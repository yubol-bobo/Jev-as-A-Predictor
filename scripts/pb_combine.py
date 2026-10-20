"""When should a System One decider overrule the LLM, and which combination of deciders is best? Offline, on the fixed
exam (scripts/pb_exam.py): every policy picks one option per question from the same states and options, so the
policies differ only in who decides.

Keys (an option is bad; a key counts on a question only where it separates the options):
  order      the graders' required orders (pb_exam.py): an option is right if it places a required order not yet placed
  finish     FINISH while a required order is still open is premature
  redundant  the option only repeats calls already executed (same tool, same arguments), or rewrites the note with
             no new tool result since the last write
Error rate = bad picks over all (question, key) pairs; per key as well.

Error types (the insight): omission = no order picked when a required order is offered; over-ordering = an order
picked when the options mix orders and non-orders and none is required; plus premature finish and redundancy.

Policies, each against the LLM's own sample (`first`, what the llm arm executes):
  first, majority (most sampled), random (expected)
  prefer_order      no model: the most-sampled option that places an order whenever one is offered, else `first`
                    (does a decider's order accuracy need more than a bias toward acting?)
  D                 decider D's argmax everywhere (S1-Select)
  route(D)          D only at commitment points (some option orders, books, messages or finishes), else `first`
  gate(D, d)        D only when it prefers its pick over `first` by a log-odds margin >= d, else `first`
  route+gate(D, d)  both
  ens(S, w)         deciders in S: mean log-probability plus w * log of the LLM's sample frequency (smoothed)
The graders do not penalise orders nobody required, so over-ordering is not in the error rate; the gates trade it
against omission, reported as a Pareto frontier (error vs over-ordering). Configurations are chosen by 5-fold cross-validation grouped by task (the chosen setting is scored only on held-out
tasks); in-sample bests are reported separately. Differences against `first` carry a task-cluster bootstrap 95% CI.

Patient content stays under data/offline/pb_exam/; only aggregates go to results/.
  .venv/bin/python scripts/pb_combine.py [--deciders clef,d3,...] [--out results/pb_combine.json]
"""

import argparse
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_exam import JOBS, OD, SLUG  # noqa: E402
from pb_offline_orders import replay  # noqa: E402
from src.s1clin.pb_decider import parse_turn  # noqa: E402

MARGINS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0]
KEYS = ("order", "finish", "redundant")


def canon(name, args):
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False)


def redundancy(q, msgs):
    """Per option: True if it adds nothing (repeats executed calls, or rewrites the note with nothing new since)."""
    done, since_write, wrote = set(), 0, False
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            f = tc["function"]
            done.add(canon(f["name"], f.get("arguments")))
            if f["name"] == "write_file":
                wrote, since_write = True, 0
            else:
                since_write += 1
    out = []
    for o, k in zip(q["options"], q["kinds"]):
        calls = [canon(n, a) for n, a in parse_turn(o) if n != "FINISH"]
        out.append(bool(calls) and all(c in done for c in calls) or (k == "document" and wrote and since_write == 0))
    return out


def load(deciders):
    qs = [json.loads(l) for l in (OD / "questions.jsonl").read_text().splitlines()]
    ans = {}
    for d in deciders:
        a = {}
        for x in map(json.loads, (OD / f"answers_{d}.jsonl").read_text().splitlines()):
            if x["pick"] is not None:
                a[x["id"]] = x["distribution"]
        ans[d] = a
    qs = [q for q in qs if all(q["id"] in ans[d] for d in deciders)]
    hist = {}
    for q in qs:
        if q["task"] not in hist:
            hist[q["task"]] = replay(JOBS / SLUG / q["task"] / "logs/agent/decisions.messages.jsonl")
        q["redundant"] = redundancy(q, hist[q["task"]][q["msg_epoch"]][:q["n_messages"]])
        n_open = q.get("n_open_required") or 0
        q["bad"] = {}
        if q["right"] and len(set(q["right"])) > 1:
            q["bad"]["order"] = [not r for r in q["right"]]
        if "finish" in q["kinds"] and n_open > 0:
            q["bad"]["finish"] = [k == "finish" for k in q["kinds"]]
        if len(set(q["redundant"])) > 1:
            q["bad"]["redundant"] = q["redundant"]
        q["p"] = {d: ans[d][q["id"]] for d in deciders}
    return qs


def argmax(xs):
    return max(range(len(xs)), key=lambda i: (xs[i], -i))


def commit_point(q):  # as the online gate arm (pb_decider.GatedDecider): an option acts on the record or finishes
    return any(k in ("act", "finish") for k in q["kinds"])


def make_policies(deciders):
    """name -> (family, function(q) -> option index)."""
    pol = {"first": ("base", lambda q: q["first"]),
           "majority": ("base", lambda q: argmax(q["counts"])),
           "prefer_order": ("base", lambda q: argmax([c + 100 * o for c, o in zip(q["counts"], q["creates"])])
                            if any(q["creates"]) else q["first"])}

    def gate(d, m, route):
        def f(q):
            if route and not commit_point(q):
                return q["first"]
            p = q["p"][d]
            k = argmax(p)
            return k if math.log(max(p[k], 1e-12)) - math.log(max(p[q["first"]], 1e-12)) >= m else q["first"]
        return f

    for d in deciders:
        pol[d] = ("decider", lambda q, d=d: argmax(q["p"][d]))
        pol[f"route({d})"] = ("route", lambda q, d=d: argmax(q["p"][d]) if commit_point(q) else q["first"])
        for m in MARGINS[1:]:
            pol[f"gate({d},{m})"] = (f"gate({d})", gate(d, m, False))
            pol[f"route+gate({d},{m})"] = (f"route+gate({d})", gate(d, m, True))

    def ens(S, w):
        def f(q):
            n, k = len(q["options"]), sum(q["counts"])
            prior = [math.log((c + 0.5) / (k + 0.5 * n)) for c in q["counts"]]
            return argmax([sum(math.log(max(q["p"][d][i], 1e-12)) for d in S) / len(S) + w * prior[i] for i in range(n)])
        return f

    for r in range(1, len(deciders) + 1):
        for S in itertools.combinations(deciders, r):
            for w in WEIGHTS:
                if r == 1 and w == 0:
                    continue  # the decider itself
                pol[f"ens({'+'.join(S)},{w})"] = ("ens", ens(S, w))
    return pol


def score(qs, f):
    """Error rate over (question, key) pairs, per key, and error types."""
    bad, n, per = 0, 0, {k: [0, 0] for k in KEYS}
    for q in qs:
        k = f(q)
        for key, b in q["bad"].items():
            bad += b[k]; n += 1; per[key][0] += b[k]; per[key][1] += 1
    return bad / n if n else None, {key: (round(b / m, 4) if m else None) for key, (b, m) in per.items()}


def error_types(qs, f):
    req = [q for q in qs if q["right"] and any(q["right"]) and len(set(q["creates"])) > 1]
    none = [q for q in qs if len(set(q["creates"])) > 1 and not (q["right"] and any(q["right"]))]
    share = lambda xs, g: round(sum(map(g, xs)) / len(xs), 4) if xs else None  # noqa: E731
    return {"omission": share(req, lambda q: not q["creates"][f(q)]),
            "over_ordering": share(none, lambda q: q["creates"][f(q)]),
            "premature_finish": share([q for q in qs if "finish" in q["bad"]], lambda q: q["bad"]["finish"][f(q)]),
            "redundant": share([q for q in qs if "redundant" in q["bad"]], lambda q: q["redundant"][f(q)]),
            "departs_from_llm": share(qs, lambda q: f(q) != q["first"]),
            "n": {"omission": len(req), "over_ordering": len(none)}}


def fold(task):
    return int(hashlib.sha256(task.encode()).hexdigest(), 16) % 5


def cross_validated(qs, pol, names):
    """Choose the best of `names` on 4 folds, score it on the fifth; returns the pooled held-out error and choices."""
    bad, n, chosen = 0, 0, []
    for k in range(5):
        train, test = [q for q in qs if fold(q["task"]) != k], [q for q in qs if fold(q["task"]) == k]
        best = min(names, key=lambda x: (score(train, pol[x][1])[0], x))
        chosen.append(best)
        for q in test:
            i = pol[best][1](q)
            for b in q["bad"].values():
                bad += b[i]; n += 1
    return bad / n, chosen


def bootstrap_diff(qs, f, g, reps=1000):
    """Task-cluster bootstrap of error(f) - error(g): 95% CI."""
    by = {}
    for q in qs:
        if q["bad"]:
            fi, gi = f(q), g(q)
            e = by.setdefault(q["task"], [0, 0, 0])
            for b in q["bad"].values():
                e[0] += b[fi]; e[1] += b[gi]; e[2] += 1
    tasks, rng, out = sorted(by), random.Random(0), []
    for _ in range(reps):
        s = [by[t] for t in (rng.choice(tasks) for _ in tasks)]
        m = sum(x[2] for x in s)
        out.append((sum(x[0] for x in s) - sum(x[1] for x in s)) / m)
    out.sort()
    return [round(out[int(0.025 * reps)], 4), round(out[int(0.975 * reps)], 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deciders", default=None, help="default: every decider that has answered >= 95%% of the exam")
    ap.add_argument("--out", default=str(REPO / "results/pb_combine.json"))
    a = ap.parse_args()
    n_q = len((OD / "questions.jsonl").read_text().splitlines())
    if a.deciders:
        deciders = a.deciders.split(",")
    else:
        deciders = sorted(f.stem[len("answers_"):] for f in OD.glob("answers_*.jsonl")
                          if "_blind" not in f.stem and len({x["id"] for x in map(json.loads, f.read_text().splitlines())
                                                             if x["pick"] is not None}) >= 0.95 * n_q)
    qs = load(deciders)
    pol = make_policies(deciders)
    keyed = [q for q in qs if q["bad"]]
    res = {"exam": SLUG, "deciders": deciders, "questions": len(qs), "tasks": len({q["task"] for q in qs}),
           "keyed": {k: sum(k in q["bad"] for q in qs) for k in KEYS}, "keyed_questions": len(keyed),
           "random_error": round(sum(sum(b) / len(b) for q in qs for b in q["bad"].values())
                                 / sum(len(q["bad"]) for q in qs), 4)}

    # error profile: LLM vs each decider (the insight)
    res["error_types"] = {x: error_types(qs, pol[x][1]) for x in ["first", "majority", "prefer_order"] + deciders}
    first_kind = Counter(q["kinds"][q["first"]] for q in qs)
    res["kind_preference"] = {d: {k: round(sum(q["kinds"][argmax(q["p"][d])] == k for q in qs) / max(c, 1), 3)
                                  for k, c in first_kind.items()} for d in deciders}  # picks of kind k / LLM's
    longest = lambda q, i: len(q["options"][i]) == max(map(len, q["options"]))  # noqa: E731
    res["picks_longest_option"] = {x: round(sum(longest(q, pol[x][1](q)) for q in qs) / len(qs), 4)
                                   for x in ["first"] + deciders}
    res["picks_longest_option"]["random"] = round(
        sum(sum(longest(q, i) for i in range(len(q["options"]))) / len(q["options"]) for q in qs) / len(qs), 4)

    # every policy in-sample, then the families cross-validated
    allp = {x: score(qs, f) for x, (_, f) in pol.items()}
    res["in_sample"] = {x: {"error": round(e, 4), "per_key": pk} for x, (e, pk) in
                        sorted(allp.items(), key=lambda kv: kv[1][0])[:25]}
    for x in ["first", "majority", "prefer_order"] + deciders + [f"route({d})" for d in deciders]:
        res["in_sample"].setdefault(x, {"error": round(allp[x][0], 4), "per_key": allp[x][1]})
    fams = {}
    for x, (fam, _) in pol.items():
        fams.setdefault(fam, []).append(x)
    fams["gate(any)"] = [x for x in pol if x.startswith("gate(")]
    fams["route+gate(any)"] = [x for x in pol if x.startswith("route+gate(")]
    fams["any"] = list(pol)
    res["cross_validated"] = {}
    for fam, names in sorted(fams.items()):
        if fam == "base" or len(names) == 1 and fam == "decider":
            continue
        e, chosen = cross_validated(qs, pol, names)
        res["cross_validated"][fam] = {"error": round(e, 4), "chosen": dict(Counter(chosen))}
    pts = {x: (allp[x][0], error_types(qs, pol[x][1])["over_ordering"]) for x in pol}
    res["pareto"] = [{"policy": x, "error": round(e, 4), "over_ordering": o, **{k: v for k, v in error_types(
        qs, pol[x][1]).items() if k in ("omission", "departs_from_llm")}}
        for x, (e, o) in sorted(pts.items(), key=lambda kv: kv[1])
        if not any(e2 <= e and o2 <= o and (e2, o2) != (e, o) for e2, o2 in pts.values())]
    best = min(pol, key=lambda x: (allp[x][0], x))
    res["vs_first_ci95"] = {x: {"diff": round(allp[x][0] - allp["first"][0], 4),
                                "ci": bootstrap_diff(qs, pol[x][1], pol["first"][1])}
                            for x in dict.fromkeys(deciders + [f"route({d})" for d in deciders] + [best, "majority", "prefer_order"])}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("deciders", "questions", "keyed", "random_error", "cross_validated")}, indent=1))
    print("in-sample best:", best, allp[best][0], "first:", allp["first"][0])


if __name__ == "__main__":
    main()
