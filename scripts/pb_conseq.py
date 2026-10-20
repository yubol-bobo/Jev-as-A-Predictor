"""Exam order key, consequential version (research loop, cycle 3, pre-registered P3; aggregates only).

The exam's per-step order key (pb_exam.py) marks an option right if it places a still-open required order. In the
LLM's trajectory 79% of those orders are placed later anyway (pb_ceiling.py), so the key mostly rewards ordering
early. Here an option is right only if it places an open required order that the trajectory never places after this
step (a consequential omission if not taken); this favours the first sample where it places the order itself (its
counterfactual is unknown). "rescue" avoids that: only steps where the executed turn takes no open required order,
and right = an option places one the trajectory never places. Scored on questions with such an option: the share of picks that take
it, for the LLM's first sample, the most-sampled option, prefer_order, and each decider.

  .venv/bin/python scripts/pb_conseq.py [--out results/pb_conseq.json]
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import matches, orders_in, required_orders  # noqa: E402
from src.s1clin.pb_decider import parse_turn  # noqa: E402

JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs/llm_log_k5/run1")
OD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_exam")


def keys():
    out = {}
    for jd in sorted(JOBS.iterdir()):
        dec = jd / "logs/agent/decisions.jsonl"
        if not dec.exists():
            continue
        specs = [s for s in required_orders(jd.name) if not s["negative"]]
        if not specs:
            continue
        rows = [json.loads(l) for l in dec.read_text().splitlines()]
        hits = [[{i for i, s in enumerate(specs) if any(matches(s, *o) for o in orders_in(parse_turn(c)))}
                 for c in r["candidates"]] for r in rows]
        placed_at = {}
        for k, r in enumerate(rows):
            for i in hits[k][r["pick"]]:
                placed_at.setdefault(i, k)
        for k, r in enumerate(rows):
            if len(r["candidates"]) < 2:
                continue
            open_ = {i for i in range(len(specs)) if placed_at.get(i, 10 ** 9) >= k}
            step = {i for i in open_ if placed_at.get(i, 10 ** 9) > k}  # not placed by the executed turn here either
            never_later = {i for i in open_ if i not in placed_at or placed_at[i] == k}
            per = [bool(h & open_) for h in hits[k]]
            cons = [bool(h & never_later) for h in hits[k]]
            # rescue: the executed turn takes no open required order here and some option places one that the
            # trajectory never places (the counterfactual is known only when the executed turn did not take it)
            lost = {i for i in open_ if i not in placed_at}
            rescue = [bool(h & lost) for h in hits[k]] if not hits[k][r["pick"]] & open_ else [False] * len(hits[k])
            out[f"{jd.name}|{r['step']}"] = {"task": jd.name, "per_step": per, "conseq": cons, "rescue": rescue}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/pb_conseq.json"))
    a = ap.parse_args()
    ks = keys()
    qs = {q["id"]: q for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines())}
    picks = {"first": {i: q["first"] for i, q in qs.items()},
             "majority": {i: max(range(len(q["counts"])), key=lambda k: (q["counts"][k], -k)) for i, q in qs.items()},
             "prefer_order": {i: next((k for k, c in enumerate(q["creates"]) if c), q["first"]) for i, q in qs.items()}}
    for f in sorted(OD.glob("answers_*.jsonl")):
        picks[f.stem[len("answers_"):]] = {x["id"]: x["pick"] for x in map(json.loads, f.read_text().splitlines())
                                           if x["pick"] is not None}
    rng = random.Random(0)
    res = {"exam": "llm_log_k5/run1"}
    for key in ("per_step", "conseq", "rescue"):
        items = [i for i, k in ks.items() if i in qs and any(k[key]) and not all(k[key])]
        common = [i for i in items if all(i in p for p in picks.values())]
        tasks = sorted({ks[i]["task"] for i in common})
        r = {"questions": len(common), "tasks": len(tasks), "accuracy": {}}
        for n, p in picks.items():
            acc = lambda ids: sum(ks[i][key][p[i]] for i in ids) / max(len(ids), 1)  # noqa: E731
            bs = []
            for _ in range(2000):
                ts = [rng.choice(tasks) for _ in tasks]
                ids = [i for t in ts for i in common if ks[i]["task"] == t]
                bs.append(acc(ids))
            bs.sort()
            r["accuracy"][n] = [round(acc(common), 4), round(bs[50], 4), round(bs[1949], 4)]
        res[key] = r
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
