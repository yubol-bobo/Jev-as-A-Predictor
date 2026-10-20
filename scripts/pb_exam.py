"""The fixed exam: every System One decider answers the same PhysicianBench decision states with the same options.

The exam comes from the LLM's own run (arm llm_log, run 1). The LLM executes its first sample, as in the llm arm, and
logs at every step its K = 5 candidate turns (deduplicated, with counts) and the full message history. Each step with
at least two distinct candidates is one question: the state (the history up to that step) and the options (the
distinct candidates). Every decider gets byte-identical requests: the same rendering, the same question wording,
the same 32k fitting rule, and the same option letters (seeded by the question id, not by the decider).

Answer key where one exists: order steps. Each task's code-graded order checkpoints give the required orders, using
the graders' own matching (pb_offline_orders.py). An option is right if it places a required order that the
trajectory has not placed yet. Elsewhere there is no key, and the report gives agreement: with the LLM's own sample,
with the most-sampled option, and between deciders.

  .venv/bin/python scripts/pb_exam.py build                 # questions from the finished llm_log tasks (rerun to extend)
  S1CLIN_DECIDER=d3 .venv/bin/python scripts/pb_exam.py select [--workers 2] [--blind]
  .venv/bin/python scripts/pb_exam.py report [--out results/pb_exam.json]
Patient content stays under data/offline/pb_exam/; only aggregates go to results/.
"""

import argparse
import hashlib
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import BLIND, matches, orders_in, replay, required_orders  # noqa: E402
from src.s1clin.pb_decider import DECISION_QUESTION, ClefDecider, parse_turn  # noqa: E402

JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
SLUG = "llm_log_k5/run1"
OD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_exam")


def kind(turn, raw):
    names = {n for n, _ in turn}
    if raw.startswith("write_file(") or "\nwrite_file(" in raw:  # a multi-line note does not parse as one call line
        names.add("write_file")
    if names & {"fhir_service_request_create", "fhir_medication_request_create", "fhir_appointment_create",
                "fhir_communication_create_message"}:
        return "act"
    if "write_file" in names:
        return "document"
    if names == {"FINISH"}:
        return "finish"
    return "retrieve"


def build():
    OD.mkdir(parents=True, exist_ok=True, mode=0o700)
    qs, n_tasks = [], 0
    for jd in sorted((JOBS / SLUG).iterdir()):
        dec = jd / "logs/agent/decisions.jsonl"
        if not (jd.is_dir() and (jd / "metadata.json").exists() and dec.exists()
                and (jd / "logs/agent/decisions.messages.jsonl").exists()):
            continue
        n_tasks += 1
        specs = [s for s in required_orders(jd.name) if not s["negative"]]
        placed = set()
        for r in map(json.loads, dec.read_text().splitlines()):
            turns = [parse_turn(c) for c in r["candidates"]]
            ords = [orders_in(t) for t in turns]
            open_specs = [i for i in range(len(specs)) if i not in placed]
            if len(r["candidates"]) > 1:
                qs.append({"id": f"{jd.name}|{r['step']}", "task": jd.name, "step": r["step"],
                           "n_messages": r["n_messages"], "msg_epoch": r["msg_epoch"], "options": r["candidates"],
                           "counts": r["counts"], "first": r["pick"], "kinds": [kind(t, c) for t, c in zip(turns, r["candidates"])],
                           "creates": [bool(o) for o in ords],
                           "right": [any(matches(specs[i], *o) for o in os_ for i in open_specs) for os_ in ords]
                           if specs else None})
            for o in orders_in(turns[r["pick"]]):
                placed |= {i for i, s in enumerate(specs) if matches(s, *o)}
    with open(OD / "questions.jsonl", "w") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"exam: {n_tasks} tasks, {len(qs)} questions, "
          f"{sum(1 for q in qs if q['right'] and len(set(q['right'])) > 1)} with an answer key that separates options")


def select(workers, blind):
    qs = [json.loads(l) for l in (OD / "questions.jsonl").read_text().splitlines()]
    dec = ClefDecider(question=DECISION_QUESTION)
    name = dec.name + ("_blind" if blind else "")
    out = OD / f"answers_{name}.jsonl"
    done = ({x["id"] for x in map(json.loads, out.read_text().splitlines()) if x["pick"] is not None}
            if out.exists() else set())  # failed answers (e.g. the server went away) are retried
    lock, cache = threading.Lock(), {}

    def state(q):
        if q["task"] not in cache:
            cache[q["task"]] = replay(JOBS / SLUG / q["task"] / "logs/agent/decisions.messages.jsonl")
        return cache[q["task"]][q["msg_epoch"]][:q["n_messages"]]

    def run(q):
        seed = hashlib.sha256(q["id"].encode()).hexdigest()  # same letters for every decider
        try:
            msgs = [{"role": "user", "content": BLIND}] if blind else state(q)
            pick, dist, info = dec.choose(msgs, q["options"], seed)
        except Exception as e:
            pick, dist, info = None, None, {"error": repr(e)[:300]}
        with lock, open(out, "a") as f:
            f.write(json.dumps({"id": q["id"], "pick": pick, "distribution": dist,
                                "info": {k: info.get(k) for k in ("error", "input_tokens", "fit") if k in info}}) + "\n")

    # questions the other deciders have already answered come first, so a late decider adds to the common set fastest
    others = Counter(x["id"] for f in OD.glob("answers_*.jsonl") if f != out
                     for x in map(json.loads, f.read_text().splitlines()) if x["pick"] is not None)
    todo = sorted((q for q in qs if q["id"] not in done), key=lambda q: -others[q["id"]])
    print(f"{name}: {len(todo)} of {len(qs)} questions to answer", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))


def report(path):
    qs = {q["id"]: q for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines())}
    ans = {}
    for f in sorted(OD.glob("answers_*.jsonl")):
        a = {x["id"]: x for x in map(json.loads, f.read_text().splitlines()) if x["pick"] is not None}
        ans[f.stem[len("answers_"):]] = a
    names = sorted(ans)
    common = sorted(set(qs).intersection(*[set(a) for a in ans.values()])) if ans else []
    res = {"exam": SLUG, "questions": len(qs), "tasks": len({q["task"] for q in qs.values()}),
           "answered_by_all": len(common), "deciders": {}, "agreement": {}}
    keyed = [i for i in common if qs[i]["right"] and len(set(qs[i]["right"])) > 1]
    mixed = [i for i in common if len(set(qs[i]["creates"])) > 1]
    req = [i for i in mixed if qs[i]["right"] and any(qs[i]["right"])]
    none = [i for i in mixed if not (qs[i]["right"] and any(qs[i]["right"]))]
    share = lambda ids, f: round(sum(f(i) for i in ids) / len(ids), 4) if ids else None  # noqa: E731
    majority = lambda i: max(range(len(qs[i]["counts"])), key=lambda k: (qs[i]["counts"][k], -k))  # noqa: E731
    res["baselines"] = {
        "first_sample": {"order_accuracy": share(keyed, lambda i: qs[i]["right"][qs[i]["first"]]),
                         "orders_when_required_available": share(req, lambda i: qs[i]["creates"][qs[i]["first"]]),
                         "orders_when_none_required": share(none, lambda i: qs[i]["creates"][qs[i]["first"]])},
        "majority": {"order_accuracy": share(keyed, lambda i: qs[i]["right"][majority(i)])},
        "random": {"order_accuracy": share(keyed, lambda i: sum(qs[i]["right"]) / len(qs[i]["right"]))}}
    res["key"] = {"keyed_questions": len(keyed), "required_available": len(req), "none_required": len(none)}
    for n in names:
        a = ans[n]
        pick = lambda i: a[i]["pick"]  # noqa: E731
        kinds = Counter(qs[i]["kinds"][pick(i)] for i in common)
        res["deciders"][n] = {
            "answered": len(a), "order_accuracy": share(keyed, lambda i: qs[i]["right"][pick(i)]),
            "orders_when_required_available": share(req, lambda i: qs[i]["creates"][pick(i)]),
            "orders_when_none_required": share(none, lambda i: qs[i]["creates"][pick(i)]),
            "agrees_with_first_sample": share(common, lambda i: pick(i) == qs[i]["first"]),
            "agrees_with_majority": share(common, lambda i: pick(i) == majority(i)),
            "picked_kind": {k: round(v / max(len(common), 1), 4) for k, v in kinds.most_common()},
            "mean_max_p": share(common, lambda i: max(a[i]["distribution"]))}
    for x in names:
        for y in names:
            if x < y:
                res["agreement"][f"{x}|{y}"] = share(common, lambda i: ans[x][i]["pick"] == ans[y][i]["pick"])
    res["offered_kind"] = dict(Counter(k for i in common for k in qs[i]["kinds"]).most_common())
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "select", "report"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--blind", action="store_true", help="options only, no state")
    ap.add_argument("--out", default=str(REPO / "results/pb_exam.json"))
    a = ap.parse_args()
    {"build": build, "select": lambda: select(a.workers, a.blind), "report": lambda: report(a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
