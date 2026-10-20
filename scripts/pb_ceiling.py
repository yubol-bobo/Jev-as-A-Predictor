"""Where required orders are lost: timing, selection, or generation (research loop, cycle 2; aggregates only).

For each code-graded required order of each task, along an arm's logged trajectories (decisions.jsonl: the K
candidates at every step and the executed pick):
  placed            the executed turns create it (and at which step)
  offered, not taken  some candidate at some step creates it, no executed turn does: a selector could have placed it
  never offered     no candidate ever creates it: only generation can fix it
And for the exam's step-level omissions (a candidate places an open required order, the executed turn does not):
the share of those orders the trajectory places later anyway (the step-level key then measures timing, not omission).

  .venv/bin/python scripts/pb_ceiling.py [--out results/pb_ceiling.json]
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import matches, orders_in, required_orders  # noqa: E402
from pb_summarize import load_job  # noqa: E402
from src.s1clin.pb_decider import parse_turn  # noqa: E402

JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
ARMS = ["llm_log_k5/run1", "clef_k5/run1", "clef_k5__pplx12/run1"]


def arm_stats(slug):
    n = {"required": 0, "placed": 0, "offered_not_taken": 0, "never_offered": 0}
    omis, later, tasks = 0, 0, 0
    per_task = {}
    for jd in sorted((JOBS / slug).iterdir()):
        dec = jd / "logs/agent/decisions.jsonl"
        if not ((jd / "metadata.json").exists() and dec.exists()):
            continue
        specs = [s for s in required_orders(jd.name) if not s["negative"]]
        if not specs:
            continue
        tasks += 1
        rows = [json.loads(l) for l in dec.read_text().splitlines()]
        placed_at, offered = {}, set()
        step_omissions = []  # (step, spec) where an option placed the open spec and the pick did not
        for k, r in enumerate(rows):
            ords = [orders_in(parse_turn(c)) for c in r["candidates"]]
            hit = [{i for i, s in enumerate(specs) if any(matches(s, *o) for o in os_)} for os_ in ords]
            for h in hit:
                offered |= h
            open_ = {i for i in range(len(specs)) if i not in placed_at}
            for i in open_ & set().union(*hit) - hit[r["pick"]]:
                step_omissions.append((k, i))
            for i in hit[r["pick"]] & open_:
                placed_at[i] = k
        n["required"] += len(specs)
        n["placed"] += len(placed_at)
        n["offered_not_taken"] += len(offered - set(placed_at))
        n["never_offered"] += len(set(range(len(specs))) - offered)
        cps = load_job(jd)["cps"]
        lost_cps = {specs[i]["cp"] for i in offered - set(placed_at)}
        failing = {c for c, ok in cps.items() if not ok}
        per_task[jd.name] = {"n": len(specs), "never_offered": len(set(range(len(specs))) - offered),
                             "offered_not_taken": len(offered - set(placed_at)), "placed": len(placed_at),
                             # the task would pass if the selector had placed the offered orders (upper bound)
                             "pass": not failing, "pass_if_offered_placed": bool(failing) and failing <= lost_cps}
        omis += len(step_omissions)
        later += sum(1 for k, i in step_omissions if i in placed_at and placed_at[i] > k)
    out = {"tasks_with_required_orders": tasks, **n,
           "share": {k: round(n[k] / n["required"], 4) for k in ("placed", "offered_not_taken", "never_offered")},
           "step_omissions": omis, "placed_later_after_step_omission": round(later / max(omis, 1), 4),
           "task_pass": sum(v["pass"] for v in per_task.values()),
           "tasks_that_would_pass_if_offered_orders_were_placed": sum(v["pass_if_offered_placed"] for v in per_task.values())}
    return out, per_task


def paired(pa, pb, key, rng, b=5000):
    """Difference b - a in the share of required orders with `key`, over tasks in both, with a task bootstrap CI."""
    ts = sorted(set(pa) & set(pb))
    def share(ids, p):
        return sum(p[t][key] for t in ids) / max(sum(p[t]["n"] for t in ids), 1)
    d = share(ts, pb) - share(ts, pa)
    bs = sorted(share(x, pb) - share(x, pa) for x in ([rng.choice(ts) for _ in ts] for _ in range(b)))
    return [round(d, 4), round(bs[int(0.025 * b)], 4), round(bs[int(0.975 * b)], 4), len(ts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/pb_ceiling.json"))
    a = ap.parse_args()
    st = {s: arm_stats(s) for s in ARMS}
    res = {s: v[0] for s, v in st.items()}
    rng = random.Random(0)
    res["vs_llm_log"] = {s: {k: paired(st[ARMS[0]][1], st[s][1], k, rng) for k in ("placed", "offered_not_taken", "never_offered")}
                         for s in ARMS[1:]}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
