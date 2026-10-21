"""Why the offline gain of the deciders does not show online (research loop, cycle 1; aggregates only).

Offline, on the fixed exam, every decider makes far fewer key errors than the LLM's own sample (pb_combine.py). Online,
Clef in the decision seat (clef_k5) does not beat the LLM on task pass. This script compares the online runs at the
checkpoint level and along the trajectory, pooling the runs of each side:
  LLM side: llm/run1, llm/run2, llm_log_k5/run1 (the same policy: execute the first sample)
  decider side: clef_k5/run1, clef_k5__pplx12/run1
Per task, each side's rate is the mean over its runs; differences are paired by task with task-cluster bootstrap CIs.

Checkpoints are split by the benchmark taxonomy, and Action Execution further into code-graded order checkpoints
(required orders: pb_offline_orders.required_orders), "must not be ordered" checkpoints, and other actions.
Trajectory measures from the executed tool calls: orders placed, required orders placed, forbidden orders placed,
orders that match no grader, notes written, steps, retrieval before the first action.

  .venv/bin/python scripts/pb_gap.py [--out results/pb_gap.json] [--decider-runs clef_k5__pplx12/run1] [--llm-runs ...]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts")]
from pb_offline_orders import CREATE, matches, required_orders  # noqa: E402
from pb_summarize import JOBS, TAXONOMY, load_job  # noqa: E402

SIDES = {"llm": ["llm/run1", "llm/run2", "llm_log_k5/run1"], "decider": ["clef_k5/run1", "clef_k5__pplx12/run1"]}
B = 5000


def calls(jd):
    out = []
    p = jd / "logs/agent/trajectory.log"
    for line in p.read_text().splitlines() if p.exists() else []:
        e = json.loads(line)
        if e["type"] == "tool_call":
            out.append((e["metadata"].get("tool_name", ""), e["metadata"].get("input") or {}))
    return out


def trajectory(jd, specs):
    cs = calls(jd)
    pos = [s for s in specs if not s["negative"]]
    neg = [s for s in specs if s["negative"]]
    orders = [(CREATE[n][0], a.get(CREATE[n][1], ""), str(a.get(CREATE[n][2], "") or "")) for n, a in cs if n in CREATE]
    first_act = next((i for i, (n, _) in enumerate(cs) if n == "write_file" or n.endswith("_create")), len(cs))
    return {"steps": len(cs), "orders": len(orders),
            "required_placed": (sum(any(matches(s, *o) for o in orders) for s in pos) / len(pos)) if pos else None,
            "forbidden_placed": (sum(any(matches(s, *o) for o in orders) for s in neg) / len(neg)) if neg else None,
            "unmatched_orders": sum(not any(matches(s, *o) for s in specs) for o in orders) if specs else None,
            "notes": sum(n == "write_file" for n, _ in cs),
            "retrieve_before_first_action": first_act}


def cp_group(task, cp, specs):
    t = TAXONOMY.get(task, {}).get(cp, "?")
    mine = [s for s in specs if s["cp"] == cp]
    if mine and all(s["negative"] for s in mine):
        return t, "AE: must not order"
    if mine:
        return t, "AE: required order"
    return t, ("AE: other action" if t == "Action Execution" else t)


def boot(d, rng):
    n = len(d)
    if not n:
        return None
    ms = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(sum(d) / n, 4), round(ms[int(0.025 * B)], 4), round(ms[int(0.975 * B)], 4), n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/pb_gap.json"))
    ap.add_argument("--decider-runs", default=None, help="comma list replacing the decider side's runs")
    ap.add_argument("--llm-runs", default=None, help="comma list replacing the LLM side's runs (e.g. a control arm)")
    ap.add_argument("--tasks", default=None, help="@file: only these tasks (e.g. results/p1_unseen_tasks.txt)")
    a = ap.parse_args()
    if a.decider_runs:
        SIDES["decider"] = a.decider_runs.split(",")
    if a.llm_runs:
        SIDES["llm"] = a.llm_runs.split(",")
    rng = random.Random(0)
    tasks = sorted(p.name for p in (JOBS / "llm/run1").iterdir() if p.is_dir())
    if a.tasks:
        keep = set(Path(a.tasks[1:]).read_text().split())
        tasks = [t for t in tasks if t in keep]
    # per side, per task: list over runs of {cp: ok} and trajectory measures
    data = {s: defaultdict(list) for s in SIDES}
    for side, runs in SIDES.items():
        for r in runs:
            for t in tasks:
                jd = JOBS / r / t
                j = load_job(jd) if jd.is_dir() else None
                if j and j["cps"]:
                    data[side][t].append({"run": r, "pass": j["pass"], "cps": j["cps"], "traj": trajectory(jd, required_orders(t))})
    paired = [t for t in tasks if data["llm"][t] and data["decider"][t]]
    res = {"sides": SIDES, "paired_tasks": len(paired),
           "runs_per_task": {s: round(sum(len(data[s][t]) for t in paired) / len(paired), 2) for s in SIDES}}

    def side_mean(s, t, f):
        v = [f(x) for x in data[s][t]]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    def compare(f):
        rows = [(side_mean("llm", t, f), side_mean("decider", t, f)) for t in paired]
        rows = [r for r in rows if None not in r]
        return {"llm": round(sum(r[0] for r in rows) / max(len(rows), 1), 4),
                "decider": round(sum(r[1] for r in rows) / max(len(rows), 1), 4),
                "diff_ci": boot([r[1] - r[0] for r in rows], rng)}

    res["task_pass"] = compare(lambda x: float(x["pass"]))
    groups = defaultdict(set)
    for t in paired:
        specs = required_orders(t)
        for x in data["llm"][t] + data["decider"][t]:
            for cp in x["cps"]:
                groups[cp_group(t, cp, specs)[1]].add((t, cp))
    res["checkpoints"] = {}
    for g, members in sorted(groups.items()):
        by_task = defaultdict(list)
        for t, cp in members:
            by_task[t].append(cp)

        def rate(s, t, cps=None):
            v = [x["cps"][c] for x in data[s][t] for c in cps if c in x["cps"]]
            return sum(v) / len(v) if v else None
        rows = [(rate("llm", t, c), rate("decider", t, c)) for t, c in by_task.items()]
        rows = [r for r in rows if None not in r]
        res["checkpoints"][g] = {"n_checkpoints": len(members), "llm": round(sum(r[0] for r in rows) / len(rows), 4),
                                 "decider": round(sum(r[1] for r in rows) / len(rows), 4),
                                 "diff_ci": boot([r[1] - r[0] for r in rows], rng)}
    res["trajectory"] = {k: compare(lambda x, k=k: x["traj"][k])
                         for k in ["steps", "orders", "required_placed", "forbidden_placed", "unmatched_orders", "notes",
                                   "retrieve_before_first_action"]}
    # within the LLM side, run-to-run noise on the same measures (llm/run1 vs llm_log_k5/run1)
    a_, b_ = "llm/run1", "llm_log_k5/run1"
    noise = {}
    for k in ["orders", "required_placed", "unmatched_orders", "notes"]:
        d = []
        for t in paired:
            xa = [x for x in data["llm"][t] if x["run"] == a_]
            xb = [x for x in data["llm"][t] if x["run"] == b_]
            if xa and xb and xa[0]["traj"][k] is not None and xb[0]["traj"][k] is not None:
                d.append(xb[0]["traj"][k] - xa[0]["traj"][k])
        noise[k] = boot(d, rng)
    res["noise_llm_vs_llm_log"] = noise
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
