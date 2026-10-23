"""R, the restart at the agent's FINISH from a shared prefix (docs/research_loop.md; aggregates only).

Cases: the A2 cases, each (logged run, task) of llm_log_k5/run1 and run3 with a required-order spec still open at
its FINISH (results/r_tasks_{1,3}.txt). Every arm replays the logged run up to that FINISH and branches there
(`--ranker restart-<arm>-p<run>`): none (the logged FINISH), full (RESTART_ASK with the full context), fresh
(RESTART_ASK with its own decisions removed); full and fresh keep only order calls, minus repeats of placed orders
(exact name, then the pplx12 repeat judge); list / listrand show O1's checklist (pplx12-ranked or random 8 of the
filtered task-prior pool) at the same point. All with K = 1 (the executed turn only). Per job: the grader's pass/fail of every checkpoint.
Per case (mean over the runs of an arm): the share of the open specs' checkpoints that pass (primary), the share of
all checkpoints that pass, orders placed after the branch, those matching no required spec (unmatched) and forbidden
ones. Paired over cases, bootstrap CIs. Replay check: agreement of `none` with the logged run, checkpoint by
checkpoint.

  .venv/bin/python scripts/pb_r.py [--out results/pb_r.json]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
from pb_gap import calls  # noqa: E402
from pb_offline_orders import CREATE, matches, required_orders  # noqa: E402
from pb_replay import cases  # noqa: E402
from pb_summarize import JOBS  # noqa: E402

RANKER = {"none": "restart-none", "full": "restart-full", "fresh": "restart-fresh", "list": "prior-pplx12-dedup",
          "listrand": "prior-random-dedup"}
ARMS = tuple(RANKER)
CP = re.compile(r"^(PASSED|FAILED) \S*test_checkpoint_(\S+)", re.M)


def checkpoints(jd):
    p = jd / "logs/verifier/pytest_output.txt"
    return {cp: st == "PASSED" for st, cp in CP.findall(p.read_text())} if p.exists() else None


def boot(d, n=2000):
    if not d:
        return None
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(b[50], 4), round(b[1949], 4)]


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/pb_r.json"))
    a = ap.parse_args()
    per, agree, n_jobs = {arm: {} for arm in ARMS}, [], {arm: 0 for arm in ARMS}
    for c in cases():
        if not c["open"] or not c["finish"]:
            continue
        p = c["run"][-1]
        specs = [s for s in required_orders(c["task"]) if not s["negative"]]
        neg = [s for s in required_orders(c["task"]) if s["negative"]]
        open_cps = {specs[i]["cp"] for i in c["open"]}
        n_placed = len(c["placed"])
        for arm in ARMS:
            rows = []
            for rd in sorted((JOBS / f"screen_{RANKER[arm]}-p{p}_b0m8_k1").glob("run[0-9]*")):
                if rd.name == "run99":  # smoke tests
                    continue
                jd = rd / c["task"]
                cps = checkpoints(jd) if (jd / "metadata.json").exists() else None
                if cps is None:
                    continue
                n_jobs[arm] += 1
                orders = [(CREATE[n][0], x.get(CREATE[n][1], ""), str(x.get(CREATE[n][2], "") or ""))
                          for n, x in calls(jd) if n in CREATE][n_placed:]  # after the branch
                rows.append({"open": mean([cps.get(cp) for cp in open_cps if cp in cps]),
                             "all": mean(list(cps.values())),
                             "orders": len(orders),
                             "unmatched": sum(not any(matches(s, *o) for s in specs) for o in orders),
                             "forbidden": sum(any(matches(s, *o) for s in neg) for o in orders)})
                if arm == "none" and rd.name == "run1":
                    orig = checkpoints(JOBS / c["run"] / c["task"])
                    if orig:
                        agree += [orig[k] == cps.get(k) for k in orig]
            if rows:
                per[arm][(c["run"], c["task"])] = {k: mean([r[k] for r in rows]) for k in rows[0]} | {"runs": len(rows)}
    res = {"jobs": n_jobs, "replay_agreement": round(mean(agree), 4) if agree else None,
           "replay_checkpoints": len(agree)}
    for arm in ARMS:
        rows = per[arm].values()
        res[arm] = {"cases": len(rows), **{k: round(mean([r[k] for r in rows]), 4) if rows else None
                                           for k in ("open", "all", "orders", "unmatched", "forbidden")}}
    for x, y in (("fresh", "full"), ("full", "none"), ("fresh", "none"), ("list", "none"), ("list", "listrand"),
                 ("listrand", "none"), ("fresh", "list")):
        ks = [k for k in per[x] if k in per[y] and per[x][k]["open"] is not None and per[y][k]["open"] is not None]
        res[f"{x}_minus_{y}"] = {"cases": len(ks)} | {
            k: {"mean": round(mean(d), 4), "ci95": boot(d)} for k in ("open", "all", "unmatched", "forbidden")
            for d in [[per[x][c][k] - per[y][c][k] for c in ks]] if d}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
