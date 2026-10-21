"""Offline (research loop cycle 8): would a larger candidate pool cover the required orders a run never proposes, and
how hard is screening it? Pools the orders proposed by any candidate in five logged runs per task. Aggregates only.

  .venv/bin/python scripts/pb_pool.py
"""
import json, sys, re
from collections import defaultdict
from pathlib import Path
REPO = Path("/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor")
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_offline_orders import matches, orders_in, required_orders
from src.s1clin.pb_decider import parse_turn
JOBS = Path("/ocean/projects/cis260032p/yliy/data/physicianbench/jobs")
RUNS = ["llm_log_k5/run1", "clef_k5/run1", "clef_k5__pplx12/run1", "gate0_orders_k5__pplx12/run1", "gate0_orders_k5__pplx12/run2"]
norm = lambda o: (o[0], o[2].strip() if o[2] else re.sub(r"\W+", " ", (o[1] or "").lower()).strip())
offered = defaultdict(dict)   # task -> run -> {orderkey: (order, n_candidates)}
placed = defaultdict(dict)
for run in RUNS:
    for jd in (JOBS / run).iterdir():
        dec = jd / "logs/agent/decisions.jsonl"
        if not ((jd / "metadata.json").exists() and dec.exists()): continue
        off, pl = {}, set()
        for l in dec.read_text().splitlines():
            r = json.loads(l)
            for j, c in enumerate(r["candidates"]):
                for o in orders_in(parse_turn(c)):
                    k = norm(o); off.setdefault(k, [o, 0]); off[k][1] += 1
                    if j == r["pick"]: pl.add(k)
        offered[jd.name][run] = off; placed[jd.name][run] = pl
tasks = [t for t in offered if len(offered[t]) == len(RUNS)]
print("tasks with all runs:", len(tasks))
tot = cov_run = cov_pool = pl_run = 0; rescued = 0; pool_sz = []; base = []
ranks = []  # (score, is_required) for frequency baseline
for t in tasks:
    specs = [s for s in required_orders(t) if not s["negative"]]
    pool = {}
    for run, off in offered[t].items():
        for k, (o, n) in off.items():
            pool.setdefault(k, [o, 0, 0]); pool[k][1] += n; pool[k][2] += 1
    pool_sz.append(len(pool))
    req_keys = {k for k, (o, n, r) in pool.items() if any(matches(s, *o) for s in specs)}
    if specs:
        base.append((len(req_keys), len(pool)))
        for k, (o, n, r) in pool.items(): ranks.append((r, n, k in req_keys))
    for run in RUNS:
        for s in specs:
            tot += 1
            in_run = any(matches(s, *o) for o, n in offered[t][run].values())
            in_pool = any(matches(s, *v[0]) for v in pool.values())
            in_other = any(matches(s, *o) for r2 in RUNS if r2 != run for o, n in offered[t][r2].values())
            pl_run += any(matches(s, *offered[t][run][k][0]) for k in placed[t][run])
            cov_run += in_run; cov_pool += in_pool; rescued += (not in_run) and in_other
print(f"required (spec x run): {tot}; placed in run {pl_run/tot:.3f}; proposed in run {cov_run/tot:.3f}; "
      f"proposed in the 5-run pool {cov_pool/tot:.3f}; never proposed in run but proposed in another run: "
      f"{rescued}/{tot-cov_run} = {rescued/max(tot-cov_run,1):.3f}")
import statistics as st
print("pool size distinct orders per task: median", st.median(pool_sz), "mean", round(st.mean(pool_sz),1))
print("tasks with required orders:", len(base), "required keys in pool", sum(a for a,b in base), "of", sum(b for a,b in base),
      f"= base rate {sum(a for a,b in base)/sum(b for a,b in base):.3f}")
def auroc(xs):
    pos = [x for x, y in xs if y]; neg = [x for x, y in xs if not y]
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
print("frequency baseline AUROC (runs proposing):", round(auroc([(r, y) for r, n, y in ranks]), 3),
      "(candidates proposing):", round(auroc([(n, y) for r, n, y in ranks]), 3))
for th in (1, 2, 3, 4, 5):
    sel = [(y) for r, n, y in ranks if r >= th]
    print(f"  proposed in >= {th} runs: {len(sel)} orders, required {sum(sel)}, precision {sum(sel)/max(len(sel),1):.3f}")
