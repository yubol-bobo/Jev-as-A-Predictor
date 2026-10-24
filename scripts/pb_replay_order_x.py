"""Exploratory after the A3/A4 verdict: the order-step conditions on the specs each run finally missed (open at its
FINISH), and order precision. Aggregates only; writes results/pb_replay_order_x.json (S1CLIN_AGENT=gemma4: XA,
results/pb_replay_order_x_gemma4.json, with plan − remind).
  .venv/bin/python scripts/pb_replay_order_x.py
"""
import json, sys
sys.path[:0] = ["scripts", "."]
from pb_replay import cases, OD, SUFFIX, turn_orders, boot
from pb_offline_orders import required_orders, matches
cs = {(c["run"], c["task"]): c for c in cases()}
per = {}
for cond in ("order_none", "order_remind", "order_plan"):
    rows = {}
    for x in map(json.loads, (OD / f"samples_{cond}.jsonl").read_text().splitlines()):
        c = cs.get((x["run"], x["task"]))
        if not c or "samples" not in x:
            continue
        specs = [s for s in required_orders(c["task"]) if not s["negative"]]
        neg = [s for s in required_orders(c["task"]) if s["negative"]]
        if not specs:
            continue
        o_cov, unm, prec, forb = [], [], [], []
        for smp in x["samples"]:
            os_ = turn_orders(smp)
            if c["open"]:
                o_cov.append(len({i for i in c["open"] for o in os_ if matches(specs[i], *o)}) / len(c["open"]))
            m = [any(matches(s, *o) for s in specs) for o in os_]
            unm.append(sum(not v for v in m)); forb.append(sum(any(matches(s, *o) for s in neg) for o in os_))
            if os_:
                prec.append(sum(m) / len(m))
        rows[(x["run"], x["task"])] = {"open": sum(o_cov) / len(o_cov) if o_cov else None, "unm": sum(unm) / len(unm),
                                       "forb": sum(forb) / len(forb), "prec": sum(prec) / len(prec) if prec else None}
    per[cond] = rows
res = {}
for cond, rows in per.items():
    res[cond] = {k: round(sum(r[k] for r in rows.values() if r[k] is not None) /
                          max(1, sum(r[k] is not None for r in rows.values())), 4) for k in ("open", "unm", "forb", "prec")}
    res[cond]["cases_with_open"] = sum(r["open"] is not None for r in rows.values())
for a, b in (("order_remind", "order_none"), ("order_plan", "order_none")) + ((("order_plan", "order_remind"),) if SUFFIX else ()):
    for k in ("open", "unm", "prec"):
        ks = [q for q in per[a] if q in per[b] and per[a][q][k] is not None and per[b][q][k] is not None]
        d = [per[a][q][k] - per[b][q][k] for q in ks]
        res[f"{a}_minus_{b[6:]}_{k}"] = {"cases": len(ks), "mean": round(sum(d) / len(d), 4), "ci95": boot(d)}
json.dump(res, open(f"/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor/results/pb_replay_order_x{SUFFIX}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
