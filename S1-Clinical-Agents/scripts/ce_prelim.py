"""Preliminary ClinEnv numbers for the paper (aggregates only): how often K=5 LLM samples at submit steps are really
different decisions once mapped to canonical codes, and how Clef picks on the few real choices.
Source: data/offline/<slug>/items.jsonl from scripts/offline_decisions.py build (online clef run 1 by default).

  .venv/bin/python -I scripts/ce_prelim.py [--slug qwen3.8-27b__clef_k5__run1]   -> results/ce_prelim.json
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--slug", default="qwen3.8-27b__clef_k5__run1")
a = ap.parse_args()
its = [json.loads(l) for l in open(f"/ocean/projects/cis260032p/yliy/data/offline/{a.slug}/items.jsonl")]
ok = lambda s: s >= 0.5
n = len(its)
one = sum(len(i["canon"]["options"]) == 1 for i in its)
real = [i for i in its if len(set(map(ok, i["raw"]["scores"]))) > 1 and i["online_distribution"]]
pos = [p for i in real for p, s in zip(i["online_distribution"], i["raw"]["scores"]) if ok(s)]
neg = [p for i in real for p, s in zip(i["online_distribution"], i["raw"]["scores"]) if not ok(s)]
res = {"SubmitSteps": n, "Cases": len({(i["subject"], i["hadm"]) for i in its}),
       "DistinctRaw": f"{100 * sum(len(i['raw']['options']) > 1 for i in its) / n:.0f}",
       "OneDecision": f"{100 * one / n:.0f}", "TwoPlus": n - one, "RealChoice": len(real),
       "RealChoicePct": f"{100 * len(real) / n:.0f}",
       "ClefPickCorrect": f"{100 * sum(ok(i['raw']['scores'][i['online_pick']]) for i in its) / n:.0f}",
       "FirstCorrect": f"{100 * sum(ok(i['raw']['scores'][0]) for i in its) / n:.0f}",
       "ClefRealAcc": f"{100 * sum(ok(i['raw']['scores'][i['online_pick']]) for i in real) / max(len(real), 1):.0f}",
       "RandRealAcc": f"{100 * sum(sum(map(ok, i['raw']['scores'])) / len(i['raw']['scores']) for i in real) / max(len(real), 1):.0f}",
       "ClefRealAUROC": f"{sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / max(len(pos) * len(neg), 1):.2f}"}
Path("results/ce_prelim.json").write_text(json.dumps(res, indent=1))
print(res)
