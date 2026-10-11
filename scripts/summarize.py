"""Collect every system's metrics on every task into results/summary_<split>.md (+ .json).

  python scripts/summarize.py            # test split
  python scripts/summarize.py --split dev
Metrics are recomputed from the prediction files (not copied from report.json), with 95% cluster
bootstrap CIs over groups (users / cases / query subjects).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics  # noqa: E402

TASKS = ["ml1m", "beauty", "fsq_nyc", "icews14", "bpic12"]
UNTRAINED = ["jev", "kev27b", "pplx_decider", "clef27b"]
BASELINES = ["popularity", "majority", "ngram", "sasrec", "recurrency", "regcn"]
KEYS = ["hr1", "ndcg10", "mrr", "nll"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out, md = {}, [f"# Untrained System One models vs baselines ({a.split})\n",
                   "HR@1 / NDCG@10 / MRR (higher is better), NLL (lower is better; probabilities floored at 0.005), "
                   "ECE (15 bins, top-1). 95% cluster-bootstrap CI shown for HR@1. Missing/error answers count as uniform.\n"]
    for task in TASKS:
        its = I.load_items(task, a.split)
        md += [f"\n## {task} (n={len(its)}, {len(its[0]['options'])} options)\n",
               "| system | HR@1 [95% CI] | NDCG@10 | MRR | NLL | ECE | missing |", "|---|---|---|---|---|---|---|"]
        for sysname in UNTRAINED + BASELINES:
            path = I.PREDS / task / sysname / f"{a.split}.jsonl"
            if not path.exists():
                continue
            s = metrics.summarize(its, I.load_predictions(task, sysname, a.split))
            out[f"{task}/{sysname}"] = s
            tag = "**" if sysname in UNTRAINED else ""
            md.append(f"| {tag}{sysname}{tag} | {s['hr1'][0]:.3f} [{s['hr1'][1]:.3f}, {s['hr1'][2]:.3f}] | "
                      f"{s['ndcg10'][0]:.3f} | {s['mrr'][0]:.3f} | {s['nll'][0]:.3f} | {s['ece']:.3f} | {s['missing']} |")
    (ROOT / "results" / f"summary_{a.split}.json").write_text(json.dumps(out, indent=1))
    (ROOT / "results" / f"summary_{a.split}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
