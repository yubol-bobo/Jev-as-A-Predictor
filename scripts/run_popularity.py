"""Popularity baseline: p(candidate) ∝ (training count + 1), normalised over the candidate set.

Counts come from data/items/<task>/train_sequences.jsonl.gz (dev/test targets excluded).

  python scripts/run_popularity.py --task ml1m
"""

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--system", default="popularity")
    a = ap.parse_args()
    counts = Counter()
    with gzip.open(I.ITEMS / a.task / "train_sequences.jsonl.gz", "rt") as f:
        for line in f:
            counts.update(json.loads(line))
    report = {"task": a.task, "system": a.system}
    for split in ("dev", "test"):
        its = I.load_items(a.task, split)
        rows = []
        for it in its:
            c = np.array([counts[i] + 1 for i in it["meta"]["option_ids"]], dtype=float)
            rows.append({"id": it["id"], "p": (c / c.sum()).round(6).tolist()})
        I.write_predictions(a.task, a.system, split, rows)
        s = metrics.summarize(its, I.load_predictions(a.task, a.system, split))
        report[split] = {k: (v if not isinstance(v, tuple) else [round(x, 4) for x in v]) for k, v in s.items()}
        print(split, {k: report[split][k] for k in ("hr1", "ndcg10", "mrr", "nll", "ece")})
    (I.PREDS / a.task / a.system / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
