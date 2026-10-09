"""Train SASRec for one task and write candidate-set predictions for dev and test.

Inputs (written by scripts/build_<task>.py):
  data/items/<task>/{dev,test,train}.jsonl     meta.option_ids / meta.history_ids are item indices
  data/items/<task>/train_sequences.jsonl.gz   one JSON list of item indices per line
  data/items/<task>/vocab.json                 {"n_items": int, ...}

  python scripts/run_sasrec.py --task ml1m --max-len 200 --dropout 0.2              # full data
  python scripts/run_sasrec.py --task ml1m --train-n 500 --system sasrec_n500      # matched-N

Full-data mode trains on every training sequence (loss at every position); --train-n trains
only on the first N train items as (history, target) pairs. Early stopping and the softmax
temperature both use the dev items. Predictions go to results/predictions/<task>/<system>/.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics, sasrec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--system", default="sasrec")
    ap.add_argument("--train-n", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    base = I.ITEMS / a.task
    n_items = json.loads((base / "vocab.json").read_text())["n_items"]
    dev, test = I.load_items(a.task, "dev"), I.load_items(a.task, "test")
    if a.train_n is None:
        with gzip.open(base / "train_sequences.jsonl.gz", "rt") as f:
            examples = [(json.loads(l), 1) for l in f]
        examples = [e for e in examples if len(e[0]) >= 2]
    else:
        tr = I.load_items(a.task, "train", n=a.train_n)
        assert len(tr) == a.train_n, f"only {len(tr)} train items"
        examples = [(it["meta"]["history_ids"] + [it["meta"]["option_ids"][it["gold"]]],
                     len(it["meta"]["history_ids"])) for it in tr]

    def hist_cands(its):
        return [it["meta"]["history_ids"] for it in its], [it["meta"]["option_ids"] for it in its]

    dh, dc = hist_cands(dev)
    dgold = [it["gold"] for it in dev]

    def eval_fn(model):  # dev NDCG@10 over the candidate sets
        sc = sasrec.score_candidates(model, dh, dc, a.device)
        return float(np.mean([metrics.item_metrics(sasrec.softmax(z), g)["ndcg10"] for z, g in zip(sc, dgold)]))

    t0 = time.time()
    model = sasrec.train(examples, n_items, max_len=a.max_len, hidden=a.hidden, dropout=a.dropout, lr=a.lr,
                         batch_size=a.batch_size, max_epochs=a.max_epochs, eval_fn=eval_fn,
                         eval_every=a.eval_every, patience=a.patience, device=a.device, seed=a.seed)
    T = sasrec.fit_temperature(sasrec.score_candidates(model, dh, dc, a.device), dgold)
    report = {"task": a.task, "system": a.system, "args": vars(a), "n_examples": len(examples),
              "temperature": T, "train_seconds": round(time.time() - t0, 1)}
    for split, its in (("dev", dev), ("test", test)):
        h, c = hist_cands(its)
        sc = sasrec.score_candidates(model, h, c, a.device)
        I.write_predictions(a.task, a.system, split,
                            [{"id": it["id"], "p": sasrec.softmax(z, T).round(6).tolist()} for it, z in zip(its, sc)])
        s = metrics.summarize(its, I.load_predictions(a.task, a.system, split))
        report[split] = {k: (v if not isinstance(v, tuple) else [round(x, 4) for x in v]) for k, v in s.items()}
        print(split, {k: report[split][k] for k in ("hr1", "ndcg10", "mrr", "nll", "ece")})
    out = I.PREDS / a.task / a.system / "report.json"
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
