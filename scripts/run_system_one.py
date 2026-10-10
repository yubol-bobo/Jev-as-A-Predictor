"""Untrained evaluation of an HTTP System One system on dev/test items of every task.

  python scripts/run_system_one.py --system jev
  SEQPRED_URL=http://127.0.0.1:26150 python scripts/run_system_one.py --system kev27b
  SEQPRED_URL=http://127.0.0.1:26210 python scripts/run_system_one.py --system pplx_decider

Writes results/predictions/<task>/<system>/{dev,test}.jsonl (+ report.json) and the raw responses to
results/raw/<system>/<task>_<split>.jsonl.gz. Errors are kept as {"status": "error"} rows and scored as
uniform by metrics.summarize; re-running retries only the failed requests (successes are cached).
"""

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics, systems  # noqa: E402

TASKS = ["ml1m", "beauty", "fsq_nyc", "icews14", "bpic12"]


def fmt(s):
    return {k: (round(v[0], 4) if isinstance(v, tuple) else v) for k, v in s.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(systems.SYSTEMS))
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    client = systems.HTTPSystemOne(a.system)
    workers = a.workers or client.workers
    for task in a.tasks.split(","):
        report = {"task": task, "system": a.system, "model": client.model}
        for split in a.splits.split(","):
            its = I.load_items(task, split)
            t0 = time.time()
            with ThreadPoolExecutor(workers) as pool:
                recs = list(pool.map(lambda it: client.call(it["request"]), its))
            rows, n_err = [], 0
            raw = I.ROOT / "results" / "raw" / a.system / f"{task}_{split}.jsonl.gz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(raw, "wt") as g:
                for it, rec in zip(its, recs):
                    g.write(json.dumps({"id": it["id"], **{k: v for k, v in rec.items() if k != "cached"}}) + "\n")
                    if rec["status"] == "ok" and it["qid"] in rec["response"]["answers"]:
                        rows.append({"id": it["id"],
                                     "p": systems.choice_distribution(rec["response"]["answers"][it["qid"]], it["options"])})
                    else:
                        n_err += 1
                        rows.append({"id": it["id"], "status": "error",
                                     "error": str(rec.get("response") or rec.get("error"))[:300]})
            I.write_predictions(task, a.system, split, rows)
            s = metrics.summarize(its, I.load_predictions(task, a.system, split))
            lat = sorted(r["latency_s"] for r in recs if r.get("latency_s") and not r.get("cached"))
            report[split] = {**fmt(s), "errors": n_err, "wall_s": round(time.time() - t0, 1),
                             "median_latency_s": lat[len(lat) // 2] if lat else None}
            print(task, split, report[split], flush=True)
        (I.PREDS / task / a.system / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
