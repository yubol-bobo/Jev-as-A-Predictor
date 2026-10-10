"""Untrained evaluation of Cloudflare Clef (27B) with its pinned native joint-head code, in process.

Uses the same scorer as the earlier JEV studies (s1eval.clef_system.ClefScorer: release
joint_schema_model.py, bf16, sdpa, full-precision softmax over the native logits). Run with the
environment from judge_study_20260920/open_s1_20261008/gpu_b_clef.sh:

  CUDA_VISIBLE_DEVICES=4 HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 \
  /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/run_clef.py --batch 4

Outputs match scripts/run_system_one.py: results/predictions/<task>/clef27b/{dev,test}.jsonl + report.json
and raw native logits in results/raw/clef27b/<task>_<split>.jsonl.gz. Finished task/splits are skipped.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UCJ = Path("/ocean/projects/cis260032p/yliy/research/jev/understanding_capability_of_jev")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(UCJ / "src"))
from seqpred import items as I, metrics  # noqa: E402

SYSTEM = "clef27b"
TASKS = ["ml1m", "beauty", "fsq_nyc", "icews14", "bpic12"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    from s1eval.clef_system import ClefScorer
    scorer = ClefScorer()
    for task in a.tasks.split(","):
        rep_path = I.PREDS / task / SYSTEM / "report.json"
        report = json.loads(rep_path.read_text()) if rep_path.exists() else {"task": task, "system": SYSTEM}
        for split in a.splits.split(","):
            if split in report:
                print("skip", task, split, flush=True)
                continue
            its = I.load_items(task, split)
            order = sorted(range(len(its)), key=lambda i: len(json.dumps(its[i]["request"])))  # less padding
            out = [None] * len(its)
            t0 = time.time()
            for b in range(0, len(order), a.batch):
                idx = order[b:b + a.batch]
                try:
                    outputs = scorer.run_items([dict(its[i]["request"], model="clef") for i in idx])
                except Exception as e:  # keep going; scored as uniform
                    outputs = [({"status": "error", "error": repr(e)[:500]}, [])] * len(idx)
                for i, (call, results) in zip(idx, outputs):
                    out[i] = (call, results)
                if (b // a.batch) % 50 == 0:
                    print(time.strftime("%H:%M:%S"), task, split, b + len(idx), "/", len(its), flush=True)
            rows, n_err = [], 0
            raw = I.ROOT / "results" / "raw" / SYSTEM / f"{task}_{split}.jsonl.gz"
            raw.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(raw, "wt") as g:
                for it, (call, results) in zip(its, out):
                    g.write(json.dumps({"id": it["id"], "status": call.get("status"),
                                        "input_tokens": call.get("input_tokens"),
                                        "native_logits": call.get("native_logits"),
                                        "error": call.get("error")}) + "\n")
                    if call.get("status") == "ok" and results:
                        d = results[0]["distribution"]
                        rows.append({"id": it["id"], "p": [float(d.get(o, 0.0)) for o in it["options"]]})
                    else:
                        n_err += 1
                        rows.append({"id": it["id"], "status": "error", "error": str(call.get("error"))[:300]})
            I.write_predictions(task, SYSTEM, split, rows)
            s = metrics.summarize(its, I.load_predictions(task, SYSTEM, split))
            report[split] = {**{k: (round(v[0], 4) if isinstance(v, tuple) else v) for k, v in s.items()},
                             "errors": n_err, "wall_s": round(time.time() - t0, 1)}
            print(task, split, report[split], flush=True)
            rep_path.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
