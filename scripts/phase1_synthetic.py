"""Phase 1: JEV on synthetic sequences with exact predictive distributions.

  python scripts/phase1_synthetic.py generate   # freeze data/items/phase1_synthetic.jsonl
  python scripts/phase1_synthetic.py run        # call JEV (cached) -> results/phase1/predictions.jsonl
  python scripts/phase1_synthetic.py analyze    # tables -> results/phase1/summary.{md,json}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevseq import client, metrics, synthetic  # noqa: E402

ITEMS = ROOT / "data" / "items" / "phase1_synthetic.jsonl"
OUT = ROOT / "results" / "phase1"


def load_items():
    return [json.loads(l) for l in ITEMS.open()]


def generate():
    items = synthetic.generate_all()
    ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with ITEMS.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    counts = defaultdict(int)
    for it in items:
        counts[it["task"]] += 1
    print(f"wrote {len(items)} items -> {ITEMS}: {dict(counts)}")


def run():
    items = load_items()
    calls = client.run_items(items, workers=16)
    OUT.mkdir(parents=True, exist_ok=True)
    n_err = 0
    with (OUT / "predictions.jsonl").open("w") as f:
        for it in items:
            rec = calls[it["id"]]
            row = {"id": it["id"], "status": rec["status"], "latency_s": rec.get("latency_s"),
                   "usage": rec.get("usage"), "key": rec["key"]}
            if rec["status"] == "ok":
                q = it["request"]["questions"][it["qid"]]
                d = client.distribution(rec["response"]["answers"][it["qid"]], q)
                row["p"] = [d[o] for o in it["options"]]
            else:
                n_err += 1
                row["error"] = str(rec.get("response") or rec.get("error"))[:500]
            f.write(json.dumps(row) + "\n")
    print(f"done: {len(items)} items, {n_err} errors")


def _cell_key(it, factors):
    return tuple(it["cond"].get(k) for k in factors)


FACTORS = {"bern": ["n", "frame"], "markov": ["K", "L", "rep", "frame"], "period": ["p", "eps", "rep"],
           "ar1": ["phi", "L"], "trend": ["rule"]}
MAIN = {"bern": ["n"], "markov": ["L"], "period": ["eps"], "ar1": ["phi", "L"], "trend": ["rule"]}


def analyze():
    items = {it["id"]: it for it in load_items()}
    preds = {r["id"]: r for r in map(json.loads, (OUT / "predictions.jsonl").open())}
    rows = []  # one row per (item, predictor)
    for iid, it in items.items():
        pr = preds.get(iid)
        predictors = {"bayes": it["q"], **it["baselines"]}
        if "q_true" in it:
            predictors["oracle_true"] = it["q_true"]
        if pr and pr["status"] == "ok":
            predictors["jev"] = pr["p"]
        for name, p in predictors.items():
            m = metrics.item_metrics(p, it["gold"], it["q"])
            if it["task"] == "ar1":
                m["rps"] = metrics.rps(np.asarray(p) / np.sum(p), it["gold"])
            rows.append({"id": iid, "task": it["task"], "cond": it["cond"], "group": it["group"],
                         "predictor": name, **m})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "item_metrics.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    keys = ["exp_acc", "acc", "exp_nll", "kl_q", "tv_q", "brier", "gold_zero", "rps"]
    summary, md = {}, ["# Phase 1 — synthetic sequences: JEV vs exact Bayes predictive\n",
                       "exp_acc = q(argmax p) (expected accuracy under the true predictive); exp_nll = H(q, p) "
                       "with p floored at 0.005; kl_q = KL(q‖p). Lower is better except accuracy. "
                       "95% CIs bootstrap over sequences (groups).\n"]
    for task in FACTORS:
        trows = [r for r in rows if r["task"] == task]
        md.append(f"\n## {task}\n")
        # overall per predictor
        md.append("| predictor | n | exp_acc | acc | exp_nll | KL(q‖p) | TV | brier | gold@0 | ECE |" +
                  (" RPS |" if task == "ar1" else ""))
        md.append("|---|---|---|---|---|---|---|---|---|---|" + ("---|" if task == "ar1" else ""))
        for name in sorted({r["predictor"] for r in trows}, key=lambda n: (n != "jev", n != "bayes", n)):
            pr = [r for r in trows if r["predictor"] == name]
            s = {k: metrics.bootstrap_ci([r[k] for r in pr], [r["group"] for r in pr])
                 for k in keys if k in pr[0]}
            s["ece"] = metrics.ece([r["conf"] for r in pr], [r["acc"] for r in pr])
            s["n"] = len(pr)
            summary[f"{task}/{name}"] = s
            fmt = lambda k: f"{s[k][0]:.3f} [{s[k][1]:.3f},{s[k][2]:.3f}]" if k in ("exp_acc", "exp_nll") else f"{s[k][0]:.3f}"
            md.append(f"| **{name}** | {s['n']} | {fmt('exp_acc')} | {fmt('acc')} | {fmt('exp_nll')} | "
                      f"{fmt('kl_q')} | {fmt('tv_q')} | {fmt('brier')} | {fmt('gold_zero')} | {s['ece']:.3f} |" +
                      (f" {fmt('rps')} |" if task == "ar1" else ""))
        # breakdown by the main factor(s), JEV vs bayes vs best baseline
        md.append(f"\nBy {', '.join(FACTORS[task])} (exp_acc / exp_nll):\n")
        names = ["jev", "bayes"] + [n for n in items[next(i for i in items if items[i]['task'] == task)]["baselines"]]
        md.append("| cell | " + " | ".join(names) + " |")
        md.append("|---" * (len(names) + 1) + "|")
        cells = sorted({_cell_key(items[r["id"]], FACTORS[task]) for r in trows}, key=str)
        for c in cells:
            line = [str(dict(zip(FACTORS[task], c)))]
            for n in names:
                pr = [r for r in trows if r["predictor"] == n and _cell_key(items[r["id"]], FACTORS[task]) == c]
                line.append(f"{np.mean([r['exp_acc'] for r in pr]):.2f} / {np.mean([r['exp_nll'] for r in pr]):.2f}"
                            if pr else "–")
            md.append("| " + " | ".join(line) + " |")
    lat = [p["latency_s"] for p in preds.values() if p.get("latency_s")]
    if lat:
        md.append(f"\nJEV latency (fresh calls only): median {np.median(lat):.3f}s, p95 {np.percentile(lat, 95):.3f}s, n={len(lat)}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    {"generate": generate, "run": run, "analyze": analyze}[sys.argv[1]]()
