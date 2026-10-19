"""Per-task LoRA finetuning of a System One model with the shared recipe (src/seqpred/ft/common.py RECIPE).

  python scripts/finetune.py --system kev27b --task bpic12 --train-n 500            # train, fit T on dev, write dev/test
  python scripts/finetune.py --system kev27b --task bpic12 --train-n 500 --max-steps 3   # quick test (separate output name)
  python scripts/finetune.py --system kev27b --task bpic12,ml1m --eval-only-untrained-check   # forward-path parity
  python scripts/finetune.py --system kev27b --task bpic12,fsq_nyc --probe-longest 3   # memory / speed of the longest items
  python scripts/finetune.py --system kev27b --task ml1m,beauty,fsq_nyc,icews14,bpic12 --token-stats   # tokenizer only

Kev-27B runs in its own environment (see scripts/slurm/ft_kev27b.sbatch for the variables):
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 \
  /ocean/projects/cis260032p/yliy/research/jev/jev_open/kev_v2/.venv/bin/python scripts/finetune.py ...

Outputs of a training run (system name <system>_ft_n<N>, plus _steps<K> with --max-steps):
  results/predictions/<task>/<name>/{dev,test}.jsonl  probabilities (temperature fitted on dev) aligned with options
  results/predictions/<task>/<name>/report.json       metrics, T, recipe/config, timing, peak memory, loss curve
  results/checkpoints/<task>/<name>/                  LoRA weights + head (gitignored)
Checks: results/ft_checks/<system>/{parity_<task>,probe,token_stats}.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I  # noqa: E402
from seqpred.ft import ADAPTERS, get_adapter  # noqa: E402
from seqpred.ft import common as C  # noqa: E402

CHECKS = ROOT / "results" / "ft_checks"
TASKS = ["ml1m", "beauty", "fsq_nyc", "icews14", "bpic12"]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))
    log(f"wrote {path}")


def parity(adapter, task, n):
    """Untrained LoRA (B = 0): probabilities at the serving temperature vs the untrained server's dev predictions."""
    its = I.load_items(task, "dev", n)
    ref = I.load_predictions(task, adapter.system, "dev")
    z = C.raw_logits(adapter, its, log)
    T = adapter.serving_temperature
    diffs, agree, rows = [], [], []
    for it, zz in zip(its, z):
        p = C.softmax(zz, T)
        q = np.asarray(ref[it["id"]]["p"])
        d = np.abs(p - q)
        diffs.append(d)
        agree.append(int(np.argmax(p) == np.argmax(q)))
        rows.append({"id": it["id"], "max_abs": round(float(d.max()), 5), "argmax_agree": agree[-1]})
    allv = np.concatenate(diffs)
    out = {"task": task, "n": len(its), "temperature": T, "max_abs_diff": float(allv.max()),
           "mean_abs_diff": float(allv.mean()), "mean_per_item_max_abs_diff": float(np.mean([d.max() for d in diffs])),
           "argmax_agreement": float(np.mean(agree)), "items": rows}
    log(f"parity {task}: max|dp| {out['max_abs_diff']:.4f} mean|dp| {out['mean_abs_diff']:.5f} "
        f"argmax agree {out['argmax_agreement']:.3f}")
    return out


def token_stats(adapter, tasks, n):
    out = {}
    for task in tasks:
        its = I.load_items(task, "train", n)
        L = np.array([adapter.n_tokens(it) for it in its])
        out[task] = {"n": len(L), "mean": round(float(L.mean()), 1), "median": float(np.median(L)),
                     "p95": float(np.percentile(L, 95)), "max": int(L.max()),
                     "mean_first_500": round(float(L[:500].mean()), 1),
                     "mean_first_2000": round(float(L[:2000].mean()), 1),
                     "argmax_ids": [its[i]["id"] for i in np.argsort(-L)[:5]]}
        log(f"tokens {task}: {out[task]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=list(ADAPTERS))
    ap.add_argument("--task", required=True, help="one task (training); comma list for the checks")
    ap.add_argument("--train-n", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=C.RECIPE["epochs"])
    ap.add_argument("--max-steps", type=int, default=None, help="stop after K optimizer steps (quick tests)")
    ap.add_argument("--eval-only-untrained-check", action="store_true",
                    help="parity of the untrained LoRA model against the untrained server dev predictions")
    ap.add_argument("--parity-n", type=int, default=50)
    ap.add_argument("--probe-longest", type=int, default=0, help="memory/speed probe on the K longest train items per task")
    ap.add_argument("--probe-pool", type=int, default=10000, help="train items searched for the longest")
    ap.add_argument("--token-stats", action="store_true")
    ap.add_argument("--no-save", action="store_true", help="do not write the LoRA/head checkpoint")
    a = ap.parse_args()
    tasks = a.task.split(",")
    adapter = get_adapter(a.system)()

    if a.token_stats:
        write_json(CHECKS / a.system / "token_stats.json", token_stats(adapter, tasks, a.probe_pool))
        return

    log(f"loading {a.system}")
    t0 = time.time()
    adapter.load()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log(f"loaded in {time.time() - t0:.0f}s; {json.dumps(adapter.describe())}")

    if a.eval_only_untrained_check:
        for task in tasks:
            write_json(CHECKS / a.system / f"parity_{task}.json", parity(adapter, task, a.parity_n))
        return

    if C.RECIPE["gradient_checkpointing"]:
        adapter.enable_gradient_checkpointing()

    if a.probe_longest:
        res = {}
        for task in tasks:
            pool = I.load_items(task, "train", a.probe_pool)
            L = [adapter.n_tokens(it) for it in pool]
            longest = [pool[i] for i in np.argsort(L)[::-1][:a.probe_longest]]
            log(f"probe {task}: longest {[L[i] for i in np.argsort(L)[::-1][:a.probe_longest]]} tokens")
            res[task] = C.probe(adapter, longest, log)
        write_json(CHECKS / a.system / "probe.json", res)
        return

    assert len(tasks) == 1 and a.train_n, "training needs one --task and --train-n"
    task, N = tasks[0], a.train_n
    name = f"{a.system}_ft_n{N}" + (f"_steps{a.max_steps}" if a.max_steps else "")
    train_items = I.load_items(task, "train", N)
    assert len(train_items) == N, f"only {len(train_items)} train items"
    log(f"train {name}: {N} items, {a.epochs} epochs, recipe {json.dumps(C.RECIPE)}")
    hist = C.train(adapter, train_items, epochs=a.epochs, max_steps=a.max_steps, log=log)
    log(f"trained in {hist['train_wall_s']}s ({hist['seconds_per_optimizer_step']} s/step, "
        f"peak {hist['peak_mem_allocated_gib']} GiB)")
    if not a.no_save:
        ck_dir = ROOT / "results" / "checkpoints" / task / name
        adapter.save(ck_dir)
        log(f"saved {ck_dir}")
    T, summ, eval_s = C.evaluate(adapter, task, name, log)
    report = {"task": task, "system": name, "base_system": a.system, "train_n": N,
              "config": {"recipe": C.RECIPE, "epochs": a.epochs, "max_steps": a.max_steps,
                         "train_file": f"data/items/{task}/train.jsonl (first {N} lines)",
                         "model": adapter.describe()},
              "temperature": T, "dev": summ["dev"], "test": summ["test"], "dev_nll_T1": summ["dev_nll_T1"],
              "eval_wall_s": eval_s, **{k: v for k, v in hist.items() if k != "loss_curve"},
              "loss_curve": hist["loss_curve"]}
    write_json(I.PREDS / task / name / "report.json", report)


if __name__ == "__main__":
    main()
