"""Shared finetuning recipe, training loop, temperature fitting and evaluation for every System One model.

Everything that must be identical across Kev-27B, pplx-decider and Clef lives here: the recipe constants (RECIPE), the
LoRA configuration (lora_config), the loss, optimizer, schedule and loop (train), the dev temperature fit and the
prediction / report writing (evaluate). A model plugs in through an `Adapter` subclass, which only knows how to load its
released checkpoint with a fresh LoRA from lora_config(), how to turn an item into option logits (its native forward
path and decision head), and how to save / reload its trained weights.

Adapter contract (see Adapter below):
  load()                 build the released model on the device, attach get_peft_model(backbone, lora_config()), keep the
                         backbone frozen (bf16) and the decision head trainable (warm-started from the release)
  param_groups()         {"lora": [...], "head": [...]} trainable parameters; nothing else may require grad
  option_logits(item)    raw head logits, shape [len(item["options"])], aligned with item["options"] (not with the
                         request's presentation order), temperature 1. Called inside the recipe's bf16 autocast; an
                         adapter may run its head in fp32 by disabling autocast locally. Gradients flow in train mode.
  train_mode()/eval_mode(), enable_gradient_checkpointing()
  save(dir)/load_trained(dir)
  n_tokens(item)         tokens of the item's request as this model encodes it (for cost estimates)
  serving_temperature    the temperature the released checkpoint serves with (None if it has none)
  describe()             JSON-able provenance (model ids, revisions, LoRA module names, parameter counts)
"""

import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .. import items as I, metrics
from ..sasrec import fit_temperature, softmax

# Qwen3.8-27B (hybrid Qwen3.5 architecture; all three models share this backbone) module names LoRA attaches to: every
# attention, MLP and Gated DeltaNet projection. This is Kev's lora_targets="all" on a hybrid backbone (kev.model
# DecisionModel.__init__); the other two adapters must target exactly these names.
LORA_TARGETS_ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
LORA_TARGETS_MLP = ["gate_proj", "up_proj", "down_proj"]
LORA_TARGETS_DELTANET = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
LORA_TARGETS = LORA_TARGETS_ATTENTION + LORA_TARGETS_MLP + LORA_TARGETS_DELTANET

RECIPE = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": LORA_TARGETS,
    "backbone_dtype": "bf16 (frozen)",
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "lr_lora": 1e-4,
    "lr_head": 2e-5,
    "warmup_frac": 0.1,           # linear warmup over the first 10% of optimizer steps, then cosine decay to 0
    "grad_clip": 1.0,             # global norm over LoRA + head
    "micro_batch": 1,             # records per forward pass
    "grad_accum": 8,              # effective batch 8
    "epochs": 2,
    "seed": 0,
    "gradient_checkpointing": True,
    "autocast": "bf16",
    "loss": "cross-entropy over the item's options on raw head logits (T=1)",
    "temperature_fit": "single T on dev NLL, grid exp(linspace(log 0.05, log 20, 121)) (seqpred.sasrec.fit_temperature)",
}


def lora_config(task_type="FEATURE_EXTRACTION"):
    """The one LoRA configuration every model uses (peft.LoraConfig)."""
    from peft import LoraConfig
    return LoraConfig(task_type=task_type, r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"],
                      lora_dropout=RECIPE["lora_dropout"], target_modules=list(RECIPE["lora_targets"]))


def lora_module_names(model):
    """Names of the modules that received a LoRA adapter (for provenance / cross-model comparison)."""
    return sorted(n for n, m in model.named_modules() if hasattr(m, "lora_A") and len(getattr(m, "lora_A")) > 0)


class Adapter:
    """Base class; see the module docstring for the contract."""
    system = None
    serving_temperature = None

    def __init__(self, device="cuda"):
        self.device = device

    def load(self):
        raise NotImplementedError

    def param_groups(self):
        raise NotImplementedError

    def option_logits(self, item):
        raise NotImplementedError

    def train_mode(self):
        raise NotImplementedError

    def eval_mode(self):
        raise NotImplementedError

    def enable_gradient_checkpointing(self):
        raise NotImplementedError

    def save(self, out_dir):
        raise NotImplementedError

    def load_trained(self, out_dir):
        raise NotImplementedError

    def n_tokens(self, item):
        raise NotImplementedError

    def describe(self):
        return {"system": self.system}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def lr_lambda(total_steps, warmup_frac):
    """Linear warmup (step 1 already has lr > 0) then cosine decay to 0 at the last step."""
    warm = max(1, math.ceil(warmup_frac * total_steps))

    def f(s):  # s = optimizer steps taken so far
        if s < warm:
            return (s + 1) / warm
        prog = (s - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def item_loss(adapter, item):
    z = adapter.option_logits(item).float()
    assert z.shape == (len(item["options"]),), f"{item['id']}: logits {tuple(z.shape)} vs {len(item['options'])} options"
    return F.cross_entropy(z[None], torch.tensor([item["gold"]], device=z.device))


def make_optimizer(adapter):
    g = adapter.param_groups()
    return torch.optim.AdamW([{"params": g["lora"], "lr": RECIPE["lr_lora"]},
                              {"params": g["head"], "lr": RECIPE["lr_head"]}],
                             weight_decay=RECIPE["weight_decay"])


def trainable(adapter):
    g = adapter.param_groups()
    return g["lora"] + g["head"]


def training_order(n, epochs, seed):
    """Record indices for the whole run: one seeded permutation per epoch, concatenated."""
    order = []
    for ep in range(epochs):
        perm = list(range(n))
        random.Random(seed * 1000 + ep).shuffle(perm)
        order += perm
    return order


def train(adapter, train_items, epochs=None, max_steps=None, log=print):
    """The shared training loop. -> history dict (loss curve per optimizer step, timing, peak memory)."""
    epochs = epochs or RECIPE["epochs"]
    accum = RECIPE["grad_accum"]
    seed_everything(RECIPE["seed"])
    params = trainable(adapter)
    opt = make_optimizer(adapter)
    order = training_order(len(train_items), epochs, RECIPE["seed"])
    total = math.ceil(len(order) / accum)
    if max_steps:
        total = min(total, max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, RECIPE["warmup_frac"]))
    adapter.train_mode()
    torch.cuda.reset_peak_memory_stats()
    curve, mb_seconds = [], []
    t_start = time.time()
    for step in range(total):
        chunk = order[step * accum:(step + 1) * accum]
        torch.cuda.synchronize()
        t0 = time.time()
        losses = []
        for idx in chunk:
            tm = time.time()
            with autocast():
                loss = item_loss(adapter, train_items[idx])
            (loss / len(chunk)).backward()
            losses.append(loss.item())
            mb_seconds.append(time.time() - tm)
        gn = float(torch.nn.utils.clip_grad_norm_(params, RECIPE["grad_clip"], error_if_nonfinite=True))
        lrs = [g["lr"] for g in opt.param_groups]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rec = {"step": step + 1, "epoch": (step * accum) // len(train_items) + 1, "loss": float(np.mean(losses)),
               "n": len(chunk), "grad_norm": gn, "lr_lora": lrs[0], "lr_head": lrs[1], "seconds": round(dt, 3)}
        curve.append(rec)
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == total:
            log(f"step {step + 1}/{total} ep {rec['epoch']} loss {rec['loss']:.4f} gn {gn:.3f} "
                f"lr {lrs[0]:.2e}/{lrs[1]:.2e} {dt:.1f}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
    wall = time.time() - t_start
    return {"optimizer_steps": total, "records_seen": min(len(order), total * accum), "epochs": epochs,
            "train_wall_s": round(wall, 1), "seconds_per_optimizer_step": round(wall / max(1, total), 3),
            "seconds_per_microbatch_mean": round(float(np.mean(mb_seconds)), 3),
            "peak_mem_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_mem_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "loss_curve": curve}


def probe(adapter, probe_items, log=print):
    """Memory / speed check: forward + backward on each item (micro-batch 1, training mode, checkpointing as configured),
    then one AdamW step so optimizer state is counted. Weights change; discard the model afterwards."""
    params = trainable(adapter)
    opt = make_optimizer(adapter)
    adapter.train_mode()
    torch.cuda.reset_peak_memory_stats()
    out = []
    for it in probe_items:
        torch.cuda.synchronize()
        t0 = time.time()
        with autocast():
            loss = item_loss(adapter, it)
        loss.backward()
        torch.cuda.synchronize()
        out.append({"id": it["id"], "tokens": adapter.n_tokens(it), "options": len(it["options"]),
                    "seconds": round(time.time() - t0, 3), "loss": round(loss.item(), 4),
                    "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
        log(json.dumps(out[-1]))
    torch.nn.utils.clip_grad_norm_(params, RECIPE["grad_clip"])
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return {"items": out, "peak_mem_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_mem_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}


@torch.no_grad()
def raw_logits(adapter, items, log=print, every=200):
    """Raw (T=1) option logits for every item, as numpy arrays aligned with item["options"]."""
    adapter.eval_mode()
    out = []
    t0 = time.time()
    for i, it in enumerate(items):
        with autocast():
            out.append(adapter.option_logits(it).float().cpu().numpy())
        if (i + 1) % every == 0:
            log(f"  logits {i + 1}/{len(items)} {time.time() - t0:.0f}s")
    return out


def prob_rows(items, logits, T):
    return [{"id": it["id"], "p": [float(x) for x in softmax(z, T)]} for it, z in zip(items, logits)]


def fmt_summary(s):
    """metrics.summarize output -> JSON: point estimates plus [lo, hi] 95% CIs."""
    out = {}
    for k, v in s.items():
        if isinstance(v, tuple):
            out[k] = round(v[0], 4)
            out[k + "_ci"] = [round(v[1], 4), round(v[2], 4)]
        else:
            out[k] = v
    return out


def evaluate(adapter, task, system_name, log=print):
    """Dev logits -> fit T on dev NLL -> write dev and test predictions; -> (T, {split: summary}, eval seconds)."""
    t0 = time.time()
    dev, test = I.load_items(task, "dev"), I.load_items(task, "test")
    z_dev = raw_logits(adapter, dev, log)
    T = fit_temperature(z_dev, [it["gold"] for it in dev])
    z_test = raw_logits(adapter, test, log)
    summ = {}
    for split, its, z in (("dev", dev, z_dev), ("test", test, z_test)):
        I.write_predictions(task, system_name, split, prob_rows(its, z, T))
        summ[split] = fmt_summary(metrics.summarize(its, I.load_predictions(task, system_name, split)))
        log(f"{task} {split} {system_name} T={T:.3f} " +
            " ".join(f"{k}={summ[split][k]}" for k in ("hr1", "ndcg10", "mrr", "nll", "ece")))
    summ["dev_nll_T1"] = round(float(np.mean([-np.log(softmax(z, 1.0)[it["gold"]] + 1e-12)
                                              for it, z in zip(dev, z_dev)])), 4)
    return T, summ, round(time.time() - t0, 1)
