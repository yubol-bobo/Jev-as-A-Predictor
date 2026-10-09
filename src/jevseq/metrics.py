"""Scoring rules for predictive distributions over a finite option set.

JEV rounds probabilities to 0.01 and often returns exact zeros, so every
log-based metric floors probabilities at EPS (half the rounding grid) and
renormalizes. `gold_zero` records how often the realized outcome got 0.00.
"""

import math

import numpy as np

EPS = 0.005


def floored(p, eps=EPS):
    v = np.maximum(np.asarray(p, dtype=float), eps)
    return v / v.sum()


def item_metrics(p, gold_idx, q=None):
    """p, q: probability vectors over the same options; gold_idx: realized outcome."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.full_like(p, 1 / len(p))
    pf = floored(p)
    onehot = np.zeros_like(p)
    onehot[gold_idx] = 1
    top = int(np.argmax(p))
    tied = np.flatnonzero(p >= p.max() - 1e-12)  # ties are broken uniformly (in expectation)
    m = {
        "acc": float(np.isin(gold_idx, tied)) / len(tied),
        "conf": float(p[top]),
        "nll": float(-math.log(pf[gold_idx])),
        "brier": float(((p - onehot) ** 2).sum()),
        "gold_zero": float(p[gold_idx] < 1e-9),
        "p_gold": float(p[gold_idx]),
    }
    if q is not None:
        q = np.asarray(q, dtype=float)
        mask = q > 0
        m["kl_q"] = float((q[mask] * (np.log(q[mask]) - np.log(pf[mask]))).sum())
        m["exp_acc"] = float(q[tied].mean())  # expected accuracy of p's argmax under the true q
        m["exp_nll"] = float(-(q * np.log(pf)).sum())  # cross-entropy H(q, p)
        m["tv_q"] = float(0.5 * np.abs(q - p).sum())
    return m


def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            total += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(total)


def rps(p, gold_idx):
    """Ranked probability score for ordinal outcomes (lower is better)."""
    cp = np.cumsum(np.asarray(p, dtype=float))
    co = np.cumsum(np.eye(len(p))[gold_idx])
    return float(((cp - co) ** 2)[:-1].sum() / (len(p) - 1))


def bootstrap_ci(values, groups=None, n_boot=2000, seed=0):
    """Mean and 95% percentile CI; resamples groups (clusters) when given."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, len(values), (n_boot, len(values)))
        boots = values[idx].mean(1)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = [np.where(groups == g)[0] for g in uniq]
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            boots.append(values[np.concatenate([members[i] for i in pick])].mean())
        boots = np.asarray(boots)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
