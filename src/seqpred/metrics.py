"""Ranking and probability-quality metrics over a supplied candidate set.

Some systems (e.g. JEV) round probabilities to 0.01 and return exact zeros, so NLL
floors probabilities at EPS (half the rounding grid) and renormalizes. Ranking ties are
averaged over every position the gold option could occupy among tied candidates.
"""

import math

import numpy as np

EPS = 0.005


def item_metrics(p, gold):
    p = np.asarray(p, dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
    pf = np.maximum(p, EPS)
    pf = pf / pf.sum()
    higher = int((p > p[gold] + 1e-12).sum())
    ties = int((np.abs(p - p[gold]) <= 1e-12).sum())  # includes gold itself
    positions = np.arange(higher + 1, higher + ties + 1)
    return {
        "hr1": float(np.mean(positions == 1)),
        "hr5": float(np.mean(positions <= 5)),
        "ndcg10": float(np.mean([1 / math.log2(k + 1) if k <= 10 else 0.0 for k in positions])),
        "mrr": float(np.mean(1 / positions)),
        "rank": float(positions.mean()),
        "nll": float(-math.log(pf[gold])),
        "p_gold": float(p[gold]),
        "gold_zero": float(p[gold] < 1e-9),
        "conf": float(p.max()),
    }


def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return float(sum((bins == b).mean() * abs(conf[bins == b].mean() - correct[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def bootstrap_ci(values, groups=None, n_boot=2000, seed=0):
    """Mean and 95% percentile CI, resampling clusters (groups) when given."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        boots = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    else:
        groups = np.asarray(groups)
        members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
        boots = np.array([values[np.concatenate([members[i] for i in
                                                 rng.integers(0, len(members), len(members))])].mean()
                          for _ in range(n_boot)])
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize(items, preds):
    """Aggregate metrics for one system on one split. preds: {id: {"p": [...]}}."""
    rows, missing = [], 0
    for it in items:
        pr = preds.get(it["id"])
        if not pr or "p" not in pr:
            missing += 1
            p = [1 / len(it["options"])] * len(it["options"])  # missing answers count as uniform
        else:
            p = pr["p"]
        rows.append((it["group"], item_metrics(p, it["gold"])))
    groups = [g for g, _ in rows]
    out = {"n": len(rows), "missing": missing}
    for k in ("hr1", "hr5", "ndcg10", "mrr", "nll"):
        out[k] = bootstrap_ci([m[k] for _, m in rows], groups)
    out["ece"] = ece([m["conf"] for _, m in rows], [m["hr1"] for _, m in rows])
    out["gold_zero"] = float(np.mean([m["gold_zero"] for _, m in rows]))
    return out
