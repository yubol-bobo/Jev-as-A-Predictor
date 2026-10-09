"""Recurrency baseline for ICEWS14 (after Gastinger et al., 2024, "History repeats itself").

  python scripts/run_icews14_recurrency.py

For a query (s, r, ?, t) with ground-truth history H = {facts with day < t}, each entity o gets
recency-weighted shares (weight 2^(-lambda * (t - t')) per past fact, normalized over all entities):
  strict(o)  = share of o among past objects of (s, r, ·)            [strict recurrency]
  so(o)      = share of o among past partners of s (any relation, either direction)
  ro(o)      = share of o among past objects of (·, r, ·)            [relaxed recurrency]
  score(o)   = alpha * strict(o) + (1 - alpha) * (beta * so(o) + (1 - beta) * ro(o))
Probabilities over the 20 candidates: softmax(log(score + eps) / T). lambda, alpha, beta, eps and T
are fitted jointly on the dev items by NLL. Predictions -> results/predictions/icews14/recurrency/.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from seqpred import items as I, metrics  # noqa: E402
from seqpred.icews14 import ICEWS14  # noqa: E402

TASK, SYSTEM = "icews14", "recurrency"
LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
ALPHAS = tuple(np.round(np.linspace(0, 1, 11), 2))
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
EPSS = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
TEMPS = np.exp(np.linspace(np.log(0.05), np.log(5), 61))


def components(ds, by_r, q, lam, cand=None):
    """(3, n) shares [strict, so, ro] for candidate entities `cand` (None = all entities)."""
    s, r, day = q["s"], q["r"], q["day"]

    def share(idx, other):
        out = np.zeros(ds.n_ent)
        if len(idx):
            w = np.exp2(-lam * (day - ds.day[idx]))
            np.add.at(out, other, w)
            out /= w.sum()
        return out

    i_sr = ds.before(ds.by_sr.get((s, r)), day)
    i_s = ds.before(ds.by_ent.get(s), day)
    i_r = ds.before(by_r.get(r), day)
    comp = np.stack([share(i_sr, ds.o[i_sr]),
                     share(i_s, np.where(ds.s[i_s] == s, ds.o[i_s], ds.s[i_s])),
                     share(i_r, ds.o[i_r])])
    return comp if cand is None else comp[:, cand]


def probs(comp, alpha, beta, eps, T):
    sc = alpha * comp[..., 0, :] + (1 - alpha) * (beta * comp[..., 1, :] + (1 - beta) * comp[..., 2, :])
    z = np.log(sc + eps) / T
    z -= z.max(-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(-1, keepdims=True)


def main():
    t0 = time.time()
    ds = ICEWS14()
    by_r = {}
    for r in range(ds.n_rel):
        by_r[r] = np.flatnonzero(ds.r == r)
    its = {sp: I.load_items(TASK, sp) for sp in ("dev", "test")}
    cand = {sp: [np.array(it["meta"]["option_ids"]) - 1 for it in v] for sp, v in its.items()}
    comps = {sp: np.stack([np.stack([components(ds, by_r, it["meta"]["query"], lam, c)
                                     for it, c in zip(its[sp], cand[sp])]) for lam in LAMBDAS])
             for sp in its}  # (n_lambda, n_items, 3, 20)
    gold = {sp: np.array([it["gold"] for it in v]) for sp, v in its.items()}

    def nll(p, g):
        return float(-np.log(np.maximum(p[np.arange(len(g)), g], 1e-12)).mean())

    best = (np.inf, None)
    for li, a, b, e, T in itertools.product(range(len(LAMBDAS)), ALPHAS, BETAS, EPSS, TEMPS):
        v = nll(probs(comps["dev"][li], a, b, e, T), gold["dev"])
        if v < best[0]:
            best = (v, (li, float(a), float(b), e, float(T)))
    li, a, b, e, T = best[1]
    params = {"lambda": LAMBDAS[li], "alpha": a, "beta": b, "eps": e, "T": round(T, 4), "dev_nll_fit": best[0]}
    print("fitted", params)

    report = {"task": TASK, "system": SYSTEM, "params": params,
              "grid": {"lambda": LAMBDAS, "alpha": list(map(float, ALPHAS)), "beta": BETAS, "eps": EPSS,
                       "T": [float(TEMPS[0]), float(TEMPS[-1]), len(TEMPS)]}}
    for sp in ("dev", "test"):
        P = probs(comps[sp][li], a, b, e, T)
        I.write_predictions(TASK, SYSTEM, sp, [{"id": it["id"], "p": p.round(6).tolist()} for it, p in zip(its[sp], P)])
        s = metrics.summarize(its[sp], I.load_predictions(TASK, SYSTEM, sp))
        report[sp] = {k: (v if not isinstance(v, tuple) else [round(x, 4) for x in v]) for k, v in s.items()}
        print(sp, {k: report[sp][k] for k in ("hr1", "ndcg10", "mrr", "nll", "ece")})

    # sanity: time-aware filtered MRR over all entities for the 1,000 test queries (ties averaged)
    rr = []
    for it in its["test"]:
        q = it["meta"]["query"]
        g = it["meta"]["option_ids"][it["gold"]] - 1
        comp = components(ds, by_r, q, LAMBDAS[li])
        sc = a * comp[0] + (1 - a) * (b * comp[1] + (1 - b) * comp[2])
        other = [o for o in ds.answers[(q["s"], q["r"], q["day"])] if o != g]
        sc[other] = -np.inf
        higher, ties = (sc > sc[g]).sum(), (sc == sc[g]).sum()
        rr.append(np.mean(1 / np.arange(higher + 1, higher + ties + 1)))
    report["test_full_ranking_filtered_mrr_1000q"] = round(float(np.mean(rr)), 4)
    report["seconds"] = round(time.time() - t0, 1)
    print("full-ranking filtered MRR (1000 test queries):", report["test_full_ranking_filtered_mrr_1000q"])
    (I.PREDS / TASK / SYSTEM / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
