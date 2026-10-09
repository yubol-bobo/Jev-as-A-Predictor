"""Phase-1 synthetic sequence tasks with exactly known predictive distributions.

Each item:
  id, task, cond (factors), group (cluster id for bootstrap), request (JEV payload),
  qid, options (ordered option names), gold (index of the realized next element),
  q (Bayes / generator predictive the item is scored against), q_true (latent truth if
  different), baselines {name: distribution}.
All randomness flows from one seeded numpy Generator, so item files are reproducible.
"""

import math

import numpy as np

LETTERS = ["A", "B", "C", "D", "E"]
WORDS = ["apple", "river", "stone", "cloud", "lamp"]

FRAMES = {
    "plain": ("Sequence so far (oldest to newest): {seq}",
              "Which element comes next in the sequence?"),
    "stochastic": ("The following sequence was produced by a random process. "
                   "Observations so far (oldest to newest): {seq}",
                   "What will the next observation be?"),
}


def _norm(v):
    v = np.asarray(v, dtype=float)
    return (v / v.sum()).tolist()


def _choice_request(seq_tokens, options, frame, rng, sep=" "):
    state_t, instr = FRAMES[frame]
    order = list(options)
    rng.shuffle(order)  # randomized option presentation order; scoring uses `options`
    return {"state": state_t.format(seq=sep.join(seq_tokens)),
            "questions": {"next": {"type": "choice", "instructions": instr,
                                   "criteria": {o: None for o in order}}}}


def gen_bern(rng, n_per=40):
    items = []
    for n in (5, 10, 20, 40):
        for frame in FRAMES:
            for s in range(n_per):
                p = rng.beta(1, 1)
                flips = rng.random(n + 1) < p
                seq = ["H" if f else "T" for f in flips[:n]]
                h = int(flips[:n].sum())
                state_t, _ = FRAMES[frame]
                req = {"state": state_t.format(seq=" ".join(seq)) + " (H = heads, T = tails)",
                       "questions": {"next": {"type": "noul",
                                              "instructions": "Will the next coin flip land heads?"}}}
                q = (h + 1) / (n + 2)
                items.append({
                    "id": f"bern-n{n}-{frame}-{s}", "task": "bern", "cond": {"n": n, "frame": frame},
                    "group": f"bern-n{n}-{s}", "request": req, "qid": "next", "options": ["true", "false"],
                    "gold": 0 if flips[n] else 1, "q": [q, 1 - q], "q_true": [p, 1 - p],
                    "baselines": {"uniform": [0.5, 0.5],
                                  "freq": [(h + 0.5) / (n + 1), (n - h + 0.5) / (n + 1)],
                                  "last": [0.75, 0.25] if flips[n - 1] else [0.25, 0.75]},
                })
    return items


def gen_markov(rng, n_per=20, alpha=0.5):
    items = []
    for K in (3, 5):
        for L in (10, 30, 60):
            for rep in ("letters", "words"):
                for frame in FRAMES:
                    for s in range(n_per):
                        sym = (LETTERS if rep == "letters" else WORDS)[:K]
                        T = rng.dirichlet([alpha] * K, size=K)
                        x = [int(rng.integers(K))]
                        for _ in range(L):
                            x.append(int(rng.choice(K, p=T[x[-1]])))
                        obs, nxt = x[:L], x[L]
                        last = obs[-1]
                        C = np.zeros((K, K))
                        for a, b in zip(obs[:-1], obs[1:]):
                            C[a, b] += 1
                        q = _norm(C[last] + alpha)
                        uni = _norm(np.bincount(obs, minlength=K) + 1)
                        items.append({
                            "id": f"markov-K{K}-L{L}-{rep}-{frame}-{s}", "task": "markov",
                            "cond": {"K": K, "L": L, "rep": rep, "frame": frame},
                            "group": f"markov-K{K}-L{L}-{s}",
                            "request": _choice_request([sym[i] for i in obs], sym, frame, rng),
                            "qid": "next", "options": sym, "gold": nxt, "q": q, "q_true": T[last].tolist(),
                            "baselines": {"uniform": [1 / K] * K, "unigram": uni,
                                          "last": _norm([0.5 if i == last else 0.5 / (K - 1) for i in range(K)])},
                        })
    return items


def gen_period(rng, n_per=20, K=4, L=24):
    items = []
    for p in (2, 3, 4):
        for eps in (0.0, 0.1, 0.2):
            for rep in ("letters", "words"):
                for s in range(n_per):
                    sym = (LETTERS if rep == "letters" else WORDS)[:K]
                    while True:
                        pat = [int(v) for v in rng.integers(K, size=p)]
                        if len(set(pat)) > 1:
                            break
                    x = []
                    for t in range(L + 1):
                        v = pat[t % p]
                        if rng.random() < eps:
                            v = int(rng.choice([k for k in range(K) if k != v]))
                        x.append(v)
                    obs, nxt = x[:L], x[L]
                    target = pat[L % p]
                    q = [1 - eps if k == target else eps / (K - 1) for k in range(K)]
                    phase = [obs[t] for t in range(L) if t % p == L % p]
                    items.append({
                        "id": f"period-p{p}-e{eps}-{rep}-{s}", "task": "period",
                        "cond": {"p": p, "eps": eps, "rep": rep, "frame": "plain"},
                        "group": f"period-p{p}-e{eps}-{s}",
                        "request": _choice_request([sym[i] for i in obs], sym, "plain", rng),
                        "qid": "next", "options": sym, "gold": nxt, "q": q,
                        "baselines": {"uniform": [1 / K] * K,
                                      "unigram": _norm(np.bincount(obs, minlength=K) + 1),
                                      "phase_count": _norm(np.bincount(phase, minlength=K) + 0.5)},
                    })
    return items


def _bin_masses(mu, sd, edges):
    cdf = [0.5 * (1 + math.erf((e - mu) / (sd * math.sqrt(2)))) for e in edges]
    cdf = [0.0] + cdf + [1.0]
    return _norm(np.maximum(np.diff(cdf), 1e-12))


def gen_ar1(rng, n_per=30, nbins=7):
    from statistics import NormalDist
    items = []
    for phi in (0.5, 0.9):
        for L in (20, 50):
            for s in range(n_per):
                sd_stat = 1 / math.sqrt(1 - phi ** 2)
                x = [rng.normal(0, sd_stat)]
                for _ in range(L):
                    x.append(phi * x[-1] + rng.normal())
                disp = [round(50 + 10 * v, 1) for v in x]          # what the model sees
                # equal-probability bins of the stationary marginal, in display units
                edges = [round(50 + 10 * sd_stat * NormalDist().inv_cdf(k / nbins), 1) for k in range(1, nbins)]
                levels = ([f"below {edges[0]}"] +
                          [f"between {edges[i]} and {edges[i + 1]}" for i in range(nbins - 2)] +
                          [f"above {edges[-1]}"])
                nxt = disp[L]
                gold = int(sum(nxt >= e for e in edges))
                obs = disp[:L]
                q = _bin_masses(50 + 10 * phi * x[L - 1], 10, edges)
                # learner baseline: OLS AR(1) plug-in on the observed series
                a, b = np.array(obs[:-1]) - 50, np.array(obs[1:]) - 50
                phi_hat = float((a * b).sum() / max((a * a).sum(), 1e-9))
                resid_sd = float(np.std(b - phi_hat * a)) or 10.0
                req = {"state": "Sequence so far (oldest to newest): " + ", ".join(f"{v:.1f}" for v in obs),
                       "questions": {"next": {"type": "score",
                                              "instructions": "What range will the next value of the sequence fall in?",
                                              "criteria": levels}}}
                items.append({
                    "id": f"ar1-phi{phi}-L{L}-{s}", "task": "ar1", "cond": {"phi": phi, "L": L, "frame": "plain"},
                    "group": f"ar1-phi{phi}-L{L}-{s}", "request": req, "qid": "next",
                    "options": [str(i) for i in range(nbins)], "gold": gold, "q": q,
                    "baselines": {"uniform": [1 / nbins] * nbins,
                                  "persistence": _bin_masses(obs[-1], 10, edges),
                                  "ols_ar1": _bin_masses(50 + phi_hat * (obs[-1] - 50), resid_sd, edges)},
                })
    return items


def gen_trend(rng, n_per=16):
    items = []
    rules = {
        "arith": lambda a, d, n: [a + d * i for i in range(n)],
        "geom": lambda a, d, n: [a * (d % 3 + 2) ** i for i in range(n)],
        "quad": lambda a, d, n: [a + d * i * i for i in range(n)],
        "fib": lambda a, d, n: _fib(a, a + d, n),
    }
    for name, f in rules.items():
        for s in range(n_per):
            a, d = int(rng.integers(1, 10)), int(rng.integers(2, 8))
            seq = f(a, d, 7)
            obs, ans = seq[:6], seq[6]
            step = abs(seq[6] - seq[5]) or 1
            linear = seq[5] + (seq[5] - seq[4])          # the "wrong rule" distractor
            pool = [linear, ans + 1, ans - 1, ans + step, ans - step, ans + 2, ans - 2, ans + 3]
            distract = []
            for c in pool:
                if c != ans and c not in distract:
                    distract.append(c)
            opts = [str(c) for c in sorted([ans] + distract[:4])]
            gold = opts.index(str(ans))
            items.append({
                "id": f"trend-{name}-{s}", "task": "trend", "cond": {"rule": name, "frame": "plain"},
                "group": f"trend-{name}-{s}",
                "request": _choice_request([str(v) for v in obs], opts, "plain", rng, sep=", "),
                "qid": "next", "options": opts, "gold": gold,
                "q": [1.0 if i == gold else 0.0 for i in range(len(opts))],
                "baselines": {"uniform": [1 / len(opts)] * len(opts)},
            })
    return items


def _fib(a, b, n):
    out = [a, b]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out


def generate_all(seed=20261009):
    rng = np.random.default_rng(seed)
    items = []
    for g in (gen_bern, gen_markov, gen_period, gen_ar1, gen_trend):
        items += g(rng)
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    return items
