# JEV as a Predictor / JEV 作为序列预测器

Can **JEV** (`jev-1.13.0`, TypeSafe System One — state + typed questions in,
probability distributions out, no decoding, ~0.17 s/call) be used or adapted as a
good predictor for **sequential prediction** (next symbol / next item / next event /
next value)?

JEV 能否（或经过适配后能否）成为一个好的**序列预测器**？

- Research plan / 研究计划: [docs/00_research_plan.md](docs/00_research_plan.md)
- Code: `src/jevseq/` (cached JEV client, metrics, task generators), `scripts/`
- Results: `results/<phase>/summary.md`

## Progress log / 进展记录

| Date (UTC) | Step | Status | Where |
|---|---|---|---|
| 2026-10-09 | Plan, cached JEV client, metrics | ✅ | `docs/00_research_plan.md`, `src/jevseq/` |
| 2026-10-09 | **Phase 1** — 1,344 synthetic sequences with exact Bayes predictive (5 tasks) | ✅ run + analyzed | [results/phase1/summary.md](results/phase1/summary.md) |
| next | Phase 1b — adaptation on synthetic (count-summary states, calibration) | ⏳ | |
| next | Phase 2 — real benchmarks (MovieLens-1M next item, process-mining next activity) | ⏳ | |

## Key findings so far / 目前的主要发现

**Phase 1 (synthetic, exact ground truth; 1,344 JEV calls, median latency 0.17 s).**
Metric `exp_acc` = probability under the true predictive that JEV's argmax is right;
`exp_nll` = cross-entropy H(q, p_JEV) (p floored at 0.005).

| task | JEV exp_acc | Bayes-optimal | best simple baseline | JEV exp_nll | Bayes exp_nll | uniform exp_nll |
|---|---|---|---|---|---|---|
| Bernoulli coin (noul) | 0.706 | 0.726 | freq 0.726 | 0.645 | 0.542 | 0.693 |
| order-1 Markov (K=3/5) | 0.474 | 0.586 | unigram 0.448 | 1.247 | 0.993 | 1.354 |
| noisy period (p=2–4) | 0.669 | 0.900 | phase-count 0.892 | 0.791 | 0.390 | 1.386 |
| AR(1) binned (score, 7 bins) | 0.313 | 0.355 | persistence 0.352 | 1.888 | 1.517 | 1.946 |
| integer trends (5 cands) | 0.906 | 1.000 | — | 0.232 | 0.020 | 1.609 |

1. **Argmax is often reasonable, probabilities are not.** On coin flips JEV's argmax
   is near-Bayes (0.706 vs 0.726), but its P(heads) is compressed toward 0.5: a
   linear fit against the Laplace rule has slope 0.21 (when the Laplace estimate is
   ≥0.8, JEV says 0.54 on average). Its log-loss (0.645) is close to the uniform
   guess (0.693). → JEV is strongly *under*-confident as a frequency estimator.
   **JEV 的 argmax 尚可，但概率被严重压向 0.5（欠自信）。**
2. **JEV behaves like a frequency counter, not a transition model.** On Markov chains
   its distribution is closer to the unigram frequency (TV 0.155) than to the Bayes
   transition predictive (TV 0.258); accuracy ≈ unigram (0.474 vs 0.448), well
   below Bayes (0.586). **JEV 更像是统计符号频率，而没有学到转移结构。**
3. **Periodic structure breaks under noise and with word tokens.** Clean period-3/4
   patterns are mostly solved, but with 10–20% noise JEV drops to 0.40–0.77 vs
   0.80–0.90 for a trivial phase counter; on clean `apple river apple river …` it
   picks the right phase only 35% of the time (letters: 100%).
4. **Numeric series:** JEV is worse than the naive persistence forecast on AR(1)
   (exp_acc 0.313 vs 0.352, NLL 1.89 vs 1.58 — barely better than uniform 1.95).
5. **Rule-based integer sequences** (arithmetic, geometric, Fibonacci) are solved
   perfectly; quadratic sequences only 62%.

Interpretation (preliminary): zero-shot JEV is a fast *pattern recognizer* but a
poor *in-context statistical learner*. Whether this can be fixed by changing what
we put in the state (pre-computed sufficient statistics), by post-hoc calibration,
or by fusing it with a trained sequence model is the subject of Phase 1b/2.
