# Research plan — JEV as a sequential predictor / JEV 作为序列预测器

*Drafted 2026-10-09. Living document; changes are recorded in the README progress log.*

## 1. Question / 研究问题

JEV (`jev-1.13.0`, TypeSafe "System One") takes a `state` and typed questions
(Choice over ≤255 supplied candidates, ordinal Score over 2–10 levels, Boolean
`noul`) and returns a full probability distribution in one prefill-only call
(~0.2 s). It is marketed as a *decision* model. We ask whether it is also a
good **predictor of what comes next in a sequence**, and whether it can be
**adapted** into one.

JEV 是一个"决策"模型：给定 state 与带候选集的问题，单次前向直接输出概率分布。
我们研究：它能否作为**序列下一步预测器**（next-item / next-event / next-value），
以及能否通过适配让它成为好的预测器。

Sub-questions:

- **RQ1 — In-context sequence inference.** On sequences whose true predictive
  distribution is known exactly (Markov chains, noisy periodic patterns,
  Bernoulli with unknown bias, AR(1) processes), how close is JEV's
  distribution to the Bayes posterior predictive? (accuracy, NLL, KL to exact
  `q`, calibration). Does it improve with more context, like a Bayesian
  learner would? This is the PFN view of a "System One" predictor.
- **RQ2 — Real sequential benchmarks.** On standard sequential prediction
  tasks — sequential recommendation (next item, candidate re-ranking) and
  process mining (next activity) — how does zero-shot JEV compare with
  (a) trivial statistics (popularity, Markov / n-gram), (b) trained sequence
  models (SASRec / GRU4Rec, LSTM), (c) open LLM log-prob readouts?
- **RQ3 — Adaptation without weights.** JEV is closed, so adaptation means:
  state serialization (raw vs. summarized sufficient statistics), candidate
  construction (two-stage: cheap retriever → JEV re-rank), post-hoc calibration
  fitted on dev, and fusion with a trained sequence model (log-linear / stacking).
  Which of these turns JEV into a competitive predictor?
- **RQ4 — Adaptation with weights (open analogs).** If the System-One interface
  is valuable, does training an open System-One model (Kev / NanoJev-style
  decision head) on sequence data close the gap? (Phase 3, only if RQ1–3 are
  promising.)

## 2. Why this is non-trivial / 为什么值得做

- JEV returns probabilities rounded to 2 decimals; many options get exactly 0.00,
  so log-loss requires a floor. We report NLL with a fixed floor (ε = 0.005,
  i.e. half the rounding grid) and the fraction of gold items assigned 0.00.
- Prior audits (see `../jev/jev_open/COMPARISON.md`) found JEV's typed-decision
  KL/Brier **worse than an input-ignoring prior**. A prior / frequency baseline is
  therefore mandatory everywhere.
- A sequence predictor must aggregate evidence over many steps (counting);
  a decision model trained for "pick the right action" may instead pattern-match
  the last element. RQ1 isolates that.

## 3. Tasks / 任务

### Phase 1 — synthetic, exact ground truth (RQ1)

| Task | Generator | Question type | Exact `q` |
|---|---|---|---|
| `bern` | Bernoulli(p), p ~ Beta(1,1), n ∈ {5,10,20,40} | noul | Laplace rule (h+1)/(n+2); also true p |
| `markov` | order-1 chain, K ∈ {3,5}, rows ~ Dir(α), length ∈ {10,30,60} | choice (K) | Dirichlet-posterior predictive given counts; also true row |
| `period` | period-p pattern (p ∈ {2,3,4}) with symbol flip ε ∈ {0, .1, .2} | choice (K) | 1−ε on pattern symbol, ε/(K−1) else |
| `ar1` | x_{t+1}=φx_t+σe, φ ∈ {0.5, 0.9}, numeric series | score (7 bins) | Gaussian bin masses |
| `trend` | linear / quadratic integer sequences | choice (5 numeric candidates) | deterministic |

Symbols are rendered as tokens (`A B C`) and also as words (`sunny rainy …`) to
test representation sensitivity.

### Phase 2 — real benchmarks (RQ2)

- **MovieLens-1M** next-item, leave-one-out, 1 positive + 19 sampled negatives
  (standard LLM-rec protocol, e.g. LlamaRec / TALLRec); metrics HR@1, NDCG@5/10, MRR.
  Baselines: popularity, item-transition Markov, SASRec.
- **Process mining next activity** (Helpdesk, BPIC 2012 / Sepsis from 4TU).
  Candidates = full activity vocabulary. Baselines: majority, prefix n-gram, LSTM.
- (Optional) a language-heavy domain where world knowledge should help
  (e.g. Amazon product sequences with titles).

### Phase 3 — adaptation (RQ3, RQ4)

state formatting ablations · count-summary states · dev-fitted temperature /
Platt · log-linear fusion JEV ⊗ SASRec · retrieve-then-rerank · (optional) LoRA
decision head on an open System-One backbone.

## 4. Systems / 系统

- `jev-1.13.0` via `api.typesafe.ai/v1/systemone` (client reused from
  `../jev/understanding_capability_of_jev/src/s1eval/systems.py`, with a local
  content-addressed cache so no request is ever paid twice).
- Statistical baselines (exact Bayes, counts, n-gram, popularity).
- Trained baselines (SASRec, LSTM) — CPU/GPU via SLURM.
- Open LLM log-prob readout (Qwen3.5-9B) and open System-One models (Kev-9B, CLM)
  via existing servers — Phase 2+.

## 5. Metrics / 指标

Accuracy (top-1), NLL (ε-floored), Brier, ECE (15 bins, top-label), KL(q‖p) for
synthetic tasks, ranking metrics (HR@k, NDCG@k, MRR) for recommendation, RPS for
ordinal Score questions. Uncertainty: bootstrap 95% CIs over sequences.

## 6. Hygiene

- Every JEV request/response stored verbatim (`results/raw/…jsonl`) with sha256.
- Generators are seeded; item files are frozen before any model call.
- Dev/test split for anything fitted (calibration, fusion weights).
- No API keys in git; keys are read from `../jev/.env` at call time.
