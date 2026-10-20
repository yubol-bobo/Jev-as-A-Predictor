# Research loop: System One deciders in the decision seat

A running log of cycles: phenomenon → closer look → hypothesis → test → keep, refine, or drop. Aggregates only (the
data policy in README.md). Claims count as established only after a test that was fixed before its data existed
(pre-registered below) or a held-out replication. Paper text waits until the experiments are done (user, 2026-10-20).

## Cycle 1 (2026-10-20 13:40): the offline–online gap

**Phenomenon.** On the fixed exam (pb_combine.py, 4 deciders, 1,461 questions, 100 tasks) every decider makes far
fewer key errors than the LLM's own first sample: 0.17–0.27 vs 0.47; omission of an available required order 2–18%
vs 55%. Online, Clef in the decision seat does not beat the LLM on task pass.

**Closer look** (`scripts/pb_gap.py`, `results/pb_gap.json`). LLM side pooled over llm/run1, llm/run2 (15 tasks),
llm_log_k5/run1 (2.15 runs per task); paired by task, task-bootstrap 95% CIs.

| | LLM | Clef (100 tasks) | pplx12 (76 tasks) |
|---|---|---|---|
| task pass | 0.41 | 0.37 (−0.04 [−0.12, 0.03]) | 0.47 (+0.06 [−0.06, 0.17]) |
| required-order checkpoints | 0.76 / 0.72 | +0.01 [−0.06, 0.07] | **+0.13 [0.04, 0.24]** |
| Documentation checkpoints | 0.70 | −0.04 [−0.11, 0.03] | −0.02 [−0.12, 0.08] |
| notes written | 1.5 | **−0.38** | **−0.49** |
| retrieval calls before the first action | 30 | **−5.5** | **−3.6** |

The grader barely penalizes over-ordering: 1 "must not order" checkpoint in 100 tasks. Orders that match no grader:
~3.2 per task in every arm.

## Cycle 2 (2026-10-20 13:50): where required orders are lost

**Question.** The exam's omission key is per step. Does the LLM omit, or only defer?

**Test** (`scripts/pb_ceiling.py`, `results/pb_ceiling.json`), on llm_log_k5/run1 (118 required orders, 54 tasks):

- 79% of the LLM's step-level omissions are placed later in the same trajectory. **The exam's omission key mostly
  measures timing, not omission**; that is why the offline gain does not carry over.
- Of all required orders: 69% placed, 11% offered by some candidate but never executed (what selection can recover),
  19% never offered by any candidate (only generation can fix). Upper bound from selection on this trajectory:
  4 more of 54 tasks pass.
- Online, pplx12 takes every offered required order (offered-not-taken 0%, −12 points [−18, −6]) and places more
  (+11 points [1, 21] vs llm_log). Clef does not (placed −6 [−15, 3]); its never-offered share rises +8 [−1, 18],
  plausibly because it acts after less retrieval.

**Reading.** A selector helps only through orders the LLM proposes and would not place later. pplx12 captures them
online; Clef does not. Deciders cost notes (fewer rewrites), with a small, uncertain documentation loss.

## Pre-registered tests (fixed 2026-10-20 14:00, before the data exist)

- **P1 (replication on unseen tasks).** On the 24 tasks clef_k5__pplx12/run1 has not run yet, pplx12 in the decision
  seat passes more required-order checkpoints than the LLM side (pooled LLM runs on the same tasks). Primary metric:
  required-order checkpoint pass, paired by task. Prediction: positive difference. Also report the 100-task pooled
  estimate with its CI.
- **P2 (method: decide only at commitment points).** gate(pplx12) online (pb_run.py --arm gate, margin fixed from the
  exam before the run) keeps the required-order gain of P1 (difference vs LLM > 0) and removes the note loss (notes
  written within the LLM's run-to-run noise). Secondary: documentation checkpoints not below the LLM.
- **P3 (offline key).** Redefine the exam's order key to count only consequential omissions (the required order is
  not placed later in the trajectory). Prediction: the decider ranking on this key matches the online ranking
  (pplx12 > Clef), unlike the per-step key.
