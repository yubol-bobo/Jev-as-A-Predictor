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

## Cycle 3 (2026-10-20 14:00): does a consequential key rank the deciders as online? (P3)

`scripts/pb_conseq.py`. Restricted to steps where the counterfactual is known (the executed turn takes no open
required order, an option places one the trajectory never places): **11 questions in 9 tasks**. Take rate: pplx12
and d3 0.91, Clef and linne 0.82, prefer_order 0.73, amos and decider31b 0.64 (CIs ±0.2–0.3). **P3 not supported**:
the exam has too few consequential order decisions to rank deciders, and on the per-step key Clef ≥ pplx12 although
online pplx12 ≫ Clef. Lesson: the exam measures decision style (action bias, finish bias, redundancy), not outcome;
only online runs rank deciders on outcome.

## Cycle 4 (2026-10-20 14:05): the note loss, and a narrower decision point

- Deciders prefer FINISH (picked 1.33× as often as the LLM's own sample does) and orders (1.5×), and retrieval less
  (0.89×) (`kind_preference`, pb_combine.py). Online they write fewer notes and stop earlier.
- Within a task, the run that writes more notes has better Documentation checkpoints: 47 vs 19 discordant pairs
  (all runs); 32 vs 19 (p = 0.09) among runs with at least one note. Suggestive only (observational).
- So the exam's redundancy key, which counts a note rewrite without new results as redundant, may penalise something
  useful; treat it with caution.
- **Method refinement: orders(D)**, D decides only where some candidate places a service or medication order;
  FINISH and notes stay with the LLM's sample. Offline (4 deciders): orders(pplx12) keeps the order key (omission
  3% vs LLM 55%), leaves finish and redundancy at the LLM's level, overrules the LLM on 11% of steps (pplx12 alone:
  73%; route(pplx12): 19%). Online: `pb_run.py --arm gate --margin 0 --points orders`.

## Pre-registered tests (fixed 2026-10-20 14:00, before the data exist)

- **P1 (replication on unseen tasks).** On the 24 tasks clef_k5__pplx12/run1 has not run yet, pplx12 in the decision
  seat passes more required-order checkpoints than the LLM side (pooled LLM runs on the same tasks). Primary metric:
  required-order checkpoint pass, paired by task. Prediction: positive difference. Also report the 100-task pooled
  estimate with its CI.
- **P2 (method: decide only at commitment points).** gate(pplx12) online (pb_run.py --arm gate, margin fixed from the
  exam before the run) keeps the required-order gain of P1 (difference vs LLM > 0) and removes the note loss (notes
  written within the LLM's run-to-run noise). Secondary: documentation checkpoints not below the LLM.
- **P2' (refinement of P2, fixed 14:10; replaces P2 for the next online run).** gate(pplx12, margin 0, points =
  orders) online, run 1, paired with the pooled LLM runs on the tasks it finishes. Primary: required-order checkpoint
  pass > LLM. Secondary: notes written and steps within the LLM's run-to-run noise (llm vs llm_log differences:
  notes −0.12 [−0.40, 0.15]); Documentation checkpoints not below the LLM; task pass reported with its CI, no claim
  unless the CI excludes 0. Started by `scripts/loop_online.sh` after the exam (2026-10-20, ~15:15–17:05).
- **P3 (offline key).** Redefine the exam's order key to count only consequential omissions (the required order is
  not placed later in the trajectory). Prediction: the decider ranking on this key matches the online ranking
  (pplx12 > Clef), unlike the per-step key.
