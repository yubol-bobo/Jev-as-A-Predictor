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

## Cycle 5 (2026-10-20 14:30): boundaries of the decider (`scripts/pb_boundary.py`)

Benefit = key errors of the LLM's sample − key errors of the decider's pick, on keyed exam questions (Clef, pplx12,
d3, amos; descriptive, no CIs).

- **Confidence is informative.** Overrides with log-odds ≥ 3 over the LLM's sample: +0.8 to +1.0 on the order key,
  +0.6 on redundancy; below 1.5: about 0. A high margin keeps the benefit with far fewer overrides.
- **The decider helps where the reasoning model is unsure.** Semantic agreement of the LLM's 5 samples with its
  first: 1–2 of 5 → benefit +0.37 to +0.40; 4 of 5 → +0.11 to +0.17; 5 of 5 → 0. Order decisions sit almost all at
  1 of 5.
- **Finishing is the decider's blind spot.** Confident overrides toward FINISH while a required order is open:
  −0.75 to −0.80 (n = 4–5); deciders finish prematurely 90% vs the LLM's 80%.
- **Correction of an earlier reading.** The deciders' low redundancy (3–8% vs 28%) is not state use: in 17–18 of
  19–22 rescues the decider picks FINISH where the LLM's sample rewrites the note without new results. Their exam
  advantages reduce to type preferences (orders 1.5×, FINISH 1.33×, retrieval 0.89× vs the LLM's sample).
- **cgate(D, a, d)** (S1 + LLM: D only if ≤ a of 5 samples agree, only with margin d, never toward FINISH) scores
  worse on the exam (CV 0.23 vs 0.17) because the redundancy key rewards finishing over a note rewrite; the keys
  conflict, so the exam cannot choose this rule; online can.

## Cycle 6 (2026-10-20 14:40): synthesis — a commitment bias

The same pattern shows in all three ways a decider sits in the loop:

| framing | what the decider does (vs the LLM's own choice) | outcome |
|---|---|---|
| exam (picks among the LLM's 5 turns) | orders 1.5×, FINISH 1.33×, retrieval 0.89× | order key up; finish key down |
| S1-Select online (clef_k5, pplx12) | retrieval before the first action 25–27 vs 30; fewer notes; shorter runs | required orders up (pplx12), notes down |
| S1-Tool online (decider picks the tool) | write_file on 44% of steps, 9.2 notes per task vs 1.5; retrieval before the first action 9.9 vs 30; 15 loop aborts | pass 0.07 vs 0.41 |

**Hypothesis H-commit:** System One deciders carry a prior toward committing actions (order, write, finish) over
information gathering, largely independent of the state. It helps where the reasoning LLM under-commits (orders it
defers) and hurts where commitment is premature (finishing with orders open, note loops when it picks tools, less
retrieval). Test of state-independence: P4 (blind exam). Test of the fix: P2' (decider only at order points).

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
- **P4 (does the decider use the patient state? fixed 14:35).** pplx12 answers the exam blind (options only, the
  same option letters; `pb_exam.py select --blind`). Style hypothesis: blind and state picks agree on ≥ 60% of
  questions (about the agreement between different deciders with state), and the order and finish key errors differ
  by ≤ 0.05. State hypothesis: agreement < 50% and blind errors clearly higher. Reported on all keys, plus agreement
  on unkeyed questions.
- **P3 (offline key).** Redefine the exam's order key to count only consequential omissions (the required order is
  not placed later in the trajectory). Prediction: the decider ranking on this key matches the online ranking
  (pplx12 > Clef), unlike the per-step key.

## Status at the end of allocation 49512149 (2026-10-20 17:20)

- **P1, P2', P4 not run.** `scripts/loop_online.sh` started both Qwen servers and pplx12 at 16:08, but neither Qwen
  server wrote a log line or took GPU memory in 70 minutes, and the pplx12 server finished loading its weights (10 min)
  and then did not open its port. The first attempt of the P4 blind exam therefore failed on all 1,466 questions
  (connection refused; answers_pplx12_blind.jsonl holds only failed answers, which `select` retries). Cause not
  diagnosed; slow model loading from shared storage is the likely suspect. No partial data exists for any of the three.
- The paper draft states that the tests are pre-registered and not yet run, and makes no claim about the gate online
  or about whether the commitment bias reads the state.
- ClinEnv is dropped from the study (user, 2026-10-20 16:05); its one line of evidence here was removed.
- Next allocation: P4 first (pplx12 only, short prompts, minutes), then P1 and P2' (~30 min per task, about two runs of
  the LLM pool per allocation), then second runs of pplx12 and the gate and LLM run 2.

## P4 result (2026-10-20 20:00, allocation 49526072): the decider reads the state, and the state is what makes it stop

`scripts/pb_blind.py --decider pplx12` -> `results/pb_blind.json`; 1,461 questions, pplx12 with the state vs blind
(options only, same letters). Differences blind − state, paired, task-bootstrap 95% CIs.

| | with state | blind | blind − state | random option |
|---|---|---|---|---|
| agreement with the state picks | — | 0.41 (keyed 0.45; random 0.22) | |
| same kind of action as the state pick | — | 0.91 (random 0.875; other deciders with state 0.95–0.98) | |
| order key error (97 q, 48 tasks) | 0.07 | 0.22 | **+0.14 [0.08, 0.21]** | 0.55 |
| redundancy key error (86 q, 61 tasks) | 0.03 | 0.24 | **+0.21 [0.11, 0.31]** | 0.35 |
| finish key error (30 q, 25 tasks) | 0.90 | 0.53 | **−0.37 [−0.53, −0.19]** | 0.73 |
| type preference act / finish / document / retrieve | 1.49 / 1.34 / 1.09 / 0.90 | 1.38 / 0.96 / 0.80 / 0.98 | |

**Verdict against the pre-registration.** Style hypothesis (agreement ≥ 60%, order and finish errors within 0.05):
**rejected**. State hypothesis (agreement < 50%, blind errors clearly higher): **supported on the order and redundancy
keys, reversed on the finish key**.

**Reading (refines H-commit).**
- pplx12 uses the patient state: blind, its picks match the kind of the state pick barely above chance (0.91 vs
  0.875), and it errs far more on redundancy (random 0.35 → blind 0.24 → state 0.03: mostly state).
- The order gain is mostly a prior. Random 0.55 → blind 0.22 → state 0.07: about two thirds of the reduction needs no
  patient state (options alone: an order-shaped turn is preferred, 1.38× blind vs 1.49× with state).
- The FINISH error is driven by the state. 73% of the options on these questions are FINISH (the LLM's own samples
  mostly stop there too). Blind, pplx12 avoids FINISH (0.53, below random); with the state it finishes with a
  required order open on 0.90, above random and above the LLM's first sample (0.80). **Reading the state is what
  makes it stop.** The truncation explanation cannot be tested here: 27 of the 30 finish questions are read with
  shortened tool results (finish error 0.93 shortened vs 0.67 full, n = 3).
- In the LLM's terms: these are late steps where the LLM itself mostly wants to stop. The decider with the state
  agrees with the LLM's (wrong) sense that the case is done, and more strongly.
- So the commitment bias has two parts: a state-free prior toward orders, and a state-driven, mistaken judgement that
  the work is done. The orders-only gate (P2') keeps the first and removes the second by design.

## Incident (2026-10-20 21:00): the first P2' tasks ran the wrong gate

`pb_run.py` did not pass `--margin`/`--points` to its per-task child, which re-parsed its own defaults (margin 1.5,
points commit) and wrote to `gate1.5_k5__pplx12/run1`. The 22 tasks finished 20:14–20:59 are therefore a gate with
margin 1.5 at every commitment point (the original P2 design), not P2'. Fixed (the child gets both flags); P2' restarted
at 21:04 into `gate0_orders_k5__pplx12/run1`, children verified to run `--margin 0.0 --points orders`. The gate1.5 runs
are kept as a separate, exploratory arm and are not counted for P2'. No earlier data was touched (the gate1.5 directory
was new). P1 (clef arm) was not affected.

## P1 result (2026-10-20 22:17): the pplx12 order gain does not replicate

`pb_gap.py --decider-runs clef_k5__pplx12/run1 --tasks @results/p1_unseen_tasks.txt` -> `results/pb_gap_p1.json`;
all 100 tasks -> `results/pb_gap_pplx12_all.json`. Paired with the pooled LLM runs, task-bootstrap 95% CIs.

| | 24 held-out tasks (P1) | all 100 tasks |
|---|---|---|
| required-order checkpoints | LLM 0.84, **−0.18 [−0.36, −0.02]** (16 tasks) | +0.04 [−0.05, 0.14] (54 tasks) |
| required orders placed | −0.15 [−0.29, −0.02] | +0.05 [−0.05, 0.15] |
| task pass | −0.02 [−0.17, 0.13] | +0.04 [−0.06, 0.14] |
| Documentation checkpoints | +0.19 [0.06, 0.33] | +0.03 [−0.05, 0.12] |
| notes written | −0.58 [−0.88, −0.33] | −0.52 [−0.68, −0.37] |
| steps | −3.0 [−6.1, 0.5] | −4.0 [−6.7, −1.5] |

**Verdict: P1 fails** (prediction: positive difference; observed significantly negative). The +13 points on the first
76 tasks were not a property of the decider. Ceiling analysis rerun on all 54 tasks with required orders
(`results/pb_ceiling.json`): pplx12 halves offered-not-taken (4% vs 11%, −7 [−13, 0]) but more required orders are never
offered (+4 [−5, 13]), so placed +3 [−8, 12]. Selection recovers part of the 11% and loses about as much on generation
(it acts after less retrieval). This fits the ceiling reading: the decision seat is not where the LLM's order errors
are. Paper corrected (abstract, findings, Table pooled, §4.1, §4.2, §4.4, §4.7).

## Allocation 49531604 (2026-10-20 23:45)

Job 49526072 had an 8 h limit but ended in TIMEOUT after 3 h 03 min (22:31; cause unknown), with P2' at 30/100 tasks.
Resumed on w001 with the node-local model copy (servers up in 9 min) and two P2' drivers (14 + 8 workers); the
P2' score (`results/pb_gap_p2prime.json`) is committed every 45 min in case this job also ends early.

## P2' result (2026-10-21 02:55): the gate keeps the agent's length and notes, not an order gain; it over-orders

`pb_gap.py --decider-runs gate0_orders_k5__pplx12/run1` -> `results/pb_gap_p2prime.json`, all 100 tasks, paired with
the pooled LLM runs (llm/run1, llm/run2, llm_log_k5/run1), task-bootstrap 95% CIs. LLM-vs-LLM noise in brackets.

| | gate (P2') − LLM | full steering (pplx12) − LLM | LLM noise |
|---|---|---|---|
| required-order checkpoints (54 tasks) | +0.05 [−0.02, 0.14] | +0.04 [−0.05, 0.14] | |
| required orders placed | +0.03 [−0.05, 0.11] | +0.05 [−0.05, 0.15] | +0.05 [−0.01, 0.11] |
| orders placed | **+1.28 [0.76, 1.89]** | +0.26 [−0.35, 0.93] | +0.18 [−0.34, 0.71] |
| orders not in the rubric (55 tasks) | **+1.13 [0.52, 1.79]** | +0.21 [−0.70, 1.11] | +0.35 [−0.31, 1.02] |
| notes written | −0.02 [−0.26, 0.24] | −0.52 [−0.68, −0.37] | −0.12 [−0.40, 0.15] |
| steps | −0.5 [−4.4, 3.9] | −4.0 [−6.7, −1.5] | |
| Documentation checkpoints | −0.03 [−0.10, 0.04] | +0.03 [−0.05, 0.12] | |
| task pass | −0.04 [−0.13, 0.04] | +0.04 [−0.06, 0.14] | |

**Verdict: P2' primary fails** (the CI includes 0). Secondary: notes and steps within the LLM's noise (the note loss and
the shortening are gone, as designed); Documentation not significantly below (point estimate −0.03); no task-pass claim.
**Unplanned observation (post hoc):** the gate places 1.3 more orders per task, almost all of them orders the rubric
does not name. Reading (H-commit): the state-free order prior acts at every order point, and the gated agent, unlike
full steering, does not stop early, so the prior has more steps to act on. Not claimed until P5.

- **P5 (replication of the over-ordering, fixed 2026-10-21 02:58, before the data exist).** gate(pplx12, margin 0,
  points = orders), run 2, same tasks, paired with the pooled LLM runs. Prediction: orders placed higher than the LLM
  with the CI excluding 0 (point estimate ≥ +0.5), and orders not in the rubric higher with the CI excluding 0.
  Required-order checkpoints reported, no new prediction. Pooled gate runs 1+2 reported as the paper's gate estimate.
  Started on allocation 49531604 at ~03:00; tasks that do not finish before the job ends are resumed later.

Exploratory, not a test: the gate1.5 arm (margin 1.5 at every order or FINISH point, the 22 tasks run by mistake on
2026-10-20) -> `results/pb_gap_gate1.5_exploratory.json`: notes −0.62 [−1.02, −0.29], steps −6.7 [−12.9, −1.9],
orders −0.13 [−0.71, 0.51], required-order checkpoints +0.08 [0.00, 0.19] (14 tasks). Gating at FINISH brings the note
loss and the shortening back; consistent with P4 (the harm sits at the finish step). Paper updated (abstract, intro,
findings, §4.7 with Table tab:gate, Discussion, Limitations).

## P5 result (2026-10-21 06:40): more orders replicates; "outside the rubric" does not reach significance alone

Gate run 2 finished all 100 tasks (03:00–06:37) -> `results/pb_gap_p5.json`; runs 1+2 pooled ->
`results/pb_gap_gate_pooled.json` (the paper's gate estimate, Table tab:gate).

| | run 2 (P5) − LLM | runs 1+2 pooled − LLM |
|---|---|---|
| orders placed | **+0.61 [0.12, 1.13]** | **+0.95 [0.51, 1.44]** |
| orders not in the rubric (55 tasks) | +0.35 [−0.29, 0.97] | **+0.74 [0.20, 1.31]** |
| required-order checkpoints | +0.06 [−0.02, 0.16] | +0.06 [−0.01, 0.14] |
| notes | −0.13 [−0.33, 0.09] | −0.07 [−0.26, 0.12] |
| steps | −0.6 [−4.5, 3.7] | −0.5 [−3.9, 3.2] |
| task pass | +0.02 [−0.07, 0.11] | −0.01 [−0.09, 0.06] |

**Verdict: P5 half holds.** More orders replicates (≥ +0.5, CI excludes 0); more orders outside the rubric does not
reach significance in run 2 alone (it does pooled). Smaller effect in run 2 (+0.61 vs +1.28): run-to-run variation of
the gate itself is large. Required-order checkpoints still not above the LLM; notes and steps at the LLM's level in
both runs. Paper updated (abstract, findings, §4.7, Table tab:gate pooled, Limitations).

## Cycle 7 (2026-10-21 07:00, offline): can a pre-finish "pending order audit" recover the selection ceiling?

Idea: before FINISH, place every order some candidate proposed but no executed turn placed (no System One needed).
On llm_log_k5/run1 (100 tasks, rough string dedup): ~9.8 distinct proposed-but-unexecuted orders per task; about 33 of
~980 match a required order that was not placed (the 13 required orders of pb_ceiling's offered-not-taken), about 3%.
Requiring support from ≥ 2 candidates does not raise it (2 of 62). Upper bound on task pass: +4 of 54 tasks
(pb_ceiling). Not pursued: it would add ~9 orders per task to recover ~0.1. Selection-side fixes are exhausted on this
benchmark; the remaining gap is generation (19% of required orders never proposed).

## Cycle 8 (2026-10-21 07:15, offline): System One as a fast screener over a wider pool (idea: Yubo)

Idea: a System One model is fast, not only cheap. Instead of picking 1 of 5 turns, let it screen many more candidates
(each with a short input, or all at once) so more information reaches a model that can judge it. The ceiling says
the gap is generation (required orders never proposed); the question is whether breadth fixes generation.

`scripts/pb_pool.py`: pool the orders proposed by any candidate in five logged runs of the same task
(llm_log_k5, clef_k5, clef_k5__pplx12, gate run 1, gate run 2; 100 tasks, 590 required order × run pairs).

- Proposed within one run: 75% of required orders (placed 68%). Proposed somewhere in the 5-run pool: **91%**.
- Of the required orders a run never proposes, **62% (91/146) are proposed in another run** of the same task.
  "Never proposed" is mostly sampling breadth, not inability: the LLM knows these orders, in some runs.
- The pool is not small: median 26.5 distinct order strings per task (mean 36.6; includes spelling variants, so the
  string-level base rate of required, 0.30, is rough). Frequency across runs does not separate them (string-level
  dedup too fine to judge; precision 0.31 at ≥ 2 runs, 0.60 at all 5 but only 15 orders).
- Speed: the decider takes 4.3 s (Clef) / 12.3 s (pplx12) median per step on ~32k-token inputs (prefill); the five LLM
  proposals take ~6k generated tokens per step. A short input (a few k tokens) and a shared prefix should put one
  screening judgement well under a second, so screening ~40 orders costs less than one LLM step.

So the idea has room: breadth would recover up to ~16 points of required orders (75% -> 91% proposed), more than the
selection ceiling (11%). It works only if the screener separates required from unneeded orders given the state;
the blind test (P4) warns that the order preference is largely state-free, which is exactly what screening cannot use.

- **P6 (offline screening test, fixed 2026-10-21 07:20, before the data exist).** For the 54 tasks with required
  orders, the state is the llm_log_k5/run1 state just before its first FINISH (the point where a missed order is
  lost), shortened to the instruction plus retrieved results (≤ 8k tokens). Options: the task's 5-run order pool,
  variants merged by rubric spec where they match one (others by normalized name). Each order is scored pointwise
  ("Should this order be placed for this patient now? Yes / No", P(Yes)) by pplx12 and Clef, with the state and blind
  (order only). Comparators: the LLM (Qwen, thinking) asked the same question, and frequency across runs. Metrics,
  per task then averaged: AUROC required vs not; recall@5 of required orders not placed by the run.
  Predictions: (a) with the state, a System One screener reaches AUROC ≥ 0.70; (b) state − blind ≥ 0.10 AUROC (the
  screener reads the state; if not, the idea fails for the reason P4 found); (c) within 0.05 AUROC of the LLM at
  < 1/10 of its wall time. If (a) and (b) hold, the online test is: before FINISH, the screener's top orders not yet
  placed are shown to the LLM as a checklist, the LLM keeps the decision (so the commitment bias cannot over-order).

## Allocation 49541996 (2026-10-21 07:19–15:19, w001): P6 running; P7 fixed

P6 built (`scripts/pb_screen.py build`): 54 tasks, 2102 pool orders after merging variants by rubric spec, 104
required (5%), median state 16.7k characters. Scoring started 07:31 (pplx12, Clef with the state; Qwen thinking).

- **P7 (online screening checklist, fixed 2026-10-21 07:40, before any P6 result was read; run only if P6 (a) and
  (b) hold).** Arm `screen` (`pb_run.py --arm screen`, `ScreenClient`): the LLM acts as in llm_log (K = 5 logged, the
  first executed); at its first FINISH, the orders proposed by any candidate so far plus those in B = 8 extra
  tool-forced samples are pooled, placed orders dropped, the rest ranked, and the top M = 3 shown to the LLM as a
  checklist ("place any that are indicated, and only those"); the LLM's answer replaces the FINISH. Ranker: the
  System One model with the higher state AUROC in P6. Control arm: the same checklist ranked by proposal frequency
  (`--ranker freq`), so the nudge itself is controlled. Tasks: the 54 with required orders (`results/p7_tasks.txt`),
  run 1, paired with the pooled LLM runs and with the control.
  Primary: required orders placed, screen − freq > 0 and screen − LLM > 0 (CIs reported; claim only where the CI
  excludes 0). Secondary: required-order checkpoints; orders placed (over-ordering: screen − LLM reported, a cost);
  task pass reported, no claim unless its CI excludes 0.

## P6 interim (2026-10-21 08:15): the screen ranks what was done, not what was missed

State scores complete for pplx12 and Clef, Qwen at 866/2102. Overall AUROC (required vs not, per task): pplx12 0.74
[0.66, 0.81], Clef 0.71 [0.63, 0.79], Qwen (thinking, partial) 0.68. Split by whether the run placed the order:
placed orders 0.67 / 0.66 / 0.65; **orders the run did not place 0.43 / 0.40 / 0.41 (21, 21, 7 tasks)**. The
overall AUROC comes from placed orders (19% required) scoring above unplaced ones (1.7% required): the models read
what the agent did. On the orders a screen could add, every model, the LLM included, ranks the missed required orders
below the unneeded ones. recall@5 of missed required orders: pplx12 0.14, Clef 0.11 (random top-5 of ~37: ~0.14).
Clef blind on unplaced: 0.55 (partial). The pre-registered frequency comparator is invalid as built: required
variants are merged by rubric spec and unneeded ones are not, so frequency inflates required orders by construction
(AUROC 0.97); it is dropped.
Reading: the missed orders do not look indicated from the state the agent built. Either the information that
makes them indicated was never retrieved, or the rubric asks for what the chart does not imply.

- **P6b (more information, fixed 2026-10-21 08:20, before the data exist).** The same questions, with a rich state:
  the run's own state plus the retrieval steps the other four pool runs took before their first order (no other
  run's orders leak in), identical results kept once, up to ~26k tokens (`pb_screen.py build_rich`, median 15 extra
  retrieval steps per task). Scored by pplx12 (`score --rich`). Prediction: AUROC on unplaced orders ≥ 0.60 and
  ≥ 0.10 above the own state (0.43). If it holds, missed orders are an information gap that a fast, wide reader can
  close (design: the System One model reads a chart dump, not the agent's trajectory); if not, the gap is not in
  what the agent read, and screening is dropped. P7 is on hold until P6b is read.

## P6b verdict (2026-10-21 09:22): fails; more of the chart does not help, less does

Unplaced-order AUROC with the rich state (other runs' retrieval added, ~18k tokens): pplx12 0.436 [0.29, 0.59]
(own state 0.430), Clef 0.451 [0.30, 0.60] (own 0.396). Neither reaches 0.60 nor +0.10. The wide-reader design is
dropped. Exploratory, seen before P6c was fixed: with **no information at all** (order text only, `--blind`),
pplx12 ranks the missed required orders above unneeded unplaced ones, 0.628 [0.50, 0.76], 21 tasks; within
ServiceRequests alone 0.61 (13 tasks), so not only a medication-vs-service split; recall@5 0.37 (random ~0.14).
Clef blind 0.51. Ordering on unplaced orders: no information > own state ≈ rich state, for both models. Reading:
the state pulls the System One model toward what the agent did (it learned to imitate agents), and what the agent
did not do looks not indicated; its order prior, free of the trajectory, carries some of what the rubric wants.
Full aggregates: `results/pb_screen.json` (with per-task unplaced AUROC and bootstrap CIs over tasks).

- **P6c (short input, fixed 2026-10-21 09:24, before the data exist).** Same questions, the task instruction only,
  no trajectory (`score --task`, a few hundred tokens: what a fast model does best). Scored by pplx12 and Clef.
  Prediction: pplx12 task-only unplaced AUROC ≥ 0.60 with the CI's lower end above 0.5. Decision for P7′: the
  ranker input is the task instruction if task-only ≥ blind (0.628), else the order alone if task-only fails and
  blind stays as is; if task-only's CI includes 0.5 and is below blind, P7′ uses blind and is reported as testing
  an order prior, not a reading of the task.
- **P7′ (replaces P7, whose P6 condition failed; fixed 09:24).** Arm and analysis as P7 (K = 5, B = 8, M = 3,
  54 tasks, run 1, paired with the pooled LLM runs and the `freq` control), the ranker being pplx12 on the short
  input chosen by P6c (`--ranker pplx12-task` or `pplx12-blind`). The freq control does not depend on P6c and starts
  first. Primary unchanged: required orders placed, screen − freq > 0 and screen − LLM > 0, claims only where the
  CI excludes 0. Note: online, the pool is one run's proposals plus 8 forced samples, not the 5-run pool of P6.

## P6c verdict (2026-10-21 09:28): holds; the short input is the System One model's best view

Task instruction only (a few hundred tokens, 0.39 s per question): pplx12 unplaced AUROC **0.652 [0.508, 0.786]**
(21 tasks; own state 0.430, rich 0.436, blind 0.628), recall@5 of missed required orders 0.32 (random ~0.14);
within ServiceRequests 0.73 (13 tasks; blind 0.61), so the task text adds information beyond the order prior.
Clef task-only 0.625 [0.503, 0.742]. Over all orders, task-only is also pplx12's best input: AUROC 0.80
[0.73, 0.87] vs 0.74 with the state and 0.57 blind. The trajectory hurts: it shows the model what the agent did,
and the model follows it. Decision as pre-registered: P7′ runs with `--ranker pplx12-task`. Control added after
the fact (exploratory, not part of P6c): Qwen thinking on the same task-only input, unplaced subset only
(`pb_screen.py llm --task`), to separate "a System One model" from "a short input".

P7′ started 09:24 (freq control, 8011) and 09:29 (pplx12-task; 8011 and a Qwen replica on GPU 3, port 8013, that
replaced the Clef server), 12 workers each.

**P7′ throughput (10:40).** One Qwen TP2 server plus one single-GPU replica carry 30 agents (K = 5 each): about
5 steps per minute in total, so the 108 task-runs need ~6.5 h and will not finish in this allocation. Decision
(made before any P7′ task finished): drivers run until ~14:45; the analysis is on the tasks finished in both arms
(both drivers walk the same sorted task list, so the two arms cover nearly the same tasks), reported as interim;
the rest resume in the next allocation (drivers skip finished tasks). Extra drivers (3 workers per arm) were pinned
to the replica at 10:12. The Qwen task-only comparator is paused (209/919 answered) to give the agents the replica.
Also from P6 data (exploratory): on orders the run *did* place, task-only beats the state as well, pplx12 0.82 vs
0.67 and Clef 0.76 vs 0.66 (42 tasks), so "the trajectory makes the System One model worse" holds on both subsets
and both models. Mechanism script for P7′: `scripts/pb_p7.py`.

## P7′ mechanism, interim (2026-10-21 12:00, 14 tasks per arm): the in-run pool cannot hold what is missed

`scripts/pb_p7.py` -> `results/pb_p7_mechanism.json`. At the first FINISH, of ~2.1 required-order specs per task
only 0.29 (freq) / 0.43 (pplx12-task) are still open; the checklist flags an open one in 1 / 2 of 14 tasks and
closes none so far. Of the 2.7 flagged orders, 1.1 / 1.7 match a required spec, but almost all of those are
variants of orders already placed under another name or code. The ceiling of an in-run screen is tiny: the
required orders a run misses are mostly not among its own proposals, whatever the ranker. P7′ continues as
planned; its outcome is expected near zero for both arms.

- **P8 (task-prior pool, offline, fixed 2026-10-21 12:05, before any generation).** A pool from outside the run,
  built from short inputs only: Qwen (no thinking) reads the task instruction alone and lists candidate orders,
  8 samples, union by normalized name (`scripts/pb_prior.py gen`); items matching the base run's orders placed
  before its first FINISH (by normalized name, as an online screen would) are dropped; pplx12 ranks the rest with the
  task-only question of P6c (`score`). Target: per task, the required-order specs not met by the base run
  (`llm_log_k5/run1`) before its first FINISH. Over tasks with ≥ 1 open spec: (a) pool coverage of open specs ≥ 0.40;
  (b) pplx12 top-5 covers ≥ 0.25 of open specs (mean per task) and more than a random 5 of the same pool
  (exact expectation). Both hold -> online arm P9 (task-prior checklist at the first FINISH); (a) fails -> the missed
  orders cannot be recalled from the task text and the order-screening line stops; (a) holds, (b) fails -> the
  ranker is the bottleneck (try the LLM or a listwise ranker).

## P8 verdict (2026-10-21 13:00): holds; the missed orders are known from the task text alone

`scripts/pb_prior.py` -> `results/pb_prior.json`. 26 tasks with 36 required-order specs open at the base run's first
FINISH. Qwen without thinking, task text only, 8 samples (177 s per task at 4 workers under load), pool 67 orders per
task after dedup: **coverage of the open specs 0.84 [0.71, 0.95]** (pooled 29/36). pplx12 task-only ranking: top-5
covers 0.35 [0.15, 0.54] vs a random 5 of the same pool 0.17, difference +0.17 [0.02, 0.33]; top-3 0.31 vs 0.11.
Both criteria hold. Reading: the LLM, asked from the task alone, names most of the required orders it then fails to
place while working in the chart: a commitment failure, not missing knowledge. Name matching against the rubric
regexes is lenient for generic names; the online test is the real check.
P7′ drivers stopped at 12:58 (tasks in flight finish): its mechanism shows a near-zero ceiling, and the remaining
allocation goes to P9. P7′ is reported as interim on the tasks finished in both arms.

- **P9 (task-prior checklist, online, fixed 2026-10-21 13:05, before any P9 run).** Arm `screen` with
  `--ranker prior-pplx12 --b 0 --m 3` (`ScreenClient._prior`): at the first FINISH, the P8 pool of the task
  (`prior_ranked_pplx12.json`, built before the run from the task text) minus orders the run has placed (by
  normalized name), top 3 by pplx12, shown with the P7 checklist text; the LLM decides. Control: `--ranker
  prior-random` (3 random orders of the same pool, same text), so the nudge is controlled and only the System One
  ranking differs. Tasks: the 26 of P8 (`results/p9_tasks.txt`), run 1, both arms concurrently. Primary: required-
  order checkpoint pass, prior-pplx12 − prior-random > 0 (task-bootstrap CI; claim only if it excludes 0).
  Secondary: vs the LLM runs llm/run1 and llm/run2 (not llm_log_k5/run1, which selected the tasks: regression to the
  mean would favour the arm); orders placed and orders outside the rubric (cost); task pass reported. Mechanism:
  open specs closed by a flagged order (`pb_p7.py`). Runs unfinished at ~15:05 resume next allocation; any earlier
  read is labelled interim.

## P7′ interim (2026-10-21 14:54, 30 tasks per arm, 27 paired with the control) and P9 status

`pb_gap.py` -> `results/pb_gap_p7_vs_freq_interim.json`, `pb_gap_p7_vs_llm_interim.json`,
`pb_gap_p7freq_vs_llm_interim.json`; mechanism `results/pb_p7_mechanism.json`.
- **Primary, screen (pplx12-task) − freq control: required-order checkpoints −0.02 [−0.15, 0.11]**, task pass −0.11
  [−0.26, 0.04]: no ranker effect. Mechanism: the checklist closed an open required spec in 3 of 29 tasks with
  pplx12-task (0.17 per task) and in 0 of 30 with freq.
- Both arms vs the pooled historical LLM runs: required-order checkpoints **+0.15 [0.04, 0.27] (pplx12-task) and
  +0.14 [0.03, 0.27] (freq)**, orders +1.0 / +1.8, notes +0.6 / +0.9. Since the checklist closes almost nothing,
  this gain arises before the first FINISH, where both arms run the same code path as llm_log. It is therefore not
  a screening effect: either the arm's client calls the LLM differently from DeciderClient (to check), or today's
  serving setup (MTP speculative decoding, load) differs from that of the historical runs. Until resolved, no
  comparison against historical LLM runs is trusted; only concurrent controls count (P7′ was built with one, P9 too).
- P9: 5 (prior-pplx12) and 4 (prior-random) of 26 tasks finished; too few to read. Resume next allocation:
  `pb_run --arm screen --ranker prior-pplx12|prior-random --b 0 --m 3 --tasks @results/p9_tasks.txt --run 1`
  (unfinished task directories restart clean). Also pending: Qwen task-only comparator
  (`pb_screen.py llm --task`, 209/919), and a concurrent plain-LLM run (llm_log) on the P9 tasks to measure the drift.
- Checked 14:58: ScreenClient and DeciderClient call the LLM identically (same temperature, reasoning effort, n = K),
  so the drift is not the client. Consequence beyond P7′: the gate results (P2′, P5; paper Table tab:gate) are also
  measured against the historical LLM runs. **First job of the next allocation: a concurrent plain LLM run
  (`--arm llm_log --run 3`) on the P7/P9 tasks under the current serving setup**, to size the drift and re-baseline
  the gate before the paper relies on it further.

## Allocation 49563852 (2026-10-21 16:32–00:26, w005): drift baseline D1, P9 resumed

Checked before any run: the Qwen serving script is unchanged since 10-19 and no agent or PhysicianBench code changed
after the historical LLM runs (llm/run1 10-18/19, llm/run2 10-19, llm_log_k5/run1 10-19/20), so the cause of the
P7′ gap is not known. All 4 GPUs serve Qwen (two TP2 replicas, 8011 and 8013); P9 reads its precomputed ranking and
needs no System One server.

- **D1 (drift baseline, fixed 2026-10-21 16:45, before the run).** `pb_run --arm llm_log --run 3` (the same policy
  as llm_log_k5/run1: K = 5 logged, the first executed) on the 54 P7 tasks, concurrently with P9. Comparison: run 3 −
  the pooled historical LLM runs (pb_gap.py, required-order checkpoints, task bootstrap). If ≥ +0.10 with a CI
  excluding 0, the drift is real: every comparison against historical LLM runs (P1, P2′, P5 gate; paper
  Table tab:gate) is re-done against run 3, and P7′ / P9 use run 3 as their LLM reference. If the CI includes 0 and
  the estimate is < +0.05, the P7′ arms' gain over the LLM is attributed to the screen arm itself (extra forced
  samples or the checklist turn) and is examined. Also reported: run 3 − P7′ freq arm (both today).
D1 and P9 (resumed: 19 + 21 tasks left) started 16:56, Qwen task-only comparator resumed 17:05 (6 workers).
Harness note: `fhir_document_reference_search_clinical_notes` (third_party/physicianbench/tools/fhir_api_functions.py:889)
has a parameter named `type`, so its error path `raise type(e)(...)` fails with "'NoneType' object is not callable":
when that search returns an HTTP error, the agent sees this message instead of the server's. It appears in 14-22% of
tasks in every run (historical and today alike), so it is not the drift; left unchanged to keep runs comparable.

**P9 interim mechanism (18:00, 11 / 6 tasks; exploratory).** Of the 33 orders the pplx12 checklist showed, 15 are
variants of required orders the run had already placed under another name (name dedup misses abbreviations and
synonyms), 2 are open required orders, 16 match no grader; random: 4 / 2 / 12 of 18. A ranker that scores what is
clearly indicated puts what was clearly done on top. Candidate fix (P9b, after P9 is read): drop pool items that a
System One model judges the same order as one already placed (a short-input question: the order and the list of
placed orders), then rank. Not applied to P9, which runs as fixed.
- **P8b (placed-aware ranking, offline, fixed 2026-10-21 18:10, before scoring).** As P8, but pplx12 also sees the
  base run's orders placed before its first FINISH (one step listing them; still a short input) and is asked whether
  the order is "indicated and still missing, not the same as or equivalent to an order already placed"
  (`pb_prior.py score --placed`, `report --placed`). Same 26 tasks, pool and target as P8. Prediction: top-3
  coverage of open specs ≥ 0.40 and above P8's 0.31 (paired by task). If it holds, P9b runs online with this ranker
  (the placed list taken from the run itself); if not, the variant problem is left to the LLM. Runs when a GPU frees.

## Research program from here (2026-10-21 18:30): where and how System One models belong in a clinical agent loop

The user's goal: a paper on System One models in clinical agent loops that every reader learns from. Working thesis,
to be confirmed or cut by the tests below: (1) the decision seat does not help, because deciding is not the
bottleneck (11% selectable, 19% never proposed; P1, P2′, P5); (2) System One models are better priors than readers:
the agent's trajectory anchors them on what the agent did (P6, P6c); (3) their useful role is a fast, cheap screener
on short inputs, paired with LLM generation, because the LLM can name most of what it misses from the task text
alone (P8). Lines of work, each pre-registered: G1 generality over all six System One models; G2 placed-aware
screening (P8b offline, then P9b online); D1 baseline drift; G3 cost (latency per judgment, System One vs LLM).

- **G1 (generality, fixed 2026-10-21 18:30, before any of these scores).** d3, linne, amos and decider31b score the
  P6 questions on the subset the two AUROCs are computed on (placed orders, and unplaced orders of the 21 tasks with
  a missed required order; 1,323 questions; `pb_screen.py score --subset`), once with the own state and once with
  the task instruction only (`--task`). With pplx12 and Clef: six models. Prediction: task-only > state on the
  unplaced AUROC for ≥ 5 of 6 models, and on the placed AUROC for ≥ 5 of 6; mean task − state difference on unplaced
  ≥ +0.10. If it holds, "the trajectory anchors System One models" is a property of the model class (claim in the
  paper); if 3-4 of 6, it is model-dependent (reported as such); ≤ 2, the claim is dropped. The Qwen comparator
  (task-only, running) says whether the LLM shares it. Runs when D1 and P9 free the GPUs (4 models in parallel).

## P9 verdict (2026-10-22 01:45, 24 tasks paired, 25 per arm): fails

The previous allocation's processes stopped at ~19:40 (session end); 25 of 26 tasks per arm had finished.
`results/pb_gap_p9_vs_random.json`, `pb_gap_p9_vs_llm.json`, `pb_gap_p9random_vs_llm.json`, `pb_p9_mechanism.json`.
**Primary, prior-pplx12 − prior-random: required-order checkpoints +0.045 [−0.07, 0.19]**; task pass +0.13 [−0.04,
0.29]; steps +6.0 [1.3, 11.3], notes −0.67 [−1.17, −0.17]. Versus llm/run1-2: pplx12 checklist +0.07 [−0.09, 0.24],
random checklist +0.01 [−0.11, 0.14]. Mechanism: the pplx12 checklist holds an open required order 0.29 per task
(random 0.16) and closes one in 3 tasks (random 3); after the pplx12 checklist the LLM places 0.42 orders, after the
random one 1.32: the top-ranked items are mostly orders already placed under another name, which the LLM rightly
declines (interim note above). The ranking signal of P8 survives online only weakly, and the slots are spent on
duplicates. Next: P8b (placed-aware ranking) decides whether P9b is worth running.

## G1 verdict (2026-10-22 03:00): holds; the trajectory anchors every System One model tested

`results/pb_screen.json`, paired per-task differences `results/pb_g1_paired.json`. AUROC (required vs not, per
task), own state -> task instruction only; unplaced orders (21 tasks with a missed required order) / placed (42):

| model | unplaced: state -> task (paired Δ, 95% CI) | placed: state -> task (paired Δ, 95% CI) | s/question (state, task) |
|---|---|---|---|
| pplx12 | 0.43 -> 0.65 (+0.22 [0.08, 0.38]) | 0.67 -> 0.82 (+0.15 [0.05, 0.25]) | 4.2, 0.39 |
| Clef | 0.40 -> 0.63 (+0.23 [0.09, 0.39]) | 0.66 -> 0.76 (+0.11 [0.00, 0.21]) | 3.7, 0.41 |
| d3 | 0.39 -> 0.73 (+0.34 [0.19, 0.50]) | 0.67 -> 0.86 (+0.20 [0.11, 0.30]) | 4.5, 0.39 |
| linne | 0.37 -> 0.61 (+0.24 [0.11, 0.38]) | 0.66 -> 0.77 (+0.11 [0.01, 0.22]) | 8.7, 0.60 |
| amos | 0.42 -> 0.69 (+0.27 [0.14, 0.43]) | 0.72 -> 0.83 (+0.11 [0.04, 0.19]) | 4.6, 0.39 |
| decider31b | 0.40 -> 0.57 (+0.17 [0.04, 0.31]) | 0.65 -> 0.74 (+0.09 [−0.01, 0.19]) | 0.34, 0.22 |

Pre-registered criteria: task > state on unplaced for 6/6 (≥ 5 required), on placed for 6/6 (≥ 5); mean
difference on unplaced +0.25 (≥ +0.10 required). Holds. With the state, all six rank the orders the agent missed
below chance (0.37-0.43); with the task text alone, all six rank them above (0.57-0.73). Claim for the paper: the
agent's trajectory anchors System One models on what the agent did; given the task alone they are useful priors.
The task-only judgment is also 7-20x faster (0.2-0.6 s per question vs 3.7-8.7 s with the state; Qwen thinking 41 s
with the state). LLM side: the Qwen task-only comparator is at 427 / 919 (interim unplaced 0.62, not read yet).

## P8b verdict (2026-10-22 03:05): fails, in the direction G1 predicts

`results/pb_prior_placed.json`. With the base run's placed orders in the input and the question "indicated and
still missing", pplx12's top-3 coverage of open specs falls from 0.31 to **0.15 [0.04, 0.31]** (random 0.11); top-5
0.35 -> 0.21, top-5 − random +0.03 [−0.10, 0.18]. Showing the System One model what the agent did makes it worse
again, even when the question is about what is missing. P9b is not run. Duplicate removal, if pursued, belongs
outside the System One model (code-level matching or the LLM).

- **G4 (causal anchoring, fixed 2026-10-22 03:10, before any counterfactual score).** For each unplaced order in
  the 21 tasks with a missed required order (919 questions), the own state plus one fabricated final step in which
  the agent places exactly that order (`fhir_service_request_create` / `fhir_medication_request_create` with the
  order's display and code, a generic "created" result); nothing about the patient changes, and the question still
  says "whether or not the agent has already placed it" (`pb_screen.py score --cf`). Models: pplx12 and d3 (state
  scores exist). Prediction: mean P(Yes) rises by ≥ 0.15 for required and for unneeded orders alike (the shift is
  label-blind), and the unplaced AUROC does not improve by more than 0.05. If so, the anchoring is causal: the
  System One judgment follows the agent's action, not the patient. If P(Yes) moves < 0.05, anchoring is not
  imitation of the agent's orders but something else in the trajectory (length, retrieval noise), to be probed next.

## G4 verdict (2026-10-22 03:45): partial; a causal, label-blind shift, smaller than predicted

`results/pb_g4.json` (919 unplaced orders, 21 tasks). One fabricated step placing the order, nothing else changed:
mean P(Yes) rises for required orders by +0.098 [0.056, 0.145] (pplx12) and +0.104 [0.066, 0.147] (d3), for
unneeded orders by +0.071 [0.063, 0.079] and +0.077 [0.069, 0.085]; unplaced AUROC 0.43 -> 0.44 (pplx12), 0.39 ->
0.42 (d3). The shift is causal and label-blind, as predicted, but below the pre-registered +0.15: the prediction
fails on magnitude. Following the agent's own orders explains part of the anchoring, not all of it (with the state,
missed required orders already score below unneeded ones before any fabricated step: 0.62 vs 0.69 for pplx12).
Next (exploratory, existing scores): whether the trajectory steers the judgment toward what the agent looked at
(salience), i.e. whether P(Yes) with the state tracks how much the order's terms appear in the trajectory.

**Salience (exploratory, 03:55).** With the state, P(Yes) tracks how many of the order's terms appear in the
trajectory twice as strongly as task-only P(Yes) (within-task Spearman 0.25-0.38 vs 0.13-0.19, 6/6 models). But
missed required orders appear in the trajectory slightly more than unneeded ones (term overlap 0.53 vs 0.46), so
salience does not explain why they score below chance.

- **G5 (what in the trajectory anchors, fixed 2026-10-22 04:00, before any score).** A third input: the
  retrieval-only state, the own state minus every step that contains an agent decision (any `*_create` call:
  orders, appointments, messages; `write_file`: notes), so the chart data the agent retrieved stays and its orders
  and written plan go (`pb_screen.py score --retrieval --subset`). Models: pplx12, d3, amos. Predictions: unplaced
  AUROC with retrieval-only ≥ state + 0.15 and ≥ task-only − 0.05, for ≥ 2 of 3 models. Reading: if it holds, what
  anchors is the agent's own decisions, not the chart, and the design rule is "give the System One model the chart,
  not the agent"; if retrieval-only also exceeds task-only by ≥ 0.05, chart data adds information once the agent's
  decisions are removed. If retrieval-only stays near the state, the agent's choice of what to retrieve is itself
  the anchor (the chart as the agent filtered it).

## D1 verdict (2026-10-22 04:10): no drift; the P7′ "gain" was task-subset noise; the gate stands

`results/pb_gap_d1_run3_vs_hist.json` and re-baselines. llm_log_k5/run3 (today's setup, 54 tasks) − the pooled
historical LLM runs: required-order checkpoints −0.004 [−0.094, 0.083], task pass −0.02 [−0.12, 0.07], orders
−0.01 [−0.56, 0.57], notes −0.16 [−0.44, 0.12]. As pre-registered (CI includes 0, estimate < +0.05) there is no
drift. On the ~30 tasks P7′ had finished, the historical runs happen to sit low (0.68-0.73 vs run 3's 0.79-0.83);
against run 3 the P7′ arms are +0.04 [−0.06, 0.15] (freq) and +0.04 [−0.06, 0.16] (pplx12-task) on required-order
checkpoints, with more orders (+1.8 / +1.1) and notes (+1.1 / +0.8): the pre-finish checklist makes the agent keep
working, it does not make it place what was missed. Gate (both runs) vs run 3 on the 54 tasks: orders +0.89
[0.28, 1.57], required-order checkpoints +0.06 [−0.02, 0.15], the same as against the historical runs (+0.88,
+0.06): Table tab:gate stands. Files: `pb_gap_p7freq_vs_run3.json`, `pb_gap_p7_vs_run3.json`,
`pb_gap_gate_vs_run3.json`, `pb_gap_gate_vs_hist_p7tasks.json`.
