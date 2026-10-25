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

## G5 verdict (2026-10-22 04:40): holds; the agent's decisions are the anchor, not the chart

`results/pb_screen.json`, paired `results/pb_g5_paired.json`. Unplaced AUROC, state -> retrieval-only (agent
decisions removed) -> task-only; paired Δ retrieval − state and retrieval − task (95% CI):

| model | state | retrieval-only | task-only | retr − state | retr − task | placed: state / retr / task |
|---|---|---|---|---|---|---|
| pplx12 | 0.43 | 0.64 | 0.65 | +0.21 [0.07, 0.37] | −0.01 [−0.13, 0.07] | 0.67 / 0.78 / 0.82 |
| d3 | 0.39 | 0.63 | 0.73 | +0.24 [0.10, 0.40] | −0.10 [−0.21, −0.02] | 0.67 / 0.83 / 0.86 |
| amos | 0.42 | 0.67 | 0.69 | +0.25 [0.11, 0.41] | −0.02 [−0.14, 0.06] | 0.72 / 0.80 / 0.83 |
| linne (ext.) | 0.37 | 0.62 | 0.61 | +0.25 [0.14, 0.38] | +0.01 [−0.10, 0.11] | 0.66 / 0.79 / 0.77 |
| decider31b (ext.) | 0.40 | 0.64 | 0.57 | +0.25 [0.13, 0.37] | +0.08 [−0.03, 0.19] | 0.65 / 0.75 / 0.74 |

Pre-registered (pplx12, d3, amos): retrieval ≥ state + 0.15 for 3/3; retrieval ≥ task − 0.05 for 2/3 (d3 misses).
Holds; the two extension models agree. Removing only the agent's decisions (164 of 729 steps: orders, appointments,
messages, notes) recovers the whole anchoring loss; the chart the agent retrieved does not mislead the System One
model, and it adds little beyond the task text either. Design rule for the paper: give the System One model the
chart, not the agent. With G4 (one fabricated order: +0.07-0.10, label-blind), placing an order is part of the
anchor but not all of it.

- **G6 (which decisions anchor, fixed 2026-10-22 04:45, before any score).** Two more inputs: the state without the
  agent's notes only (`write_file` steps removed, orders kept) and without its orders only (`*_create` steps removed,
  notes kept). Models pplx12, d3, amos; unplaced AUROC. Reported: the share of the retrieval-only recovery that each
  removal achieves. No directional prediction is made; a share ≥ 0.6 for one type and ≤ 0.4 for the other, in ≥ 2 of
  3 models, names that type as the main anchor; otherwise both contribute.

## The LLM is anchored too (Qwen task-only comparator, complete 2026-10-22 04:50; control added after P6c)

Qwen3.8-27B with thinking, task text only, unplaced subset: 759 of 919 questions answered (the rest ran out of the
8,192-token budget without a "P=" line, as in the state run). On the 674 questions both Qwen runs answered (15 tasks
with a computable AUROC): **state 0.46 -> task-only 0.62, paired +0.17 [0.07, 0.27]**. The same questions for the
System One models: pplx12 0.43 -> 0.66, d3 0.38 -> 0.72. Anchoring on the agent's trajectory is therefore not a
System One quirk: it is what happens to any judge that reads the agent's actions. What is specific to the System One
models is the cost of the remedy: with the task text alone they match or beat the thinking LLM (0.66-0.72 vs 0.62)
at 0.2-0.6 s per judgment vs ~80 s (about 150-350x less time), which is what makes wide screening affordable.
- **G5-LLM (fixed 2026-10-22 05:00, before the run).** Qwen thinking scores the unplaced subset (919) with the
  retrieval-only state (`pb_screen.py llm --retrieval`), same prompt and budget as its state and task-only runs.
  Prediction: on questions all three Qwen runs answer, retrieval-only ≥ state + 0.10 (state 0.46, task-only 0.62).
  If it holds, "strip the agent's decisions" is a rule for any judge, not only for System One models.

## G6 verdict (2026-10-22 06:01): the agent's orders are the anchor; its notes are not

`results/pb_g6_paired.json` (unplaced AUROC, 21 tasks, paired by task vs the state):

| model | state | no notes | no orders | retrieval-only | share of recovery: no notes / no orders |
|---|---|---|---|---|---|
| pplx12 | 0.43 | 0.43 (+0.00 [−0.04, 0.03]) | 0.60 (+0.17 [0.06, 0.30]) | 0.64 | 0.01 / 0.83 |
| d3 | 0.39 | 0.40 (+0.01 [−0.02, 0.04]) | 0.57 (+0.18 [0.05, 0.33]) | 0.63 | 0.04 / 0.74 |
| amos | 0.42 | 0.41 (−0.00 [−0.03, 0.03]) | 0.59 (+0.18 [0.06, 0.32]) | 0.67 | −0.00 / 0.71 |

By the pre-registered rule (one type ≥ 0.6, the other ≤ 0.4, in ≥ 2 of 3 models) the orders (every `*_create` step:
orders, appointments, messages) are the main anchor, in 3 of 3 models. Removing the agent's notes alone changes
nothing; the remaining 17-29% of the retrieval-only recovery appears only when both go (the note restates the orders,
so it anchors only once the orders themselves are gone). Placed AUROC moves the same way (no orders 0.74-0.83 vs
state 0.67-0.72). Reading: the System One model takes the agent's set of orders as the reference for what the task
needs, so whatever the agent never ordered looks unneeded: the checker inherits the agent's blind spot.

## G5-LLM verdict (2026-10-22 06:01): holds on the pre-registered point estimate, with low power

`results/pb_g5_llm.json`. Qwen thinking with the retrieval-only state answered 636 of 919 (the rest ran out of the
8,192-token budget, as in its other runs). On the 538 questions all three Qwen runs answered (10 tasks with a
computable AUROC): state 0.52, retrieval-only 0.64, task-only 0.68; retrieval − state +0.12 [−0.005, 0.26] (the
prediction, ≥ +0.10, holds on the point estimate; the CI touches 0), task − state +0.16 [0.01, 0.32]. The System One
models on the same questions: state 0.42-0.50 -> retrieval 0.55-0.65 -> task 0.71-0.79. Stripping the agent's
decisions helps the LLM judge as it helps the System One models; on these questions the System One models with the
task text alone (0.71-0.79) beat every Qwen input (≤ 0.68) at a median 0.2-0.6 s vs 41-69 s per judgment, and they
always return an answer, while Qwen fails to answer 17-31% of the questions within its budget.

- **G6-ext and G6-LLM (fixed 2026-10-22 06:02, before any score).** (a) linne and decider31b score the no-notes and
  no-orders states (generality of G6). Prediction: no-orders share ≥ 0.6 and no-notes share ≤ 0.4 for both. (b) Qwen
  thinking scores the no-orders state (`pb_screen.py llm --drop orders`), same prompt and budget. Prediction: on
  questions all of its state, retrieval-only and no-orders runs answer, no-orders achieves ≥ 0.6 of the
  retrieval-only gain over the state. If it holds, "the agent's orders anchor the judge" holds for the LLM too.

**Plan conformity (exploratory, 06:03, existing scores; `results/pb_plan_sim.json`).** For each unplaced order, its
lexical similarity to the agent's placed orders (max token Jaccard of the names). Missed required orders are less like
the agent's orders than the unneeded candidates are (mean 0.26 vs 0.34; similarity as a scorer: unplaced AUROC 0.41).
With the state, P(Yes) tracks that similarity within task: Spearman 0.28-0.38 for the five System One models and 0.47
for Qwen; with the task text only 0.05 (amos 0.16; Qwen 0.05); without the orders 0.18-0.19; retrieval-only
0.13-0.21. So a judge that reads the trajectory scores an order by how much it looks like what the agent already
did, and because the missed orders are by definition unlike the agent's plan, they land below chance: the judge
inherits the agent's blind spot. Caveats: lexical similarity only; the unneeded candidates come from runs' proposals,
so they are agent-adjacent by construction (as an in-run pool would be).

**Lexical dedup of the task-prior pool (exploratory, 06:04).** Dropping pool items whose name resembles a placed
order (token Jaccard ≥ 0.75 / 0.5 / 0.34) before the pplx12 ranking leaves top-3 coverage of the open specs at 0.35 /
0.27 / 0.27 (no dedup 0.31): missed orders often resemble placed ones by name (another test from the same family),
so name-based dedup removes them too. P9's failure is not fixed by a cheaper dedup; the task-prior route is capped
near a third of the missed orders at a 3-5 item budget.

- **G7 (can an instruction undo the anchoring? fixed 2026-10-22 06:05, before any score).** The state, with a
  debiasing sentence before the question ("Agents like this one often overlook orders that the task requires, so the
  orders it has placed may be incomplete. Judge the order from the task and the patient's data, not from whether it
  resembles what the agent has done."; `pb_screen.py score --warn`, unplaced subset, 919). Models pplx12, d3, amos.
  Prediction: the warning recovers < 0.25 of the retrieval-only gain over the state in ≥ 2 of 3 models (anchoring is
  in the input, not fixable by telling the model). If it recovers ≥ 0.6, the cheap remedy is an instruction and the
  design rule weakens to "warn the checker".

## G6-LLM verdict (2026-10-22 06:37): not testable; the questions Qwen answers in every condition carry no anchoring

`results/pb_g6_llm.json`. Qwen thinking with the no-orders state answered 656 of 919. On the 507 questions that its
state, retrieval-only and no-orders runs all answered (7 tasks with a computable AUROC) there is no anchoring to
recover: state 0.54, retrieval-only 0.55 (+0.02 [−0.04, 0.08]), no-orders 0.52 (−0.02 [−0.10, 0.05]), and the
System One models show none either on the same questions (pplx12 0.54 / 0.55, d3 0.47 / 0.43, amos 0.57 / 0.57,
state / retrieval-only). Requiring an answer from all three Qwen runs keeps a subset of tasks on which the agent's
orders happen not to anchor (not a length effect: answered and unanswered questions come from states of similar
length, median ~18.7k characters). The prediction can be neither confirmed nor refuted; the share is undefined. What
the run does show: Qwen leaves 29% unanswered (median 44 s per answered question), which itself limits it as a checker.
- **G7-LLM (fixed 2026-10-22 06:37, before the run).** Qwen thinking scores the unplaced subset with the state and
  the same debiasing sentence before the question (`pb_screen.py llm --warn`), same budget. Compared on questions its
  state, task-only and warned runs all answer. No directional prediction (a reasoning model may follow the
  instruction where a System One model cannot); reported: warned − state and the share of the task-only gain it
  recovers. Caveat from G6-LLM: the commonly answered subset may carry little anchoring; if state and task-only
  differ by < 0.05 there, the comparison is not interpretable.

## G6-ext verdict (2026-10-22 07:25): holds for decider31b, narrowly misses for linne; the pattern is the same

`results/pb_g6_paired.json` (now 5 models). Unplaced AUROC, state -> no notes / no orders (paired Δ vs state), share
of the retrieval-only recovery:

| model | state | no notes | no orders | retrieval-only | share: no notes / no orders |
|---|---|---|---|---|---|
| linne | 0.37 | 0.35 (−0.02 [−0.07, 0.04]) | 0.51 (+0.15 [0.05, 0.27]) | 0.62 | −0.07 / 0.57 |
| decider31b | 0.40 | 0.39 (−0.01 [−0.04, 0.03]) | 0.56 (+0.17 [0.07, 0.28]) | 0.64 | −0.02 / 0.68 |

Pre-registered: no-orders share ≥ 0.6 and no-notes share ≤ 0.4 for both. decider31b meets it; linne meets the notes
part and misses the orders threshold by 0.03. Across all five models the orders carry 0.57-0.83 of the recovery and
the notes −0.07-0.04 (removing the orders alone: +0.15 to +0.18, every CI excludes 0; removing the notes alone: no
CI excludes 0).

## G7 / G7-LLM verdict (2026-10-22 07:45): holds; an instruction does not undo the anchoring

`results/pb_g7.json`. Unplaced AUROC, state -> state with the debiasing sentence (paired Δ), share of the
remedy's gain recovered (System One: retrieval-only; Qwen: task-only on the 564 questions all three of its runs
answered, 11 tasks, where task-only is +0.18 over the state, so the comparison is interpretable):

| model | state | warned | Δ | remedy | share |
|---|---|---|---|---|---|
| pplx12 | 0.43 | 0.43 | +0.00 [−0.02, 0.03] | 0.64 | 0.02 |
| d3 | 0.39 | 0.39 | +0.00 [−0.01, 0.02] | 0.63 | 0.01 |
| amos | 0.42 | 0.43 | +0.02 [−0.01, 0.05] | 0.67 | 0.06 |
| Qwen (thinking) | 0.53 | 0.54 | +0.02 [−0.09, 0.11] | 0.71 | 0.08 |

Pre-registered (< 0.25 in ≥ 2 of 3 System One models): holds, 3 of 3. The models do read the sentence: mean P(Yes)
rises by +0.01-0.03 for both required and unneeded orders (Qwen +0.06 / +0.02), but the ranking does not change.
Telling the checker that the agent may have missed orders does not let it find them; what it reads has to change.
Qwen has 225 of 919 unanswered with the warning (24%).

## Toward the next online test (2026-10-22 08:17)

Why P9 failed, in numbers (existing scores, 26 tasks): in the pplx12-ranked task-prior list, 35% of the top-3 slots
(31% of the top-8) are variants of required orders the agent had already placed; top-m coverage of the missed specs
is 0.31 / 0.35 / 0.36 / 0.40 at m = 3 / 5 / 8 / 10 (random 0.11 / 0.17 / 0.26 / 0.31). The list spends its slots on
what the agent already did, and the indication judgment cannot be given the placed orders without being anchored
(P8b, G6). The design that follows from G6: ask the two questions separately.

- **P10 (factorized screen, offline; fixed 2026-10-22 08:17, before any score).** Each item of the whole task-prior
  pool gets a second, separate System One judgment: given only the placed orders (no task text), "is this candidate
  the same as or equivalent to one of the orders already placed?" (`pb_prior.py dedup`, pplx12). Ranking:
  P(indicated | task) × (1 − P(repeat)). Compared with the P8 ranking (name-based drop): top-3 coverage of the open
  specs and the share of top-3 slots that repeat a met spec. Prediction: factorized top-3 coverage ≥ 0.39 (P8
  0.31) and variant share ≤ 0.20 (P8 0.35). If it holds, P10 becomes the list of the next online arm (O1 below); if
  it fails, the dedup belongs to the LLM and O1 tests the uptake side only.

## P10 verdict (2026-10-22 08:20): fails on coverage; the repeat judge itself works

`results/pb_p10.json` (26 tasks). Factorized ranking (P(indicated | task) × (1 − P(repeat))) vs the P8 name-drop
ranking: top-3 coverage of the open specs 0.15 vs 0.31 (−0.15 [−0.31, −0.02]); top-5 0.29 vs 0.35; top-8 0.37 vs
0.36. Variant share of the top-3 slots 0.01 vs 0.35. Pre-registered (coverage ≥ 0.39 and variant share ≤ 0.20): the
variant part holds, the coverage part fails, so P10 fails. Diagnosis (exploratory): the repeat judgment is accurate
(pplx12 from the placed orders alone: mean P(repeat) 0.74 for variants of met specs, 0.12 for the missed specs, 0.16
for other items; per-task AUROC variants vs missed 0.93), but the product reorders the near-tied top of the
indication ranking by small differences in P(repeat), and generic high-indication items take the slots. A hard
filter instead of the product (drop P(repeat) ≥ 0.5, rank the rest by indication; chosen after seeing P10, so
exploratory): top-3 0.27, top-5 0.27, **top-8 0.55 vs 0.36 (+0.19 [0.01, 0.38])**, variant share 0.06, random 8
from the filtered pool 0.28. The separation of the two judgments works; what a 3-5 item list cannot do is cover
more than about a third of the missed orders.

- **O1 (online, the next allocation; fixed 2026-10-22 08:21, before any run).** The factorized checklist at the
  agent's first FINISH, on all 54 tasks of llm_log_k5/run3 (`results/o1_tasks.txt`; not selected on a past miss).
  Arms, both `pb_run --arm screen --b 0 --m 8 --tasks @results/o1_tasks.txt --run 1`:
  A `--ranker prior-pplx12-dedup`: the task-prior pool (built before the run from the task text alone, ranked by
  pplx12 P(indicated | task)), minus the run's placed orders by name, minus every item that pplx12, given only the
  run's placed orders, judges a repeat (P ≥ 0.5); the top 8 are shown with the P9 checklist text; the LLM decides.
  B `--ranker prior-random-dedup`: the same pool and repeat filter, 8 at random (control for the ranking).
  Reference: llm_log_k5/run3 (no checklist; D1 found no drift), with the historical LLM runs as a check.
  Primary: required-order checkpoints per task, A − run3, paired over 54 tasks; prediction ≥ +0.05 with the 95% CI
  excluding 0. Secondary: A − B ≥ +0.03; uptake (shown items placed) and the share of shown items that repeat a placed
  order (≤ 0.10 online); harms A − run3: not-in-rubric orders ≤ +1.0, no increase in forbidden orders, task pass not
  lower by more than 0.05. Reading: A > run3 with A ≈ B means the gain comes from the separated repeat check and the
  list, not the indication ranking (the System One role is then the repeat judge); A > B as well means both System One
  judgments matter. O1 uses m = 8 and the hard filter from the P10 exploration; the online outcome is new data.
  Smoke test first: 2 tasks, check `screen.pool`, `pool_after_dedup`, `flagged` and the screening time per FINISH.
  Smoke test (2026-10-22 08:31, run 99, 1 task, not part of O1): the arm runs end to end; at the first FINISH the
  pool held 77 items, 71 after the repeat filter, 8 were shown, and the LLM answered with 3 orders; screening plus
  the LLM's second turn took 104 s. Job exit code 1 is the usual "not every checkpoint passed" (41 of 52 P9 runs).

## O1 launched (2026-10-22 09:30, allocation 49587179, w007)

Both arms, 54 tasks, run 1, 14 workers each: `prior-pplx12-dedup` (port base 18100) and `prior-random-dedup`
(18300). Qwen TP2 on GPUs 0,1 (8011) and TP1 on GPU 2 (8013), tasks routed by name with
`8011/v1*2,8013/v1*1`, so a task meets the same replica in both arms (run 3's routing could not be reproduced exactly;
the replicas serve the same weights and sampling); pplx12 on GPU 3. Logs `logs/o1_{pplx12,random}_run1.log`.

## Why the agent errs: anatomy (exploratory, 2026-10-22 11:40; existing logs, CPU only)

The user's success criteria for this stage: (Q1) why a required order is missed in the first place (context drift,
a plan fixed too early, or a wrong task decomposition); (Q2) whether anchoring on its own orders causes the omission,
not only blocks recovery; (Q3) how Clinical Reasoning and Documentation checkpoints fail.

`scripts/pb_omission.py` -> `results/pb_omission.json` (llm_log_k5/run1, 100 tasks, 54 with a required-order spec;
run3, 54 tasks; 236 specs; 76 missed at the end of the run).
- **Where a missed order is lost** (first class that applies; the note's and text's classes use the grader's name
  regexes on free text, so they are upper bounds that include negated mentions): **noted** in the agent's own note
  but never placed 53% (run1) / 53% (run3); proposed by a sampled candidate but never executed 8% / 18%; only
  mentioned in a candidate's text 3% / 0%; **silent** (never surfaces) 36% / 30%.
- **The task-text prior knows them**: of the noted ones 100% (40/40) are in the task-only Qwen list (P8 pool); of the
  silent ones about half (14/25); placed ones 99%. About 14% of all misses (silent and not in the prior) look like a
  knowledge or decomposition gap; most of the rest the model can name and often writes down.
- **The order of work**: the executed turns end, in 91 of 154 runs, as retrieval -> one burst of orders -> the note
  -> FINISH ('ONF' or 'O.NF'); 20 runs write the note and finish with no order. Orders are one burst (73/100 and 50/54
  runs) at 73-76% of the run; placed required orders come at 74% of the run, at ~42-49k prompt tokens, the same
  point and context length at which the missed-but-proposed ones first appear (69-76%, 49-55k tokens): no sign that
  orders surfacing later, at longer contexts, are the ones lost. The note, the most explicit clinical reasoning of
  the run, comes after the orders, and the agent rarely returns to ordering after it.
- **Q3, first pass**: of the failed Clinical Reasoning checkpoints (161, four LLM runs) 23% fail because no note was
  written or it is empty; Documentation 10% (9/86); the rest are the grader's content judgments (to classify, A6).
Reading (hypothesis, to test): the dominant omission is an execution gap at the end of the episode: the agent orders,
then reasons in the note, and the note's further orders are never executed; context drift has no support yet.

- **A1-A6 (fixed 2026-10-22 11:40, before any sample; run after O1 on Qwen, all four GPUs).** `scripts/pb_replay.py`
  rebuilds the agent's exact context (its system prompt, its 14 tools, Qwen thinking, 16k tokens, K = 5 samples) at
  the first FINISH (51 cases with an open required spec, 69 open specs) and at the first order step (102 cases).
  Outcome per sample: share of the target specs that the sampled turn orders (finish: the open specs; order step: all
  required specs), duplicates of placed orders, number of orders; paired by case, bootstrap CIs.
  - A1 (Q1, execution gap): finish_nudge ("check that every order your note or plan recommends has been placed;
    place any not yet placed") − finish_none: coverage ≥ +0.15 with the CI excluding 0. If it holds, most omissions
    are said-not-done and are closed by a self-consistency check; if not, the noted mentions are not plans.
  - A2 (Q2, does seeing its own decisions suppress the missed orders at generation): finish_fresh − finish_ask, both
    with the same request ("place every order this task requires and that has not been placed yet"), fresh = the
    context with every order/message/appointment/note turn and its results removed: coverage ≥ +0.10 with the CI
    excluding 0. If it holds, the agent's own decisions suppress the orders it would otherwise generate (anchoring is
    a cause of omission, at least at the end of the run); duplicates are reported as the price.
  - A3 (Q1, context drift): order_remind (the task instruction repeated just before the first order step) −
    order_none: coverage of all required specs ≥ +0.05 with the CI excluding 0 supports drift; otherwise no support.
  - A4 (Q1, plan before acting / decomposition): order_plan ("write out your full assessment and plan, listing every
    order the task requires, then place every order in it") − order_none: coverage ≥ +0.05 with the CI excluding 0
    supports that acting before full reasoning loses orders.
  - A5 (Q1, what the note says, `pb_classify.py noted`): of the noted misses, RECOMMENDS + CLAIMS_DONE ≥ 0.5.
  - A6 (Q3, `pb_classify.py reasons`): shares of NO_NOTE / MISSING / WRONG / HEDGED / INCOMPLETE among failed
    Clinical Reasoning and Documentation checkpoints; no directional prediction.
  Order of runs: A5, A6 (minutes), A2, A1 (the finish point), then A3, A4.

## A5 / A6 verdicts (2026-10-22 16:34; local Qwen, no thinking, temperature 0; `results/pb_classify.json`)

Run early on the O1 replicas at low concurrency (the O1 drivers ran past 16:30, so the A1-A4 replay moves to the next
allocation; `a_run.sh` was stopped by hand at 16:21 before it could stop pplx12 or start anything).
- **A5 fails (0.40 < 0.5).** How the agent's own note names the 40 missed required orders it mentions (llm_log_k5 run1
  and run3): CLAIMS_DONE 30%, CONDITIONAL 25%, NEGATED 17.5%, OTHER 17.5%, RECOMMENDS 10%. RECOMMENDS + CLAIMS_DONE =
  0.40. So "noted" (53% of misses) is not mostly said-not-done: about a quarter are deferred or conditional and about
  a sixth are explicit decisions against. The large part is CLAIMS_DONE: the note says the order was placed and no
  executed order matches the grader's spec. Two readings, to separate before any claim: (a) the agent believes it
  placed something it did not (an execution or bookkeeping error), (b) it placed an equivalent order under a name the
  spec's pattern does not match (a grader artifact). Check: for each CLAIMS_DONE case, does any placed order of the
  run judge as the same order (local judge, placed orders vs the spec's names)?
  Consequence for A1: the pool of orders a self-consistency nudge can recover is about RECOMMENDS + CLAIMS_DONE of the
  noted misses, ~0.4 × 0.53 ≈ 21% of all misses (if CLAIMS_DONE is real); A1's +0.15 is ambitious on this base.
- **A6 (descriptive; 291 failed checkpoints, four LLM runs).** Clinical Reasoning (161): WRONG 41%, INCOMPLETE 25.5%,
  NO_NOTE 23%, HEDGED 5%, MISSING 4%. Documentation (86): WRONG 55%, INCOMPLETE 29%, NO_NOTE 10.5%, MISSING 6%. Data
  Retrieval (44): MISSING 61%, WRONG 18%, INCOMPLETE 11%, NO_NOTE 9%. The same pattern in llm and llm_log_k5 runs.
  Reading: reasoning and documentation fail mainly on content (a wrong value, interpretation or decision; then a
  missing required detail), not on hedging; about a fifth of reasoning failures are simply no note. Next split
  (proposed): WRONG into a misread or missing chart fact vs a wrong clinical judgment from correct facts, which
  decides whether retrieval or knowledge is the lever.
- **A5b (exploratory, 16:35; `pb_classify.py placed`, the check above).** Local Qwen, given the spec's name patterns
  and every order the run placed: of the 12 CLAIMS_DONE misses, 9 have no equivalent placed order (3 do: grader
  pattern misses). Over all noted misses, 4 of 37 judged have an equivalent placed order. So reading (a) dominates:
  in about a quarter of the noted misses (9/37) the agent's note states an order as placed that was never executed, a
  false completion claim, not a grader artifact. Together with RECOMMENDS (4), about a third of the noted misses
  (13/37) are orders the agent intended and believes or says it placed: the part a check of the note against the
  executed orders could close. The rest of "noted" is deferral (CONDITIONAL) or a decision against (NEGATED).

## O1 interim (2026-10-22 16:53, NOT the verdict: 49/54 pplx12, 51/54 random; the allocation ends ~17:00)

The remaining tasks (5 + 3) restart clean on the next allocation (stale claims, `pb_run` removes the unfinished job
dirs). Interim, `results/pb_gap_o1_*.json`, `results/pb_o1_mechanism.json`: required-order checkpoints A − run3
+0.009 [−0.07, 0.09] (49 tasks); B − run3 +0.06 [−0.03, 0.16] (51); A − B −0.045 [−0.15, 0.06] (48). Harms: orders
+1.4 / +1.7 per task, unmatched +1.3 / +1.7 (above the +1.0 bound), forbidden 0 -> 0. Mechanism, inside the run: the
pplx12 list held an open required spec in 15 tasks (random 11) and the agent placed it in 5 (random 1); variant
share of the list 0.04 / 0.015; uptake 6% / 4% of shown items. Between runs, the arms' own pre-checklist misses
differ (open specs at FINISH 0.81 vs 0.61 per task), which swamps the within-run closures at n ≈ 50.

## Allocation 49632386 (2026-10-22 22:48, w010): O1 resumed; A1-A4 replay alongside (`scripts/b_run.sh`)

- **A7 (Q3, fixed 22:50 before any label; descriptive, no directional prediction).** The A6 WRONG checkpoints
  (121) split by local Qwen from the grader's message: FACT (a record fact misstated or overlooked) vs JUDGMENT
  (facts right, the interpretation or decision wrong) vs UNCLEAR (`pb_classify.py wrong`). Reading: a FACT majority
  points to retrieval and reading the chart as the lever; a JUDGMENT majority to clinical knowledge or reasoning.
- **A7 result (23:22; `results/pb_classify.json` "wrong").** Of the WRONG failures: Clinical Reasoning (66) JUDGMENT
  70%, FACT 30%; Documentation (47) FACT 62%, JUDGMENT 38%; Data Retrieval (8) FACT 100%. Combined with A6, the
  reasoning failures are mainly judgment (0.41 × 0.70 ≈ 29% of failed reasoning checkpoints are a wrong decision from
  correct facts; 12% a misread fact; 26% incomplete; 23% no note), while documentation failures are mainly the record
  copied wrong (0.55 × 0.62 ≈ 34% a misstated fact). Two different levers: knowledge/reasoning for the plan,
  faithful reading of the chart for the note. Caveat: labels come from the grader's message, one local model, no
  human check.

## A1 / A2 verdicts (2026-10-23 02:02; `results/pb_replay.json`, 51 finish cases, 69 open specs, K = 5)

Coverage of the open required specs by the one sampled turn at the agent's first FINISH: as it was (none) 0.00
(0.06 orders; 72% finish again), nudge 0.078, ask 0.127, fresh 0.197 (any of 5 samples: 0, 0.14, 0.24, 0.47).
- **A1 fails on size: nudge − none +0.078 [0.026, 0.143]** (pre-registered ≥ +0.15). Real but small, consistent
  with A5/A5b: only about a third of the noted misses are intended-and-believed-placed, the part a self-check closes.
- **A2 fails on size: fresh − ask +0.071 [0.006, 0.139]** (pre-registered ≥ +0.10), the CI excludes 0. With the
  same request, removing its own decisions from the context raises coverage of the missed orders by about half
  (0.127 -> 0.197; any-of-5 0.24 -> 0.47), at a price: per sample 0.35 exact repeats of placed orders and 0.80
  variants of already-met specs (ask: 0 and 0), and 2.9 other orders vs 1.3. Reading: the agent's own decisions do
  suppress orders it would otherwise generate (Q2: some support that anchoring contributes to omission, smaller than
  predicted), but the fresh context also simply orders more, and part of the gain may be volume.
- Exploratory: ask − none +0.127 [0.063, 0.202]; asking for the remaining orders is itself worth more than the nudge.
- For R (online): an exact-name filter would remove only 0.35 of the 1.15 repeats per sample; the variants need a
  semantic repeat judge. The pplx12 repeat judge from P10 (placed orders only, AUROC 0.93 variants vs missed) is the
  natural System One role here.

## Allocation 49673289 (2026-10-23 03:42, w005)

49632386 ended before its time limit (by 03:41 a new allocation was running); O1 was still 3 tasks short (52/54,
53/54): the replay requests (n = 5, 16k thinking tokens) had filled the Qwen KV cache from about 23:00 and the O1
agents queued behind them (8-9 steps in 3 h); the order-step replay was paused at 02:03 and had written nothing.
Here O1 gets its own replica (Qwen TP2, 8011) and the replay the other (8013), so neither waits on the other.
- **R filter check (exploratory, 04:08; `pb_replay.py filter` -> `results/pb_replay_filter.json`).** R's online
  filter (exact repeats, then the pplx12 repeat judge from the placed orders alone) applied to the A2 samples: fresh
  0.174 vs ask 0.127 coverage of the open specs, +0.047 [−0.012, 0.112]; variants of met specs per sample 0.80 ->
  0.08; orders kept 1.5 vs 1.3 (fresh, ask). The filter equalizes volume and most of A2's gain survives it. Spread
  over all required-order checkpoints of 54 tasks this is about +0.03, far below the run-to-run noise of an
  independent online run (O1: ±0.1), so R is redesigned around a shared prefix (below).
- **Shared-prefix runs (code, 04:10-04:21).** `--ranker restart-{none,full,fresh}-p3`: each run replays the logged
  llm_log_k5/run3 trajectory of its task (the assistant turns, re-executed against a fresh environment, no LLM call)
  up to its FINISH and branches there; every arm starts from the same state. Smoke test (run 99, one task):
  restart-none reproduces run 3 exactly (4/7, the same failed checkpoints), restart-fresh executed 2 orders after
  the filter dropped 1 exact repeat and 4 variants. **Incident, fixed:** the logged turns name run 3's workspace by
  absolute path, so the first smoke test wrote its notes into run 3's job directory (one task,
  alcohol_use_disorder; for about two minutes its note held the smoke run's text). The note was restored from run 3's
  own logged write_file calls and verified identical; no other file was touched; run 3's grading (metadata) predates
  it. The replay now rewrites the logged run's directory to its own and refuses to start if any reference remains.

## O1 verdict (2026-10-23 04:38; 54/54 tasks per arm; `results/pb_gap_o1_*.json`, `results/pb_o1_mechanism.json`)

**Fails.** Primary, required-order checkpoints A (pplx12-ranked factorized checklist) − run3: −0.020 [−0.100,
+0.063] (pre-registered ≥ +0.05, CI excluding 0). Secondary A − B (random 8 from the same filtered pool): −0.090
[−0.201, +0.020] (pre-registered ≥ +0.03); B − run3 +0.069 [−0.020, +0.170]. Variant share of the shown list 0.041
(A) / 0.017 (B), within the ≤ 0.10 bound: the repeat judge works online. Harms: orders +1.15 (A) / +1.69 (B) per
task, unmatched (no rubric match) +1.13 / +1.67, above the +1.0 bound; forbidden 0 -> 0; task pass +0.02 / 0.00.
Mechanism inside the runs (screened FINISHes, 52 per arm): the A list held an open required spec in 18 tasks (B 12),
covering 0.46 of the open specs per task (B 0.25; P10 offline 0.55 vs random 0.28); the agent placed 0.17 open
specs per task from it (B 0.04), 6 tasks closed (B 2); uptake 5.5% / 4.1% of shown items. So the ranking puts the
missed orders on the list twice as often and they get placed four times as often, but the arms' own runs differed
before the checklist (open specs at FINISH 0.90 vs 0.62 per task) and that between-run difference is larger than
everything the checklist adds; at 54 tasks an independent online run cannot see an effect of this size.
Reading: the System One parts did their jobs (ranking from the task text; repeat judge from the placed orders), the
agent accepts about a third of the correct flags and adds about one non-rubric order per task; the design question
is now measurement, which the shared-prefix runs answer (R below).

- **R, the restart and the checklist from a shared prefix (fixed 2026-10-23 04:40, before any R run; the user asked
  for the restart experiment to be pre-registered after O1).** Cases: the A2 cases, 51 (logged run, task) pairs of
  llm_log_k5 run1 (25) and run3 (26) with a required-order spec open at the first FINISH (69 open specs;
  `results/r_tasks_{1,3}.txt`). Each job replays the logged run up to that FINISH (`--ranker <arm>-p{1,3}`, verified
  to reproduce run 3 exactly on one task) and branches there; K = 1; Qwen 8011, pplx12 for the repeat judge and the
  ranking. Arms: none (the logged FINISH; 1 run, the replay check); full (RESTART_ASK on the full context); fresh
  (RESTART_ASK on the context without its own decision turns); for both, only order calls are executed, minus exact
  repeats and minus items the pplx12 repeat judge (placed orders only) scores P ≥ 0.5; list (O1's pplx12-ranked
  factorized checklist, 8 items); listrand (8 random items of the same filtered pool). Full, fresh, list and listrand
  5 runs each (`scripts/r_run.sh`). Outcome per case, mean over runs: the share of the open specs' checkpoints that
  pass in the grader (`scripts/pb_r.py`). Paired over cases, bootstrap 95% CIs.
  - R1 (Q2, does seeing its own decisions suppress the missed orders, online): fresh − full ≥ +0.03 with the CI
    excluding 0. Basis: A2 after the same filter, +0.047 [−0.012, 0.112]; the expected SE is about 0.03, so the
    power is about one half; a null here is weak evidence.
  - R2 (asking at FINISH, online): full − none ≥ +0.08 with the CI excluding 0 (offline ask − none +0.127).
  - R3 (O1's question with the between-run noise removed): list − none ≥ +0.10 and list − listrand ≥ +0.05, each with
    the CI excluding 0 (O1 in-run: 0.17 vs 0.04 open specs closed per task, of 0.90 / 0.62 open).
  - Harms: unmatched orders (no required spec matched) per case fresh − full ≤ +0.5 and list − none ≤ +1.0; no
    forbidden orders; share of all checkpoints passing fresh − full and list − none ≥ −0.03.
  - Replay check: none agrees with the logged run on ≥ 0.90 of the checkpoints (the grader's LLM judge varies);
    below that, every result carries the caveat.
  - Exploratory: fresh − list (a free-form restart vs a System One checklist), cost per case.
  Not a held-out test of A2 (the same cases); what is new is the online grading, the executed filter, and the
  agent's continuation after the branch.

### R progress (2026-10-23 06:33)
Run 1 of all arms done at 06:31 (none at 04:45): 1h45 per run on 8011 alone, so runs 3-5 would not fit the
allocation (ends ~11:40). The order-step replay (A3, A4) on 8013 was paused (~8 min per case per condition at one
worker, 15-18 of 108 cases done; resumable, `pb_replay.py run` skips finished cases). A first split (8013 alone takes runs 3 and 5)
was stopped after 15 min with no job finished: both replicas are KV-bound (8011 holds ~16 agent contexts at a time,
8013 ~3), so 8013 alone would need ~9 h per run. Now (`scripts/r_run2.sh`, 06:52) run 3 uses 8013 at one worker per
driver while run 2 holds 8011, then runs 3-5 go one after another on both replicas (least-loaded routing). Replicas are identical; a task may meet a different replica
than in runs 1-2. Replay check after run 1: none agrees with the logged runs on 0.997 of 352 checkpoints. No verdict
numbers are read before the last run.

### R interim on runs 1-3 of 5 (2026-10-23 11:50; not the verdict)
Allocation 49673289 was killed at 09:19 (SLURM FAILED, exit 9:0, cause not shown; MaxRSS 157 GB of 1000 GB) during
run 4; runs 1-3 are complete for every arm (153 jobs per arm, none 51). Allocation 49691349 (w008, 11:30) gives this
project 2 GPUs, and pplx12 (49 GB) does not fit beside a TP2 Qwen, so runs 4-5 (they need pplx12 for the repeat
judge and the ranking) wait; the order-step replay (A3, A4; Qwen only) runs first. `scripts/pb_r.py --runs 1,2,3`:
- Replay check: 0.997 of 352 checkpoints (holds).
- R1 fresh − full +0.065 [−0.003, +0.144]: the size is past the bar (+0.03), the CI touches 0 (not yet held).
- R2 full − none +0.085 [+0.029, +0.160]: holds.
- R3 list − none +0.144 [+0.059, +0.239] holds; list − listrand +0.052 [−0.003, +0.114]: size past the bar (+0.05),
  the CI touches 0 (not yet held). listrand − none +0.092 [+0.039, +0.154].
- Harms: unmatched fresh − full +0.43 [−0.05, +0.92] (bound +0.5, holds), list − none +1.44 [+1.10, +1.82] (bound
  +1.0, fails; listrand − none +2.10); forbidden 0 in every arm; all-checkpoint pass fresh − full +0.029 [+0.004,
  +0.059], list − none +0.045 [+0.016, +0.082] (hold).
- Exploratory: fresh − list +0.007 [−0.088, +0.105]; fresh places 1.93 orders per case after the branch, list 1.71.
Reading so far: asking at FINISH recovers missed orders (+0.085), and removing the agent's own decisions from the
context adds about as much again (+0.065, CI touching 0); the pplx12 checklist reaches the same level as the fresh
restart, beats the random list mostly by adding fewer non-rubric orders (−0.65 unmatched). Runs 4-5 decide R1 and R3.

## A3 / A4 verdicts (2026-10-23 15:20; order-step replay, 102 cases, K = 5; `results/pb_replay.json`)
Run on allocation 49673289 (8013, slow) and 49691349 (Qwen TP2 on GPUs 0,1); a Lustre eviction of the node at 13:26
killed the replay processes silently once (no errored records; they resumed from the written cases).
- **A3 holds as pre-registered.** order_remind − order_none: coverage of all required specs at the first order step
  +0.108 [+0.063, +0.152] (0.333 -> 0.440); orders per sample 2.35 -> 2.88; finish 0.01 -> 0.03.
- **A4 holds as pre-registered.** order_plan − order_none: +0.186 [+0.132, +0.240] (-> 0.519; any of 5 samples 0.96);
  orders per sample 3.91.
- Exploratory (`scripts/pb_replay_order_x.py`, `results/pb_replay_order_x.json`), on the specs the logged run finally
  missed (open at its FINISH, 48 cases): remind − none +0.019 [−0.019, +0.054] (no recovery); plan − none +0.069
  [+0.019, +0.126]. Precision of placed orders (share matching a required spec): none 0.41, remind 0.41, plan 0.37
  (−0.037 [−0.076, −0.000]); unmatched orders per sample remind +0.28 [−0.02, +0.58], plan +1.15 [+0.80, +1.50];
  forbidden 0 everywhere.
Reading: the pre-registered metric counts all required specs at the first order step, so it mixes timing with
omission (79% of step omissions are placed later in the run, cycle 1). Repeating the task before the order step makes
the agent place more of the orders it would place anyway, at the same precision, and does not reach the orders the
run finally misses: context drift explains the step-level shortfall, not the final omissions. Planning first does
reach them (+0.07, about two thirds of A2's fresh-context effect at FINISH) at the cost of about one more non-rubric
order per step. So for Q1: the final omissions come less from losing the instruction than from acting before the
whole plan is written, and (A2, R) from anchoring on the orders already placed; a quarter of the misses the note
mentions are false completion claims (A5b).

### R runs 4-5 resumed (2026-10-23 20:13, allocation 49716204, w004, GPUs 0,1)
Allocation 49691349 ended at 17:19 like 49673289 at 09:19: the interactive step was killed (SIGKILL, CANCELLED, after
5 h 49 min and 5 h 40 min; MaxRSS 69 GB / 157 GB of 1000 GB), not a resource limit seen by SLURM. Run 4 stood at 75 of
204 jobs. `scripts/r_run3.sh`: Qwen TP1 on GPU 0 (8011) + pplx12 on GPU 1 (8022), runs 4 and 5, about 0.7 jobs/min
(129 run-4 jobs left, about 3 h).

## R verdict on runs 1-4 of 5 (2026-10-24 03:52; provisional until run 5; `results/pb_r.json`, `pb_r.py --runs 1,2,3,4`)
Allocation 49716204 was also killed (interactive step, after 55 min); run 4 finished on 49754860 (w009, GPUs 0,1) at
03:50. 204 jobs per arm, none 51. Run 5 runs now and may not fit this allocation; the final verdict uses all 5.
- Replay check 0.997 (holds).
- **R1 holds (narrowly):** fresh − full +0.066 [+0.003, +0.140] (bar +0.03, CI excluding 0). On runs 1-3 the CI
  touched 0; the lower bound is fragile.
- **R2 holds:** full − none +0.086 [+0.029, +0.159].
- **R3 holds:** list − none +0.150 [+0.069, +0.243]; list − listrand +0.059 [+0.007, +0.113].
- Harms: unmatched orders fresh − full +0.39 [−0.02, +0.82] (bound +0.5, holds); list − none +1.44 [+1.11, +1.79]
  (bound +1.0, **fails**; listrand − none +2.07, so the ranking cuts the random list's excess by a third); forbidden
  0 in every arm; all-checkpoint pass fresh − full +0.027 [+0.005, +0.053], list − none +0.047 [+0.018, +0.086].
- Exploratory: fresh − list +0.003 [−0.083, +0.093]; orders after the branch: full 1.26, fresh 1.90, list 1.71,
  listrand 2.21.
Reading: online, from the same point in the same run, (1) asking the agent at FINISH recovers missed orders
(+0.09); (2) taking its own decisions out of the context recovers more (+0.07 on top, Q2: its own decisions suppress
orders it can still generate); (3) a System One checklist built from the task text alone gets as far as the fresh
restart (0.53 vs 0.53), and its ranking beats a random list of the same pool (+0.06) with a third fewer non-rubric
orders. O1's null was between-run noise, as its in-run mechanism suggested. The price of every recovery arm is about
1-1.5 orders per case that no rubric item names; whether they are clinically reasonable is not measured here.

- **R3-d3, the checklist ranked by a second System One model (fixed 2026-10-24 10:27, before any d3 score of the
  prior pool exists).** Question: is R3 a property of pplx12, or of a System One prior read from the task text? d3
  is the best task-only ranker in G1 (unplaced 0.73 vs pplx12 0.65). d3 scores the same task-prior pool (Qwen,
  task text only, 8 samples) with the same task-only question (`S1CLIN_DECIDER=d3 pb_prior.py score --full`, export
  `--name d3`); the online arm `prior-d3-dedup-p{1,3}` is O1's list arm with d3's ranking, the same pplx12 repeat
  filter and the same 8 items, from the same shared prefix on the same 51 cases; K = 1; 3 runs (allocation time; the
  pplx12 list has 5). `pb_r.py` arm `listd3`, per case mean over its runs, paired over cases, bootstrap 95% CIs.
  - R3-d3a: listd3 − none ≥ +0.10 with the CI excluding 0.
  - R3-d3b: listd3 − listrand (all 5 listrand runs) ≥ +0.05 with the CI excluding 0. Power is limited (pplx12's
    list − listrand was +0.059 [+0.007, +0.113] on 4 runs); a miss here with R3-d3a holding reads as "a System One
    list helps, the ranking's edge over random is not shown for d3".
  - Harms as for list: unmatched listd3 − none ≤ +1.0, forbidden 0, all-checkpoint pass listd3 − none ≥ −0.03.
  - Exploratory: listd3 − list (d3 vs pplx12 ranking, same pool and filter); offline top-8 coverage of the open
    specs by d3's ranking after the repeat filter vs pplx12's (P10's function).

## R verdict, final (2026-10-24 11:35; all 5 runs; `results/pb_r.json`, `pb_r.py`)
Run 5 finished at 11:30 on allocation 49766322 (w008, GPUs 0,1). 255 jobs per arm (51 cases × 5 runs), none 51.
The verdicts are unchanged from the provisional reading on runs 1-4.
- Replay check 0.997 of 352 checkpoints (holds).
- **R1 holds:** fresh − full +0.071 [+0.010, +0.137] (bar +0.03, CI excluding 0). The lower bound is closer to 0
  than the mean suggests; the effect is real but small.
- **R2 holds:** full − none +0.080 [+0.026, +0.151] (bar +0.08; the point estimate sits at the bar).
- **R3 holds:** list − none +0.147 [+0.067, +0.241]; list − listrand +0.059 [+0.006, +0.118].
- Harms:
  - unmatched orders fresh − full +0.45 [+0.09, +0.85] (bound +0.5 on the mean, holds; the CI now excludes 0);
  - list − none +1.47 [+1.15, +1.82] (bound +1.0, **fails**; listrand − none +1.98, list − listrand −0.51
    [−0.86, −0.18]: the ranking cuts the random list's excess by a quarter);
  - forbidden 0 in every arm;
  - all-checkpoint pass fresh − full +0.027 [+0.005, +0.052], list − none +0.047 [+0.018, +0.086].
- Exploratory: fresh − list +0.004 [−0.078, +0.088]. Orders after the branch: full 1.20, fresh 1.87, list 1.75,
  listrand 2.11. Open-spec pass: none 0.38, full 0.46, fresh 0.53, list 0.53, listrand 0.47.

Reading (final; the provisional reading stands):
1. Asking at FINISH recovers missed orders (+0.08).
2. Removing the agent's own decision turns recovers more (+0.07 on top). This is the online answer to Q2: its own
   decisions suppress orders it can still generate.
3. A System One checklist from the task text alone recovers as much as the fresh restart (0.53 vs 0.53), and it
   beats a random list of the same pool.
4. Every recovery arm costs about 1-1.5 orders per case that no rubric item names (list − none fails the
   pre-registered harm bound). Whether these orders are clinically reasonable is not measured.

### R3-d3 progress and the offline coverage (exploratory; 2026-10-24 11:45)
d3 scored the whole task-prior pool (3576 orders, 54 tasks, none missing; 11:33-11:39) and its ranking is exported.
The online d3 list runs 1-3 started at 11:39 (`scripts/e_run.sh`, pplx12 back on GPU 1 for the repeat filter).
Offline, P10's function on its 26 tasks (`pb_prior.py factorized --name d3`, `results/pb_p10_d3.json`). Ranking
`filtered` is the online arm's: pplx12's repeat judge drops items with P ≥ 0.5, and the rest are ranked by the
decider. Coverage of the open specs:
- top 3: d3 0.33, pplx12 0.27; d3 − pplx12 +0.06 [0.00, +0.15].
- top 5: d3 0.44, pplx12 0.27; +0.17 [+0.06, +0.33].
- top 8 (the online arm's size): d3 0.57, pplx12 0.54; +0.03 [−0.04, +0.12].
At 8 items the two rankings cover about the same specs, so offline predicts listd3 ≈ list online. d3's edge is
in the first five slots.

## X, the agent-side findings with a second agent LLM: Gemma 4 31B-it (fixed 2026-10-24 12:01, before any Gemma output)
Every agent-side finding so far (Q1: A3/A4; Q2: A2, R1/R2; Q3: A5-A7) comes from one agent, Qwen3.8-27B. The user
chose Gemma 4 31B-it (google/gemma-4-31B-it, local cache) as the second agent.

Setup:
- Gemma runs as the agent in the same harness: MiniAgent, the same tools, prompts, 100 steps and 16k completion
  tokens. It is served bf16 with thinking on, 128k context, and sampling from its generation config
  (`scripts/vllm_gemma_up.sh`).
- Qwen stays the grader's judge (thinking off) and the Q3 labeller, so only the agent changes.
- `pb_run --agent gemma4 --agent-url ...`; job tags get the suffix `__gemma4`.
- Hardware: GPU 0 runs the Qwen judge (8011) and GPU 1 runs Gemma (8031). No new Qwen agent runs; Qwen comparisons
  use the existing numbers.

Tests:
- **X1, base runs.** Arm llm_log, K = 1 (each step logged, its one sample executed), all 100 tasks, runs 1 and 2.
  Descriptive only, no prediction:
  - checkpoint pass by category;
  - required-order checkpoints;
  - open required specs at the first FINISH;
  - coverage of the required orders by run 1 alone vs runs 1+2 pooled (Qwen: 75% per run, 91% over 5 runs);
  - each of these set beside Qwen's `llm` run 1.
- **XR, Q2 online (R with Gemma).** Cases are the (run, task) pairs of Gemma runs 1 and 2 whose first FINISH leaves
  a required-order spec open, as in R. Arms none (the replay check), full and fresh; K = 1; 3 runs each.
  - One change from R: the restart arms drop only exact-name repeats of placed orders, with no pplx12 repeat judge.
    pplx12 does not fit beside Gemma and the Qwen judge on two GPUs. Both arms use the same filter.
  - XR1: fresh − full ≥ +0.03 with the CI excluding 0 (Qwen R1 +0.071).
  - XR2: full − none ≥ +0.08 with the CI excluding 0 (Qwen R2 +0.080).
  - Harms: unmatched orders fresh − full ≤ +0.5; forbidden 0; all-checkpoint pass fresh − full ≥ −0.03.
  - Replay check ≥ 0.90.
  - A verdict needs at least 25 cases; with fewer, the result is reported as descriptive.
- **XA, Q1 offline (A3/A4 with Gemma).** `pb_replay.py` order-step replay at the first order step of Gemma run 1,
  K = 5 Gemma samples, the same three conditions (none, remind, plan). The confirmatory outcome is the one that
  carried the Qwen reading (exploratory there): coverage of the specs the run finally missed (open at its FINISH).
  - XA4: plan − none ≥ +0.03 with the CI excluding 0 (Qwen +0.069).
  - XA3: plan − remind ≥ +0.03 with the CI excluding 0 (Qwen: remind +0.019 n.s., plan +0.069). This is the claim
    that acting before the plan is written, rather than drift, loses the final orders.
  - The A3/A4 step-level metric (all required specs at the first order step) is reported as before.
- **XQ3, Q3.** A5-A7 classification of Gemma run 1's failed checkpoints with the same Qwen labeller and prompts
  (`pb_classify.py`). Prediction: for each checkpoint type, the dominant category matches Qwen's:
  - clinical reasoning: WRONG, mostly judgment;
  - documentation: WRONG, mostly fact;
  - data retrieval: MISSING.
  The A5 share of noted misses that are false completion claims is descriptive.

Order: X0 smoke test (2 tasks, run 99, not analysed), then X1 run 1, then XQ3 and XA (offline, from run 1), then
X1 run 2, then XR.

## R3-d3 verdict (2026-10-24 15:40; 3 runs, 153 jobs, 51 cases; `results/pb_r.json`)
The d3 list runs 1-3 ran 11:39-15:34. Every job finished and was graded.
- **R3-d3a holds:** listd3 − none +0.160 [+0.078, +0.252] (bar +0.10). Open-spec pass is 0.54 (pplx12 list 0.53).
- **R3-d3b holds:** listd3 − listrand +0.072 [+0.011, +0.133] (bar +0.05).
- Harms:
  - unmatched orders listd3 − none +1.52 [+1.16, +1.92] (bound +1.0, **fails**, as for the pplx12 list);
  - listd3 − listrand −0.46 [−0.83, −0.09];
  - forbidden 0;
  - all-checkpoint pass +0.051 [+0.023, +0.087].
- Exploratory: listd3 − list +0.013 [−0.014, +0.043] (unmatched +0.05). This matches the offline prediction: at 8
  items the two rankings cover about the same specs (0.57 vs 0.54).
Reading: R3 is not a property of pplx12. A second System One model ranking the same pool from the task text alone
gives the same recovery, and the same edge over a random list. The ranked lists also add a third fewer non-rubric
orders than the random list. The harm (about 1.5 non-rubric orders per case) is shared by every list.

### X amendment (2026-10-24 15:58, before any Gemma output)
Gemma 4 31B-it in bf16 does not fit on one H100. Its weights take 58 GiB and leave 11.8 GiB of KV
cache, while one 128k context needs 23.3 GiB, so vLLM refused to start. The other GPU holds the Qwen judge. Gemma
therefore runs with FP8 weights (vLLM online dynamic quantization) and a bf16 KV cache. Qwen stays bf16. Everything
else in X is unchanged. Any Gemma-vs-Qwen difference carries this caveat (a quantized agent), and the within-Gemma
contrasts (XR, XA) do not.

### X progress (2026-10-24 16:05)
Gemma (FP8) is served on GPU 1 (8031), with the Qwen judge on GPU 0 (8011). A toy tool-call request returns a parsed
tool call and separate reasoning. The X0 smoke test (run 99) completed one task end to end: retrieval, two orders, a
note and FINISH, graded by Qwen. X1 run 1 (100 tasks, 8 workers) started at 16:05.

## X1 run 1 (descriptive) and XQ3 verdict (2026-10-24 17:05; `results/pb_x1_run1.json`, `pb_omission_gemma4.json`, `pb_classify_gemma4.json`)
Run 1 took 100 tasks in 51 min (16:05-16:56, 8 workers).

**X1, Gemma beside Qwen (descriptive).** Qwen's numbers are from `llm` run 1 and the executed path of
llm_log_k5 run 1. Gemma − Qwen is paired over tasks.

| | Gemma | Qwen | Gemma − Qwen |
|---|---|---|---|
| task pass | 0.08 | 0.41 | −0.33 [−0.42, −0.25] |
| checkpoint pass | 0.51 | 0.81 | −0.30 [−0.35, −0.24] |
| required-order specs placed | 0.40 | 0.67-0.72 | −0.29 [−0.40, −0.18] |
| steps | 16 | 36 | −20 |
| orders | 1.0 | 3.1 | |
| unmatched orders | 0.7 | 3.0 | |

- Documentation checkpoints pass at 0.22 for Gemma vs 0.69-0.71 for Qwen.
- 56 of 100 Gemma runs place no order (Qwen 27).
- Not a harness artifact:
  - no FINISH holds an unparsed tool call;
  - no turn reaches the token cap;
  - 90 runs end in a FINISH with a text answer (median 1.5k characters, after a median of 8 steps);
  - 8 runs end on the harness's own repeated-call abort, as can happen for Qwen;
  - 2 runs end on a request error.
- Gemma is a much weaker agent in this harness: it stops early and acts little.
- Missed required specs (Q1 anatomy):
  - Gemma: silent 0.58, noted 0.35, mentioned 0.07;
  - Qwen run 1: silent 0.36, noted 0.53, proposed 0.08.

**XQ3 fails as pre-registered (1 of 3 checkpoint types match).**
- Data retrieval: MISSING is dominant for both (Gemma 0.55, Qwen 0.61). **Matches.**
- Clinical reasoning: Gemma's dominant category is INCOMPLETE (0.38; WRONG 0.19), Qwen's is WRONG (0.41). **Does
  not match.** Within WRONG the split does match: judgment 0.62 (Qwen 0.70).
- Documentation: Gemma's dominant category is INCOMPLETE (0.57; WRONG 0.17), Qwen's is WRONG (0.55). **Does not
  match.** Within WRONG (n = 13): judgment 0.54 (Qwen fact 0.62).
- A5 (descriptive): Gemma's 25 noted misses are mostly RECOMMENDS (0.60; Qwen 0.10). CLAIMS_DONE is 0.16, and for
  0.96 of them no placed order is equivalent, so about 0.16 are false completion claims (Qwen about a quarter).

Reading:
- Qwen fails by saying the wrong thing; Gemma fails by saying too little. The Q3 error profile depends on the agent.
- Within "wrong", clinical reasoning failures are judgment for both agents.
- Gemma also shows an execution gap that Qwen does not: it recommends orders in its note and never places them
  (15 of 25 noted misses). For Qwen, A1's nudge recovered only +0.08.

### XA verdict (2026-10-24 18:20; results/pb_replay_order_x_gemma4.json, pb_replay_gemma4.json; Gemma run 1, K=5)
Finally-missed specs (18 cases with an open spec): none 0.10, remind 0.17, plan 0.28.
- XA4 plan − none +0.177 [0.041, 0.322]: **holds** (bar +0.03, CI excludes 0).
- XA3 plan − remind +0.114 [0.022, 0.228]: **holds**.
- remind − none +0.063 [−0.026, 0.152], n.s. (as for Qwen: a reminder alone does not close misses).
- Step-level coverage (33 cases): plan − none +0.180 [0.073, 0.276]; remind − none +0.017 n.s.
- Cost: plan adds unmatched orders +0.42 [0.18, 0.68] per step and precision −0.03 n.s.; forbidden 0.
Reading: the Qwen pattern replicates and is larger on Gemma (Qwen plan +0.07 on finally-missed specs): asking the
agent to plan its orders first, not reminding it, recovers omissions; the price is extra unmatched orders.
Allocation 49766322 ended 18:19 during X1 run 2 (54 rc lines logged) and before XR.

### X1 full (2026-10-24 21:25; results/pb_x1.json; descriptive)
Gemma runs 1 and 2 agree: task pass 0.08 / 0.05, checkpoint 0.51 / 0.49, steps 16 / 15, orders 1.0 / 1.0, required
specs placed 0.40 / 0.40. Gemma − Qwen over 100 tasks (mean of each agent's two runs): checkpoint −0.31 [−0.36, −0.25],
task pass −0.35 [−0.43, −0.27], required placed −0.29 [−0.40, −0.19] (54 tasks with a spec), orders −2.2, steps −20.
Two-run pool coverage of required specs: Gemma 0.40 → 0.46, Qwen 0.67 → 0.73 (+0.06 for both). XR cases: run 1 32,
run 2 33 (65 ≥ 25).

### XR verdict (2026-10-24 22:52; results/pb_r_gemma4.json; 65 cases = Gemma run 1 (32) + run 2 (33); none ×1, full and fresh ×3)
Replay agreement 0.996 (441 checkpoints; ≥ 0.90 ok). Open required-order checkpoints: none 0.29, full 0.47, fresh 0.56.
- XR1 fresh − full +0.089 [0.032, 0.153]: **holds** (Qwen R1 +0.071 [0.010, 0.137]).
- XR2 full − none +0.178 [0.106, 0.257]: **holds** (Qwen R2 +0.080).
- Harms: unmatched fresh − full +0.23 [−0.07, 0.51] (≤ +0.5 ok); forbidden 0; all-checkpoint fresh − full +0.024
  [0.010, 0.042] (≥ −0.03 ok).
- Descriptive: fresh − none +0.267 [0.185, 0.353]; unmatched full − none +0.95, fresh − none +1.18.
Reading: Q2 replicates on a second, much weaker agent from another family, and both effects are larger. Asking at
FINISH recovers omitted orders (the agent can still generate them), and removing its own decisions from the context
recovers more (anchoring on its own orders suppresses them). Caveat: restart arms use the exact-name repeat filter
only (no pplx12 judge), as pre-registered.

### XR3 pre-registration (2026-10-24 22:53, before any XR3 job; Gemma agent)
Question: does the System One checklist from the task text (R3: pplx12-ranked task-prior pool, top 8, shown at the
agent's first FINISH) also help the second agent? Same 65 XR cases and shared prefixes; arms list
(`--ranker prior-pplx12-p{1,2}`) and listrand (`prior-random-p{1,2}`, the same pool shuffled); K = 1; 3 runs each;
compared with XR's none (and descriptively with full / fresh).
- One change from R3, as in XR: no `-dedup` repeat judge (pplx12 does not fit beside Gemma and the Qwen judge);
  only orders already placed by exact name leave the pool. Both list arms share this.
- XR3a: list − none ≥ +0.08 with the CI excluding 0 (Qwen R3 +0.147).
- XR3b: list − listrand ≥ +0.03 with the CI excluding 0 (Qwen +0.059).
- Harms: forbidden 0; unmatched list − none ≤ +1.0 (the R bound; Qwen failed it at +1.47, reported either way).
- Descriptive: list − fresh, all-checkpoint pass.
Script: scripts/x_run3.sh; analysis scripts/pb_r.py with S1CLIN_AGENT=gemma4 (arms list, listrand added).

### XR3 verdict (2026-10-25 00:58; results/pb_r_gemma4.json; 65 cases, list and listrand ×3 runs, 195 jobs each)
Open required-order checkpoints: none 0.29, full 0.47, fresh 0.56, list 0.58, listrand 0.45.
- XR3a list − none +0.292 [0.200, 0.395]: **holds** (Qwen R3 +0.147).
- XR3b list − listrand +0.136 [0.065, 0.217]: **holds** (Qwen +0.059).
- Harms: forbidden 0; unmatched list − none +1.56 [1.18, 1.97]: **fails** the +1.0 bound (as for Qwen, +1.47).
  The random list costs more (+2.27) and the ranking cuts it (list − listrand −0.71 [−1.00, −0.41]).
- Descriptive: list − fresh −0.026 [−0.10, 0.04] (as for Qwen: the checklist matches a fresh restart); all-checkpoint
  list − none +0.078 [0.053, 0.107].
Reading: the S1 checklist from the task text helps the second agent too, about twice as much in absolute terms
(lower baseline), and its ranking matters more than for Qwen. The cost, orders no rubric item names, replicates.

### H pre-registration (2026-10-25 07:29, before any rating; scripts/pb_extra.py)
Question: the checklist arms fail the unmatched-order harm bound for both agents (R3 +1.47, R3-d3 +1.52, XR3 +1.56).
Are those orders, which no rubric item names, clinically inappropriate, or reasonable care the rubric does not list?
- Units: per case (R: 51 Qwen cases; XR: 65 Gemma cases), the distinct orders placed by the agent before its first
  FINISH (`base`) and after the branch by each arm in any run (full, fresh, list, listrand; listd3 for Qwen). Each is
  matched (meets a required spec) or unmatched.
- Judges: Qwen3.8-27B and Gemma 4 31B-it (FP8), thinking on, temperature 0, each rating every order of every case.
  Input: the fresh context at the FINISH (task and retrieved chart, no decision turns) and the case's orders in a
  seeded shuffle with no source shown. Scale: A indicated, B reasonable, C low value, D potentially harmful.
  Inappropriate = C or D. Shares pooled over orders, CIs by cluster bootstrap over cases.
- Validity: matched (required) orders rated inappropriate ≤ 0.20 by a judge, else that judge is descriptive only.
- H1 (non-inferiority): inappropriate share of the list arm's unmatched orders − that of the agent's own (base)
  unmatched orders, upper 95% bound < +0.10. Per agent × valid judge; holds if it holds in all of them.
- H2: list − listrand inappropriate share < 0, CI excluding 0 (the S1 ranking picks better extras); same rule.
- H3: potentially harmful (D) share of the list arm's unmatched orders ≤ 0.05, upper bound ≤ 0.10; same rule.
- Descriptive: full, fresh, listd3, judge agreement (Cohen's κ, 4-way and inappropriate yes/no).
- Caveats fixed in advance: LLM judges, not physicians; each agent's own model is one of the judges (self-preference
  is possible, hence both); the judge does not see the orders already placed, so duplication is not rated here.

### H verdict (2026-10-25 11:06; results/pb_extra.json)
Coverage: Gemma judge rated all cases (Qwen agent 51, Gemma agent 65); the Qwen judge returned no parsable answer
(3 tries) for 19/51 and 13/65 cases (long thinking on lists of up to 99 orders).
- Validity (matched orders rated inappropriate ≤ 0.20): Gemma judge 0.17 (Qwen-agent cases), 0.06 (Gemma-agent
  cases): valid. Qwen judge 0.51, 0.40: **fails** → descriptive only, as pre-registered.
- Gemma judge, inappropriate (C or D) share of unmatched orders: Qwen agent base 0.21, list 0.26, listrand 0.34,
  fresh 0.32, full 0.28, listd3 0.23; Gemma agent base 0.26, list 0.17, listrand 0.25, fresh 0.19, full 0.19.
- H1 list − base, upper bound < +0.10: Qwen agent +0.05 [−0.09, 0.20] (fails, inconclusive); Gemma agent −0.08
  [−0.28, 0.08] (holds). **Does not hold** (1 of 2).
- H2 list − listrand < 0, CI excluding 0: Qwen agent −0.08 [−0.18, 0.03] (fails); Gemma agent −0.07 [−0.14, −0.003]
  (holds). **Does not hold** (1 of 2).
- H3 potentially harmful share of list unmatched orders ≤ 0.05, upper ≤ 0.10: 0.000 [0, 0] for both agents:
  **holds** (Qwen judge also 0 D among list orders).
- Agreement between judges: κ 0.31 / 0.28 (4-way), 0.31 / 0.21 (inappropriate yes/no): fair.
Reading: by the valid judge about three quarters of the orders that no rubric item names are indicated or reasonable,
at the rate of the agent's own unmatched orders, and none from the S1 list is rated potentially harmful; the formal
non-inferiority and ranking claims are not established (wide CIs, one judge).
Post-hoc observation (not a test): each case's orders were rated in one list mixing all arms, so near-duplicate
variants of one order sit side by side, and the Qwen judge marks the later copies low value (its matched orders
placed after the branch: fresh 0.61 inappropriate vs base 0.02). List composition thus confounds arm comparisons.
H-b below rates each order alone.

### H-b specification (2026-10-25 11:07, before any H-b rating; exploratory follow-up, not a new confirmatory test)
Each order of the H items rated alone (one call per order; same context, scale and judges; `pb_extra.py judge
--single`, max 8192 tokens, 3 tries), which removes the list-composition confound. Analysis as H (validity rule,
H1-H3, κ), reported as H-b in results/pb_extra_single.json; it replaces neither H's verdict nor its numbers.
H-b amendment (11:28, before any H-b result was looked at): the Qwen judge rates ~3 orders/min with thinking (≈16 h for
all 2879), so it rates a seeded random sample of 150 orders per agent (`--sample 150`; its first 50 sequential
ratings are set aside unused). The Gemma judge rates all orders. Qwen-judge numbers are for validity and agreement.

### H-b result (2026-10-25 12:43; results/pb_extra_single.json; exploratory, each order rated alone)
Gemma judge: all 2879 orders; Qwen judge: seeded samples (148 of 150 and 150 of 150 parsed).
- Validity (matched rated inappropriate ≤ 0.20): Gemma-agent cases, Gemma judge 0.06, Qwen judge 0.05: valid.
  Qwen-agent cases, Gemma judge 0.21, Qwen judge 0.24 (n = 17): just over the bar, so those are descriptive.
- Gemma-agent cases, unmatched orders inappropriate (Gemma judge / Qwen judge): base 0.20 / 0.30, list 0.09 / 0.05,
  listrand 0.20 / 0.41, fresh 0.11 / 0.26, full 0.16 / 0.32.
  - H1 list − base: −0.11 [−0.26, 0.02] / −0.25 [−0.60, 0.08]: upper bounds < +0.10 under both judges.
  - H2 list − listrand: −0.11 [−0.18, −0.05] / −0.36 [−0.52, −0.23]: both exclude 0.
  - H3 harmful share of list orders: 0 / 0.
- Qwen-agent cases (Gemma judge, descriptive): base 0.22, list 0.19, listrand 0.33, fresh 0.33, full 0.33, listd3
  0.23; list − base −0.03 [−0.14, 0.08], list − listrand −0.14 [−0.22, −0.06], list − fresh −0.14 [−0.25, −0.02];
  harmful list 0 (fresh 0.018, full 0.028, base 0.029).
- Agreement κ (inappropriate yes/no) 0.49 / 0.48, up from 0.31 / 0.21 in H (the list confound added noise).
Reading: rated one at a time, the orders that the pplx12 checklist adds and no rubric item names are as appropriate as
the agent's own unmatched orders (or more), clearly more appropriate than a random list's, more than the restart's
extras for Qwen, and never rated potentially harmful. The "unmatched-order harm" in R3, R3-d3 and XR3 is mostly
care the rubric does not list, not inappropriate care, by two LLM judges (no physician check yet).
