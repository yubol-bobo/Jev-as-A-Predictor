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
