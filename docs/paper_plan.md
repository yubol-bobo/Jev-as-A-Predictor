# Paper plan (2026-10-22, updated 2026-10-25: paper restructured as "Fresh Eyes", main.tex; this file is now background)

Goal: a paper on System One (S1) models inside clinical agent loops that every reader takes something from. The
current draft (`paper/main.tex`, "Who Decides?") covers the decision seat. The 2026-10-21/22 loop (`research_loop.md`,
P6-G7) adds a second, more general finding. Proposed arc: **where an S1 model helps a clinical agent, and why
reading the agent's work is the wrong input.**

## Claims, in order, with evidence

1. **The decision seat does not help; deciding is not the bottleneck** (existing §5.1-5.4). 11% of steps are
   selectable; 19% of required orders are never proposed; S1 deciders show a commitment bias (finish early, fewer
   notes). Gate (S1 decides only at order points) adds orders, +0.89 [0.28, 1.57], mostly outside the rubric
   (`tab:gate`; D1 confirms vs a same-day LLM run, `pb_gap_gate_vs_run3.json`).
2. **What is missing is the orders nobody proposes, and the knowledge to name them exists.** A 5-run pool covers 91%
   of required orders vs 75% per run. Qwen without thinking, given only the task text, names 84% [0.71, 0.95] of the
   orders its own run misses (P8, `pb_prior.json`): a knowing-doing gap.
3. **A checker that reads the agent's work inherits its blind spot** (new headline).
   - With the agent's trajectory, all 6 S1 models rank the orders the agent missed *below chance* (unplaced AUROC
     0.37-0.43); given the task text alone 0.57-0.73, paired +0.17 to +0.34, all CIs exclude 0 (G1, `pb_g1_paired.json`).
   - Not an S1 quirk: Qwen with thinking goes 0.46 -> 0.62 (task-only) on the same questions.
   - What anchors: the agent's decisions, not the chart (G5: retrieval-only recovers +0.21-0.25, 5/5 models), and
     among decisions the orders, not the notes (G6: 71-83% of the recovery from removing orders, ~0 from notes).
   - Causal: one fabricated step placing an order raises P(Yes) by +0.07-0.10 regardless of the label (G4).
   - Mechanism: with the trajectory, scores follow similarity to the agent's orders (Spearman 0.28-0.47; task-only
     0.05); missed orders are unlike the agent's plan (AUROC 0.41), so they land below chance (`pb_plan_sim.json`).
   - Not instruction-fixable: a debiasing sentence ("the agent may have missed orders; judge independently")
     recovers 1-6% of the gap for S1 models and 8% for Qwen thinking (G7, `pb_g7.json`): the input must change.
   - G6 generality: across 5 models orders carry 57-83% of the recovery, notes ~0 (linne 0.57 just under the bar).
     G6-LLM untestable (Qwen's commonly answered questions show no anchoring).
4. **The S1 advantage is that the remedy is cheap.** Task-only S1 judgments take 0.2-0.6 s and always answer; Qwen
   takes 41-69 s and fails to answer 17-31% within an 8k budget; on common questions task-only S1 (0.66-0.79) matches
   or beats every Qwen input. That is what makes screening hundreds of candidate orders per task affordable.
5. **Closing the loop online works when measured from a shared prefix** (updated 2026-10-24; R, 51 cases, 5 runs,
   `pb_r.json`). O1's independent-run null (−0.02) was between-run noise. Branching the same run at its first
   FINISH, a System One checklist from the task text alone (pplx12-ranked, repeat-filtered, 8 items) raises the open
   required-order checkpoints from 0.38 to 0.53 (+0.15 [0.07, 0.24]). It beats a random list of the same pool (+0.06
   [0.01, 0.12]) and matches a fresh-context restart (0.53). The price is about 1.5 orders per case that no rubric item
   names (harm bound +1.0 fails; random list +2.0), and their clinical value is unmeasured. R3-d3 (d3-ranked list)
   tests whether this is pplx12-specific: offline at 8 items d3 ≈ pplx12 (0.57 vs 0.54); online it holds as
   pre-registered (3 runs: listd3 − none +0.16 [0.08, 0.25], − random list +0.07 [0.01, 0.13], ≈ pplx12 list
   +0.01 n.s.; same unmatched-order cost +1.5). The effect belongs to S1 ranking of the task prior, not to one model.
   What the unmatched orders are (H, H-b; two LLM judges, blinded, each order rated alone): as appropriate as the
   agent's own unmatched orders (Gemma judge, Qwen agent 0.19 vs 0.22 inappropriate), more appropriate than a random
   list's (−0.14 [−0.22, −0.06]) or the restart's extras, and none rated potentially harmful. The "harm" is mostly
   care the rubric does not list. Physician check still missing.

## Why the agent misses orders (Q1-Q3; Qwen3.8-27B, with Gemma 4 31B-it (FP8) as a second agent, X)
- **Q1, when the omission happens.** Not drift: reminding the agent of the task at its first order step moves orders
  earlier (+0.11) but not into the final record (+0.02 n.s. on specs finally missed). Asking for a full plan first
  helps the final record (+0.07 [0.02, 0.13]) (A3/A4). The agent acts before it has enumerated the plan, and orders
  it never enumerated are never placed. 19% of required orders are never proposed in a run; 91% are covered across 5
  runs, so the miss is a sampling miss rather than a knowledge gap (claim 2).
- **Q2, anchoring.** Online: a restart without its own decision turns recovers +0.07 [0.01, 0.14] more than the same
  question with them (R1). Asking at FINISH recovers +0.08 (R2). The agent's own orders suppress orders it can still
  generate. This mirrors claim 3, where the checker's input is anchored by the same orders.
- **Second agent (X, Gemma 4 31B-it, FP8).** Gemma is a much weaker agent here (checkpoint pass 0.51 vs 0.81; 56% of
  its runs place no order; it FINISHes after ~8 steps with text). Q1 replicates and is larger: planning first closes
  +0.18 [0.04, 0.32] of finally-missed specs, a reminder does not (+0.06 n.s.), plan − remind +0.11 [0.02, 0.23]
  (XA). Q3 does not replicate (XQ3 fails, 1 of 3 checkpoint types): Gemma's errors are INCOMPLETE where Qwen's are
  WRONG, and 60% of the misses it notes are recommendations it never executes (Qwen 10%): an execution gap rather
  than a judgment gap. The error profile is agent-specific; the planning remedy is not. Q2 replicates (XR, 65 cases): fresh − full +0.09 [0.03, 0.15], full − none +0.18.
  Claim 5 replicates (XR3): the pplx12 task-prior checklist raises open required checkpoints 0.29 → 0.58
  (+0.29 [0.20, 0.40]), beats a random list +0.14 [0.07, 0.22], matches a fresh restart; unmatched cost +1.6.
- **Q3, other errors.** About a quarter of the misses the agent notes are false completion claims (A5b). Clinical
  reasoning errors are mostly wrong judgment (70%), documentation errors mostly wrong facts (62%), and data retrieval
  errors are omissions (A6/A7; Qwen-labelled, no human check).
- **Limits.** Two agent models, the second FP8 and much weaker. Qwen labels the classifiers and is the grader's judge. Extra orders are not assessed
  clinically. No held-out end-to-end test of the best combination.

## Design rules for practitioners (Discussion)
- Do not put an S1 model in the decision seat for open-ended clinical work; use it at commitment points at most.
- To check an agent, give the checker the task (and the chart), not the agent's decisions, whatever the checker is.
- Use the S1 model's speed for breadth (screen many candidates), and let the LLM generate the candidates.

## Proposed restructure
- Keep §§1-4 and §5.1-5.6 (shortened); add §6 "Reading the agent: the checker inherits the blind spot"
  (claims 3-4: one figure with state / retrieval / no-orders / no-notes / task across models, one table of cost),
  and §7 "Closing the loop online" (claim 5: O1's null, then R from the shared prefix), and §8 "Why the agent
  misses orders" (Q1-Q3). The arc becomes a diagnosis paper; retitle.
- Appendix: P6b (rich state), P8b, salience, lexical dedup, G4 details.

## Open questions for the user
- Retitle? The current title fits claims 1-2 only. Candidates: "Fresh Eyes: System One Models as Checkers for
  Clinical Agents", or keep "Who Decides?" with a subtitle on checking.
- Co-authors: Yubo Li, Rema Krishnan (set 2026-10-22; affiliation line to confirm).
- Second agent model: Gemma 4 31B-it chosen by the user (2026-10-24), run in FP8 (KV memory; amendment a04045c).
