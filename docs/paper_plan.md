# Paper plan (2026-10-22, draft for discussion; no paper text changed yet)

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
   - Pending: G7 (can an instruction undo it?), G6-ext (linne, decider31b), G6-LLM.
4. **The S1 advantage is that the remedy is cheap.** Task-only S1 judgments take 0.2-0.6 s and always answer; Qwen
   takes 41-69 s and fails to answer 17-31% within an 8k budget; on common questions task-only S1 (0.66-0.79) matches
   or beats every Qwen input. That is what makes screening hundreds of candidate orders per task affordable.
5. **Honest limit: closing the loop online is not solved.** In-run screening (P7′) and a task-prior checklist (P9)
   did not raise required-order checkpoints (+0.04 / +0.045, n.s.): the in-run pool rarely holds the missed orders,
   and a 3-5 item list from the task prior covers about a third of them; the agent mostly declines to act on it.

## Design rules for practitioners (Discussion)
- Do not put an S1 model in the decision seat for open-ended clinical work; use it at commitment points at most.
- To check an agent, give the checker the task (and the chart), not the agent's decisions, whatever the checker is.
- Use the S1 model's speed for breadth (screen many candidates), and let the LLM generate the candidates.

## Proposed restructure
- Keep §§1-4 and §5.1-5.6 (shortened); add §6 "Reading the agent: the checker inherits the blind spot"
  (claims 3-4: one figure with state / retrieval / no-orders / no-notes / task across models, one table of cost),
  and §7 "Closing the loop online" (claim 5, short, with the negative results).
- Appendix: P6b (rich state), P8b, salience, lexical dedup, G4 details.

## Open questions for the user
- Retitle? The current title fits claims 1-2 only. Candidates: "Fresh Eyes: System One Models as Checkers for
  Clinical Agents", or keep "Who Decides?" with a subtitle on checking.
- Co-author names (`\pending{co-authors}`).
