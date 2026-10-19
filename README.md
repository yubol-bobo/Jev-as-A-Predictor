# Who Decides? System One models in the decision seat of clinical agents

How does a System One decision model compare with an LLM when it makes the decisions inside real clinical agent
loops? A System One model reads a state and a set of options, and returns a probability for each in one forward
pass, without generating text.

System One 决策模型在真实临床智能体任务中做决策，与 LLM 相比表现如何？

- **Paper draft:** [paper/main.pdf](paper/main.pdf) (source `paper/main.tex`, figures `paper/figures/`)
- **Design and running log:** [docs/full_experiment.md](docs/full_experiment.md)
- **Environments:** [PhysicianBench](https://github.com/HealthRex/PhysicianBench) (100 consultation tasks against a
  FHIR record of real de-identified patients) and [ClinEnv](https://github.com/ylin766/ClinEnv) (staged MIMIC-IV
  admissions). Everything runs on local models only.

## Setup

Everything except the decision maker is held fixed: the benchmarks' own agent loops, tools and graders, the
hardware, and the backbone. The LLM is Qwen3.8-27B, and Clef-27B is post-trained from it.

| Arm | Who decides |
|---|---|
| LLM | Qwen3.8-27B, one sample per step (leaderboard setting) |
| S1-Select | Qwen proposes K = 5 complete next turns, and a System One model picks one |
| S1-Tool | a System One model picks the next tool (or finish); Qwen writes the call |
| LLM-Tool | control for S1-Tool: the same one-tool-per-turn restriction, but Qwen picks the tool |

## Findings so far (PhysicianBench, run 1, Clef-27B as the System One model)

- **Selecting among proposals, it matches the LLM:** 80.5% vs 81.3% of graded checkpoints, with no significant
  difference in task success (38.1% vs 42.3%, p = 0.52), and with fewer tool calls.
- **It acts when action is required:** when a required order is among the proposals, it places it in 100% of
  those steps; the LLM's own sample does so in 38%.
- **It should not steer alone:** choosing the tool, it writes the note again and again instead of gathering data,
  and checkpoint success falls to 43.7%. The same restriction with the LLM choosing costs only 2.2 points.
- **Selection is bounded by the proposer:** on MIMIC-IV, 80% of the LLM's five-sample candidate sets hold a single
  clinical decision.

## In progress

- **Other System One models in Clef's seat:** pplx-decider-v1.2, d3 v3.1.0, linne-oak-1-0, amos-1-large and
  decider-31b, run in both System One arms ([docs §7](docs/full_experiment.md)).
- LLM-Tool control to 100 tasks; PhysicianBench runs 2–3 (pass@3); the ClinEnv online arms.

## Layout

| Path | What |
|---|---|
| `src/s1clin/` | arms, deciders, the job runners for both benchmarks, item claims and replica routing |
| `scripts/` | serving (`vllm_qwen_up.sh`, `clef_server.py`, `s1_up.sh`, `s1_proxy.py`), run drivers, analysis (`pb_analyze.py`, `pb_offline_orders.py`, `offline_decisions.py`), tables and figures (`make_tables.py`, `build_paper.sh`) |
| `results/` | aggregate results only |
| `paper/` | the draft, generated tables and numbers, vector figures |
| `third_party/physicianbench/` | vendored PhysicianBench (Apache-2.0), local changes marked `S1CLIN PATCH` |

**Data policy.** Patient-level data (MIMIC-IV, the PhysicianBench record, trajectories, notes, decision logs) never
enters this repository and is never sent to external services. Only aggregate metrics are committed.
ClinEnv has no upstream license, so its vendored code is not included here (see
`third_party/clinenv/NOT_INCLUDED.md`).

This is the working repository. Experiment progress (aggregates, tables, figures, the paper) is committed and pushed
every 20 minutes by `scripts/progress_commit.sh`. Commits up to 2026-10-19 17:00 were developed in a local
repository and published here as snapshots; each snapshot lists the local commits it covers. The earlier
sequential-predictor study (abandoned) is kept at the tag `archive/seqpred-2026-10-19`.
