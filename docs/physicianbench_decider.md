# PhysicianBench with a System One decider in the leaderboard loop (protocol A)

*2026-10-18. User decisions:*
- *local models only (Qwen3.8-27B as the LLM and judge);*
- *a System One model is the decision maker inside the leaderboard agent loop: the LLM proposes, Clef-27B decides;*
- *no Laya, no JEV.*

## 1. Question

**What changes when the decision at each step of a clinical agent is made by a System One model instead of by the LLM?** Everything else is held fixed at the PhysicianBench leaderboard setting: the instruction, the tools, the loop and the graders.

## 2. Leaderboard harness (unchanged)

These parts are kept exactly as upstream (`third_party/physicianbench`, vendored at `c7efa8f`, local edits marked `S1CLIN PATCH`):

- `MiniAgent`, with its minimal system prompt and `instruction.md` as the user turn;
- the 14 FHIR and file tools;
- tool output capped at 10k characters;
- the loop safeguards;
- 100 steps;
- provider-default sampling, with reasoning effort "high";
- 3 independent runs;
- graders: `pytest tests/test_outputs.py`. A task passes only if every checkpoint passes.

**Differences forced by local-only execution** (they apply to every arm):

| Leaderboard | Here |
|---|---|
| API models (GPT-5.5 … Kimi-K2.6) | Qwen3.8-27B on vLLM (bf16, one H100, 128k context, `qwen3_coder` tool parser, `qwen3` reasoning parser) |
| `reasoning effort: high` | Qwen thinking on (`enable_thinking=True`). Sampling uses the model's generation config (temp 1.0, top-p 0.95, top-k 20) = "provider default" |
| GPT-5 judge | Qwen3.8-27B judge, thinking off, temperature 0 |
| `max_completion_tokens=32000` | 16384, so prompt + completion stays within 128k |
| Docker FHIR | Apptainer, a fresh copy of the pristine H2 database for every task run (`scripts/pb_fhir_up.sh`) |

**Consequence:** absolute pass rates are **not comparable to the published leaderboard**. They are comparable only across our arms, which share the same LLM, judge and environment.

## 3. Arms

| Arm | Each step |
|---|---|
| `llm` | Qwen samples one turn and it is executed. This is the leaderboard procedure. |
| `clef` | Qwen samples K = 5 turns in one request (n = 5). Duplicates are removed. **Clef-27B chooses** one, and it is executed. |
| `llm_select` | The same K samples. **Qwen chooses**, seeing its full history and the lettered candidates, with thinking on. |

- **No random-choice arm is needed.** Choosing uniformly among K i.i.d. samples has the same distribution as `llm`.
- **A candidate** is a complete assistant turn:
  - a batch of tool calls with all arguments, rendered `name(arg=value, …)`; a `write_file` candidate therefore contains the full note;
  - or `FINISH with final answer: …`.
  
  Any free text the LLM writes alongside tool calls is not shown to Clef, so Clef judges actions, not rationales.
- **If all K samples are identical**, the step is forced: no decision is made, and the step is logged as forced.
- **If Clef fails** (the server errors), the first sample is executed. That is an `llm`-arm draw, and it is logged.

## 4. Clef input (`scripts/clef_server.py`)

- **Request:** one choice question.
  - Instruction: "…choose the candidate that best advances completing the task correctly, completely and safely."
  - Options are named by letters assigned per (task, run, step) with a seeded shuffle. Clef presents choice options sorted by name, so the shuffle sets the presentation order.
  - The readout is the native joint-head logits with a full-precision softmax (the scorer from the earlier JEV studies).
- **State:** "TASK GIVEN TO THE AGENT" (the full instruction), then each step's ACTION and RESULTs exactly as the LLM saw them.
- **Fitting into 32,768 tokens** is deterministic and uses Clef's own tokenizer:
  1. The instruction and every action are always kept.
  2. Tool results are shortened oldest-first to 800 characters.
  3. Then they are omitted oldest-first.
  4. Each candidate rendering is capped at 12k characters.

  The amount shortened or omitted is logged per decision.
- **Known asymmetry:** the LLM (and `llm_select`) reads the full history, up to 128k tokens. Clef reads a fitted view. We report decision quality against the fraction of history shortened.

## 5. Measurements

- **Native outcomes:** for each arm, report:
  - task pass@1 (mean over 3 runs), pass@3 and pass^3;
  - checkpoint pass rate by type (Data Retrieval, Clinical Reasoning, Action Execution, Documentation);
  - #turns.
- **Decision-level** (from `decisions.jsonl`):
  - the share of steps that are real decisions (more than one distinct candidate);
  - how often Clef picks the majority sample;
  - Clef confidence (max p) and entropy;
  - for `llm_select`, the parse rate;
  - how often the deciders agree.
- **Calibration:** link each decision step to graded outcomes:
  - Action Execution checkpoints are code-graded on the FHIR end state, so a chosen order (or not ordering) is attributable to a step;
  - Data Retrieval checkpoints test whether a tool was called.
  
  Reliability of Clef's chosen-option probability is measured against "this step's chosen action contributed to a passing checkpoint". Task-level: Clef's mean confidence against task success.
- **Cost:** generated tokens per task, and decider wall time.

## 6. Plan

1. **Smoke:** 3 tasks × 1 run × 3 arms. Check the trajectory format, the graders, the decision logs and timing.
2. **Pilot:** 20 tasks (stratified by specialty) × 1 run × 3 arms. Make a go/no-go on runtime and on whether decisions are frequent enough.
3. **Full:** 100 tasks × 3 runs × 3 arms.

## 7. Compliance

- **Local-only guard.** `src/s1clin/guard_site/sitecustomize.py` loads the localhost-only socket guard in the agent and in the grader subprocess. Every LLM base URL is asserted to be loopback, and the FHIR server listens on 127.0.0.1 only.
- **Where data lives.** Job directories (trajectories with patient data, notes, decision logs) are under `/ocean/projects/cis260032p/yliy/data/physicianbench/jobs/` with mode 700, outside git. Only aggregate metrics are committed.
