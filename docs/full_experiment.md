# Full experiment: LLM vs System One as the decision maker in clinical agent loops

*2026-10-19. The user asked for the full runs and left the design to me. The core question is fixed: how does an LLM differ from a System One model in ability and performance when either makes the decisions inside an agent loop? Everything lives under `/ocean/projects/cis260032p` (cis260025p is retired).*

## 1. Question and the one manipulated variable

The two benchmarks are real clinical agent loops:

- **PhysicianBench:** 100 consult tasks on a FHIR EHR. Graded by code and an LLM judge; a task passes only if every checkpoint passes.
- **ClinEnv:** multi-stage inpatient admissions. The agent queries a patient, a nurse, a lab and the history, then submits medications, procedures and diagnoses. These are scored against gold with an ontology-aware hierarchical F1.

Each benchmark runs in its own unchanged harness, with an unchanged loop, tools, prompts, simulators and graders. **The only thing varied is who chooses the next turn:**

| Arm | Who decides each turn |
|---|---|
| `llm` | **The LLM.** Qwen3.8-27B samples one turn and it is executed. This is the benchmark's own procedure. |
| `clef` (main) | **A System One model.** Qwen samples K = 5 complete candidate turns; duplicates are removed; **Clef-27B** reads the state and the candidates and outputs a probability distribution; the argmax is executed. |
| `llm_select` (control) | **The LLM, given the same candidates.** The same K samples; **Qwen** chooses among them by generation, with thinking on. This separates "a System One model decides" from "someone chooses among K samples". |
| `llm_t0` (ClinEnv only) | The paper's protocol: greedy decoding (T = 0), one sample. It links our numbers to the published table. |

**No random-choice arm is needed.** A uniform pick among K i.i.d. samples has the same distribution as `llm`.

**Fixed for every arm:**
- Qwen3.8-27B, bf16, thinking on, the model's own sampling defaults (temp 1.0, top-p 0.95, top-k 20);
- the same vLLM build: 3 identical replicas, decode CUDA graphs, MTP-2 speculative decoding (which leaves the output distribution unchanged);
- local judges and simulators: Qwen with thinking off at T = 0, and bge-large for ICD retrieval;
- one decision procedure, shared by both benchmarks (`src/s1clin/pb_decider.py`): candidate rendering, de-duplication, seeded option letters per (case, run, step), and the Clef input fitted deterministically into 32k tokens.

## 1b. Revised design (2026-10-19, agreed with the user)

The online arms compare whole agent loops, but after a few steps each arm faces different patient states, so a
difference mixes decision quality with error compounding and sampling luck. The revision adds an **offline,
step-paired evaluation** and a **tool-level System One arm**, and drops the online `llm_select` arm (it cost 2.5–3 h
per PhysicianBench task; it is evaluated offline instead).

| Part | What | Question |
|---|---|---|
| online `llm` | the benchmark's own procedure | baseline |
| online `clef` | Clef picks among K = 5 whole turns | System One deciding the whole action, in the loop |
| online `s1_tool` (new) | Clef picks the tool (or FINISH) from the full tool menu; the LLM writes the call, constrained to that tool (ClinEnv: `tool_choice` named; PhysicianBench: `required`, keeping only calls of the chosen tool, else regenerating with the tool named) | System One deciding only the kind of action: strategy and timing |
| online `llm_log` (new) | K = 5 samples per step, the first is executed (the same distribution as `llm`), all K and the message history logged | `llm` behaviour plus the candidate sets |
| offline | on the `llm_log` states and candidate sets: Clef, Clef blind (options only), LLM select (thinking on), majority vote, first sample | same state, same options: who picks right, and calibration |

**Offline items (ClinEnv)** are the submit steps: every candidate is a submission of one type. Each candidate is
scored against the stage gold still open at that step with ClinEnv's own pair scores (`scripts/offline_decisions.py`);
correct = score ≥ 0.5. Three versions of each set:
- `raw`: as the online deciders see them;
- `canon`: action + canonical name (ICD title of the mapped code, ATC-preprocessed drug name), without the agent's
  `reasoning` argument, so surface form carries no cue;
- `inj`: `canon`, and when no candidate is correct one wrong option is replaced by an open gold item rendered the
  same way (a single wrong option gets the gold one added). Every set then has a right answer.

The natural sets (`raw`, `canon`) carry the calibration question, including whether Clef is less confident when all
options are wrong. The injected sets measure discrimination against chance 1/n. **Clef blind** checks for cues in
the options themselves: if it beats chance on `inj` without the patient, the injected set leaks.

**PhysicianBench offline** covers order steps (ServiceRequest / MedicationRequest / Appointment / Communication
creates) scored with each task's code-grader patterns, on natural sets only: there is no gold order to inject.

**First finding from the smoke data.** ClinEnv candidates that are distinct strings are often the *same* decision:
the submit tools take a free-text `reasoning` argument, so five samples of one diagnosis render as five options.
After canonical mapping most submit steps have one or two distinct options. Real choice is rarer than raw counts
suggest, and the online `clef` arm often chooses among wordings of one decision.

**Question wording.** Until 2026-10-19 the ClinEnv `clef` arm sent Clef the PhysicianBench question ("…works in an
electronic health record through FHIR tools…"). Each benchmark now has its own wording, and decision rows carry
`q` (question version); ClinEnv `clef` cases run under version 1 are rerun.

## 2. Data and sizes

| | PhysicianBench | ClinEnv |
|---|---|---|
| Items | all 100 tasks | 392 admissions: those scored by all 7 paper models in the authors' release (899 stages, 2,714 decisions) |
| Runs | 3 per arm (pass@1, pass@3, pass^3, as the leaderboard reports) | 1 per arm, the same as the paper (all runs are single) |
| Arms | llm, clef, llm_select | llm, clef, llm_select, llm_t0 |
| Data handling | the FHIR image from Redivis; per-run copy of the database; localhost only | the authors' prepared cases (the HF release), **handled as MIMIC data**: local, mode 700, never in git, never sent to an external API |

**Checks on the ClinEnv cases.** They match the paper's Table 4 exactly: 9,297 stages and 26,043 decisions; diagnosis 71.6%, medication 21.4%, procedure 6.9%; medication actions start 79.1%, stop 7.5%, adjust 9.5%, switch 3.9%. The release also holds the authors' runs and scores for the 7 paper models on the same cases. We summarise those next to ours as a reference. They are **not paired on the judge**: the authors used GPT-5.4-mini, we use Qwen.

## 3. Measurements

**Native outcomes**
- **PhysicianBench:**
  - task pass@1, pass@3 and pass^3;
  - checkpoint pass rate by type: Data Retrieval, Clinical Reasoning, Action Execution, Documentation;
  - number of turns and tokens.
- **ClinEnv:**
  - stage F1 and F1 by type (diagnosis, medication, procedure), from ClinEnv's Hungarian scorer;
  - information coverage and efficiency;
  - information calls per stage;
  - long-horizon F1 (cases with 3 or more stages).

**Statistics.** The arms are compared paired: on items finished by every arm, with a cluster bootstrap over tasks or admissions.

**Decision level**, from the per-step `decisions.jsonl`:
- **Step types.** Steps are split into *information steps* (ask, order or search) and *decision steps* (any candidate submits or places an order).
- **How often there is a real choice.** This means more than one distinct candidate, both exactly and after merging semantic duplicates.
- **The decider's behaviour:**
  - confidence (max p) and normalised entropy;
  - how often it picks the most-sampled candidate;
  - how often Clef and the LLM agree;
  - decider latency.

**Calibration**, the System One question:
- **ClinEnv:**
  - every candidate at a decision step can be scored offline against the stage gold with the same scorer, so we know which candidates were right;
  - this gives Clef's reliability (probability against correctness), ECE, Brier, and the AUROC of p(candidate) for ranking right against wrong candidates;
  - it also gives oracle best-of-K, decider pick, LLM pick and first-sample accuracy, all on the same candidate sets.
- **PhysicianBench:**
  - order steps are linked to the code-graded Action Execution checkpoints;
  - at task level, mean Clef confidence is compared with task success.

**Cost**
- generated tokens and wall time per item and per arm;
- decider time.

## 4. Order of work and compute

**Measured in the first 8-hour allocation (w004, one Qwen replica per arm on one GPU each).** The replicas were limited by KV-cache memory, not compute:
- One H100 holds the 52 GiB of bf16 weights plus only about 16 GiB, or 231k tokens, of KV cache. That is about 5 long agent contexts.
- The `clef` and `llm_select` replicas, which serve K = 5 candidates per step, ran 5–7 sequences, with about 20 requests waiting and 85% KV use. Each GPU generated only about 200 tok/s.
- Per item, under that load:

  | | `llm` | `clef` | `llm_select` |
  |---|---|---|---|
  | PhysicianBench, per task | 10–30 min | 20–110 min | 2.5–3 h |
  | ClinEnv, per case | 3–8 min | 0.5–2 h | — |

  The `llm_select` ClinEnv cases hit the client's default 600 s timeout and its retries. These were infrastructure failures, with no output, so they are rerun.

**Layout from 2026-10-19 (`scripts/slurm/full.sbatch`, `scripts/run_full.sh`).** Each node runs:
- Qwen3.8-27B with **tensor parallelism 2 on GPUs 0–1** (port 8011);
- one more Qwen3.8-27B on **GPU 2** (port 8013);
- **Clef on GPU 3**.

TP only splits the same bf16 weights over two GPUs, so the model, dtype and sampling are unchanged. It frees about 2.7 times as much KV cache per GPU. Items finished on the w004 layout are kept, and every item's metadata records its replica.

**Routing.** Every arm uses both replicas. Until 09:00 on 2026-10-19 each item was routed by a weighted hash of its name, at 5 : 1 by KV capacity; that left the small replica idle at times. Since then each item goes, when it starts, to the replica with the lowest (running + waiting requests) / capacity (`src/s1clin/claim.py`). A whole item stays on one replica, and the replicas are the same model and weights, so routing does not depend on the arm. Compute stays on the one interactive node (4 × H100); no other SLURM jobs.

**Client timeout.** It is 3600 s for the model under test.

**Scheduling.** Six streams run at once: PhysicianBench and ClinEnv for each arm.
- Each stream does its own runs in priority order, **run 1 of every arm first** (the main comparison), then runs 2–3 and ClinEnv T = 0.
- It then joins the slower arms, mostly `llm_select`.
- Items are claimed atomically before they run, so the streams on a node, and several SLURM nodes at once, share the work without overlap.
- Everything can be resumed. The chained 8-hour jobs (account cis260032p) pick up where the last one stopped.

## 5. Progress, git and data

- **Progress commits.** `scripts/progress_commit.sh` commits only the aggregate summaries (`results/pb_summary.json`, `results/clinenv_summary.json`), every 20 minutes.
- **Where data lives.** Trajectories, dialogues, notes, decision logs and cases stay under `/ocean/projects/cis260032p/yliy/data/`.
- **Commit guard.** The pre-commit hook refuses data-like files, MRNs, FHIR JSON, MIMIC ids and anything over 2 MB.
- **No push.** There is no git remote.

## 6. Status and resume (2026-10-19, interactive node w002 until 14:55)

The user asked for PhysicianBench first and complete. State at 14:30:
- **Done, PhysicianBench run 1:** `llm` 100/100, `clef_k5` 100/100, `s1_tool` 97/100 (3 tasks were in flight when the
  node expired and are picked up again on resume), `llm_tool` 36 (stopped at 13:17; 35 tasks paired with `s1_tool`).
  The paper is built from this state. Serving about 22 tasks at a time kept the prefix cache near 80% hit rate. More concurrency thrashed the KV / prefix cache:
  at 48 tasks, 80 requests queued and generation fell to 160–900 tok/s.
- **Paused, ClinEnv:** all arms. The remaining items are claimed by holder pid 100708 (`scripts/hold_items.py`). The
  ClinEnv streams were stopped; cases in flight were killed and rerun from scratch on resume. The `clef` cases run
  under question version 1 were moved to `data/clinenv/archive_q1/` and are rerun.
- **Paused, PhysicianBench `llm_log`:** limited to the 20 pilot tasks. Non-pilot tasks are held by holder pid 26000;
  the pilot driver was stopped at 10:08 to give the K=5 decode capacity to the three main arms.
- **Paused, PhysicianBench runs 2–3** (10:53) of `llm`, `clef`, `s1_tool`: held by pids 1803, 1836, 1845, 1857,
  1861, 1864 (`logs/ce_hold.txt`), so the capacity goes to `clef` run 1 and the new control arm `llm_tool` run 1.
- **Added, `llm_tool`** (10:45): the control for `s1_tool`. It applies the same one-tool-per-turn restriction, but the
  LLM chooses the tool: its own turn is kept to the calls of its first tool.
- **Resume:** finish `s1_tool` run 1 and extend `llm_tool` to 100 (`bash scripts/run_pb_full.sh`). Then
  `.venv/bin/python scripts/hold_items.py release --pid 100708` (and the other holder pids), then
  `bash scripts/run_full.sh s_ce_llm s_ce_clef s_ce_log s_ce_tool s_pb_log`. A new allocation has new pids, so the
  holds are stale anyway: release first, so the claim files are removed rather than left as stale claims.

## 7. Other System One deciders in Clef's seat (2026-10-19, w008)

The user asked to swap Clef for five other System One models: perplexity-ai/pplx-decider-v1.2, vllm-sr/d3 v3.1.0,
PNPM244/linne-oak-1-0, Mapika/decider-31b, and mukaddamzaid/amos-1-large. Only the decider changes. The proposer,
the K = 5 candidates, the question wording, the state rendering, the 32k fitting rule, the seeds and the graders
stay the same.
- **Serving:** `scripts/s1_up.sh NAME GPU` starts the authors' own server, revisions pinned. pplx-decider-v1.2 ships
  only a library, so `scripts/s1_servers/pplx12_server.py` wraps its reference path. In front of it,
  `scripts/s1_proxy.py` gives Clef's `/decide` API. Rendering and fitting are shared with `clef_server.py` through
  `scripts/clef_render.py`; tokens are counted with the decider's own tokenizer. Everything runs locally with
  `HF_HUB_OFFLINE=1`, and no server logs requests (amos: no `--log-dir`, no `AMOS_THINK_URL`).
- **decider-31b** is NVFP4. H100 has no native FP4, so vLLM runs its weight-only FP4 path; the authors used
  Blackwell.
- **Runs:** `scripts/run_deciders.sh 3` runs one decider at a time on GPU 3: S1-Select (`--arm clef --decider D`,
  tag `clef_k5__D`) and S1-Tool (`--arm s1_tool --decider D`, tag `s1_tool__D`), run 1, all 100 tasks. Order:
  pplx12, d3, linne, amos, decider31b.
- **Analysis:** `pb_analyze.py` adds `deciders.<D>`, which compares each new arm with the LLM, with LLM-Tool, and
  with Clef in the same arm, on the tasks each pair has in common.
- **Capacity:** about 18 decider-arm tasks share the Qwen replicas with the llm_tool control (4 workers,
  `scripts/restart_llm_tool.sh`). LLM runs 2–3 are paused.
