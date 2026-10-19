# PhysicianBench study: harness, tasks, checkpoints, and System One protocols

*2026-10-18. Read-and-design only. Nothing was run, no LLM API was called, and no patient data was touched (the FHIR image is not downloaded yet).*

- **Code:** `research/external/PhysicianBench` (HealthRex/PhysicianBench @ `c7efa8f`, Apache-2.0). Below, `file:line` refers to this repo.
- **Paper:** arXiv 2605.02240v1, 30 pages, read in full including the appendices. PDF and text are in `research/external/PhysicianBench_paper/physicianbench.{pdf,txt}`.
- **Per-checkpoint classification** of all 670 checkpoints: see §3. The machine-readable file is `research/S1-Clinical-Agents//ocean/projects/cis260032p/yliy/data/physicianbench/derived/physicianbench_checkpoints.jsonl (kept out of git)`.

---

## 1. Harness

### 1.1 How a task runs

`scripts/run_task.py` handles one task. `scripts/run_batch_task.sh` loops over tasks *sequentially*, using one port (`:168-221`).

1. **Start the FHIR server.** `start_fhir_container()` runs `docker run -d --name fhir-bench-<uuid8> -p <port>:8080 fhir-full:v1` (`run_task.py:69-91`, call at `:74-77`). It then polls `GET http://localhost:<port>/fhir/metadata` every 3 s, for up to 180 s (`:55-66`).
2. **Data import.** None happens: the patient data is baked into the image's H2 database (`:301`).
3. **Run the agent** in-process (`run_agent`, `:144-199`).
   - It reads `instruction.md` and rewrites `/workspace/` to the job workspace. It appends a "Working Directory" block.
   - It sets `FHIR_BASE_URL=<url>/` (`:160`). Builds `LLMClient(model_id)` + `ToolRegistry` + `MiniAgent`.
   - The workspace is `jobs/<batch>/<task>/workspace/output/`. `input_files/` is symlinked if present, but no v1 task has one: every task dir holds only `instruction.md`, `task.toml` and `tests/test_outputs.py`.
4. **Evaluate.** `scripts/run_eval.py` runs `pytest tasks/v1/<task>/tests/test_outputs.py -v -rA` with `cwd` set to the job workspace and the env vars `FHIR_BASE_URL` and `JOB_DIR` (`run_eval.py:31-58`). Output goes to `logs/verifier/pytest_output.txt` (`run_task.py:202-230`).
5. **Tear down.** `docker rm -f <name>` (`run_task.py:94-98`). Then `metadata.json` is written (`:328-341`).
6. **Task success** means pytest exits 0, i.e. **every** checkpoint passes. An LLM-judge PARTIAL counts as a fail (`utils/eval_helpers.py:672`).

### 1.2 Agent loop (`agent/mini_agent.py`, about 300 lines)

- **Messages:** the system prompt (`agent/prompts.py:3-14`, minimal) plus the user instruction (`mini_agent.py:66-69`). There is no task-specific scaffolding.
- **Each step** (`:80-291`):
  - It calls `client.chat(messages, tools=14 schemas, parallel_tool_calls=True, reasoning_effort)`.
  - If there are **no tool calls**, the content is the final answer and the loop stops (`:153-157`).
  - Otherwise it runs every tool call in order and appends `role=tool` messages.
  - Tool output sent to the LLM is **truncated to 10,000 chars**, with a hint to use filters (`:20, :189-196`). The full output goes to `logs/agent/trajectory.log` (JSONL; `agent/trajectory.py`).
- **Safeguards** (`:46-50, :209-285`), each aborting the run:
  - the same error 5 times in a row;
  - the same (tool, args, output) 5 times in a row;
  - the same batch 5 times within 10 steps;
  - no new (tool, args) pair for 15 steps.
- **Step limits:** `run_task.py` defaults to `max_steps` 200 (`:245`); the batch script uses 100 (`run_batch_task.sh:32`). The paper uses 100.
- **The graders read the trajectory.** Data-retrieval checkpoints check which tools appear in the trajectory (e.g. `tasks/v1/aortic_aneurysm_cad/tests/test_outputs.py:64-98`). **Any replacement agent must therefore keep the trajectory format and the tool names.**

### 1.3 The 14 tools (`agent/tool_registry.py:64-424`, implementations in `tools/fhir_api_functions.py`)

| # | Tool | HTTP | Notes |
|---|---|---|---|
| 1 | `fhir_condition_search_problems` | GET Condition | count 10, 3 pages by default |
| 2 | `fhir_observation_search_labs` | GET Observation | LOINC `code`, `date` |
| 3 | `fhir_observation_search_vitals` | GET Observation | |
| 4 | `fhir_patient_search_demographics` | GET Patient | |
| 5 | `fhir_procedure_search_orders` | GET Procedure | |
| 6 | `fhir_medication_request_search_orders` | GET MedicationRequest | |
| 7 | `fhir_document_reference_search_clinical_notes` | GET DocumentReference | **base64 note text is decoded inline** (`fhir_api_functions.py:40-64, :917`). The description mentions "Binary.Read", but no such tool exists. |
| 8 | `fhir_service_request_search` | GET ServiceRequest | |
| 9 | `fhir_observation_search_social_history` | GET Observation (category = social-history) | |
| 10 | `fhir_medication_request_create` | POST MedicationRequest | `authoredOn = now()` (UTC wall clock, `:1590-1592`) |
| 11 | `fhir_communication_create_message` | POST Communication | |
| 12 | `fhir_service_request_create` | POST ServiceRequest | labs, imaging, referrals |
| 13 | `fhir_appointment_create` | POST Appointment | |
| 14 | `write_file` | local FS | `tools/file_tools.py:11-26`, no path sandboxing |

- **Connection:** every function re-reads `FHIR_BASE_URL` from the environment and uses plain `requests`, with no auth, 2 retries and a 20 s timeout (`fhir_api_functions.py:17, :108, :130-160`).
- **Date filter:** the graders find agent-created resources with a date filter `authored(on) ge <task date>` (`eval_helpers.py:223-240`). This works because `authoredOn` is the real clock time (2026), which is later than every task date (2021–2024).

### 1.4 The LLM client: can it point at a local OpenAI-compatible endpoint? **Yes, with no code change.**

- `agent/llm_client.py:27-43` picks the first backend whose key is set, in the order OpenRouter → Anthropic → OpenAI. The base URL comes from `OPENAI_BASE_URL` if it is set (`:40`).
- Set `OPENAI_API_KEY=EMPTY` and `OPENAI_BASE_URL=http://127.0.0.1:8011/v1`. Also set `--model <vLLM served-model-name>`, and make sure `OPENROUTER_API_KEY` and `ANTHROPIC_API_KEY` are **unset** and no `.env` sets them (`load_dotenv()` at `llm_client.py:17`, `run_task.py:41`).
- **Things to fix or set:**
  - **`max_completion_tokens=32000` on every call** (`llm_client.py:101`). vLLM rejects requests where prompt + max_tokens exceeds `max_model_len`. Contexts grow to about 100k tokens: about 30–40 tool results of up to 10k chars each. Either serve with `--max-model-len` of at least 131072, or patch this to about 8k.
  - **`reasoning_effort`** is sent as `extra_body={"reasoning":{"effort":...}}` (`:119-120`). vLLM ignores it, but `run_batch_task.sh:30` defaults to `high`. For Qwen, add `extra_body["chat_template_kwargs"]={"enable_thinking": bool}` instead (a 3-line patch).
  - **vLLM serve flags:** `--enable-auto-tool-choice --tool-call-parser hermes` (Qwen3 family) and `--reasoning-parser qwen3`. Without the tool parser, `tool_calls` comes back empty and the agent stops at step 1.
  - **Our compliance rule (design §7):** add an assertion that the base URL host is `127.0.0.1`/`localhost`. Also guard `run_task.py:105-121`, which calls `openrouter.ai/api/v1/credits` whenever `OPENROUTER_API_KEY` is set.

### 1.5 Every `docker` usage

| Location | Command | Replace with |
|---|---|---|
| `scripts/run_task.py:74-77` | `docker run -d --name fhir-bench-XXXX -p <port>:8080 <image>` | `apptainer instance start …` (§1.6) |
| `scripts/run_task.py:97` | `docker rm -f <name>` | `apptainer instance stop <name>` |
| `README.md:51` | `gunzip -c physicianbench-fhir-v1.tar.gz \| docker load` | `apptainer build` (§1.6) |

- `run_batch_task.sh` itself makes no docker calls. It only passes `--fhir-image` and `--port` through (`:34-35, :196-197`).
- The website text mentions "fresh Docker container" (`website/src/components/Methodology.tsx:12`). This is documentation only.
- The paper says "each task ships as a self-contained Docker image" (§5.1, App. E). The code uses **one image, `fhir-full:v1`, for all 100 tasks** (`run_task.py:46`). The 100 tasks have 100 distinct patient IDs, so presumably all 100 patients are in one H2 database. This must be confirmed when the image arrives.

### 1.6 Running the FHIR server under Apptainer (no Docker/root)

What we know:
- The server is HAPI FHIR JPA (starter) on R4, backed by a **persistent H2 file database inside the image** (paper §3.2, §5.1). There is no external DB and no Elasticsearch. Lucene index files are possible if HAPI fulltext indexing is on.
- It listens on 8080 at `/fhir`.
- The site config on this machine (`/etc/apptainer/apptainer.conf`) has:
  - `sessiondir max size = 64` → `--writable-tmpfs` gives only **64 MB**. That is too small if the H2 file is copied up on first write: overlayfs copies the *whole* file.
  - `enable overlay = yes` and `allow setuid-mount extfs = yes` → `--overlay x.img` works.
  - Apptainer is version 1.5.4.

Recipe (to check against the real image):

```bash
# 0) inspect the image config without docker (Entrypoint/Cmd/Env/WorkingDir/User/Volumes)
mkdir -p $D/img && cd $D/img && gunzip -k physicianbench-fhir-v1.tar.gz
tar -xOf physicianbench-fhir-v1.tar manifest.json            # -> Config: <sha>.json
tar -xOf physicianbench-fhir-v1.tar <sha>.json | python -m json.tool
export APPTAINER_TMPDIR=/ocean/projects/cis260032p/yliy/apptainer_tmp APPTAINER_CACHEDIR=/ocean/projects/cis260032p/yliy/apptainer_cache
# 1) sandbox (lets us find the H2 path with host tools; hapiproject/hapi images are distroless => no shell inside)
apptainer build --sandbox fhir_sbx/ docker-archive://physicianbench-fhir-v1.tar
find fhir_sbx -name '*.mv.db' -o -name 'application*.y*ml' -o -name lucenefiles   # expect /app/target/database/h2.mv.db (starter default)
# 2) immutable SIF for runs
apptainer build physicianbench-fhir-v1.sif docker-archive://physicianbench-fhir-v1.tar
# 3) per run: fresh copy of the pristine DB dir on node-local disk, bind it over the image path, pick a port
RUN=$LOCAL/fhir_$PORT; mkdir -p $RUN/tmp; cp -a fhir_sbx/app/target/database $RUN/db
apptainer instance start --containall --no-home \
   --bind $RUN/db:/app/target/database --bind $RUN/tmp:/tmp --writable-tmpfs \
   --env SERVER_PORT=$PORT --env JAVA_TOOL_OPTIONS="-Xmx8g -Djava.io.tmpdir=/tmp" \
   physicianbench-fhir-v1.sif fhir_$PORT
# (instance start runs the image's startscript = Docker ENTRYPOINT/CMD; if not, use `apptainer run` in background)
curl -s localhost:$PORT/fhir/metadata | head -c 200
apptainer instance stop fhir_$PORT; rm -rf $RUN
```

Points to check:
- **Networking.** Apptainer shares the host network, and `-p` mapping needs root/CNI. So each server binds its port **directly on the node**. Use `SERVER_PORT` (Spring Boot relaxed binding) or append `--server.port=$PORT` to the entrypoint arguments. Choose a distinct port per instance and per job; SLURM jobs on a shared node may collide.
- **Writable paths:**
  - the H2 directory (bind a fresh copy);
  - the Lucene directory if present (bind it too);
  - `/tmp` (Tomcat work dir);
  - any log dir. `--writable-tmpfs`, at 64 MB, covers small stray writes.
- **Alternative to copying the DB each time:** `apptainer overlay create --size 8192 --sparse ov_$i.img`, then `--overlay ov_$i.img`. This works, but the first write copies the whole H2 file into the overlay. A plain `cp -a` of the DB directory is simpler and just as fast.
- **User.** The distroless image declares `USER 65532`. Apptainer runs as us, which is fine, but the sandbox/DB copy must be owned by us (it is, when we build it).
- **Startup.** The JVM plus H2 open takes about 30–120 s, so the 180 s timeout in `wait_for_fhir` is fine.

Patches needed:
- `run_task.py`: add `--fhir-backend {docker,apptainer,external}`.
  - `apptainer` wraps the commands above (start/stop functions at `:69-98`).
  - `external` takes `--fhir-url` and does nothing at start/stop. This is useful for offline protocol A and for a long-lived shared server.
- `run_batch_task.sh`:
  - remove the interactive `read -p "Proceed?"` (`:155-159`), which blocks SLURM;
  - replace `uv run python` (`:117, :212`) with our env's python;
  - add `--workers N` with port = base + worker id;
  - add `--fhir-backend` passthrough.
- `scripts/score_jobs.py:80` only scores directories whose name starts with `task_`. The current runner names job dirs after the bare task name (`run_batch_task.sh:209`), so **`score_jobs.py` finds 0 tasks on new batches**. Fix: drop the prefix check.
- Judge locality: §2.3.

---

## 2. Tasks and checkpoints

### 2.1 Structure of `tasks/v1/`
- **100 task directories.** Each holds `instruction.md`, `task.toml` (`[metadata] tags=[specialty, task type]`) and `tests/test_outputs.py`. There are no input files and no reference solutions; the paper's "solution summary" is not released.
- **`instruction.md`** averages about 210 words. It covers:
  - the context: current date/time, role, Practitioner ID, patient MRN, trigger event;
  - numbered task steps;
  - deliverables, usually one text file under `/workspace/output/`.
- **Patient ID and time.** `PATIENT_ID = "MRN…"` and `TASK_TIMESTAMP` are hard-coded at the top of each test file (e.g. `aortic_aneurysm_cad/tests/test_outputs.py:32-33`). There are **100 distinct patients, one per task**.
- **Checkpoints per task:** 5–9 (5: 8 tasks; 6: 35; 7: 38; 8: 17; 9: 2). The total is **670**.
- **Taxonomy** (`scripts/checkpoint_taxonomy_v1.json`, paper §3.3):

  | Type | Count |
  |---|---|
  | Clinical Reasoning | 356 |
  | Action Execution | 112 |
  | Data Retrieval | 103 |
  | Documentation | 99 |

  The test docstrings label determinism as deterministic 299, semi 164 and non-deterministic 203. Paper App. A has the task-type taxonomy, and `scripts/task_taxonomy_v1.json` maps tasks to specialty groups and types.

### 2.2 How each checkpoint is graded (from parsing every `test_checkpoint_*` body)

| Grader (helpers called) | n | Typical checkpoint |
|---|---|---|
| LLM judge only (`llm_judge` / `llm_extract*` on the output note) | 452 | 353 Clinical Reasoning + 99 Documentation |
| FHIR end state only (`validate_service_order(s)` / `validate_medication_order`) | 105 | Action Execution |
| Trajectory only (tool names called) | 53 | Data Retrieval |
| Trajectory + LLM judge (stated values) | 44 | Data Retrieval (43) |
| FHIR + LLM judge | 9 | 7 Action Execution with a reasoning part, 1 "no ENT referral", 1 Data Retrieval |
| Trajectory + direct FHIR value checks | 6 | Data Retrieval: `get_lab_value(LOINC)` range asserts (e.g. `iron_deficiency_anemia/tests/test_outputs.py:93-115`) |
| Other | 1 | `adc_pulmonary_toxicity` cp2 |

- **Code graders** (`eval_helpers.py:379-608`) match regexes over `code.text`/`coding.display` or code regexes (CPT/LOINC/RxNorm). They check status ∈ {active, completed} and intent order/plan, the dose within ±10% or a range, and frequency patterns. They only look at resources with authored date ≥ the task date.
- **Orders that are not required are never penalised.** Only 2 checks assert that something is absent (`oral_contraceptive_switch/...:399`, `pulsatile_tinnitus_workup` cp6 via the judge). So **the code graders give positive gold only**; verified negatives must come from rubric text.
- **Ground truth lives in the test files.** Rubric strings give "Ground truth from EHR", "Clinical question", "Expected answer" and PASS/PARTIAL/FAIL criteria, as in `aortic_aneurysm_cad/tests/test_outputs.py:107-139, 197-231`.
- `get_lab_value("<LOINC>")` appears 71 times in 26 test files. These are ready-made deterministic fact probes, with gold computed from FHIR.

### 2.3 The LLM judge must become local
- **Where it is set:** `eval_helpers.py:59-76`. The backend is OpenRouter (`openai/gpt-5`) or OpenAI (`gpt-5`), chosen by `LLM_JUDGE_BACKEND`, with the model from `LLM_JUDGE_MODEL`. The call (`:616-626`) uses temperature 0 and `max_completion_tokens=4000`. The prompt is at `:636-649`, and PASS is required.
- **To make it local, set** `LLM_JUDGE_BACKEND=openai`, `OPENAI_API_KEY=EMPTY`, `OPENAI_BASE_URL=http://127.0.0.1:8011/v1` and `LLM_JUDGE_MODEL=<served name>`. The `openai` SDK reads `OPENAI_BASE_URL` itself, so no code change is needed.
- **Disable Qwen thinking for the judge,** or raise the 4000-token budget. If the reasoning eats the budget, `content` is `None` and the result is "LLM returned empty response" = FAIL (`:654-655`). Disabling thinking means adding `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` in `call_llm`.
- **Caveat for reporting:** a Qwen-27B judge is not GPT-5. Native PhysicianBench pass rates we report are therefore "local-judge" numbers and not comparable to the leaderboard. Our System One metrics should not depend on the judge (§4).

---

## 3. Where the discrete decisions with deterministic gold are

### 3.1 How the labels were made

- **Method:** all 670 checkpoints were labelled by reading every `test_checkpoint_*` body, its rubric and context, and the matching `instruction.md`. The work was split into four batches of 25 tasks, validated so that each test function appears exactly once.
- **Output:** one row per checkpoint in `/ocean/projects/cis260032p/yliy/data/physicianbench/derived/physicianbench_checkpoints.jsonl (kept out of git)`, with these fields:
  - the label, the grader, and the question;
  - 2–6 options and the gold;
  - a `negative` flag;
  - the rubric's ground-truth `facts`;
  - a `leak_risk` note.
- **Status:** this is a **first-pass curation and must be reviewed** before it becomes a frozen item set. In particular, the distractor options were written by us and are not physician-verified.
- **Labels:**
  - **A_action:** code-graded order. A discrete decision: order X or not, which drug or test, dose range.
  - **A_judgment:** an LLM-judged reasoning checkpoint whose rubric commits to **one** categorical conclusion with a small option set.
  - **B_lookup:** the gold is a fact in the EHR.
  - **B_compute:** the gold is a deterministic function of EHR facts (score, threshold, criteria count).
  - **C_free:** multi-element free text, documentation or counselling, or a rubric that accepts opposite conclusions.

### 3.2 Counts

| Taxonomy \ label | A_action | A_judgment | B_lookup | B_compute | C_free | Total |
|---|---|---|---|---|---|---|
| Action Execution | **111** | – | 1 | – | – | 112 |
| Clinical Reasoning | 1 | **209** | 16 | **45** | 85 | 356 |
| Data Retrieval | – | – | **103** | – | – | 103 |
| Documentation | – | – | – | – | 99 | 99 |
| **Total** | **112** | **209** | **120** | **45** | **184** | **670** |

Mapped onto the requested classes:
- **(a) Discrete decision with an enumerable option set: 321 (48%).**
  - 112 are code-graded on the FHIR end state: 83 ServiceRequest (labs, imaging, referrals) and 29 MedicationRequest.
  - 209 are LLM-judged but have a single categorical gold.
  - Every one of the 100 tasks has at least 1; the median is 3 per task (distribution 1:6, 2:21, 3:38, 4:19, 5:13, 6:3).
- **(b) Retrieval / reading facts: 165 checkpoints** (120 lookup, 45 compute).
  - The rubrics' ground-truth blocks list **1,190 individual facts**, a median of 11 per task (min 2, max 29). 789 of them come from the 103 Data Retrieval checkpoints.
  - These are the natural factual probes. Examples:
    - "LDL 79 mg/dL (prior 153)", "aortic root 5.1 cm echo / 5.2 cm CTA", "rosuvastatin 20 mg daily";
    - the 71 `get_lab_value(LOINC)` assertions, whose gold can be recomputed from FHIR.
- **(c) Free text / LLM-only: 184** (99 Documentation + 85 Clinical Reasoning).

### 3.3 Properties that matter for System One items

1. **Gold is overwhelmingly positive.**
   - Only **34/321** A-items have a "do not / not indicated" gold: 33 A_judgment and 1 A_action. Examples: cancel fungal blood culture; no CMV treatment; defer rate control to EP; no CBI; withhold iron; avoid deferiprone; no ENT referral yet.
   - The code graders never penalise extra orders (§2.2). So "order X? = no" negatives must come from rubric text, or from protocol C below.
2. **The instruction often gives the answer away.** For **176/321** A-items, the instruction names the action:
   - "Refer to endocrinology";
   - "Order symptomatic therapy for cough relief";
   - "place a vascular surgery referral";
   - 89 instructions contain "determine/assess whether …" phrasing.

   For **56/321**, the rubric attributes the gold to a specialist recommendation that is probably **in an EHR note** (e-consult replies, consult notes). Only **102/321** carry no leak flag. This is the main validity risk for decision items. See the "neutralised state" requirement in §4.
3. **Option-format cues.** The gold, copied from the rubric, tends to be longer and more specific than hand-written distractors: "Increase by 400–800 mg/day, preferably calcium citrate" vs "No change". Before freezing, options must be rewritten to matched length and specificity, or turned into yes/no per option.
4. **Multi-answer golds.** Some code graders accept a set of answers (statin: atorvastatin 40–80 **or** rosuvastatin 20–40; migraine preventive: 4 drugs; LMWH: 3 drugs). For a choice question, either all accepted members go in as gold (a set-valued gold), or only one accepted member appears among the options.
5. **Checkpoints we dropped from (a).** When a rubric accepts *opposite* conclusions (ARB continue vs hold; start vs withhold anticoagulation; biopsy for vs against), we labelled it C_free. Four rows have imperfect gold and are noted in the JSONL (`oligomenorrhea_pcos/cp2`, `osteoarthritis_nsaid_intolerant/cp5,cp6`, `post_gi_bleed_followup/cp3`).

### 3.4 Examples

| Class | Task / cp | Question | Options | Gold |
|---|---|---|---|---|
| A_action | `aortic_aneurysm_cad/cp4` | Coronary evaluation order? | coronary CTA / invasive angio / stress test / none | coronary CTA (CPT 75574) |
| A_action | `pulsatile_tinnitus_workup/cp4` | Which imaging? | CTA head & neck / MRI IAC / carotid US / none | CTA head & neck |
| A_action | `vte_prophylaxis_substitution/cp4` | Substitute for certoparin? | enoxaparin 40 mg SC daily / UFH / DOAC / none | enoxaparin (dalteparin and fondaparinux also accepted) |
| A_action | `adrenal_incidentaloma/cp4` | Hyperaldosteronism screen? | aldosterone + renin / none / K only | aldosterone + PRA |
| A_judgment | `aortic_aneurysm_cad/cp3` | Is coronary CTA indicated? | yes / no | yes |
| A_judgment | `mgus_in_ckd/cp3` | Cause of elevated FLC? | CKD-related / LC-MGUS / myeloma / AL | CKD-related |
| A_judgment | `thyroid_medication_management/cp5` | Levothyroxine change? | reduce to 75–88 / continue 100 / increase 112 / reduce 50 / hold | reduce to 75–88 mcg |
| A_judgment (neg) | `fungal_blood_culture_review/cp3` | Proceed with fungal blood culture? | cancel / proceed / routine cultures | cancel |
| B_lookup | `iron_deficiency_anemia/cp1` | Most recent ferritin? | value bins | 14.8–16.8 (FHIR-checked, LOINC 2276-4) |
| B_compute | `oligomenorrhea_pcos/cp2` | Number of Rotterdam criteria met? | 0/1/2/3 | 2 |
| B_compute | `hyponatremia_siadh_workup/cp2` | SIADH criteria met? | SIADH / hypovolemic / pseudo / hypervolemic | SIADH |
| C_free | `*/cp_documentation` | — | — | 7-element rubric |

---

## 4. Evaluation protocols for System One models

### Shared building blocks (needed by every protocol)

1. **`pb_dump.py`**
   - Pulls each task patient's full record from the Apptainer server. Use `GET /fhir/Patient/<id>/$everything` (HAPI JPA supports it); otherwise use the 9 search endpoints with `_count`/paging.
   - Writes the raw JSON to `data/physicianbench/fhir_dump/<task>.json` (mode 700).
   - This is a one-off of about 100 calls. After it, all of protocol A is offline.
2. **`pb_render.py`** turns the FHIR dump into the text `state` deterministically:
   - one line per resource, e.g. `[LAB 2023-11-02] LDL-C 79 mg/dL (ref <100)` or `[MED active] rosuvastatin 20 mg PO daily`;
   - decoded note text, in sections;
   - only data dated at or before `TASK_TIMESTAMP`;
   - FHIR boilerplate removed and duplicate lines collapsed.

   It produces three state sizes:
   - **`S_full`:** everything, most recent first, cut to the token budget.
   - **`S_rel(cp)`:** the resources that match that checkpoint's rubric facts (by LOINC, drug, condition or note keywords), plus demographics, the active problem list and active meds. This is "oracle evidence selection".
   - **`S_agent`:** the tool observations an LLM agent actually retrieved (protocol B).
3. **Context scaffold.** Every state starts with a *neutralised* header: the instruction's Context paragraph (date, role, trigger), **without** the numbered "Your Task" steps. As a variant, the full instruction is used too. The difference between the two measures how much instruction leakage helps.
4. **`pb_items.py`** turns the curated JSONL into System One questions:
   - choice questions and per-option yes/no ("noul") questions for A-items;
   - choice, score or yes/no probes for B-facts.

   Probe gold is **recomputed from the FHIR dump**, not copied from the rubric. Values were perturbed, and the rubric text may contain mistakes. For example, ferritin probe gold = `get_lab_value("2276-4")` binned.
5. **Leakage audit, automated.** For each A-item, run its own grader regexes (`name_patterns`/`code_patterns` from the test, plus gold keywords) over the rendered state:
   - pre-existing ServiceRequests and MedicationRequests (is the order already placed?);
   - notes (is the specialist's answer written down?).

   Flagged items are either dropped or kept in a separate "answer-in-chart" stratum.

### Protocol A: offline, oracle-retrieval (recommended first; the E2 analogue of ClinEnv "direct mode")

- **State:** the neutralised header plus `S_rel(cp)`. `S_full` is the long-context condition.
- **Decision questions (from the 321 (a)-items):**
  - one choice question per item, options as curated after the length/specificity rewrite;
  - plus per-option yes/no ("Should <option> be done now for this patient?");
  - set-valued gold where the grader accepts several answers.
- **Probes (from the 165 (b)-checkpoints / 1,190 facts):**
  - lab value → choice over bins: reference-range position, or clinically meaningful cut-points from the rubric (e.g. aortic root <4.0 / 4.0–4.9 / 5.0–5.4 / ≥5.5 cm);
  - medication or condition presence → yes/no;
  - B_compute (score, criteria count) → score levels.
- **Linking decisions to probes.** The relevant probes for a decision are the facts in **that checkpoint's** rubric context: the physicians' own "what you must know to decide". Controls are facts from other checkpoints of the same patient. This gives the reading-vs-deciding 2×2 and the interventions from design §6: (a) gold-facts block, (b) the model's own probe answers, (c) composition in code. Composition (c) is possible where the rubric states a threshold rule, mostly B_compute and threshold-type A_judgment; roughly 40–60 items are expected.
- **Gold:**
  - A_action: the grader patterns (positive). Verified negatives come only from the 34 negative-gold items and from rubric "FAIL if …" clauses.
  - Unverified distractors are allowed **only** in "most appropriate" choice framing, never as yes/no gold.
- **Expected number of items:**
  - about 321 decision items, about 290 after the leakage audit. Times 3 option orders, that is about 900 decision requests.
  - about 1,200 yes/no option items.
  - about 900–1,100 probes after FHIR verification.
  - All from 100 patients, so use a cluster bootstrap over tasks and a dev/test split by task (e.g. 30/70, stratified by specialty).
- **Risks:**
  - **State length:**
    - `S_full` with notes is likely tens of thousands to over 100k tokens per patient. This is unmeasured until the dump.
    - Laya-EN (512 tokens) only fits probe micro-states: one resource plus the header. It cannot do decisions.
    - Laya-multilingual (8k) needs `S_rel`.
    - The 27B models take `S_rel` and a capped `S_full`.
    - Report accuracy against state length.
  - **Leakage:** handled by the audit and the neutralised header.
  - **Gold quality:** the e-consult answer is one specialist's choice, and our distractors are unverified. Ask a clinician to spot-check about 50 items.
  - **Selection bias of `S_rel`:** it is built from the rubric facts, which makes it easier than a real retrieval. Report it together with `S_full`.
- **Code to write:** dump, render, items, leak audit (about 600 lines). The E1 runner and metrics are reused unchanged.

### Protocol B: hybrid loop (local Qwen agent gathers, System One decides; the realistic Q3 setting)

- **Agent:** the unchanged `MiniAgent` with local Qwen3.8-27B, on the Apptainer FHIR server, with the local judge.
- **Decision points**, defined in three ways (B2 is the main one):
  - **B1, intercepting writes.** Wrap the 4 create tools. When the LLM proposes an order, System One gets `S_agent` and answers "Place <display>? yes/no", or chooses among {proposal, curated alternatives for that checkpoint, no order}. The write executes only if System One approves.
    - Measured: native code-grader outcomes, plus System One's calibration against "this order matches a checkpoint pattern".
    - Caveat: a non-matching order is *not* a verified negative.
  - **B2, a scheduled decision phase.** Let the agent retrieve until its first create/`write_file` call or its final answer, then pause. Pose that task's protocol-A questions with `S_agent` as the state.
    - System One's choices are executed by templates: `fhir_service_request_create` / `medication_request_create` with the gold-pattern display and code for the chosen option.
    - The LLM then writes the note, *conditioned on the System One decisions*.
    - Native graders give end-to-end pass@1 and checkpoint rates. Per-decision metrics use protocol-A gold.
    - Compare four arms: LLM-only, LLM-retrieval + System One decision, System One on `S_rel` (A), and the same backbone's LM readout.
  - **B3, System One controls gathering.** At each step System One chooses among the 9 search tools and "decide now". Arguments are templated: patient id, category, and LOINC/drug codes from the item bank.
    - Measured: evidence recall against the `S_rel` facts, downstream decision accuracy, and number of steps. This mirrors ClinEnv's query-vs-submit.
- **Reading vs deciding in B:** probes are asked on `S_agent`. A wrong probe whose fact is absent from `S_agent` is a **retrieval** failure, not a reading failure. That gives a third failure class: retrieval → reading → deciding.
- **Expected size:** 100 tasks × 1–3 runs. The decision items are the same ~321 per run, but the states differ.
- **Risks:**
  - `S_agent` can reach about 100k tokens (40 × 10k-char tool outputs), so it must be compressed with the same renderer. Laya cannot run here.
  - The local judge drifts relative to GPT-5.
  - Runtime: about 10–50 h per run (§5).
  - Agent nondeterminism: fix the seed and temperature 0, and log the exact request bytes.
- **Code to write:**
  - `run_task.py` with the Apptainer and external backends;
  - a `MiniAgent` subclass with a `decision_hook` and a template executor (about 400 lines).

### Protocol C: counterfactual pairs (best answer to "does the decision follow the facts?", and fixes the positive-only gold)

- **Applies to:** items whose rubric gives an explicit rule, e.g.:
  - aortic root vs 5.0/5.5 cm;
  - LDL vs goal;
  - ≥6-week urticaria;
  - Rotterdam count;
  - eGFR cut-offs for metformin or NSAIDs;
  - SIADH criteria.
- **Method:** edit only the decisive value(s) in the rendered state and keep every other line identical. The gold flips by the rule. Ask the same decision and probes on both states.
- **Metrics:**
  - P(decision flips | the fact flipped and the probe was read correctly): a direct "uses the fact" measure;
  - balanced yes/no gold;
  - "ignores the state" detection, matching E1's changed-question control.
- **Expected size:** about 40–80 rule-backed items × 2–3 perturbations each, so roughly 150–250 pairs.
- **Risks:**
  - Clinical plausibility of the edited states: keep edits within a single resource, and get a clinician spot-check.
  - Hidden dependencies, e.g. a note that restates the value. The renderer must edit every occurrence, or drop notes that mention it.

### Optional D: System One as a checkpoint verifier

Use System One to score agent outputs with PASS/PARTIAL/FAIL score levels. Gold comes only from the **code-graded** 112 checkpoints, by running the native code grader on protocol-B trajectories. This tests System One as a calibrated verifier. It is cheap once protocol B has produced trajectories. LLM-judged checkpoints should not serve as gold here, since that would be circular with a local judge.

**Recommended order:**
1. Protocol A on `S_rel`: Laya-multilingual smoke test → 27B models.
2. Protocol C on the rule-backed subset.
3. Protocol B2 once Apptainer and the vLLM agent are validated.
4. B3 and D last.

---

## 5. Practicalities

### 5.1 Python dependencies
- **Declared** (`pyproject.toml:10-15`): `openai`, `requests`, `python-dotenv` and `pytest>=8`.
- **Python version:** 3.10 or later, because of PEP 604 types (`run_task.py:105`). The cluster's `/usr/bin/python3` is 3.6 and too old.
- `uv` is only a convenience (`run_batch_task.sh:117,212`).
- **Pytest config:** `--import-mode=importlib` and `pythonpath=["."]` (`pyproject.toml:20-22`). Tests import `utils.eval_helpers`, so pytest's rootdir must be the repo root. That already holds, because the test path lives under the repo.
- **Plan:** a tiny conda env (`s1eval_pb`) with these 4 packages. Our renderer and question builder need nothing more.

### 5.2 Runtime per task
- **The paper reports no wall-clock time**, only #turns (tool calls): 16–42 per task depending on the model (Table 2; the abstract says about 27 on average).
- **Our estimate for a local Qwen3.8-27B agent:**
  - about 15–40 LLM steps per task, with contexts growing to about 50–100k tokens, at about 10–60 s per step;
  - so about **5–30 min of agent time per task**;
  - plus about 1–2 min for FHIR start, plus 3–7 judge calls (seconds each).
- **Full run:** the 100-task benchmark is about 10–50 h sequentially per run. Use N workers (§5.3).
- **Protocol A** needs no agent. It is one FHIR dump of about 1 min per patient, then System One inference only.

### 5.3 Can tasks run in parallel on one FHIR server?
**Yes, with care.**
- **Why it works:**
  - every task has its own patient;
  - every grader query is scoped to `subject=Patient/<task patient>` and `authored ≥ task date` (`eval_helpers.py:223-240, 395-406, 529-542`);
  - HAPI/H2 in one JVM handles concurrent requests.
- **What breaks it:**
  1. **Repeated runs of the same task** (`--n_runs`, or a rerun) on the same DB. Orders from run 1 satisfy run 2's code graders. The fix is a fresh DB copy per (task, run), or per batch-run with each task once.
  2. **Agents writing to the wrong patient.** This is rare, and the graders would not count it anyway.
  3. **Unfiltered searches.** The agent could see other patients through searches with no patient filter (the Patient search allows that). This leaks no gold.
- **Recommendation:** one server per worker (distinct port and DB copy), each task once per server lifetime, and DB reset between runs. That is cheap, because a server is just `cp -a` of the DB plus a JVM.
- **Protocol A:** dump all 100 patients once from a single server. After that, everything is offline from JSON.

### 5.4 Image size
- **The paper and the repo do not state it.** The README only names `physicianbench-fhir-v1.tar.gz` on Redivis.
- **Expectation:** the HAPI base is about 0.3–0.5 GB, plus an H2 file for 100 longitudinal patients with notes, plausibly 1–5 GB. Check the size on the Redivis download page.
- **Disk budget:** reserve about 4× the archive size during conversion, for `.tar.gz`, `.tar`, sandbox and `.sif`. Use `/ocean/.../apptainer_tmp` as `APPTAINER_TMPDIR`. Keep the image and every FHIR dump/rendering under `data/physicianbench/` (mode 700), never in git (design §7).

### 5.5 Smallest next steps (after the image is downloaded)
1. Build the SIF and sandbox, find the H2 and Lucene paths, and start one instance under Apptainer.
2. Dump the 100 patients (via `$everything`, or the 9 search endpoints).
3. Leakage audit: for every A-item, run its own `code_patterns`/`name_patterns` regex over the rendered state and over pre-existing ServiceRequests and MedicationRequests.
4. Smoke test: run Laya-multilingual on protocol A with the `S_rel` state for about 20 tasks.
5. In parallel, one native run of the Qwen agent on 5 tasks with the local judge, to validate the Apptainer/vLLM patches before protocol B.
