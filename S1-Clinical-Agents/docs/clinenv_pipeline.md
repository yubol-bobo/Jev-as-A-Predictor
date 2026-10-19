# ClinEnv pipeline (local, DUA-compliant)

*Status 2026-10-18.* Steps 1 and 2 are done and tested: ClinEnv is vendored and patched, and the timeline builder works.
The builder was tested on the MIMIC-IV demo and on a synthetic MIMIC-shaped set. The local planner runs end to end on
synthetic and demo inputs. Full MIMIC-IV 3.1 and Note 2.2 are **not downloaded yet** (the compute node gets about 28 KB/s;
the user will run `scripts/download_data.sh` on a login node). So the 20 real cases (step 3) are **not built yet**.
Running `scripts/clinenv_build_cases.sh 20` does all of it once the data is in place.

## 1. Layout

| What | Where | In git? |
|---|---|---|
| ClinEnv upstream `c033d74` plus our patches | `third_party/clinenv/` (`PROVENANCE.md`) | yes (no upstream license; do not publish) |
| Timeline builder (App. B.2) | `src/s1clin/clinenv_timeline.py` | yes |
| Case statistics vs. paper Table 4 | `src/s1clin/clinenv_case_stats.py` | yes |
| End-to-end driver | `scripts/clinenv_build_cases.sh` | yes |
| Tests (synthetic + demo) | `tests/` (`synthetic_mimic.py` makes fake MIMIC CSVs) | yes |
| Python env (duckdb, pandas, openai, pyhealth 1.1.6 `--no-deps`, CPU torch, sentence-transformers) | `.venv/` (`uv`, Python 3.11) | no |
| Raw PhysioNet data | `/ocean/projects/cis260032p/yliy/data/physionet/` | no |
| Everything derived (counts, timelines, cases, logs) | `/ocean/projects/cis260032p/yliy/data/clinenv/` (mode 700) | no |

Subdirectories of the derived-data root:

- `demo/`: demo timelines;
- `synthetic/`: synthetic set and cases;
- `demo_timing/`: planner timing runs;
- `full/`: real MIMIC (empty for now).

## 2. Commands

```bash
REPO=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
PY=$REPO/.venv/bin/python; cd $REPO

# tests (19; synthetic + demo; the guard blocks all non-loopback traffic in-process)
$PY -m pytest -q tests

# timelines, demo (no notes -> --no-note-filter; structure only)
$PY -m src.s1clin.clinenv_timeline all --mimic-dir /ocean/projects/cis260032p/yliy/data/physionet/mimic-iv-demo/2.2 \
    --out-dir /ocean/projects/cis260032p/yliy/data/clinenv/demo --no-note-filter --n 20

# full data, 20 cases (scales to 500 with N=500; the 20 are a prefix of the 500, same seed)
scripts/clinenv_build_cases.sh 20          # count -> select -> build -> plan -> stats

# planner on its own (run from third_party/clinenv)
cd third_party/clinenv
CLINENV_EHR_ROOT=$OUT/ehr_by_subject CLINENV_CASES_DIR=$OUT/cases S1CLIN_LOG_CONNECTIONS=$OUT/connections.log \
  $PY -m preprocess.main --manifest $OUT/env_ready_admissions.jsonl --quiet
```

The local LLM is vLLM serving `qwen3.8-27b` at `127.0.0.1:8011` on GPU 3. The first launch crashed during CUDA-graph
capture (a torch stable-ABI op error), so it was relaunched with `--enforce-eager` and
`CUDA_HOME=/opt/packages/cuda/v12.6.1` (FlashInfer JIT needs `nvcc`). The log is `logs/vllm_qwen38_27b_eager2.log`.
The server runs **without** `--enable-auto-tool-choice` or a reasoning parser.

## 3. Timeline builder (`src/s1clin/clinenv_timeline.py`)

It reimplements App. B.2, which upstream does not ship. The output is exactly what `preprocess/loaders/ehr_loader.py`
reads: `ehr_by_subject/{subject_id}/{hadm_id}.json`, a JSON array of events of the form
`{"source_table", "event_time"?, ...payload}`.

**Three subcommands.** All of them stream the `.csv.gz` files with DuckDB and never decompress them to disk.

1. **`count`** makes one pass over every table and records row counts per (subject, hadm, table). It also flags the
   four required note headings on the longest discharge note of each admission, using `clean.py`'s regexes
   translated to RE2. Results are cached as parquet in `stats/`.
2. **`select`** applies `clean.py`'s rule exactly:
   - required tables: admissions, diagnoses_icd, labevents, note_df and procedures_icd;
   - prescriptions **or** pharmacy;
   - all 4 headings: CC, HPI, BHC and Discharge Diagnosis;
   - `timeline_event_count` ≤ the p50 of the **whole eligible population**, using the same index arithmetic as
     `clean.py:185-191`.

   It then draws a seeded permutation (seed `20261018`). The first N entries form the sample, so the 20-case set is
   a subset of the 500-case set.
3. **`build`** extracts rows for the selected **subjects**, not only the selected admissions. Every admission of
   those subjects is written, so `load_prior_admissions` finds the history. A manifest in `clean.py` format
   (`env_ready_admissions.jsonl`) is written as well.

**Tables, with the joins that supply the fields ClinEnv reads:**

| `source_table` | source | join / derived fields |
|---|---|---|
| `hosp_admissions_df` | admissions | `gender`, `dod` from patients; `age = anchor_age + year(admittime) − anchor_year` |
| `hosp_diagnoses_icd_df` | diagnoses_icd | `long_title_diagnosis` |
| `hosp_procedures_icd_df` | procedures_icd | `long_title_procedure` |
| `hosp_labevents_df` | labevents | `label`, `fluid`, `category` from d_labitems |
| `hosp_microbiologyevents_df` | microbiologyevents | — |
| `hosp_prescriptions_df`, `hosp_pharmacy_df` | prescriptions, pharmacy | — |
| `hosp_emar_detail_df` | emar_detail ⋈ emar on `emar_id` | `hadm_id`, `charttime`, `medication`, `event_txt`, `scheduletime` from emar |
| `ehr_{chart,datetime_,input,output,ingredient,procedure}events_df` | icu tables | `label` from d_items |
| `note_df`, `radiology_note` | note/discharge, note/radiology | — |

**Step 2: keys.** A row is dropped if `subject_id` or `hadm_id` is empty. This drops many outpatient labs.

**Step 3: event time.** `event_time` is the first non-null of `charttime, starttime, admittime, chartdate, stoptime, endtime`.

**Step 4: payload.**
- Removed: identifier, provider and internal columns (`*_id`, `poe_*`, `emar_seq`, `orderid`, …) and null fields.
- Kept: the original time columns.
- All columns are read as text, so ICD codes keep their leading zeros (`0389`).
- Converted: `icd_version`, `seq_num`, `itemid` and `age` to int; numeric fields such as `valuenum`, `ref_range_*`,
  `amount` and `rate` to float.

**Step 5: sort.** Events are sorted by (event_time with nulls last, source_table, natural key). Diagnoses have no
time, so they come last.

**Checks.**
- Tests on synthetic data confirm that the upstream `load_admission`, `build_readviews`, `load_prior_admissions`
  and `clean.build_manifest` accept the output. They also confirm that `clean.py`'s per-admission
  `timeline_event_count` and table set match our count pass.
- On the demo, all 20 sampled admissions reproduce `timeline_event_count` exactly.

**Demo run** (100 patients, no notes): 275 admissions; 91 are eligible without the note filter; the p50 cutoff is
1,918 events (demo patients are ICU-heavy); 20 sampled. It writes 83 files (the 20 plus the same subjects' other
admissions) in 7 s.

## 4. Patches to ClinEnv

Every change is tagged `S1CLIN PATCH` in the code: run `grep -rn "S1CLIN PATCH" third_party/clinenv`. The diff is
in git: the unmodified vendor commit `5eaefab` comes first, then the patch commits.

### Compliance (no MIMIC-derived content leaves 127.0.0.1)

| Patch | Files | Effect |
|---|---|---|
| Socket-level guard | `env/local_guard.py`, imported by `env/__init__.py` and at the top of every CLI script (`preprocess/main.py`, `preprocess/planners/runner.py`, `env/main.py`, `evaluation/main.py`, `run.py`) | `socket.connect` to any non-loopback address raises `LocalOnlyViolation`. This covers openai, anthropic, httpx, requests, urllib and pyhealth. It also scrubs `OPENAI/ANTHROPIC/DEEPSEEK/GEMINI/GOOGLE_API_KEY` and `HF_TOKEN`, and sets HF offline. `S1CLIN_LOG_CONNECTIONS=<file>` logs every connect attempt. |
| No `.env` loading | same scripts | upstream `load_dotenv(override=True)` could re-inject cloud keys |
| Local-only clients | `env/llm_client.py` | `get_openai_client(context=planner/runtime/eval)` requires a localhost `base_url`. A missing config raises instead of falling back to OpenAI. `get_anthropic_client()` raises. The wrapper injects `chat_template_kwargs.enable_thinking=false` and strips `<think>` blocks. |
| Configs | `config/{planner,runtime,eval}.json` | every model is `qwen3.8-27b` at `http://127.0.0.1:8011/v1`. Upstream used gpt-5.4 for the planner, deepseek-chat (api.deepseek.com) for the model under test, and gpt-5.4-mini (OpenAI) for the simulators and judges. |
| Planner client | `preprocess/planners/runner.py` | uses the planner config. Upstream used the **runtime** client, i.e. DeepSeek, with the planner's model name. |
| Embeddings | `env/local_embeddings.py`, `eval.json` | in-process `BAAI/bge-large-en-v1.5` from the local HF cache, CPU by default (`device` is configurable). Qwen3-Embedding, e5 and gte are not cached. bge-small, MiniLM and a MiniLM cross-encoder are also present but weaker. Nothing was downloaded. |
| RxNav → offline ATC | `evaluation/scorers/atc_scoring.py` | drug names are matched exactly against WHO ATC level-5 names from pyhealth's ATC vocabulary, cached in `~/.cache/pyhealth/medcode`. Matching uses upstream's cleaning and aliases, plus a US→INN alias table (acetaminophen→paracetamol, albuterol→salbutamol, …) and 1–2-token prefixes. The fallback is the upstream LLM normalizer, now local. `_rxcui`, `_atc` and `_ndcs` are offline stubs. |

**Effect of the offline ATC lookup on scoring:**
- **Coverage:** 82% of the demo's 584 distinct MAIN prescription drug names resolve, covering 95% of prescription
  rows.
- **Unresolved names** (vaccines, multivitamin brands, some combinations) go through the local LLM normalizer. If
  that also fails, they score 0 unless the names match exactly. Medication scores can therefore only be lower than
  or equal to RxNav-based scores.
- **One exception:** a substance with several ATC codes (e.g. aspirin: A01AD05, B01AC06, N02BA01) is scored on its
  best-matching code pair, where RxNav returned a single code.
- **Unavailable:** `medication_cost`, which needs NDCs. It is disabled in `eval.json`, as upstream.

### Bug fixes

| Bug | Fix |
|---|---|
| `evaluation/scorer.py:46-49`: any code starting with a digit is treated as ICD-9, so ICD-10-PCS codes were looked up in ICD9PROC | uses `icd_version` from the GT record (copied from MIMIC by Phase E); the heuristic is only a fallback |
| *(found while testing)* `icd_scoring._normalize` turned `0DJ08ZZ` into `0D.J08ZZ`, which is not in pyhealth's ICD10PROC, so HDF1 was always 0 even with the right vocabulary | no decimal for ICD10PROC; ICD-9 E-codes split after 4 characters (`E878.6`) |
| *(found)* pyhealth 1.1.6 range nodes without a name (`M00.-M25`) make `lookup()` raise `KeyError` while the ICD index is built | skip them; the cache filename now includes the embedding model |
| `phase_a.py:21-24` searched `record.timeline` for `note_df`, but the loader moves the note out of the timeline, so Phase A always used the 4-section fallback | uses `record.get_notes()`, the full discharge note (paper §3.2). A unit test checks that a section outside the fallback reaches the prompt. |

### Robustness and instrumentation (behaviour-neutral for a working run)

- `helpers.call_llm`:
  - **Timeout:** 45 s → 600 s (configurable).
  - **Retries:** bounded to 6. Upstream retried forever, and `"500"` matched any message.
  - **Text-mode tool calls (`planner.json: "tool_mode": "text"`):** tools are still sent with `tool_choice="none"`,
    so Qwen's template renders them. Qwen3.8 emits `<tool_call><function=…><parameter=…>` in text, which is parsed
    with `skip_special_tokens=false`. This is needed because the vLLM server has no tool-call parser. Phase B code
    is unchanged.
  - **Statistics:** per-phase counts of calls, tokens, LLM seconds and errors.
- `phase_b.py`: the per-window tool loop is capped at 4 rounds. Upstream's `while True` never ends if the model keeps
  issuing invalid indices.
- `runner.py`:
  - writes `plan/planner_stats.json` per case (calls, tokens, wall time per phase, status);
  - **re-raises errors**, where upstream returned `[]` and the failure surfaced later only as a schema-validation
    error.
- `env/paths.py`: `CLINENV_EHR_ROOT` and `CLINENV_CASES_DIR` replace the hard-coded in-repo `data/` paths.

## 5. Deviations from the paper / upstream (deliberate or unavoidable)

1. **Planner model:** Qwen3.8-27B with thinking disabled, at temperature 0, instead of Claude-Sonnet-4.6 (paper) or
   gpt-5.4 (code config). Thinking is off because upstream prompts expect bare JSON and Sonnet ran without extended
   thinking. The simulators and judges are also Qwen3.8-27B (paper: GPT-5.4-mini). Embeddings are bge-large-en-v1.5
   (paper: text-embedding-3-small). The ATC table is offline (paper: RxNav).
2. **Tool calling is emulated in text** (see above). The model sees the same tool schema through its own chat template.
3. **Subset:** we build 20 cases, then up to 500 (paper: 3,509 cases). Selection uses clean.py's p50 over the full
   eligible population, so the case distribution should match. Sampling is seeded uniform random; the paper does not
   say how its 3,509 were chosen from the eligible set.
4. **Timeline details that App. B.2 leaves open:**
   - *emar:* `hosp_emar_detail_df` = emar_detail rows joined to their emar parent. All detail rows are kept, both the
     parent row and the per-product rows, so one administration can appear 2–3 times.
   - *Labels:* ICU tables get only `label` from d_items.
   - *Tie-break:* "original row order" is approximated by each table's natural key (labevent_id, pharmacy_id,
     emar_seq, …), because DuckDB does not expose a CSV row number. This only affects events with the same timestamp
     and table.
   - *Date-only times:* kept as in the CSV (`--date-only keep`, the default). **Consequence:** `procedures_icd.chartdate`
     ('YYYY-MM-DD') string-sorts before every timed event that day. Procedures dated on the admission day are
     therefore flagged `pre_admission`, and Phase B refuses to anchor them. In the demo, 85 of 233 procedure rows
     (36%) are affected. `--date-only end_of_day` (`YYYY-MM-DD 23:59:59`) avoids this, at the cost of placing the
     procedure after that day's events. **This needs a decision before the 500-case build**; we cannot tell what
     upstream did.
5. **Upstream gaps kept as in the code:**
   - the "merge stages with fewer than 10 events" safeguard is not implemented (`MIN_CONTEXT_SPAN` is unused);
   - process metrics are cumulative across stages;
   - there is no `history` key in readviews;
   - the manifest `path` lacks the `case/` segment;
   - the final diagnosis stage is skipped when the last decision anchors to the last event;
   - Phase C ignores bundle anchors when computing the final stage start.
6. **Model under test on this server:** `env/runtime/environment_controller._call_openai` always sends
   `tool_choice="auto"`, which this vLLM server rejects. Running models under test therefore needs either a server
   restart with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` or a text-mode patch like the planner's.
   This is not needed for case building.

## 6. Timings and extrapolation

**Planner measurements.** All runs had concurrency up to 4 on one H100, Qwen3.8-27B in eager mode. The connection
audit showed only `127.0.0.1:8011`.

| Input | Timeline events | LLM calls / case (D, A, B, E) | Wall time / case | Notes |
|---|---|---|---|---|
| Synthetic case (`tests/synthetic_mimic.py`) | 30 | 17 (1, 1, 11, 4) | 31–33 s (1 case alone) | 6 decisions → 4 stages |
| 3 demo admissions + synthetic note | 195–216 | 17.3 mean (D 3–4, B 6–12) | 41 s mean; 3 cases in 47 s concurrently | timing only; the note does not match the events |

**Extrapolation (estimates, not measured on real notes).** Real eligible admissions have ≤ p50 events (paper Fig. 5:
about 25–225). The paper's average is about 7.4 decisions per case, more than our tests. That points to roughly 20–60
calls per case (most of them in Phase B) and 1–2 min per case sequentially.
- **500 cases:** about 2–5 h on one GPU at concurrency 4–8.
- **Prompt sizes:** Phase D uses about 4–5k tokens per 100-event window, well inside the 32k context. Phase A holds
  the full note (usually 2–6k tokens).

**Timeline build on full MIMIC (not measured yet).**
- **`count`** is a single pass over about 10 GB of gzip, dominated by chartevents (about 433M rows). Estimate: 30–60
  min with DuckDB (gzip decompression is single-threaded per file).
- **`build`** for 500 subjects makes a second filtered pass of similar cost.
- **Disk:** only small parquet and JSON files are written. The project quota has about 100 GB free.

**ICD embedding index** (needed only for scoring, not for case building). This is a one-time job: bge-large on CPU
takes about 81 min for ICD10CM, 162 min for ICD10PROC and 15 min for ICD9CM. It would be much faster on a free GPU
(set `embedding.device`).

## 7. Known issues / next steps

- Full data is pending. After it lands, run `scripts/clinenv_build_cases.sh 20`. It reports the counts and statistics
  against Table 4 (stages per case 2.65, ≥3 stages 49.8%, type mix 71.7 / 21.4 / 6.9% dx / med / proc, actions
  79 / 7.5 / 3.8 / 9.5%).
- Decide the `--date-only` policy (see §5.4).
- Phase B quality with Qwen: in a probe, the model sometimes calls `mark_single` even when the decision is absent
  from the window. These false anchors need checking on real cases (user review gate P1).
- The model-under-test tool-call path (§5.6).
