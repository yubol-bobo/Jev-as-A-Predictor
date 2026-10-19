# Design — System One decision models in clinical agent loops

*Status: DRAFT v0 (2026-10-18). User decisions so far: ClinEnv + PhysicianBench; local open System One models only (no JEV) on both; download all PhysioNet data (MIMIC-IV 3.1, Note 2.2, Ext-CDM 1.1); first exploratory runs with Laya on both benchmarks.*

## 1. Research question

Can **System One decision models**, which take a text state plus a typed question and return a probability distribution over supplied options without generating text, make **clinical decisions inside multi-turn agent loops**? Are their probabilities **calibrated** there? And when they fail, is it because they **misread the state** or because they **fail to decide from facts they read correctly**?

Three sub-questions:

1. **Decision quality and calibration.** Measure per-stage decision accuracy/F1, NLL, ECE, Brier, risk–coverage, and how well confidence ranks the model's own errors (AUROC), on real admissions and real consult tasks.
2. **Reading vs deciding.** Use the framework of the user's arXiv 2610.18579, applied to clinical states. Each decision is paired with factual probe questions about the *same* state, each with a deterministic gold answer. We measure P(decision wrong | all probes right), and run interventions: supply the true facts; supply the model's own probe answers; compose the probe answers in code.
3. **Agent-loop behaviour.** When the System One model controls information gathering (query vs submit, which source next), how does that change decision quality, cost and calibration? How does it compare to an LLM agent and to an LLM + System One hybrid?

Why this is new: every medical System One paper so far is static and single-step, and 2610.18529 names this gap. The agent-loop papers are all non-clinical. No paper pairs fact probes with clinical decisions.

## 2. Environments

| | **E1 ClinEnv** (arXiv 2606.02568) | **E2 PhysicianBench** (arXiv 2605.02240) |
|---|---|---|
| Data | MIMIC-IV v3.1 (hosp + icu) + MIMIC-IV-Note v2.2, real inpatient admissions | Stanford STARR records (de-identified, perturbed) on a HAPI FHIR R4 server, from Redivis (approved) |
| Unit | admission → ordered decision stages (median-size admissions; paper: 3,509 adm / 9,297 stages / 26,043 decisions) | 100 consult tasks, 670 checkpoints, ~27 tool calls per task |
| Decisions | medication start/stop/adjust/switch, procedures (ICD), diagnoses (ICD) | free-form FHIR actions (orders, notes); checkpoints graded by scripts and an LLM judge |
| Interaction | the agent queries 4 simulated sources (patient, nurse, lab, history), then submits; or "direct" mode with the full visible context | the agent calls 14 FHIR tools |
| SOTA | GPT-5.4 decision F1 0.306 (diagnosis 0.51, management 0.17) | about 46–50% pass@1 |
| External APIs | **forbidden** (PhysioNet 2025 policy) → all models local | **not used** (user decision 2026-10-18): local open models only |
| Systems | local only: Laya, Kev-27B, pplx-decider, Clef-27B, local LLM reference | the same (local only; **no JEV**) |

**Order of work: E1 first.** It has per-stage gold labels, so it is the clean testbed for Q1 and Q2. E2 follows as the realistic agent setting for Q3.

## 3. E1 ClinEnv: what we must build or fix

The analysis of the ClinEnv repo is in `research/external/CLINENV_NOTES.md`. These issues bear on the design:

- **The timeline builder is missing from the repo.** It turns raw MIMIC tables into `ehr_by_subject/{subject}/{hadm}.json`. We reimplement it from paper App. B.2: 5 deterministic steps that merge hosp, icu and note events per admission, in time order.
- **The case planner is LLM-based.** It extracts decisions from the discharge note and anchors each one to an event. It must run on a **local LLM** (vLLM, e.g. Qwen3.8-27B) instead of gpt-5.4. Gold drug names and ICD codes come from EHR rows, not from the LLM. The planner adds about 20–60 calls per admission, so we build a subset (e.g. 300–500 admissions), not all 3,509.
- **The public Hugging Face tarball of prepared cases** (`ylin766gatech/ClinEnv`, 140 MB, ungated) probably contains MIMIC-derived note text. **We do not download or use it.** Possible follow-up: email the authors about a credentialed release.
- **Known bugs to fix or document:**
  - ICD-10 procedure codes starting with a digit are scored as ICD-9 (`evaluation/scorer.py:46-49`).
  - Phase A uses only 4 note sections, not the full note.
  - Process metrics are cumulative across stages.
  - The stage-merge safeguard is not implemented.
- **Making it local:**
  - simulators (patient, nurse) → local vLLM;
  - the ICD mapper (embeddings + LLM rerank) → local embedding model + local LLM;
  - RxNav (public NIH service) → only drug names leave the cluster, but we will use an offline RxNorm/ATC table to be safe.

## 4. Turning stages into System One questions (E1)

For each stage, the **state** is the "direct mode" text: patient context plus all visible events up to the stage, and earlier gold decisions shown as already done.

- **Decision questions:** one choice question per gold decision.
  - Options: the gold plus 19 negatives (20 total, matching the protocol in Jev-as-A-Predictor).
  - Negatives are hierarchy neighbours: same ICD chapter or same ATC level-2 class, excluding anything recorded anywhere in the admission. A second, easier option set uses random negatives.
  - Medication action is a 4-way choice (start / stop / adjust / switch).
- **Fact probes:** several per stage, built automatically from the structured fields in the visible events, each with a deterministic gold answer. Examples:
  - "Is the most recent potassium above its reference range?"
  - "Has drug Y been administered before now?"
  - "Is there an active order for X?"
  - "Was procedure Z already performed?"
  
  For each decision we pick the probes **relevant** to it (the facts a clinician would check), plus a few irrelevant control probes.
- **Interventions** (on the same items):
  - (a) state plus a gold-facts block;
  - (b) state plus the model's own probe answers;
  - (c) probe answers combined by a simple rule in code, where a rule exists.
- **Interactive mode (Q3, later):**
  - The System One model decides `query_patient | query_nurse | query_lab | query_history | submit` at each turn. The simulators run on a local LLM.
  - Baselines: a local LLM agent doing the same; a hybrid where the LLM gathers information and System One makes the final choice.

## 5. Systems

| System | Where | Notes |
|---|---|---|
| Laya (EN 421M, 512 tokens; multilingual 322M, up to 8k) | local, `s1eval_laya_cu128` | smoke-test model; the 512-token EN context is too short for full states, so smoke uses the multilingual model |
| Kev-27B, pplx-decider-v1.1-27b, Clef-27B | local (existing runners) | main open models; later finetuned (stage 2) |
| Same-backbone LM readout (Qwen3.8-27B option-token probabilities) | local | tests whether a System One head adds anything (survey 2609.32160) |
| Local generative LLM agent (e.g. Qwen3.8-27B via vLLM) | local | the "System 2" reference; also runs the simulators and the planner |

## 6. Metrics and controls

- **Decision quality:**
  - choice: HR@1, MRR, NDCG@10;
  - ClinEnv native hierarchical F1, after fixing the procedure-code bug.
- **Calibration:**
  - NLL with a 0.005 probability floor;
  - ECE (15 bins), Brier, signed calibration gap;
  - risk–coverage curve / AURC;
  - error-ranking AUROC;
  - temperature refit on a dev split.
- **Reading vs deciding:**
  - probe accuracy;
  - 2×2 table of (all relevant probes right) × (decision right);
  - P(decision wrong | probes right);
  - the base-rate-adjusted statistic, avoiding the independence confound flagged for 2610.18579;
  - intervention deltas.
- **Robustness controls** (from the literature review):
  - 3 seeded option orders per item, with flip rate;
  - a repeat-call noise floor on 300 items;
  - one question per request;
  - log exact request bytes;
  - check probabilities sum to 1 and count exact zeros.
- **Statistics:** 95% cluster bootstrap over patients/admissions; a dev/test split by patient.

## 7. Compliance and data handling

- **Where the data lives:** `/ocean/projects/cis260032p/yliy/data/physionet/` and `.../data/physicianbench/`, mode 700, never in git.
- **Logs and caches:** raw request/response logs and derived cases are also kept outside git. Only code, aggregate metrics and de-identified summary tables are pushed.
- **No MIMIC-derived text to any external API.** Every MIMIC LLM call goes to localhost vLLM. Enforce this with an outbound-URL allowlist in our client, and keep OpenAI/Anthropic/DeepSeek keys unset in the ClinEnv env.
- **Redivis:** no data to external APIs; local models only (user decision 2026-10-18).

## 8. Plan

| Phase | Work | Gate |
|---|---|---|
| P0 | repo; download MIMIC-IV 3.1 + Note 2.2 (after the user signs); PhysicianBench image → Apptainer `.sif`; local vLLM for Qwen3.8-27B | data present, checksums OK |
| P1 | timeline builder (App. B.2) + local planner on **20 admissions**; inspect stages against the paper's examples | user reviews sample stages |
| **P2 (smoke)** | **Laya end-to-end on those ~20 admissions**: decision + probe questions → Laya server → predictions → metrics; GPUs 2–3 | report to the user |
| P3 | build 300–500 admissions; untrained Kev/pplx/Clef/LM-readout/LLM on E1 direct mode, with probes and interventions | report |
| P4 | E1 interactive mode (query vs submit) | report |
| P5 | E2 PhysicianBench (FHIR on Apptainer), local models only | report |
| P6 | finetuning (LoRA on train-split stages; train with changed questions over the same state, to catch "ignores the question" collapse) | — |

**Before MIMIC access lands:** P1/P2 plumbing can be developed on the open **MIMIC-IV demo** (100 patients, hosp + icu, ODbL). It has no notes, so it covers only timelines, structured probes and the Laya pipeline; decisions need the notes.

## 9. Open decisions for the user

1. GitHub repo for this project: name and private/public. (`gh` is not installed. Create an empty repo, and I push.)
2. ~~Redivis / JEV~~ — decided: no JEV, local open models only on both benchmarks.
3. Subset size for E1 (proposed 300–500 admissions) and the local planner model (proposed Qwen3.8-27B).
4. OK to contact the ClinEnv authors about the public HF tarball / a credentialed release?
5. Confirm E1-first ordering, and that the smoke test runs on the MIMIC-IV demo first, then on 20 real admissions.
