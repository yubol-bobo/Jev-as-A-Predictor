# S1-Clinical-Agents

System One decision models (Laya, Kev-27B, pplx-decider, Clef, JEV) as decision makers inside multi-turn clinical agent loops: decision quality, calibration, and reading-vs-deciding failures.

Environments: ClinEnv (MIMIC-IV, local models only) and PhysicianBench (Stanford FHIR).

Design: [docs/design.md](docs/design.md) — draft, under discussion.
ClinEnv pipeline (vendored code, local-only patches, timeline builder, planner): [docs/clinenv_pipeline.md](docs/clinenv_pipeline.md).

| Phase | Status |
|---|---|
| P0 data + infra | in progress |
| P1 ClinEnv case building (20 admissions) | code done and tested: ClinEnv vendored + patched (localhost-only guard, local Qwen3.8-27B, local bge embeddings, offline ATC, ICD-10-PCS scoring and Phase A fixes); timeline builder tested on MIMIC-IV demo and synthetic data; local planner runs end to end on synthetic/demo inputs (all traffic to 127.0.0.1:8011). **Waiting for full MIMIC-IV 3.1 + Note 2.2**, then `scripts/clinenv_build_cases.sh 20` |
| P2 Laya smoke test | toy call OK on GPU 2 |
