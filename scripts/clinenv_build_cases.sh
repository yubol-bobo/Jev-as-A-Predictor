#!/bin/bash
# Build ClinEnv cases from full MIMIC-IV 3.1 + Note 2.2 with local models only.
#   scripts/clinenv_build_cases.sh [N=20] [SEED=20261018]
# Steps: count (one pass over all CSVs, cached) -> select (clean.py rules, seeded) ->
#        build timelines -> local planner (vLLM 127.0.0.1:8011) -> stats.
# Everything derived goes to $OUT (mode 700, outside git).
set -euo pipefail
N=${1:-20}
SEED=${2:-20261018}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$REPO/.venv/bin/python
PHYS=/ocean/projects/cis260032p/yliy/data/physionet
OUT=${OUT:-/ocean/projects/cis260032p/yliy/data/clinenv/full}
mkdir -p "$OUT" && chmod 700 "$OUT" /ocean/projects/cis260032p/yliy/data/clinenv

# Compliance: no cloud keys in this process tree; the in-process guard also scrubs them.
unset OPENAI_API_KEY ANTHROPIC_API_KEY DEEPSEEK_API_KEY HF_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

curl -sf http://127.0.0.1:8011/v1/models | grep -q qwen3.8-27b || { echo "local vLLM (qwen3.8-27b @ :8011) not up"; exit 1; }

cd "$REPO"
if [ ! -f "$OUT/stats/count_log.json" ]; then
  $PY -m src.s1clin.clinenv_timeline count --mimic-dir $PHYS/mimiciv/3.1 --note-dir $PHYS/mimic-iv-note/2.2 \
      --out-dir "$OUT" --threads 32 --mem 200GB
fi
$PY -m src.s1clin.clinenv_timeline select --out-dir "$OUT" --n "$N" --seed "$SEED"
$PY -m src.s1clin.clinenv_timeline build --mimic-dir $PHYS/mimiciv/3.1 --note-dir $PHYS/mimic-iv-note/2.2 \
    --out-dir "$OUT" --threads 32 --mem 200GB

cd "$REPO/third_party/clinenv"
export CLINENV_EHR_ROOT=$OUT/ehr_by_subject CLINENV_CASES_DIR=$OUT/cases
export S1CLIN_LOG_CONNECTIONS=$OUT/connections.log
/usr/bin/time -v $PY -m preprocess.main --manifest "$OUT/env_ready_admissions.jsonl" --quiet \
    > "$OUT/plan_run.log" 2>&1 || true
tail -5 "$OUT/plan_run.log"
cd "$REPO"
$PY -m src.s1clin.clinenv_case_stats --cases-dir "$OUT/cases" --manifest "$OUT/env_ready_admissions.jsonl" \
    --json "$OUT/case_stats.json" | head -60
echo "connections made by the planner (must all be 127.0.0.1:8011):"
sort "$OUT/connections.log" | uniq -c
