#!/bin/bash
# PhysicianBench first (user, 2026-10-19): more workers on all 100 tasks, alongside the drivers already running from
# run_full.sh (claims share the work). ClinEnv is paused meanwhile (scripts/hold_items.py).
# Order: run 1 of llm / clef / s1_tool / llm_log, then runs 2-3 of llm / clef / s1_tool (pass@3, pass^3).
#   bash scripts/run_pb_full.sh
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
PY=.venv/bin/python
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
export S1CLIN_LLM_SERVER="scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2); 8011 = TP2 on 2 GPUs, 8013 = TP1"
H=$(hostname -s)
pb() {  # arm workers portbase run
  $PY -m src.s1clin.pb_run --arm "$1" --tasks all --run "$4" --k 5 --workers "$2" --port-base "$3" \
    --llm-url "$POOL" >> "logs/full_pb_$1_run$4.$H.log" 2>&1
}
if [ "${1:-}" = extra ]; then
  ( for r in 1 2 3; do pb s1_tool 9 19400 $r; done ) &
  ( for r in 1 2 3; do pb clef 4 19500 $r; done ) &
  wait; echo "extra PhysicianBench streams finished $(date)"; exit 0
fi
( for r in 1 2 3; do pb llm 5 19000 $r; done ) &
( for r in 1 2 3; do pb clef 6 19100 $r; done ) &
( for r in 1 2 3; do pb s1_tool 8 19200 $r; done ) &
( pb llm_log 3 19300 1; for r in 2 3; do pb clef 3 19300 $r; done ) &
wait
echo "PhysicianBench streams finished $(date)"
# 10:00 addendum (run as: bash scripts/run_pb_full.sh extra): s1_tool is latency-bound (one tool per turn, cheap in
# tokens) and clef is the slowest K=5 arm, so both get more workers.
