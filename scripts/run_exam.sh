#!/bin/bash
# The fixed exam for every System One decider (user, 2026-10-19): the LLM runs PhysicianBench as usual (arm llm_log:
# it executes its first sample, which is the llm arm's distribution) and logs, at every step, its K = 5 candidate
# turns and the full message history. Afterwards every decider answers the same logged states with the same options
# in the same order (pb_offline_orders.py and the all-step replay).
#   bash scripts/run_exam.sh [workers] [port-base]
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
export S1CLIN_LLM_SERVER="scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2); 8011 = TP2 on 2 GPUs, 8013 = TP1"
.venv/bin/python -m src.s1clin.pb_run --arm llm_log --tasks all --run 1 --k 5 --workers "${1:-6}" --port-base "${2:-19800}" \
  --llm-url "$POOL" >> "logs/full_pb_llm_log_run1.$(hostname -s).log" 2>&1
echo "$(date) llm_log run 1 finished"
