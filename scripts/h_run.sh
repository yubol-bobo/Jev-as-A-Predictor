#!/bin/bash
# H (docs/research_loop.md): Qwen (GPU 0, 8011) and Gemma (GPU 1, 8031) each rate every order of the R and XR cases
# (items from `pb_extra.py build`), then the report. Resumes (rated cases are kept).
#   nohup bash scripts/h_run.sh > logs/h_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
curl -s -m 5 localhost:8011/v1/models | grep -q qwen || bash scripts/vllm_qwen_up.sh 0 8011 > logs/qwen.pid
curl -s -m 5 localhost:8031/v1/models | grep -q gemma || bash scripts/vllm_gemma_up.sh 1 8031 > logs/gemma4.pid
until curl -s -m 5 localhost:8011/v1/models | grep -q qwen && curl -s -m 5 localhost:8031/v1/models | grep -q gemma; do
  timeout 15 tail -f /dev/null; done
echo "$(date +%T) Qwen and Gemma up"
for A in qwen gemma4; do
  S1CLIN_AGENT=$A .venv/bin/python scripts/pb_extra.py judge --judge qwen --url http://127.0.0.1:8011/v1 >> logs/h_${A}_qwen.log 2>&1 &
  S1CLIN_AGENT=$A .venv/bin/python scripts/pb_extra.py judge --judge gemma4 --url http://127.0.0.1:8031/v1 >> logs/h_${A}_gemma4.log 2>&1 &
done
wait
.venv/bin/python scripts/pb_extra.py report > /dev/null
echo "$(date +%T) H done"
