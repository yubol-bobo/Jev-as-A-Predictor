#!/bin/bash
# Evening of 2026-10-19 (w008): the exam has priority, so pplx12's online arms are paused (drivers stopped at 19:07,
# resumable). GPU 3 then serves the exam.
#   1. once at most 6 of pplx12's in-flight online tasks remain, a second llm_log driver uses the freed Qwen capacity
#   2. pplx12 stays up until its in-flight online tasks are done (they need it for every decision) and it has
#      answered the exam as it stands; then Clef answers it, then d3 (exam_deciders.sh sees whichever decider is up)
#   bash scripts/exam_sequence.sh GPU
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
GPU=$1
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_exam
inflight() { ps -u "$USER" -o args= | awk '$3=="src.s1clin.pb_run" && $4=="--one"' | grep -c -- "--decider pplx12"; }
caught_up() { [ "$(wc -l < "$OD/questions.jsonl")" -le "$(wc -l < "$OD/answers_$1.jsonl" 2>/dev/null || echo 0)" ]; }
gpu_free() { for i in $(seq 1 60); do
  [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return; sleep 5; done; }
while [ "$(inflight)" -gt 6 ]; do sleep 60; done
echo "$(date) $(inflight) pplx12 online tasks left; second llm_log driver starts"
setsid nohup bash scripts/run_exam.sh 8 20000 > logs/run_exam2.log 2>&1 < /dev/null &
while [ "$(inflight)" -gt 0 ]; do sleep 60; done
until caught_up pplx12; do sleep 60; done
echo "$(date) pplx12 done (online tasks in flight finished, exam answered so far); Clef next"
kill -TERM -- -"$(cat logs/s1_pplx12.pgid)" 2>/dev/null; kill "$(cat logs/s1_pplx12_proxy.pid)" 2>/dev/null; gpu_free
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
  setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
  > logs/clef_server_gpu$GPU.log 2>&1 < /dev/null &
CLEF=$!
until grep -q "clef decider on" "logs/clef_server_gpu$GPU.log"; do sleep 15; done
sleep 120
until caught_up clef; do sleep 60; done
echo "$(date) Clef answered the exam so far; d3 next"
kill "$CLEF"; gpu_free
bash scripts/s1_up.sh d3 "$GPU"
echo "$(date) d3 is up and answers the exam (exam_deciders.sh)"
