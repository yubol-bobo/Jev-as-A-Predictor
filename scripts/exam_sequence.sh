#!/bin/bash
# Evening of 2026-10-19 (w008): once pplx12's online arms finish, GPU 3 serves the exam instead of more online runs.
#   1. a second llm_log driver uses the freed Qwen capacity, so the exam grows faster
#   2. pplx12 stays up until it has answered the exam as it stands, then Clef, then d3 answer it (exam_deciders.sh
#      sees whichever decider is up)
#   bash scripts/exam_sequence.sh GPU
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
GPU=$1
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_exam
online() { ps -u "$USER" -o args= | awk '$3=="src.s1clin.pb_run" && $4=="--arm" && $7=="pplx12"' | grep -c .; }
caught_up() {  # this decider has answered every question built so far
  [ "$(wc -l < "$OD/questions.jsonl")" -le "$(wc -l < "$OD/answers_$1.jsonl" 2>/dev/null || echo 0)" ]
}
stop_decider() {
  local pg; pg=$(cat "logs/s1_$1.pgid" 2>/dev/null) && kill -TERM -- -"$pg" 2>/dev/null
  kill "$(cat "logs/s1_$1_proxy.pid" 2>/dev/null)" 2>/dev/null
  for i in $(seq 1 60); do
    [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break; sleep 5
  done
}
while [ "$(online)" -gt 0 ]; do sleep 60; done
echo "$(date) pplx12 online arms finished; second llm_log driver starts"
setsid nohup bash scripts/run_exam.sh 10 20000 > logs/run_exam2.log 2>&1 < /dev/null &
until caught_up pplx12; do sleep 60; done
echo "$(date) pplx12 answered the exam so far; Clef next"
stop_decider pplx12
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
  setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
  > logs/clef_server_gpu$GPU.log 2>&1 < /dev/null &
CLEF=$!
until grep -q "clef decider on" "logs/clef_server_gpu$GPU.log"; do sleep 15; done
sleep 120
until caught_up clef; do sleep 60; done
echo "$(date) Clef answered the exam so far; d3 next"
kill "$CLEF"
for i in $(seq 1 60); do [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break; sleep 5; done
bash scripts/s1_up.sh d3 "$GPU"
echo "$(date) d3 is up and answers the exam (exam_deciders.sh)"
