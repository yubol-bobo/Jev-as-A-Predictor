#!/bin/bash
# 2026-10-20 (w002): finish the fixed exam with all six deciders on one node. Clef (GPU 2) and pplx12 (GPU 3) answer
# while the LLM's llm_log run finishes its last tasks on Qwen TP2 (GPUs 0-1). The exam needs no LLM after that, so:
#   1. llm_log done -> Qwen stops; d3 on GPU 0, linne on GPU 1
#   2. Clef and pplx12 have answered the final exam -> amos on GPU 2, decider31b on GPU 3
# exam_deciders.sh (running alongside) rebuilds the exam and lets every decider that is up answer it, in parallel.
#   bash scripts/exam_finish.sh
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_exam
gpu_free() { for i in $(seq 1 60); do
  [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return; sleep 5; done; }
answered() { [ "$(grep -vc '"pick": null' "$OD/answers_$1.jsonl" 2>/dev/null || echo 0)" -ge "$(wc -l < "$OD/questions.jsonl")" ]; }

while pgrep -u "$USER" -f "src.s1clin.pb_run --arm llm_log" > /dev/null; do sleep 60; done
echo "$(date) llm_log run 1 finished: $(ls /ocean/projects/cis260032p/yliy/data/physicianbench/jobs/llm_log_k5/run1/*/metadata.json | wc -l) tasks"
sleep 240  # exam_deciders.sh rebuilds the exam with the last tasks
pkill -u "$USER" -f "vllm serve .*--port 8011"; gpu_free 0; gpu_free 1
bash scripts/s1_up.sh d3 0; bash scripts/s1_up.sh linne 1
echo "$(date) d3 and linne up"

until answered clef && answered pplx12; do sleep 60; done
echo "$(date) Clef and pplx12 answered the final exam ($(wc -l < "$OD/questions.jsonl") questions)"
pkill -u "$USER" -f "scripts/clef_server.py --port 8021"
kill -TERM -- -"$(cat logs/s1_pplx12.pgid)" 2>/dev/null; kill "$(cat logs/s1_pplx12_proxy.pid)" 2>/dev/null
gpu_free 2; gpu_free 3
bash scripts/s1_up.sh amos 2; bash scripts/s1_up.sh decider31b 3
echo "$(date) amos and decider31b up"
