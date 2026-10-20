#!/bin/bash
# 2026-10-20 (w002): finish the fixed exam with all six deciders on one node. Clef (GPU 2) and pplx12 (GPU 3) answer
# while the LLM's llm_log run finishes its last tasks on Qwen TP2 (GPUs 0-1). The exam needs no LLM after that, so:
#   1. llm_log done -> Qwen stops; d3 on GPU 0, linne on GPU 1 (skipped when Qwen is already down)
#   2. Clef done -> linne on GPUs 1+2 (32k inputs need two); pplx12 done -> amos on GPU 3; d3 done -> decider31b on GPU 0
# exam_deciders.sh (running alongside) rebuilds the exam and lets every decider that is up answer it, in parallel.
#   bash scripts/exam_finish.sh
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_exam
gpu_free() { for i in $(seq 1 60); do
  [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return; sleep 5; done; }
answered() { [ "$(grep -vc '"pick": null' "$OD/answers_$1.jsonl" 2>/dev/null || echo 0)" -ge "$(wc -l < "$OD/questions.jsonl")" ]; }

if pgrep -u "$USER" -f "vllm serve .*--port 8011" > /dev/null; then
  while pgrep -u "$USER" -f "src.s1clin.pb_run --arm llm_log" > /dev/null; do sleep 60; done
  echo "$(date) llm_log run 1 finished: $(ls /ocean/projects/cis260032p/yliy/data/physicianbench/jobs/llm_log_k5/run1/*/metadata.json | wc -l) tasks"
  sleep 240  # exam_deciders.sh rebuilds the exam with the last tasks
  pkill -u "$USER" -f "vllm serve .*--port 8011"; gpu_free 0; gpu_free 1
  bash scripts/s1_up.sh d3 0; bash scripts/s1_up.sh linne 1
  echo "$(date) d3 and linne up"
fi

# each GPU changes hands as soon as its own decider is done. linne needs two GPUs for 32k-token inputs (s1_up.sh):
# GPUs 1+2 once Clef is done; amos gets GPU 3 after pplx12, decider31b GPU 0 after d3
(until answered clef; do sleep 60; done
 echo "$(date) Clef answered the final exam ($(wc -l < "$OD/questions.jsonl") questions)"
 pkill -u "$USER" -f "scripts/clef_server.py --port 8021"
 kill -TERM -- -"$(cat logs/s1_linne.pgid)" 2>/dev/null; kill "$(cat logs/s1_linne_proxy.pid)" 2>/dev/null; gpu_free 1; gpu_free 2
 bash scripts/s1_up.sh linne 1,2; echo "$(date) linne up on GPUs 1,2") &
(until answered pplx12; do sleep 60; done
 echo "$(date) pplx12 answered the final exam ($(wc -l < "$OD/questions.jsonl") questions)"
 kill -TERM -- -"$(cat logs/s1_pplx12.pgid)" 2>/dev/null; kill "$(cat logs/s1_pplx12_proxy.pid)" 2>/dev/null; gpu_free 3
 bash scripts/s1_up.sh amos 3; echo "$(date) amos up") &
(until answered d3; do sleep 60; done
 echo "$(date) d3 answered the final exam ($(wc -l < "$OD/questions.jsonl") questions)"
 kill -TERM -- -"$(cat logs/s1_d3.pgid)" 2>/dev/null; kill "$(cat logs/s1_d3_proxy.pid)" 2>/dev/null; gpu_free 0
 bash scripts/s1_up.sh decider31b 0; echo "$(date) decider31b up") &
wait
