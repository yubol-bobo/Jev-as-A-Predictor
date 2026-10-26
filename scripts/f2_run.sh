#!/bin/bash
# F role 2 (docs/research_loop.md), after role 1 (waits for "F role 1 done" in logs/f_run.log on this allocation).
# Phase 1: Qwen on all four GPUs (8011-8014); the 539 branch rollouts (runs 101-120, scripts/pb_f2.py pick) and the
# Qwen selector judgments A, D, C2, C8 over the four replicas. Phase 2: the selector's AG judgments on Gemma and the
# timing of every selector judge, one per GPU at a time (Qwen configurations on 8011 alone; Gemma; the six System One
# servers). Every step resumes: pb_run skips graded jobs, judge and timing skip what is done. On a new allocation copy
# the weights to /local first (as logs/local_copy_<job>.log), then run this again.
#   nohup bash scripts/f2_run.sh > logs/f2_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
echo $$ > logs/f2_run.pid
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
export S1CLIN_BRANCH_SPEC=/ocean/projects/cis260032p/yliy/data/offline/pb_f/branch_spec.json
PY=.venv/bin/python
FD=/ocean/projects/cis260032p/yliy/data/offline/pb_f
gpu_free() { until [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done; }
wait_url() { until curl -s -m 5 "$1" | grep -q "$2"; do timeout 15 tail -f /dev/null; done; }
timed() { grep -q "\"judge\": \"$1\"" "$FD/sel_timing.jsonl" 2>/dev/null; }
time_judge() { timed "$1" || $PY scripts/pb_f2.py timing --judge "$1" --url "${2:-http://127.0.0.1:8011/v1}" >> logs/f2_timing.log 2>&1; echo "$(date +%T) timed $1"; }
s1_stop() { kill -TERM -- -"$(cat "logs/s1_$1.pgid")" 2>/dev/null; kill "$(cat "logs/s1_$1_proxy.pid")" 2>/dev/null; }
s1_time() { timed "$1" || { bash scripts/s1_up.sh "$1" "$2" > "logs/f2_up_$1.log" 2>&1; time_judge "$1"; s1_stop "$1"; gpu_free "$2"; }; }
if kill -0 "$(cat logs/f_run.pid 2>/dev/null)" 2>/dev/null; then
  until grep -q "F role 1 done" logs/f_run.log || ! kill -0 "$(cat logs/f_run.pid)" 2>/dev/null; do timeout 30 tail -f /dev/null; done
fi
echo "$(date +%T) role 2 starts"

# ---------------------------------------------------------------- phase 1: rollouts and Qwen selector judgments
if ! grep -q "F2 rollouts done" logs/f2_run.log 2>/dev/null; then
  for G in 0 1 2 3; do
    curl -s -m 5 localhost:$((8011 + G))/v1/models | grep -q qwen || { gpu_free $G; bash scripts/vllm_qwen_up.sh $G $((8011 + G)) > logs/qwen_g$G.pid; }
  done
  for G in 0 1 2 3; do wait_url localhost:$((8011 + G))/v1/models qwen; done; echo "$(date +%T) four Qwen replicas up"
  Q4=$(for p in 8011 8012 8013 8014; do echo -n "http://127.0.0.1:$p/v1*1,"; done); Q4=${Q4%,}
  declare -A PORT=([A]=8011 [D]=8012 [C2]=8013 [C8]=8014)  # 60 questions each: one replica per configuration
  for C in A D C2 C8; do
    $PY scripts/pb_f2.py judge --cfg $C --url "http://127.0.0.1:${PORT[$C]}/v1" --workers 8 >> logs/f2_$C.log 2>&1 &
  done
  for N in $(seq 101 120); do
    [ -s "$FD/branch_tasks_run$N.txt" ] || continue
    $PY -m src.s1clin.pb_run --arm screen --ranker branch-p1 --b 0 --m 8 --k 1 --tasks "@$FD/branch_tasks_run$N.txt" \
      --run $N --workers 3 --port-base $((20000 + 10 * (N - 100))) --llm-url "$Q4" >> logs/f2_rollouts.log 2>&1 &
  done
  wait
  echo "$(date +%T) F2 rollouts done: $(grep -c ' rc [0-9]' logs/f2_rollouts.log) jobs logged"
fi

# ---------------------------------------------------------------- phase 2: AG and timing
for G in 1 2 3; do [ -s logs/qwen_g$G.pid ] && kill -- -"$(cat logs/qwen_g$G.pid)" 2>/dev/null; done
curl -s -m 5 localhost:8011/v1/models | grep -q qwen || { gpu_free 0; bash scripts/vllm_qwen_up.sh 0 8011 > logs/qwen.pid; }
( wait_url localhost:8011/v1/models qwen; for J in A D C2 C8; do time_judge $J; done ) &
( gpu_free 2; bash scripts/vllm_gemma_up.sh 2 8031 > logs/gemma4.pid; wait_url localhost:8031/v1/models gemma
  $PY scripts/pb_f2.py judge --cfg AG --url http://127.0.0.1:8031/v1 --workers 8 >> logs/f2_AG.log 2>&1
  time_judge AG http://127.0.0.1:8031/v1; kill -- -"$(cat logs/gemma4.pid)" 2>/dev/null; gpu_free 2
  s1_time amos 2; s1_time decider31b 2 ) &
( gpu_free 1; s1_time pplx12 1; s1_time linne 1 ) &
( gpu_free 3; s1_time d3 3
  if ! timed clef; then
    CUDA_VISIBLE_DEVICES=3 HF_HOME=$S1CLIN_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
      setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
      > logs/clef_server_gpu3.log 2>&1 < /dev/null &
    echo $! > logs/clef_f2.pid
    until grep -q "clef decider on" logs/clef_server_gpu3.log; do timeout 15 tail -f /dev/null; done
    time_judge clef; kill -- -"$(cat logs/clef_f2.pid)" 2>/dev/null
  fi ) &
wait
$PY scripts/pb_f2.py report > /dev/null
echo "$(date +%T) F role 2 done"
