#!/bin/bash
# F role 1 (docs/research_loop.md) on the allocation's four GPUs. Phase 1: Qwen on all four GPUs (8011 GPU 0, 8014
# GPU 1, 8012 GPU 2, 8013 GPU 3; pplx12 and Gemma make way once AG is judged) and every Qwen configuration judges the
# pool over the four replicas. Phase 2, timing, one judge per GPU at a time: the Qwen configurations on 8011 (GPU 0);
# GPU 2 Gemma (AG), then amos and decider31b; GPU 1 pplx12, then linne; GPU 3 d3, then Clef. Every step resumes (judge
# skips judged items; timing skips judges already timed).
#   nohup bash scripts/f_run.sh > logs/f_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
echo $$ > logs/f_run.pid
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
PY=.venv/bin/python
FD=/ocean/projects/cis260032p/yliy/data/offline/pb_f
gpu_free() { until [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done; }
wait_url() { until curl -s -m 5 "$1" | grep -q "$2"; do timeout 15 tail -f /dev/null; done; }
timed() { grep -q "\"judge\": \"$1\"" "$FD/timing.jsonl" 2>/dev/null; }
time_judge() { timed "$1" || $PY scripts/pb_f.py timing --judge "$1" --url "${2:-http://127.0.0.1:8011/v1}" >> logs/f_timing.log 2>&1; echo "$(date +%T) timed $1"; }
s1_stop() { kill -TERM -- -"$(cat "logs/s1_$1.pgid")" 2>/dev/null; kill "$(cat "logs/s1_$1_proxy.pid")" 2>/dev/null; }
s1_time() { timed "$1" || { bash scripts/s1_up.sh "$1" "$2" > "logs/f_up_$1.log" 2>&1; time_judge "$1"; s1_stop "$1"; gpu_free "$2"; }; }

# ---------------------------------------------------------------- phase 1: judgments on four Qwen replicas
if ! grep -q "F judgments done" logs/f_run.log 2>/dev/null; then
  $PY scripts/pb_f.py judge --cfg AG --url http://127.0.0.1:8031/v1 --workers 16 >> logs/f_AG.log 2>&1
  s1_stop pplx12; kill -- -"$(cat logs/gemma4.pid)" 2>/dev/null; gpu_free 1; gpu_free 2
  curl -s -m 5 localhost:8014/v1/models | grep -q qwen || bash scripts/vllm_qwen_up.sh 1 8014 > logs/qwen1.pid
  curl -s -m 5 localhost:8012/v1/models | grep -q qwen || bash scripts/vllm_qwen_up.sh 2 8012 > logs/qwen2.pid
  wait_url localhost:8014/v1/models qwen; wait_url localhost:8012/v1/models qwen; echo "$(date +%T) four Qwen replicas up"
  Q4=$(for p in 8011 8012 8013 8014; do echo -n "http://127.0.0.1:$p/v1*1,"; done); Q4=${Q4%,}
  for C in B D; do $PY scripts/pb_f.py judge --cfg $C --url "$Q4" --workers 24 >> logs/f_$C.log 2>&1 & done
  for C in C2 C8; do $PY scripts/pb_f.py judge --cfg $C --url "$Q4" --workers 96 >> logs/f_$C.log 2>&1 & done
  wait
  echo "$(date +%T) F judgments done: $(for C in AG B D C2 C8; do echo -n "$C $(wc -l < $FD/checker_$C.jsonl) "; done)"
fi

# ---------------------------------------------------------------- phase 2: timing
( for J in A B D C2 C8; do time_judge $J http://127.0.0.1:8011/v1; done ) &
( kill -- -"$(cat logs/qwen2.pid)" 2>/dev/null; gpu_free 2
  if ! timed AG; then bash scripts/vllm_gemma_up.sh 2 8031 > logs/gemma4.pid; wait_url localhost:8031/v1/models gemma
    time_judge AG http://127.0.0.1:8031/v1; fi
  kill -- -"$(cat logs/gemma4.pid)" 2>/dev/null; gpu_free 2
  s1_time amos 2; s1_time decider31b 2 ) &
( kill -- -"$(cat logs/qwen1.pid)" 2>/dev/null; gpu_free 1; s1_time pplx12 1; s1_time linne 1 ) &
( kill -- -"$(cat logs/qwen3.pid)" 2>/dev/null; gpu_free 3; s1_time d3 3
  if ! timed clef; then
    CUDA_VISIBLE_DEVICES=3 HF_HOME=$S1CLIN_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
      setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
      > logs/clef_server_gpu3.log 2>&1 < /dev/null &
    echo $! > logs/clef_f.pid
    until grep -q "clef decider on" logs/clef_server_gpu3.log; do timeout 15 tail -f /dev/null; done
    time_judge clef; kill -- -"$(cat logs/clef_f.pid)" 2>/dev/null
  fi ) &
wait
$PY scripts/pb_f.py report > /dev/null
echo "$(date +%T) F role 1 done"
