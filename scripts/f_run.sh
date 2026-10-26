#!/bin/bash
# F role 1 (docs/research_loop.md) on the allocation's four GPUs, from M's phase B servers: Qwen 8011 (GPU 0) and 8013
# (GPU 3), pplx12 8022 (GPU 1), Gemma 8031 (GPU 2). Phase 1: every LLM configuration judges the pool (AG on Gemma;
# B, D, C2, C8 over both Qwen replicas). Phase 2, timing, one judge per GPU at a time: the Qwen configurations on 8011,
# AG on 8031, pplx12 on GPU 1, and on GPU 3 (Qwen 8013 stopped) d3 then Clef; then GPUs 1-2 take linne, amos and
# decider31b. Every step resumes (judge skips judged items; timing skips judges already timed).
#   nohup bash scripts/f_run.sh > logs/f_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
echo $$ > logs/f_run.pid
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
PY=.venv/bin/python
FD=/ocean/projects/cis260032p/yliy/data/offline/pb_f
Q2="http://127.0.0.1:8011/v1*1,http://127.0.0.1:8013/v1*1"
gpu_free() { until [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done; }
timed() { grep -q "\"judge\": \"$1\"" "$FD/timing.jsonl" 2>/dev/null; }
time_judge() { timed "$1" || $PY scripts/pb_f.py timing --judge "$1" --url "${2:-http://127.0.0.1:8011/v1}" >> logs/f_timing.log 2>&1; echo "$(date +%T) timed $1"; }
s1_stop() { kill -TERM -- -"$(cat "logs/s1_$1.pgid")" 2>/dev/null; kill "$(cat "logs/s1_$1_proxy.pid")" 2>/dev/null; }

# ---------------------------------------------------------------- phase 1: judgments
$PY scripts/pb_f.py judge --cfg AG --url http://127.0.0.1:8031/v1 --workers 16 >> logs/f_AG.log 2>&1 &
for C in B D; do $PY scripts/pb_f.py judge --cfg $C --url "$Q2" --workers 16 >> logs/f_$C.log 2>&1 & done
for C in C2 C8; do $PY scripts/pb_f.py judge --cfg $C --url "$Q2" --workers 48 >> logs/f_$C.log 2>&1 & done
wait
echo "$(date +%T) F judgments done: $(for C in AG B D C2 C8; do echo -n "$C $(wc -l < $FD/checker_$C.jsonl) "; done)"

# ---------------------------------------------------------------- phase 2: timing
( for J in A B D C2 C8; do time_judge $J http://127.0.0.1:8011/v1; done ) &
( time_judge AG http://127.0.0.1:8031/v1 ) &
( time_judge pplx12 ) &
( kill -- -"$(cat logs/qwen3.pid)" 2>/dev/null; gpu_free 3
  for N in d3 clef; do
    if [ $N = clef ]; then
      CUDA_VISIBLE_DEVICES=3 HF_HOME=$S1CLIN_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
        setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
        > logs/clef_server_gpu3.log 2>&1 < /dev/null &
      echo $! > logs/clef_f.pid
      until grep -q "clef decider on" logs/clef_server_gpu3.log; do timeout 15 tail -f /dev/null; done
      time_judge clef; kill -- -"$(cat logs/clef_f.pid)" 2>/dev/null
    else
      timed $N || { bash scripts/s1_up.sh $N 3 > logs/f_up_$N.log 2>&1; time_judge $N; s1_stop $N; gpu_free 3; }
    fi
  done ) &
wait
s1_stop pplx12; kill -- -"$(cat logs/gemma4.pid)" 2>/dev/null; gpu_free 1; gpu_free 2
( timed linne || { bash scripts/s1_up.sh linne 1 > logs/f_up_linne.log 2>&1; time_judge linne; s1_stop linne; } ) &
( for N in amos decider31b; do timed $N || { bash scripts/s1_up.sh $N 2 > logs/f_up_$N.log 2>&1; time_judge $N; s1_stop $N; gpu_free 2; }; done ) &
wait
$PY scripts/pb_f.py report > /dev/null
echo "$(date +%T) F role 1 done"
