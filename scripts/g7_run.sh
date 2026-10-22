#!/bin/bash
# G7 (docs/research_loop.md): System One models score the unplaced subset with the state plus a debiasing instruction.
# Waits until the given log reports its server stopped, then runs the models one after another on one GPU.
#   S1CLIN_HF_HOME=/local/slurm-<job>/local/hf2 bash scripts/g7_run.sh GPU WAIT_LOG MODEL [MODEL ...]  -> logs/g7_<model>.log
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
GPU=$1; WAIT=$2; shift 2
until grep -q "stopped" "$WAIT"; do timeout 30 tail -f /dev/null; done
for M in "$@"; do
  for _ in $(seq 60); do  # wait until the GPU is released
    [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break
    timeout 5 tail -f /dev/null
  done
  {
    if bash scripts/s1_up.sh "$M" "$GPU"; then
      S1CLIN_DECIDER=$M .venv/bin/python scripts/pb_screen.py score --warn --workers 4
      echo "$(date +%T) $M G7 done"
    else
      echo "$(date +%T) $M failed to start"
    fi
    kill "$(cat "logs/s1_${M}_proxy.pid")" 2>/dev/null
    kill -- "-$(cat "logs/s1_${M}.pgid")" 2>/dev/null
  } > "logs/g7_$M.log" 2>&1
done
