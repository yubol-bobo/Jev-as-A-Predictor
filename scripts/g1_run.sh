#!/bin/bash
# G1 (docs/research_loop.md): further System One models score the P6 subset with the own state and with the task
# instruction only. The models given run one after another on one GPU; each server is stopped after its scoring.
#   S1CLIN_HF_HOME=/local/slurm-<job>/local/hf2 bash scripts/g1_run.sh GPU MODEL [MODEL ...]   -> logs/g1_<model>.log
# Refresh results/pb_screen.json afterwards with `pb_screen.py report`.
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
GPU=$1; shift
for M in "$@"; do
  {
    if bash scripts/s1_up.sh "$M" "$GPU"; then
      for MODE in "" "--task"; do
        S1CLIN_DECIDER=$M .venv/bin/python scripts/pb_screen.py score --subset $MODE --workers 4
      done
      echo "$(date +%T) $M done"
    else
      echo "$(date +%T) $M failed to start"
    fi
    kill "$(cat "logs/s1_${M}_proxy.pid")" 2>/dev/null
    kill -- "-$(cat "logs/s1_${M}.pgid")" 2>/dev/null
  } > "logs/g1_$M.log" 2>&1
  for _ in $(seq 60); do  # wait until the GPU is released before the next model loads
    [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break
    timeout 5 tail -f /dev/null
  done
done
