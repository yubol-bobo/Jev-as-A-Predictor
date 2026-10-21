#!/bin/bash
# G1 (docs/research_loop.md): the four further System One models score the P6 subset with the own state and with the
# task instruction only, one model per GPU, all four in parallel; then the P6 report is refreshed. Needs four free GPUs.
#   S1CLIN_HF_HOME=/local/slurm-<job>/local/hf2 bash scripts/g1_run.sh       -> logs/g1_<model>.log
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
GPU=0
for M in d3 linne decider31b amos; do
  (
    bash scripts/s1_up.sh "$M" "$GPU" || { echo "$M failed to start"; exit 1; }
    for MODE in "" "--task"; do
      S1CLIN_DECIDER=$M .venv/bin/python scripts/pb_screen.py score --subset $MODE --workers 4
    done
    echo "$(date +%T) $M done"
  ) > "logs/g1_$M.log" 2>&1 &
  GPU=$((GPU + 1))
done
wait
.venv/bin/python scripts/pb_screen.py report --out results/pb_screen.json > /dev/null
echo "$(date +%T) G1 scoring done"
