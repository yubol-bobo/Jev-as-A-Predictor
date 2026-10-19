#!/bin/bash
# Commit experiment progress (aggregate summary only) every INTERVAL seconds while runs are going.
#   bash scripts/pb_progress_commit.sh NAME [INTERVAL]     e.g. NAME=pilot20 -> results/pb_pilot20_summary.json
# Job directories with patient data stay in data/physicianbench/jobs (outside the repo); the pre-commit hook
# (.githooks) refuses anything data-like. There is no git remote; nothing is pushed.
set -uo pipefail
NAME=$1; INTERVAL=${2:-1200}
REPO=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
OUT=results/pb_${NAME}_summary.json
cd "$REPO"
while true; do
  .venv/bin/python scripts/pb_summarize.py --out "$OUT" > /dev/null 2>&1
  done_n=$(ls /ocean/projects/cis260032p/yliy/data/physicianbench/jobs/*/run*/*/metadata.json 2>/dev/null | wc -l)
  git add "$OUT"
  if ! git diff --cached --quiet -- "$OUT"; then
    git commit -q -m "PhysicianBench $NAME progress: $done_n task-runs finished ($(date '+%F %H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "$OUT" && echo "$(date +%T) committed ($done_n done)"
  fi
  [ -f "$REPO/logs/pb_${NAME}.stop" ] && break
  sleep "$INTERVAL"
done
