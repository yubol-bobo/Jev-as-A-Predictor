#!/bin/bash
# Commit experiment progress (aggregate summaries only) every INTERVAL seconds:
#   results/pb_summary.json (PhysicianBench, all runs), results/clinenv_summary.json (ClinEnv, ours + authors')
#   bash scripts/progress_commit.sh INTERVAL [once]     (stops when logs/commits.stop exists)
# Job and run directories with patient data stay under /ocean/projects/cis260032p/yliy/data (outside the repo);
# the pre-commit hook (.githooks) refuses anything data-like. There is no git remote; nothing is pushed.
set -uo pipefail
INTERVAL=${1:-1200}; ONCE=${2:-}
REPO=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
cd "$REPO"
while true; do
  .venv/bin/python scripts/pb_summarize.py --out results/pb_summary.json > /dev/null 2>&1
  .venv/bin/python scripts/clinenv_summarize.py --authors --out results/clinenv_summary.json > /dev/null 2>&1
  bash scripts/build_paper.sh > /dev/null 2>&1  # pb_analysis.json, paper/generated/*, paper/main.pdf
  n_pb=$(ls /ocean/projects/cis260032p/yliy/data/physicianbench/jobs/*/run*/*/metadata.json 2>/dev/null | wc -l)
  n_ce=$(ls -d /ocean/projects/cis260032p/yliy/data/clinenv/tree/*/*/evals/qwen3.8-27b__* 2>/dev/null | wc -l)
  git add results paper/generated paper/main.pdf paper/figures/*.pdf
  if ! git diff --cached --quiet -- results paper; then
    git commit -q -m "Full experiment progress: PhysicianBench $n_pb task-runs, ClinEnv $n_ce case-runs scored ($(date '+%F %H:%M'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- results paper \
      && echo "$(date +%T) committed (pb $n_pb, ce $n_ce)"
  fi
  [ -n "$ONCE" ] && break
  [ -f logs/commits.stop ] && break
  sleep "$INTERVAL"
done
