#!/bin/bash
# H-b (docs/research_loop.md): each order rated alone by both judges (servers as in scripts/h_run.sh). Resumes.
#   nohup bash scripts/h_run2.sh > logs/h_run2.log 2>&1 &
set -uo pipefail
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
for A in qwen gemma4; do
  S1CLIN_AGENT=$A .venv/bin/python scripts/pb_extra.py judge --single --judge qwen --url http://127.0.0.1:8011/v1 --workers 16 >> logs/hb_${A}_qwen.log 2>&1 &
  S1CLIN_AGENT=$A .venv/bin/python scripts/pb_extra.py judge --single --judge gemma4 --url http://127.0.0.1:8031/v1 --workers 16 >> logs/hb_${A}_gemma4.log 2>&1 &
done
wait
.venv/bin/python scripts/pb_extra.py report --single > /dev/null
echo "$(date +%T) H-b done"
