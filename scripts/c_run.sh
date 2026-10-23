#!/bin/bash
# Allocation 49673289: the last O1 tasks on their own replica (Qwen TP2 8011; pplx12 on GPU 3 for arm A), the
# order-step replay (A3, A4) on the other (Qwen 8013, GPU 2), so neither queues behind the other (b_run.sh did).
# The O1 report is written when both O1 drivers exit.
#   nohup bash scripts/c_run.sh > logs/c_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
O1="http://127.0.0.1:8011/v1*1"
A="http://127.0.0.1:8013/v1*1"
.venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-pplx12-dedup --b 0 --m 8 --tasks @results/o1_tasks.txt \
  --run 1 --workers 14 --port-base 18100 --llm-url "$O1" >> logs/o1_pplx12_run1.log 2>&1 &
P1=$!
.venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-random-dedup --b 0 --m 8 --tasks @results/o1_tasks.txt \
  --run 1 --workers 14 --port-base 18300 --llm-url "$O1" >> logs/o1_random_run1.log 2>&1 &
P2=$!
echo "$(date +%T) O1 drivers $P1 $P2"
( while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do timeout 60 tail -f /dev/null; done
  bash scripts/o1_report.sh > logs/o1_report.txt 2>&1; echo "$(date +%T) O1 done, report written" ) &
W=()
for COND in order_none order_remind order_plan; do
  .venv/bin/python scripts/pb_replay.py run --cond $COND --url "$A" --workers 2 >> logs/a_$COND.log 2>&1 &
  W+=($!)
done
wait "${W[@]}"; echo "$(date +%T) order conditions done"
.venv/bin/python scripts/pb_replay.py score > /dev/null; echo "$(date +%T) scored"
wait
