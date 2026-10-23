#!/bin/bash
# Allocation 49632386: finish O1 (the tasks left unfinished when 49587179 ended restart clean) and, alongside it on the
# same Qwen replicas, the A1-A4 replay (docs/research_loop.md); the O1 report is written when both O1 drivers exit.
# Qwen 8011 (TP2, GPUs 0,1) and 8013 (GPU 2); pplx12 on GPU 3 (O1 arm A needs it).
#   nohup bash scripts/b_run.sh > logs/b_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
POOL="http://127.0.0.1:8011/v1*2,http://127.0.0.1:8013/v1*1"
.venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-pplx12-dedup --b 0 --m 8 --tasks @results/o1_tasks.txt \
  --run 1 --workers 14 --port-base 18100 --llm-url "$POOL" >> logs/o1_pplx12_run1.log 2>&1 &
P1=$!
.venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-random-dedup --b 0 --m 8 --tasks @results/o1_tasks.txt \
  --run 1 --workers 14 --port-base 18300 --llm-url "$POOL" >> logs/o1_random_run1.log 2>&1 &
P2=$!
echo "$(date +%T) O1 drivers $P1 $P2"
( while kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; do timeout 60 tail -f /dev/null; done
  bash scripts/o1_report.sh > logs/o1_report.txt 2>&1; echo "$(date +%T) O1 done, report written" ) &
W=()
for COND in finish_fresh finish_ask finish_none finish_nudge; do
  .venv/bin/python scripts/pb_replay.py run --cond $COND --url "$POOL" --workers 8 >> logs/a_$COND.log 2>&1 &
  W+=($!)
done
wait "${W[@]}"; echo "$(date +%T) finish conditions done"
.venv/bin/python scripts/pb_replay.py score > /dev/null; echo "$(date +%T) finish scored"
W=()
for COND in order_none order_remind order_plan; do
  .venv/bin/python scripts/pb_replay.py run --cond $COND --url "$POOL" --workers 10 >> logs/a_$COND.log 2>&1 &
  W+=($!)
done
wait "${W[@]}"; echo "$(date +%T) order conditions done"
.venv/bin/python scripts/pb_replay.py score > /dev/null; echo "$(date +%T) scored"
wait
