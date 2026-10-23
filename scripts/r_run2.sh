#!/bin/bash
# R runs 3-5 (r_run.sh ran runs 1-2 on 8011 alone, ~1h45 per run: too slow for the allocation). Both replicas are
# KV-bound (8011 holds ~16 agent contexts at once, 8013 ~3), so the runs go one after another on a shared pool
# (claim.py routes each task to the less loaded replica, weights ~ capacity). While run 2 still holds 8011, run 3
# starts on 8013 alone at one worker per driver; after run 2 the pooled run-3 drivers take the rest (tasks are
# claimed, none runs twice). The order-step replay (A3, A4) on 8013 is paused for this (resumable). Then pb_r.py.
#   nohup bash scripts/r_run2.sh <run-2 driver pids...> > logs/r_run2.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
drv() {  # ranker prefix run port_base llm workers
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "$1-p$2" --b 0 --m 8 --k 1 \
    --tasks "@results/r_tasks_$2.txt" --run "$3" --workers "$6" --port-base "$4" --llm-url "$5" >> "logs/r_$1_p$2.log" 2>&1
}
wave() {  # run port_base llm workers
  local P=$2
  for R in restart-full restart-fresh prior-pplx12-dedup prior-random-dedup; do
    drv $R 1 $1 $P "$3" $4 & drv $R 3 $1 $((P + 50)) "$3" $4 &
    P=$((P + 100))
  done
  wait
}
( wave 3 20000 "http://127.0.0.1:8013/v1*1" 1; echo "$(date +%T) run 3 on 8013 alone stopped" ) &
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do timeout 60 tail -f /dev/null; done; done
echo "$(date +%T) run 2 done"; .venv/bin/python scripts/pb_r.py > /dev/null
for RUN in 3 4 5; do
  wave $RUN 19000 "$POOL" 5
  [ $RUN = 3 ] && wait  # the 8013-only run-3 drivers too
  echo "$(date +%T) run $RUN done"; .venv/bin/python scripts/pb_r.py > /dev/null
done
echo "$(date +%T) R done"
