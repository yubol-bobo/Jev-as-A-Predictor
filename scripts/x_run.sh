#!/bin/bash
# X (docs/research_loop.md) after X1 run 1, on GPUs 0 (Qwen judge and labeller, 8011) and 1 (Gemma, 8031):
#   X1 run 2  |  XA (order-step replay of run 1, Gemma samples K = 5)  |  XQ3 (pb_omission + pb_classify, Qwen labels)
# then XR (restart from the shared prefix of runs 1 and 2; none once, full and fresh 3 runs; exact-name repeat filter).
# Every step resumes (claimed tasks / written cases); rerun the script on a new allocation.
#   nohup bash scripts/x_run.sh [<run-1 driver pid>] > logs/x_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
GURL=http://127.0.0.1:8031/v1
QURL="http://127.0.0.1:8011/v1*1"
X="--agent gemma4 --agent-url $GURL"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do timeout 60 tail -f /dev/null; done; done
base() {  # run
  .venv/bin/python -m src.s1clin.pb_run --arm llm_log --k 1 $X --tasks all --run "$1" --workers 8 --port-base 19400 \
    --llm-url "$QURL" >> "logs/x_run$1.log" 2>&1
}
base 1; echo "$(date +%T) X1 run 1 done"
r_tasks() {  # run -> results/x_r_tasks_<run>.txt: the run's tasks with a required-order spec open at its first FINISH
  S1CLIN_AGENT=gemma4 .venv/bin/python -c "
import sys; sys.path[:0] = ['scripts', '.']
from pb_replay import cases
print('\n'.join(c['task'] for c in cases(('llm_log_k1__gemma4/run$1',)) if c['open'] and c['finish']))" > "results/x_r_tasks_$1.txt"
  echo "$(date +%T) XR cases of run $1: $(wc -l < "results/x_r_tasks_$1.txt")"
}
r_tasks 1
( base 2; echo "$(date +%T) X1 run 2 done" ) &
( for C in order_none order_remind order_plan; do
    S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_replay.py run --cond $C --k 5 --url "$GURL*1" --workers 2 >> logs/xa_$C.log 2>&1 &
  done; wait
  S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_replay.py score > /dev/null
  S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_replay_order_x.py > /dev/null; echo "$(date +%T) XA done" ) &
( S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_omission.py > /dev/null
  for K in noted placed reasons wrong; do
    S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_classify.py $K --url http://127.0.0.1:8011/v1 --workers 8 >> logs/xq3.log 2>&1
  done
  S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_classify.py report > /dev/null; echo "$(date +%T) XQ3 done" ) &
wait
r_tasks 2
xr() {  # arm prefix run port_base
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "restart-$1-p$2" --b 0 --m 8 --k 1 $X --repeat-judge none \
    --prefix-arm llm_log_k1__gemma4 --tasks "@results/x_r_tasks_$2.txt" --run "$3" --workers 4 --port-base "$4" \
    --llm-url "$QURL" >> "logs/xr_$1_p$2.log" 2>&1
}
xr none 1 1 19600 & xr none 2 1 19650 &
for RUN in 1 2 3; do
  xr full 1 $RUN 19700 & xr full 2 $RUN 19750 & xr fresh 1 $RUN 19800 & xr fresh 2 $RUN 19850 &
  wait
  echo "$(date +%T) XR run $RUN done"
  S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_r.py > /dev/null
done
echo "$(date +%T) X done"
