#!/bin/bash
# X after XA and XQ3 (done on allocation 49766322): starts both servers on this project's GPUs (0: Qwen judge 8011,
# 1: Gemma agent 8031), then X1 run 2 alongside XR on the run-1 prefixes; XR on the run-2 prefixes once run 2 is done.
# Same commands as scripts/x_run.sh (every step resumes); only the order changes, the p1 arms no longer wait for run 2.
#   nohup bash scripts/x_run2.sh > logs/x_run2_driver.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
bash scripts/vllm_qwen_up.sh 0 8011 > logs/qwen.pid
bash scripts/vllm_gemma_up.sh 1 8031 > logs/gemma4.pid
until curl -s -m 5 localhost:8011/v1/models | grep -q qwen && curl -s -m 5 localhost:8031/v1/models | grep -q gemma; do
  timeout 15 tail -f /dev/null; done
echo "$(date +%T) Qwen and Gemma up"
GURL=http://127.0.0.1:8031/v1
QURL="http://127.0.0.1:8011/v1*1"
X="--agent gemma4 --agent-url $GURL"
base() {  # run
  .venv/bin/python -m src.s1clin.pb_run --arm llm_log --k 1 $X --tasks all --run "$1" --workers 8 --port-base 19400 \
    --llm-url "$QURL" >> "logs/x_run$1.log" 2>&1
}
r_tasks() {  # run -> results/x_r_tasks_<run>.txt: the run's tasks with a required-order spec open at its first FINISH
  S1CLIN_AGENT=gemma4 .venv/bin/python -c "
import sys; sys.path[:0] = ['scripts', '.']
from pb_replay import cases
print('\n'.join(c['task'] for c in cases(('llm_log_k1__gemma4/run$1',)) if c['open'] and c['finish']))" > "results/x_r_tasks_$1.txt"
  echo "$(date +%T) XR cases of run $1: $(wc -l < "results/x_r_tasks_$1.txt")"
}
xr() {  # arm prefix run port_base
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "restart-$1-p$2" --b 0 --m 8 --k 1 $X --repeat-judge none \
    --prefix-arm llm_log_k1__gemma4 --tasks "@results/x_r_tasks_$2.txt" --run "$3" --workers 4 --port-base "$4" \
    --llm-url "$QURL" >> "logs/xr_$1_p$2.log" 2>&1
}
arms() {  # prefix port_base
  xr none "$1" 1 "$2" &
  for RUN in 1 2 3; do
    xr full "$1" $RUN $(($2 + 100)) & xr fresh "$1" $RUN $(($2 + 200)) &
    wait
    echo "$(date +%T) XR p$1 run $RUN done"
  done
}
( arms 1 19600 ) &
( base 2; echo "$(date +%T) X1 run 2 done"; .venv/bin/python scripts/pb_x1.py > /dev/null; r_tasks 2; arms 2 19650 ) &
wait
S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_r.py > /dev/null
echo "$(date +%T) X done"
