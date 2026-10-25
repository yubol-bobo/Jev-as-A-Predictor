#!/bin/bash
# XR3 (docs/research_loop.md): the task-prior checklist (pplx12-ranked vs random, no repeat judge) at the Gemma agent's
# first FINISH, from the XR shared prefixes, runs 1-3. Servers as in scripts/x_run2.sh (Qwen 8011 GPU 0, Gemma 8031
# GPU 1) must be up. Resumes.
#   nohup bash scripts/x_run3.sh > logs/x_run3_driver.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
X="--agent gemma4 --agent-url http://127.0.0.1:8031/v1"
xl() {  # ranker prefix run port_base
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "$1-p$2" --b 0 --m 8 --k 1 $X --repeat-judge none \
    --prefix-arm llm_log_k1__gemma4 --tasks "@results/x_r_tasks_$2.txt" --run "$3" --workers 4 --port-base "$4" \
    --llm-url "http://127.0.0.1:8011/v1*1" >> "logs/xr3_$1_p$2.log" 2>&1
}
for RUN in 1 2 3; do
  xl prior-pplx12 1 $RUN 19600 & xl prior-pplx12 2 $RUN 19650 & xl prior-random 1 $RUN 19700 & xl prior-random 2 $RUN 19750 &
  wait
  echo "$(date +%T) XR3 run $RUN done"
  S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_r.py > /dev/null
done
echo "$(date +%T) XR3 done"
