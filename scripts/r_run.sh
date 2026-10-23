#!/bin/bash
# R (docs/research_loop.md): the restart at the agent's FINISH from a shared prefix, on the A2 cases (llm_log_k5 run1
# and run3 tasks with a required-order spec open at FINISH). Arm none once (the replay check); full, fresh, list
# (O1's pplx12 checklist) and listrand (O1's random checklist) five runs each; K = 1; Qwen 8011, pplx12 on its
# usual port. Then scripts/pb_r.py.
#   nohup bash scripts/r_run.sh > logs/r_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
LLM="http://127.0.0.1:8011/v1*1"
drv() {  # ranker prefix run port_base
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "$1-p$2" --b 0 --m 8 --k 1 \
    --tasks "@results/r_tasks_$2.txt" --run "$3" --workers 5 --port-base "$4" --llm-url "$LLM" >> "logs/r_$1_p$2.log" 2>&1
}
drv restart-none 1 1 19000 & drv restart-none 3 1 19100 & wait
echo "$(date +%T) none done"
for RUN in 1 2 3 4 5; do
  P=19000
  for R in restart-full restart-fresh prior-pplx12-dedup prior-random-dedup; do
    drv $R 1 $RUN $P & drv $R 3 $RUN $((P + 50)) &
    P=$((P + 100))
  done
  wait
  echo "$(date +%T) run $RUN done"
  .venv/bin/python scripts/pb_r.py > /dev/null
done
echo "$(date +%T) R done"
