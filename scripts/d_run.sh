#!/bin/bash
# Allocation 49691349 (w008; this project may use GPUs 0 and 1 only, 2 and 3 belong to another project): the
# order-step replay (A3, A4) on Qwen TP2 (GPUs 0,1, 8011) first; when it ends, score it, then switch to Qwen TP1 on
# GPU 0 (8011) + pplx12 on GPU 1 (8022; 49 GB, does not fit beside TP2 Qwen) and run R run 4, then run 5 as far as the
# allocation goes (tasks are claimed, an unfinished run resumes later). Then scripts/pb_r.py --runs 1,2,3,4.
#   nohup bash scripts/d_run.sh > logs/d_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID:-49691349}/local/hf2
# the replay (resumable) is relaunched until no case is left: a Lustre eviction (13:26) killed it silently once
bash scripts/vllm_qwen_up.sh 0,1 8011 > logs/qwen_tp2.pid
until curl -s -m 5 localhost:8011/v1/models | grep -q qwen; do timeout 15 tail -f /dev/null; done
echo "$(date +%T) Qwen TP2 up"
for TRY in 1 2 3 4 5; do
  for C in order_none order_remind order_plan; do
    .venv/bin/python scripts/pb_replay.py run --cond $C --url "http://127.0.0.1:8011/v1*1" --workers 2 >> logs/a_$C.log 2>&1 &
  done
  wait
  left=$(for C in order_none order_remind order_plan; do tail -n 1 logs/a_$C.log; done | grep -cv ": 0 cases")
  echo "$(date +%T) replay pass $TRY, conditions with cases left at its start: $left"
  [ "$left" = 0 ] && break
done
echo "$(date +%T) order-step replay done"
.venv/bin/python scripts/pb_replay.py score > /dev/null; echo "$(date +%T) scored"
PG=$(ps -o pgid= -p "$(cat logs/qwen_tp2.pid)" | tr -d ' ')  # the TP2 server's process group (vllm_qwen_up.sh's pid)
[ -n "$PG" ] && kill -- -"$PG"
until [ "$(nvidia-smi -i 0,1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')" -lt 2000 ]; do
  timeout 10 tail -f /dev/null; done
echo "$(date +%T) GPUs 0,1 free"
bash scripts/vllm_qwen_up.sh 0 8011
bash scripts/s1_up.sh pplx12 1
until curl -s -m 5 localhost:8011/v1/models | grep -q qwen; do timeout 15 tail -f /dev/null; done
echo "$(date +%T) Qwen TP1 and pplx12 up"
drv() {  # ranker prefix run port_base
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "$1-p$2" --b 0 --m 8 --k 1 \
    --tasks "@results/r_tasks_$2.txt" --run "$3" --workers 2 --port-base "$4" \
    --llm-url "http://127.0.0.1:8011/v1*1" >> "logs/r_$1_p$2.log" 2>&1
}
for RUN in 4 5; do
  P=19000
  for R in restart-full restart-fresh prior-pplx12-dedup prior-random-dedup; do
    drv $R 1 $RUN $P & drv $R 3 $RUN $((P + 50)) &
    P=$((P + 100))
  done
  wait
  echo "$(date +%T) run $RUN done"
  .venv/bin/python scripts/pb_r.py --runs "$(seq -s, 1 $RUN)" > /dev/null
done
echo "$(date +%T) R done"
