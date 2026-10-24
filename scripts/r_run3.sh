#!/bin/bash
# R runs 4-5 on two GPUs (this project's share of the allocation; the others belong to another project): Qwen TP1 on
# GPU 0 (8011) + pplx12 on GPU 1 (8022). Unfinished jobs of an earlier allocation are redone clean (claims of a
# finished SLURM job are stale). Then scripts/pb_r.py --runs 1,..,RUN after each run.
#   nohup bash scripts/r_run3.sh > logs/r_run3.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
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
