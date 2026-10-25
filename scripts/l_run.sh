#!/bin/bash
# L (docs/research_loop.md) on this project's two GPUs: Qwen on GPU 0 (8011: agent, judge and the L ranker) with
# Gemma on GPU 1 (8031) for the ranker scores and the Gemma arm; then pplx12 replaces Gemma on GPU 1 (the R3 repeat
# judge) for the Qwen arm. Every step resumes.
#   nohup bash scripts/l_run.sh > logs/l_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
up() { until curl -s -m 5 "localhost:$1/v1/models" | grep -q "$2"; do timeout 15 tail -f /dev/null; done; }
curl -s -m 5 localhost:8011/v1/models | grep -q qwen || bash scripts/vllm_qwen_up.sh 0 8011 > logs/qwen.pid
QURL="http://127.0.0.1:8011/v1*1"
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_screen
if [ ! -s "$OD/prior_ranked_qwen.json" ] || ! grep -q "L Gemma arm done" logs/l_run.log 2>/dev/null; then
  curl -s -m 5 localhost:8031/v1/models | grep -q gemma || bash scripts/vllm_gemma_up.sh 1 8031 > logs/gemma4.pid
fi
up 8011 qwen; echo "$(date +%T) Qwen up"
S1CLIN_DECIDER=qwen .venv/bin/python scripts/pb_prior.py score --full --workers 16 >> logs/l_score.log 2>&1
.venv/bin/python scripts/pb_prior.py export --name qwen || { echo "$(date +%T) export failed"; exit 1; }
.venv/bin/python scripts/pb_prior.py factorized --name qwen > /dev/null; echo "$(date +%T) L ranker scored and exported"
G="--agent gemma4 --agent-url http://127.0.0.1:8031/v1 --repeat-judge none --prefix-arm llm_log_k1__gemma4"
if ! grep -q "L Gemma arm done" logs/l_run.log; then
  up 8031 gemma; echo "$(date +%T) Gemma up"
  for RUN in 1 2 3; do
    for P in 1 2; do
      .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker "prior-qwen-p$P" --b 0 --m 8 --k 1 $G \
        --tasks "@results/x_r_tasks_$P.txt" --run $RUN --workers 8 --port-base $((19600 + 50 * P)) --llm-url "$QURL" \
        >> "logs/l_gemma_p$P.log" 2>&1 &
    done
    wait; echo "$(date +%T) L Gemma run $RUN done"
  done
  S1CLIN_AGENT=gemma4 .venv/bin/python scripts/pb_r.py > /dev/null; echo "$(date +%T) L Gemma arm done"
fi
kill -- -"$(cat logs/gemma4.pid)" 2>/dev/null
until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done
curl -s -m 5 localhost:8022/health | grep -q . || bash scripts/s1_up.sh pplx12 1
until curl -s -m 5 localhost:8022/health | grep -q .; do timeout 15 tail -f /dev/null; done
echo "$(date +%T) pplx12 up"
for RUN in 1 2 3; do
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-qwen-dedup-p1 --b 0 --m 8 --k 1 --tasks @results/r_tasks_1.txt \
    --run $RUN --workers 8 --port-base 19000 --llm-url "$QURL" >> logs/l_qwen_p1.log 2>&1 &
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-qwen-dedup-p3 --b 0 --m 8 --k 1 --tasks @results/r_tasks_3.txt \
    --run $RUN --workers 8 --port-base 19100 --llm-url "$QURL" >> logs/l_qwen_p3.log 2>&1 &
  wait; echo "$(date +%T) L Qwen run $RUN done"
  .venv/bin/python scripts/pb_r.py > /dev/null
done
echo "$(date +%T) L done"
