#!/bin/bash
# M (docs/research_loop.md) on the allocation's four GPUs (the user's leave, 2026-10-26). Phase A: the four System One
# models without pool scores (Clef GPU 0, linne GPU 1, amos GPU 2, decider31b GPU 3) score the task-prior pool from the
# task text alone; then the committee `ens` is built and exported. Phase B: Qwen on GPUs 0 and 3 (8011, 8013: agent,
# judge), pplx12 on GPU 1 (8022, the Qwen arm's repeat judge), Gemma on GPU 2 (8031); the Qwen arm (runs 1 and 2
# together, then run 3) and the Gemma arm (runs 1-3) run side by side. Every step resumes.
#   nohup bash scripts/m_run.sh > logs/m_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
echo $$ > logs/m_run.pid
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_screen
PY=.venv/bin/python
wait_url() { until curl -s -m 5 "$1" | grep -q "$2"; do timeout 15 tail -f /dev/null; done; }
gpu_free() { until [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done; }
until [ "$(grep -c done logs/local_copy_${SLURM_JOB_ID}.log)" -ge 8 ]; do timeout 30 tail -f /dev/null; done
echo "$(date +%T) weights local"

# ---------------------------------------------------------------- phase A: pool scores
if [ ! -s "$OD/prior_ranked_ens.json" ]; then
  CUDA_VISIBLE_DEVICES=0 HF_HOME=$S1CLIN_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
    setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
    > logs/clef_server_gpu0.log 2>&1 < /dev/null &
  echo $! > logs/clef_m.pid
  bash scripts/s1_up.sh linne 1 > logs/m_up_linne.log 2>&1 &
  bash scripts/s1_up.sh amos 2 > logs/m_up_amos.log 2>&1 &
  bash scripts/s1_up.sh decider31b 3 > logs/m_up_decider31b.log 2>&1 &
  score() {  # NAME URL-to-wait-for: score the whole pool, then stop the server
    until grep -q "clef decider on" logs/clef_server_gpu0.log 2>/dev/null || [ "$1" != clef ]; do timeout 15 tail -f /dev/null; done
    [ "$1" != clef ] && until curl -s -m 5 "$2" > /dev/null; do timeout 15 tail -f /dev/null; done
    echo "$(date +%T) $1 up"
    S1CLIN_DECIDER=$1 $PY scripts/pb_prior.py score --full --workers 8 >> "logs/m_score_$1.log" 2>&1
    echo "$(date +%T) $1 scored: $(grep -c '"p": null' "$OD/prior_scores_$1.jsonl") missing of $(wc -l < "$OD/prior_scores_$1.jsonl")"
    if [ "$1" = clef ]; then kill -- -"$(cat logs/clef_m.pid)" 2>/dev/null
    else kill -TERM -- -"$(cat "logs/s1_$1.pgid")" 2>/dev/null; kill "$(cat "logs/s1_$1_proxy.pid")" 2>/dev/null; fi
  }
  score clef "" &
  score linne http://127.0.0.1:8024/health &
  score amos http://127.0.0.1:8026/health &
  score decider31b http://127.0.0.1:8025/health &
  wait
  $PY scripts/pb_prior.py ensemble && $PY scripts/pb_prior.py export --name ens || { echo "$(date +%T) ensemble failed"; exit 1; }
  for N in ens clef linne amos decider31b; do $PY scripts/pb_prior.py factorized --name $N > /dev/null; done
  echo "$(date +%T) M committee scored and exported"
fi

# ---------------------------------------------------------------- phase B: the online arms
for G in 0 1 2 3; do gpu_free $G; done
curl -s -m 5 localhost:8011/v1/models | grep -q qwen || bash scripts/vllm_qwen_up.sh 0 8011 > logs/qwen.pid
curl -s -m 5 localhost:8013/v1/models | grep -q qwen || bash scripts/vllm_qwen_up.sh 3 8013 > logs/qwen3.pid
curl -s -m 5 localhost:8031/v1/models | grep -q gemma || bash scripts/vllm_gemma_up.sh 2 8031 > logs/gemma4.pid
curl -s -m 5 localhost:8022/health | grep -q . || bash scripts/s1_up.sh pplx12 1 > logs/m_up_pplx12.log 2>&1
wait_url localhost:8011/v1/models qwen; wait_url localhost:8013/v1/models qwen
wait_url localhost:8031/v1/models gemma; wait_url localhost:8022/health .
echo "$(date +%T) phase B servers up"
QURL="http://127.0.0.1:8011/v1*1,http://127.0.0.1:8013/v1*1"
qwen_arm() {  # RUN PORT-OFFSET
  $PY -m src.s1clin.pb_run --arm screen --ranker prior-ens-dedup-p1 --b 0 --m 8 --k 1 --tasks @results/r_tasks_1.txt \
    --run "$1" --workers 8 --port-base $((19000 + $2)) --llm-url "$QURL" >> logs/m_qwen_p1.log 2>&1 &
  $PY -m src.s1clin.pb_run --arm screen --ranker prior-ens-dedup-p3 --b 0 --m 8 --k 1 --tasks @results/r_tasks_3.txt \
    --run "$1" --workers 8 --port-base $((19100 + $2)) --llm-url "$QURL" >> logs/m_qwen_p3.log 2>&1 &
  wait
}
gemma_arm() {
  G="--agent gemma4 --agent-url http://127.0.0.1:8031/v1 --repeat-judge none --prefix-arm llm_log_k1__gemma4"
  for RUN in 1 2 3; do
    for P in 1 2; do
      $PY -m src.s1clin.pb_run --arm screen --ranker "prior-ens-p$P" --b 0 --m 8 --k 1 $G \
        --tasks "@results/x_r_tasks_$P.txt" --run $RUN --workers 8 --port-base $((19800 + 50 * P)) --llm-url "$QURL" \
        >> "logs/m_gemma_p$P.log" 2>&1 &
    done
    wait; echo "$(date +%T) M Gemma run $RUN done"
  done
  S1CLIN_AGENT=gemma4 $PY scripts/pb_r.py > /dev/null; echo "$(date +%T) M Gemma arm done"
}
gemma_arm &
GA=$!
qwen_arm 1 0 & Q1=$!
qwen_arm 2 200 & Q2=$!
wait $Q1 $Q2
echo "$(date +%T) M Qwen runs 1-2 done"
qwen_arm 3 0; echo "$(date +%T) M Qwen run 3 done"
wait $GA
$PY scripts/pb_r.py > /dev/null
$PY scripts/pb_m.py > /dev/null
echo "$(date +%T) M done"
