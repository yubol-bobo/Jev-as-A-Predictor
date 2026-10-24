#!/bin/bash
# R3-d3 (docs/research_loop.md), on this project's two GPUs (0: Qwen TP1 8011; 1: one System One model at a time).
# After R run 5 (r_run3.sh) ends: the final R report; swap pplx12 for d3 on GPU 1 and score the whole task-prior pool
# with d3 (task text only), export d3's ranking; swap back to pplx12 (the repeat filter) and run the d3 checklist arm
# from the shared prefix, runs 1-3. Then scripts/pb_r.py.
#   nohup bash scripts/e_run.sh <r_run3 pid> > logs/e_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
H=/ocean/projects/cis260032p/yliy/huggingface_cache/hub
D=$S1CLIN_HF_HOME/hub/models--vllm-sr--d3
[ -d "$D/snapshots" ] || { mkdir -p "$D" && cp -rL $H/models--vllm-sr--d3/snapshots "$D/"; }
echo "$(date +%T) d3 weights local"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do timeout 60 tail -f /dev/null; done; done
echo "$(date +%T) R run 5 ended"
.venv/bin/python scripts/pb_r.py > /dev/null; echo "$(date +%T) R report (all runs)"
s1_down() {  # name: the backend's process group and the proxy
  kill -- -"$(cat logs/s1_$1.pgid)" 2>/dev/null; kill "$(cat logs/s1_$1_proxy.pid)" 2>/dev/null
  until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done
}
s1_down pplx12; bash scripts/s1_up.sh d3 1; echo "$(date +%T) d3 up"
S1CLIN_DECIDER=d3 .venv/bin/python scripts/pb_prior.py score --full --workers 8
.venv/bin/python scripts/pb_prior.py export --name d3 || { echo "$(date +%T) export failed (missing d3 scores)"; exit 1; }
s1_down d3; bash scripts/s1_up.sh pplx12 1; echo "$(date +%T) pplx12 up"
for RUN in 1 2 3; do
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-d3-dedup-p1 --b 0 --m 8 --k 1 --tasks @results/r_tasks_1.txt \
    --run $RUN --workers 8 --port-base 19000 --llm-url "http://127.0.0.1:8011/v1*1" >> logs/r_prior-d3-dedup_p1.log 2>&1 &
  .venv/bin/python -m src.s1clin.pb_run --arm screen --ranker prior-d3-dedup-p3 --b 0 --m 8 --k 1 --tasks @results/r_tasks_3.txt \
    --run $RUN --workers 8 --port-base 19100 --llm-url "http://127.0.0.1:8011/v1*1" >> logs/r_prior-d3-dedup_p3.log 2>&1 &
  wait
  echo "$(date +%T) d3 list run $RUN done"
  .venv/bin/python scripts/pb_r.py > /dev/null
done
echo "$(date +%T) R3-d3 done"
