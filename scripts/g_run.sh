#!/bin/bash
# X (docs/research_loop.md), on this project's two GPUs (0: the Qwen judge, 8011; 1: Gemma 4 31B-it, the agent, 8031).
# After R3-d3 (e_run.sh) ends: stop pplx12 on GPU 1, serve Gemma there, and run the X0 smoke test (2 tasks, run 99).
# The base runs start by hand after the smoke test has been checked.
#   nohup bash scripts/g_run.sh <e_run pid> > logs/g_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
export S1CLIN_HF_HOME=/local/slurm-${SLURM_JOB_ID}/local/hf2
H=/ocean/projects/cis260032p/yliy/huggingface_cache/hub/models--google--gemma-4-31B-it
G=$S1CLIN_HF_HOME/hub/models--google--gemma-4-31B-it
S=842da3794eaa0b77d5f08bae87a17459d91ff475
[ -d "$G/snapshots/$S" ] || { mkdir -p "$G/snapshots" && cp -rL "$H/snapshots/$S" "$G/snapshots/"; }
echo "$(date +%T) Gemma weights local"
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do timeout 60 tail -f /dev/null; done; done
echo "$(date +%T) R3-d3 ended"
kill -- -"$(cat logs/s1_pplx12.pgid)" 2>/dev/null; kill "$(cat logs/s1_pplx12_proxy.pid)" 2>/dev/null
until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do timeout 10 tail -f /dev/null; done
bash scripts/vllm_gemma_up.sh 1 8031 > logs/gemma4.pid
for i in $(seq 120); do curl -s -m 5 localhost:8031/v1/models | grep -q gemma && break; timeout 15 tail -f /dev/null; done
curl -s -m 5 localhost:8031/v1/models | grep -q gemma || { echo "$(date +%T) Gemma did not come up"; exit 1; }
echo "$(date +%T) Gemma up"
.venv/bin/python -m src.s1clin.pb_run --arm llm_log --k 1 --agent gemma4 --agent-url http://127.0.0.1:8031/v1 \
  --tasks aortic_aneurysm_cad,iron_deficiency_anemia --run 99 --workers 2 --port-base 19300 \
  --llm-url "http://127.0.0.1:8011/v1*1" >> logs/x_smoke.log 2>&1
echo "$(date +%T) X0 smoke test done"
