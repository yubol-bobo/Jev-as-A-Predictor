#!/bin/bash
# PhysicianBench pilot / full runs on one 4xH100 node: Qwen replicas on GPUs 0, 2, 3 (one per arm), Clef on GPU 1.
# Resumable: finished task-runs (metadata.json) are skipped, unfinished ones restart clean.
#   sbatch --export=ALL,TASKS=@configs/pb_pilot20.txt,RUN=1,NAME=pilot20 scripts/slurm/pb_pilot.sbatch
#SBATCH -J pb_pilot
#SBATCH -A cis260032p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents/logs/slurm_%x_%j.out
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
cd "$REPO"
TASKS=${TASKS:-@configs/pb_pilot20.txt}; RUN=${RUN:-1}; NAME=${NAME:-pilot20}
umask 077
nvidia-smi -L

# Clef weights are read much faster after a sequential pass into the page cache
C=/ocean/projects/cis260032p/yliy/huggingface_cache/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c
for f in "$C"/*.safetensors; do cat "$f" > /dev/null & done; wait

bash scripts/vllm_qwen_up.sh 0 8011; bash scripts/vllm_qwen_up.sh 2 8012; bash scripts/vllm_qwen_up.sh 3 8013
CUDA_VISIBLE_DEVICES=1 HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
  setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
  > logs/clef_server_gpu1.log 2>&1 &
for i in $(seq 1 90); do
  ok=0
  for p in 8011 8012 8013; do curl -s -m 2 -o /dev/null 127.0.0.1:$p/v1/models && ok=$((ok + 1)); done
  grep -q "clef decider on" logs/clef_server_gpu1.log && ok=$((ok + 1))
  [ $ok -eq 4 ] && break; sleep 10
done
echo "servers ready: $ok/4 after $((i * 10)) s"; [ $ok -eq 4 ] || exit 1

bash scripts/pb_progress_commit.sh "$NAME" 1200 > "logs/pb_${NAME}_commits.log" 2>&1 &
COMMIT_LOOP=$!
PIDS=()
for spec in "llm 4 18100 8011" "clef 4 18200 8012" "llm_select 6 18300 8013"; do
  set -- $spec
  .venv/bin/python -m src.s1clin.pb_run --arm $1 --tasks "$TASKS" --run "$RUN" --k 5 --workers $2 --port-base $3 \
    --llm-url http://127.0.0.1:$4/v1 >> "logs/pb_${NAME}_$1.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
kill $COMMIT_LOOP 2>/dev/null
.venv/bin/python scripts/pb_summarize.py --out "results/pb_${NAME}_summary.json" > /dev/null
git add "results/pb_${NAME}_summary.json" && git commit -q -m "PhysicianBench $NAME run$RUN: job finished

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "results/pb_${NAME}_summary.json"
echo done
