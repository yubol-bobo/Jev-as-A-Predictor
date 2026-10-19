#!/bin/bash
# Full experiment (docs/full_experiment.md) on one 4xH100 node: Qwen3.8-27B TP=2 on GPUs 0,1 (8011), TP=1 on GPU 2
# (8013), Clef on GPU 3 (8021), then scripts/run_full.sh. Resumable; chain jobs with --dependency=afterany:<previous>.
# Several nodes may run this at once: items are claimed (src/s1clin/claim.py), so they share the work.
#   sbatch scripts/slurm/full.sbatch
#SBATCH -J s1clin_full
#SBATCH -A cis260032p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor/logs/slurm_%x_%j.out
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
nvidia-smi -L

# Clef weights are read much faster after a sequential pass into the page cache
C=/ocean/projects/cis260032p/yliy/huggingface_cache/hub/models--Cloudflare--clef/snapshots/2f3de3dd85f379784083b0814d997ab627200f0c
for f in "$C"/*.safetensors; do cat "$f" > /dev/null & done; wait

bash scripts/vllm_qwen_up.sh 0,1 8011; bash scripts/vllm_qwen_up.sh 2 8013
CUDA_VISIBLE_DEVICES=3 HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
  setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
  > logs/clef_server_gpu3.log 2>&1 &
for i in $(seq 1 120); do
  ok=0
  for p in 8011 8013; do curl -s -m 2 -o /dev/null 127.0.0.1:$p/v1/models && ok=$((ok + 1)); done
  grep -q "clef decider on" logs/clef_server_gpu3.log && ok=$((ok + 1))
  [ $ok -eq 3 ] && break; sleep 10
done
echo "servers ready: $ok/3 after $((i * 10)) s"; [ $ok -eq 3 ] || exit 1

bash scripts/run_full.sh
