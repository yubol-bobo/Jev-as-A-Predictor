#!/bin/bash
# Step 2: untrained Cloudflare Clef (27B) on all 5 tasks' dev/test items (in-process native joint head).
# Resumable: task/splits already in results/predictions/<task>/clef27b/report.json are skipped.
#SBATCH -J seqpred_clef
#SBATCH -A cis260032p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/ocean/projects/cis260025p/yliy/research/Jev-as-A-Predictor/logs/slurm_%x_%j.out
set -uo pipefail
cd /ocean/projects/cis260025p/yliy/research/Jev-as-A-Predictor
export HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312
nvidia-smi -L
/ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python -u scripts/run_clef.py --batch 4
