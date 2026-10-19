#!/bin/bash
# Step 3: one Kev-27B LoRA finetuning run (shared recipe, src/seqpred/ft/common.py) on one H100.
#   sbatch --export=ALL,TASK=bpic12,N=500 scripts/slurm/ft_kev27b.sbatch
#SBATCH -J seqpred_ft_kev
#SBATCH -A cis260032p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/ocean/projects/cis260025p/yliy/research/Jev-as-A-Predictor/logs/slurm_%x_%j.out
set -uo pipefail
: "${TASK:?set TASK}" "${N:?set N}"
cd /ocean/projects/cis260025p/yliy/research/Jev-as-A-Predictor
export HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/ocean/projects/cis260032p/yliy/cache/triton_kev27b XDG_CACHE_HOME=/ocean/projects/cis260032p/yliy/cache/xdg_kev27b
unset KEV_TEMPERATURE KEV_DTYPE KEV_FUSED KEV_CUDA_GRAPHS KEV_BACKEND KEV_MERGE KEV_TRUNCATE_STATES KEV_DATE_FACTS
nvidia-smi -L
/ocean/projects/cis260032p/yliy/research/jev/jev_open/kev_v2/.venv/bin/python -u scripts/finetune.py \
  --system kev27b --task "$TASK" --train-n "$N" ${EXTRA:-}
