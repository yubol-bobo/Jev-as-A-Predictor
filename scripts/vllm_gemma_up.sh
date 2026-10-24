#!/bin/bash
# Serve Gemma 4 31B-it as the second agent LLM (the judge stays Qwen, scripts/vllm_qwen_up.sh; pb_run --agent gemma4).
#   bash scripts/vllm_gemma_up.sh GPUS PORT      -> log at logs/vllm_gemma4_gpu<GPUS>.log, prints PID (group leader)
# - bf16 as Qwen; 128k context as Qwen; prefix caching; text only (no image or audio inputs)
# - gemma4 tool and reasoning parsers; thinking is switched on per request (chat_template_kwargs enable_thinking,
#   S1CLIN_QWEN_THINKING=1 as for Qwen); sampling from the model's generation_config
# - no speculative decoding (Qwen's MTP-2 keeps the output distribution, so the two agents differ in speed only)
set -euo pipefail
GPUS=$1; PORT=$2; TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
V=/ocean/projects/cis260032p/yliy/venvs/vllm_new_py311
HF=${S1CLIN_HF_HOME:-/ocean/projects/cis260032p/yliy/huggingface_cache}
M=$HF/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475/
PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$GPUS CUDA_HOME=/opt/packages/cuda/v12.6.1 HF_HUB_OFFLINE=1 \
HF_HOME=$HF setsid nohup $V/bin/vllm serve $M \
  --host 127.0.0.1 --port "$PORT" --served-model-name gemma-4-31b-it --dtype bfloat16 --max-model-len 131072 \
  --tensor-parallel-size "$TP" --max-num-seqs 32 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  > "$REPO/logs/vllm_gemma4_gpu${GPUS//,/}.log" 2>&1 &
echo $!
