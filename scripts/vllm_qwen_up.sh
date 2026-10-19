#!/bin/bash
# Serve Qwen3.8-27B for PhysicianBench agents, proposers and the local judge. One replica per GPU, all identical.
#   bash scripts/vllm_qwen_up.sh GPUS PORT       -> log at logs/vllm_qwen38_gpu<GPUS>.log, prints PID (group leader)
#   GPUS is one GPU ("2") or a comma list ("0,1") for tensor parallelism over those GPUs. TP only splits the same
#   bf16 weights across GPUs; it frees memory for KV cache (the bottleneck: one H100 holds ~230k KV tokens next to
#   the 52 GiB of weights, about 5 long agent contexts), and the model and sampling are unchanged.
# - bf16, 128k context, prefix caching, qwen3_coder tool parser, qwen3 reasoning parser (thinking -> reasoning_content)
# - decode CUDA graphs without torch.compile (-cc.mode=none): compiled mode crashes in this vLLM build
#   (torch stable-ABI aten::new_empty); --max-num-seqs 64 because the Mamba cache caps CUDA-graph batch sizes
# - MTP speculative decoding, 2 tokens: about 2.5x faster than eager decode, and rejection sampling keeps the
#   output distribution unchanged
# - ninja (FlashInfer sampler JIT) comes from the venv bin; nvcc from CUDA_HOME
set -euo pipefail
GPUS=$1; PORT=$2; TP=$(echo "$GPUS" | tr ',' '\n' | wc -l)
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
V=/ocean/projects/cis260032p/yliy/venvs/vllm_new_py311
M=/ocean/projects/cis260032p/yliy/huggingface_cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/
PATH=$V/bin:$PATH CUDA_VISIBLE_DEVICES=$GPUS CUDA_HOME=/opt/packages/cuda/v12.6.1 HF_HUB_OFFLINE=1 \
HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache setsid nohup $V/bin/vllm serve $M \
  --host 127.0.0.1 --port "$PORT" --served-model-name qwen3.8-27b --dtype bfloat16 --max-model-len 131072 --tensor-parallel-size "$TP" \
  -cc.mode=none -cc.cudagraph_mode=FULL_DECODE_ONLY --max-num-seqs 64 --gpu-memory-utilization 0.9 \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  > "$REPO/logs/vllm_qwen38_gpu${GPUS//,/}.log" 2>&1 &
echo $!
