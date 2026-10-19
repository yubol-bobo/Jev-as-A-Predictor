#!/bin/bash
# Start one alternative System One decider on one GPU: the author's own server (or, for pplx-decider-v1.2, which ships
# none, a minimal one around its library), then scripts/s1_proxy.py, which gives it Clef's /decide API, state
# rendering and 32k fitting (src/s1clin/pb_decider.DECIDER_URLS: pplx12 8022, d3 8023, linne 8024, decider31b 8025,
# amos 8026). Everything is local and offline (HF_HUB_OFFLINE=1); no request is logged by the servers.
#   bash scripts/s1_up.sh NAME GPU          prints the backend's process group id; logs in logs/s1_<name>_*.log
# Serving, as the authors ship it (revisions pinned):
#   pplx12      perplexity-ai/pplx-decider-v1.2 @ cce0f3f8, pplx_decider.Decider reference path (fast=False), saved T
#   d3          vllm-sr/d3 @ v3.1.0 (cbff3a3c), its d3_server.py (batch 8, verify fast)
#   linne       PNPM244/linne-oak-1-0 @ 20da7bf3, linne_engine (noncausal, token_budget 16384) behind linne_shim.py
#   decider31b  Mapika/decider-31b @ e12ddf20, decider.serve_vllm (decider-ai 1.9.0, vLLM 0.29.0) with the repo's
#               decider_config.json; H100 has no native NVFP4, so vLLM runs the NVFP4 MLP weights through its
#               weight-only FP4 path (the authors ran Blackwell)
#   amos        mukaddamzaid/amos-1-large @ d67e051c, its amos.serve (one option order, calibration.json)
set -euo pipefail
NAME=$1; GPU=$2
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
V=/ocean/projects/cis260032p/yliy/venvs
export HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
unset AMOS_THINK_URL DECISION_API_KEY AUTOJEV_API_KEY DECIDER_TEMPERATURE DECIDER_LAYOUT
C=/ocean/projects/cis260032p/yliy/cache/s1clin_$NAME; mkdir -p "$C"
export TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg VLLM_CACHE_ROOT=$C/vllm
snap() { echo "$HF_HOME/hub/models--${1//\//--}/snapshots/$2"; }
LOG=$REPO/logs/s1_${NAME}_server.log
cd "$C"  # nothing importable next to the servers
case $NAME in
  pplx12) S=$(snap perplexity-ai/pplx-decider-v1.2 cce0f3f80033e9d89d80831d5281fd72124aacce); BP=27340; PORT=8022
    PYTHONPATH=$S/source/src setsid nohup $V/d3/bin/python -u "$REPO/scripts/s1_servers/pplx12_server.py" \
      --checkpoint "$S" --port $BP > "$LOG" 2>&1 < /dev/null & ;;
  d3) S=$(snap vllm-sr/d3 cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75); BP=27330; PORT=8023
    PYTHONPATH=$V/s1clin_web setsid nohup $V/d3/bin/python -B -u "$S/d3_server.py" --model "$S" --device cuda:0 \
      --batch-size 8 --verify fast --host 127.0.0.1 --port $BP > "$LOG" 2>&1 < /dev/null & ;;
  linne) S=$(snap PNPM244/linne-oak-1-0 20da7bf3eb4d13b21ae0403e4be834ca19ec5cd0); BP=27320; PORT=8024
    PYTHONSAFEPATH=1 setsid nohup $V/linne/bin/python -u "$REPO/scripts/s1_servers/linne_shim.py" --snapshot "$S" \
      --repo PNPM244/linne-oak-1-0 --revision 20da7bf3eb4d13b21ae0403e4be834ca19ec5cd0 --host 127.0.0.1 --port $BP \
      > "$LOG" 2>&1 < /dev/null & ;;
  decider31b) S=$(snap Mapika/decider-31b e12ddf20fbc8aa975622414c6b3568804717c8ae); BP=27350; PORT=8025
    export DECIDER_MODEL=$S DECIDER_VLLM_MAX_MODEL_LEN=40960 DECIDER_MAX_STATE_TOKENS=36864 \
      DECIDER_VLLM_GPU_MEMORY_UTILIZATION=0.90 DECIDER_VLLM_MAX_NUM_SEQS=16 VLLM_USE_FLASHINFER_SAMPLER=0
    PYTHONSAFEPATH=1 setsid nohup $V/decider/bin/python -m uvicorn decider.serve_vllm:app --host 127.0.0.1 --port $BP \
      > "$LOG" 2>&1 < /dev/null & ;;
  amos) S=$(snap mukaddamzaid/amos-1-large d67e051c763dc88a1e3662a5003c1e1ea5ffaa74); BP=27360; PORT=8026
    PYTHONPATH=$S/serve setsid nohup $V/d3/bin/python -u -m amos.serve "$S" --device cuda --host 127.0.0.1 \
      --port $BP --max-len 131072 --name amos-1-large > "$LOG" 2>&1 < /dev/null & ;;
  *) echo "unknown decider $NAME" >&2; exit 1 ;;
esac
PG=$!
HP=/health; [ "$NAME" = amos ] && HP=/v1/models  # amos.serve has no /health
echo "$PG" > "$REPO/logs/s1_${NAME}.pgid"
for i in $(seq 1 240); do
  curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$BP$HP" 2>/dev/null | grep -q 200 && break
  kill -0 "$PG" 2>/dev/null || { echo "$NAME backend died" >&2; tail -30 "$LOG" >&2; exit 1; }
  sleep 10
done
echo "$NAME backend: $(curl -s -m 5 http://127.0.0.1:$BP$HP | cut -c1-600)"
TOK=$S/tokenizer.json; [ -f "$TOK" ] || TOK=$S/model/tokenizer.json
setsid nohup "$REPO/.venv/bin/python" "$REPO/scripts/s1_proxy.py" --name "$NAME" --port $PORT \
  --backend "http://127.0.0.1:$BP/v1/systemone" --tokenizer "$TOK" > "$REPO/logs/s1_${NAME}_proxy.log" 2>&1 < /dev/null &
echo $! > "$REPO/logs/s1_${NAME}_proxy.pid"
sleep 2; curl -s -m 5 "http://127.0.0.1:$PORT/health"; echo
