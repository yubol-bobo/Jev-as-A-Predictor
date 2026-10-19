#!/bin/bash
# Swap Clef for other System One deciders (user, 2026-10-19): for each decider, in turn on one GPU, run 1 of the
# PhysicianBench S1-Select arm (`clef` arm with --decider: K=5 Qwen proposals, the decider picks) and the S1-Tool arm
# (the decider picks the tool, Qwen writes the call), on all 100 tasks, everything else as for Clef. Job tags get the
# suffix __<decider> (clef_k5__d3, s1_tool__d3, ...). Resumable: finished tasks are skipped, items are claimed.
#   bash scripts/run_deciders.sh GPU [decider ...]     default order: pplx12 d3 linne amos decider31b
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
cd "$REPO"
umask 077
GPU=$1; shift
DECIDERS=${*:-pplx12 d3 linne amos decider31b}
PY=.venv/bin/python
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
export S1CLIN_LLM_SERVER="scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2); 8011 = TP2 on 2 GPUs, 8013 = TP1"
H=$(hostname -s)
pb() {  # arm decider workers portbase
  $PY -m src.s1clin.pb_run --arm "$1" --decider "$2" --tasks all --run 1 --k 5 --workers "$3" --port-base "$4" \
    --llm-url "$POOL" >> "logs/full_pb_$1__$2_run1.$H.log" 2>&1
}
stop() {  # the backend's process group and the proxy
  local pg; pg=$(cat "logs/s1_$1.pgid" 2>/dev/null) && kill -TERM -- -"$pg" 2>/dev/null
  kill "$(cat "logs/s1_$1_proxy.pid" 2>/dev/null)" 2>/dev/null
  for i in $(seq 1 60); do
    [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break; sleep 5
  done
}
for d in $DECIDERS; do
  echo "$(date) $d: starting on GPU $GPU"
  if ! bash scripts/s1_up.sh "$d" "$GPU"; then echo "$(date) $d: failed to start, skipped"; stop "$d"; continue; fi
  pb clef "$d" 12 19600 &
  pb s1_tool "$d" 6 19700 &
  wait
  echo "$(date) $d: done"
  stop "$d"
done
echo "deciders finished $(date)"
