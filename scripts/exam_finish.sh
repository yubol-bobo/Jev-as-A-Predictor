#!/bin/bash
# Finish the fixed exam with the System One deciders on one 4-GPU node. Safe to start in a fresh allocation: answers
# are resumable, a decider that has answered the whole exam is skipped, and only servers started by this script in
# this allocation are stopped (remove stale logs/*.pgid and logs/*_proxy.pid from an earlier allocation first).
# Each lane is a GPU (or a GPU pair) and the deciders it runs in turn; the next starts when the current one has
# answered the exam. linne needs two GPUs for 32k-token inputs (s1_up.sh).
#   bash scripts/exam_finish.sh "0:pplx12,decider31b" "3:d3,amos" "1,2:linne"
# exam_deciders.sh (running alongside) rebuilds the exam and lets every decider that is up answer it, in parallel.
# 2026-10-20: the w002 allocation 49510083 ended at 09:28 (its interactive step was killed); resumed in 49512149.
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
gpu_free() { for i in $(seq 1 60); do
  [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return; sleep 5; done; }
answered() {  # every question has an answer, or a final refusal (its options alone exceed the 32k fitting rule)
  .venv/bin/python - "$1" <<'PY'
import json, sys
od = "/ocean/projects/cis260032p/yliy/data/offline/pb_exam"
qs = {json.loads(l)["id"] for l in open(f"{od}/questions.jsonl")}
done = {x["id"] for x in map(json.loads, open(f"{od}/answers_{sys.argv[1]}.jsonl"))
        if x["pick"] is not None or "does not fit" in (x["info"].get("error") or "")}
sys.exit(0 if qs <= done else 1)
PY
}
stop() {  # backend process group and proxy, as recorded by s1_up.sh
  [ -f "logs/s1_$1.pgid" ] && kill -TERM -- -"$(cat "logs/s1_$1.pgid")" 2>/dev/null
  [ -f "logs/s1_$1_proxy.pid" ] && kill "$(cat "logs/s1_$1_proxy.pid")" 2>/dev/null
  rm -f "logs/s1_$1.pgid" "logs/s1_$1_proxy.pid"
}
lane() {
  local gpus=${1%%:*} list=${1#*:} d g
  for d in ${list//,/ }; do
    if answered "$d" 2>/dev/null; then echo "$(date) $d has answered the exam; skipped"; continue; fi
    for g in ${gpus//,/ }; do gpu_free "$g"; done
    if ! bash scripts/s1_up.sh "$d" "$gpus" > "logs/s1_up_$d.log" 2>&1; then
      echo "$(date) $d failed to start on GPU $gpus (logs/s1_up_$d.log)"; stop "$d"; continue
    fi
    echo "$(date) $d up on GPU $gpus"
    until answered "$d" 2>/dev/null; do sleep 60; done
    echo "$(date) $d answered the exam"; stop "$d"
  done
}
for l in "$@"; do lane "$l" & done
wait
