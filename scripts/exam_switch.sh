#!/bin/bash
# Evening of 2026-10-19 (w008), replacing exam_sequence.sh: the exam grows faster than one decider answers it, so each
# decider answers the exam as it stands when it comes up, then hands GPU 3 to the next one: pplx12 -> Clef -> d3.
# pplx12's last in-flight online tasks get until DEADLINE; if they are still running, they are interrupted (SIGINT, so
# their FHIR servers are stopped) and rerun next allocation, rather than finishing with no decider behind them.
#   bash scripts/exam_switch.sh GPU DEADLINE(HH:MM)
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
GPU=$1; DEADLINE=$2
OD=/ocean/projects/cis260032p/yliy/data/offline/pb_exam
inflight() { ps -u "$USER" -o pid=,args= | awk '$4=="src.s1clin.pb_run" && $5=="--one"' | grep -- "--decider pplx12" | awk '{print $1}'; }
gpu_free() { for i in $(seq 1 60); do
  [ "$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return; sleep 5; done; }
answer_snapshot() {  # wait until decider $1 has answered every question that existed when it came up
  local n; n=$(wc -l < "$OD/questions.jsonl")
  until [ "$(grep -vc '"pick": null' "$OD/answers_$1.jsonl" 2>/dev/null || echo 0)" -ge "$n" ]; do sleep 30; done
}
while [ -n "$(inflight)" ] && [ "$(date +%H:%M)" \< "$DEADLINE" ]; do sleep 30; done
if [ -n "$(inflight)" ]; then
  echo "$(date) interrupting pplx12 online tasks: $(inflight | tr '\n' ' ')"; inflight | xargs -r kill -INT; sleep 60
fi
echo "$(date) pplx12 off GPU $GPU (answered $(wc -l < "$OD/answers_pplx12.jsonl") questions); Clef next"
kill -TERM -- -"$(cat logs/s1_pplx12.pgid)" 2>/dev/null; kill "$(cat logs/s1_pplx12_proxy.pid)" 2>/dev/null; gpu_free
CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/ocean/projects/cis260032p/yliy/huggingface_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=/ocean/projects/cis260032p/yliy/venvs/clef_deps_py312 PYTHONUNBUFFERED=1 \
  setsid nohup /ocean/projects/cis260032p/yliy/venvs/s1eval_kev/bin/python scripts/clef_server.py --port 8021 \
  > "logs/clef_server_gpu$GPU.log" 2>&1 < /dev/null &
CLEF=$!
until grep -q "clef decider on" "logs/clef_server_gpu$GPU.log"; do sleep 15; done
echo "$(date) Clef up"
answer_snapshot clef
echo "$(date) Clef answered $(wc -l < "$OD/answers_clef.jsonl") questions; d3 next"
kill "$CLEF"; gpu_free
bash scripts/s1_up.sh d3 "$GPU"
echo "$(date) d3 up"
answer_snapshot d3
echo "$(date) d3 answered $(wc -l < "$OD/answers_d3.jsonl") questions"
