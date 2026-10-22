#!/bin/bash
# A1-A6 (docs/research_loop.md), chained after O1 on allocation 49587179: when both O1 drivers have exited, stop
# pplx12 (GPU 3), serve a third Qwen replica there (8014), classify (A5, A6), then sample the replay conditions,
# the finish point first. Nothing new starts after STOP (HHMM); every step resumes from what is already written.
#   S1CLIN_HF_HOME=/local/slurm-<job>/local/hf2 nohup bash scripts/a_run.sh 1630 > logs/a_run.log 2>&1 &
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
cd "$REPO"
umask 077
STOP=$1
POOL="http://127.0.0.1:8011/v1*2,http://127.0.0.1:8013/v1*1,http://127.0.0.1:8014/v1*1"
late() { [ "$(date +%H%M)" -ge "$STOP" ]; }
while ps -eo args | grep -q "[p]b_run --arm screen --ranker prior-.*-dedup"; do timeout 60 tail -f /dev/null; done
echo "$(date +%T) O1 drivers done"
bash scripts/o1_report.sh > logs/o1_report.txt 2>&1; echo "$(date +%T) O1 report written"
kill "$(cat logs/s1_pplx12_proxy.pid)" 2>/dev/null; kill -- "-$(cat logs/s1_pplx12.pgid)" 2>/dev/null
for _ in $(seq 60); do [ "$(nvidia-smi -i 3 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && break; timeout 5 tail -f /dev/null; done
bash scripts/vllm_qwen_up.sh 3 8014
for _ in $(seq 80); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8014/v1/models)" = 200 ] && break; timeout 15 tail -f /dev/null; done
echo "$(date +%T) Qwen on GPU 3 up"
.venv/bin/python scripts/pb_classify.py noted --url http://127.0.0.1:8014/v1 --workers 16
.venv/bin/python scripts/pb_classify.py reasons --url http://127.0.0.1:8014/v1 --workers 32
.venv/bin/python scripts/pb_classify.py report > /dev/null; echo "$(date +%T) A5 A6 classified"
for COND in finish_fresh finish_ask finish_none finish_nudge; do
  late || .venv/bin/python scripts/pb_replay.py run --cond $COND --url "$POOL" --workers 13 > logs/a_$COND.log 2>&1 &
done
wait; echo "$(date +%T) finish conditions done"
for COND in order_none order_remind order_plan; do
  late || .venv/bin/python scripts/pb_replay.py run --cond $COND --url "$POOL" --workers 17 > logs/a_$COND.log 2>&1 &
done
wait; echo "$(date +%T) order conditions done"
.venv/bin/python scripts/pb_replay.py score > /dev/null; echo "$(date +%T) scored"
