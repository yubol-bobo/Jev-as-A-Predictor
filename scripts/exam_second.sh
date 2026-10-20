#!/bin/bash
# A second instance of a slow decider: once the lane before it is done (WAIT, a pattern in logs/exam_finish.log) and
# its GPUs are free, it answers the exam from the other end of the list (pb_exam.py select --reverse); both instances
# skip what the other has answered. It stops itself when the exam is answered.
#   bash scripts/exam_second.sh NAME GPUS "WAIT"
#   bash scripts/exam_second.sh decider31b 3 "amos (answered the exam|failed to start)"
# 2026-10-20: decider31b answers ~210 questions an hour (NVFP4 weight-only on H100, KV cache for ~3 32k requests).
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
NAME=$1; GPUS=$2; WAIT=$3
declare -A PORT=([pplx12]=8022 [d3]=8023 [linne]=8024 [decider31b]=8025 [amos]=8026)
until grep -qE "$WAIT" logs/exam_finish.log; do sleep 60; done
.venv/bin/python scripts/pb_exam.py answered "$NAME" && { echo "$(date) $NAME has answered the exam; no second instance"; exit 0; }
for g in ${GPUS//,/ }; do until [ "$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done; done
bash scripts/s1_up.sh "$NAME" "$GPUS" 2 > "logs/s1_up_${NAME}_2.log" 2>&1 || { echo "$(date) $NAME instance 2 failed to start"; exit 1; }
echo "$(date) $NAME instance 2 up on GPU $GPUS"
kill "$(cat "logs/exam_select_$NAME.pid")" 2>/dev/null  # exam_deciders.sh restarts it with the code that skips
until .venv/bin/python scripts/pb_exam.py answered "$NAME"; do
  S1CLIN_DECIDER=$NAME S1CLIN_DECIDER_URL="http://127.0.0.1:$((PORT[$NAME] + 100))/decide" \
    .venv/bin/python scripts/pb_exam.py select --workers 2 --reverse
  sleep 30
done
echo "$(date) $NAME has answered the exam; instance 2 stops"
kill -TERM -- -"$(cat "logs/s1_${NAME}_2.pgid")" 2>/dev/null; kill "$(cat "logs/s1_${NAME}_2_proxy.pid")" 2>/dev/null
rm -f "logs/s1_${NAME}_2.pgid" "logs/s1_${NAME}_2_proxy.pid"
