#!/bin/bash
# Research loop, online tests (docs/research_loop.md, pre-registered P1 and P2'), in the time left on one 4-GPU node
# once the exam is answered. Qwen TP2 on GPUs 0,1 (8011) and TP1 on GPU 3 (8013); pplx12 on GPU 2 (8022).
#   P1   clef --decider pplx12, run 1: the tasks that arm has not run yet (unseen replication)
#   P2'  gate --margin 0 --points orders --decider pplx12, run 1: pplx12 decides only where a candidate places an
#        order, so FINISH and notes stay with the LLM
#   P4   pplx12 answers the exam blind (pb_exam.py select --blind), on the same pplx12 server
# At DEADLINE (HHMM, the next such time, so it may fall after midnight) the drivers stop, so no new task starts that
# cannot finish (~30 min each); tasks in flight finish or are cut when the allocation ends and restart cleanly later
# (stale claims are taken over, half-done dirs removed). With S1CLIN_HF_HOME set, the models load from that local copy.
#   S1CLIN_HF_HOME=/local/slurm-<job>/local/hf bash scripts/loop_online.sh 0255
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
DEADLINE=${1:-1705}
END=$(date -d "$DEADLINE" +%s); [ "$END" -le "$(date +%s)" ] && END=$((END + 86400))
PY=.venv/bin/python
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
export S1CLIN_LLM_SERVER="scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2); 8011 = TP2 on 2 GPUs, 8013 = TP1"
until $PY scripts/pb_exam.py answered linne; do sleep 60; done
echo "$(date) linne has answered the exam"
for g in 0 1 2 3; do until [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done; done
[ "$(date +%s)" -ge "$((END - 1800))" ] && { echo "$(date) too late to start online runs"; exit 0; }
touch logs/exam.stop  # exam_deciders.sh: the exam is done
bash scripts/vllm_qwen_up.sh 0,1 8011 > logs/qwen_8011.pid
bash scripts/vllm_qwen_up.sh 3 8013 > logs/qwen_8013.pid
bash scripts/s1_up.sh pplx12 2 > logs/s1_up_pplx12.log 2>&1 || { echo "$(date) pplx12 failed to start"; exit 1; }
for p in 8011 8013; do
  until curl -s -m 5 "http://127.0.0.1:$p/v1/models" | grep -q qwen; do sleep 15; done
done
echo "$(date) Qwen and pplx12 up"
until S1CLIN_DECIDER=pplx12 $PY scripts/pb_exam.py answered pplx12_blind; do  # P4; failed answers are retried
  S1CLIN_DECIDER=pplx12 $PY scripts/pb_exam.py select --blind --workers 2 >> logs/p4_blind.log 2>&1; sleep 60
done &
pb() {  # arm workers portbase run extra...; exec, so $! is the driver itself
  local arm=$1 w=$2 pbase=$3 run=$4; shift 4
  exec $PY -m src.s1clin.pb_run --arm "$arm" --tasks all --run "$run" --k 5 --workers "$w" --port-base "$pbase" \
    --llm-url "$POOL" --decider pplx12 "$@" >> "logs/loop_pb_${arm}_run$run.log" 2>&1
}
( pb clef 8 19000 1 ) & P1=$!
( pb gate 14 19200 1 --margin 0 --points orders ) & P2=$!
echo "$P1 $P2" > logs/loop_online.pids
echo "$(date) P1 (driver $P1) and P2' (driver $P2) running"
until [ "$(date +%s)" -ge "$END" ] || ! { kill -0 $P1 2>/dev/null || kill -0 $P2 2>/dev/null; }; do sleep 60; done
# stop the drivers only: no new tasks start; tasks in flight finish if they can before the allocation ends
echo "$(date) drivers stop (deadline $DEADLINE or both done)"
kill $P1 $P2 2>/dev/null
