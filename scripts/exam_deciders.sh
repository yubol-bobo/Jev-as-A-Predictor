#!/bin/bash
# Keep the fixed exam current and have every System One decider that is up answer it (scripts/pb_exam.py). The exam
# grows as the LLM's llm_log run finishes tasks; answers are resumable, so each decider answers only new questions.
# Clef (8021) answers too when it is up, with the same requests as everyone else.
#   bash scripts/exam_deciders.sh        (loops; stops when logs/exam.stop exists)
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
declare -A PORT=([clef]=8021 [pplx12]=8022 [d3]=8023 [linne]=8024 [decider31b]=8025 [amos]=8026)
while [ ! -f logs/exam.stop ]; do
  .venv/bin/python scripts/pb_exam.py build
  for d in "${!PORT[@]}"; do
    if [ "$d" = clef ]; then up=$(curl -s -m 5 -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:8021/decide" -d '{}')
    else up=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT[$d]}/health"); fi
    [ "$up" = 000 ] && continue
    S1CLIN_DECIDER=$d .venv/bin/python scripts/pb_exam.py select --workers 2
  done
  .venv/bin/python scripts/pb_exam.py report > /dev/null
  sleep 120
done
