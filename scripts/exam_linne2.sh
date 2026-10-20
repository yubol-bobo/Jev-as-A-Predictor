#!/bin/bash
# linne answers about 165 questions an hour on two GPUs, too slow to finish the exam in one allocation. Once the
# lanes "0:pplx12,decider31b" and "3:d3,amos" of exam_finish.sh are done (answered, or failed to start), a second
# linne instance on GPUs 0+3 answers from the other end of the list (pb_exam.py select --reverse); both instances
# skip what the other has answered.
#   bash scripts/exam_linne2.sh
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
umask 077
done_lane() { grep -qE "$1 (answered the exam|has answered the exam|failed to start)" logs/exam_finish.log; }
answered() {
  .venv/bin/python - <<'PY'
import json, sys
od = "/ocean/projects/cis260032p/yliy/data/offline/pb_exam"
qs = {json.loads(l)["id"] for l in open(f"{od}/questions.jsonl")}
done = {x["id"] for x in map(json.loads, open(f"{od}/answers_linne.jsonl"))
        if x["pick"] is not None or "does not fit" in (x["info"].get("error") or "")}
sys.exit(0 if qs <= done else 1)
PY
}
until done_lane decider31b && done_lane amos; do sleep 60; done
for g in 0 3; do until [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done; done
bash scripts/s1_up.sh linne 0,3 2 > logs/s1_up_linne_2.log 2>&1 || { echo "$(date) linne instance 2 failed"; exit 1; }
echo "$(date) linne instance 2 up on GPUs 0,3"
kill "$(cat logs/exam_select_linne.pid)" 2>/dev/null  # exam_deciders.sh restarts it with the code that skips
until answered; do
  S1CLIN_DECIDER=linne S1CLIN_DECIDER_URL=http://127.0.0.1:8124/decide \
    .venv/bin/python scripts/pb_exam.py select --workers 1 --reverse
  sleep 30
done
echo "$(date) linne has answered the exam"
kill -TERM -- -"$(cat logs/s1_linne_2.pgid)" 2>/dev/null; kill "$(cat logs/s1_linne_2_proxy.pid)" 2>/dev/null
