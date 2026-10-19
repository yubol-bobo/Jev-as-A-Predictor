#!/bin/bash
# Same-state comparison of System One deciders: whenever one of them is up (scripts/run_deciders.sh brings them up in
# turn), replay it once on the order steps of Clef's online S1-Select run (pb_offline_orders.py select, with the
# state and options-only), next to the online arms. A replay costs about 170 decisions on the decider GPU, which the
# online arms leave half idle.
#   bash scripts/offline_deciders.sh        (loops until every decider has been replayed)
cd /ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
umask 077
declare -A PORT=([pplx12]=8022 [d3]=8023 [linne]=8024 [decider31b]=8025 [amos]=8026)
SEL=/ocean/projects/cis260032p/yliy/data/offline/pb_clef_k5_run1/selections.jsonl
while true; do
  left=0
  for d in "${!PORT[@]}"; do
    n=$(grep -c "\"decider\": \"$d\"" "$SEL" 2>/dev/null || true)
    [ "${n:-0}" -ge 70 ] && continue
    left=1
    curl -s -m 5 "http://127.0.0.1:${PORT[$d]}/health" > /dev/null || continue
    echo "$(date +%T) replaying $d"
    S1CLIN_DECIDER=$d .venv/bin/python scripts/pb_offline_orders.py select --slug clef_k5/run1 --workers 2
  done
  [ $left = 0 ] && break
  sleep 120
done
echo "$(date +%T) all deciders replayed"
