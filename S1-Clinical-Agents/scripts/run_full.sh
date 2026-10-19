#!/bin/bash
# Full experiment driver (docs/full_experiment.md): LLM vs System One as the decision maker in clinical agent loops.
# Assumes the servers are up on this node (scripts/slurm/full.sbatch layout): Qwen3.8-27B TP=2 on GPUs 0,1 (port
# 8011), Qwen3.8-27B TP=1 on GPU 2 (port 8013), Clef on GPU 3 (port 8021). Every arm uses both Qwen replicas; items
# are routed by name with weights ~ KV-cache capacity (src/s1clin/claim.py), so the replica is independent of the arm.
#
# Online arms: llm (baseline), clef (Clef picks among K=5 whole turns), s1_tool (Clef picks the tool, the LLM writes
# the call), llm_log (= llm, logging K=5 candidates and the history for the offline decision evaluation). llm_select
# is evaluated offline only (logs/disabled_arms).
# One stream per (benchmark, arm) runs its own work in priority order (run 1 first), then helps the others. Items are
# claimed before they run, so streams on this node and drivers on other SLURM nodes share the work without overlap;
# every step is resumable, and a job that hits its time limit is resumed by the next job in the chain.
#   bash scripts/run_full.sh              all streams + the progress-commit loop
#   bash scripts/run_full.sh s_ce_log ... only the named streams (no commit loop)
set -uo pipefail
REPO=/ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
cd "$REPO"
umask 077
PY=.venv/bin/python
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
export S1CLIN_LLM_SERVER="scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2); 8011 = TP2 on 2 GPUs, 8013 = TP1"
H=$(hostname -s)

pb() {  # arm workers portbase run
  $PY -m src.s1clin.pb_run --arm "$1" --tasks all --run "$4" --k 5 --workers "$2" --port-base "$3" \
    --llm-url "$POOL" >> "logs/full_pb_$1_run$4.$H.log" 2>&1
}
ce() {  # arm workers run [temperature]
  $PY -m src.s1clin.clinenv_run --arm "$1" --run "$3" --k 5 --workers "$2" --llm-url "$POOL" \
    --temperature "${4:-default}" >> "logs/full_ce_$1_run$3${4:+_t$4}.$H.log" 2>&1
}
help_pb() { for r in 1 2 3; do pb s1_tool "$1" "$2" $r; pb clef "$1" "$2" $r; done; }  # workers portbase
help_ce() { ce llm_log "$1" 1; ce s1_tool "$1" 1; ce clef "$1" 1; }

s_pb_llm()  { for r in 1 2 3; do pb llm 3 18100 $r; done; help_pb 3 18100; }
s_ce_llm()  { ce llm 4 1; ce llm 4 1 0; help_ce 4; }
s_pb_clef() { for r in 1 2 3; do pb clef 4 18200 $r; done; help_pb 4 18200; }
s_ce_clef() { ce clef 4 1; help_ce 4; }
s_pb_log()  { pb llm_log 3 18800 1; help_pb 3 18800; }
s_ce_log()  { ce llm_log 4 1; help_ce 4; }
s_pb_tool() { for r in 1 2 3; do pb s1_tool 3 18900 $r; done; help_pb 3 18900; }
s_ce_tool() { ce s1_tool 4 1; help_ce 4; }
ALL="s_pb_llm s_ce_llm s_pb_clef s_ce_clef s_pb_log s_ce_log s_pb_tool s_ce_tool"

STREAMS=${*:-$ALL}
if [ $# -eq 0 ]; then
  rm -f logs/full.stop
  bash scripts/progress_commit.sh 1200 >> logs/full_commits.log 2>&1 &
  COMMIT_LOOP=$!
fi
PIDS=()
for s in $STREAMS; do $s & PIDS+=($!); done
wait "${PIDS[@]}"
if [ $# -eq 0 ]; then
  touch logs/full.stop; kill $COMMIT_LOOP 2>/dev/null
  bash scripts/progress_commit.sh 0 once
fi
echo "streams finished: $STREAMS $(date)"
