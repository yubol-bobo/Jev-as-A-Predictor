#!/bin/bash
# Resume after the w002 allocation (2026-10-19 15:50, w008): finish PhysicianBench run 1 (s1_tool, llm_tool control to
# 100 tasks), then fill the spare generation capacity with llm runs 2-3. About 22 tasks at a time keeps the prefix
# cache healthy (docs/full_experiment.md section 6). Clef runs 2-3 wait: GPU 3 goes to the other System One deciders.
#   bash scripts/run_pb_resume.sh
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
( pb s1_tool 3 19200 1; pb llm 3 19200 2 ) &
( pb llm_tool 14 19000 1; pb llm 5 19000 3 ) &
( pb llm 5 19400 2 ) &
wait
echo "resume streams finished $(date)"
