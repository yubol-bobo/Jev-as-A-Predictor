#!/bin/bash
# Once no orphaned llm / llm_tool task processes remain (their drivers were stopped to relieve the Qwen replicas),
# restart the llm_tool control on run 1 with 4 workers, next to the decider arms (scripts/run_deciders.sh).
cd /ocean/projects/cis260032p/yliy/research/S1-Clinical-Agents
umask 077
busy() { ps -u "$USER" -o args= | awk '$3=="src.s1clin.pb_run" && $4=="--one" {for (i=1;i<=NF;i++) if ($i=="--arm") print $(i+1)}' | grep -cxE "llm|llm_tool"; }
while [ "$(busy)" -gt 0 ]; do sleep 30; done
echo "$(date) orphans done; llm_tool restarts"
POOL="http://127.0.0.1:8011/v1*5,http://127.0.0.1:8013/v1*1"
export S1CLIN_LLM_SERVER="scripts/vllm_qwen_up.sh (bf16, 128k, decode CUDA graphs, MTP-2); 8011 = TP2 on 2 GPUs, 8013 = TP1"
.venv/bin/python -m src.s1clin.pb_run --arm llm_tool --tasks all --run 1 --k 5 --workers 4 --port-base 19000 \
  --llm-url "$POOL" >> "logs/full_pb_llm_tool_run1.$(hostname -s).log" 2>&1
echo "$(date) llm_tool run 1 finished"
