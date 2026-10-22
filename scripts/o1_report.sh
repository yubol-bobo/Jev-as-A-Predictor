#!/bin/bash
# O1 (docs/research_loop.md): outcomes of the two checklist arms vs llm_log_k5/run3 and vs each other (pb_gap.py),
# and what the checklist showed and what the LLM did with it (pb_p7.py). Aggregates only.
set -euo pipefail
cd /ocean/projects/cis260032p/yliy/research/Jev-as-A-Predictor
A=screen_prior-pplx12-dedup_b0m8_k5/run1; B=screen_prior-random-dedup_b0m8_k5/run1; R=llm_log_k5/run3
.venv/bin/python scripts/pb_gap.py --decider-runs $A --llm-runs $R --out results/pb_gap_o1_pplx12_vs_run3.json > /dev/null
.venv/bin/python scripts/pb_gap.py --decider-runs $B --llm-runs $R --out results/pb_gap_o1_random_vs_run3.json > /dev/null
.venv/bin/python scripts/pb_gap.py --decider-runs $A --llm-runs $B --out results/pb_gap_o1_pplx12_vs_random.json > /dev/null
.venv/bin/python scripts/pb_p7.py --arms $A,$B --out results/pb_o1_mechanism.json > /dev/null
for f in pplx12_vs_run3 random_vs_run3 pplx12_vs_random; do
  .venv/bin/python - "results/pb_gap_o1_$f.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
t, c = d["trajectory"], d["checkpoints"]
print(sys.argv[1].split("pb_gap_o1_")[1][:-5], "tasks", d["paired_tasks"],
      "| req-order ckpt", c["AE: required order"]["diff_ci"], "| pass", d["task_pass"]["diff_ci"],
      "| orders", t["orders"]["diff_ci"], "| unmatched", t["unmatched_orders"]["diff_ci"],
      "| forbidden", t["forbidden_placed"]["llm"], "->", t["forbidden_placed"]["decider"])
PY
done
