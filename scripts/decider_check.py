"""Quick health of the decider arms: decisions logged, errors, latency, confidence (aggregates only)."""
import glob
import json
import statistics as st
import sys

J = "/ocean/projects/cis260032p/yliy/data/physicianbench/jobs"
for d in sys.argv[1:]:
    for tag in (f"clef_k5__{d}", f"s1_tool__{d}"):
        rows = [json.loads(l) for f in glob.glob(f"{J}/{tag}/run1/*/logs/agent/decisions.jsonl") for l in open(f)]
        real = [r for r in rows if r.get("distribution")]
        errs = [r for r in rows if (r.get("info") or {}).get("error")]
        done = len(glob.glob(f"{J}/{tag}/run1/*/metadata.json"))
        line = f"{tag}: tasks done {done}, decisions {len(rows)}, scored {len(real)}, errors {len(errs)}"
        if real:
            line += (f", decide_s median {st.median(r['decide_s'] for r in real):.2f}, mean max p "
                     f"{st.mean(max(r['distribution']) for r in real):.2f}, median input tokens "
                     f"{st.median((r.get('info') or {}).get('input_tokens') or 0 for r in real):.0f}")
        print(line)
        if errs:
            print("   first error:", str(errs[0]["info"]["error"])[:300])
