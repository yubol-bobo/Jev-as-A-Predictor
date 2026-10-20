"""Where a System One decider is reliable in the agent loop, and where it is not (research loop, cycle 5;
aggregates only). On the fixed exam (pb_exam.py, keys from pb_combine.py), per decider D, the outcome of executing
D's argmax instead of the LLM's own sample (`first`) on keyed questions: benefit = bad keys of first - bad keys of D's
pick (positive: D helps), split by
  confidence   D's log-odds of its pick over first (is D's confidence informative? -> gating)
  context      input tokens of D's request, and whether the 32k fitting rule shortened tool results
  position     how far into the trajectory the step is
  llm_agree    how many of the LLM's K = 5 samples equal its first sample (the LLM's self-consistency: where is the
               reasoning model reliable on its own? -> S1 + LLM combination)
Each cell: mean benefit, n questions, and D's and first's error rates. Questions are clustered by task; cells are
descriptive (no CIs) and feed hypotheses, not claims.

  .venv/bin/python scripts/pb_boundary.py [--deciders clef,pplx12,...] [--out results/pb_boundary.json]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from pb_combine import argmax, load  # noqa: E402
from pb_exam import OD  # noqa: E402
from pb_summarize import semantic_key  # noqa: E402


def bins(x, edges):
    for e in edges:
        if x < e:
            return f"<{e:g}"
    return f">={edges[-1]:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deciders", default="clef,pplx12,d3,amos,decider31b,linne")
    ap.add_argument("--out", default=str(REPO / "results/pb_boundary.json"))
    a = ap.parse_args()
    res = {}
    for d in a.deciders.split(","):
        info = {x["id"]: x["info"] for x in map(json.loads, (OD / f"answers_{d}.jsonl").read_text().splitlines())
                if x["pick"] is not None}
        qs = [q for q in load([d]) if q["bad"]]
        n_steps = defaultdict(int)
        for q in qs:
            n_steps[q["task"]] = max(n_steps[q["task"]], q["step"])
        cells = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # benefit sum, n, bad_d, bad_first
        for q in qs:
            p = q["p"][d]
            k, f = argmax(p), q["first"]
            bad_d = sum(b[k] for b in q["bad"].values())
            bad_f = sum(b[f] for b in q["bad"].values())
            lo = math.log(max(p[k], 1e-12)) - math.log(max(p[f], 1e-12))
            it = info[q["id"]]
            sem = [semantic_key(o) for o in q["options"]]
            agree = sum(c for c, sk in zip(q["counts"], sem) if sk == sem[f])  # samples semantically equal to first
            keys = {"all": "all",
                    "confidence": "same pick" if k == f else bins(lo, [0.5, 1.5, 3.0]),
                    "max_p": bins(max(p), [0.3, 0.5, 0.7]),
                    "context": bins((it.get("input_tokens") or 0) / 1000, [8, 16, 24]),
                    "shortened": "shortened" if (it.get("fit") or {}).get("shortened") else "full",
                    "position": bins(q["step"] / max(n_steps[q["task"]], 1), [0.34, 0.67]),
                    "llm_agree": f"{agree}/5"}
            for dim, v in keys.items():
                c = cells[dim][v]
                c[0] += bad_f - bad_d; c[1] += 1; c[2] += bad_d; c[3] += bad_f
                if dim in ("shortened", "position", "llm_agree", "confidence"):  # per key: separates key mix from the split
                    for key, b in q["bad"].items():
                        c = cells[f"{dim}|{key}"][v]
                        c[0] += b[f] - b[k]; c[1] += 1; c[2] += b[k]; c[3] += b[f]
        res[d] = {dim: {v: {"benefit": round(c[0] / c[1], 3), "n": c[1], "err_decider": round(c[2] / c[1], 3),
                            "err_first": round(c[3] / c[1], 3)} for v, c in sorted(vs.items())}
                  for dim, vs in cells.items()}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for d, r in res.items():
        print(f"== {d}")
        for dim, vs in r.items():
            print(f"  {dim:11s}", "  ".join(f"{v}: {c['benefit']:+.2f} (n={c['n']}, D {c['err_decider']:.2f} / LLM {c['err_first']:.2f})"
                                          for v, c in vs.items()))


if __name__ == "__main__":
    main()
