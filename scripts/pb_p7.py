"""P7' mechanism: what the pre-finish screen flagged and what the LLM did with it (aggregates only).

For each finished task of a screen arm: the screen record of the first FINISH (pool size, the M flagged orders, the
LLM's answer), the flagged orders that match a required-order grader (pb_offline_orders.required_orders), and the
orders placed in the answer. Outcome comparisons are pb_gap.py's job; this explains them.

  .venv/bin/python scripts/pb_p7.py [--arms screen_freq_b8m3_k5,screen_pplx12-task_b8m3_k5] [--out results/pb_p7_mechanism.json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
from pb_offline_orders import matches, required_orders  # noqa: E402
from pb_summarize import JOBS  # noqa: E402
from src.s1clin.pb_decider import turn_orders  # noqa: E402

FLAG = re.compile(r"^(MedicationRequest|ServiceRequest): (.*?)(?: \(code (.*)\))?$")


def is_required(o, specs):
    return any(matches(s, o[0], o[1], o[2]) for s in specs)


def arm(slug):
    rows = []
    for jd in sorted(p for p in (JOBS / slug).iterdir() if p.is_dir()):
        dec = jd / "logs/agent/decisions.jsonl"
        if not (jd / "metadata.json").exists() or not dec.exists():
            continue
        specs = [s for s in required_orders(jd.name) if not s["negative"]]
        screen = next((r["screen"] for r in map(json.loads, dec.read_text().splitlines()) if "screen" in r), None)
        row = {"task": jd.name, "screened": screen is not None and "error" not in screen}
        if row["screened"]:
            flagged = []
            for f in screen.get("flagged", []):
                m = FLAG.match(f["order"])
                if m:
                    flagged.append(("medication" if m[1] == "MedicationRequest" else "service", m[2], m[3] or ""))
            placed = turn_orders(screen.get("answer", ""))
            row.update(pool=screen.get("pool", 0), flagged=len(flagged), s=screen.get("s"),
                       flagged_required=sum(is_required(o, specs) for o in flagged),
                       answer_orders=len(placed), answer_required=sum(is_required(o, specs) for o in placed))
        rows.append(row)
    sc = [r for r in rows if r["screened"]]
    mean = lambda k: round(sum(r[k] for r in sc) / len(sc), 3) if sc else None  # noqa: E731
    return {"tasks": len(rows), "screened": len(sc), "pool": mean("pool"), "flagged": mean("flagged"),
            "flagged_required": mean("flagged_required"), "answer_orders": mean("answer_orders"),
            "answer_required": mean("answer_required"), "screen_s": mean("s"),
            "tasks_with_flagged_required": sum(r["flagged_required"] > 0 for r in sc),
            "tasks_with_answer_required": sum(r["answer_required"] > 0 for r in sc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="screen_freq_b8m3_k5/run1,screen_pplx12-task_b8m3_k5/run1")
    ap.add_argument("--out", default=str(REPO / "results/pb_p7_mechanism.json"))
    a = ap.parse_args()
    res = {s: arm(s) for s in a.arms.split(",")}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
