"""G7 (docs/research_loop.md): does a debiasing instruction undo the anchoring? Paired by task, unplaced AUROC.
System One models: share of the retrieval-only gain over the state that the warned state recovers. Qwen: on the
questions its state, task-only and warned runs all answered, the share of the task-only gain.
  .venv/bin/python scripts/pb_g7.py   -> results/pb_g7.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from pb_screen import OD, _answers, auroc  # noqa: E402


def boot(d, n=2000):
    rng = random.Random(0)
    b = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(b[50], 4), round(b[1949], 4)]


def compare(base, ref, arm, ids, qs, scores):
    """Per-task AUROC on `ids`, paired arm - base, and the share of (ref - base) the arm recovers."""
    by = defaultdict(list)
    for i in ids:
        by[qs[i]["task"]].append(i)
    per = {n: {} for n in (base, ref, arm)}
    for t, ii in by.items():
        for n in per:
            x = auroc([scores[n][i] for i in ii if qs[i]["label"] == "required"],
                      [scores[n][i] for i in ii if qs[i]["label"] is None])
            if x is not None:
                per[n][t] = x
    ts = [t for t in per[base] if all(t in per[n] for n in per)]
    mean = {n: round(sum(per[n][t] for t in ts) / len(ts), 4) for n in per}
    d = [per[arm][t] - per[base][t] for t in ts]
    gain = sum(per[ref][t] - per[base][t] for t in ts)
    return {"questions": len(ids), "tasks": len(ts), "auroc": mean,
            f"{arm}_minus_{base}": {"delta": round(sum(d) / len(d), 4), "ci95": boot(d)},
            "share_of_gain": round(sum(d) / gain, 3) if gain else None}


def main():
    qs = {q["id"]: q for q in map(json.loads, (OD / "questions.jsonl").read_text().splitlines())}
    res = {}
    for m in ("pplx12", "d3", "amos", "llm"):
        ref = m + ("_task" if m == "llm" else "_retr")
        names = (m, ref, m + "_warn")
        scores = {n: {i: a["p"] for i, a in _answers(n)[1].items()} for n in names}
        if not scores[m + "_warn"]:
            continue
        ids = [i for i in scores[m + "_warn"] if all(i in scores[n] for n in names)
               and not qs[i]["placed"] and qs[i]["label"] != "forbidden"]
        res[m] = compare(m, ref, m + "_warn", ids, qs, scores)
        res[m]["valid_warn"] = len(scores[m + "_warn"])
    (REPO / "results/pb_g7.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
