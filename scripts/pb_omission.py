"""Anatomy of the agent's errors (research loop Q1/Q3; aggregates only).

Q1, where a missed required order is lost. For each run with K candidates logged at every step (llm_log_k5), each
required-order spec (pb_offline_orders.required_orders, positive only) that no executed order matches is placed in
the first class that applies:
  noted      - the agent's own note (workspace/output/*.txt) names it (spec name pattern): said, not done
  proposed   - some sampled candidate at some step creates a matching order, but it was never executed
  mentioned  - some candidate's text names it without creating it
  silent     - it never surfaces: no note, no candidate order, no mention
Name patterns are the grader's own regexes, applied to free text, so `noted`/`mentioned` also count negated mentions
("no need for X"); read them as upper bounds. For orders that were placed: the step and the context length (prompt
tokens) at placement, relative to the run's length. Per run: the order phase (first to last step with an order) and
how many distinct order bursts it has (a gap of more than 2 steps without an order starts a new burst).

Q3, how the other checkpoints fail. For every failed Clinical Reasoning / Documentation / Data Retrieval checkpoint of
the LLM runs, the grader's assertion is classified: no_output (no note or empty: "Output file not found or empty"),
judge_partial, judge_fail, other. The grader's reasons go to data/offline/pb_omission/reasons.jsonl (patient
content stays there) for a later local classification.

  .venv/bin/python scripts/pb_omission.py [--runs llm_log_k5/run1,llm_log_k5/run3] [--out results/pb_omission.json]
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
from pb_gap import calls  # noqa: E402
from pb_offline_orders import CREATE, matches, required_orders  # noqa: E402
from pb_summarize import JOBS, TAXONOMY  # noqa: E402
from src.s1clin.pb_decider import turn_orders  # noqa: E402

OD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_omission")
BLOCK = re.compile(r"^_+ test_checkpoint_(\S+) _+$", re.M)
TOOL_LINE = re.compile(r"^\w+\(.*$", re.M)
SNIPPETS = []


def mentions(spec, text):
    return any(re.search(p, text or "", re.IGNORECASE) for p in spec["names"])


def failures(jd):
    """checkpoint -> (category, assertion text) for each failed checkpoint."""
    p = jd / "logs/verifier/pytest_output.txt"
    out = p.read_text() if p.exists() else ""
    sec = out.split("= FAILURES =", 1)[1].split("= PASSES =", 1)[0] if "= FAILURES =" in out else ""
    parts = BLOCK.split(sec)
    res = {}
    for cp, body in zip(parts[1::2], parts[2::2]):
        e = " ".join(line[1:].strip() for line in body.splitlines() if line.startswith("E "))
        cat = ("no_output" if re.search(r"Output file.{0,80}not found|not found or empty", e) else
               "judge_partial" if re.search(r"""['"]score['"]: ['"]PARTIAL""", e) else
               "judge_fail" if re.search(r"""['"]score['"]: ['"]FAIL""", e) else "other")
        res[cp] = (cat, e)
    return res


def anatomy(jd):
    task = jd.name
    specs = [s for s in required_orders(task) if not s["negative"]]
    dec = [json.loads(l) for l in (jd / "logs/agent/decisions.jsonl").read_text().splitlines()]
    executed = [(CREATE[n][0], a.get(CREATE[n][1], ""), str(a.get(CREATE[n][2], "") or "")) for n, a in calls(jd) if n in CREATE]
    notes = "\n".join(p.read_text(errors="ignore") for p in (jd / "workspace/output").glob("*.txt")
                      if not p.name.startswith(".")) if (jd / "workspace/output").exists() else ""
    steps = len(dec)
    order_steps, placed_at = [], {}
    for r in dec:
        chosen = r["candidates"][r.get("pick", 0)]
        os_ = turn_orders(chosen)
        if os_:
            order_steps.append(r["step"])
        for i, s in enumerate(specs):
            if i not in placed_at and any(matches(s, *o) for o in os_):
                placed_at[i] = (r["step"], r.get("prompt_tokens") or 0)
    rows = []
    for i, s in enumerate(specs):
        if any(matches(s, *o) for o in executed):
            st, tok = placed_at.get(i, (None, None))
            rows.append({"cls": "placed", "step": st, "rel": st / steps if st else None, "tok": tok})
            continue
        prop = [r["step"] for r in dec if any(matches(s, *o) for c in r["candidates"] for o in turn_orders(c))]
        ment = [r["step"] for r in dec if any(mentions(s, TOOL_LINE.sub("", c)) for c in r["candidates"])]
        cls = "noted" if mentions(s, notes) else "proposed" if prop else "mentioned" if ment else "silent"
        if cls == "noted":  # the note's words around each mention, for the local classification (data dir only)
            spans = [notes[max(0, m.start() - 400):m.end() + 400] for p in s["names"]
                     for m in re.finditer(p, notes, re.IGNORECASE)][:3]
            SNIPPETS.append({"run": jd.parent.name, "task": task, "cp": s["cp"], "order_names": s["names"], "spans": spans})
        first = (prop or ment or [None])[0]
        rows.append({"cls": cls, "first": first, "rel": first / steps if first else None,
                     "tok": next((r.get("prompt_tokens") or 0 for r in dec if r["step"] == first), None) if first else None})
    bursts, last = 0, None
    for st in order_steps:
        if last is None or st - last > 2:
            bursts += 1
        last = st
    run = {"steps": steps, "orders": len(executed), "bursts": bursts,
           "order_phase": [order_steps[0] / steps, order_steps[-1] / steps] if order_steps else None,
           "final_tokens": dec[-1].get("prompt_tokens") if dec else None}
    return rows, run


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="llm_log_k5/run1,llm_log_k5/run3")
    ap.add_argument("--fail-runs", default="llm/run1,llm/run2,llm_log_k5/run1,llm_log_k5/run3",
                    help="runs for the Q3 failure classification")
    ap.add_argument("--out", default=str(REPO / "results/pb_omission.json"))
    a = ap.parse_args()
    os.umask(0o077)
    OD.mkdir(parents=True, exist_ok=True, mode=0o700)
    res = {"q1": {}, "q3": {}}

    for slug in a.runs.split(","):
        rows, runs = [], []
        for jd in sorted(p for p in (JOBS / slug).iterdir() if p.is_dir()):
            if not (jd / "metadata.json").exists() or not (jd / "logs/agent/decisions.jsonl").exists():
                continue
            r, run = anatomy(jd)
            rows += r
            runs.append(run)
        cls = Counter(r["cls"] for r in rows)
        missed = [r for r in rows if r["cls"] != "placed"]
        res["q1"][slug] = {
            "tasks": len(runs), "specs": len(rows), "classes": dict(cls),
            "missed_share": {k: round(v / len(missed), 3) for k, v in cls.items() if k != "placed"} if missed else {},
            "placed_rel_step_mean": mean(r["rel"] for r in rows if r["cls"] == "placed"),
            "placed_tokens_mean": mean(r["tok"] for r in rows if r["cls"] == "placed"),
            "missed_first_rel_step_mean": {k: mean(r["rel"] for r in missed if r["cls"] == k) for k in ("noted", "proposed", "mentioned")},
            "missed_first_tokens_mean": {k: mean(r["tok"] for r in missed if r["cls"] == k) for k in ("proposed", "mentioned")},
            "runs": {"steps_mean": mean(x["steps"] for x in runs), "orders_mean": mean(x["orders"] for x in runs),
                     "bursts": dict(Counter(x["bursts"] for x in runs)),
                     "order_phase_start_mean": mean(x["order_phase"][0] for x in runs if x["order_phase"]),
                     "order_phase_end_mean": mean(x["order_phase"][1] for x in runs if x["order_phase"]),
                     "final_tokens_mean": mean(x["final_tokens"] for x in runs)}}

    by = defaultdict(Counter)
    with open(OD / "reasons.jsonl", "w") as f:
        for slug in a.fail_runs.split(","):
            for jd in sorted(p for p in (JOBS / slug).iterdir() if p.is_dir()):
                if not (jd / "metadata.json").exists():
                    continue
                tax = TAXONOMY.get(jd.name, {})
                for cp, (cat, e) in failures(jd).items():
                    typ = tax.get(f"cp{cp}" if not cp.startswith("cp") else cp) or tax.get(cp) or "unknown"
                    by[typ][cat] += 1
                    f.write(json.dumps({"run": slug, "task": jd.name, "cp": cp, "type": typ, "cat": cat, "assert": e[:3000]},
                                       ensure_ascii=False) + "\n")
    res["q3"] = {t: dict(c) for t, c in by.items()}
    with open(OD / "noted_snippets.jsonl", "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in SNIPPETS)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
