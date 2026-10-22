"""Local classification of the error anatomy texts (research loop A5, A6; Qwen without thinking, temperature 0;
patient text goes only to the local model; aggregates only in results/).

A5: for each missed required order that the agent's own note names (pb_omission.py class `noted`), how the note
names it: RECOMMENDS (plans or recommends placing it as part of this task), CLAIMS_DONE (says it was ordered or
placed), CONDITIONAL (only if / consider / later / defers to another clinician), NEGATED (not indicated, not needed,
declined), OTHER (a past result, history, or other context).
A6: for each failed Clinical Reasoning / Documentation / Data Retrieval checkpoint (pb_omission.py reasons.jsonl),
what the grader's reason says: NO_NOTE (no output file or empty), MISSING (the note does not address the required
element), WRONG (an incorrect value, interpretation or decision), HEDGED (non-committal or conditional where a
decision was required), INCOMPLETE (addressed but a required detail is missing: threshold, dose, timing, specific
test), OTHER.

  .venv/bin/python scripts/pb_classify.py noted|reasons [--url http://127.0.0.1:8011/v1] [--workers 16]
  .venv/bin/python scripts/pb_classify.py report [--out results/pb_classify.json]
"""

import argparse
import json
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]
OD = Path("/ocean/projects/cis260032p/yliy/data/offline/pb_omission")
NOTED = ["RECOMMENDS", "CLAIMS_DONE", "CONDITIONAL", "NEGATED", "OTHER"]
REASONS = ["NO_NOTE", "MISSING", "WRONG", "HEDGED", "INCOMPLETE", "OTHER"]
NOTED_PROMPT = """An AI agent working on a clinical task wrote a note. The order below was required by the task but the agent never placed it. The excerpts are the parts of the agent's note that mention it.

Order (name patterns): {names}

Note excerpts:
{spans}

How does the note refer to this order? Answer with one category:
RECOMMENDS - the note plans or recommends placing this order as part of the current task
CLAIMS_DONE - the note says this order was ordered or placed
CONDITIONAL - only if something, consider, at a later visit, or deferred to another clinician
NEGATED - not indicated, not needed, or declined
OTHER - a past result, history, or other context, not a plan for this order
Reply with JSON only: {{"category": "..."}}"""
REASON_PROMPT = """A grader checked an AI agent's work on a clinical task and the checkpoint below failed. The text is the grader's failure message.

Checkpoint: {cp} ({type})
Grader's failure message:
{msg}

What kind of failure is it? Answer with one category:
NO_NOTE - no output file, or it is empty
MISSING - the note does not address the required element at all
WRONG - the note states an incorrect value, interpretation, or decision
HEDGED - the note is non-committal or conditional where a decision was required
INCOMPLETE - the note addresses it but a required detail is missing (threshold, dose, timing, specific test)
OTHER - none of these
Reply with JSON only: {{"category": "..."}}"""


def ask(url, prompt, labels):
    r = requests.post(f"{url}/chat/completions", timeout=600, json={
        "model": "qwen3.8-27b", "temperature": 0, "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    m = re.search(r'"category"\s*:\s*"([A-Z_]+)"', r.json()["choices"][0]["message"]["content"] or "")
    return m[1] if m and m[1] in labels else "UNPARSED"


def classify(kind, url, workers):
    src = OD / ("noted_snippets.jsonl" if kind == "noted" else "reasons.jsonl")
    rows = [json.loads(l) for l in src.read_text().splitlines()]
    if kind == "reasons":
        rows = [r for r in rows if r["type"] in ("Clinical Reasoning", "Documentation", "Data Retrieval")]
    out, lock = OD / f"classified_{kind}.jsonl", threading.Lock()
    open(out, "w").close()

    def one(r):
        if kind == "noted":
            cat = ask(url, NOTED_PROMPT.format(names=", ".join(r["order_names"]), spans="\n---\n".join(r["spans"])), NOTED)
        elif r["cat"] == "no_output":
            cat = "NO_NOTE"
        else:
            cat = ask(url, REASON_PROMPT.format(cp=r["cp"], type=r["type"], msg=r["assert"][:2500]), REASONS)
        with lock, open(out, "a") as f:
            f.write(json.dumps({**{k: r[k] for k in ("run", "task", "cp") if k in r}, "type": r.get("type", "noted"),
                                "category": cat}) + "\n")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, rows))
    print(out.name, len(rows))


def report(path):
    res = {}
    for kind in ("noted", "reasons"):
        p = OD / f"classified_{kind}.jsonl"
        if not p.exists():
            continue
        by = defaultdict(Counter)
        for r in map(json.loads, p.read_text().splitlines()):
            by[r["type"]][r["category"]] += 1
        res[kind] = {t: {"n": sum(c.values()), "share": {k: round(v / sum(c.values()), 3) for k, v in c.most_common()}}
                     for t, c in by.items()}
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["noted", "reasons", "report"])
    ap.add_argument("--url", default="http://127.0.0.1:8011/v1")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(REPO / "results/pb_classify.json"))
    a = ap.parse_args()
    os.umask(0o077)
    if a.cmd == "report":
        report(a.out)
    else:
        classify(a.cmd, a.url, a.workers)


if __name__ == "__main__":
    main()
