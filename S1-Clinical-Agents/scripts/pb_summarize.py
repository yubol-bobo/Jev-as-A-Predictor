"""Summarise PhysicianBench decider runs (docs/physicianbench_decider.md §5). Aggregates only; no patient content.

  .venv/bin/python scripts/pb_summarize.py [--tasks a,b,c] [--out results/pb_summary.json]

Per arm: task pass@1 per run and mean, checkpoint pass rate by type, #turns, tokens, wall time; for decider arms,
the share of steps with a real choice, majority agreement, confidence/entropy and decider time.
Only tasks finished by every arm are compared (paired).
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get("PB_JOBS", "/ocean/projects/cis260032p/yliy/data/physicianbench/jobs"))
TAXONOMY = json.loads((REPO / "third_party/physicianbench/scripts/checkpoint_taxonomy_v1.json").read_text())
LINE = re.compile(r"^(PASSED|FAILED|ERROR) \S*::test_checkpoint_(\S+)", re.M)


def load_job(jd: Path):
    md = jd / "metadata.json"
    if not md.exists():
        return None
    meta = json.loads(md.read_text())
    out = (jd / "logs/verifier/pytest_output.txt").read_text() if (jd / "logs/verifier/pytest_output.txt").exists() else ""
    cps = {cp: status == "PASSED" for status, cp in LINE.findall(out)}
    traj = jd / "logs/agent/trajectory.log"
    turns = ptok = ctok = 0
    if traj.exists():
        for l in traj.read_text().splitlines():
            e = json.loads(l)
            if e["type"] == "tool_call":
                turns += 1
            elif e["type"] == "llm_response":
                ptok += e["metadata"].get("prompt_tokens") or 0
                ctok += e["metadata"].get("completion_tokens") or 0
    dec = []
    if (jd / "logs/agent/decisions.jsonl").exists():
        dec = [json.loads(l) for l in (jd / "logs/agent/decisions.jsonl").read_text().splitlines()]
    tr = meta.get("test_results", {})
    return {"pass": bool(tr.get("total")) and tr.get("failed", 1) == 0 and tr.get("passed") == tr.get("total"),
            "cps": cps, "turns": turns, "prompt_tokens": ptok, "completion_tokens": ctok,
            "wall": meta.get("wall_s", {}), "decisions": dec}


ARG = re.compile(r'(?:medication_display|code_display|display|code|category|file_path|path)="([^"]+)"')


def semantic_key(c):
    """Coarse identity of a candidate: tool names plus the key clinical arguments, case-folded.
    Candidates that differ only in argument order, casing or incidental fields count as one."""
    names = re.findall(r"^(\w+)\(", c, re.M) or ["FINISH"]
    return tuple(sorted(names)), tuple(sorted({v.lower() for v in ARG.findall(c)}))


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", default=str(REPO / "results/pb_summary.json"))
    a = ap.parse_args()
    arms = sorted(p.name for p in JOBS.iterdir() if p.is_dir()) if JOBS.exists() else []
    jobs = {arm: {run.name: {t.name: load_job(t) for t in sorted(run.iterdir()) if t.is_dir()}
                  for run in sorted((JOBS / arm).iterdir()) if run.is_dir()} for arm in arms}
    for arm in jobs:
        for run in jobs[arm]:
            jobs[arm][run] = {t: j for t, j in jobs[arm][run].items() if j is not None}
    want = set(a.tasks.split(",")) if a.tasks else None
    common = None
    for arm in jobs:
        for run, tj in jobs[arm].items():
            s = set(tj) if want is None else set(tj) & want
            common = s if common is None else common & s
    common = sorted(common or [])
    summary = {"tasks": common, "n_tasks": len(common), "arms": {}}
    for arm in jobs:
        runs = {}
        for run, tj in jobs[arm].items():
            js = [tj[t] for t in common]
            by_type = defaultdict(lambda: [0, 0])
            for t, j in zip(common, js):
                for cp, ok in j["cps"].items():
                    typ = TAXONOMY.get(t, {}).get(cp, "?")
                    by_type[typ][0] += ok
                    by_type[typ][1] += 1
            n_cp = sum(v[1] for v in by_type.values())
            dec = [d for j in js for d in j["decisions"]]
            real = [d for d in dec if d["n_unique"] > 1 and d.get("distribution")]
            r = {"pass@1": sum(j["pass"] for j in js) / max(len(js), 1),
                 "checkpoint_pass": sum(v[0] for v in by_type.values()) / max(n_cp, 1),
                 "checkpoint_pass_by_type": {k: round(v[0] / v[1], 3) for k, v in sorted(by_type.items())},
                 "mean_turns": sum(j["turns"] for j in js) / max(len(js), 1),
                 "mean_completion_tokens": sum(j["completion_tokens"] for j in js) / max(len(js), 1),
                 "mean_agent_wall_s": sum(j["wall"].get("agent", 0) for j in js) / max(len(js), 1),
                 "per_task_pass": {t: j["pass"] for t, j in zip(common, js)}}
            if dec:
                # steps where some candidate was sampled more than once: did the decider pick the mode?
                moded = [d for d in real if sorted(d["counts"])[-1] > ([0] + sorted(d["counts"]))[-2]]
                maj = [d for d in moded if d["counts"][d["pick"]] == max(d["counts"])]
                sem_real = [d for d in real if len({semantic_key(c) for c in d["candidates"]}) > 1]
                r["decisions"] = {
                    "steps": len(dec), "real_choices": len(real),
                    "share_real": round(len(real) / len(dec), 3),
                    "mean_unique": round(sum(d["n_unique"] for d in dec) / len(dec), 2),
                    "semantic_real_choices": len(sem_real),
                    "share_semantic_real": round(len(sem_real) / len(dec), 3),
                    "steps_with_unique_mode": len(moded),
                    "pick_is_mode": round(len(maj) / max(len(moded), 1), 3),
                    "mean_max_p": round(sum(max(d["distribution"]) for d in real) / max(len(real), 1), 3),
                    "mean_entropy_norm": round(sum(entropy(d["distribution"]) / math.log(d["n_unique"])
                                                   for d in real) / max(len(real), 1), 3),
                    "decider_errors": sum(1 for d in dec if d["info"].get("error")),
                    "mean_decide_s": round(sum(d["decide_s"] for d in real) / max(len(real), 1), 2),
                    "fit_shortened_steps": sum(1 for d in real if (d["info"].get("fit") or {}).get("shortened")),
                    "fit_omitted_steps": sum(1 for d in real if (d["info"].get("fit") or {}).get("omitted")),
                }
            runs[run] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        summary["arms"][arm] = runs
    # progress view: every arm on its own finished tasks (unpaired; for monitoring only, not for comparison)
    summary["progress"] = {}
    for arm in jobs:
        for run, tj in jobs[arm].items():
            js = list(tj.values())
            cps = [ok for j in js for ok in j["cps"].values()]
            summary["progress"][f"{arm}/{run}"] = {
                "finished": len(js), "passed": sum(j["pass"] for j in js),
                "checkpoint_pass": round(sum(cps) / max(len(cps), 1), 3)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
