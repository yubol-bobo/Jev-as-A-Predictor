"""Profile of PhysicianBench: the public task definitions (third_party/physicianbench/tasks/v1) and, with --fhir, the
EHR as resource counts only (FHIR `_summary=count`; no records are read).

  .venv/bin/python -I scripts/pb_data_profile.py --out results/data_profile/pb_profile.json [--fhir http://127.0.0.1:18999/fhir]
"""

import argparse
import json
import re
import statistics as S
import tomllib
from collections import Counter
from pathlib import Path

import requests

TASKS = Path(__file__).resolve().parents[1] / "third_party" / "physicianbench" / "tasks" / "v1"
RESOURCES = ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "ServiceRequest", "Procedure",
             "DocumentReference", "DiagnosticReport", "AllergyIntolerance", "Immunization", "Practitioner",
             "Communication", "Appointment"]


def cp_role(name, grader):
    n = name.lower()
    if "retriev" in n or "data_" in n:
        return "data_retrieval"
    if "document" in n or "note" in n or "letter" in n:
        return "documentation"
    if re.search(r"order|referral|refer|prescri|start|stop|discontinu|message|appointment|schedul", n) \
            or "fhir" in grader.lower():
        return "action"
    return "reasoning"


def checkpoints(test_file):
    src = test_file.read_text()
    out = []
    for m in re.finditer(r'def test_checkpoint_(\w+)\(\):\s*"""(.*?)"""', src, re.S):
        doc = m.group(2)
        g = re.search(r"Grader:\s*(.+)", doc)
        d = re.search(r"Determinism:\s*(\S+)", doc)
        grader = g.group(1).strip() if g else ""
        out.append({"name": m.group(1), "grader": grader, "determinism": d.group(1) if d else "?",
                    "role": cp_role(m.group(1), grader)})
    return out


def dist(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": round(S.mean(xs), 2), "min": xs[0], "median": q(.5), "max": xs[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fhir", default=None)
    a = ap.parse_args()
    spec, cat, n_cp, det, role, role_task, deliv, words = Counter(), Counter(), [], Counter(), Counter(), Counter(), [], []
    grader_kind, actions_per_task = Counter(), []
    for t in sorted(p for p in TASKS.iterdir() if p.is_dir()):
        tags = tomllib.loads((t / "task.toml").read_text()).get("metadata", {}).get("tags", [])
        spec[tags[0] if tags else "?"] += 1
        for x in tags[1:]:
            cat[x] += 1
        cps = checkpoints(t / "tests" / "test_outputs.py")
        n_cp.append(len(cps))
        for c in cps:
            det[c["determinism"]] += 1
            role[c["role"]] += 1
            grader_kind["llm_judge" if "llm" in c["grader"].lower() else
                        "code+llm" if "hybrid" in c["grader"].lower() else "code"] += 1
        for r in set(c["role"] for c in cps):
            role_task[r] += 1
        actions_per_task.append(sum(c["role"] == "action" for c in cps))
        ins = (t / "instruction.md").read_text()
        words.append(len(ins.split()))
        deliv.append(len(re.findall(r"`/workspace/output/[^`]+`", ins)))
    res = {"tasks": len(n_cp), "checkpoints": sum(n_cp), "checkpoints_per_task": dist(n_cp),
           "specialties": dict(spec.most_common()), "task_categories": dict(cat.most_common()),
           "checkpoint_determinism": dict(det), "checkpoint_grader": dict(grader_kind),
           "checkpoint_role": dict(role), "tasks_with_role": dict(role_task),
           "action_checkpoints_per_task": dist(actions_per_task),
           "instruction_words": dist(words), "file_deliverables_per_task": dist(deliv)}
    if a.fhir:
        res["fhir_resource_counts"] = {
            r: requests.get(f"{a.fhir}/{r}", params={"_summary": "count"}, timeout=60).json().get("total")
            for r in RESOURCES}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
