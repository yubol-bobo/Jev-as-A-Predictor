"""Aggregate statistics for ClinEnv cases built locally, vs. paper Table 4.

Reads   {cases_dir}/manifest.jsonl, {cases_dir}/{sid}/{hid}/case/case.json and
        {cases_dir}/{sid}/{hid}/plan/planner_stats.json
Prints  counts only (no patient text), suitable for committing as a summary.

Usage:  python -m src.s1clin.clinenv_case_stats --cases-dir DIR [--manifest M] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

PAPER = {  # ClinEnv paper Table 4 / Sec. 4
    "admissions": 3509, "stages": 9297, "decisions": 26043,
    "stages_per_case": 2.65, "frac_cases_ge3_stages": 0.498,
    "decisions_per_stage": 26043 / 9297,
    "type_mix": {"diagnosis": 0.717, "medication": 0.214, "procedure": 0.069},
    "med_action_mix": {"start": 0.791, "stop": 0.075, "switch": 0.038, "adjust": 0.095},
}


def collect(cases_dir: Path, manifest: Path | None = None) -> dict:
    ids = []
    if manifest and manifest.exists():
        ids = [(r["subject_id"], r["hadm_id"]) for r in map(json.loads, manifest.read_text().splitlines()) if r]
    else:
        ids = [(p.parent.parent.parent.name, p.parent.parent.name)
               for p in sorted(cases_dir.glob("*/*/case/case.json"))]
    per_case, stages_n, dec_per_stage, types, actions = [], [], [], Counter(), Counter()
    unscoreable, final_dx_stage, ctx_spans, timeline_lens = 0, 0, [], []
    phase_calls, phase_secs, wall, failures, statuses = Counter(), Counter(), [], [], Counter()
    for sid, hid in ids:
        sp = cases_dir / sid / hid / "plan" / "planner_stats.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            statuses[s.get("status")] += 1
            wall.append(s.get("wall_seconds_total") or 0)
            if s.get("timeline_events") is not None:
                timeline_lens.append(s["timeline_events"])
            for ph, b in (s.get("llm_by_phase") or {}).items():
                phase_calls[ph] += b.get("calls", 0)
                phase_secs[ph] += b.get("llm_seconds", 0.0)
            if s.get("status") != "ok":
                failures.append({"subject_id": sid, "hadm_id": hid, "error": (s.get("error") or "")[:200]})
        cp = cases_dir / sid / hid / "case" / "case.json"
        if not cp.exists():
            if not sp.exists():
                failures.append({"subject_id": sid, "hadm_id": hid, "error": "no case.json / planner_stats"})
            continue
        case = json.loads(cp.read_text())
        stages = case["stages"]
        stages_n.append(len(stages))
        n_dec = 0
        for s in stages:
            gts = s.get("gts", [])
            dec_per_stage.append(len(gts))
            n_dec += len(gts)
            a, b = s["context_range"]
            ctx_spans.append(b - a + 1)
            if gts and all(g.get("type") == "diagnosis" for g in gts):
                final_dx_stage += 1
            for g in gts:
                types[g.get("type")] += 1
                if g.get("type") == "medication":
                    actions[g.get("action")] += 1
                if g.get("unscoreable"):
                    unscoreable += 1
        per_case.append({"subject_id": sid, "hadm_id": hid, "stages": len(stages), "decisions": n_dec,
                         "prior_admissions": len(case.get("prior_admissions", []))})

    n_dec_total = sum(types.values())
    n_med = sum(actions.values())
    out = {
        "cases_in_manifest": len(ids),
        "cases_with_case_json": len(per_case),
        "planner_status": dict(statuses),
        "failures": failures,
        "stages_total": sum(stages_n),
        "decisions_total": n_dec_total,
        "stages_per_case_mean": round(st.mean(stages_n), 3) if stages_n else None,
        "stages_per_case_hist": dict(sorted(Counter(stages_n).items())),
        "frac_cases_ge3_stages": round(sum(1 for x in stages_n if x >= 3) / len(stages_n), 3) if stages_n else None,
        "decisions_per_stage_mean": round(st.mean(dec_per_stage), 3) if dec_per_stage else None,
        "final_diagnosis_stages": final_dx_stage,
        "type_mix": {k: round(v / n_dec_total, 3) for k, v in types.most_common()} if n_dec_total else {},
        "type_counts": dict(types),
        "med_action_mix": {k: round(v / n_med, 3) for k, v in actions.most_common()} if n_med else {},
        "unscoreable_gts": unscoreable,
        "context_span_median": st.median(ctx_spans) if ctx_spans else None,
        "context_spans_lt10": sum(1 for x in ctx_spans if x < 10),
        "timeline_events_median": st.median(timeline_lens) if timeline_lens else None,
        "llm_calls_by_phase_total": dict(phase_calls),
        "llm_calls_per_case_mean": round(sum(phase_calls.values()) / len(wall), 1) if wall else None,
        "llm_seconds_by_phase_total": {k: round(v, 1) for k, v in phase_secs.items()},
        "wall_seconds_per_case_mean": round(st.mean(wall), 1) if wall else None,
        "wall_seconds_per_case_median": round(st.median(wall), 1) if wall else None,
        "paper": PAPER,
        "per_case": per_case,
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = collect(Path(a.cases_dir), Path(a.manifest) if a.manifest else None)
    txt = json.dumps(res, indent=2)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
