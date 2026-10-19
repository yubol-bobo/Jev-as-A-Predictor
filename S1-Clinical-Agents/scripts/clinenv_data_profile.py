"""Aggregate profile of the ClinEnv cases (handled as MIMIC data): counts and distributions only, no case content.

Counts below 10 are suppressed ("<10"), following the usual MIMIC small-cell practice. Ontology titles (ICD long
titles, drug names) appear only as frequent categories (>= 10 gold items), never tied to a case.

  .venv/bin/python -I scripts/clinenv_data_profile.py [--subset data/clinenv/eval_subset_paper7.txt] --out FILE
  .venv/bin/python -I scripts/clinenv_data_profile.py --example-to FILE   # one rendered case for the user, local
"""

import argparse
import json
import statistics as S
from collections import Counter
from datetime import datetime
from pathlib import Path

DATA = Path("/ocean/projects/cis260032p/yliy/data/clinenv")
HF = DATA / "hf"


def sup(n):
    return n if n >= 10 else "<10"


def dist(xs):
    xs = sorted(xs)
    if not xs:
        return None
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "mean": round(S.mean(xs), 2), "p10": q(.1), "median": q(.5), "p90": q(.9)}


def top(counter, k=15):
    return [[name, n] for name, n in counter.most_common(k) if n >= 10]


def hours(evs):
    ts = [datetime.fromisoformat(e["event_time"]) for e in evs if e.get("event_time")]
    return (max(ts) - min(ts)).total_seconds() / 3600 if len(ts) > 1 else 0.0


def profile(cases):
    n_st, ages, sex, long_h, hist, empty = [], [], Counter(), 0, 0, 0
    g_n, g_type, med_act, unscor, icdv, prim = [], Counter(), Counter(), 0, Counter(), []
    ev_n, vis_n, src, span = [], [], Counter(), []
    rv = {"hpi_chars": [], "nurse_events": [], "lab_events": [], "history_chars": []}
    dx, drugs, procs, dx_chapter = Counter(), Counter(), Counter(), Counter()
    for s, h in cases:
        c = json.loads((HF / s / h / "case" / "case.json").read_text())
        st = c["stages"]
        if not st:  # no scorable stage (not part of the benchmark's 3,561)
            empty += 1
            continue
        n_st.append(len(st))
        long_h += len(st) >= 3
        p = (st[0].get("readviews") or {}).get("patient") or {}
        a_ = p.get("age")  # MIMIC's shifted year when the age is unavailable; ClinEnv then shows "? y/o"
        ages.append("age" if isinstance(a_, (int, float)) and a_ <= 120 else "shifted_year_or_missing")
        sex[p.get("gender")] += 1
        hist += any("history" in x.get("available_agents", []) for x in st)
        for x in st:
            gts = x.get("gts", [])
            g_n.append(len(gts))
            prim.append(sum(1 for g in gts if g.get("type") == "diagnosis" and g.get("primary")))
            for g in gts:
                g_type[g["type"]] += 1
                unscor += bool(g.get("unscoreable"))
                if g["type"] == "medication":
                    med_act[g.get("action")] += 1
                    drugs[(g.get("drug_name") or "").strip().lower()] += 1
                elif g["type"] == "diagnosis":
                    icdv[g.get("icd_version")] += 1
                    dx[g.get("diagnosis_title") or g.get("description")] += 1
                    code = g.get("icd_code") or ""
                    dx_chapter[f"ICD{g.get('icd_version')}:{code[:1] if g.get('icd_version') == 10 else code[:1] + 'xx'}"] += 1
                else:
                    procs[g.get("procedure_title") or g.get("description")] += 1
            ev_n.append(len(x.get("events", [])))
            vis_n.append(len(x.get("visible_events", [])))
            span.append(round(hours(x.get("events", [])), 1))
            for e in x.get("events", []):
                src[e.get("source_table")] += 1
            r = x.get("readviews") or {}
            rv["hpi_chars"].append(len(((r.get("patient") or {}).get("hpi")) or ""))
            rv["nurse_events"].append(len((r.get("nurse") or {}).get("events", [])))
            rv["lab_events"].append(len((r.get("lab") or {}).get("events", [])))
            if r.get("history"):
                rv["history_chars"].append(len(json.dumps(r["history"])))
    tot = sum(g_type.values())
    return {
        "cases": len(n_st), "case_dirs_without_stages": empty, "stages": sum(n_st), "decisions": tot,
        "stages_per_case": dist(n_st), "cases_with_3plus_stages": sup(long_h),
        "cases_with_history_agent": sup(hist),
        "age_field": {k: sup(v) for k, v in Counter(ages).items()}, "gender": {k: sup(v) for k, v in sex.items()},
        "gold_per_stage": dist(g_n), "primary_dx_per_stage": dist(prim),
        "gold_type_share": {k: round(v / tot, 3) for k, v in g_type.items()},
        "medication_action_share": {k: round(v / sum(med_act.values()), 3) for k, v in med_act.items()},
        "unscoreable_share": round(unscor / tot, 4), "diagnosis_icd_version": {str(k): sup(v) for k, v in icdv.items()},
        "events_per_stage": dist(ev_n), "visible_events_per_stage": dist(vis_n), "stage_span_hours": dist(span),
        "event_source_tables": {k: sup(v) for k, v in src.most_common()},
        "readview_sizes_per_stage": {k: dist(v) for k, v in rv.items()},
        "top_diagnoses": top(dx), "top_dx_chapters": top(dx_chapter, 25),
        "top_drugs": top(drugs), "top_procedures": top(procs),
    }


def render_example(s, h, out):
    """Readable markdown of one case, written locally for the user (never printed)."""
    c = json.loads((HF / s / h / "case" / "case.json").read_text())
    L = [f"# ClinEnv case {s}/{h} (MIMIC-derived: local only)\n"]
    for i, x in enumerate(c["stages"]):
        r = x.get("readviews") or {}
        L.append(f"\n## Stage {i + 1}: events {x['context_range']}, agents {x.get('available_agents')}\n")
        if i == 0 and r.get("patient"):
            L.append("### Patient readview\n```json\n" + json.dumps(r["patient"], indent=1) + "\n```\n")
        for k in ("nurse", "lab"):
            evs = (r.get(k) or {}).get("events", [])
            L.append(f"### {k} readview: {len(evs)} events (first 8)\n```json\n"
                     + "\n".join(json.dumps(e) for e in evs[:8]) + "\n```\n")
        L.append(f"### Visible events: {len(x.get('visible_events', []))} (first 8)\n```json\n"
                 + "\n".join(json.dumps(e) for e in x.get("visible_events", [])[:8]) + "\n```\n")
        L.append("### Gold decisions at this stage\n```json\n" + "\n".join(json.dumps(g) for g in x["gts"]) + "\n```\n")
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    out.write_text("".join(L))
    out.chmod(0o600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--example-to", default=None)
    a = ap.parse_args()
    sub = [tuple(l.split()) for l in Path(a.subset or DATA / "eval_subset_paper7.txt").read_text().splitlines() if l.strip()]
    if a.example_to:  # a 3-stage case from the eval subset, chosen by position only
        for s, h in sub:
            if len(json.loads((HF / s / h / "case" / "case.json").read_text())["stages"]) == 3:
                render_example(s, h, Path(a.example_to))
                print("wrote", a.example_to)
                return
    allc = [(p.parent.parent.name, p.parent.name) for p in sorted(HF.glob("*/*/case"))]
    res = {"all_cases": profile(allc), "eval_subset": profile(sub)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
