"""Summarise ClinEnv runs per arm (aggregates only; no patient content), paired on cases finished by every run.

  .venv/bin/python scripts/clinenv_summarize.py [--out results/clinenv_summary.json] [--authors]

Per run directory qwen3.8-27b__<arm>__run<N> under data/clinenv/tree (and, with --authors, the authors' own scored
runs in data/clinenv/hf/*/*/evals/<model>):
  stage F1 (mean over stages, ClinEnv's Hungarian scorer), per-type F1 (diagnosis / medication / procedure),
  information coverage and efficiency, information calls per stage;
  decider statistics split into information-gathering steps and decision steps (a step is a decision step when any
  candidate submits a medication, diagnosis or procedure).
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/ocean/projects/cis260032p/yliy/data/clinenv")
SUBMIT = re.compile(r"^(submit_medication|submit_diagnosis|submit_procedure)\(", re.M)


def stage_metrics(ev_stage, out_stage):
    o = ev_stage.get("outcome_score") or {}
    p = (ev_stage.get("process_score") or {}).get("info_coverage") or {}
    hits, gts, subs = defaultdict(float), defaultdict(int), defaultdict(int)
    for m in o.get("matches") or []:
        if m.get("gt_item"):
            hits[m["gt_item"]["type"]] += m.get("score") or 0.0
    for g in (out_stage or {}).get("gts", []):
        if not g.get("unscoreable"):
            gts[g.get("type")] += 1
    for s in (out_stage or {}).get("submissions", []):
        subs[s.get("type")] += 1
    per_type = {}
    for t in ("diagnosis", "medication", "procedure"):
        if gts[t]:
            pr = hits[t] / subs[t] if subs[t] else 0.0
            rc = hits[t] / gts[t]
            per_type[t] = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"f1": o.get("f1"), "types": per_type, "coverage": p.get("coverage_overall"),
            "efficiency": p.get("efficiency"), "info_calls": p.get("n_info_calls")}


def load_case(case_dir, slug, root_runs):
    ev = case_dir / "evals" / slug / "latest.json"
    out = root_runs / "runs" / slug / "output.json"
    if not ev.exists():
        return None
    e = json.loads(ev.read_text())
    o = json.loads(out.read_text()) if out.exists() else {"stages": []}
    st = [stage_metrics(es, o["stages"][i] if i < len(o["stages"]) else None) for i, es in enumerate(e["stages"])]
    dec = root_runs / "runs" / slug / "decisions.jsonl"
    rows = [json.loads(l) for l in dec.read_text().splitlines()] if dec.exists() else []
    return {"stages": st, "decisions": rows}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def decider_stats(rows):
    real = [r for r in rows if r["n_unique"] > 1 and r.get("distribution")]
    out = {"steps": len(rows), "real_choices": len(real)}
    for name, sel in (("decision_steps", [r for r in real if any(SUBMIT.search(c) for c in r["candidates"])]),
                      ("info_steps", [r for r in real if not any(SUBMIT.search(c) for c in r["candidates"])])):
        out[name] = {"n": len(sel),
                     "mean_max_p": mean([max(r["distribution"]) for r in sel]),
                     "mean_entropy_norm": mean([-sum(p * math.log(p) for p in r["distribution"] if p > 0)
                                                / math.log(r["n_unique"]) for r in sel]),
                     "mean_decide_s": mean([r["decide_s"] for r in sel])}
    out["decider_errors"] = sum(1 for r in rows if (r.get("info") or {}).get("error"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/clinenv_summary.json"))
    ap.add_argument("--authors", action="store_true", help="also summarise the authors' scored runs (same cases)")
    a = ap.parse_args()
    tree, hf = DATA / "tree", DATA / "hf"
    runs = defaultdict(dict)  # slug -> {(s,h): case}
    for case_dir in sorted(p for p in tree.glob("*/*") if p.is_dir() and not p.parent.name.startswith("_")):
        for slug_dir in (case_dir / "evals").glob("*") if (case_dir / "evals").exists() else []:
            c = load_case(case_dir, slug_dir.name, case_dir)
            if c:
                runs[slug_dir.name][(case_dir.parent.name, case_dir.name)] = c
    if a.authors:
        mine = set().union(*[set(v) for v in runs.values()]) if runs else set()
        for s, h in mine:
            for slug_dir in (hf / s / h / "evals").glob("*") if (hf / s / h / "evals").exists() else []:
                c = load_case(hf / s / h, slug_dir.name, hf / s / h)
                if c:
                    runs["authors:" + slug_dir.name][(s, h)] = c
    ours = [k for k in runs if not k.startswith("authors:")]
    common = set.intersection(*[set(runs[k]) for k in ours]) if ours else set()
    summary = {"paired_cases": len(common), "runs": {}}
    for slug, cases in sorted(runs.items()):
        cs = [cases[k] for k in sorted(common) if k in cases]
        st = [x for c in cs for x in c["stages"]]
        r = {"cases": len(cs), "cases_finished": len(cases), "stages": len(st),
             "stage_f1": mean([x["f1"] for x in st]),
             "f1_by_type": {t: mean([x["types"].get(t) for x in st]) for t in ("diagnosis", "medication", "procedure")},
             "coverage": mean([x["coverage"] for x in st]), "efficiency": mean([x["efficiency"] for x in st]),
             "info_calls_per_stage": mean([x["info_calls"] for x in st])}
        rows = [d for c in cs for d in c["decisions"]]
        if rows:
            r["decider"] = decider_stats(rows)
        summary["runs"][slug] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
