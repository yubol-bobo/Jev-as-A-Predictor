"""X1 (docs/research_loop.md): the Gemma 4 agent's base runs beside Qwen's, descriptive only (aggregates only).

Per run: tasks, task pass@1, checkpoint pass overall and by group (pb_gap.cp_group), steps, orders, notes, required
order specs placed by the end, orders matching no spec. Per agent: required specs placed by run 1
alone vs by runs 1 and 2 pooled. Gemma − Qwen paired over tasks (mean over each agent's runs), bootstrap CIs.

  .venv/bin/python scripts/pb_x1.py [--out results/pb_x1.json]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO)]
from pb_gap import boot, calls, cp_group, trajectory  # noqa: E402
from pb_offline_orders import CREATE, matches, required_orders  # noqa: E402
from pb_summarize import JOBS, load_job  # noqa: E402

# Qwen: its `llm` run 1 and the executed path of llm_log_k5 run 1 (the first of K samples is executed, as in `llm`;
# `llm` run 2 has 15 tasks)
AGENTS = {"qwen": ("llm/run1", "llm_log_k5/run1"), "gemma4": ("llm_log_k1__gemma4/run1", "llm_log_k1__gemma4/run2")}


def mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 4) if v else None


def per_task(slug):
    out = {}
    for jd in sorted(p for p in (JOBS / slug).iterdir() if p.is_dir()) if (JOBS / slug).exists() else []:
        j = load_job(jd)
        if j is None:
            continue
        specs = required_orders(jd.name)
        pos = [s for s in specs if not s["negative"]]
        orders = [(CREATE[n][0], a.get(CREATE[n][1], ""), str(a.get(CREATE[n][2], "") or "")) for n, a in calls(jd) if n in CREATE]
        groups = defaultdict(list)
        for cp, ok in j["cps"].items():
            groups[cp_group(jd.name, cp, specs)[1]].append(ok)
        t = trajectory(jd, specs)
        out[jd.name] = {"pass": float(j["pass"]), "cps": mean(list(j["cps"].values())),
                        "groups": {g: sum(v) / len(v) for g, v in groups.items()},
                        "steps": t["steps"], "orders": t["orders"], "notes": t["notes"],
                        "required_placed": t["required_placed"], "unmatched": t["unmatched_orders"],
                        "met": [i for i, s in enumerate(pos) if any(matches(s, *o) for o in orders)], "n_pos": len(pos)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/pb_x1.json"))
    a = ap.parse_args()
    res, runs = {}, {}
    for agent, slugs in AGENTS.items():
        for slug in slugs:
            r = runs[slug] = per_task(slug)
            if not r:
                continue
            gs = sorted({g for x in r.values() for g in x["groups"]})
            res[slug] = {"tasks": len(r), **{k: mean([x[k] for x in r.values()]) for k in
                                              ("pass", "cps", "steps", "orders", "notes", "required_placed", "unmatched")},
                         "groups": {g: mean([x["groups"].get(g) for x in r.values()]) for g in gs}}
        r1, r2 = runs.get(slugs[0], {}), runs.get(slugs[1], {})
        both = [t for t in r1 if t in r2 and r1[t]["n_pos"]]
        if both:
            res[f"{agent}_pool"] = {"tasks": len(both),
                                    "run1": mean([len(r1[t]["met"]) / r1[t]["n_pos"] for t in both]),
                                    "runs_1_2": mean([len(set(r1[t]["met"]) | set(r2[t]["met"])) / r1[t]["n_pos"] for t in both])}
    qa, ga = [runs.get(s, {}) for s in AGENTS["qwen"]], [runs.get(s, {}) for s in AGENTS["gemma4"]]
    tasks = sorted(set().union(*qa) & set().union(*ga))
    rng = random.Random(0)
    diff = {}
    for k in ("pass", "cps", "required_placed", "orders", "unmatched", "steps"):
        d = []
        for t in tasks:
            q, g = mean([r[t][k] for r in qa if t in r]), mean([r[t][k] for r in ga if t in r])
            if q is not None and g is not None:
                d.append(g - q)
        diff[k] = {"cases": len(d), "mean": mean(d), "ci95": boot(d, rng) if d else None}
    res["gemma4_minus_qwen"] = diff
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
