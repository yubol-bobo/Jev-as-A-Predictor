"""PhysicianBench analysis for the paper: LLM vs System One as the decision maker (aggregates only).

Arms: llm (benchmark procedure), clef_k5 (Clef picks among K=5 whole turns), s1_tool (Clef picks the tool, the LLM
writes the call), llm_log_k5 (executes the first of K=5 samples: the same distribution as llm, so llm vs llm_log is
the run-to-run noise floor).

Per arm, on the tasks finished by every compared arm (paired):
  task pass@1 (run 1; mean over runs when several), pass@k and pass^k over the runs every arm has,
  checkpoint pass rate overall and by the benchmark's checkpoint taxonomy, turns, generated tokens, wall time;
  differences vs llm with 95% task-cluster bootstrap CIs and exact McNemar p for task pass;
  decision statistics per arm (how often a real choice exists, decider confidence, tool choices for s1_tool).

  .venv/bin/python scripts/pb_analyze.py [--arms llm,clef_k5,s1_tool,llm_log_k5] [--out results/pb_analysis.json]
"""

import argparse
import json
import math
import random
import re
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

from pb_summarize import JOBS, REPO, TAXONOMY, load_job, semantic_key

TASKS = REPO / "third_party/physicianbench/tasks/v1"
B = 10000


def mcnemar(b, c):
    """Exact two-sided McNemar p from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(per_task_a, per_task_b, rng):
    """Mean difference a - b over tasks, with a 95% bootstrap CI over tasks."""
    d = [x - y for x, y in zip(per_task_a, per_task_b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(sum(d) / n, 4), round(means[int(0.025 * B)], 4), round(means[int(0.975 * B)], 4)


def task_metrics(j, task):
    by = defaultdict(lambda: [0, 0])
    for cp, ok in j["cps"].items():
        t = TAXONOMY.get(task, {}).get(cp, "?")
        by[t][0] += ok
        by[t][1] += 1
    n = sum(v[1] for v in by.values())
    return {"pass": float(j["pass"]), "cp": sum(v[0] for v in by.values()) / max(n, 1),
            "by": {k: v[0] / v[1] for k, v in by.items()}}


def ending(jd):
    """How the episode ended, from the agent's final message (MiniAgent)."""
    out = jd / "logs/agent/stdout.txt"
    t = out.read_text()[:300] if out.exists() else ""
    if t.startswith("LLM call failed"):
        return "context_overflow" if "maximum context length" in t else "llm_error"
    if t.startswith("Agent reached maximum steps"):
        return "max_steps"
    if t.startswith("Agent aborted"):
        return "loop_abort"
    return "final_answer" if out.exists() else "no_output"


def tool_category(name):
    if name == "write_file":
        return "document"
    if name.endswith("_create"):
        return "act"  # orders, prescriptions, messages, appointments
    return "retrieve"


def tool_profile(jd):
    """Tool calls by category, and retrieval calls before the first act/document call (aggregates only)."""
    traj = jd / "logs/agent/trajectory.log"
    names = []
    if traj.exists():
        for line in traj.read_text().splitlines():
            e = json.loads(line)
            if e["type"] == "tool_call":
                names.append(e["metadata"].get("tool_name", ""))
    cats = [tool_category(n) for n in names]
    first = next((i for i, c in enumerate(cats) if c != "retrieve"), len(cats))
    return {"calls": len(cats), "retrieve": cats.count("retrieve"), "act": cats.count("act"),
            "document": cats.count("document"), "retrieve_before_first_action": first}


def decision_stats(arm, js):
    dec = [d for j in js for d in j["decisions"]]
    if not dec:
        return None
    if arm.split("__")[0] in ("s1_tool", "llm_tool"):  # <arm>__<decider>: the same arm with another decider
        tools = Counter(d["tool"] for d in dec)
        real = [d for d in dec if d.get("distribution")]
        return {"steps": len(dec), "steps_per_task": round(len(dec) / len(js), 1),
                "tool_share": {k: round(v / len(dec), 3) for k, v in tools.most_common()},
                "mean_max_p": round(sum(max(d["distribution"]) for d in real) / max(len(real), 1), 3),
                "regenerated_share": round(sum(bool(d.get("regenerated")) for d in dec) / len(dec), 3),
                "steps_with_dropped_calls": round(sum(bool(d.get("dropped_calls")) for d in dec) / len(dec), 3),
                "decider_errors": sum(1 for d in dec if (d.get("info") or {}).get("error"))}
    real = [d for d in dec if d["n_unique"] > 1]
    sem = [d for d in dec if len({semantic_key(c) for c in d["candidates"]}) > 1]
    out = {"steps": len(dec), "steps_per_task": round(len(dec) / len(js), 1),
           "share_distinct_strings": round(len(real) / len(dec), 3),
           "share_semantically_distinct": round(len(sem) / len(dec), 3),
           "mean_unique": round(sum(d["n_unique"] for d in dec) / len(dec), 2)}
    withp = [d for d in real if d.get("distribution")]
    if withp:
        out.update({"mean_max_p": round(sum(max(d["distribution"]) for d in withp) / len(withp), 3),
                    "mean_entropy_norm": round(sum(-sum(p * math.log(p) for p in d["distribution"] if p > 0)
                                                   / math.log(d["n_unique"]) for d in withp) / len(withp), 3),
                    "pick_is_first_sample": round(sum(d["pick"] == 0 for d in withp) / len(withp), 3),
                    "decider_errors": sum(1 for d in dec if (d.get("info") or {}).get("error")),
                    "share_fit_shortened": round(sum(1 for d in withp if ((d.get("info") or {}).get("fit") or {}).get("shortened"))
                                                 / len(withp), 3),
                    "share_fit_omitted": round(sum(1 for d in withp if ((d.get("info") or {}).get("fit") or {}).get("omitted"))
                                               / len(withp), 3),
                    "mean_decide_s": round(sum(d["decide_s"] for d in withp) / len(withp), 2)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="llm,clef_k5,s1_tool")
    ap.add_argument("--repeat-arm", default="llm_log_k5", help="same distribution as llm: noise floor, own pairing")
    ap.add_argument("--out", default=str(REPO / "results/pb_analysis.json"))
    a = ap.parse_args()
    arms = a.arms.split(",")
    jobs = {}
    for arm in arms:
        for rd in sorted((JOBS / arm).glob("run[1-9]")):
            tj = {t.name: load_job(t) for t in sorted(rd.iterdir()) if t.is_dir()}
            jobs.setdefault(arm, {})[int(rd.name[3:])] = {t: j for t, j in tj.items() if j is not None}
    deciders = sorted({p.name.split("__")[1] for p in JOBS.glob("*__*") if p.is_dir()})  # other System One models
    for extra in (a.repeat_arm, "llm_tool", *(f"{arm}__{d}" for d in deciders for arm in ("clef_k5", "s1_tool"))):
        for rd in sorted((JOBS / extra).glob("run1")):
            tj = {t.name: load_job(t) for t in sorted(rd.iterdir()) if t.is_dir()}
            jobs.setdefault(extra, {})[1] = {t: j for t, j in tj.items() if j is not None}
    rng = random.Random(20261019)
    res = {"arms": arms, "note": "llm_log_k5 executes the first of K i.i.d. samples: same distribution as llm"}
    # run 1, paired over tasks finished by every arm
    common = sorted(set.intersection(*[set(jobs.get(arm, {}).get(1, {})) for arm in arms]))
    res["run1_paired_tasks"] = len(common)
    m = {arm: [task_metrics(jobs[arm][1][t], t) for t in common] for arm in arms}
    types = sorted({k for arm in arms for x in m[arm] for k in x["by"]})
    res["run1"] = {}
    for arm in arms:
        js = [jobs[arm][1][t] for t in common]
        r = {"pass@1": round(sum(x["pass"] for x in m[arm]) / max(len(common), 1), 4),
             "checkpoint_pass": round(sum(x["cp"] for x in m[arm]) / max(len(common), 1), 4),
             "checkpoint_pass_by_type": {}, "mean_turns": round(sum(j["turns"] for j in js) / max(len(js), 1), 1),
             "mean_completion_tokens": round(sum(j["completion_tokens"] for j in js) / max(len(js), 1)),
             "mean_agent_wall_min": round(sum(j["wall"].get("agent", 0) for j in js) / max(len(js), 1) / 60, 1),
             "decisions": decision_stats(arm, js),
             "episode_end": dict(Counter(ending(JOBS / arm / "run1" / t) for t in common))}
        tp = [tool_profile(JOBS / arm / "run1" / t) for t in common]
        r["tools"] = {k: round(sum(x[k] for x in tp) / max(len(tp), 1), 1) for k in
                      ("calls", "retrieve", "act", "document", "retrieve_before_first_action")}
        for t in types:
            v = [x["by"][t] for x in m[arm] if t in x["by"]]
            r["checkpoint_pass_by_type"][t] = {"n_tasks": len(v), "rate": round(sum(v) / max(len(v), 1), 4)}
        if arm != "llm" and common:
            pa, pl = [x["pass"] for x in m[arm]], [x["pass"] for x in m["llm"]]
            b = sum(1 for x, y in zip(pa, pl) if x and not y)
            c = sum(1 for x, y in zip(pa, pl) if y and not x)
            r["vs_llm"] = {"pass@1_diff_ci": boot_ci(pa, pl, rng), "mcnemar_p": round(mcnemar(b, c), 4),
                           "tasks_won": b, "tasks_lost": c,
                           "checkpoint_diff_ci": boot_ci([x["cp"] for x in m[arm]], [x["cp"] for x in m["llm"]], rng)}
            for t in types:
                both = [(x["by"][t], y["by"][t]) for x, y in zip(m[arm], m["llm"]) if t in x["by"] and t in y["by"]]
                if both:
                    r["vs_llm"].setdefault("by_type_diff_ci", {})[t] = boot_ci(*zip(*both), rng)
        res["run1"][arm] = r
    # noise floor: llm vs its repeat (identical distribution), paired on their common run-1 tasks
    rep = a.repeat_arm
    if rep in jobs and 1 in jobs.get("llm", {}):
        ct = sorted(set(jobs["llm"][1]) & set(jobs[rep][1]))
        if ct:
            ml = [task_metrics(jobs["llm"][1][t], t) for t in ct]
            mr = [task_metrics(jobs[rep][1][t], t) for t in ct]
            pa, pl = [x["pass"] for x in mr], [x["pass"] for x in ml]
            res["noise_floor"] = {
                "arm": rep, "paired_tasks": len(ct), "pass@1_llm": round(sum(pl) / len(ct), 4),
                "pass@1_repeat": round(sum(pa) / len(ct), 4), "pass@1_diff_ci": boot_ci(pa, pl, rng),
                "discordant_tasks": sum(x != y for x, y in zip(pa, pl)),
                "checkpoint_diff_ci": boot_ci([x["cp"] for x in mr], [x["cp"] for x in ml], rng),
                "decisions": decision_stats(rep, [jobs[rep][1][t] for t in ct]),
                "episode_end": dict(Counter(ending(JOBS / rep / "run1" / t) for t in ct))}
    # pairwise comparisons on each pair's common run-1 tasks: who chooses the tool under the same restriction
    # (s1_tool vs llm_tool), and the restriction itself (llm_tool vs llm)
    res["pairs"] = {}

    def pair(x, y):
        if not (x in jobs and y in jobs and 1 in jobs[x] and 1 in jobs[y]):
            return None
        ct = sorted(set(jobs[x][1]) & set(jobs[y][1]))
        if ct:
            mx = [task_metrics(jobs[x][1][t], t) for t in ct]
            my = [task_metrics(jobs[y][1][t], t) for t in ct]
            px, py = [m["pass"] for m in mx], [m["pass"] for m in my]
            b = sum(1 for u, v in zip(px, py) if u and not v)
            c = sum(1 for u, v in zip(px, py) if v and not u)
            byt = {}
            for t in types:
                both = [(u["by"][t], v["by"][t]) for u, v in zip(mx, my) if t in u["by"] and t in v["by"]]
                if both:
                    byt[t] = boot_ci(*zip(*both), rng)
            return {
                "paired_tasks": len(ct), f"pass@1_{x}": round(sum(px) / len(ct), 4), f"pass@1_{y}": round(sum(py) / len(ct), 4),
                f"checkpoint_{x}": round(sum(m["cp"] for m in mx) / len(ct), 4),
                f"checkpoint_{y}": round(sum(m["cp"] for m in my) / len(ct), 4),
                "pass@1_diff_ci": boot_ci(px, py, rng), "mcnemar_p": round(mcnemar(b, c), 4),
                "checkpoint_diff_ci": boot_ci([m["cp"] for m in mx], [m["cp"] for m in my], rng), "by_type_diff_ci": byt,
                "tools": {arm: {k: round(sum(tool_profile(JOBS / arm / "run1" / t)[k] for t in ct) / len(ct), 1)
                                for k in ("calls", "retrieve", "act", "document")} for arm in (x, y)},
                "decisions": {arm: decision_stats(arm, [jobs[arm][1][t] for t in ct]) for arm in (x, y)}}

    for x, y in (("s1_tool", "llm_tool"), ("llm_tool", "llm"), ("clef_k5", "llm")):
        q = pair(x, y)
        if q:
            res["pairs"][f"{x}_vs_{y}"] = q
    # the same arms with other System One deciders: against the LLM, the control, and Clef in the same arm
    res["deciders"] = {}
    for d in deciders:
        res["deciders"][d] = {k: q for k, (x, y) in {
            "select_vs_llm": (f"clef_k5__{d}", "llm"), "select_vs_clef": (f"clef_k5__{d}", "clef_k5"),
            "tool_vs_llm_tool": (f"s1_tool__{d}", "llm_tool"), "tool_vs_clef": (f"s1_tool__{d}", "s1_tool"),
            "tool_vs_llm": (f"s1_tool__{d}", "llm")}.items() if (q := pair(x, y))}
    # several runs: pass@k / pass^k over the runs every arm has, on tasks finished in all of them
    runs = sorted(set.intersection(*[set(jobs.get(arm, {})) for arm in arms]))
    if len(runs) > 1:
        ct = sorted(set.intersection(*[set(jobs[arm][r]) for arm in arms for r in runs]))
        res["multi_run"] = {"runs": runs, "paired_tasks": len(ct), "arms": {}}
        for arm in arms:
            ps = [[jobs[arm][r][t]["pass"] for r in runs] for t in ct]
            res["multi_run"]["arms"][arm] = {
                "pass@1_mean": round(sum(sum(p) / len(p) for p in ps) / max(len(ct), 1), 4),
                f"pass@{len(runs)}": round(sum(any(p) for p in ps) / max(len(ct), 1), 4),
                f"pass^{len(runs)}": round(sum(all(p) for p in ps) / max(len(ct), 1), 4)}
    # progress (unpaired, monitoring only)
    res["finished"] = {f"{arm}/run{r}": len(tj) for arm in jobs for r, tj in sorted(jobs[arm].items())}
    # breakdown by task category (task.toml tags), run 1 paired
    cat = {}
    for t in common:
        tags = tomllib.loads((TASKS / t / "task.toml").read_text()).get("metadata", {}).get("tags", [])
        for g in tags[1:]:
            cat.setdefault(g, []).append(common.index(t))
    res["run1_by_category"] = {g: {arm: round(sum(m[arm][i]["pass"] for i in idx) / len(idx), 3) for arm in arms}
                               | {"n": len(idx)} for g, idx in sorted(cat.items())}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
