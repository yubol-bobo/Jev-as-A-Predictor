"""Generate the paper's tables and numbers from results/*.json (aggregates only).

  .venv/bin/python scripts/make_tables.py      -> paper/generated/{numbers.tex, tab_pb_main.tex, tab_pb_types.tex,
                                                   tab_pb_decisions.tex, tab_pb_offline.tex, tab_ce_prelim.tex}
Every number in the paper comes from here, so rerunning after more runs finish updates the draft.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R, G = REPO / "results", REPO / "paper" / "generated"
NAMES = {"llm": r"\textsc{LLM}", "clef_k5": r"\textsc{S1-Select}", "s1_tool": r"\textsc{S1-Tool}",
         "llm_log_k5": r"\textsc{LLM} (repeat)", "llm_tool": r"\textsc{LLM-Tool}"}
TYPES = ["Data Retrieval", "Clinical Reasoning", "Action Execution", "Documentation"]


def pval(p):
    if not isinstance(p, (int, float)):
        return p
    return "$<$0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def prel(p):
    if not isinstance(p, (int, float)):
        return "=" + str(p)
    return "<0.001" if p < 0.001 else "=" + (f"{p:.2f}" if p >= 0.01 else f"{p:.3f}")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def sgn(x):
    return f"{x:+.1f}".replace("-", "$-$")


def ci(t):
    if not t:
        return "--"
    m, lo, hi = t
    return f"{sgn(100 * m)} [{sgn(100 * lo)}, {sgn(100 * hi)}]"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main():
    G.mkdir(parents=True, exist_ok=True)
    pb = load("pb_analysis.json") or {}
    nums = ""
    run1 = pb.get("run1", {})
    nums += macro("PBpaired", pb.get("run1_paired_tasks", "--"))
    for arm, key in (("llm", "LLM"), ("clef_k5", "Sel"), ("s1_tool", "Tool"), ("llm_log_k5", "Rep")):
        r = run1.get(arm, {})
        nums += macro(f"PBpass{key}", pct(r.get("pass@1")))
        nums += macro(f"PBcp{key}", pct(r.get("checkpoint_pass")))
        v = r.get("vs_llm", {})
        nums += macro(f"PBdiff{key}", ci(v.get("pass@1_diff_ci")))
        nums += macro(f"PBp{key}", prel(v.get("mcnemar_p", "--")))
        nums += macro(f"PBtok{key}", f"{r.get('mean_completion_tokens', 0) / 1000:.0f}k" if r else "--")
    nf = pb.get("noise_floor") or {}
    nums += macro("PBnfTasks", nf.get("paired_tasks", "--")) + macro("PBnfDiff", ci(nf.get("pass@1_diff_ci")))
    nums += macro("PBnfDisc", nf.get("discordant_tasks", "--"))
    if nf:
        run1 = dict(run1)
        run1["llm_log_k5"] = {"pass@1": nf["pass@1_repeat"], "checkpoint_pass": None, "mean_turns": "--",
                              "mean_completion_tokens": 0, "mean_agent_wall_min": "--",
                              "vs_llm": {"pass@1_diff_ci": nf["pass@1_diff_ci"], "mcnemar_p": "--",
                                         "checkpoint_diff_ci": nf["checkpoint_diff_ci"]},
                              "checkpoint_pass_by_type": {}, "decisions": nf.get("decisions")}
    # pairwise comparisons (each on its own common tasks)
    pairs = pb.get("pairs") or {}
    rows = []
    for key, label in (("clef_k5_vs_llm", "\\sel{} vs \\llm"), ("s1_tool_vs_llm_tool", "\\tool{} vs \\textsc{LLM-Tool}"),
                       ("llm_tool_vs_llm", "\\textsc{LLM-Tool} vs \\llm")):
        q = pairs.get(key)
        if not q:
            continue
        x, y = key.split("_vs_")
        bt = q.get("by_type_diff_ci", {})
        rows.append(f"{label} & {q['paired_tasks']} & {pct(q[f'pass@1_{x}'])} / {pct(q[f'pass@1_{y}'])} & "
                    f"{ci(q['pass@1_diff_ci'])} & {pval(q['mcnemar_p'])} & {ci(q['checkpoint_diff_ci'])} & "
                    + " & ".join(sgn(100 * bt[t][0]) if t in bt else "--" for t in TYPES) + " \\\\")
        tag = {"clef_k5_vs_llm": "SelLLM", "s1_tool_vs_llm_tool": "ToolCtl", "llm_tool_vs_llm": "CtlLLM"}[key]
        nums += macro(f"PR{tag}n", q["paired_tasks"]) + macro(f"PR{tag}diff", ci(q["pass@1_diff_ci"]))
        nums += macro(f"PR{tag}cp", ci(q["checkpoint_diff_ci"])) + macro(f"PR{tag}p", prel(q["mcnemar_p"]))
        nums += macro(f"PR{tag}x", pct(q[f"pass@1_{x}"])) + macro(f"PR{tag}y", pct(q[f"pass@1_{y}"]))
        for t, short in zip(TYPES, ("DR", "CR", "AE", "DO")):
            nums += macro(f"PR{tag}{short}", ci(bt.get(t)))
    (G / "tab_pb_pairs.tex").write_text(
        "\\begin{tabular}{lrrrrr" + "r" * len(TYPES) + "}\n\\toprule\n"
        "Comparison & Tasks & Pass@1 & $\\Delta$ pass@1 [95\\% CI] & $p$ & $\\Delta$ checkpoints [95\\% CI] & "
        + " & ".join(f"$\\Delta$ {t}" for t in ("DR", "CR", "AE", "Doc")) + " \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
    # tool use, episode endings, decider latency (main paired set; control from its pair)
    for arm, key in (("llm", "LLM"), ("clef_k5", "Sel"), ("s1_tool", "Tool")):
        r = run1.get(arm) or {}
        t = r.get("tools") or {}
        for k, m in (("calls", "Calls"), ("retrieve", "Ret"), ("act", "Act"), ("document", "Doc"),
                     ("retrieve_before_first_action", "First")):
            nums += macro(f"TL{m}{key}", t.get(k, "--"))
        ends = r.get("episode_end") or {}
        nums += macro(f"END{key}", sum(v for k, v in ends.items() if k != "final_answer"))
        d = r.get("decisions") or {}
        nums += macro(f"DEC{key}maxp", d.get("mean_max_p", "--")) + macro(f"DEC{key}s", d.get("mean_decide_s", "--"))
        nums += macro(f"DEC{key}sem", pct(d.get("share_semantically_distinct"), 0))
        nums += macro(f"DEC{key}first", pct(d.get("pick_is_first_sample"), 0))
        nums += macro(f"DEC{key}depart", pct(1 - d["pick_is_first_sample"], 0) if "pick_is_first_sample" in d else "--")
        nums += macro(f"DEC{key}err", d.get("decider_errors", "--"))
        nums += macro(f"DEC{key}short", pct(d.get("share_fit_shortened"), 0)) + macro(f"DEC{key}omit", pct(d.get("share_fit_omitted"), 0))
        nums += macro(f"TOK{key}", f"{r.get('mean_completion_tokens', 0) / 1000:.0f}k" if r else "--")
    ctl = (pb.get("pairs", {}).get("s1_tool_vs_llm_tool") or {}).get("tools", {})
    for arm, key in (("llm_tool", "Ctl"), ("s1_tool", "ToolP")):
        t = ctl.get(arm, {})
        for k, m in (("calls", "Calls"), ("retrieve", "Ret"), ("act", "Act"), ("document", "Doc")):
            nums += macro(f"TL{m}{key}", t.get(k, "--"))
    st = (run1.get("s1_tool") or {}).get("decisions") or {}
    nums += macro("TOOLregen", pct(st.get("regenerated_share"), 0)) + macro("TOOLdropped", pct(st.get("steps_with_dropped_calls"), 0))
    # main table, ACL style: bold best per column among the decision makers
    arms = [a for a in ("llm", "clef_k5", "s1_tool") if a in run1]
    def best(vals):
        m = max(v for v in vals if v is not None)
        return [v == m for v in vals]
    cols = [[run1[a]["pass@1"] for a in arms], [run1[a]["checkpoint_pass"] for a in arms]] + \
           [[run1[a]["checkpoint_pass_by_type"].get(t, {}).get("rate") for a in arms] for t in TYPES]
    bold = [best(c) for c in cols]
    rows = []
    for i, a in enumerate(arms):
        r = run1[a]
        v = r.get("vs_llm", {})
        cells = []
        for j, c in enumerate(cols):
            x = pct(c[i])
            cells.append(f"\\textbf{{{x}}}" if bold[j][i] else x)
        d1 = ci(v.get("pass@1_diff_ci")) if v else "\\textit{ref.}"
        d2 = ci(v.get("checkpoint_diff_ci")) if v else "\\textit{ref.}"
        rows.append(f"{NAMES[a]} & {cells[0]} & {d1} & {cells[1]} & {d2} & "
                    + " & ".join(cells[2:]) + f" & {r['tools']['calls']} & {r['mean_completion_tokens'] / 1000:.0f}k \\\\")
    (G / "tab_main_acl.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}\\toprule\n"
        "& \\multicolumn{2}{c}{Task pass@1} & \\multicolumn{2}{c}{All checkpoints} & "
        "\\multicolumn{4}{c}{Checkpoints by type (\\%)} & & \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}\n"
        "Decision maker & \\% & $\\Delta$ [95\\% CI] & \\% & $\\Delta$ [95\\% CI] & DR & CR & AE & Doc & "
        "\\shortstack{Calls\\\\/task} & \\shortstack{Tokens\\\\/task} \\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n")
    # main table (v1)
    rows = []
    for arm in ("llm", "clef_k5", "s1_tool", "llm_log_k5"):
        r = run1.get(arm)
        if not r:
            continue
        v = r.get("vs_llm", {})
        name = NAMES[arm] + (f"$^\\dagger$" if arm == "llm_log_k5" else "")
        tok = f"{r['mean_completion_tokens'] / 1000:.0f}k" if r["mean_completion_tokens"] else "--"
        rows.append(f"{name} & {pct(r['pass@1'])} & {ci(v.get('pass@1_diff_ci')) if v else '--'} & "
                    f"{pval(v.get('mcnemar_p', '--')) if v else '--'} & {pct(r['checkpoint_pass'])} & "
                    f"{ci(v.get('checkpoint_diff_ci')) if v else '--'} & {r['mean_turns']} & "
                    f"{tok} & {r['mean_agent_wall_min']} \\\\")
    (G / "tab_pb_main.tex").write_text(
        "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
        "Decision maker & Pass@1 & $\\Delta$ vs \\textsc{LLM} [95\\% CI] & McNemar $p$ & Checkpoints & "
        "$\\Delta$ [95\\% CI] & Turns & Gen.\\ tokens & Min \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
    # checkpoint types
    rows = []
    for arm in ("llm", "clef_k5", "s1_tool"):
        r = run1.get(arm)
        if not r:
            continue
        cells = []
        for t in TYPES:
            x = r["checkpoint_pass_by_type"].get(t, {})
            d = (r.get("vs_llm", {}).get("by_type_diff_ci") or {}).get(t)
            cells.append(pct(x.get("rate")) + (f" ({100 * d[0]:+.1f})" if d else ""))
        rows.append(f"{NAMES[arm]} & " + " & ".join(cells) + " \\\\")
    ntask = {t: run1.get("llm", {}).get("checkpoint_pass_by_type", {}).get(t, {}).get("n_tasks", "--") for t in TYPES}
    (G / "tab_pb_types.tex").write_text(
        "\\begin{tabular}{l" + "r" * len(TYPES) + "}\n\\toprule\nDecision maker & " +
        " & ".join(f"{t} ($n$={ntask[t]})" for t in TYPES) + " \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
    # decision statistics
    rows = []
    for arm in ("llm_log_k5", "clef_k5", "s1_tool"):
        d = (run1.get(arm) or {}).get("decisions")
        if not d:
            continue
        if arm == "s1_tool":
            esc = lambda k: k.replace("fhir_", "").replace("_", "\\_")
            top = ", ".join(f"{esc(k)} {100 * v:.0f}\\%" for k, v in list(d["tool_share"].items())[:3])
            rows.append(f"{NAMES[arm]} & {d['steps_per_task']} & -- & -- & {d.get('mean_max_p', '--')} & -- & "
                        f"\\multicolumn{{1}}{{l}}{{{top}}} \\\\")
        else:
            rows.append(f"{NAMES[arm]} & {d['steps_per_task']} & {pct(d['share_distinct_strings'], 0)} & "
                        f"{pct(d['share_semantically_distinct'], 0)} & {d.get('mean_max_p', '--')} & "
                        f"{d.get('mean_entropy_norm', '--')} & {pct(d.get('pick_is_first_sample'), 0)} \\\\")
    (G / "tab_pb_decisions.tex").write_text(
        "\\begin{tabular}{lrrrrrl}\n\\toprule\n& Steps/task & Distinct (\\%) & Semantically distinct (\\%) & "
        "Mean max $p$ & Norm.\\ entropy & Pick = 1st sample (\\%) / top tools \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
    # tool choices under the one-tool restriction: Clef (S1-Tool) vs the LLM (LLM-Tool), same tasks
    q = (pb.get("pairs") or {}).get("s1_tool_vs_llm_tool") or {}
    dd = q.get("decisions") or {}
    if dd.get("s1_tool") and dd.get("llm_tool"):
        a, b = dd["s1_tool"]["tool_share"], dd["llm_tool"]["tool_share"]
        keys = sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) + b.get(k, 0)))[:8]
        short = {"write_file": "Write note", "FINISH": "Finish", "observation_search_labs": "Search labs",
                 "document_reference_search_clinical_notes": "Search notes",
                 "condition_search_problems": "Search problems", "patient_search_demographics": "Demographics",
                 "medication_request_search_orders": "Search med.\\ orders", "service_request_create": "Place order"}
        nice = lambda k: short.get(k.replace("fhir_", "")) or k.replace("fhir_", "").replace("_", " ")
        rows = [f"{nice(k)} & {pct(a.get(k, 0), 0)} & {pct(b.get(k, 0), 0)} \\\\" for k in keys]
        (G / "tab_tool_choice.tex").write_text(
            "\\begin{tabular}{@{}lrr@{}}\\toprule\nTool & \\shortstack{Clef\\\\(\\tool)} & "
            "\\shortstack{LLM\\\\(\\ctl)} \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
        nums += macro("TCwriteClef", pct(a.get("write_file", 0), 0)) + macro("TCwriteLLM", pct(b.get("write_file", 0), 0))
        nums += macro("TCfinClef", pct(a.get("FINISH", 0), 0)) + macro("TCfinLLM", pct(b.get("FINISH", 0), 0))
        lab = lambda d: sum(v for k, v in d.items() if "search" in k)
        nums += macro("TCsearchClef", pct(lab(a), 0)) + macro("TCsearchLLM", pct(lab(b), 0))
        nums += macro("TCsteps", f"{dd['s1_tool']['steps_per_task']} vs {dd['llm_tool']['steps_per_task']}")
    else:
        (G / "tab_tool_choice.tex").write_text("(pending)\n")
    # decision statistics, ACL style
    rows = []
    for arm in ("clef_k5",):
        d = (run1.get(arm) or {}).get("decisions")
        if d:
            rows.append(f"{NAMES[arm]} & {d['steps_per_task']} & {pct(d['share_distinct_strings'], 0)} & "
                        f"{pct(d['share_semantically_distinct'], 0)} & {d.get('mean_max_p', '--')} & "
                        f"{d.get('mean_entropy_norm', '--')} & {pct(1 - d['pick_is_first_sample'], 0)} & "
                        f"{d.get('mean_decide_s', '--')} \\\\")
    (G / "tab_decisions_acl.tex").write_text(
        "\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n& \\shortstack{Steps\\\\/task} & \\shortstack{Distinct\\\\strings (\\%)} & "
        "\\shortstack{Distinct\\\\actions (\\%)} & \\shortstack{Mean\\\\max $p$} & \\shortstack{Norm.\\\\entropy} & "
        "\\shortstack{Departs from\\\\LLM sample (\\%)} & \\shortstack{Decide\\\\(s)} \\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n")
    # offline order steps
    off = load("pb_offline_orders.json")
    if off:
        dd = off["deciders"]
        cell = lambda k: (pct(dd[k]["accuracy"]) if isinstance(dd.get(k), dict) else pct(dd.get(k))) if k in dd else "--"
        au = lambda k: dd[k]["auroc_candidate"] if isinstance(dd.get(k), dict) else "--"
        (G / "tab_pb_offline.tex").write_text(
            "\\begin{tabular}{lrr}\n\\toprule\nDecider & Accuracy (\\%) & Candidate AUROC \\\\\n\\midrule\n"
            f"Random & {pct(off['random_expected'])} & 0.5 \\\\\nFirst sample (\\textsc{{LLM}}) & {cell('first')} & -- \\\\\n"
            f"Majority of $K$ & {cell('majority')} & -- \\\\\nClef & {cell('clef')} & {au('clef')} \\\\\n"
            f"Clef, options only & {cell('clef_blind')} & {au('clef_blind')} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
        nums += macro("PBoffSteps", off["order_steps"]) + macro("PBoffDisc", off["discriminative_steps"])
        nums += macro("PBoffTasks", off["tasks"])
    else:
        (G / "tab_pb_offline.tex").write_text("(offline order-step evaluation pending)\n")
        nums += macro("PBoffSteps", "--") + macro("PBoffDisc", "--") + macro("PBoffTasks", "--")
    # online Clef on its own order steps: whether-to-order
    on = load("pb_offline_orders_clef_online.json")
    if on:
        w = on["whether_to_order"]
        rows = []
        for key, label in (("required_available", "A required order is among the candidates"),
                           ("none_required", "No candidate places a required order")):
            r = w.get(key, {})
            rows.append(f"{label} & {r.get('n', '--')} & {pct(r.get('share_creating'), 0)} & {pct(r.get('first'), 0)} & "
                        f"{pct(r.get('clef_online'), 0)} \\\\")
        (G / "tab_pb_whether.tex").write_text(
            "\\begin{tabular}{lrrrr}\n\\toprule\nStep type & Steps & Order-creating candidates (\\%) & "
            "First sample orders (\\%) & Clef orders (\\%) \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
        d = on["deciders"]
        nums += macro("ONsteps", on["order_steps"]) + macro("ONtasks", on["tasks"]) + macro("ONdisc", on["discriminative_steps"])
        nums += macro("ONclefAcc", pct((d.get("clef_online") or {}).get("accuracy"), 0))
        nums += macro("ONfirstAcc", pct(d.get("first"), 0)) + macro("ONrandAcc", pct(on.get("random_expected"), 0))
        nums += macro("ONmajAcc", pct(d.get("majority"), 0))
        au = (d.get("clef_online") or {}).get("auroc_candidate")
        nums += macro("ONauroc", f"{au:.2f}" if au is not None else "--")
        nums += macro("ONwhich", on.get("which_order_steps", "--"))
        for key, k2 in (("required_available", "Req"), ("none_required", "None")):
            r = w.get(key, {})
            nums += macro(f"ON{k2}n", r.get("n", "--")) + macro(f"ON{k2}share", pct(r.get("share_creating"), 0))
            nums += macro(f"ON{k2}first", pct(r.get("first"), 0)) + macro(f"ON{k2}clef", pct(r.get("clef_online"), 0))
    else:
        (G / "tab_pb_whether.tex").write_text("(pending)\n")
    # ClinEnv offline (llm_log states, canonical and gold-injected sets)
    co = load("offline_qwen3.8-27b__llm_log_k5__run1.json")
    if co:
        rows = []
        for v, label in (("canon", "Canonical (natural)"), ("inj", "Gold-injected")):
            r = co["versions"].get(v, {})
            dd = r.get("deciders", {})
            acc = lambda k: pct(dd[k]["accuracy"], 0) if k in dd else "--"
            rows.append(f"{label} & {r.get('items_with_choice', '--')} & {r.get('mean_options', '--')} & "
                        f"{pct(r.get('random_expected'), 0)} & {acc('clef')} & {acc('llm_select')} & {acc('clef_blind')} \\\\")
        (G / "tab_ce_offline.tex").write_text(
            "\\begin{tabular}{lrrrrrr}\n\\toprule\nCandidate set & Items & Options & Random & Clef & "
            "\\textsc{LLM} select & Clef, options only \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n")
        inj = co["versions"].get("inj", {})
        nums += macro("COitems", co["items"]) + macro("COinjN", inj.get("items_with_choice", "--"))
        for k, m in (("clef", "Clef"), ("llm_select", "Sel"), ("clef_blind", "Blind")):
            nums += macro(f"COinj{m}", pct((inj.get("deciders", {}).get(k) or {}).get("accuracy"), 0))
        nums += macro("COcanonN", co["versions"].get("canon", {}).get("items_with_choice", "--"))
    else:
        (G / "tab_ce_offline.tex").write_text("(pending)\n")
    # ClinEnv preliminary (online clef run 1, canonicalised submit steps)
    ce = load("ce_prelim.json")
    if ce:
        for k, v in ce.items():
            nums += macro("CE" + k, v)
    # the fixed exam: every System One decider on the LLM's own logged states and candidates (pb_exam.py)
    DEC = {"clef": ("Clef-27B", "Clef"), "pplx12": ("pplx-decider-v1.2", "Pplx"), "d3": ("d3", "Dthree"),
           "linne": ("linne-oak-1.0", "Linne"), "amos": ("amos-1-large", "Amos"), "decider31b": ("decider-31b", "Mapika")}
    ex = load("pb_exam.json")
    if ex and ex.get("deciders"):
        k, b = ex["key"], ex["baselines"]
        nums += macro("EXquestions", ex["questions"]) + macro("EXtasks", ex["tasks"]) + macro("EXcommon", ex["answered_by_all"])
        nums += macro("EXkeyed", k["keyed_questions"]) + macro("EXreqn", k["required_available"])
        nums += macro("EXnonen", k["none_required"])
        nums += macro("EXaccFirst", pct(b["first_sample"]["order_accuracy"], 0))
        nums += macro("EXaccRandom", pct(b["random"]["order_accuracy"], 0))
        nums += macro("EXreqFirst", pct(b["first_sample"]["orders_when_required_available"], 0))
        rows = [f"Random & {pct(b['random']['order_accuracy'], 0)} & -- & -- & -- & -- & -- \\\\",
                f"\\llm{{}} (own sample) & {pct(b['first_sample']['order_accuracy'], 0)} & "
                f"{pct(b['first_sample']['orders_when_required_available'], 0)} & "
                f"{pct(b['first_sample']['orders_when_none_required'], 0)} & 100 & -- & -- \\\\",
                f"\\llm{{}} (most sampled) & {pct(b['majority']['order_accuracy'], 0)} & -- & -- & -- & -- & -- \\\\",
                "\\midrule"]
        for d in [d for d in DEC if d in ex["deciders"]]:
            v, (label, tag) = ex["deciders"][d], DEC[d]
            kinds = v["picked_kind"]
            rows.append(f"{label} & {pct(v['order_accuracy'], 0)} & {pct(v['orders_when_required_available'], 0)} & "
                        f"{pct(v['orders_when_none_required'], 0)} & {pct(v['agrees_with_first_sample'], 0)} & "
                        f"{pct(kinds.get('act'), 0)} & {pct(kinds.get('document'), 0)} \\\\")
            for key, short in (("order_accuracy", "acc"), ("orders_when_required_available", "req"),
                               ("orders_when_none_required", "none"), ("agrees_with_first_sample", "agree")):
                nums += macro(f"EX{short}{tag}", pct(v[key], 0))
        (G / "tab_exam.tex").write_text(
            "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n"
            "& \\multicolumn{3}{c}{Order steps (answer key)} & \\multicolumn{3}{c}{All questions} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
            "Decider & Right & \\shortstack{Orders,\\\\required} & \\shortstack{Orders,\\\\none req.} & "
            "\\shortstack{Agrees\\\\w/ LLM} & Act & Note \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\\end{tabular}\n")
    else:
        (G / "tab_exam.tex").write_text("(pending)\n")
    # online runs with other deciders in Clef's seat (each pair on its own common tasks)
    rows = []
    for d, (label, tag) in DEC.items():
        if d == "clef":
            sel, tool = pairs.get("clef_k5_vs_llm"), pairs.get("s1_tool_vs_llm_tool")
        else:
            q = (pb.get("deciders") or {}).get(d) or {}
            sel, tool = q.get("select_vs_llm"), q.get("tool_vs_llm_tool")
        if not (sel or tool):
            continue
        cell = lambda q: (f"{q['paired_tasks']} & {ci(q['checkpoint_diff_ci'])}" if q else "-- & --")  # noqa: E731
        rows.append(f"{label} & {cell(sel)} & {cell(tool)} \\\\")
        for q, arm in ((sel, "Sel"), (tool, "Tool")):
            if q:
                nums += macro(f"ON{arm}{tag}n", q["paired_tasks"]) + macro(f"ON{arm}{tag}cp", ci(q["checkpoint_diff_ci"]))
    (G / "tab_deciders_online.tex").write_text(
        "\\begin{tabular}{@{}lrrrr@{}}\\toprule\n"
        "& \\multicolumn{2}{c}{\\sel{} vs \\llm} & \\multicolumn{2}{c}{\\tool{} vs \\textsc{LLM-Tool}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "Decider & Tasks & $\\Delta$ checkpoints & Tasks & $\\Delta$ checkpoints \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
    (G / "numbers.tex").write_text(nums)
    print("wrote", sorted(p.name for p in G.iterdir()))


if __name__ == "__main__":
    main()
