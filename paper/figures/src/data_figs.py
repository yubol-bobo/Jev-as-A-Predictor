"""Data figures for the paper, from results/*.json (aggregates only). Vector PDF.

  /ocean/projects/cis260032p/yliy/venvs/figs/bin/python paper/figures/src/data_figs.py
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[3]
R, OUT = REPO / "results", REPO / "paper" / "figures"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelsize": 9.5, "axes.labelweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.linestyle": (0, (4, 3)), "grid.linewidth": 0.6, "grid.color": "#D5D9E2",
    "axes.axisbelow": True,
})
SERIF = {"family": "DejaVu Serif", "weight": "bold"}
C = {"llm": "#6C5CE7", "clef_k5": "#1E7BEA", "s1_tool": "#F08A24", "llm_tool": "#9AA3B5", "llm_log_k5": "#B9B2F5",
     "ink": "#16213E", "sub": "#5A6378", "good": "#13A36E", "bad": "#E0533D"}
NAME = {"llm": "LLM", "clef_k5": "S1-Select", "s1_tool": "S1-Tool", "llm_tool": "LLM-Tool", "llm_log_k5": "LLM (repeat)"}
TYPES = [("Data Retrieval", "Data"), ("Clinical Reasoning", "Reason"),
         ("Action Execution", "Action"), ("Documentation", "Notes")]


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def title(ax, s):
    ax.set_title(s, fontdict=SERIF, loc="left", pad=6)


def label(ax, x=None, y=None):
    if x:
        ax.set_xlabel(x, fontdict=SERIF)
    if y:
        ax.set_ylabel(y, fontdict=SERIF)


# ---------------------------------------------------------------- Figure 2: headline results
def fig_main():
    pb = load("pb_analysis.json")
    if not pb:
        return
    run1, pairs = pb["run1"], pb.get("pairs", {})
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.78)

    # (a) checkpoint pass by type, main paired set
    ax = fig.add_subplot(gs[0])
    arms = [a for a in ("llm", "clef_k5", "s1_tool") if a in run1]
    cats = TYPES + [(None, "All")]
    width = 0.8 / len(arms)
    for k, arm in enumerate(arms):
        vals = [100 * (run1[arm]["checkpoint_pass"] if t is None else run1[arm]["checkpoint_pass_by_type"][t]["rate"])
                for t, _ in cats]
        xs = [i + (k - (len(arms) - 1) / 2) * width for i in range(len(cats))]
        ax.bar(xs, vals, width * 0.92, color=C[arm], label=NAME[arm], zorder=3, edgecolor="white", linewidth=0.6)
        if arm == "s1_tool":
            for x, v in zip(xs, vals):
                ax.text(x, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8, color=C[arm], fontweight="bold")
    ax.set_xticks(range(len(cats)), [c for _, c in cats], fontsize=8)
    ax.set_ylim(0, 105)
    ax.axvline(len(cats) - 1.5, color="#C3C9D6", lw=0.8, zorder=1)
    label(ax, y="Checkpoints passed (%)")
    title(ax, f"(a) PhysicianBench, {pb['run1_paired_tasks']} paired tasks")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=8.2, handlelength=1.2,
              columnspacing=1.2, prop={"weight": "bold", "size": 8.2})

    # (b) paired differences with 95% CIs
    ax = fig.add_subplot(gs[1])
    rows = [("clef_k5_vs_llm", "S1-Select − LLM", C["clef_k5"], "who decides the action"),
            ("s1_tool_vs_llm_tool", "S1-Tool − LLM-Tool", C["s1_tool"], "who chooses the tool"),
            ("llm_tool_vs_llm", "LLM-Tool − LLM", C["llm_tool"], "one tool per turn"),
            ("noise", "LLM repeat − LLM", C["llm_log_k5"], "run-to-run noise")]
    nf = pb.get("noise_floor")
    y = 0
    yt, yl = [], []
    for key, name, col, note in rows:
        q = pairs.get(key) if key != "noise" else nf
        if not q or q["paired_tasks"] < 10:  # the LLM rerun is too small to show until runs 2-3 land
            continue
        for metric, off, mk in (("checkpoint_diff_ci", 0.17, "o"), ("pass@1_diff_ci", -0.17, "D")):
            m, lo, hi = (100 * v for v in q[metric])
            ax.plot([lo, hi], [y + off] * 2, color=col, lw=2.2, solid_capstyle="round", zorder=3)
            ax.plot(m, y + off, mk, ms=5.2 if mk == "o" else 4.4, color=col, mec="white", mew=0.8, zorder=4)
        yt.append(y)
        yl.append(f"{name}\n{note} (n={q['paired_tasks']})")
        y -= 1
    ax.axvline(0, color=C["ink"], lw=0.9, zorder=2)
    ax.set_yticks(yt, yl, fontsize=7.2)
    ax.set_ylim(y + 0.5, 0.6)
    ax.grid(axis="y", visible=False)
    label(ax, x="Difference (points), 95% CI")
    title(ax, "(b) Paired differences")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C["ink"], ms=5, label="checkpoints"),
                       Line2D([], [], marker="D", ls="", color=C["ink"], ms=4.4, label="task pass")],
              loc="upper left", fontsize=7.5, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(OUT / "fig2_main.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3: Clef orders when an order is required
def fig_orders():
    on = load("pb_offline_orders_clef_online.json")
    if not on:
        return
    w = on["whether_to_order"]
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    groups = [("required_available", "A required order\nis among candidates"),
              ("none_required", "No candidate places\na required order")]
    bars = [("share_creating", "Ordering candidates\n(base rate)", "#C9CFDB"),
            ("first", "LLM's own sample", C["llm"]), ("clef_online", "Clef's pick", C["clef_k5"])]
    width = 0.26
    for k, (key, name, col) in enumerate(bars):
        vals = [100 * (w[g].get(key) or 0) for g, _ in groups]
        xs = [i + (k - 1) * width for i in range(len(groups))]
        ax.bar(xs, vals, width * 0.92, color=col, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold",
                    color=C["ink"] if key != "share_creating" else C["sub"])
    ax.set_xticks(range(len(groups)), [f"{g}\n(n={w[k]['n']})" for k, g in groups], fontsize=7.8)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    label(ax, y="Steps choosing to order (%)")
    title(ax, "Order steps: when does each decider act?")
    ax.legend(loc="upper right", fontsize=7, ncol=1, handlelength=1, prop={"size": 7})
    fig.savefig(OUT / "fig3_orders.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4: tool-use profile
def fig_tools():
    pb = load("pb_analysis.json")
    if not pb:
        return
    prof = {a: pb["run1"][a]["tools"] for a in ("llm", "clef_k5", "s1_tool") if a in pb["run1"]}
    ctl = (pb.get("pairs", {}).get("s1_tool_vs_llm_tool") or {}).get("tools", {}).get("llm_tool")
    if ctl:
        prof["llm_tool"] = ctl
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    parts = [("retrieve", "retrieve data", "#7FB2F0"), ("act", "order / prescribe / message", C["s1_tool"]),
             ("document", "write note", "#13A36E")]
    arms = list(prof)
    for i, arm in enumerate(arms[::-1]):
        left = 0
        for key, name, col in parts:
            v = prof[arm][key]
            ax.barh(i, v, left=left, color=col, height=0.62, label=name if i == 0 else None, zorder=3,
                    edgecolor="white", linewidth=0.6)
            if v >= 2.5:
                ax.text(left + v / 2, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center",
                        fontsize=7.2, color="white", fontweight="bold")
            left += v
    ax.set_yticks(range(len(arms)), [NAME[a] for a in arms[::-1]], fontweight="bold", fontsize=8.2)
    ax.grid(axis="y", visible=False)
    label(ax, x="Tool calls per task")
    title(ax, "What each decision maker does")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, fontsize=6.9, handlelength=1,
              columnspacing=0.8)
    fig.savefig(OUT / "fig4_tools.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5: ClinEnv candidate diversity and gold injection
def fig_clinenv():
    pre, off = load("ce_prelim.json"), load("offline_qwen3.8-27b__llm_log_k5__run1.json")
    if not pre:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.35), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.75})
    stages = [("strings\ndiffer", float(pre["DistinctRaw"])),
              ("decisions\ndiffer", 100 - float(pre["OneDecision"])),
              ("right and\nwrong", float(pre["RealChoicePct"]))]
    cols = ["#B9D4F6", "#6EA6EC", C["clef_k5"]]
    for i, ((name, v), col) in enumerate(zip(stages, cols)):
        a1.barh(2 - i, v, 0.66, color=col, zorder=3)
        a1.text(v + 3, 2 - i, f"{v:.0f}%", ha="left", va="center", fontsize=7.8, fontweight="bold")
    a1.set_yticks([2, 1, 0], [s for s, _ in stages], fontsize=7.2)
    a1.set_xlim(0, 135)
    a1.set_xticks([0, 50, 100])
    a1.grid(axis="y", visible=False)
    label(a1, x="Submit steps (%)")
    title(a1, "(a) Five samples")
    if off and "inj" in off["versions"]:
        dd = off["versions"]["inj"]["deciders"]
        names = [("clef", "Clef", C["clef_k5"]), ("llm_select", "LLM", C["llm"]), ("clef_blind", "Clef\nblind", "#9EC5F5")]
        for i, (k, n, col) in enumerate(names):
            v = 100 * dd[k]["accuracy"] if k in dd else 0
            a2.bar(i, v, 0.66, color=col, zorder=3)
            a2.text(i, v + 2.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
        a2.axhline(50, color=C["bad"], lw=1.1, ls=(0, (4, 2)), zorder=2)
        a2.text(2.45, 52, "chance", color=C["bad"], fontsize=7, ha="right", va="bottom", fontweight="bold")
        a2.set_xticks(range(3), [n for _, n, _ in names], fontsize=7)
        a2.set_ylim(0, 100)
        label(a2, y="Picks injected gold (%)")
        title(a2, "(b) Gold added")
    fig.savefig(OUT / "fig5_clinenv.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 6: the checker inherits the blind spot
S1 = [("pplx12", "pplx12", "#2a78d6", "o"), ("d3", "d3", "#eb6834", "s"), ("amos", "amos", "#1baf7a", "D"),
      ("linne", "linne", "#eda100", "^"), ("decider31b", "decider-31b", "#e87ba4", "v"), ("clef", "Clef", "#008300", "P")]


def fig_anchor():
    """Unplaced AUROC (missed required vs unneeded orders) of each System One model by what it reads. The row marked *
    has 3 models (pplx12, d3, amos); each black line is the mean over the models shown in its row (say so in the caption)."""
    sc = (load("pb_screen.json") or {}).get("scorers")
    if not sc:
        return
    rows = [("", "Agent's full trajectory"), ("_warn", "+ debiasing instruction*"), ("_nonotes", "− agent's notes"),
            ("_noorders", "− agent's orders"), ("_retr", "Chart only (no decisions)"), ("_task", "Task text only")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for y, (suf, _) in enumerate(rows):
        vals = []
        for i, (m, name, col, mk) in enumerate(S1):
            v = (sc.get(m + suf) or {}).get("auroc_unplaced_mean_per_task")
            if v is None:
                continue
            vals.append(v)
            ax.scatter(v, y + (i - 2.5) * 0.07, s=26, color=col, marker=mk, edgecolor="white", linewidth=0.6, zorder=4,
                       label=name if y == 0 else None)
        mean = sum(vals) / len(vals)
        ax.plot([mean, mean], [y - 0.3, y + 0.3], color=C["ink"], lw=1.6, zorder=5)
        ax.text(0.83, y, f"{mean:.2f}", va="center", ha="left", fontsize=7.6, color=C["ink"], fontweight="bold")
    ax.axvline(0.5, color=C["bad"], lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.text(0.5, -0.75, "chance", color=C["bad"], fontsize=7, ha="center", va="bottom", fontweight="bold")
    ax.set_yticks(range(len(rows)), [r for _, r in rows], fontsize=7.8)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.set_xlim(0.3, 0.82)
    ax.grid(axis="y", visible=False)
    label(ax, x="AUROC, missed required vs unneeded orders")
    title(ax, "What the checker reads")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.32, -0.27), fontsize=7, handletextpad=0.2,
              columnspacing=0.8, markerscale=0.9)
    fig.savefig(OUT / "fig6_anchor.pdf")
    fig.savefig(OUT / "fig6_anchor_preview.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- Figures 7-8: why agents miss orders, and fresh eyes
AGENTS = [("", "Qwen3.8\n27B"), ("_gemma4", "Gemma 4\n31B")]
ARM = {"none": ("agent finishes", "#C9CFDB"), "full": ("asked again,\nfull context", "#B9B2F5"),
       "fresh": ("asked again,\nown decisions removed", C["llm"]), "listrand": ("random checklist", "#9EC5F5"),
       "list": ("S1 checklist (pplx12)", C["clef_k5"]), "listd3": ("S1 checklist (d3)", "#0B4FA8")}


def bar_group(ax, groups, arms, get, fmt="{:.0f}", scale=100, ylim=None):
    width = 0.8 / len(arms)
    present = {g: [a for a in arms if get(g, a) is not None] for g in groups}
    for arm in arms:
        xs, vs = [], []
        for i, g in enumerate(groups):
            if arm not in present[g]:
                continue
            k, n = present[g].index(arm), len(present[g])
            xs.append(i + (k - (n - 1) / 2) * width)
            vs.append(scale * get(g, arm))
        ax.bar(xs, vs, width * 0.9, color=ARM.get(arm, (arm, "#999"))[1], label=ARM.get(arm, (arm,))[0], zorder=3,
               edgecolor="white", linewidth=0.6)
        for x, v in zip(xs, vs):
            ax.text(x, v + (ylim or 100) * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=6.6,
                    fontweight="bold", color=C["ink"])


def fig_why():
    """(a) the agent knows the orders it misses; (b) planning first recovers them, a reminder does not;
    (c) asked again at FINISH, it recovers more when its own decisions are removed (anchoring)."""
    rx = {suf: load(f"pb_replay_order_x{suf}.json") for suf, _ in AGENTS}
    rr = {suf: load(f"pb_r{suf}.json") for suf, _ in AGENTS}
    if not all(rx.values()) or not all(rr.values()):
        return
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.8, 0.8], wspace=0.62)

    ax = fig.add_subplot(gs[0])  # numbers from research_loop Cycle 8 and P8 (results/pb_prior.json)
    rows = [("placed in a run", 68, "#C9CFDB"), ("proposed in a run", 75, "#B9B2F5"),
            ("proposed in any of 5 runs", 91, C["llm"]), ("missed ones, named from\nthe task text alone", 84, C["good"])]
    for i, (name, v, col) in enumerate(rows):
        ax.barh(len(rows) - 1 - i, v, 0.66, color=col, zorder=3)
        ax.text(v + 2, len(rows) - 1 - i, f"{v}%", va="center", fontsize=7.6, fontweight="bold", color=C["ink"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=7.2)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 50, 100])
    ax.grid(axis="y", visible=False)
    label(ax, x="Required orders (%)")
    title(ax, "(a) The agent knows")

    ax = fig.add_subplot(gs[1])
    names = {"order_none": ("as logged", "#C9CFDB"), "order_remind": ("+ task reminder", "#B9B2F5"),
             "order_plan": ("+ plan first", C["good"])}
    width = 0.26
    for k, (cond, (name, col)) in enumerate(names.items()):
        vs = [100 * rx[suf][cond]["open"] for suf, _ in AGENTS]
        xs = [i + (k - 1) * width for i in range(len(AGENTS))]
        ax.bar(xs, vs, width * 0.9, color=col, label=name, zorder=3, edgecolor="white", linewidth=0.6)
        for x, v in zip(xs, vs):
            ax.text(x, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_xticks(range(len(AGENTS)), [n for _, n in AGENTS], fontsize=7.6, fontweight="bold")
    ax.set_ylim(0, 36)
    label(ax, y="Finally-missed orders\nplaced (%)")
    title(ax, "(b) Plan first")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), fontsize=6.6, handlelength=1, prop={"size": 6.6})

    ax = fig.add_subplot(gs[2])
    bar_group(ax, [suf for suf, _ in AGENTS], ["none", "full", "fresh"], lambda g, a: rr[g][a]["open"], ylim=70)
    ax.set_xticks(range(len(AGENTS)), [n for _, n in AGENTS], fontsize=7.6, fontweight="bold")
    ax.set_ylim(0, 70)
    label(ax, y="Open required-order\ncheckpoints passed (%)")
    title(ax, "(c) Its own orders anchor it")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), fontsize=6.6, handlelength=1, prop={"size": 6.6})
    fig.savefig(OUT / "fig7_why.pdf")
    fig.savefig(OUT / "fig7_why_preview.png", dpi=200)
    plt.close(fig)


def fig_loop():
    """(a) the S1 checklist at the agent's FINISH, both agents, against a random list of the same pool and a fresh
    restart; (b) what the added orders are, rated alone by a blinded LLM judge (Gemma, H-b)."""
    rr = {suf: load(f"pb_r{suf}.json") for suf, _ in AGENTS}
    hb = load("pb_extra_single.json")
    if not all(rr.values()) or not hb:
        return
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.5)

    ax = fig.add_subplot(gs[0])
    arms = ["none", "listrand", "list", "listd3", "fresh"]
    bar_group(ax, [suf for suf, _ in AGENTS], arms, lambda g, a: (rr[g].get(a) or {}).get("open"), ylim=70)
    ax.set_xticks(range(len(AGENTS)), [f"{n.replace(chr(10), ' ')}\n({rr[s]['none']['cases']} cases)" for s, n in AGENTS],
                  fontsize=7.6, fontweight="bold")
    ax.set_ylim(0, 72)
    label(ax, y="Open required-order\ncheckpoints passed (%)")
    title(ax, "(a) Fresh eyes at the agent's finish")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.6, handlelength=1,
              columnspacing=0.8, prop={"size": 6.6})

    ax = fig.add_subplot(gs[1])
    srcs = [("base", "agent's own"), ("list", "S1 checklist"), ("listrand", "random list"), ("fresh", "fresh restart")]
    cols = {"A": ("indicated", C["good"]), "B": ("reasonable", "#8BD3B5"), "C": ("low value", "#C9CFDB"),
            "D": ("potentially harmful", C["bad"])}
    y, yt, yl = 0, [], []
    for suf, (key, aname) in zip(("", "_gemma4"), (("qwen__judge_gemma4", "Qwen agent"), ("gemma4__judge_gemma4", "Gemma agent"))):
        cells = hb[key]["cells"]
        ax.text(-2, y + 0.75, aname, ha="right", va="center", fontsize=7.2, fontweight="bold", color=C["ink"])
        for src, sname in srcs:
            d = cells[src]["dist"]
            n = sum(d.values())
            left = 0
            for g, (gname, col) in cols.items():
                v = 100 * d[g] / n
                ax.barh(y, v, left=left, height=0.7, color=col, zorder=3, edgecolor="white", linewidth=0.5,
                        label=gname if y == 0 else None)
                left += v
            ax.text(101.5, y, f"n={n}", va="center", fontsize=6.2, color=C["sub"])
            yt.append(y)
            yl.append(sname)
            y -= 1
        y -= 0.7
    ax.set_yticks(yt, yl, fontsize=6.9)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    label(ax, x="Orders outside the rubric (%)")
    title(ax, "(b) What the extra orders are")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.35, -0.2), fontsize=6.6, handlelength=1,
              columnspacing=0.8, prop={"size": 6.6})
    fig.savefig(OUT / "fig8_loop.pdf")
    fig.savefig(OUT / "fig8_loop_preview.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_main, fig_orders, fig_tools, fig_clinenv, fig_anchor, fig_why, fig_loop):
        f()
    print("figures:", sorted(p.name for p in OUT.glob("fig*.pdf")))
