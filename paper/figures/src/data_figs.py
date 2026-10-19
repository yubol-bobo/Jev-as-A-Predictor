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


if __name__ == "__main__":
    for f in (fig_main, fig_orders, fig_tools, fig_clinenv):
        f()
    print("figures:", sorted(p.name for p in OUT.glob("fig*.pdf")))
