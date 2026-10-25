"""Figure 1 (story): the agent misses orders it knows; a checker that reads its work inherits the blind spot; a System
One model that reads only the task finds them (vector illustration, SVG -> PDF). Numbers are aggregates from
docs/research_loop.md (Cycle 8, P8, G1, R, XR3); no patient data.

  /ocean/projects/cis260032p/yliy/venvs/figs/bin/python paper/figures/src/fig_fresh.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from illus import *  # noqa: E402,F403

BAD, GOOD = "#E0533D", "#13A36E"
W, H = 1500, 560
begin(W, H)


def stat(x, y, big, line, color):
    text(x, y, big, 26, 900, color)
    text(x + 92, y - 3, line, 14.5, 400, INK)


def row(x, y, w, ok, color):
    """One order line in a list: a check (placed) or a dashed empty slot (missed)."""
    if ok:
        add(f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="7" fill="#FFFFFF" stroke="{color}" stroke-width="1.3"/>')
        add(f'<rect x="{x + 12}" y="{y + 11}" width="{w * 0.55:.0f}" height="5" rx="2.5" fill="{color}" opacity="0.4"/>')
        cx, cy = x + w - 16, y + 13
        add(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{GOOD}"/>')
        add(f'<polyline points="{cx - 4},{cy} {cx - 1},{cy + 3.5} {cx + 4.5},{cy - 3.5}" fill="none" stroke="#fff" '
            f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        add(f'<rect x="{x}" y="{y}" width="{w}" height="26" rx="7" fill="#FFF5F3" stroke="{BAD}" stroke-width="1.5" '
            f'stroke-dasharray="5 4"/>')
        text(x + 12, y + 18, "required order, never placed", 12.5, 700, BAD)


# ------------------------------------------------------------------ panel 0: the agent misses orders it knows
x0 = 18
card(x0, 18, 470, 524, fill="url(#panelL)", stroke="#E1DBFF", r=22, shadow=False)
header(x0 + 22, 64, LLM, "The agent misses orders", "that it already knows")
server(x0 + 34, 140, LLM, LLM_L, "LLM")
arrow(x0 + 168, 190, x0 + 214, 190, LLM)
card(x0 + 224, 116, 222, 176, fill="url(#paper)", stroke=LLM, r=12)
text(x0 + 240, 142, "orders at FINISH", 13.5, 700, LLM)
for i, ok in enumerate((True, True, True, False)):
    row(x0 + 236, 154 + i * 33, 198, ok, LLM)
stat(x0 + 30, 348, "75%", "of required orders proposed in a run", LLM)
stat(x0 + 30, 386, "91%", "proposed in one of five runs", LLM)
stat(x0 + 30, 424, "84%", "of the missed ones named from the task", GOOD)
footer(x0 + 22, 486, "Two causes, on two agents", "Plans late", "and anchors on its own orders", LLM)

# ------------------------------------------------------------------ panel 1: a checker that reads the agent's work
x1 = 508
card(x1, 18, 470, 524, fill="#FFF8F6", stroke="#F6D5CE", r=22, shadow=False)
header(x1 + 22, 64, BAD, "A checker reads its work", "a System One model, or the LLM itself")
text(x1 + 30, 116, "agent's trajectory", 13.5, 700, LLM)
for i in range(4):
    doc(x1 + 30 + i * 12, 128 + i * 9, 74, 66, LLM, lines=3, faded=(i != 3))
arrow(x1 + 146, 178, x1 + 192, 178, BAD)
chip(x1 + 206, 128, 88)
text(x1 + 330, 150, "scores each", 13.5, 400, SUB)
text(x1 + 330, 168, "candidate order", 13.5, 400, SUB)
labels = ("like a placed order", "unneeded", "unneeded", "missed, required")
probs = (0.84, 0.55, 0.47, 0.21)
for i, (lab, p) in enumerate(zip(labels, probs)):
    yy = 262 + i * 24
    col = BAD if i == 3 else S1
    text(x1 + 172, yy + 10, lab, 13, 700 if i == 3 else 400, BAD if i == 3 else SUB, "end")
    add(f'<rect x="{x1 + 184}" y="{yy}" width="200" height="12" rx="6" fill="{col}" opacity="0.10"/>')
    add(f'<rect x="{x1 + 184}" y="{yy}" width="{200 * p:.0f}" height="12" rx="6" fill="{col}" opacity="{1 if i == 3 else 0.5}"/>')
    text(x1 + 394, yy + 11, f"{p:.2f}", 13, 900 if i == 3 else 400, BAD if i == 3 else SUB)
stat(x1 + 30, 410, "0.40", "AUROC for missed orders: below chance", BAD)
text(x1 + 122, 432, "6 System One models (0.37–0.43), Qwen 0.46", 13, 400, SUB)
footer(x1 + 22, 486, "Anchored by the agent's orders, not its notes", "Inherits", "the blind spot", BAD)

# ------------------------------------------------------------------ panel 2: fresh eyes
x2 = 998
card(x2, 18, 484, 524, fill="url(#panelS)", stroke="#D3E5FF", r=22, shadow=False)
header(x2 + 22, 64, S1, "Fresh eyes", "a System One model reads only the task")
doc(x2 + 30, 128, 80, 92, CLIN, lines=5)
text(x2 + 70, 240, "task text", 13.5, 700, CLIN, "middle")
arrow(x2 + 118, 174, x2 + 158, 174, S1)
chip(x2 + 170, 128, 88)
text(x2 + 214, 240, "0.4 s / order", 13.5, 700, S1, "middle")
arrow(x2 + 268, 174, x2 + 300, 174, S1)
card(x2 + 308, 112, 150, 176, fill="url(#paper)", stroke=S1, r=12)
text(x2 + 322, 138, "checklist, top 8", 13, 700, S1)
for i in range(5):
    yy = 152 + i * 25
    add(f'<rect x="{x2 + 322}" y="{yy}" width="14" height="14" rx="3" fill="#FFFFFF" stroke="{S1}" stroke-width="1.6"/>')
    add(f'<rect x="{x2 + 344}" y="{yy + 5}" width="{96 - (i % 3) * 14}" height="5" rx="2.5" fill="{S1}" opacity="0.4"/>')
text(x2 + 383, 310, "shown at the agent's FINISH", 13, 700, S1, "middle")
text(x2 + 30, 352, "Open required-order checkpoints passed", 14, 700, INK)
for i, (agent, a, b, d) in enumerate((("Qwen3.8-27B", 38, 53, "+15"), ("Gemma 4 31B", 29, 58, "+29"))):
    yy = 376 + i * 36
    text(x2 + 30, yy + 14, agent, 13.5, 700, SUB)
    add(f'<rect x="{x2 + 140}" y="{yy + 2}" width="200" height="16" rx="8" fill="{S1}" opacity="0.10"/>')
    add(f'<rect x="{x2 + 140}" y="{yy + 2}" width="{2 * b}" height="16" rx="8" fill="{S1}"/>')
    add(f'<rect x="{x2 + 140}" y="{yy + 2}" width="{2 * a}" height="16" rx="8" fill="#C9CFDB"/>')
    text(x2 + 352, yy + 15, f"{a} → {b}%", 13.5, 400, INK)
    text(x2 + 448, yy + 15, d, 15, 900, S1)
footer(x2 + 22, 486, "Beats a random list · extra orders not harmful", "Recovers", "what the agent missed", S1)

finish("fig0_fresh")
