"""Figure 1 (opening): one real PhysicianBench case, and what changes when a System One model makes the decision.

The case is task nafld_ferritin_workup (public instruction and graders, paraphrased) in the S1-Select run (clef_k5,
run 1). The timeline shows the tool types the agent executed at each step. Step 8 shows the five proposals by tool
type, Clef's probabilities, and which proposal was the LLM's own first sample. The two order names come from the
task's public graders. No patient data appears: no values, notes or arguments. The statistics in the outcome boxes
come from results/pb_offline_orders_clef_online.json (order steps where a proposal places a required order).

  /ocean/projects/cis260032p/yliy/venvs/figs/bin/python paper/figures/src/fig0_case.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from illus import *  # noqa: E402,F403

REPO = Path(__file__).resolve().parents[3]
on = json.loads((REPO / "results/pb_offline_orders_clef_online.json").read_text())["whether_to_order"]["required_available"]
N_STEPS, P_FIRST, P_S1 = on["n"], round(100 * on["first"]), round(100 * on["clef_online"])

W, H = 1500, 668
begin(W, H)
RET, DOC = CLIN, "#8A94A8"


def num(cx, cy, k, color, filled=True, r=13):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color if filled else "#FFFFFF"}" stroke="{color}" stroke-width="1.8"/>')
    text(cx, cy + 4.5, str(k), 12.5, 900, "#FFFFFF" if filled else color, "middle")


def tag(x, y, label, color, w=None, h=22, size=12):
    w = w or (len(label) * 6.9 + 18)
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{color}" opacity="0.12"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    text(x + w / 2, y + h / 2 + 4.3, label, size, 700, color, "middle")
    return w


def tags(x, y, items, gap=6):
    for label, color in items:
        x += tag(x, y, label, color) + gap


def bell(cx, cy, color):
    add(f'<circle cx="{cx}" cy="{cy}" r="22" fill="{color}" opacity="0.14"/>')
    add(f'<path d="M{cx - 9},{cy + 6} Q{cx - 9},{cy - 12} {cx},{cy - 12} Q{cx + 9},{cy - 12} {cx + 9},{cy + 6} '
        f'L{cx + 12},{cy + 9} L{cx - 12},{cy + 9} Z" fill="{color}"/>')
    add(f'<circle cx="{cx}" cy="{cy + 13}" r="3.2" fill="{color}"/>')


def check(cx, cy, color, r=9):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}"/>')
    add(f'<polyline points="{cx - r * 0.45},{cy} {cx - r * 0.1},{cy + r * 0.38} {cx + r * 0.5},{cy - r * 0.38}" '
        f'fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')


def dice(x, y, s, color):
    add(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="{s * 0.22}" fill="#FFFFFF" stroke="{color}" stroke-width="2"/>')
    for dx, dy in ((0.28, 0.28), (0.72, 0.72), (0.5, 0.5), (0.72, 0.28), (0.28, 0.72)):
        add(f'<circle cx="{x + dx * s:.1f}" cy="{y + dy * s:.1f}" r="{s * 0.08:.1f}" fill="{color}"/>')


# ------------------------------------------------------------------ panel A: the task
xa, wa = 18, 424
card(xa, 18, wa, 576, fill="url(#panelA)", stroke="#CDEDEA", r=22, shadow=False)
header(xa + 22, 64, CLIN, "A real consultation", "PhysicianBench · hepatology e-consult")
card(xa + 22, 104, wa - 44, 98, fill="#FFFFFF", stroke="#F6C99A", r=14)
bell(xa + 62, 152, ACT)
text(xa + 98, 138, "Lab alert in the inbox", 16, 900, INK)
text(xa + 98, 162, "Abnormal iron studies and liver tests in", 14, 400, SUB)
text(xa + 98, 181, "a patient with known fatty liver disease.", 14, 400, SUB)
text(xa + 22, 236, "The agent must", 15, 900, INK)
for k, s in enumerate(("Review the patient's record", "Interpret the iron and liver labs",
                       "Order the diagnostic workup", "Write the assessment note"), 1):
    num(xa + 38, 256 + (k - 1) * 30, k, CLIN, filled=False, r=11)
    text(xa + 60, 261 + (k - 1) * 30, s, 14.5, 700 if k == 3 else 400, ACT if k == 3 else INK)
# record strip
card(xa + 22, 378, wa - 44, 52, fill="url(#clingrad)", stroke=CLIN, r=12, shadow=False)
add(f'<rect x="{xa + 38}" y="{390}" width="40" height="28" rx="5" fill="#FFFFFF" stroke="{CLIN}" stroke-width="1.4"/>')
add(f'<polyline points="{xa + 42},{405} {xa + 50},{405} {xa + 54},{396} {xa + 60},{413} {xa + 65},{402} {xa + 74},{405}" '
    f'fill="none" stroke="{CLIN}" stroke-width="2" stroke-linejoin="round"/>')
text(xa + 90, 400, "Real de-identified patient record", 14, 900, CLIN)
text(xa + 90, 419, "FHIR API · search, order and note tools", 13, 400, SUB)
text(xa + 22, 462, "Graded by 7 checkpoints", 15, 900, INK)
tags(xa + 22, 474, [("data retrieval", RET), ("ferritin", LLM), ("Gilbert", LLM), ("lifestyle", LLM)])
tags(xa + 22, 504, [("HFE gene test (order)", ACT), ("FibroScan (order)", ACT)])
tags(xa + 22, 534, [("assessment note", DOC)])
text(xa + 22, 578, "retrieval · reasoning · action · documentation", 12.5, 400, SUB)

# ------------------------------------------------------------------ panel B: the agent at work
xb, wb = 460, 392
card(xb, 18, wb, 576, fill="url(#panelA)", stroke="#CDEDEA", r=22, shadow=False)
header(xb + 22, 64, CLIN, "The agent at work", "one decision per step, then the tools run")
steps = [("demographics", [("patient", RET)]), ("history", [("problems", RET), ("labs", RET), ("meds", RET), ("social", RET)]),
         ("", [("labs", RET)]), ("", [("labs ×4", RET)]), ("", [("labs", RET)]), ("", [("labs ×6", RET)]),
         ("", [("notes", RET), ("past orders", RET)]), ("", []), ("", [("write note", DOC)]), ("", [("finish", DOC)])]
x_line, y0, dy = xb + 46, 116, 45
add(f'<line x1="{x_line}" y1="{y0}" x2="{x_line}" y2="{y0 + 9 * dy}" stroke="{CLIN}" stroke-width="2" opacity="0.35"/>')
for k, (_, items) in enumerate(steps, 1):
    y = y0 + (k - 1) * dy
    if k == 8:
        card(xb + 22, y - 19, wb - 44, 38, fill=S1_L, stroke=S1, r=12, sw=2, shadow=False)
        num(x_line, y, k, S1)
        text(xb + 72, y + 5.5, "decision point: which turn next?", 15, 900, S1)
        arrow(xb + wb - 30, y, xb + wb + 26, y, S1, w=3)
        continue
    num(x_line, y, k, CLIN if k < 8 else DOC)
    tags(xb + 72, y - 11, items)
text(xb + 22, 578, "The executed run passes 7 / 7 checkpoints.", 12.5, 400, SUB)

# ------------------------------------------------------------------ panel C: step 8
xc, wc = 878, 604
card(xc, 18, wc, 576, fill="url(#panelS)", stroke="#D3E5FF", r=22, shadow=False)
header(xc + 22, 64, S1, "Step 8: same state, five proposals", "the LLM proposes K = 5 turns; who picks one?")
text(xc + 64, 118, "Proposals (sampled from the LLM)", 13, 700, LLM)
text(xc + 360, 118, "System One: p(option)", 13, 700, S1)
props = [("Search labs ×3 · vitals", "", RET), ("Search clinical notes", "", RET), ("Search notes · labs ×2", "", RET),
         ("Search clinical notes", "", RET), ("Place 6 orders", "incl. HFE gene test and FibroScan", ACT)]
probs = [0.05, 0.16, 0.09, 0.23, 0.47]
for i, ((label, sub, col), p) in enumerate(zip(props, probs)):
    y = 130 + i * 54
    chosen = i == 4
    card(xc + 64, y, 280, 44, fill="#FFFFFF", stroke=ACT if chosen else "#D9E2F2", r=10, sw=2 if chosen else 1.3,
         shadow=chosen)
    add(f'<rect x="{xc + 64}" y="{y}" width="7" height="44" rx="3.5" fill="{col}"/>')
    text(xc + 90, y + (19 if sub else 27), "ABCDE"[i], 14, 900, SUB)
    text(xc + 112, y + (19 if sub else 27), label, 14.5, 900 if chosen else 400, ACT if chosen else INK)
    if sub:
        text(xc + 112, y + 36, sub, 11.5, 700, ACT)
    # S1 probability bar
    bx, bw = xc + 360, 150
    add(f'<rect x="{bx}" y="{y + 15}" width="{bw}" height="13" rx="6.5" fill="{S1}" opacity="0.10"/>')
    add(f'<rect x="{bx}" y="{y + 15}" width="{bw * p / 0.5:.0f}" height="13" rx="6.5" fill="{S1}" opacity="{1 if chosen else 0.45}"/>')
    text(bx + bw + 10, y + 26.5, f"{p:.2f}", 13.5, 900 if chosen else 400, INK if chosen else SUB)
# who picked what
add(f'<rect x="{xc + 14}" y="{141}" width="42" height="22" rx="11" fill="{LLM}"/>')
text(xc + 35, 156.3, "LLM", 11.5, 900, "#FFFFFF", "middle")
add(f'<rect x="{xc + 14}" y="{357}" width="42" height="22" rx="11" fill="{S1}"/>')
text(xc + 35, 372.3, "S1", 11.5, 900, "#FFFFFF", "middle")

# outcome boxes
yo, ho = 420, 158
card(xc + 22, yo, 272, ho, fill="#FFFFFF", stroke="#D8D0FF", r=16)
dice(xc + 40, yo + 18, 34, LLM)
text(xc + 88, yo + 32, "LLM decides", 17, 900, LLM)
text(xc + 88, yo + 51, "samples one turn: A", 13.5, 400, INK)
text(xc + 40, yo + 84, "Keeps searching; no order", 14, 700, INK)
text(xc + 40, yo + 102, "at this step.", 14, 700, INK)
text(xc + 40, yo + 132, f"Across {N_STEPS} such steps its own", 12.5, 400, SUB)
add(f'<text x="{xc + 40}" y="{yo + 149}" font-family="Lato" font-size="12.5" fill="{SUB}">sample orders in '
    f'<tspan font-weight="900" fill="{LLM}">{P_FIRST}%</tspan>.</text>')
card(xc + 310, yo, 272, ho, fill="#FFFFFF", stroke="#BBD6FF", r=16)
add(f'<rect x="{xc + 328}" y="{yo + 16}" width="38" height="38" rx="10" fill="url(#s1grad)"/>')
text(xc + 347, yo + 41, "S1", 15, 900, "#FFFFFF", "middle")
text(xc + 382, yo + 32, "System One decides", 17, 900, S1)
text(xc + 382, yo + 51, "one pass over all five: E", 13.5, 400, INK)
text(xc + 328, yo + 84, "Places both required orders", 14, 700, INK)
check(xc + 336, yo + 98, ACT, 7)
check(xc + 354, yo + 98, ACT, 7)
text(xc + 368, yo + 103, "HFE gene test, FibroScan", 13, 700, ACT)
text(xc + 328, yo + 132, f"Across {N_STEPS} such steps it places", 12.5, 400, SUB)
add(f'<text x="{xc + 328}" y="{yo + 149}" font-family="Lato" font-size="12.5" fill="{SUB}">the order in '
    f'<tspan font-weight="900" fill="{S1}">{P_S1}%</tspan>.</text>')

# ------------------------------------------------------------------ banner
card(18, 610, W - 36, 44, fill="#F4F7FC", stroke="#DCE4F2", r=22, shadow=False)
text(W / 2 + 112, 638, "Same task · same record · same tools · same proposals · same graders —", 16.5, 400, INK, "end")
text(W / 2 + 120, 638, "only the decision maker changes.", 16.5, 900, S1)

finish("fig0_case")
