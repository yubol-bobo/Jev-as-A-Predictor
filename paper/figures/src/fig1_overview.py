"""Figure 2 (overview): who makes the decision inside a clinical agent loop (vector illustration, SVG -> PDF).

  /ocean/projects/cis260032p/yliy/venvs/figs/bin/python paper/figures/src/fig1_overview.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from illus import *  # noqa: E402,F403

W, H = 1500, 560
begin(W, H)


# ------------------------------------------------------------------ panel 0: the clinical agent loop
x0 = 18
card(x0, 18, 330, 524, fill="url(#panelA)", stroke="#CDEDEA", r=22, shadow=False)
header(x0 + 22, 64, CLIN, "Clinical agent", "the loop we keep fixed")
# EHR monitor
mx, my = x0 + 92, 112
card(mx, my, 150, 104, fill="url(#clingrad)", stroke=CLIN, r=12)
add(f'<rect x="{mx + 14}" y="{my + 14}" width="122" height="62" rx="6" fill="#FFFFFF" stroke="{CLIN}" stroke-width="1.2"/>')
add(f'<polyline points="{mx + 20},{my + 52} {mx + 44},{my + 52} {mx + 54},{my + 30} {mx + 66},{my + 66} {mx + 78},{my + 44} '
    f'{mx + 90},{my + 52} {mx + 130},{my + 52}" fill="none" stroke="{CLIN}" stroke-width="2.6" stroke-linejoin="round"/>')
add(f'<rect x="{mx + 63}" y="{my + 104}" width="24" height="12" fill="{CLIN}" opacity="0.5"/>'
    f'<rect x="{mx + 45}" y="{my + 115}" width="60" height="6" rx="3" fill="{CLIN}" opacity="0.6"/>')
text(mx + 75, my + 98, "EHR · FHIR", 12.5, 700, CLIN, "middle")
# loop
cx, cy = x0 + 167, 352
add(f'<circle cx="{cx}" cy="{cy}" r="86" fill="none" stroke="{CLIN}" stroke-width="2.5" stroke-dasharray="7 7" opacity="0.6"/>')
for (lx, ly, lab, col) in ((cx, cy - 86, "state", CLIN), (cx + 86, cy, "decide", S1), (cx, cy + 86, "act", ACT),
                           (cx - 86, cy, "observe", CLIN)):
    w = pill(lx - (len(lab) * 7.6 + 22) / 2, ly - 12, lab, col, filled=(lab == "decide"))
text(cx, cy - 6, "next turn =", 14.5, 400, SUB, "middle")
text(cx, cy + 16, "tool calls + args", 15.5, 900, INK, "middle")
text(cx, cy + 36, "or a final answer", 14.5, 400, SUB, "middle")
footer(x0 + 22, 486, "PhysicianBench · ClinEnv", "Same", "tools, data, graders", CLIN)

# ------------------------------------------------------------------ panel 1: LLM decides
x1 = 368
card(x1, 18, 360, 524, fill="url(#panelL)", stroke="#E1DBFF", r=22, shadow=False)
header(x1 + 22, 64, LLM, "LLM decides", "the benchmark's own procedure")
server(x1 + 40, 150, LLM, LLM_L, "LLM")
# thought bubble
add(f'<ellipse cx="{x1 + 250}" cy="{150}" rx="70" ry="34" fill="#FFFFFF" stroke="{LLM}" stroke-width="1.5" opacity="0.95"/>')
add(f'<circle cx="{x1 + 186}" cy="{184}" r="7" fill="#FFFFFF" stroke="{LLM}" stroke-width="1.5"/>')
add(f'<circle cx="{x1 + 172}" cy="{198}" r="4" fill="#FFFFFF" stroke="{LLM}" stroke-width="1.5"/>')
for i in range(3):
    add(f'<rect x="{x1 + 208}" y="{136 + i * 10}" width="{84 - i * 16}" height="4" rx="2" fill="{LLM}" opacity="0.4"/>')
text(x1 + 250, 128, "thinking …", 12.5, 700, LLM, "middle")
arrow(x1 + 178, 262, x1 + 232, 262, LLM)
doc(x1 + 242, 222, 82, 92, LLM, lines=5, check=True)
text(x1 + 283, 342, "one sampled turn", 13.5, 700, LLM, "middle")
text(x1 + 40, 300, "every step: generate", 13.5, 400, SUB)
text(x1 + 40, 318, "reasoning, then the action", 13.5, 700, LLM)
footer(x1 + 22, 486, "Generates every action", "Thinks", "thousands of tokens", LLM)

# ------------------------------------------------------------------ panel 2: S1 selects
x2 = 748
card(x2, 18, 360, 524, fill="url(#panelS)", stroke="#D3E5FF", r=22, shadow=False)
header(x2 + 22, 64, S1, "System One selects", "S1-Select: picks the whole turn")
text(x2 + 30, 116, "LLM proposes K = 5 turns", 13.5, 700, LLM)
for i in range(5):
    doc(x2 + 30 + i * 13, 128 + i * 9, 70, 62, LLM, lines=3, faded=(i != 3))
arrow(x2 + 150, 182, x2 + 196, 182, S1)
chip(x2 + 212, 134, 92)
bars(x2 + 30, 262, [0.08, 0.19, 0.11, 0.52, 0.10], S1, chosen=3, width=150)
for i, lab in enumerate("ABCDE"):
    text(x2 + 22, 271 + i * 17, lab, 12, 700, SUB, "end")
arrow(x2 + 252, 300, x2 + 252, 338, S1)
doc(x2 + 214, 344, 76, 70, S1, lines=3, check=True)
text(x2 + 30, 370, "one forward pass,", 13.5, 400, SUB)
text(x2 + 30, 388, "a probability per option", 13.5, 700, S1)
footer(x2 + 22, 486, "Decides among proposals", "One pass", "no generated text", S1)

# ------------------------------------------------------------------ panel 3: S1 steers
x3 = 1128
card(x3, 18, 354, 524, fill="url(#panelS)", stroke="#D3E5FF", r=22, shadow=False)
header(x3 + 22, 64, S1, "System One steers", "S1-Tool: picks the kind of action")
chip(x3 + 30, 120, 86)
labels = [("labs", 0), ("notes", 0), ("problems", 0), ("order", 1), ("write note", 0), ("finish", 0)]
px, py = x3 + 140, 118
for i, (lab, on) in enumerate(labels):
    xx, yy = px + (i % 2) * 96, py + (i // 2) * 34
    pill(xx, yy, lab, ACT if on else S1, filled=bool(on), w=88)
arrow(x3 + 184, 230, x3 + 184, 266, ACT)
server(x3 + 40, 286, LLM, LLM_L, "LLM")
arrow(x3 + 166, 336, x3 + 210, 336, LLM)
doc(x3 + 222, 298, 80, 80, ACT, lines=4, check=True)
text(x3 + 262, 400, "order( …args… )", 13, 700, ACT, "middle")
footer(x3 + 22, 486, "Sets strategy, LLM fills in", "Steers", "data → action → stop", S1)

finish("fig1_overview")
