"""Figure 1: who makes the decision inside a clinical agent loop (vector illustration, SVG -> PDF).

  /ocean/projects/cis260032p/yliy/venvs/figs/bin/python paper/figures/src/fig1_overview.py
"""

from pathlib import Path

import cairosvg

OUT = Path(__file__).resolve().parents[1]
W, H = 1500, 560
INK, SUB = "#16213E", "#5A6378"
LLM, LLM_L = "#6C5CE7", "#EFEBFF"
S1, S1_L = "#1E7BEA", "#E6F1FF"
CLIN, CLIN_L = "#0FA3A3", "#E3F7F6"
ACT = "#F08A24"

parts = []


def add(s):
    parts.append(s)


def text(x, y, s, size=18, weight=400, fill=INK, anchor="start", style="normal"):
    add(f'<text x="{x}" y="{y}" font-family="Lato" font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" font-style="{style}">{s}</text>')


def card(x, y, w, h, fill="#FFFFFF", stroke="#D9E2F2", r=14, sw=1.5, shadow=True):
    if shadow:
        add(f'<rect x="{x + 3}" y="{y + 5}" width="{w}" height="{h}" rx="{r}" fill="#1B2A4A" opacity="0.07"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x1, y1, x2, y2, color, w=3.5, head=11):
    import math
    a = math.atan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * math.cos(a), y2 - head * math.sin(a)
    add(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')
    p1 = (bx + head * 0.55 * math.sin(a), by - head * 0.55 * math.cos(a))
    p2 = (bx - head * 0.55 * math.sin(a), by + head * 0.55 * math.cos(a))
    add(f'<polygon points="{x2},{y2} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')


def doc(x, y, w, h, color, lines=4, check=False, faded=False):
    op = 0.45 if faded else 1
    add(f'<g opacity="{op}">')
    card(x, y, w, h, fill="url(#paper)", stroke=color, r=8, sw=1.6, shadow=not faded)
    for i in range(lines):
        lw = (w - 22) * (0.95 if i % 2 == 0 else 0.7)
        add(f'<rect x="{x + 11}" y="{y + 12 + i * 10}" width="{lw:.0f}" height="4" rx="2" fill="{color}" opacity="0.35"/>')
    if check:
        cx, cy = x + w - 6, y + h - 6
        add(f'<circle cx="{cx}" cy="{cy}" r="14" fill="{color}"/>')
        add(f'<polyline points="{cx - 6},{cy} {cx - 1.5},{cy + 5} {cx + 7},{cy - 5}" fill="none" stroke="#fff" '
            f'stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>')
    add("</g>")


def server(x, y, color, light, label):
    """Stacked-server LLM icon."""
    for i in range(3):
        yy = y + i * 34
        add(f'<rect x="{x}" y="{yy}" width="108" height="30" rx="8" fill="url(#llmgrad)" stroke="{color}" stroke-width="1.5"/>')
        for k in range(3):
            add(f'<rect x="{x + 12 + k * 9}" y="{yy + 11}" width="4" height="9" rx="2" fill="{color}" opacity="0.7"/>')
    add(f'<rect x="{x + 52}" y="{y - 6}" width="66" height="106" rx="10" fill="url(#llmgrad2)" stroke="{color}" stroke-width="1.6"/>')
    text(x + 85, y + 58, label, 17, 900, color, "middle")
    add(f'<circle cx="{x + 72}" cy="{y + 22}" r="5" fill="{color}"/><circle cx="{x + 98}" cy="{y + 22}" r="5" fill="{color}"/>'
        f'<circle cx="{x + 85}" cy="{y + 6}" r="5" fill="{color}"/>'
        f'<polyline points="{x + 72},{y + 22} {x + 85},{y + 6} {x + 98},{y + 22} {x + 72},{y + 22}" fill="none" '
        f'stroke="{color}" stroke-width="2"/>')


def chip(x, y, s=96):
    """System One model: a decision chip with a probability readout."""
    add(f'<rect x="{x + 4}" y="{y + 6}" width="{s}" height="{s}" rx="20" fill="#1B2A4A" opacity="0.08"/>')
    add(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" rx="20" fill="url(#s1grad)" stroke="{S1}" stroke-width="1.8"/>')
    for k in range(4):  # pins
        add(f'<rect x="{x - 8}" y="{y + 18 + k * 18}" width="8" height="6" rx="2" fill="{S1}" opacity="0.55"/>')
        add(f'<rect x="{x + s}" y="{y + 18 + k * 18}" width="8" height="6" rx="2" fill="{S1}" opacity="0.55"/>')
    text(x + s / 2, y + 34, "S1", 22, 900, "#FFFFFF", "middle")
    for k, hgt in enumerate((12, 20, 30, 42)):
        add(f'<rect x="{x + 20 + k * 15}" y="{y + 84 - hgt}" width="10" height="{hgt}" rx="2.5" fill="#FFFFFF" opacity="{0.55 + 0.15 * k}"/>')


def bars(x, y, probs, color, chosen, width=120):
    for i, p in enumerate(probs):
        yy = y + i * 17
        add(f'<rect x="{x}" y="{yy}" width="{width}" height="10" rx="5" fill="{color}" opacity="0.10"/>')
        add(f'<rect x="{x}" y="{yy}" width="{width * p:.0f}" height="10" rx="5" fill="{color}" '
            f'opacity="{1 if i == chosen else 0.45}"/>')
        text(x + width + 8, yy + 9.5, f"{p:.2f}", 12.5, 900 if i == chosen else 400, INK if i == chosen else SUB)


def pill(x, y, label, color, filled=False, w=None):
    w = w or (len(label) * 7.6 + 22)
    add(f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="12" fill="{color if filled else "#FFFFFF"}" '
        f'stroke="{color}" stroke-width="1.6" opacity="{1 if filled else 0.9}"/>')
    text(x + w / 2, y + 16.5, label, 12.5, 700, "#FFFFFF" if filled else color, "middle")
    return w


def header(x, y, dot, title, sub):
    add(f'<circle cx="{x + 9}" cy="{y - 8}" r="9" fill="{dot}"/>')
    text(x + 26, y, title, 25, 900, INK)
    text(x + 26, y + 24, sub, 14.5, 400, SUB)


def footer(x, y, line1, key, rest, color):
    text(x, y, line1, 15, 400, INK)
    add(f'<text x="{x}" y="{y + 24}" font-family="Lato" font-size="18"><tspan font-weight="900" fill="{color}">{key}</tspan>'
        f'<tspan fill="{color}" dx="6">· {rest}</tspan></text>')


# ------------------------------------------------------------------ canvas and gradients
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('''<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F6F9FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
<linearGradient id="paper" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#F3F6FC"/></linearGradient>
<linearGradient id="llmgrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F7F5FF"/><stop offset="1" stop-color="#DCD6FF"/></linearGradient>
<linearGradient id="llmgrad2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E4DFFF"/></linearGradient>
<linearGradient id="s1grad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5AA6FF"/><stop offset="1" stop-color="#1361C9"/></linearGradient>
<linearGradient id="clingrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#DDF4F2"/></linearGradient>
<linearGradient id="panelA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F2FBFA"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
<linearGradient id="panelL" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F6F4FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
<linearGradient id="panelS" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F0F6FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
</defs>''')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

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

add("</svg>")
svg = "\n".join(parts)
(OUT / "fig1_overview.svg").write_text(svg)
cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / "fig1_overview.pdf"))
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "fig1_overview_preview.png"), output_width=1500)
print("wrote", OUT / "fig1_overview.pdf")
