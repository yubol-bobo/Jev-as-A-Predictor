"""Shared drawing kit for the paper's vector illustrations (SVG -> PDF): palette, cards, icons, arrows.

  from illus import *; begin(W, H); ...; finish("name")
"""

from pathlib import Path

import cairosvg

OUT = Path(__file__).resolve().parents[1]
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


DEFS = '''<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F6F9FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
<linearGradient id="paper" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#F3F6FC"/></linearGradient>
<linearGradient id="llmgrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#F7F5FF"/><stop offset="1" stop-color="#DCD6FF"/></linearGradient>
<linearGradient id="llmgrad2" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#E4DFFF"/></linearGradient>
<linearGradient id="s1grad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5AA6FF"/><stop offset="1" stop-color="#1361C9"/></linearGradient>
<linearGradient id="clingrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFFFFF"/><stop offset="1" stop-color="#DDF4F2"/></linearGradient>
<linearGradient id="panelA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F2FBFA"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
<linearGradient id="panelL" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F6F4FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
<linearGradient id="panelS" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#F0F6FF"/><stop offset="1" stop-color="#FFFFFF"/></linearGradient>
</defs>'''


def begin(w, h):
    parts.clear()
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    add(DEFS)
    add(f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>')


def finish(name, width=1500):
    add("</svg>")
    svg = "\n".join(parts)
    (OUT / f"{name}.svg").write_text(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT / f"{name}.pdf"))
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / f"{name}_preview.png"), output_width=width)
    print("wrote", OUT / f"{name}.pdf")
