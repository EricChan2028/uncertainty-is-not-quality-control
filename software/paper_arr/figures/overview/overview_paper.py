"""Paper version of the overview figure, drawn at its printed size (ACL \\textwidth = 6.3 in).

Every font size below is the printed size in points; figqa.check() enforces >= 6 pt (math
sub/superscripts included), no text-text or text-graphic overlap, and no text leaving its card.
Numbers come from the paper text; tokens and pool rows are illustrative.
Run:  python overview_paper.py  ->  overview_paper.pdf / .png / .svg
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figqa  # noqa: E402

for f in ("comic.ttf", "comicbd.ttf", "comici.ttf", "comicz.ttf", "seguisym.ttf"):
    p = os.path.join(r"C:\Windows\Fonts", f)
    if os.path.exists(p):
        fm.fontManager.addfont(p)
plt.rcParams.update({
    "font.family": ["Comic Sans MS", "Segoe UI Symbol", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# 1 unit = 0.01 in; font sizes are printed points
W, H = 630, 340
fig = plt.figure(figsize=(W / 100, H / 100), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
REND = fig.canvas.get_renderer()

INK, MUTE = "#222222", "#555555"
RED_T, GRN_T = "#C62828", "#2E7D32"
C_SOLV, C_ILL = "#7CC47F", "#F08A80"
C_A, C_B, C_C = "#F08A80", "#F7B764", "#B39DF2"
C_BLUE, C_GRAYTOK, C_YEL = "#8DB6F0", "#D3D3D3", "#FFD966"
P_GRAY, P_PEACH, P_LAV, P_PINK = "#EDEDED", "#FDECDF", "#EAE6FA", "#FCE5E2"
SH = dict(dx=1.1, dy=-1.4, alpha=0.10)


def darker(c, f=0.66):
    r, g, b = mcolors.to_rgb(c)
    return (r * f, g * f, b * f)


def txt(x, y, s, size=6.3, weight="normal", color=INK, ha="left", va="center", style="normal", z=30, **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, fontstyle=style,
                   zorder=z, **kw)


def rbox(x0, y0, x1, y1, fc="white", ec="none", lw=0.8, r=6, z=3, ls="-", alpha=None):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, ls=ls, alpha=alpha)
    ax.add_patch(p)
    return p


def card(x0, y0, x1, y1, ec="#DADADA", lw=0.8, r=6, z=4):
    rbox(x0 + SH["dx"], y0 + SH["dy"], x1 + SH["dx"], y1 + SH["dy"], fc="#000000", r=r, z=z - 0.2, alpha=SH["alpha"])
    return rbox(x0, y0, x1, y1, fc="white", ec=ec, lw=lw, r=r, z=z)


def header_strip(x0, x1, y0, y1, fc, ec, r=6):
    rbox(x0 + 0.6, y0, x1 - 0.6, y1 - 0.6, fc=fc, r=r - 0.6, z=4.2)
    ax.add_patch(Rectangle((x0 + 0.6, y0), x1 - x0 - 1.2, r, fc=fc, ec="none", zorder=4.25))
    ax.plot([x0 + 0.4, x1 - 0.4], [y0, y0], color=ec, lw=0.7, zorder=4.3)


def soft_panel(x0, y0, x1, y1, fc, title):
    rbox(x0, y0, x1, y1, fc=fc, r=9, z=1)
    txt((x0 + x1) / 2, y1 - 12, title, 7.5, "bold", ha="center")


def token(cx, cy, s, fc, label=None, lsize=6.0, lcolor=INK, z=8, ghost=False):
    if ghost:
        rbox(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, fc="none", ec="#9E9E9E", lw=0.6, r=0.3 * s, z=z, ls=(0, (2, 1.5)))
        return
    rbox(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, fc=fc, ec=darker(fc), lw=0.6, r=0.3 * s, z=z)
    if label:
        txt(cx, cy - 0.2, label, lsize, "bold", color=lcolor, ha="center", z=z + 1)


def arrow(p0, p1, color=INK, lw=0.9, ms=5.5, z=12, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=z, ls=ls,
                                 shrinkA=0, shrinkB=0))


def curved_arrow(p0, p1, bulge, depth, color=INK, lw=0.9, ms=5.5, z=12, ls="-"):
    """Arc from p0 to p1 bowing toward the direction `bulge` by `depth` units.

    matplotlib's arc3 puts the control point at mid + rad * (dy, -dx), i.e. positive rad bows to the
    right of the direction of travel; the sign is derived here instead of guessed.
    """
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    side = np.sign(bulge[0] * dy - bulge[1] * dx)
    rad = side * 2.0 * depth / L
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, zorder=z, ls=ls,
                        connectionstyle=f"arc3,rad={rad:.4f}", shrinkA=0, shrinkB=0)
    a._bulge_note = (f"{p0} -> {p1}: bows toward {bulge} (rad {rad:+.3f}); "
                     f"control point offset {tuple(np.round(np.array([dy, -dx]) * rad, 1))}")
    ax.add_patch(a)


def block_arrow(p0, p1, fc, ms=6.5, z=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="simple,head_length=0.9,head_width=1.6,tail_width=0.72",
                                 mutation_scale=ms, fc=fc, ec=darker(fc), lw=0.7, zorder=z, shrinkA=0, shrinkB=0))


def pill(x, y, s, fc, ec, size=6.0, color=INK, ha="left"):
    return txt(x, y, s, size, color=color, ha=ha, z=25,
               bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.8", fc=fc, ec=ec, lw=0.7))


def twidth(t):
    bb = t.get_window_extent(renderer=REND)
    return (bb.x1 - bb.x0) * W / (fig.get_figwidth() * fig.dpi)


def pill_row(x, y, items, fc, ec, size=6.0, gap=4.0):
    for s in items:
        t = pill(x, y, s, fc, ec, size=size)
        x += twidth(t) + 2 * 0.32 * size * 100 / 72 + gap


def check_badge(cx, cy, r, fc="#5DB36B"):
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=darker(fc), lw=0.7, zorder=16))
    ax.plot([cx - 0.48 * r, cx - 0.1 * r, cx + 0.52 * r], [cy + 0.02 * r, cy - 0.38 * r, cy + 0.42 * r],
            color="white", lw=1.2, solid_capstyle="round", solid_joinstyle="round", zorder=16.1)


def warn_badge(cx, cy, s):
    rbox(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, fc="#F26B5E", ec="#B83A2F", lw=0.7, r=0.3 * s, z=16)
    txt(cx, cy - 0.3, "!", 7, "bold", color="white", ha="center", z=17)


def robot(cx, cy, s, fc):
    ax.add_patch(FancyBboxPatch((cx - s, cy - 0.78 * s), 2 * s, 1.56 * s, boxstyle=f"round,pad=0,rounding_size={0.42 * s}",
                                fc=fc, ec=INK, lw=0.7, zorder=14))
    ax.plot([cx, cx], [cy + 0.78 * s, cy + 1.08 * s], color=INK, lw=0.7, zorder=14)
    ax.add_patch(Circle((cx, cy + 1.18 * s), 0.14 * s, fc="#F6B93B", ec=INK, lw=0.5, zorder=14.1))
    for dx in (-0.42, 0.42):
        ax.add_patch(Circle((cx + dx * s, cy + 0.1 * s), 0.2 * s, fc="white", ec=INK, lw=0.5, zorder=14.1))
        ax.add_patch(Circle((cx + dx * s, cy + 0.1 * s), 0.09 * s, fc=INK, lw=0, zorder=14.2))
    ax.plot([cx - 0.3 * s, cx + 0.3 * s], [cy - 0.42 * s, cy - 0.42 * s], color=INK, lw=0.6, zorder=14.2)


def snowflake(cx, cy, r, color="#2F6FC0"):
    for k in range(6):
        a = math.radians(90 + 60 * k)
        ax.plot([cx, cx + r * math.cos(a)], [cy, cy + r * math.sin(a)], color=color, lw=0.7, zorder=15,
                solid_capstyle="round")


def doc_icon(x, y, w, h, mark=None, mcolor=INK):
    fold = 0.32 * min(w, h)
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - fold), (x + w - fold, y + h), (x, y + h)], closed=True,
                         fc="white", ec=INK, lw=0.7, zorder=14))
    ax.add_patch(Polygon([(x + w - fold, y + h), (x + w - fold, y + h - fold), (x + w, y + h - fold)], closed=True,
                         fc="#DADADA", ec=INK, lw=0.5, zorder=14.1))
    top = 0.8 if mark else 0.68
    for i in range(1 if mark else 3):
        yy = y + h * (top - 0.17 * i)
        ax.plot([x + 0.18 * w, x + (0.55 if i == 0 else 0.78) * w], [yy, yy], color="#8A8A8A", lw=0.5, zorder=14.2)
    if mark:
        txt(x + 0.45 * w, y + 0.42 * h, mark, 6.5, "bold", color=mcolor, ha="center", z=14.3)


def funnel(cx, cy, s):
    ax.add_patch(Polygon([(cx - s, cy + 0.8 * s), (cx + s, cy + 0.8 * s), (cx + 0.22 * s, cy - 0.1 * s),
                          (cx + 0.22 * s, cy - 0.85 * s), (cx - 0.22 * s, cy - 0.85 * s), (cx - 0.22 * s, cy - 0.1 * s)],
                         closed=True, fc="#C9B8F4", ec=INK, lw=0.6, zorder=14))


def gear(cx, cy, r, fc="#F7B98C"):
    pts = []
    for i in range(32):
        a = 2 * math.pi * i / 32
        rr = r if (i // 2) % 2 == 0 else 0.74 * r
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=INK, lw=0.6, zorder=14))
    ax.add_patch(Circle((cx, cy), 0.3 * r, fc="white", ec=INK, lw=0.5, zorder=14.1))


# ============================================================================ row 1: contrast
card(2, 262, 277, 337)
txt(139.5, 327, "Hard problem", 7.5, "bold", ha="center")
S1, ST1 = 12, 14.5
for i, fc in enumerate([C_BLUE, C_BLUE, C_GRAYTOK, C_BLUE, C_BLUE, C_GRAYTOK, C_GRAYTOK, C_BLUE]):
    token(15 + i * ST1, 300, S1, fc)
arrow((126, 300), (142, 300), color=MUTE)
txt(134, 312, "self-play", 6.0, color=MUTE, ha="center")
for i in range(8):
    token(152 + i * ST1, 300, S1, C_BLUE)
check_badge(268, 300, 6.5)
txt(66, 277, "errors scatter", 6.3, color=MUTE, ha="center")
txt(206, 277, "leaves the band", 6.3, color=MUTE, ha="center")

card(353, 262, 628, 337)
txt(490.5, 327, "Ill-posed problem", 7.5, "bold", ha="center")
rd = {"A": C_A, "B": C_B, "C": C_C}
for i, c in enumerate("ABACBACB"):
    token(366 + i * ST1, 300, S1, rd[c], label=c)
arrow((477, 300), (493, 300), color=MUTE)
txt(485, 312, "self-play", 6.0, color=MUTE, ha="center")
for i, c in enumerate("AABAACAB"):
    token(503 + i * ST1, 300, S1, rd[c], label=c)
warn_badge(619, 300, 12)
txt(417, 277, "readings split", 6.3, color=MUTE, ha="center")
txt(557, 277, "stays in the band", 6.3, color=MUTE, ha="center")

txt(284.0, 305, "}", 24, color="#E8894E", ha="center", z=26)
txt(346.0, 305, "{", 24, color="#E8894E", ha="center", z=26)
txt(315, 305, "same\nsplit,\ndifferent\ncause", 6.3, "bold", ha="center", linespacing=1.0)
txt(315, 271, "readings:\n94% vs 26%", 6.0, color=MUTE, ha="center", linespacing=1.05)

# ============================================================================ panels
soft_panel(2, 2, 140, 256, P_GRAY, "(a) Label-free loop")
soft_panel(145, 2, 283, 256, P_PEACH, "(b) The drift")
soft_panel(288, 2, 473, 256, P_LAV, "(c) Why: solvable ones exit")
soft_panel(478, 2, 628, 256, P_PINK, "(d) What it learns")

# ---- (a)
nodes = [("Frozen proposer", "#DCE8FB", "#6F9BE0", "robot"), ("1,200 problems", "#FFF1C2", "#E3B341", "doc"),
         ("Solver: 8 samples", "#DDF2DE", "#5DB36B", "solver"), ("Keep if 2\u20136 agree", "#E7DFFB", "#9B7ED9", "funnel"),
         ("Majority-vote GRPO", "#FDE1CC", "#E8894E", "gear")]
NY = [218, 186, 154, 122, 90]
for (lab, fc, ec, icon), y in zip(nodes, NY):
    rbox(8 + SH["dx"], y - 12 + SH["dy"], 126 + SH["dx"], y + 12 + SH["dy"], fc="#000000", r=10, z=3.8, alpha=0.08)
    rbox(8, y - 12, 126, y + 12, fc=fc, ec=ec, lw=0.9, r=10, z=4)
    if icon == "robot":
        robot(19, y - 1, 6.0, "#FFFFFF")
        snowflake(28.5, y + 6.5, 3.2)
    elif icon == "solver":
        robot(19, y - 1, 6.0, "#BFE5C1")
    elif icon == "doc":
        doc_icon(14, y - 7.5, 11, 15)
    elif icon == "funnel":
        funnel(19.5, y, 7.0)
    else:
        gear(19.5, y, 7.0)
    txt(78, y, lab, 6.3, "bold", ha="center")
for y0, y1 in zip(NY[:-1], NY[1:]):
    arrow((67, y0 - 12.5), (67, y1 + 12.5), color="#555555", lw=0.8, ms=5)
curved_arrow((128, 90), (128, 154), bulge=(1, 0), depth=7.5, color="#E07A2C", lw=1.1, ms=6)

card(8, 14, 132, 64, ec="#E8A77A", lw=0.9)
header_strip(8, 132, 48, 64, "#FDE1CC", "#E8A77A")
txt(70, 56, "R-Zero\u2019s promise", 6.5, "bold", ha="center")
txt(15, 36, "\u201cimplicit quality", 6.3, style="italic")
txt(15, 25, "control\u201d", 6.3, style="italic")
txt(118, 31, "?", 13, "bold", color="#D84315", ha="center")

# ---- (b)
card(149, 160, 279, 232, ec="#E8894E", lw=0.9)
header_strip(149, 279, 214, 232, "#F9CBA7", "#E8894E")
txt(214, 223, "Qwen3-4B, ill-posed share", 6.5, "bold", ha="center")
for lab, n_ill, val, y in (("R1", 6, "57%", 199), ("R3", 8, "80%", 181)):
    txt(152, y, lab, 6.3, color=MUTE)
    for i in range(10):
        token(167 + i * 8.9, y, 7.6, C_ILL if i < n_ill else C_SOLV)
    txt(256.0, y, val, 6.5, "bold", color=RED_T)
txt(214, 168, "Opus judge, 200 per round", 6.0, color=MUTE, ha="center")
for lab, a, b, y in (("Qwen3-8B", 56, 73, 145), ("Phi-4-mini", 60, 82, 127), ("R-Zero code", 33, 58, 109)):
    txt(151, y, lab, 6.5, "bold")
    txt(278, y, f"{a}% \u2192 {b}%", 6.5, "bold", color=RED_T, ha="right")
    ax.plot([151, 278], [y - 9, y - 9], color="#E9C9B3", lw=0.5, zorder=5)
card(149, 20, 279, 90, ec="#E8A77A", lw=0.9)
txt(214, 79, "same drift with", 6.5, "bold", ha="center")
pill_row(156, 60, ["\u2713 blind", "\u2713 open judge"], "#E6F4E7", "#5DB36B")
pill_row(156, 38, ["\u2713 humans", "\u2713 no judge"], "#E6F4E7", "#5DB36B")

# ---- (c)
txt(380.5, 231, "one fixed pool\u2020, each new solver", 6.0, style="italic", color=MUTE, ha="center")
BX0, BX1 = 334, 434
ax.add_patch(Rectangle((BX0, 100), BX1 - BX0, 120, fc="#D6CCF6", ec="none", zorder=2))
for xx in (BX0, BX1):
    ax.plot([xx, xx], [100, 220], color="#7E57C2", lw=0.7, ls=(0, (3, 2)), zorder=3)
slot = lambda i: 339 + i * 10
# (label, share, reds kept, red ghost slots, greens kept (right-aligned), green ghosts, exited greens, exited reds)
rows = [("base", "53% ill", 206, 5, 0, 5, 0, 0, 0),
        ("round 1", None, 176, 5, 0, 3, 2, 2, 0),
        ("round 3", None, 146, 5, 0, 2, 3, 3, 0),
        ("round 5", "81% ill", 116, 4, 1, 1, 4, 4, 1)]
for lab, share, y, n_r, g_r, n_g, g_g, out_g, out_r in rows:
    txt(292, y + (4.5 if share else 0), lab, 6.3, "bold" if share else "normal")
    if share:
        txt(292, y - 5.5, share, 6.0, "bold", color=RED_T)
    for i in range(n_r):
        token(slot(i), y, 8.4, C_ILL)
    for i in range(g_r):
        token(slot(n_r + i), y, 8.4, None, ghost=True)
    for i in range(n_g):
        token(slot(9 - i), y, 8.4, C_SOLV)
    for i in range(g_g):
        token(slot(9 - n_g - i), y, 8.4, None, ghost=True)
    outs = [C_SOLV] * out_g + [C_ILL] * out_r
    for k, fc in enumerate(outs):
        cx = 443 + (k % 3) * 10.5
        cy = y + (5.0 if k < 3 else -5.0)
        token(cx, cy, 8.4, fc, label=None)
        if fc == C_SOLV:
            ax.plot([cx - 2.3, cx - 0.5, cx + 2.6], [cy, cy - 1.8, cy + 2.0], color="white", lw=0.8, zorder=10)
        else:
            txt(cx, cy - 0.2, "!", 6.0, "bold", color="white", ha="center", z=10)
curved_arrow((435.5, 211), (443, 185.8), bulge=(1, 1), depth=4.0, color="#7E57C2", lw=0.8, ms=5, ls=(0, (2.5, 1.5)))
txt(384, 91, "admission band", 6.3, "bold", color="#5E35B1", ha="center")
txt(453, 91, "exited", 6.0, color=MUTE, ha="center")
legend = [(C_SOLV, None, "solvable: rate 3.75\u2020, exits"),
          (C_ILL, None, "ill-posed: rate 1.40\u20131.60\u2020, stays"),
          (None, "ghost", "empty slot: its problem has exited"),
          (C_SOLV, "check", "exited: answered unanimously"),
          (C_ILL, "!", "ill-posed exit: self-consistent")]
for k, (fc, mark, s) in enumerate(legend):
    y = 72 - 13 * k
    token(297, y, 8.0, fc, ghost=(mark == "ghost"))
    if mark == "check":
        ax.plot([297 - 2.2, 297 - 0.5, 297 + 2.5], [y, y - 1.7, y + 1.9], color="white", lw=0.8, zorder=10)
    elif mark == "!":
        txt(297, y - 0.2, "!", 6.0, "bold", color="white", ha="center", z=10)
    txt(305, y, s, 6.3)

# ---- (d)
card(482, 136, 624, 234, ec="#5DB36B", lw=0.9)
header_strip(482, 624, 198, 234, "#DDF2DE", "#5DB36B")
doc_icon(487, 206, 12, 16, mark="?", mcolor=RED_T)
txt(504, 216, "\u201cIf he can fit some books\non a shelf, how many\nshelves will he need?\u201d", 6.0, style="italic",
    linespacing=1.08)
txt(514, 189, "8 answers, 7 values", 6.0, color=MUTE)
txt(486, 176, "base", 6.3, color=MUTE)
pal = ["#F4A9A0", "#BFE3C1", "#B9D3F6", "#D9C9F7", "#FFD9A3", "#B7E6E9", "#F6BFD6", "#BFE3C1"]
for i, fc in enumerate(pal):
    token(519 + i * 12.8, 176, 11, fc)
txt(486, 159, "R5", 6.3, color=MUTE)
for i in range(8):
    token(519 + i * 12.8, 159, 11, C_YEL if i < 6 else C_GRAYTOK, label="27" if i < 6 else None, lsize=6.0)
txt(514, 145, "6 of 8 say 27", 6.3, "bold")

block_arrow((518, 133), (518, 121), "#F4A79D")
block_arrow((590, 133), (590, 121), "#9ED39F")
txt(518, 112, "plain prompt", 6.3, "bold", color=RED_T, ha="center")
txt(518, 101, "self-consistent:", 6.0, ha="center")
txt(518, 90, "28% \u2192 41\u201348%", 6.8, "bold", color=RED_T, ha="center")
txt(518, 79, "accuracy flat", 6.0, color=MUTE, ha="center")
txt(590, 112, "with \u201cunknown\u201d", 6.3, "bold", color=GRN_T, ha="center")
txt(590, 101, "abstains:", 6.0, ha="center")
txt(590, 90, "68\u201376%", 6.8, "bold", color=GRN_T, ha="center")
txt(590, 79, "2\u20133% false", 6.0, color=MUTE, ha="center")

ang = -4
cx, cy, w_, h_ = 553.0, 40.0, 134.0, 44.0
rot = math.radians(ang)
corners = [(-w_ / 2, -h_ / 2), (w_ / 2, -h_ / 2), (w_ / 2, h_ / 2), (-w_ / 2, h_ / 2)]
pts = [(cx + a * math.cos(rot) - b * math.sin(rot), cy + a * math.sin(rot) + b * math.cos(rot)) for a, b in corners]
ax.add_patch(Polygon([(x + 1.1, y - 1.4) for x, y in pts], closed=True, fc="#000000", ec="none", alpha=0.12, zorder=13))
ax.add_patch(Polygon(pts, closed=True, fc="#F9C0B6", ec="#E08A7D", lw=0.7, zorder=14))
txt(cx, cy, "\u201c\u2026I will assume that each\nshelf can hold a maximum\nof 1 book.\u201d", 6.0, style="italic",
    ha="center", rotation=ang, linespacing=1.08, z=15)

# ---- connectors between panels
block_arrow((131, 190), (148, 190), "#F7B98C")
block_arrow((280, 136), (297, 136), "#C4B4F2")
block_arrow((470, 191), (486, 191), "#F4A79D")

# ============================================================================ QA + save
problems = figqa.check(fig, ax, scale=1.0, min_pt=6.0, big_in=0.9)
for n in figqa.arrow_report(ax):
    print("arrow:", n)
print(f"{len(problems)} layout problems")
for p_ in problems:
    print("  ", p_)
fonts = sorted({round(t.get_fontsize(), 2) for t in ax.texts})
print("font sizes used (pt):", fonts)
for ext in ("pdf", "png", "svg"):
    fig.savefig(os.path.join(HERE, f"overview_paper.{ext}"), dpi=300 if ext == "png" else None)
print("saved")
