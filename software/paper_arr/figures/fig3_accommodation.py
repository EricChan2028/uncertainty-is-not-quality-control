"""Figure 4 (file name kept): what the solver says to unanswerable questions, before and after self-play.

Qwen3-4B-Base vs its own self-play solver after round 5 (seeds 0-2), on 112 UMWP unanswerable items
and 112 bin-matched hard answerable items, 8 fresh samples per item, labelled blind.
r2 (2026-10-06, review): four categories instead of three, with the answer status from the strict parser
(paper_arr/analyses/rev5/strict_parse.py, validated against the human annotators) instead of the probe's parser, which
read a number from many verbal declines. Stacked horizontal bars: share of SAMPLES that
  silent_answer      - give a numeric answer, no gap/assumption flagged,
  conditional_answer - give a numeric answer and flag a missing/ambiguous premise or state an assumption,
  flag_and_decline   - give no numeric answer and flag a gap,
  other_non_answer   - give no numeric answer and flag nothing (cut off, symbolic, unrelated),
(paper_arr/rev6/fig4_categories.json, "llm_flags_strict_parser", cells '<model>|U' / '<model>|A').
Right column: mean number of distinct reading clusters per item
(dirA5/results/a8/a8_read.json, cells['<model>|<U/A>']['mean_reading_clusters']).
Data are read read-only.  python3 fig3_accommodation.py [--preview /tmp/x.png]
Layout: 3.03 in wide (ACL column width), included at 100% scale; all text >= 7.5 pt.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ROOT = "/home/tione/notebook/rsi_rl"
CAT4 = f"{ROOT}/paper_arr/rev6/fig4_categories.json"
READ = f"{ROOT}/dirA5/results/a8/a8_read.json"
OUT = f"{ROOT}/paper_arr/figures/fig3_accommodation.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "grid.linewidth": 0.35,
    "grid.alpha": 0.5,
    "hatch.linewidth": 0.5,
})

cat4 = json.load(open(CAT4))["llm_flags_strict_parser"]
cells = json.load(open(READ))["cells"]

MODELS = [("base", "Base"), ("r5_s0", "Round 5, s0"), ("r5_s1", "Round 5, s1"), ("r5_s2", "Round 5, s2")]
# (key, legend label, colour, hatch, text colour); colours chosen for CVD separation
CATS = [("silent_answer", "Silent answer", "#3366aa", "", "white"),
        ("conditional_answer", "Answer, gap flagged", "#e8a33d", "////", "#1a1a1a"),
        ("flag_and_decline", "Decline, gap flagged", "#5aa36e", "", "#1a1a1a"),
        ("other_non_answer", "Other non-answer", "#cfcfcf", "", "#1a1a1a")]
INK = "#1a1a1a"
MUTED = "#555555"

for m, _ in MODELS:
    for g in ("U", "A"):
        s = cat4[f"{m}|{g}"]
        assert s["n"] == 896, (m, g, s)
        assert abs(sum(s.get(c[0], 0.0) for c in CATS) - 1.0) < 1e-9, (m, g, s)
        assert cells[f"{m}|{g}"]["n"] == 112 and cells[f"{m}|{g}"]["valid"] == 112

def pct_label(v):
    return str(int(round(100 * v + 1e-9)))

Y_HDR_U, Y_U, H_U = -0.05, [0.95, 1.95, 2.95, 3.95], 0.74
Y_HDR_A, Y_A, H_A = 4.9, [5.6, 6.3, 7.0, 7.7], 0.46
groups = [("U", "Unanswerable items (n = 112)", Y_HDR_U, Y_U, H_U, True),
          ("A", "Hard answerable items (n = 112)", Y_HDR_A, Y_A, H_A, False)]

FIG_W, FIG_H = 3.03, 2.5
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_axisbelow(True)
ax.grid(axis="x", color="#bbbbbb")
rows = []
for g, hdr, yh, ys, h, annotate in groups:
    ax.text(0.0, yh, hdr, ha="left", va="center", fontsize=8, fontweight="bold", color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    for (m, lab), y in zip(MODELS, ys):
        s = cat4[f"{m}|{g}"]
        left = 0.0
        for key, _, col, hatch, tcol in CATS:
            w = 100.0 * s.get(key, 0.0)
            ax.barh(y, w, left=left, height=h, color=col, edgecolor="white", linewidth=0.6,
                    hatch=hatch if hatch else None)
            if annotate and w >= 8 and key != "other_non_answer":
                ax.text(left + w / 2, y, pct_label(s[key]), ha="center", va="center", fontsize=7.5,
                        color=tcol, bbox=(dict(facecolor=col, edgecolor="none", pad=0.4) if hatch else None))
            left += w
        rows.append((y, lab))

tr = blended_transform_factory(ax.transAxes, ax.transData)
XCOL = 1.10
ax.text(XCOL, Y_HDR_U, "Readings", ha="center", va="center", fontsize=8, fontweight="bold",
        color=INK, transform=tr, clip_on=False)
for g, hdr, yh, ys, h, annotate in groups:
    for (m, _), y in zip(MODELS, ys):
        r = cells[f"{m}|{g}"]["mean_reading_clusters"]
        ax.text(XCOL, y, f"{r:.2f}", ha="center", va="center", fontsize=8,
                color=INK if g == "U" else MUTED, transform=tr, clip_on=False)

ax.set_yticks([y for y, _ in rows])
ax.set_yticklabels([lab for _, lab in rows])
for t, (y, _) in zip(ax.get_yticklabels(), rows):
    if y in Y_A:
        t.set_color(MUTED)
ax.set_ylim(Y_A[-1] + 0.45, Y_HDR_U - 0.42)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("Share of samples (%)")
ax.spines["left"].set_visible(False)

handles = [Patch(facecolor=c[2], edgecolor="white", hatch=c[3] if c[3] else None, label=c[1]) for c in CATS]
handles = [handles[0], handles[2], handles[1], handles[3]]  # ncol=2 fills column-major: rows read silent, flagged / decline, other
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
           handlelength=1.1, handleheight=0.8, handletextpad=0.35, columnspacing=1.2, borderaxespad=0.15,
           labelspacing=0.3)

fig.tight_layout(pad=0.25, rect=(0, 0, 1, 0.875))
fig.savefig(OUT)
if "--preview" in sys.argv:
    fig.savefig(sys.argv[sys.argv.index("--preview") + 1], dpi=300)
print("wrote", OUT)
for g in ("U", "A"):
    for m, _ in MODELS:
        s = cat4[f"{m}|{g}"]
        print(f"{m}|{g}", {k: round(s.get(k, 0.0), 3) for k, *_ in CATS}, "readings", round(cells[f"{m}|{g}"]["mean_reading_clusters"], 4))
