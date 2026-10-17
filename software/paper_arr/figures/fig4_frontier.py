"""Appendix figure: selective answering on UMWP under the corrected abstention prompt (Qwen3-4B, round 3).

Frontier = strict Pareto points of the label-free rules in dirA5/code/a5_frontier.py (modal-share threshold, abstention
count), linearly interpolated, as stored per checkpoint in the registered A11 read output (descriptive.umwp_per_tag.*.new).
r2 (2026-10-06): one line per training condition instead of one per seed: the mean of the seeds' frontiers on a common
grid of false-abstention rates, with the range over seeds shaded; hollow markers: the mean default point (plain-prompt
self-consistent-answer rate at the abstention-prompt false-abstention rate), bars: range over seeds.
Data are read read-only from ext/a11_runs/a11_read.json. 3.03 in wide (ACL column), text >= 7.5 pt.
r3 (2026-10-06, review): --source rev6_strict (default) reads the same quantities from paper_arr/rev6/rev6_read.json
(frontier.STRICT.<tag>: pareto, default_point), i.e. the rev6 samples with the answers re-parsed by the strict parser;
--source rev6_original the same samples with the probe's parser; --source a11 the figure as before.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "/home/tione/notebook/rsi_rl"
READ = f"{ROOT}/ext/a11_runs/a11_read.json"
OUT = f"{ROOT}/paper_arr/figures/fig4_frontier.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.linewidth": 0.35, "grid.alpha": 0.5,
})

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="rev6_strict", choices=["rev6_strict", "rev6_original", "a11"])
ap.add_argument("--out", default=OUT); A = ap.parse_args(); OUT = A.out
if A.source == "a11":
    u = json.load(open(READ))["descriptive"]["umwp_per_tag"]
else:
    F = json.load(open(f"{ROOT}/paper_arr/rev6/rev6_read.json"))["frontier"]["STRICT" if A.source == "rev6_strict" else "ORIGINAL"]
    u = {t: {"new": {"pareto_abs": d["pareto"], "default_point": {"FA_A": d["default_point"][0], "CA_U": d["default_point"][1]}}}
         for t, d in F.items() if "default_point" in d}
GROUPS = [
    ("Base model", ["qwen3_4b_base"], "#555555", "-", "o"),
    ("Full curriculum (5 seeds)", [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)], "#2b5c9e", "-", "o"),
    ("Solvable only (4 seeds)", [f"qwen3_4b_Base_own_clean_a3_s{s}_r3" for s in range(4)], "#e08a2e", "--", "s"),
    ("Abstention guard (3 seeds)", [f"qwen3_4b_Base_own_ag_a4_s{s}_r3" for s in range(3)], "#4d9a6a", (0, (1.2, 1.2)), "^"),
]
XMAX = 15.0
GRID = np.linspace(0.0, XMAX, 301)

def frontier(tag):
    pts = sorted(u[tag]["new"]["pareto_abs"])
    xs = [0.0] + [100 * p[0] for p in pts]   # flat at the first Pareto point below the smallest false-abstention rate
    ys = [100 * pts[0][1]] + [100 * p[1] for p in pts]
    return np.interp(GRID, xs, ys)

fig, ax = plt.subplots(figsize=(3.03, 2.3))
handles = []
for name, tags, col, ls, mk in GROUPS:
    F = np.array([frontier(t) for t in tags])
    if len(tags) > 1:
        ax.fill_between(GRID, F.min(0), F.max(0), color=col, alpha=0.16, lw=0)
    ax.plot(GRID, F.mean(0), color=col, ls=ls, lw=1.3)
    d = np.array([[100 * u[t]["new"]["default_point"]["FA_A"], 100 * u[t]["new"]["default_point"]["CA_U"]] for t in tags])
    mx, my = d.mean(0)
    if len(tags) > 1:
        ax.errorbar(mx, my, xerr=[[mx - d[:, 0].min()], [d[:, 0].max() - mx]], yerr=[[my - d[:, 1].min()], [d[:, 1].max() - my]],
                    fmt="none", ecolor=col, elinewidth=0.8, capsize=1.5, capthick=0.8, zorder=4)
    ax.plot(mx, my, marker=mk, ms=4.5, mfc="white", mec=col, mew=1.0, ls="none", zorder=5)
    handles.append(Line2D([0], [0], color=col, ls=ls, lw=1.3, label=name))
handles.append(Patch(facecolor="#888888", alpha=0.3, lw=0, label="Range over seeds"))
handles.append(Line2D([0], [0], marker="o", ms=4.5, mfc="white", mec="#333333", mew=1.0, ls="none", label="Without the option"))
ax.axvline(5.0, color="#999999", lw=0.5, ls=(0, (2, 2)))
ax.set_xlim(0, XMAX)
ax.set_ylim(0, 80)
ax.set_xlabel("False abstention on answerable questions (%)")
ax.set_ylabel("Committed answers to\nunanswerable questions (%)")
ax.grid(True)
ax.legend(handles=handles, loc="upper right", frameon=False, handlelength=2.2, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(OUT)
print("wrote", OUT)
