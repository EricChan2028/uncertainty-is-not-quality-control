"""Figure 2: class-conditional sharpening of the admission band (Qwen3-4B own loop).

(a) fit pool (qwen_s0): fraction of each audited class inside the admission band,
    observed (markers) vs fitted per-class law (lines).
(b) held-out pool (qwen_s1): admitted ill-posed share, observed vs prediction of the
    per-class law fitted on the other pool (with bootstrap 95% band) vs single shared rate.
Data are read read-only from dirA4/results.
Layout (r1, 2026-09-26): 3.03 in wide (ACL column width), included at 100% scale; all text >= 8 pt.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

ROOT = "/home/tione/notebook/rsi_rl"
LAW = f"{ROOT}/dirA4/results/law_stable_2026-09-24.json"
BOOT = f"{ROOT}/dirA4/results/law_bootstrap.json"
OUT = f"{ROOT}/paper_arr/figures/fig2_mechanism.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.linewidth": 0.35,
    "grid.alpha": 0.5,
})

law = json.load(open(LAW))
boot = json.load(open(BOOT))

CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
x = list(range(len(CKS)))
CLASSES = [("SOLVABLE", "Solvable", "#4c72b0", "o"),
           ("AMBIGUOUS", "Ambiguous", "#dd8452", "s"),
           ("UNSOLVABLE", "Unsolvable", "#8c1c1f", "^")]
QWEN4B = "#1f77b4"
PRED = "#222222"

s0 = law["qwen_s0"]
s1 = law["qwen_s1"]
obs_test = boot["point"]["obs_test"]
pred_test = boot["point"]["pred_test"]
ci = boot["ci95"].get("pred_test_by_ck")
single = s1["single_rate_law"]
# consistency checks between the two files
assert obs_test == s1["observed"], (obs_test, s1["observed"])
assert pred_test == s1["per_class_law"], (pred_test, s1["per_class_law"])
mae = sum(abs(a - b) for a, b in zip(obs_test, pred_test)) / len(obs_test)
mae_single = sum(abs(a - b) for a, b in zip(obs_test, single)) / len(obs_test)

FIG_W, FIG_H = 3.03, 2.7  # ACL \columnwidth; the PDF is included at 100% scale
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)

# ---------- panel (a) ----------
handles, labels = [], []
for key, name, col, mk in CLASSES:
    ob = s0["in_band_obs"][key]
    fit = s0["in_band_law"][key]
    ln, = ax1.plot(x, fit, color=col, lw=1.1, zorder=2)
    sc, = ax1.plot(x, ob, ls="none", marker=mk, ms=4.0, mfc=col, mec="white",
                   mew=0.4, color=col, zorder=3)
    handles.append((sc, ln))
    labels.append(name)
leg1 = ax1.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
                  loc="upper right", ncol=3, frameon=False, handlelength=1.4,
                  columnspacing=0.7, handletextpad=0.35, borderaxespad=0.1, borderpad=0.2)
ax1.set_ylim(0, 1.12)
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.set_yticklabels(["0", ".25", ".50", ".75", "1"])
ax1.set_ylabel("Fraction in band")
ax1.grid(True, axis="y")
ax1.text(0.012, 0.99, "(a)", transform=ax1.transAxes, fontsize=9, fontweight="bold",
         va="top", ha="left")
ax1.text(0.995, 0.285, "Markers: observed; lines: fitted model", transform=ax1.get_yaxis_transform(),
         fontsize=8, va="center", ha="right", color="#444444")

# ---------- panel (b) ----------
if ci is not None:
    lo = [c[0] for c in ci]
    hi = [c[1] for c in ci]
    ax2.fill_between(x, lo, hi, color=PRED, alpha=0.13, lw=0, zorder=1)
l_law, = ax2.plot(x, pred_test, color=PRED, lw=1.1, zorder=2)
l_single, = ax2.plot(x, single, color=PRED, lw=1.1, ls=(0, (4, 2)), zorder=2)
l_obs, = ax2.plot(x, obs_test, ls="none", marker="o", ms=3.8, mfc=QWEN4B, mec="white",
                  mew=0.4, zorder=3)
if ci is not None:
    band = plt.Rectangle((0, 0), 1, 1, fc=PRED, alpha=0.13, lw=0)
    law_handle = (band, l_law)
else:
    law_handle = l_law
ax2.legend([l_obs, law_handle, l_single],
           ["Observed (held-out pool)", "Class-conditional model (fit on other pool)", "Single rate"],
           handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
           loc="lower right", frameon=False, handlelength=1.8, handletextpad=0.4,
           borderaxespad=0.1, borderpad=0.2, labelspacing=0.2)
ax2.set_ylim(35, 90)
ax2.set_yticks([40, 60, 80])
ax2.set_ylabel("Admitted ill-posed (%)")
ax2.grid(True, axis="y")
ax2.set_xticks(x)
ax2.set_xticklabels(CKS)
ax2.set_xlim(-0.3, len(CKS) - 0.7)
ax2.set_xlabel("Solver checkpoint")
ax2.text(0.012, 0.99, "(b)", transform=ax2.transAxes, fontsize=9, fontweight="bold",
         va="top", ha="left")

fig.tight_layout(pad=0.3, h_pad=0.4)
fig.align_ylabels([ax1, ax2])
fig.savefig(OUT)
if len(sys.argv) > 1:
    fig.savefig(sys.argv[1], dpi=300)

print("PANEL A (qwen_s0, file", LAW, ")")
for key, name, _, _ in CLASSES:
    print(" ", name.lower(), "obs", s0["in_band_obs"][key], "law", s0["in_band_law"][key])
print("beta_per_class", law["beta_per_class"])
print("PANEL B (qwen_s1)")
print("  obs_test", obs_test)
print("  pred_test", pred_test, "mae %.2f (file %.2f)" % (mae, boot["point"]["mae_test"]))
print("  ci95 pred_test_by_ck", ci)
print("  single_rate_law", single, "mae %.2f (file %.2f)" % (mae_single, s1["mae_single"]))
print("wrote", OUT)
