"""Appendix figures that replace eight numeric appendix tables (2026-10-05). Values are those the tables reported, which come
from the registered read outputs named per figure below (all in data.zip, reads/). Single-column figures: the width of each
saved figure is fitted to 3.03 in (ACL column width), so that it is included at about 100% scale; text >= 7 pt.
  figA_validation.pdf  agreement checks of the judges          (Appendix B; judge validation, a7b and a10 reads)
  figA_human.pdf       human annotation of 240 problems        (Appendix B; dirA5/human_validation_drift/human_read.json)
  figA_prereg.pdf      outcomes of the 136 predictions         (Appendix C; dirA5/docs/prereg_ledger_2026-09-25.md, A11-A13 reads)
  figA_rzacc.pdf       R-Zero benchmark accuracy               (Appendix E; ext/a12_draft a12_p2_read, a12_plain_eval, a12_rzchat_eval)
  figA_rzero.pdf       R-Zero final solvers on UMWP            (Appendix E; ext/rzero_design rz_read, a12 reads)
  figA_perseed.pdf     ill-posed share per lineage             (Appendix F; a3/a4 reads, rz_read)
  figA_sizes.pdf       training-set sizes                      (Appendix F; training logs)
  figA_thresholds.pdf  self-consistency at stricter thresholds (Appendix G; paper_arr/analyses/rev3/ca_thresholds_2026-10-04.json)
python3 figA_appendix.py [--outdir DIR]"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default="/home/tione/notebook/rsi_rl/paper_arr/figures")
# r7 (2026-10-06, review): thresholds figure from a ca_thresholds.py output (default: the strict-parser rerun on the rev6 samples)
ap.add_argument("--thr_json", default="/home/tione/notebook/rsi_rl/paper_arr/rev6/reruns_strict/ca_thresholds.json")
ap.add_argument("--only", default=""); args = ap.parse_args()
W = 3.03
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.linewidth": 0.35, "grid.alpha": 0.5, "hatch.linewidth": 0.5, "legend.frameon": False,
})
INK, MUTED, GRID = "#1a1a1a", "#555555", "#d9d9d9"
BLUE, LBLUE, ORANGE, GREEN, GREY, PURPLE = "#2b5c9e", "#8fb3e0", "#e08a2e", "#4d9a6a", "#9e9e9e", "#7a4f9e"

def save(fig, name):
    for _ in range(6):  # fit the tight bounding box to the column width (label sizes are fixed in inches)
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if abs(w - W) < 0.005: break
        fw, fh = fig.get_size_inches(); fig.set_size_inches(fw + (W - w), fh)
    fig.savefig(f"{args.outdir}/{name}", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)

# ---------------------------------------------------------------- validation
def fig_validation():
    rows = [("Opus vs. UMWP\nhuman labels (400)", (97.0, 97.0), "97.0%, $\\kappa$ 0.94"),
            ("Opus vs. planted\nstatus (200)", (90.0, 90.0), "90.0%"),
            ("Opus re-judged, re-\nversed order (2$\\times$200)", (90.0, 92.0), "90–92%, $\\kappa$ 0.74–0.76"),
            ("Blind vs. original,\nQwen3-4B (900)", (90.8, 90.8), "90.8%, $\\kappa$ 0.78"),
            ("Blind vs. original,\nR-Zero (1,200)", (97.2, 97.2), "97.2%, $\\kappa$ 0.94"),
            ("Open-weight vs.\nblind Opus (900)", (70.6, 70.6), "70.6%, $\\kappa$ 0.38")]
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ys = list(range(len(rows)))[::-1]
    for y, (lab, (lo, hi), txt) in zip(ys, rows):
        if hi > lo: ax.plot([lo, hi], [y, y], color=BLUE, lw=3.0, solid_capstyle="butt")
        else: ax.plot([lo], [y], "o", color=BLUE, ms=4.2)
        if lo > 85: ax.text(lo - 1.6, y, txt, ha="right", va="center", fontsize=7, color=INK)
        else: ax.text(hi + 1.6, y, txt, ha="left", va="center", fontsize=7, color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(50, 100); ax.set_xticks([50, 60, 70, 80, 90, 100]); ax.set_xlabel("Agreement on ill-posed vs. solvable (%)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.set_ylim(-0.6, len(rows) - 0.4)
    save(fig, "figA_validation.pdf")

# ---------------------------------------------------------------- human
def fig_human():
    R = [1, 3, 5]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"wspace": 0.5})
    fig.subplots_adjust(bottom=0.36, top=0.9)
    hum = [56.2, 78.8, 76.2]
    a1.plot(R, hum, "-o", color=INK, ms=3.5, lw=1.2)
    for r, v, dx, ha in zip(R, hum, (0.35, 0, 0), ("left", "center", "center")):
        a1.text(r + dx, v - (1.8 if dx else -2.6), f"{v:.1f}", ha=ha, va="center" if dx else "bottom", fontsize=7)
    a1.text(3.0, 47.5, "R1$\\to$R5: +20.0\n[6.2, 33.8]", ha="center", va="center", fontsize=7, color=MUTED)
    a1.set_ylabel("Human ill-posed share (%)"); a1.set_title("(a)", loc="left", pad=3)
    a1.set_ylim(40, 92); a1.set_xticks(R); a1.set_xticklabels(["R1", "R3", "R5"]); a1.set_xlim(0.2, 5.8)
    judges = [("Opus, original", [11.4, 41.2, 36.8], ORANGE, "o", "$\\kappa$ 0.75, acc. 90%, drift +27.5"),
              ("Opus, blind", [2.9, 23.5, 21.1], BLUE, "s", "$\\kappa$ 0.89, acc. 95%, drift +26.2"),
              ("Qwen2.5-32B", [20.0, 23.5, 63.2], GREEN, "^", "$\\kappa$ 0.35, acc. 70%, drift +25.0")]
    for name, v, c, m, note in judges:
        a2.plot(R, v, "-", marker=m, color=c, ms=3.5, lw=1.1, label=f"{name}: {note}")
    a2.set_ylabel("Judge false positives (%)"); a2.set_title("(b)", loc="left", pad=3)
    a2.set_xticks(R); a2.set_xticklabels(["R1\n$n$=35", "R3\n$n$=17", "R5\n$n$=19"]); a2.set_xlim(0.2, 5.8); a2.set_ylim(0, 72)
    for ax in (a1, a2): ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    fig.legend(*a2.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1, handlelength=1.6, borderaxespad=0.1)
    save(fig, "figA_human.pdf")

# ---------------------------------------------------------------- prereg
def fig_prereg():
    rows = [("Drift across models (§3)", 5, 8, 0, 1), ("Filters, proposers,\nrewards (§3, §6)", 26, 29, 1, 2),
            ("Validity, controls\n(§3, §5, §6)", 13, 11, 2, 11), ("Difficulty, swaps,\nsets (§4, §5)", 5, 5, 0, 0),
            ("Exit through round 5 (§4)", 1, 1, 0, 0), ("Official R-Zero code\n(§3, §5)$^{\\ast}$", 2, 0, 0, 0),
            ("Blind re-judging,\ntwo studies (§3)", 6, 0, 0, 0), ("Readings (§4, §5)", 2, 1, 0, 0),
            ("Sharpening model in\ntwo more models (§4)", 2, 0, 0, 0), ("Open-weight judge (§3)", 1, 1, 0, 0)]
    tot = [sum(r[i] for r in rows) for i in range(1, 5)]
    assert tot == [63, 56, 3, 14] and sum(tot) == 136
    cats = [("met", BLUE, None, "white"), ("not met", ORANGE, None, INK), ("inconclusive", GREY, None, INK), ("not run", "white", "////", INK)]
    fig, ax = plt.subplots(figsize=(W, 3.75))
    fig.subplots_adjust(bottom=0.22, top=0.99)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        n, left = sum(r[1:]), 0.0
        for (name, col, hat, tcol), v in zip(cats, r[1:]):
            if v == 0: continue
            w = 100.0 * v / n
            ax.barh(y, w, left=left, color=col, edgecolor=MUTED if col == "white" else col, hatch=hat, lw=0.5, height=0.66)
            if w >= 5.2:  # wide enough for the count
                ax.text(left + w / 2, y, str(v), ha="center", va="center", fontsize=7, color=tcol,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none") if hat else None)
            left += w
    ax.set_yticks(ys); ax.set_yticklabels([f"{r[0]}, {sum(r[1:])}" for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of the topic's predictions (%)"); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    handles = [Patch(facecolor=c, edgecolor=MUTED if c == "white" else c, hatch=h, lw=0.5, label=f"{nm.capitalize()} ({t})") for (nm, c, h, _), t in zip(cats, tot)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, handlelength=1.2, columnspacing=1.2, handletextpad=0.4)
    save(fig, "figA_prereg.pdf")

# ---------------------------------------------------------------- R-Zero accuracy
def fig_rzacc():
    groups = [("MATH-500", [71.4, 74.0, 75.4], (68.2, 79.6)), ("GSM8K", [83.6, 92.0, 91.6], (87.8, 92.1)),
              ("MATH-500", [63.6, 71.8, 73.8], None), ("GSM8K", [82.2, 83.6, 84.6], None)]
    cols = [(GREY, "Base"), (BLUE, "Seed 0"), (LBLUE, "Seed 1")]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    fig.subplots_adjust(bottom=0.24, top=0.82)
    bw = 0.22
    for g, (name, vals, pub) in enumerate(groups):
        for i, ((c, lab), v) in enumerate(zip(cols, vals)):
            x = g + (i - 1) * bw
            ax.bar(x, v, width=bw * 0.92, color=c, label=lab if g == 0 else None)
            ax.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
        if pub:
            xp = g + 1.95 * bw
            ax.annotate("", xy=(xp, pub[1]), xytext=(xp, pub[0]), arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=6))
            ax.plot([xp], [pub[0]], "o", mfc="white", mec=INK, ms=3, mew=0.8)
    ax.plot([], [], marker="o", mfc="white", mec=INK, ms=3, ls="-", color=INK, lw=0.8, label="Published")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.tick_params(axis="x", length=0)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for (a, b, txt) in ((0, 1, "R-Zero chat format"), (2, 3, "our plain prompt")):
        ax.plot([a - 0.32, b + 0.32], [-0.13, -0.13], color=MUTED, lw=0.6, transform=tr, clip_on=False)
        ax.text((a + b) / 2, -0.15, txt, ha="center", va="top", fontsize=7.5, color=INK, transform=tr)
    ax.set_ylim(55, 100); ax.set_ylabel("Greedy accuracy (%)"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(groups) - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, handlelength=1.2, columnspacing=0.9, borderaxespad=0)
    save(fig, "figA_rzacc.pdf")

# ---------------------------------------------------------------- R-Zero UMWP
def fig_rzero():  # r7 (2026-10-06): rev6 runs, strict reading (rev6/reruns_strict/rz_paired_desc.py.log); abstention row from A11
    rows = [("Plain prompt:\nself-consistent", [28.2, 33.0, 29.0]), ("Abstention\nprompt: abstains", [53.4, 61.3, 60.2]),
            ("Chat format:\nself-consistent$^{\\dagger}$", [15.8, 44.5, 39.7]), ("Chat format:\nno answer", [41.3, 29.0, 32.8]),
            ("Answerable\nquestions: accuracy", [91.5, 90.2, 89.8])]
    cols = [(GREY, "Base"), (BLUE, "Seed 0"), (LBLUE, "Seed 1")]
    fig, ax = plt.subplots(figsize=(W, 2.75))
    fig.subplots_adjust(top=0.9)
    bh = 0.26
    ys = list(range(len(rows)))[::-1]
    for y, (name, vals) in zip(ys, rows):
        for i, ((c, lab), v) in enumerate(zip(cols, vals)):
            yy = y + (1 - i) * bh
            ax.barh(yy, v, height=bh * 0.9, color=c, label=lab if y == ys[0] else None)
            ax.text(v + 1.2, yy, f"{v:.1f}", ha="left", va="center", fontsize=7)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Share of UMWP questions (%)")
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, borderaxespad=0.2)
    save(fig, "figA_rzero.pdf")

# ---------------------------------------------------------------- per seed
def fig_perseed():
    D = {"Qwen3-4B": {"s0": [55.0, 72.5, 76.0, 80.5, 81.0, 83.0], "s1": [56.0, 75.5, 83.0, 80.0, 74.0, 82.5], "s2": [59.0, 67.5, 78.0, 78.5, 80.5, 76.5],
                      "s3": [56.0, 72.0, 78.0], "s4": [57.5, 76.0, 84.5]},
         "Qwen3-8B": {"s0": [57.5, 68.0, 76.5], "s1": [58.0, 69.5, 69.5, 78.5, 77.0, 77.5], "s2": [52.5, 66.5, 74.0, 75.5, 73.0, 75.0]},
         "Phi-4-mini": {"s0": [59.5, 76.5, 83.5], "s1": [61.0, 76.0, 80.5]},
         "Llama-3.2-3B": {"s2": [65.0, 74.5, 71.5], "s3": [67.0, 73.5, 69.0], "s4": [67.0, 76.5, 72.0]},
         "Qwen3-14B": {"s0": [57.0, 72.0]},
         "R-Zero, official code": {"s0": [32.5, 44.0, 55.5], "s1": [34.0, 42.5, 61.0]}}
    SC = {"s0": (BLUE, "o"), "s1": (ORANGE, "s"), "s2": (GREEN, "^"), "s3": (PURPLE, "D"), "s4": (GREY, "v")}
    fig, axs = plt.subplots(3, 2, figsize=(W, 3.6), sharey=True, gridspec_kw={"hspace": 0.62, "wspace": 0.12})
    fig.subplots_adjust(bottom=0.16)
    for ax, (name, seeds) in zip(axs.flat, D.items()):
        for s, v in seeds.items():
            c, m = SC[s]; ax.plot(range(1, len(v) + 1), v, "-", marker=m, ms=2.8, lw=1.0, color=c)
        ax.set_title(name, loc="left", pad=2)
        n = 3 if name.startswith("R-Zero") else 6
        ax.set_xticks(range(1, n + 1)); ax.set_xlim(0.6, n + 0.4); ax.set_ylim(25, 92); ax.set_yticks([30, 50, 70, 90])
        ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    for ax in axs[:, 0]: ax.set_ylabel("Ill-posed (%)")
    axs[2, 0].set_xlabel("Round"); axs[2, 1].set_xlabel("Iteration")
    handles = [Line2D([], [], color=c, marker=m, ms=2.8, lw=1.0, label=f"seed {s[1]}") for s, (c, m) in SC.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, handlelength=1.4, columnspacing=0.7, handletextpad=0.3)
    save(fig, "figA_perseed.pdf")

# ---------------------------------------------------------------- sizes
def fig_sizes():
    C = [("Full curriculum", INK, "o", {1: (944, 1006), 2: (668, 755), 3: (586, 593), 4: (553, 593), 5: (520, 533), 6: (444, 506)}),
         ("Absolute-Zero-style band", PURPLE, "D", {1: (1370, 1392), 2: (992, 1013), 3: (812, 880)}),
         ("Co-evolving proposer", GREEN, "^", {1: (944, 1006), 2: (584, 690), 3: (515, 563)}),
         ("Guard (= random loop)", ORANGE, "s", {1: (687, 799), 2: (404, 466), 3: (367, 412), 4: (386, 408), 5: (325, 380)}),
         ("Solvable only", BLUE, "v", {1: (430, 491), 2: (183, 216), 3: (124, 168)}),
         ("Same-size contrast", GREY, "P", {3: (148, 177)})]
    fig, ax = plt.subplots(figsize=(W, 2.35))
    off = {0: -0.15, 1: 0.15, 2: -0.05, 3: 0.05, 4: -0.1, 5: 0.1}
    for k, (name, c, m, d) in enumerate(C):
        xs = [r + off[k] for r in d]; mids = [(lo + hi) / 2 for lo, hi in d.values()]
        ax.vlines(xs, [lo for lo, _ in d.values()], [hi for _, hi in d.values()], color=c, lw=2.0)
        ax.plot(xs, mids, "-" if len(d) > 1 else "none", marker=m, color=c, ms=3.2, lw=0.9, label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("Training-set size (problems)"); ax.set_xticks(range(1, 7)); ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 2250); ax.set_yticks([0, 400, 800, 1200, 1600, 2000]); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    def n2p(n):  # 1920 / n is its own inverse
        n = np.asarray(n, dtype=float)
        return 1920.0 / np.clip(n, 1e-9, None)
    sec = ax.secondary_yaxis("right", functions=(n2p, n2p))
    sec.set_yticks([1.5, 2, 3, 5, 10]); sec.set_yticklabels(["1.5", "2", "3", "5", "10"]); sec.set_ylabel("Passes per problem")
    sec.spines["right"].set_visible(True); sec.spines["right"].set_linewidth(0.6)
    ax.legend(loc="upper right", ncol=1, handlelength=1.5, borderaxespad=0.2)
    save(fig, "figA_sizes.pdf")

# ---------------------------------------------------------------- thresholds
def fig_thresholds():
    T = ["$\\geq$4\n(main text)", "$\\geq$4,\nno tie", "$\\geq$5", "$\\geq$6", "$\\geq$7", "8 of 8"]
    import json
    J = json.load(open(args.thr_json)); K = ["paper_CA", "no_tie_at_top", ">=5", ">=6", ">=7", "8of8"]
    rng = lambda g: [(min(v[k] for v in J[g].values()), max(v[k] for v in J[g].values())) for k in K]
    assert len(J["base"]) == 1 and len(J["own_r3"]) == 5 and len(J["solvable_only_r3"]) == 4 and len(J["own_r5"]) == 3, {g: len(v) for g, v in J.items()}
    base = [next(iter(J["base"].values()))[k] for k in K]
    S = [("Round 3 (5 seeds)", BLUE, rng("own_r3")),
         ("Solvable-only round 3 (4 seeds)", ORANGE, rng("solvable_only_r3")),
         ("Round 5 (3 seeds)", PURPLE, rng("own_r5"))]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = list(range(len(T)))
    off = {0: -0.13, 1: 0.0, 2: 0.13}
    for k, (name, c, rg) in enumerate(S):
        xs = [i + off[k] for i in x]
        ax.vlines(xs, [lo for lo, _ in rg], [hi for _, hi in rg], color=c, lw=2.6, label=f"{name}: range")
    bplot = [max(v, 0.12) for v in base]  # a zero cannot be drawn on the log axis: drawn at the floor, labelled 0
    ax.plot(x, bplot, "-o", color=INK, ms=3.2, lw=1.0, label="Base model")
    for i, v in enumerate(base): ax.text(i - 0.2, bplot[i], f"{v:g}", ha="right", va="center", fontsize=7, color=INK)
    S = [(n, c, [(max(lo, 0.12), max(hi, 0.12)) for lo, hi in rg]) for n, c, rg in S]
    ax.set_yscale("log"); ax.set_ylim(0.1, 100); ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_xticks(x); ax.set_xticklabels(T); ax.tick_params(axis="x", length=0); ax.set_xlim(-0.6, len(T) - 0.5)
    ax.set_ylabel("Self-consistent answers (%, log scale)"); ax.set_xlabel("Samples that must give the modal answer")
    ax.grid(axis="y", color=GRID, which="major"); ax.set_axisbelow(True)
    ax.legend(loc="lower left", handlelength=1.2, borderaxespad=0.2)
    save(fig, "figA_thresholds.pdf")

for f in (fig_validation, fig_human, fig_prereg, fig_rzacc, fig_rzero, fig_perseed, fig_sizes, fig_thresholds):
    if not args.only or f.__name__ in args.only.split(","): f()
print("done")
