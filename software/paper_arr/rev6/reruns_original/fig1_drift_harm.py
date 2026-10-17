# rev6 rerun copy of paper_arr/figures/fig1_drift_harm.py (UMWP dir -> dumps_original, 1 occurrence(s); output -> reruns_original)
"""Figure 1: drift of the admitted curriculum and the downstream harm.

(a) share of admitted problems that are ill-posed (UNSOLVABLE + AMBIGUOUS in the 200-item Opus audit)
    per self-play round: own loops of Qwen3-4B-Base, Qwen3-8B-Base, Phi-4-mini-instruct (mean over seeds,
    min-max band), the official R-Zero code on Qwen3-4B-Base (iterations 1-3), blind re-audits (hollow),
    and the ill-posed share of the frozen proposer's raw round-1 output (dashed).
(b) confident answering of the 600 unanswerable UMWP questions (numeric modal answer with share >= 0.5
    over 8 samples, plain prompt) vs mean greedy lenient accuracy on MATH-500, MATH held-out-1000, GSM8K-500.
All data are read read-only.
Layout (r1, 2026-09-26): 6.30 in wide (ACL text width), included at 100% scale; all text >= 8 pt.
In (b) per-seed points are small and semi-transparent; the seed means are large outlined markers and
the arrows (base model -> mean of its round-3 solvers) end at the means.
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/home/tione/notebook/rsi_rl"
RUNS = f"{ROOT}/dirB/runs"
EVALS = f"{ROOT}/dirB/results/evals"
UMWP = f"{ROOT}/paper_arr/rev6/dumps_original"
RZ = f"{ROOT}/ext/rzero_runs/rz_read_final_2026-09-26.json"
A75D = f"{ROOT}/dirA5/results/a75d_blind/a75d_read.json"
A7B = f"{ROOT}/dirA5/results/a7b_blind/a7b_read.json"
DATASHEET = f"{ROOT}/release_v1/DATASHEET.md"
OUT = f"{ROOT}/paper_arr/rev6/reruns_original/fig1_drift_harm.pdf"
PREVIEW = "/tmp/fig1_drift_harm_preview.png"

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

C = {"q4": "#1f77b4", "q8": "#2ca02c", "phi": "#9467bd", "rz": "#d62728",
     "clean": "#ff7f0e", "base": "#7f7f7f", "ink": "#333333"}
M = {"q4": "s", "q8": "^", "phi": "D", "rz": "o", "clean": "X"}
plotted = {"a": {}, "b": {}}


# ---------------------------------------------------------------- panel (a) data
def ill_share(run):
    """UNSOLVABLE + AMBIGUOUS share (%) of the 200-item audit of admitted problems."""
    d = json.load(open(f"{RUNS}/{run}/audit_admitted.json"))
    c = d["counts"]
    n = sum(c.values())
    assert n == 200, (run, n)
    assert len(d["verdicts"]) == 200, (run, len(d["verdicts"]))
    return 100.0 * (c["UNSOLVABLE"] + c["AMBIGUOUS"]) / n, c


LOOPS = {
    "q4": ("qwen3_4b_Base_own_a3_s{s}_r{r}", {1: range(5), 2: range(5), 3: range(5),
                                               4: range(3), 5: range(3), 6: range(3)}),
    "q8": ("qwen3_8b_Base_own_a3_s{s}_r{r}", {1: range(3), 2: range(3), 3: range(3),
                                               4: (1, 2), 5: (1, 2), 6: (1, 2)}),
    "phi": ("phi4mini_own_a4_s{s}_r{r}", {1: range(2), 2: range(2), 3: range(2)}),
}
series = {}
for key, (pat, seeds_by_round) in LOOPS.items():
    rows = []
    for r, seeds in seeds_by_round.items():
        vals = {}
        for s in seeds:
            run = pat.format(s=s, r=r)
            v, c = ill_share(run)
            vals[s] = v
            plotted["a"][f"{key}_s{s}_r{r}"] = {"value_pct": v, "counts": c,
                                                 "source": f"dirB/runs/{run}/audit_admitted.json:counts"}
        vs = list(vals.values())
        rows.append((r, sum(vs) / len(vs), min(vs), max(vs), sorted(vals)))
    series[key] = rows

rz = json.load(open(RZ))
rows = []
for it in (1, 2, 3):
    vals = {}
    for s in ("0", "1"):
        a = rz["per_seed"][s]["audit"][str(it)]
        assert a["complete"] and a["n"] == 200
        c = a["counts"]
        chk = 100.0 * (c["UNSOLVABLE"] + c["AMBIGUOUS"]) / 200
        assert abs(chk - 100 * a["ill"]) < 1e-9, (s, it)
        vals[int(s)] = 100.0 * a["ill"]
        plotted["a"][f"rz_s{s}_it{it}"] = {"value_pct": vals[int(s)], "counts": c,
                                           "source": f"ext/rzero_runs/rz_read_final_2026-09-26.json:per_seed.{s}.audit.{it}.ill"}
    vs = list(vals.values())
    rows.append((it, sum(vs) / len(vs), min(vs), max(vs), sorted(vals)))
series["rz"] = rows

# blind re-audits
a75d = json.load(open(A75D))
assert a75d["complete"]
rz_blind = []
for it in (1, 2, 3):
    vs = []
    for s in (0, 1):
        cell = a75d["cells"][f"s{s}_it{it}"]
        vs.append(cell["blind"])
        plotted["a"][f"rz_blind_s{s}_it{it}"] = {"value_pct": cell["blind"], "n": cell["n"], "orig_same_items": cell["orig"],
                                                 "source": f"dirA5/results/a75d_blind/a75d_read.json:cells.s{s}_it{it}.blind"}
    rz_blind.append((it, sum(vs) / len(vs)))
    plotted["a"][f"rz_blind_mean_it{it}"] = {"value_pct": sum(vs) / len(vs), "plotted": True}

a7b = json.load(open(A7B))
assert a7b["complete"]
q4_blind = []
for rd in (1, 3, 5):
    vs = []
    for s in (0, 1, 2):
        cell = a7b["cells"][f"s{s}_r{rd}"]
        vs.append(cell["blind"])
        plotted["a"][f"q4_blind_s{s}_r{rd}"] = {"value_pct": cell["blind"], "n": cell["n"], "orig_same_items": cell["orig"],
                                                "source": f"dirA5/results/a7b_blind/a7b_read.json:cells.s{s}_r{rd}.blind"}
    q4_blind.append((rd, sum(vs) / len(vs)))
    plotted["a"][f"q4_blind_mean_r{rd}"] = {"value_pct": sum(vs) / len(vs), "plotted": True}

# frozen proposer raw output (fixed pool qwen_s0_r1), checked against the datasheet row
ds = open(DATASHEET).read()
assert "fixed_pool_audits/qwen_s0_r1.jsonl | 1158 of 1158 | 639 / 339 / 180 |" in ds
PROPOSER = 100.0 * (339 + 180) / 1158
plotted["a"]["proposer_output"] = {"value_pct": PROPOSER, "counts": "339 U + 180 A of 1158",
                                   "source": "release_v1/DATASHEET.md fixed_pool_audits/qwen_s0_r1.jsonl row"}
for key, rows in series.items():
    for (r, mean, lo, hi, seeds) in rows:
        plotted["a"][f"{key}_mean_r{r}"] = {"mean": mean, "min": lo, "max": hi, "seeds": seeds, "plotted": True}


# ---------------------------------------------------------------- panel (b) data
def ca(tag):
    n = c = 0
    for line in open(f"{UMWP}/{tag}.jsonl"):
        r = json.loads(line)
        if r["answerable"]:
            continue
        n += 1
        if r["mode"] is not None and r["p_mode"] >= 0.5:
            c += 1
    assert n == 600, (tag, n)
    return 100.0 * c / n, c


BENCH = ("math500_train", "math_heldout1000", "gsm8k_test500")


def acc(run):
    p = f"{EVALS}/{run}.summary.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    v = [d[k]["greedy_acc_lenient"] for k in BENCH]
    return 100.0 * sum(v) / 3, v


POINTS = {  # family -> (base (umwp tag, eval run), [(group, umwp tag, eval run)])
    "q4": (("plain_qwen3_4b_base", "qwen3_4b_base_step0"),
           [("q4", f"plain_qwen3_4b_Base_own_a3_s{s}_r3", f"qwen3_4b_Base_own_a3_s{s}_r3_step60") for s in range(5)]
           + [("clean", f"plain_qwen3_4b_Base_own_clean_a3_s{s}_r3", f"qwen3_4b_Base_own_clean_a3_s{s}_r3_step60")
              for s in range(4)]),
    "q8": (("plain_base_Qwen3-8B-Base", "qwen3_8b_base_step0"),
           [("q8", f"plain_qwen3_8b_Base_own_a3_s{s}_r3", f"qwen3_8b_Base_own_a3_s{s}_r3_step60") for s in range(3)]),
    "phi": (("plain_base_Phi-4-mini-instruct", "phi4mini_step0"),
            [("phi", f"plain_phi4mini_own_a4_s{s}_r3", f"phi4mini_own_a4_s{s}_r3_step60") for s in range(2)]),
}
bpts = {}
missing = []
for fam, (base, pts) in POINTS.items():
    for grp, tag, run in [("base_" + fam, base[0], base[1])] + pts:
        y, c = ca(tag)
        x, v = acc(run)
        if x is None:
            missing.append(run)
            continue
        bpts.setdefault(grp, []).append((x, y))
        plotted["b"][tag] = {"group": grp, "x_acc_pct": x, "acc_math500_heldout1000_gsm8k500": v,
                             "y_CA_pct": y, "n_confident": c, "n_unanswerable": 600,
                             "sources": [f"dirA4/results/umwp/{tag}.jsonl",
                                         f"dirB/results/evals/{run}.summary.json:greedy_acc_lenient"]}
plotted["missing_accuracy_summaries"] = missing
if missing:
    print("MISSING accuracy summaries (points omitted):", missing, file=sys.stderr)


def mean_pt(p):
    return (sum(a for a, _ in p) / len(p), sum(b for _, b in p) / len(p))


for grp, p in bpts.items():
    plotted["b"][f"mean_{grp}"] = {"x": mean_pt(p)[0], "y": mean_pt(p)[1], "n": len(p)}

# ---------------------------------------------------------------- figure
FIG_W, FIG_H = 6.30, 2.35  # ACL \textwidth, included at 100% scale; at 2.50 in the paper body overflowed page 8
fig, (ax, bx) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
DX = 0.13  # horizontal offset of the blind re-audit markers
MS = 4.4   # line-marker size in panel (a) and in the legend

for key in ("phi", "q8", "rz", "q4"):  # Qwen3-4B (primary, 5 seeds) drawn last so its markers stay visible
    rows = series[key]
    xs = [r for r, *_ in rows]
    ax.fill_between(xs, [lo for _, _, lo, _, _ in rows], [hi for _, _, _, hi, _ in rows],
                    color=C[key], alpha=0.14, lw=0, zorder=1)
    ax.plot(xs, [m for _, m, *_ in rows], color=C[key], lw=1.2, marker=M[key], ms=MS,
            mec="white", mew=0.5, zorder=3)
ax.plot([r + DX for r, _ in rz_blind], [v for _, v in rz_blind], ls="none", marker=M["rz"], ms=MS,
        mfc="white", mec=C["rz"], mew=0.9, zorder=4)
ax.plot([r + DX for r, _ in q4_blind], [v for _, v in q4_blind], ls="none", marker=M["q4"], ms=MS,
        mfc="white", mec=C["q4"], mew=0.9, zorder=4)
# R-Zero challenger output before any filtering (A12 part 1, registered blind audit, 400 raw questions per iteration)
A12P1 = f"{ROOT}/dirA5/results/a12_blind/a12_p1_read.json"
raw = json.load(open(A12P1))["primary"]["pooled_raw_share"]
rz_raw = [(1, raw["it1"]), (2, raw["it2"]), (3, raw["it3"])]
plotted["a"]["rz_raw_challenger"] = {"values_pct": rz_raw, "source": "dirA5/results/a12_blind/a12_p1_read.json:primary.pooled_raw_share"}
ax.plot([r for r, _ in rz_raw], [v for _, v in rz_raw], color=C["rz"], ls=(0, (1.2, 1.4)), lw=1.1, marker=M["rz"],
        ms=MS - 0.6, mfc="white", mec=C["rz"], mew=0.8, zorder=3)
ax.text(3.2, rz_raw[-1][1] - 0.5, "challenger output", ha="left", va="center", fontsize=8, color=C["rz"])
ax.axhline(PROPOSER, color=C["ink"], ls=(0, (4, 2)), lw=0.8, zorder=2)
ax.text(6.38, PROPOSER - 1.8, f"Proposer output ({PROPOSER:.1f}%)", ha="right", va="top",
        fontsize=8, color=C["ink"])
ax.set_xlim(0.7, 6.45)
ax.set_ylim(0, 100)
ax.set_xticks(range(1, 7))
ax.set_yticks(range(0, 101, 20))
ax.set_xlabel("Self-play round (R-Zero: iteration)")
ax.set_ylabel("Ill-posed admitted problems (%)")
ax.grid(True, color="#cccccc")
ax.set_axisbelow(True)

# panel (b): per-seed round-3 solvers small and semi-transparent, seed means large and outlined,
# arrows from each base model end at the mean of its round-3 solvers
SEED_MS, MEAN_MS, BASE_MS = 3.6, 6.4, 5.4
SEED_ALPHA = 0.5
MEAN_EDGE = "#1a1a1a"
for fam in ("q4", "q8", "phi"):
    bxy = bpts["base_" + fam][0]
    for grp in ([fam, "clean"] if fam == "q4" else [fam]):
        if grp not in bpts:
            continue
        mx, my = mean_pt(bpts[grp])
        bx.annotate("", xy=(mx, my), xytext=bxy,
                    arrowprops=dict(arrowstyle="-|>", color=C[grp], lw=0.9, alpha=0.9,
                                    mutation_scale=7.5, shrinkA=BASE_MS / 2 + 0.8, shrinkB=MEAN_MS / 2 + 0.4),
                    zorder=2)
for grp in ("q4", "clean", "q8", "phi"):
    p = bpts.get(grp, [])
    bx.plot([a for a, _ in p], [b for _, b in p], ls="none", marker=M[grp],
            ms=SEED_MS if grp != "clean" else SEED_MS + 0.5,
            color=C[grp], mec="none", alpha=SEED_ALPHA, zorder=3)
for grp in ("q4", "clean", "q8", "phi"):  # seed means on top
    if grp not in bpts:
        continue
    mx, my = mean_pt(bpts[grp])
    bx.plot([mx], [my], ls="none", marker=M[grp], ms=MEAN_MS if grp != "clean" else MEAN_MS + 0.6,
            mfc=C[grp], mec=MEAN_EDGE, mew=0.8, zorder=5)
LBL = {"q4": ("Qwen3-4B base", (0.0, -2.6), "center", "top"),  # r6 (7 Oct): positions for the strict-reading values
       "q8": ("Qwen3-8B base", (0.3, -1.0), "left", "top"),  # left of the marker: its arrow leaves to the right
       "phi": ("Phi-4-mini base", (0.0, -2.0), "center", "top")}  # above the Qwen3-8B arrow
for fam in ("q4", "q8", "phi"):
    x0, y0 = bpts["base_" + fam][0]
    bx.plot([x0], [y0], ls="none", marker=M[fam], ms=BASE_MS, color=C["base"], mec="white", mew=0.5, zorder=4)
    t, (dx, dy), ha, va = LBL[fam]
    bx.text(x0 + dx, y0 + dy, t, fontsize=8, color=C["ink"], ha=ha, va=va, zorder=6)
bx.set_xlim(71, 85)
bx.set_xticks(range(72, 85, 2))
bx.set_ylim(0, 100)
bx.set_yticks(range(0, 101, 20))
bx.set_xlabel("Benchmark accuracy (%)")
bx.set_ylabel("Self-consistent answers to\nunanswerable questions (%)")
bx.grid(True, color="#cccccc")
bx.set_axisbelow(True)
key = [Line2D([], [], ls="none", marker="o", ms=SEED_MS, color="#555555", mec="none", alpha=SEED_ALPHA,
              label="Round 3, per seed"),
       Line2D([], [], ls="none", marker="o", ms=MEAN_MS, mfc="#8a8a8a", mec=MEAN_EDGE, mew=0.8,
              label="Round 3, mean")]
bx.legend(handles=key, loc="upper right", frameon=False, handlelength=1.0, handletextpad=0.4,
          borderaxespad=0.2, labelspacing=0.3, borderpad=0.2)

for a, lab in ((ax, "(a)"), (bx, "(b)")):  # panel labels inside the axes
    a.text(0.018, 0.985, lab, transform=a.transAxes, fontsize=9, fontweight="bold", va="top", ha="left",
           zorder=7)

handles = [  # column-major order for ncol=3: rows read q4 q8 phi / R-Zero clean blind
    Line2D([], [], color=C["q4"], lw=1.2, marker=M["q4"], ms=MS, mec="white", mew=0.5, label="Qwen3-4B-Base"),
    Line2D([], [], color=C["rz"], lw=1.2, marker=M["rz"], ms=MS, mec="white", mew=0.5, label="R-Zero (official code)"),
    Line2D([], [], color=C["q8"], lw=1.2, marker=M["q8"], ms=MS, mec="white", mew=0.5, label="Qwen3-8B-Base"),
    Line2D([], [], color=C["clean"], ls="none", marker=M["clean"], ms=MS + 0.8, mec="white", mew=0.5,
           label="Solvable-only curriculum"),
    Line2D([], [], color=C["phi"], lw=1.2, marker=M["phi"], ms=MS, mec="white", mew=0.5, label="Phi-4-mini-instruct"),
    Line2D([], [], color=C["ink"], ls="none", marker="o", ms=MS, mfc="white", mec=C["ink"], mew=0.9,
           label="Blind re-audit"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.8, columnspacing=1.6, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.3)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 1 - 0.3125 / FIG_H), w_pad=1.5)  # 0.3125 in band for the legend
fig.savefig(OUT)
if "--preview" in sys.argv:
    fig.savefig(PREVIEW, dpi=220)
print(json.dumps(plotted, indent=1, default=str))
