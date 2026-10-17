"""Print the paper's parser-dependent numbers side by side: OLD (original runs, original parser; the stored outputs the paper
cites), NEW-ORIG (rev6 samples, original parser) and NEW-STRICT (rev6 samples, strict parser). Read-only.
python3 rev6_numbers.py > rev6_numbers.txt"""
import json, os, statistics
R = "/home/tione/notebook/rsi_rl"; D6 = f"{R}/paper_arr/rev6"
rr = json.load(open(f"{D6}/rev6_read.json")); ck = json.load(open(f"{D6}/rev6_read_check.json"))
def g(d, path):
    for k in path: d = d[k]
    return d
def rng(xs, f=lambda x: 100 * x, nd=1):
    xs = [f(x) for x in xs]; return f"{min(xs):.{nd}f}-{max(xs):.{nd}f} (mean {statistics.mean(xs):.{nd}f})" if len(xs) > 1 else f"{xs[0]:.{nd}f}"
Q4 = "qwen3_4b_Base_%s_s%d_r%d"
GROUPS = {"4B base": ["qwen3_4b_base"], "4B own r3": [Q4 % ("own_a3", s, 3) for s in range(5)], "4B own r5": [Q4 % ("own_a3", s, 5) for s in range(3)],
          "4B clean r3": [Q4 % ("own_clean_a3", s, 3) for s in range(4)], "4B rnd r3": [Q4 % ("ctrl_rnd_a5", s, 3) for s in range(3)],
          "4B guard r3": [Q4 % ("own_ag_a4", s, 3) for s in range(3)], "4B cln1": [Q4 % ("cln1_a6", s, 3) for s in range(5)],
          "4B rnd1": [Q4 % ("rnd1_a6", s, 3) for s in range(5)], "4B r0": [Q4 % ("r0_a13", s, 3) for s in range(5)],
          "8B base": ["base_Qwen3-8B-Base"], "8B r3": ["qwen3_8b_Base_own_a3_s%d_r3" % s for s in range(3)],
          "Phi base": ["base_Phi-4-mini-instruct"], "Phi r3": ["phi4mini_own_a4_s%d_r3" % s for s in range(2)]}
print("== per checkpoint group, plain prompt: SC | SC_6 | SC_8 | NUM_U | ACC_A   (OLD / NEW-ORIG / NEW-STRICT)")
for name, tags in GROUPS.items():
    for m in ("SC", "SC_6", "SC_8", "NUM_U", "ACC_A"):
        old = [ck["per_tag"][f"plain_{t}"]["ORIGINAL"][m] for t in tags]
        no = [rr["per_tag"][f"plain_{t}"]["ORIGINAL"][m] for t in tags if f"plain_{t}" in rr["per_tag"]]
        ns = [rr["per_tag"][f"plain_{t}"]["STRICT"][m] for t in tags if f"plain_{t}" in rr["per_tag"]]
        print(f"{name:12s} {m:6s} OLD {rng(old):24s} NEW-ORIG {rng(no) if no else '-':24s} NEW-STRICT {rng(ns) if ns else '-'}")
print("\n== STRICT non-answer branches on U samples (share of all U samples)")
for name in ("4B base", "4B own r3", "4B own r5", "4B clean r3", "8B base", "8B r3", "Phi base", "Phi r3"):
    for t in GROUPS[name]:
        e = rr["per_tag"].get(f"plain_{t}")
        if e: print(f"{name:12s} {t:40s}", {k: round(100 * v, 1) for k, v in sorted(e["STRICT"]["U_samples_by_branch"].items())},
                    "mismatch", e["STRICT"]["branch_vs_strict_mismatches"], "| stored-vs-probe-mode mismatches", e["stored_vs_probe_mode_mismatches"],
                    "| strict_parse.original vs stored", e["strict_parse_original_vs_stored_mismatches"])
print("\n== abstention prompt (absfix): ABS_U | ABS_A | CA_abs(SC) | NUM_U   (OLD / NEW-ORIG / NEW-STRICT)")
for name in ("4B base", "4B own r3", "4B clean r3", "4B guard r3", "4B cln1"):
    tags = GROUPS[name] if name != "4B own r3" else GROUPS[name]
    for m in ("ABS_U", "ABS_A", "SC", "NUM_U", "ACC_A"):
        old = [ck["per_tag"][f"absfix_{t}"]["ORIGINAL"][m] for t in tags if f"absfix_{t}" in ck["per_tag"]]
        no = [rr["per_tag"][f"absfix_{t}"]["ORIGINAL"][m] for t in tags if f"absfix_{t}" in rr["per_tag"]]
        ns = [rr["per_tag"][f"absfix_{t}"]["STRICT"][m] for t in tags if f"absfix_{t}" in rr["per_tag"]]
        if old: print(f"{name:12s} {m:6s} OLD {rng(old):24s} NEW-ORIG {rng(no) if no else '-':24s} NEW-STRICT {rng(ns) if ns else '-'}")
print("\n== contrasts (points; mean [CI]; n_neg/n)")
for k in sorted({k.split("|")[1] for k in rr["contrasts"]}):
    row = []
    for src, P in ((ck, "ORIGINAL"), (rr, "ORIGINAL"), (rr, "STRICT")):
        c = src["contrasts"].get(f"{P}|{k}")
        row.append("n/a" if not isinstance(c, dict) else f"{c['mean']:+.1f} [{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}] neg {c['n_neg']}/{len(c['diffs'])}")
    print(f"{k:22s} OLD {row[0]:32s} NEW-ORIG {row[1]:32s} NEW-STRICT {row[2]}")
print("\n== cross-seed agreement (round 5)")
for k in ("umwp_r5", "a8_r5"):
    for src, P, lab in ((ck, "ORIGINAL", "OLD"), (rr, "ORIGINAL", "NEW-ORIG"), (rr, "STRICT", "NEW-STRICT")):
        c = src["cross_seed"].get(f"{P}|{k}")
        if not c: continue
        S, I = c["S"], c["identical"]
        print(f"{k:8s} {lab:10s} S {S} identical {I} ({100 * I / S:.1f}%) null3base {100 * c['mode_null_three_base_runs']:.1f}% eq_base_mode {c['eq_base_mode']} ({100 * c['eq_base_mode'] / I:.1f}%)"
              f" tied_top {c['tied_top_base']} ({100 * c['tied_top_base'] / I:.1f}%) in_base_8 {c['in_base_8']} ({100 * c['in_base_8'] / I:.1f}%) mean_count {c['mean_base_count']:.2f}"
              f" base_SC {c['base_SC']} base_not_SC {c['base_not_SC']} per_seed_eq_base {[round(100 * x, 1) for x in c['per_seed_mode_eq_base_mode_over_S']]}"
              + (f" eq_ref {c['eq_answerable_original_ref']}" if "eq_answerable_original_ref" in c else ""))
print("\n== frontier (CA_U in %)")
for src, P, lab in ((ck, "ORIGINAL", "OLD"), (rr, "ORIGINAL", "NEW-ORIG"), (rr, "STRICT", "NEW-STRICT")):
    F = src["frontier"][P]
    for t in ["qwen3_4b_base"] + [Q4 % ("own_a3", s, 3) for s in range(3)]:
        d = F.get(t)
        if not d: continue
        s = f"{lab:10s} {t:32s} CA@5% {100 * d['CA_at_FA_0.05']:.1f} default (FA {100 * d['default_point'][0]:.1f}, plain SC {100 * d['default_point'][1]:.1f})"
        if "CA_at_guard_FA" in d:
            hg = d["heldout_guardFA"]; h5 = d["heldout_FA5"]
            s += (f" | guardFA {100 * d['guard_FA']:.1f}: in-sample {100 * d['CA_at_guard_FA']:.1f} held-out {100 * hg['split_20260927']['CA_mix']:.1f} (achieved FA {100 * hg['split_20260927']['FA_mix']:.1f})"
                  f" 200-split {100 * hg['200_splits_mean']['CA_mix']:.1f} | 5%: held-out {100 * h5['split_20260927']['CA_mix']:.1f} (FA {100 * h5['split_20260927']['FA_mix']:.1f}) 200-split {100 * h5['200_splits_mean']['CA_mix']:.1f}")
        elif "heldout_FA5" in d:
            s += f" | 5% held-out {100 * d['heldout_FA5']['split_20260927']['CA_mix']:.1f} 200-split {100 * d['heldout_FA5']['200_splits_mean']['CA_mix']:.1f}"
        print(s)
    # mean frontier per condition at 5% and guard plain SC
    for name in ("4B own r3", "4B clean r3", "4B guard r3", "4B cln1"):
        v = [F[t]["CA_at_FA_0.05"] for t in GROUPS[name] if t in F]
        if v: print(f"{lab:10s} {name:12s} CA@5% {rng(v)}")
print("\n== reruns")
def load(p):
    return json.load(open(p)) if os.path.exists(p) else None
for lab, base in (("OLD", None), ("NEW-ORIG", f"{D6}/reruns_original"), ("NEW-STRICT", f"{D6}/reruns_strict")):
    J = load(f"{R}/dirA5/results/j1_judge_free.json" if base is None else f"{base}/j1_judge_free.json")
    if J:
        f = [x["ratio_r3"] / x["ratio_base"] for x in J["JP2"] if x.get("ratio_r3") and x.get("ratio_base")]
        print(f"{lab:10s} J-P2 UMWP in-band ratio growth factor per lineage: {[round(x, 2) for x in f]} range {min(f):.2f}-{max(f):.2f}; all > 1: {all(x > 1 for x in f)}; verdict {J['JP2_verdict']}")
    A6 = load(f"{R}/dirA5/results/a6_matched.json" if base is None else f"{base}/a6_matched.json")
    if A6:
        d = A6["D1_pooled"]; print(f"{lab:10s} A6 D1 pooled matched ratio {d['ratio']:.3f} CI {[round(x, 3) for x in d['ci95_registered']]}")
        for k in ("D2", "D2_pooled"):
            if k in A6: print(f"{lab:10s} A6 {k}:", json.dumps(A6[k], default=str)[:700])
    H = load(f"{R}/paper_arr/analyses/hardtraj.json" if base is None else f"{base}/hardtraj.json")
    if H:
        pe = H["families"]["Qwen3-4B"]["seed_means"]["r3"]["pooled_exits"]
        print(f"{lab:10s} hardtraj Qwen3-4B r3 pooled exits: hard_A {json.dumps(pe['hard_A'])} | U {json.dumps(pe['U'])}")
        fr = H["families"]["Qwen3-4B"].get("fresh_resample_check", {})
        print(f"{lab:10s} hardtraj fresh check (keys): {list(fr.keys())[:6]}")
    T = load(f"{R}/paper_arr/analyses/rev3/ca_thresholds_2026-10-04.json" if base is None else f"{base}/ca_thresholds.json")
    if T:
        for grp, per in T.items():
            keys = list(next(iter(per.values())).keys())
            print(f"{lab:10s} thresholds {grp:18s}", {k: [min(v[k] for v in per.values()), max(v[k] for v in per.values())] for k in keys})
    FR = load(f"{R}/dirA5/results/frontier_read.json" if base is None else f"{base}/frontier_read.json")
    if FR: print(f"{lab:10s} a5_frontier F_P4 (AUROC):", json.dumps(FR.get("F_P4"), default=str)[:900])
for lab, base in (("NEW-ORIG", f"{D6}/reruns_original"), ("NEW-STRICT", f"{D6}/reruns_strict")):
    p = f"{base}/rz_paired_desc.py.log"
    if os.path.exists(p):
        txt = open(p).read().split("\n--- stderr")[0]
        try:
            d = json.loads(txt)
            for k, v in d.items():
                if k.startswith("plain"): print(f"{lab:10s} R-Zero {k}: CA {v['CA_unanswerable']['base']}->{v['CA_unanswerable']['final']} diff {v['CA_unanswerable']['diff']} CI {v['CA_unanswerable']['ci95']} | ACC {v['ACC_answerable']['base']}->{v['ACC_answerable']['final']}")
            for t, v in d["response_rates"].items(): print(f"{lab:10s} R-Zero rates {t}: {v}")
        except Exception as e: print(lab, "rz log not JSON:", e, txt[:300])
print("\n== frontier CA_U at FA_A = 0.05 (a5_frontier.py curve/frontier_at on the pseudo-dumps; OLD = original absfix dumps)")
src = open(f"{R}/dirA5/code/a5_frontier.py").read()
TAG2 = ["qwen3_4b_base"] + [Q4 % ("own_a3", s, 3) for s in range(5)] + [Q4 % ("own_clean_a3", s, 3) for s in range(4)] + [Q4 % ("own_ag_a4", s, 3) for s in range(3)] \
    + [Q4 % ("rescue_uv_a4", s, 3) for s in (0, 1, 2, 4)] + [Q4 % ("rescue_abs_a4", s, 3) for s in range(3)] + [Q4 % ("loop_uv_a4", s, 4) for s in range(3)] \
    + [Q4 % ("coev_ag_a4", s, 3) for s in range(2)] + [Q4 % ("tg_abs4_a5", s, 4) for s in range(3)] + [Q4 % ("tg_t4_a5", 1, 4)]
CA5 = {}
for lab, d in (("OLD", None), ("NEW-ORIG", f"{D6}/dumps_original"), ("NEW-STRICT", f"{D6}/dumps_strict")):
    s2 = src if d is None else src.replace("/dirA4/results/umwp", d.replace(R, ""))
    nsf = {}; exec(compile(s2[:s2.index("res = {")], "a5_frontier_defs", "exec"), nsf)
    for t in TAG2:
        it = nsf["load_umwp"](t, "absfix" if d is None else "abs")
        CA5[(lab, t)] = None if it is None else nsf["frontier_at"](nsf["curve"](it), 0.05)
    row = " ".join(f"{t.replace('qwen3_4b_Base_', '')}:{'-' if CA5[(lab, t)] is None else round(100 * CA5[(lab, t)], 1)}" for t in TAG2)
    print(lab, row)
    for arm, ss in (("rescue_uv_a4", (0, 1, 2, 4)), ("rescue_abs_a4", (0, 1, 2)), ("coev_ag_a4", (0, 1)), ("tg_abs4_a5", (0, 1, 2)), ("own_ag_a4", (0, 1, 2)), ("own_clean_a3", (0, 1, 2, 3))):
        rnd = 4 if arm.startswith(("tg_",)) else 3
        dif = [CA5[(lab, Q4 % (arm, s, rnd))] - CA5[(lab, Q4 % ("own_a3", s, 3))] for s in ss
               if CA5.get((lab, Q4 % (arm, s, rnd))) is not None and CA5.get((lab, Q4 % ("own_a3", s, 3))) is not None]
        if dif: print(f"   {lab:10s} {arm:14s} minus own r3 (same seed), points: {[round(100 * x, 1) for x in dif]} mean {100 * statistics.mean(dif):+.1f}")
    dif = [CA5[(lab, Q4 % ("loop_uv_a4", s, 4))] - CA5[(lab, Q4 % ("own_a3", s, 3))] for s in range(3)
           if CA5.get((lab, Q4 % ("loop_uv_a4", s, 4))) is not None and CA5.get((lab, Q4 % ("own_a3", s, 3))) is not None]
    if dif: print(f"   {lab:10s} loop_uv r4     minus own r3 (same seed), points: {[round(100 * x, 1) for x in dif]} mean {100 * statistics.mean(dif):+.1f}")
