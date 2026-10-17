# rev6 rerun copy of dirA5/code/a7_exit.py (UMWP dir -> dumps_original, 2 occurrence(s); output -> reruns_original)
"""A7 item 4 (registered 15:10 24 Sep): differential exit over time on UMWP, using the A6 D2 definitions (functions imported
unchanged from a6_matched.py v2: load_umwp, cells, std_ratio; hard answerable = base gold accuracy <= 3/8, reference required;
base bins k = round(8 m_base) in 2..6). For Qwen3-4B own seeds 0-2, at r3 and at r5:
  ratio = D2 bin-matched standardized retention ratio (unanswerable vs hard answerable) - R-P1: ratio(r5) >= ratio(r3) + 0.10
          in >= 2 of 3 seeds;
  hard retention = the in-band share at round r of the hard-answerable items in base bins 2..6 (A6's per-class retention,
          unweighted) - R-P2: hard retention(r5) < hard retention(r3) in >= 2 of 3 seeds.
v2 after a two-reviewer check (v1 never run). Rules fixed before any r5 statistic is computed or viewed:
- D2 underpowered rule (n matched U >= 30) applies; an underpowered seed does not hold and is reported "underpowered, not
  read" (it does not trigger the registered downside sentence).
- If ratio(r3) is infinite (no hard-answerable item in band) R-P1 does not hold for that seed; inf is serialized as "inf".
- Verdicts are three-state (A6 tri): True if holds >= 2; False if holds + unread < 2; else None.
- Sensitivity (descriptive): R-P2 with the U-standardized hard retention (the D2 ratio denominator); per-bin counts; the
  nA >= 5 bin sensitivity of the ratio.
- Disclosure: the r3 terms (A6 D2 ratios, n) were on disk since 09:53, before this item was registered; the confirmatory
  content is the r5 read.
Item 2 (descriptive, no prediction registered): per UMWP category code, the J-P2 in-band ratio (in-band rate of unanswerable
items of the category / in-band rate of all 600 answerable items; J-P2 in-band definition; the denominator is shared, so
differences between categories reflect the unanswerable side only) at base and r3 and its x change, for the 10 own lineages,
with in-band counts and small-count flags (< 10 in band at base). Category names are attached only if the code mapping is
verified; otherwise codes are reported.
python3 a7_exit.py"""
import json, math, os
from collections import defaultdict
R = "/home/tione/notebook/rsi_rl"
src = open(f"{R}/dirA5/code/a6_matched.py").read().replace("/paper_arr/rev6/dumps_original", "/paper_arr/rev6/dumps_original")
ns = {}; exec(src[:src.index("LIN = [")], ns)
load, cells, std_ratio, tri, BINS = ns["load_umwp"], ns["cells"], ns["std_ratio"], ns["tri"], ns["BINS"]
hard = lambda it: it["gold"] is not None and it["gold"] <= 3 / 8 + 1e-9
def ret_raw(rA):
    xs = [x for k in BINS for _, x in rA[k]]; return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
def ret_std(rU, rA):
    bins = [k for k in BINS if rU[k] and rA[k]]; n = sum(len(rU[k]) for k in bins)
    return None if n == 0 else sum(len(rU[k]) / n * (sum(x for _, x in rA[k]) / len(rA[k])) for k in bins)
base = load("qwen3_4b_base")
item4 = {"seeds": []}
rp1, rp2, rp2s = [], [], []
for s in range(3):
    rec = {"seed": s}
    for rd in (3, 5):
        d = load(f"qwen3_4b_Base_own_a3_s{s}_r{rd}")
        if d is None or base is None: rec[f"r{rd}"] = None; continue
        rU, rA = cells(base, d, subset_A=hard)
        ratio, n, bins = std_ratio(rU, rA); ratio5, _, _ = std_ratio(rU, rA, min_nA=5)
        hr, nA = ret_raw(rA)
        rec[f"r{rd}"] = {"ratio": ratio, "n_matched_U": n, "powered": n >= 30, "hard_ret": hr, "n_hard_A": nA, "hard_ret_std": ret_std(rU, rA),
                         "ratio_bins_nA_ge5": ratio5, "per_bin_nU_nA": {k: [len(rU[k]), len(rA[k])] for k in BINS}}
    a, b = rec.get("r3"), rec.get("r5")
    if a is None or b is None: p1 = p2 = p2s = None
    elif not (a["powered"] and b["powered"]): p1 = p2 = p2s = False; rec["note"] = "underpowered, not read"
    else:
        p1 = False if (a["ratio"] is None or b["ratio"] is None or math.isinf(a["ratio"])) else b["ratio"] >= a["ratio"] + 0.10 - 1e-9
        p2 = None if a["hard_ret"] is None or b["hard_ret"] is None else b["hard_ret"] < a["hard_ret"]
        p2s = None if a["hard_ret_std"] is None or b["hard_ret_std"] is None else b["hard_ret_std"] < a["hard_ret_std"]
    rec.update({"RP1": p1, "RP2": p2, "RP2_std_sensitivity": p2s}); rp1.append(p1); rp2.append(p2); rp2s.append(p2s)
    item4["seeds"].append(rec)
    print({k: v for k, v in rec.items() if k not in ("r3", "r5")}, {rd: (rec[rd] and {k: rec[rd][k] for k in ("ratio", "n_matched_U", "hard_ret", "n_hard_A", "hard_ret_std")}) for rd in ("r3", "r5")})
v = lambda l: tri(sum(x is True for x in l), sum(x is not None for x in l), 3, 2)
item4["verdicts"] = {"R-P1": v(rp1), "R-P2": v(rp2), "R-P2_std_sensitivity": v(rp2s),
                     "underpowered_seeds": [r["seed"] for r in item4["seeds"] if r.get("note")]}
print("A7 item 4 verdicts:", item4["verdicts"])
# ---- item 2 (descriptive)
UM = f"{R}/paper_arr/rev6/dumps_original"
MAP_F = f"{R}/dirA5/results/umwp_category_map.json"
CAT = json.load(open(MAP_F)) if os.path.exists(MAP_F) else None
def raw(tag):
    f = f"{UM}/plain_{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    return rows if len(rows) >= 1200 else None
LIN = [("qwen3_4b_base", f"qwen3_4b_Base_own_a3_s{s}_r3") for s in range(5)] + [("base_Qwen3-8B-Base", f"qwen3_8b_Base_own_a3_s{s}_r3") for s in range(3)] + \
      [("base_Phi-4-mini-instruct", f"phi4mini_own_a4_s{s}_r3") for s in range(2)]
inb = lambda r: 0.25 <= ns["nshare"](r["answers"]) <= 0.75
def cat_stats(rows):
    A = [inb(r) for r in rows.values() if r["answerable"]]; ra = sum(A) / len(A)
    by = defaultdict(list)
    for r in rows.values():
        if not r["answerable"]:
            assert r["category"] in (1, 2, 3, 4, 5), r["category"]
            by[str(r["category"])].append(inb(r))
    return ra, sum(A), {c: {"n": len(x), "n_inband": sum(x), "rate": sum(x) / len(x), "ratio": (sum(x) / len(x)) / ra if ra else None} for c, x in sorted(by.items())}
item2 = {"tag": "descriptive: no prediction registered; no thresholds apply per category", "mapping": CAT or "codes only (mapping not verified)", "lineages": {}}
bc = {}
for b, run in LIN:
    if b not in bc: bc[b] = raw(b)
    B_, Rr = bc[b], raw(run)
    if B_ is None or Rr is None: item2["lineages"][run] = None; continue
    ra_b, nAb, cb = cat_stats(B_); ra_r, nAr, cr = cat_stats(Rr)
    ch = {}
    for c in cb:
        if cb[c]["ratio"] is None or cr[c]["ratio"] is None: ch[c] = None
        elif cb[c]["n_inband"] == 0: ch[c] = "base in-band count 0"
        else: ch[c] = cr[c]["ratio"] / cb[c]["ratio"]
    item2["lineages"][run] = {"powered_JP2": nAb >= 30 and nAr >= 30, "n_inband_A": [nAb, nAr], "base": cb, "r3": cr, "change_x": ch,
                              "small_count_flag": {c: cb[c]["n_inband"] < 10 for c in cb}}
print("item 2 (descriptive) x change of in-band U_cat/A ratio, base -> r3:")
for run, x in item2["lineages"].items():
    if x: print(run, {c: (round(y, 2) if isinstance(y, float) else y) for c, y in x["change_x"].items()})
def clean(o):
    if isinstance(o, float) and math.isinf(o): return "inf"
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    return o
json.dump(clean({"item4_confirmatory": item4, "item2_descriptive": item2}), open(f"{R}/paper_arr/rev6/reruns_original/a7_exit_categories.json", "w"), indent=1)
