"""A6 D1-D4 (registered 09:32 on 24 Sep), v2 after a two-reviewer adversarial check (v1 was never run). Definitions: see
dirA5/docs/A6_registration_2026-09-24.md section D. Fixes and disclosed choices (made before any run):
- D3 uses the exact standardized (weighted) median of A (weights nU_k/nA_k), not a random resample; ties are recorded.
- Underpowered lineages (< 30 matched U) are excluded from the primary pooled ratio and CI (registration: "excluded").
- D1 primary CI = the registered stratified bootstrap (lineage x bin x class); an item-clustered bootstrap CI (items resampled
  once per replicate within class and applied across lineages, since lineages share the UMWP items) is reported as a
  disclosed secondary sensitivity.
- D4 (twins): base items restricted to the band 0.25 <= m32 <= 0.75 before binning (k = round(8 m32), Python rounding);
  D4 is read with the full D1 criterion, including a pooled CI excluding 1 (the stricter reading of "the D1 criterion").
- D2: answerable items whose reference did not parse (ref None) are excluded from the hard subset; counts are recorded.
- Verdicts are three-state: True (threshold met, and CI excludes 1 where required), False (cannot be met: holds + unread <
  threshold, underpowered lineages counting as not holding), None (not read: too many lineages missing to decide).
- den = 0 gives an infinite ratio when U retention > 0; None results are dropped from bootstraps (count recorded).
- Descriptive only: per-bin counts and rates per lineage; a sensitivity dropping bins with nA < 5; per-class count of matched
  items with m_r3 == 0.
python3 a6_matched.py"""
import glob, json, math, os, random
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"; UM = f"{R}/dirA4/results/umwp"; H = f"{R}/dirB/results/hp"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
BINS = [2, 3, 4, 5, 6]


def nshare(ans):
    c = Counter(a for a in ans if a is not None and a != "unknown")
    return (c.most_common(1)[0][1] if c else 0) / len(ans)


def load_umwp(tag):
    f = f"{UM}/plain_{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    if sum(not r["answerable"] for r in rows.values()) != 600 or sum(r["answerable"] for r in rows.values()) != 600: return None
    out = {}
    for i, r in rows.items():
        ans = r["answers"]
        gold = None
        if r["answerable"] and r.get("ref") is not None:
            gold = sum(1 for a in ans if a is not None and a == r["ref"]) / len(ans)
        out[i] = {"U": not r["answerable"], "m": nshare(ans), "gold": gold}
    return out


def load_twins(run=None, pattern=None, rng=None):
    fs = glob.glob(pattern) if pattern else glob.glob(f"{H}/hp_{run}_shard*.jsonl")
    rows = {}
    for f in fs:
        for l in open(f):
            try: r = json.loads(l); rows[(str(r["fam"]), int(r["task"]), r["form"])] = r
            except ValueError: pass
    U = [r for k, r in rows.items() if k[2] == "deleted" and k[:2] in UNS]
    uk = {(str(r["fam"]), int(r["task"])) for r in U}
    A = [r for k, r in rows.items() if k[2] == "intact" and k[:2] in uk]
    if len(U) != 412 or len(A) != 412: return None
    out = {}
    for r in U + A:
        ans = r["answers"]   # m = full-sample modal share; only the in-band probability is subsampled to G = 8
        if len(ans) > 8:
            draws = [rng.sample(ans, 8) for _ in range(2000)]
            inb = sum(0.25 <= nshare(d) <= 0.75 for d in draws) / 2000
        else:
            inb = float(0.25 <= nshare(ans) <= 0.75)
        out[(str(r["fam"]), int(r["task"]), r["form"])] = {"U": r["form"] == "deleted", "m": nshare(ans), "inband": inb, "gold": None}
    return out


def cells(base, r3, subset_A=None, twins=False):
    """Per base bin: lists of (item, retention) for U and for A."""
    rU, rA = defaultdict(list), defaultdict(list)
    for i, b in base.items():
        if i not in r3: continue
        if twins and not (0.25 <= b["m"] <= 0.75): continue
        k = round(8 * b["m"])
        if k not in BINS: continue
        inb = r3[i]["inband"] if twins else float(0.25 <= r3[i]["m"] <= 0.75)
        if b["U"]: rU[k].append((i, inb))
        elif subset_A is None or subset_A(b): rA[k].append((i, inb))
    return rU, rA


def std_ratio(rU, rA, min_nA=1):
    bins = [k for k in BINS if rU[k] and len(rA[k]) >= min_nA]
    n = sum(len(rU[k]) for k in bins)
    if n == 0: return None, 0, bins
    num = sum(len(rU[k]) / n * (sum(x for _, x in rU[k]) / len(rU[k])) for k in bins)
    den = sum(len(rU[k]) / n * (sum(x for _, x in rA[k]) / len(rA[k])) for k in bins)
    return (num / den if den > 0 else (math.inf if num > 0 else None)), n, bins


def pooled(rows):
    g = defaultdict(lambda: {"U": [], "A": []})
    for lin, k, c, i, x in rows: g[(lin, k)][c].append(x)
    cs = [v for v in g.values() if v["U"] and v["A"]]
    n = sum(len(v["U"]) for v in cs)
    if n == 0: return None
    num = sum(len(v["U"]) / n * (sum(v["U"]) / len(v["U"])) for v in cs); den = sum(len(v["U"]) / n * (sum(v["A"]) / len(v["A"])) for v in cs)
    return num / den if den > 0 else (math.inf if num > 0 else None)


def boot_strat(rows, B=2000, seed=20260924):
    cs = defaultdict(list)
    for row in rows: cs[(row[0], row[1], row[2])].append(row)
    rng = random.Random(seed); out = []
    for _ in range(B):
        rs = []
        for key, rr in cs.items(): rs += [rr[rng.randrange(len(rr))] for _ in rr]
        out.append(pooled(rs))
    return out


def boot_items(rows, B=2000, seed=20260924):
    """Item-clustered: resample item ids within class once per replicate; apply the draw across lineages."""
    byitem = defaultdict(list)
    for row in rows: byitem[(row[2], row[3])].append(row)
    ids = {c: sorted(i for (cc, i) in byitem if cc == c) for c in ("U", "A")}
    rng = random.Random(seed); out = []
    for _ in range(B):
        rs = []
        for c in ("U", "A"):
            for _ in ids[c]: rs += byitem[(c, ids[c][rng.randrange(len(ids[c]))])]
        out.append(pooled(rs))
    return out


def ci(bs):
    v = sorted(x for x in bs if x is not None)
    if len(v) < 0.95 * len(bs): return None, len(bs) - len(v)
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]], len(bs) - len(v)


def sharp(qb, qr):
    qb, qr = min(max(qb, 1 / 16), 15 / 16), min(max(qr, 1 / 16), 15 / 16)
    return math.log(qb) / math.log(qr)


def median(x):
    x = sorted(x); n = len(x)
    return None if n == 0 else (x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2)


def wmedian(p):
    p = sorted(p); tot = sum(w for _, w in p); c = 0.0
    for j, (v, w) in enumerate(p):
        c += w
        if c > tot / 2 + 1e-12: return v
        if abs(c - tot / 2) <= 1e-12: return (v + p[j + 1][0]) / 2 if j + 1 < len(p) else v
    return None


def tri(h, n_read, n_total, thr):
    if h >= thr: return True
    if h + (n_total - n_read) < thr: return False
    return None


LIN = [("qwen3_4b_base", f"qwen3_4b_Base_own_a3_s{s}_r3") for s in range(5)] + [("base_Qwen3-8B-Base", f"qwen3_8b_Base_own_a3_s{s}_r3") for s in range(3)] + \
      [("base_Phi-4-mini-instruct", f"phi4mini_own_a4_s{s}_r3") for s in range(2)]
TWB = {"qwen3_4b_base": f"{R}/dirA/results/qwen3_4b_base_shard*of8.jsonl", "base_Qwen3-8B-Base": f"{R}/dirA/results/qwen3_8b_base_shard*.jsonl",
       "base_Phi-4-mini-instruct": f"{R}/dirA/results/phi4mini_shard*of8.jsonl"}
res = {"D1": [], "D2": [], "D3": [], "D4": [], "descriptive": {}}
pool, pool_all, pool4 = [], [], []
bc = {}
for base_tag, run in LIN:
    if base_tag not in bc: bc[base_tag] = load_umwp(base_tag)
    b, r = bc[base_tag], load_umwp(run)
    if b is None or r is None:
        for k in ("D1", "D2", "D3"): res[k].append({"run": run, "read": False})
        print(f"{run}: UMWP not read"); continue
    rU, rA = cells(b, r)
    ratio, n, bins = std_ratio(rU, rA); powered = n >= 30
    ratio5, n5, _ = std_ratio(rU, rA, min_nA=5)
    res["D1"].append({"run": run, "read": True, "ratio": ratio, "n_matched_U": n, "powered": powered, "holds": bool(powered and ratio is not None and ratio >= 1.25)})
    res["descriptive"][run] = {"bins": {k: {"nU": len(rU[k]), "nA": len(rA[k]), "rU": (sum(x for _, x in rU[k]) / len(rU[k])) if rU[k] else None,
                                            "rA": (sum(x for _, x in rA[k]) / len(rA[k])) if rA[k] else None} for k in BINS},
                               "ratio_bins_nA_ge5": ratio5, "n_answerable_without_ref": sum(1 for it in b.values() if not it["U"] and it["gold"] is None)}
    for k in bins:
        for i, x in rU[k]:
            pool_all.append((run, k, "U", i, x)); powered and pool.append((run, k, "U", i, x))
        for i, x in rA[k]:
            pool_all.append((run, k, "A", i, x)); powered and pool.append((run, k, "A", i, x))
    hU, hA = cells(b, r, subset_A=lambda it: it["gold"] is not None and it["gold"] <= 3 / 8 + 1e-9)
    ratio2, n2, _ = std_ratio(hU, hA)
    res["D2"].append({"run": run, "read": True, "ratio": ratio2, "n_matched_U": n2, "holds": bool(n2 >= 30 and ratio2 is not None and ratio2 >= 1.25)})
    sU, sA = defaultdict(list), defaultdict(list); zero = {"U": 0, "A": 0}
    for i, it in b.items():
        k = round(8 * it["m"])
        if k not in BINS or i not in r: continue
        (sU if it["U"] else sA)[k].append(sharp(it["m"], r[i]["m"]))
        if r[i]["m"] == 0: zero["U" if it["U"] else "A"] += 1
    bins3 = [k for k in BINS if sU[k] and sA[k]]; nU3 = sum(len(sU[k]) for k in bins3)
    mU = median([x for k in bins3 for x in sU[k]])
    mA = wmedian([(x, len(sU[k]) / (nU3 * len(sA[k]))) for k in bins3 for x in sA[k]]) if nU3 else None
    res["D3"].append({"run": run, "read": True, "median_s_U": mU, "median_s_A_std": mA, "tie": mU == mA, "n_U": nU3, "m_r3_zero": zero,
                      "holds": bool(nU3 >= 30 and mU is not None and mA is not None and mU < mA)})
    print(f"{run}: D1 {ratio} (n {n}); D2 {ratio2} (n {n2}); D3 {mU} vs {mA}")
bs_strat = boot_strat(pool); bs_item = boot_items(pool)
c1, drop1 = ci(bs_strat); c1i, drop1i = ci(bs_item)
res["D1_pooled"] = {"ratio": pooled(pool), "ci95_registered": c1, "ci95_item_clustered_sensitivity": c1i, "dropped_boots": [drop1, drop1i],
                    "ratio_all_lineages_descriptive": pooled(pool_all)}
print("D1 pooled", res["D1_pooled"])
rngt = random.Random(20260924); tb = {}
for base_tag, run in LIN:
    if base_tag not in tb: tb[base_tag] = load_twins(pattern=TWB[base_tag], rng=rngt)
    b, r = tb[base_tag], load_twins(run=run, rng=rngt)
    if b is None or r is None:
        res["D4"].append({"run": run, "read": False}); print(f"{run}: twins not read"); continue
    rU, rA = cells(b, r, twins=True)
    ratio, n, bins = std_ratio(rU, rA); powered = n >= 30
    res["D4"].append({"run": run, "read": True, "ratio": ratio, "n_matched_U": n, "holds": bool(powered and ratio is not None and ratio >= 1.25)})
    if powered:
        for k in bins:
            for i, x in rU[k]: pool4.append((run, k, "U", i, x))
            for i, x in rA[k]: pool4.append((run, k, "A", i, x))
    print(f"{run}: D4 {ratio} (n {n})")
c4, drop4 = ci(boot_strat(pool4))
res["D4_pooled"] = {"ratio": pooled(pool4), "ci95_registered": c4, "dropped_boots": drop4}
def summ(lst):
    rd = [x for x in lst if x.get("read")]; return sum(x["holds"] for x in rd), len(rd), len(lst)
h1, r1, t1 = summ(res["D1"]); h2, r2, t2 = summ(res["D2"]); h3, r3n, t3 = summ(res["D3"]); h4, r4, t4 = summ(res["D4"])
v1 = tri(h1, r1, t1, 8); v1 = (v1 and c1 is not None and c1[0] > 1.0) if v1 else v1
v4 = tri(h4, r4, t4, 7); v4 = (v4 and c4 is not None and c4[0] > 1.0) if v4 else v4
res["verdicts"] = {"D1": v1, "D2": tri(h2, r2, t2, 7), "D3": tri(h3, r3n, t3, 8), "D4": v4,
                   "counts": {"D1": [h1, r1, t1], "D2": [h2, r2, t2], "D3": [h3, r3n, t3], "D3_ties": sum(1 for x in res["D3"] if x.get("tie")), "D4": [h4, r4, t4]}}
print("verdicts:", res["verdicts"])
json.dump(res, open(f"{R}/dirA5/results/a6_matched.json", "w"), indent=1, default=lambda o: "inf" if o == math.inf else str(o))
