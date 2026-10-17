"""check_hardtraj (independent re-derivation of hardtraj; read-only over the UMWP plain dumps and A8 fresh texts).
k = count of the most frequent answer that is not None and not "unknown" (out of G = 8); band = 2 <= k <= 6.
hard = answerable, ref not None, number of answers equal to ref at the family base <= 3. other = remaining answerable. U = unanswerable.
acc_plur = mean dump field correct; acc_samp = mean share of answers equal to ref; CA = dump mode not None and p_mode >= 0.5.
solved = k >= 7 and dump correct == 1 (also checked: >= 7 answers equal to ref). python3 check_hardtraj.py"""
import json, os
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; UM = R + "/dirA4/results/umwp"; A8 = R + "/dirA5/results/a8"
OUT = R + "/paper_arr/analyses/check_hardtraj.json"

def readrows(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            d[int(r["id"])] = r
    return d

def kof(ans):
    vals = [a for a in ans if a is not None and a != "unknown"]
    return max(Counter(vals).values()) if vals else 0

def goldk(r):
    return sum(1 for a in r["answers"] if a is not None and a == r["ref"])

def avg(x):
    x = list(x); return sum(x) / len(x) if x else None

def integrity(rows):
    out = {"n": len(rows), "nU": sum(1 for r in rows.values() if not r["answerable"]), "nA": sum(1 for r in rows.values() if r["answerable"])}
    out["len_not_8"] = sum(1 for r in rows.values() if len(r["answers"]) != 8)
    out["str_answers"] = dict(Counter(a for r in rows.values() for a in r["answers"] if isinstance(a, str)))
    bad_pm = 0; bad_corr = 0
    for r in rows.values():
        c = Counter(r["answers"]); mx = max(c.values())
        if abs(mx / len(r["answers"]) - r["p_mode"]) > 1e-9: bad_pm += 1
        if r["mode"] not in [a for a, n in c.items() if n == mx]: bad_pm += 1
        corr = int(r["mode"] is not None and r.get("ref") is not None and r["mode"] == r["ref"])
        if corr != r["correct"]: bad_corr += 1
    out["p_mode_or_mode_mismatch"] = bad_pm; out["correct_mismatch"] = bad_corr
    return out

def gstats(rows, ids, U):
    rs = [rows[i] for i in ids if i in rows]
    ks = [kof(r["answers"]) for r in rs]
    o = {"n": len(rs), "missing": len(ids) - len(rs), "mean_m": avg(k / 8 for k in ks), "in_band": avg(2 <= k <= 6 for k in ks),
         "mean_p_mode": avg(r["p_mode"] for r in rs)}
    if U:
        o["CA"] = avg(r["mode"] is not None and r["p_mode"] >= 0.5 for r in rs)
    else:
        o["acc_plur"] = avg(r["correct"] for r in rs)
        o["acc_samp"] = avg(goldk(r) / 8 for r in rs if r.get("ref") is not None)
    return o

def dec(rows, ids, U):
    c = Counter(); n = 0; solved_alt = 0
    for i in ids:
        if i not in rows: continue
        r = rows[i]; k = kof(r["answers"]); n += 1
        if 2 <= k <= 6: c["in"] += 1
        elif k <= 1: c["below"] += 1
        elif (not U) and r["correct"] == 1: c["solved"] += 1
        else: c["above_wrong"] += 1
        if not U and r.get("ref") is not None and goldk(r) >= 7: solved_alt += 1
    return {"n": n, "counts": {k: c[k] for k in ("in", "solved", "above_wrong", "below")},
            "share": {k: (c[k] / n if n else None) for k in ("in", "solved", "above_wrong", "below")}, "solved_alt_ge7_eq_ref": solved_alt}

FAM = {"Qwen3-4B": ("qwen3_4b_base", {"r3": ["qwen3_4b_Base_own_a3_s%d_r3" % s for s in range(5)], "r5": ["qwen3_4b_Base_own_a3_s%d_r5" % s for s in range(3)]}),
       "Qwen3-8B": ("base_Qwen3-8B-Base", {"r3": ["qwen3_8b_Base_own_a3_s%d_r3" % s for s in range(3)]}),
       "Phi-4-mini": ("base_Phi-4-mini-instruct", {"r3": ["phi4mini_own_a4_s%d_r3" % s for s in range(2)]})}
res = {}
for fam, (bt, rounds) in FAM.items():
    base = readrows(UM + "/plain_" + bt + ".jsonl")
    F = {"integrity": {bt: integrity(base)}}
    hard = [i for i, r in base.items() if r["answerable"] and r.get("ref") is not None and goldk(r) <= 3]
    hs = set(hard)
    other = [i for i, r in base.items() if r["answerable"] and i not in hs]
    U = [i for i, r in base.items() if not r["answerable"]]
    bk = {i: kof(r["answers"]) for i, r in base.items()}
    G = {"hard": hard, "other": other, "U": U}
    GB = {g: [i for i in v if 2 <= bk[i] <= 6] for g, v in G.items()}
    F["sizes"] = {g: [len(G[g]), len(GB[g])] for g in G}
    F["ref_none_answerable"] = sum(1 for r in base.values() if r["answerable"] and r.get("ref") is None)
    F["hard_base_k_all"] = dict(sorted(Counter(bk[i] for i in hard).items()))
    F["U_base_k_band"] = dict(sorted(Counter(bk[i] for i in GB["U"]).items()))
    F["base"] = {"all": {g: gstats(base, G[g], g == "U") for g in G}, "band": {g: gstats(base, GB[g], g == "U") for g in G}}
    F["rounds"] = {}
    for rn, tags in rounds.items():
        per = {}
        for t in tags:
            p = UM + "/plain_" + t + ".jsonl"
            if not os.path.exists(p): per[t] = None; continue
            rows = readrows(p); F["integrity"][t] = integrity(rows)
            per[t] = {"all": {g: gstats(rows, G[g], g == "U") for g in G}, "dec": {g: dec(rows, GB[g], g == "U") for g in G}}
            hb = Counter(bk[i] for i in GB["hard"]); nh = sum(hb.values()); uw = 0.0; hw = 0.0
            for k, nk in hb.items():
                uk = [i for i in GB["U"] if bk[i] == k]; hk = [i for i in GB["hard"] if bk[i] == k]
                uw += nk / nh * avg(2 <= kof(rows[i]["answers"]) <= 6 for i in uk)
                hw += nk / nh * avg(2 <= kof(rows[i]["answers"]) <= 6 for i in hk)
            per[t]["U_ret_hardbin_weighted"] = uw; per[t]["hard_ret"] = hw
        ok = [t for t in tags if per[t] is not None]
        def agg(fn):
            v = [fn(per[t]) for t in ok]; return {"mean": avg(v), "min": min(v), "max": max(v), "per_seed": v}
        S = {"seeds": ok, "all": {}, "dec": {}}
        for g in G:
            keys = list(per[ok[0]]["all"][g].keys())
            S["all"][g] = {k: agg(lambda x, g=g, k=k: x["all"][g][k]) for k in keys if k not in ("n", "missing")}
            S["dec"][g] = {k: agg(lambda x, g=g, k=k: x["dec"][g]["share"][k]) for k in ("in", "solved", "above_wrong", "below")}
            S["dec"][g]["pooled_counts"] = {k: sum(per[t]["dec"][g]["counts"][k] for t in ok) for k in ("in", "solved", "above_wrong", "below")}
            S["dec"][g]["pooled_solved_alt"] = sum(per[t]["dec"][g]["solved_alt_ge7_eq_ref"] for t in ok)
            S["dec"][g]["solved_counts_per_seed"] = [per[t]["dec"][g]["counts"]["solved"] for t in ok]
        S["U_ret_hardbin_weighted"] = agg(lambda x: x["U_ret_hardbin_weighted"]); S["hard_ret"] = agg(lambda x: x["hard_ret"])
        if rn == "r3" and fam == "Qwen3-4B":
            s3 = ok[:3]
            S["s0_s2_only"] = {"hard_in_all": avg(per[t]["all"]["hard"]["in_band"] for t in s3), "U_in_all": avg(per[t]["all"]["U"]["in_band"] for t in s3),
                               "CA": avg(per[t]["all"]["U"]["CA"] for t in s3)}
        F["rounds"][rn] = S
    if fam == "Qwen3-4B":
        it = json.load(open(A8 + "/items.json")); hid = [int(i) for i in it["hard_A"]]; uid = [int(i) for i in it["U"]]
        fc = {"hard_ids_eq_band_hard": sorted(hid) == sorted(GB["hard"]), "U_subset": set(uid) <= set(GB["U"]),
              "U_bins": dict(sorted(Counter(bk[i] for i in uid).items())), "hard_bins": dict(sorted(Counter(bk[i] for i in hid).items())),
              "orig_base_hard": dict(dec(base, hid, False), **{k: gstats(base, hid, False)[k] for k in ("acc_plur", "acc_samp")}),
              "orig_base_U": dec(base, uid, True), "fresh": {}, "orig_same_items": {}}
        for ck in ["base"] + ["own_s%d_r%d" % (s, r) for r in (3, 5) for s in range(3)]:
            p = A8 + "/texts/a8_" + ck + ".jsonl"
            if not os.path.exists(p): fc["fresh"][ck] = None; continue
            rows = readrows(p)
            gh = gstats(rows, hid, False); gu = gstats(rows, uid, True)
            fc["fresh"][ck] = {"n_rows": len(rows), "len_not_8": sum(1 for r in rows.values() if len(r["answers"]) != 8),
                               "ref_match_base": sum(1 for i in hid if i in rows and rows[i].get("ref") == base[i].get("ref")),
                               "hard": dict(dec(rows, hid, False), acc_plur=gh["acc_plur"], acc_samp=gh["acc_samp"]),
                               "U": dict(dec(rows, uid, True), CA=gu["CA"])}
            if ck != "base":
                parts = ck.split("_")
                orows = readrows(UM + "/plain_qwen3_4b_Base_own_a3_" + parts[1] + "_" + parts[2] + ".jsonl")
                fc["orig_same_items"][ck] = {"hard": dec(orows, hid, False)["counts"], "U": dec(orows, uid, True)["counts"]}
        F["fresh"] = fc
    res[fam] = F
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
