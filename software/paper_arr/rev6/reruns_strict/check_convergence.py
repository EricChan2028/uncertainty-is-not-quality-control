# rev6 rerun copy of paper_arr/analyses/check_convergence.py (UMWP dir -> dumps_strict, 1 occurrence(s); output -> reruns_strict)
"""Independent re-derivation of paper_arr/analyses/convergence.json (verifier; read-only on all inputs).
Own code, not importing convergence.py. Adds: tie diagnostics, base count of the shared answer, a mode-level null
(3 fresh 8-sample runs of the base empirical distribution, conditional on all three confident), paired SE for coupling.
Output: paper_arr/analyses/check_convergence.json
"""
import json, math
from collections import Counter
from itertools import combinations

R = "/home/tione/notebook/rsi_rl"
UM = R + "/paper_arr/rev6/dumps_strict/plain_%s.jsonl"

def load_dump(tag):
    out = {}
    for line in open(UM % tag):
        x = json.loads(line)
        key = ("A" if x["answerable"] else "U", x["id"])
        assert key not in out
        out[key] = x
    return out

base = load_dump("qwen3_4b_base")
runs = {rnd: [load_dump("qwen3_4b_Base_own_a3_s%d_%s" % (s, rnd)) for s in (0, 1, 2)] for rnd in ("r5", "r3")}
U = sorted(k for k in base if k[0] == "U"); A = sorted(k for k in base if k[0] == "A")
assert len(U) == 600 and len(A) == 600

def confident(x):
    return (x["mode"] is not None) and (x["p_mode"] >= 0.5)

def wil(k, n):
    z = 1.959964; p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]

def fr(k, n):
    return {"k": k, "n": n, "share": round(k / n, 4) if n else None, "wilson95": wil(k, n) if n else None}

def avg(v):
    v = list(v); return round(sum(v) / len(v), 4) if v else None

def recheck_fields(d):
    bad = 0
    for x in d.values():
        c = Counter(x["answers"]); top = max(c.values())
        if abs(x["p_mode"] - top / len(x["answers"])) > 1e-9 or c[x["mode"]] != top or len(x["answers"]) != 8:
            bad += 1
    return bad

def numcounts(ans):
    return Counter(a for a in ans if a is not None)

def q_stats(ans):
    c = numcounts(ans); n = sum(c.values())
    q3 = sum((v / n) ** 3 for v in c.values())
    q3a = sum((v / 8) ** 3 for v in c.values())
    q3u = sum(v * (v - 1) * (v - 2) for v in c.values()) / (n * (n - 1) * (n - 2)) if n >= 3 else None
    return q3, q3a, q3u

def mode_null(ans):
    """P(confident numeric mode = a) for one fresh 8-sample run from the empirical distribution of the 8 answers
    (None kept as a category). Exact: count>=5 wins; count 4 wins unless another category also has 4 (then 1/2)."""
    c = Counter(ans); p = {a: v / 8 for a, v in c.items()}
    m = {}
    for a, pa in p.items():
        if a is None:
            continue
        b5 = sum(math.comb(8, k) * pa ** k * (1 - pa) ** (8 - k) for k in range(5, 9))
        b4 = math.comb(8, 4) * pa ** 4 * (1 - pa) ** 4
        tie = sum((pb / (1 - pa)) ** 4 for b, pb in p.items() if b != a) if pa < 1 else 0.0
        m[a] = b5 + b4 * (1 - 0.5 * tie)
    tot = sum(m.values())
    return (sum(v ** 3 for v in m.values()) / tot ** 3) if tot > 0 else None, sum(v ** 3 for v in m.values()), tot

def top_set(x):
    c = numcounts(x["answers"])
    if not c: return set()
    t = max(c.values()); return {a for a, v in c.items() if v == t}

def analyse(keys, ss, pair=None):
    S = [k for k in keys if all(confident(s[k]) for s in ss)]
    res = {"n": len(keys), "S": len(S)}
    idn = [k for k in S if ss[0][k]["mode"] == ss[1][k]["mode"] == ss[2][k]["mode"]]
    res["identical"] = fr(len(idn), len(S))
    res["pairwise"] = avg(sum(a[k]["mode"] == b[k]["mode"] for a, b in combinations(ss, 2)) / 3 for k in S)
    Q = [q_stats(base[k]["answers"]) for k in S]
    res["q3_num"] = avg(q[0] for q in Q); res["q3_all8"] = avg(q[1] for q in Q); res["q3_unb"] = avg(q[2] for q in Q if q[2] is not None)
    res["q3_unb_n"] = sum(q[2] is not None for q in Q)
    res["q2_num"] = avg(sum((v / sum(numcounts(base[k]["answers"]).values())) ** 2 for v in numcounts(base[k]["answers"]).values()) for k in S)
    MN = [mode_null(base[k]["answers"]) for k in S]
    res["mode_null_conditional_mean"] = avg(m[0] for m in MN if m[0] is not None)
    res["mode_null_P_all3_conf_mean"] = avg(m[2] ** 3 for m in MN)
    res["base_conf_share_S"] = round(sum(confident(base[k]) for k in S) / len(S), 4)
    res["base_mean_p_mode_S"] = avg(base[k]["p_mode"] for k in S)
    res["S_items_with_a_seed_4-4_numeric_tie"] = sum(any(len(top_set(s[k])) > 1 for s in ss) for k in S)
    res["identical_tie_aware"] = fr(sum(bool(top_set(ss[0][k]) & top_set(ss[1][k]) & top_set(ss[2][k])) for k in S), len(S))
    cm = {k: ss[0][k]["mode"] for k in idn}
    res["eq_base_mode"] = fr(sum(base[k]["mode"] == cm[k] for k in idn), len(idn))
    cnt = {k: numcounts(base[k]["answers"]).get(cm[k], 0) for k in idn}
    res["base_count_of_shared_answer_hist"] = dict(sorted(Counter(cnt.values()).items()))
    tt = [k for k in idn if cnt[k] > 0 and cnt[k] == max(numcounts(base[k]["answers"]).values())]
    res["tied_top"] = fr(len(tt), len(idn))
    res["tied_top_by_base_count"] = dict(sorted(Counter(cnt[k] for k in tt).items()))
    res["tied_top_with_base_count_ge2"] = fr(sum(cnt[k] >= 2 for k in tt), len(idn))
    res["eq_base_mode_by_base_count"] = dict(sorted(Counter(cnt[k] for k in idn if base[k]["mode"] == cm[k]).items()))
    res["eq_base_mode_and_count_ge2"] = fr(sum(base[k]["mode"] == cm[k] and cnt[k] >= 2 for k in idn), len(idn))
    res["in_base_8"] = fr(sum(cnt[k] > 0 for k in idn), len(idn))
    res["mean_q0_shared"] = avg(cnt[k] / 8 for k in idn)
    for lab, cond in (("base_conf", True), ("base_notconf", False)):
        sub = [k for k in S if confident(base[k]) == cond]
        si = [k for k in sub if k in cm]
        res[lab] = {"identical": fr(len(si), len(sub)), "q3_num": avg(q_stats(base[k]["answers"])[0] for k in sub),
                    "q3_unb": avg(q_stats(base[k]["answers"])[2] for k in sub if q_stats(base[k]["answers"])[2] is not None),
                    "mode_null_conditional": avg(mode_null(base[k]["answers"])[0] for k in sub),
                    "eq_base_mode": fr(sum(base[k]["mode"] == cm[k] for k in si), len(si)),
                    "shared_base_count_hist": dict(sorted(Counter(cnt[k] for k in si).items())),
                    "shared_count_ge2": fr(sum(cnt[k] >= 2 for k in si), len(si)),
                    "base_max_numeric_count_hist": dict(sorted(Counter(max(numcounts(base[k]["answers"]).values()) for k in sub).items()))}
    if pair is not None:
        res["eq_original_ref"] = fr(sum(pair[k] == cm[k] for k in idn if pair[k] is not None), sum(pair[k] is not None for k in idn))
        res["base_mode_eq_original_ref_S"] = fr(sum(base[k]["mode"] == pair[k] for k in S), len(S))
    return res, idn

def coupling(keys, X, Y):
    al = []; mi = []; diffs = []
    for k in keys:
        x = X[k]["answers"]; y = Y[k]["answers"]
        a_ = [int(x[i] == y[i]) for i in range(8) if x[i] is not None and y[i] is not None]
        m_ = [int(x[i] == y[j]) for i in range(8) for j in range(8) if i != j and x[i] is not None and y[j] is not None]
        al += a_; mi += m_
        if a_ and m_:
            diffs.append(sum(a_) / len(a_) - sum(m_) / len(m_))
    md = sum(diffs) / len(diffs); sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1))
    return {"aligned": round(sum(al) / len(al), 4), "n_al": len(al), "misaligned": round(sum(mi) / len(mi), 4), "n_mis": len(mi),
            "per_q_diff": round(md, 4), "se": round(sd / math.sqrt(len(diffs)), 4), "z": round(md / (sd / math.sqrt(len(diffs))), 2), "n_q": len(diffs)}

SD = {}
for line in open(R + "/data_ext/UMWP/data/StandardDataset.jsonl"):
    x = json.loads(line); SD[x["id"]] = x
u2a = {k: ("A", SD[k[1]]["relevant_ids"][0]) for k in U}
assert len(set(u2a.values())) == 600 and all(v in base for v in u2a.values())
assert all(SD[k[1]]["category"] == base[k]["category"] for k in U)
refpair = {k: base[u2a[k]]["ref"] for k in U}
cat_of = {k: base[k]["category"] for k in U}; cat_of.update({u2a[k]: base[k]["category"] for k in U})
CATN = json.load(open(R + "/dirA5/results/umwp_category_map.json"))

out = {"field_recheck_bad_rows": {"base": recheck_fields(base)},
       "sanity": {"base_CA": round(sum(confident(base[k]) for k in U) / 600, 4), "base_ACC_A": round(sum(base[k]["correct"] for k in A) / 600, 4),
                  "base_ACC_A_from_ref": round(sum(base[k]["mode"] is not None and base[k]["mode"] == base[k]["ref"] for k in A) / 600, 4),
                  "n_A_ref_None": sum(base[k]["ref"] is None for k in A)},
       "rounds": {}}
for rnd, ss in runs.items():
    for i, s in enumerate(ss):
        assert set(s) == set(base)
        out["field_recheck_bad_rows"]["%s_s%d" % (rnd, i)] = recheck_fields(s)
        assert all(s[k]["ref"] == base[k]["ref"] for k in base)
    e = {"CA": [round(sum(confident(s[k]) for k in U) / 600, 4) for s in ss],
         "ACC_A": [round(sum(s[k]["correct"] for k in A) / 600, 4) for s in ss]}
    e["U"], _ = analyse(U, ss, refpair)
    e["A"], idA = analyse(A, ss)
    e["A"]["identical_correct"] = fr(sum(ss[0][k]["mode"] == base[k]["ref"] for k in idA), len(idA))
    e["by_cat"] = {}
    for c in (1, 2, 3, 4, 5):
        ku = [k for k in U if cat_of[k] == c]; ka = [k for k in A if cat_of[k] == c]
        ru, _ = analyse(ku, ss, refpair); ra, _ = analyse(ka, ss)
        e["by_cat"]["%d %s" % (c, CATN[str(c)])] = {"U": {x: ru[x] for x in ("n", "S", "identical", "q3_num", "q3_unb", "mode_null_conditional_mean", "eq_base_mode", "tied_top", "eq_original_ref")},
                                                  "A": {x: ra[x] for x in ("n", "S", "identical", "q3_num")}}
    e["coupling"] = {}
    for lab, keys in (("U", U), ("A", A)):
        for i, j in combinations(range(3), 2):
            e["coupling"]["%s s%d-s%d" % (lab, i, j)] = coupling(keys, ss[i], ss[j])
    for i in range(3):
        e["coupling"]["U base-s%d" % i] = coupling(U, base, ss[i])
    out["rounds"][rnd] = e

json.dump(out, open(R + "/paper_arr/rev6/reruns_strict/check_convergence.json", "w"), indent=1)
print(json.dumps(out, indent=1))
