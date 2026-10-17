"""A8 read (registered 23:16 24 Sep, dirA5/docs/A8_registration_2026-09-24.md; clarification A8a). Inputs: dirA5/results/a8_key/
units_main_key.jsonl and units_meta.json, dirA5/results/a8/labels_main.json (blind Opus labels: key, clusters [{samples,
reading}], gap_flag [8 bools], disagreement_source), labels_second.json (second pass on the registered 25% base subsample),
optionally labels_r3.json + a8_key/units_r3_key.jsonl (descriptive). A label is valid if: key is a string; disagreement_source in
NONE/READINGS/ERRORS/MIXED; gap_flag is 8 booleans; clusters is a non-empty list of dicts, each with a non-empty 'reading' string
and a non-empty list of ints, partitioning 0..7 exactly (a label whose indices are exactly {1..8} is shifted to 0-based). Invalid,
missing or conflicting-duplicate labels count as missing; a cell (model x class) with > 5% missing makes the predictions using it
None. U = UMWP unanswerable, A = hard answerable (base gold <= 3/8).
M-P1: at base, share(READINGS or MIXED | U) - share(... | A) >= 0.20 and the item bootstrap 95% CI (4,000 replicates, seed
      20260925, units resampled within class) excludes 0.
M-P2: gap_flag rate over the samples of U units: base - r5(seed s) >= 0.05 in >= 2 of 3 seeds.
M-P3: mean number of reading clusters (excluding the NO_READING cluster; A8a.3) on U units: r5(seed s) < base in >= 2 of 3.
2-of-3 verdicts are three-state (True if >= 2 hold, False if >= 2 fail, else None).
Descriptive: A units; per UMWP category; cluster count including every cluster; M-P2 on untruncated samples only; second-pass
agreement (kappa on disagreement_source, per-sample kappa on gap_flag, exact agreement on the reading-cluster count); r3 if
labelled; the reproducibility and cut/truncation statistics. python3 a8_read.py"""
import json, os, random
from collections import Counter, defaultdict
D = "/home/tione/notebook/rsi_rl/dirA5/results"
KEY = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8_key/units_main_key.jsonl")}
META = json.load(open(f"{D}/a8_key/units_meta.json"))
SRC = {"NONE", "READINGS", "ERRORS", "MIXED"}
def norm(x):
    try:
        if not isinstance(x, dict) or not isinstance(x.get("key"), str) or x.get("disagreement_source") not in SRC: return None
        g = x["gap_flag"]
        if not (isinstance(g, list) and len(g) == 8 and all(isinstance(b, bool) for b in g)): return None
        cl = x["clusters"]
        if not (isinstance(cl, list) and cl and all(isinstance(c, dict) and isinstance(c.get("reading"), str) and c["reading"].strip()
                and isinstance(c.get("samples"), list) and c["samples"] and all(type(i) is int for i in c["samples"]) for c in cl)): return None
        idx = sorted(i for c in cl for i in c["samples"])
        if idx == list(range(1, 9)): cl = [{"samples": [i - 1 for i in c["samples"]], "reading": c["reading"]} for c in cl]; idx = list(range(8))
        if idx != list(range(8)): return None
        return {"key": x["key"], "clusters": cl, "gap_flag": g, "disagreement_source": x["disagreement_source"]}
    except Exception: return None
def load(f, keys):
    raw = json.load(open(f)); seen, bad, conflict, extra = {}, 0, set(), 0
    for x in raw:
        y = norm(x)
        if y is None: bad += 1; continue
        if y["key"] not in keys: extra += 1; continue
        k = y["key"]
        if k in seen and json.dumps(seen[k], sort_keys=True) != json.dumps(y, sort_keys=True): conflict.add(k)
        seen.setdefault(k, y)
    return {k: v for k, v in seen.items() if k not in conflict}, {"invalid": bad, "conflicting": len(conflict), "extra_keys": extra}
isread = lambda c: not c["reading"].strip().upper().startswith("NO_READING")
rd = lambda lab: lab["disagreement_source"] in ("READINGS", "MIXED")
gap = lambda xs, m=None: (lambda v: sum(v) / len(v) if v else None)([b for u, l in xs for j, b in enumerate(l["gap_flag"]) if m is None or not u[m][j]])
ncl = lambda xs: sum(sum(map(isread, l["clusters"])) for _, l in xs) / len(xs) if xs else None
ncl_all = lambda xs: sum(len(l["clusters"]) for _, l in xs) / len(xs) if xs else None
def cellname(u): return "base" if u["model"] == "base" else f"r{u['round']}_s{u['seed']}"
L, qc = load(f"{D}/a8/labels_main.json", set(KEY))
out = {"n_units": len(KEY), "n_labelled_valid": sum(k in L for k in KEY), "label_qc": qc, "repro": META.get("repro"), "cut_stats": META.get("cut_stats"), "cells": {}}
cells = defaultdict(list); miss = Counter(); tot = Counter()
for k, u in KEY.items():
    assert u["model"] == "base" or u["round"] == 5
    c = (cellname(u), "U" if not u["answerable"] else "A"); tot[c] += 1
    if k in L: cells[c].append((u, L[k]))
    else: miss[c] += 1
ok = {c: miss[c] <= 0.05 * tot[c] for c in tot}
for c in sorted(tot):
    xs = cells[c]
    out["cells"][f"{c[0]}|{c[1]}"] = {"n": tot[c], "valid": len(xs), "missing": miss[c], "usable": ok[c],
        "share_readings_or_mixed": (sum(rd(l) for _, l in xs) / len(xs)) if xs else None, "gap_flag_rate": gap(xs),
        "gap_flag_rate_untruncated": gap(xs, "trunc"), "mean_reading_clusters": ncl(xs), "mean_clusters_all": ncl_all(xs),
        "sources": dict(Counter(l["disagreement_source"] for _, l in xs))}
bU, bA = cells[("base", "U")], cells[("base", "A")]
mp1 = None; diff = ci = None
if ok.get(("base", "U")) and ok.get(("base", "A")) and bU and bA:
    diff = sum(rd(l) for _, l in bU) / len(bU) - sum(rd(l) for _, l in bA) / len(bA)
    rng = random.Random(20260925); bs = []
    for _ in range(4000):
        su = [bU[rng.randrange(len(bU))] for _ in bU]; sa = [bA[rng.randrange(len(bA))] for _ in bA]
        bs.append(sum(rd(l) for _, l in su) / len(su) - sum(rd(l) for _, l in sa) / len(sa))
    bs.sort(); ci = [bs[100], bs[3899]]; mp1 = diff >= 0.20 - 1e-9 and ci[0] > 0
def tri(l):
    t = sum(x is True for x in l); f = sum(x is False for x in l)
    return True if t >= 2 else (False if f >= 2 else None)
p2, p3, p2u, p2A, p3A = [], [], [], [], []
for s in range(3):
    c, cA = (f"r5_s{s}", "U"), (f"r5_s{s}", "A")
    use = ok.get(c) and ok.get(("base", "U")) and cells[c] and bU
    p2.append((gap(bU) - gap(cells[c]) >= 0.05 - 1e-9) if use else None)
    p3.append((ncl(cells[c]) < ncl(bU)) if use else None)
    p2u.append((gap(bU, "trunc") - gap(cells[c], "trunc") >= 0.05 - 1e-9) if use and gap(cells[c], "trunc") is not None and gap(bU, "trunc") is not None else None)
    useA = cells[cA] and bA
    p2A.append((gap(bA) - gap(cells[cA]) >= 0.05 - 1e-9) if useA else None); p3A.append((ncl(cells[cA]) < ncl(bA)) if useA else None)
out["M-P1"] = {"diff": diff, "ci95": ci, "holds": mp1}
out["M-P2"] = {"per_seed": p2, "holds": tri(p2), "sensitivity_untruncated": {"per_seed": p2u, "holds": tri(p2u)}}
out["M-P3"] = {"per_seed": p3, "holds": tri(p3)}
out["descriptive_hard_A"] = {"M-P2_analogue": {"per_seed": p2A, "holds": tri(p2A)}, "M-P3_analogue": {"per_seed": p3A, "holds": tri(p3A)}}
cat = defaultdict(list)
for k, u in KEY.items():
    if k in L and not u["answerable"]: cat[(str(u["category"]), "base" if u["model"] == "base" else "r5")].append((u, L[k]))
out["per_category"] = {f"{a}|{b}": {"n": len(xs), "share_readings_or_mixed": sum(rd(l) for _, l in xs) / len(xs), "gap_flag_rate": gap(xs),
                                    "mean_reading_clusters": ncl(xs)} for (a, b), xs in sorted(cat.items())}
def kappa(pairs, labels):
    n = len(pairs)
    if n == 0: return None
    po = sum(a == b for a, b in pairs) / n; ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in labels) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else None
try:
    SK = set(META["second_pass_keys"]); L2, qc2 = load(f"{D}/a8/labels_second.json", SK)
    both = [k for k in sorted(SK) if k in L2 and k in L]
    out["second_pass"] = {"n_registered": len(SK), "n_both_valid": len(both), "qc": qc2,
        "agreement_source": sum(L[k]["disagreement_source"] == L2[k]["disagreement_source"] for k in both) / len(both) if both else None,
        "kappa_source": kappa([(L[k]["disagreement_source"], L2[k]["disagreement_source"]) for k in both], SRC),
        "kappa_gap_flag_per_sample": kappa([(a, b) for k in both for a, b in zip(L[k]["gap_flag"], L2[k]["gap_flag"])], {True, False}),
        "exact_reading_cluster_count_agreement": sum(sum(map(isread, L[k]["clusters"])) == sum(map(isread, L2[k]["clusters"])) for k in both) / len(both) if both else None}
except Exception as e: out["second_pass"] = f"not available: {type(e).__name__}: {e}"
try:
    K3 = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8_key/units_r3_key.jsonl")}
    L3, qc3 = load(f"{D}/a8/labels_r3.json", set(K3)); c3 = defaultdict(list)
    for k, u in K3.items():
        if k in L3: c3[(cellname(u), "U" if not u["answerable"] else "A")].append((u, L3[k]))
    out["r3_descriptive"] = {"qc": qc3, "cells": {f"{a}|{b}": {"n": len(xs), "share_readings_or_mixed": sum(rd(l) for _, l in xs) / len(xs),
                             "gap_flag_rate": gap(xs), "mean_reading_clusters": ncl(xs)} for (a, b), xs in sorted(c3.items())}}
except Exception as e: out["r3_descriptive"] = f"not available: {type(e).__name__}"
json.dump(out, open(f"{D}/a8/a8_read.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("n_units", "n_labelled_valid", "label_qc", "M-P1", "M-P2", "M-P3", "descriptive_hard_A")}, indent=1))
for c, v in out["cells"].items(): print(c, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
print("second pass:", out["second_pass"])
