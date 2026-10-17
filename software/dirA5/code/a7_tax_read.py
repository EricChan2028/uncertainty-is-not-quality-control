"""A7 item 1 (registered 15:10, descriptive; no thresholds): type mix of audited ill-posed problems per round and its change,
and type agreement between two independent Opus typing passes on the registered 25% subsample (Cohen's kappa on the
primary type; also raw agreement and a 4-class collapse merging AMBIGUOUS_REFERENCE + MULTIPLE_READINGS). Labels come from
workflows wf_423f2732-53f (pass 1, 660 items, 22 typers x 30) and wf_addeb178-8f3 (pass 2, 165 items, 6 typers), blind to
seed, round and source. python3 a7_tax_read.py"""
import json
from collections import Counter, defaultdict
D = "/home/tione/notebook/rsi_rl/dirA5/results/a7_taxonomy"
S = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/sample_full.jsonl")}
L1 = {x["key"]: x for x in json.load(open(f"{D}/labels_pass1.json"))}
L2 = {x["key"]: x for x in json.load(open(f"{D}/labels_pass2.json"))}
TYPES = ["MISSING_PREMISE", "CONTRADICTION", "AMBIGUOUS_REFERENCE", "MULTIPLE_READINGS", "UNDETERMINED_ASK", "OTHER"]
out = {"n_items": len(S), "n_pass1": len(L1), "n_pass2": len(L2), "missing_pass1": sorted(set(S) - set(L1)),
       "strata": {}, "by_round_pooled": {}, "by_audit_verdict": {}, "agreement": {}}
def mix(keys):
    c = Counter(L1[k]["type"] for k in keys if k in L1); n = sum(c.values())
    return {"n": n, "counts": {t: c.get(t, 0) for t in TYPES}, "share": {t: (c.get(t, 0) / n if n else None) for t in TYPES}}
by_stratum, by_round, by_verdict = defaultdict(list), defaultdict(list), defaultdict(list)
for k, it in S.items():
    by_stratum[it["stratum"]].append(k); by_round["raw_r1_pool" if it["source"] == "raw_r1_pool" else f"r{it['round']}"].append(k)
    by_verdict[it["audit"]].append(k)
out["strata"] = {s: mix(v) for s, v in sorted(by_stratum.items())}
out["by_round_pooled"] = {r: mix(v) for r, v in sorted(by_round.items())}
out["by_audit_verdict"] = {r: mix(v) for r, v in sorted(by_verdict.items())}
r1, r5 = out["by_round_pooled"].get("r1"), out["by_round_pooled"].get("r5")
if r1 and r5: out["change_r1_to_r5_points"] = {t: 100 * (r5["share"][t] - r1["share"][t]) for t in TYPES}
per_seed = {}
for s in range(3):
    a, b = out["strata"].get(f"s{s}_r1"), out["strata"].get(f"s{s}_r5")
    if a and b: per_seed[s] = {t: 100 * (b["share"][t] - a["share"][t]) for t in TYPES}
out["change_r1_to_r5_points_per_seed"] = per_seed
def kappa(pairs):
    n = len(pairs)
    if n == 0: return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / n ** 2
    return {"n": n, "raw_agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None}
pairs = [(L1[k]["type"], L2[k]["type"]) for k in L2 if k in L1]
col = lambda t: "AMBIGUITY" if t in ("AMBIGUOUS_REFERENCE", "MULTIPLE_READINGS") else t
out["agreement"] = {"six_class": kappa(pairs), "five_class_ambiguity_merged": kappa([(col(a), col(b)) for a, b in pairs]),
                    "confusion": Counter(f"{a}|{b}" for a, b in pairs if a != b).most_common(10)}
out["pass1_says_well_posed_other"] = sum(1 for x in L1.values() if x["type"] == "OTHER" and "well" in x.get("note", "").lower())
json.dump(out, open(f"{D}/a7_taxonomy_read.json", "w"), indent=1)
print(json.dumps({"by_round": {r: {t: round(v["share"][t], 3) for t in TYPES} | {"n": v["n"]} for r, v in out["by_round_pooled"].items()},
                  "change_r1_r5": {t: round(x, 1) for t, x in out.get("change_r1_to_r5_points", {}).items()},
                  "agreement": out["agreement"]}, indent=1))
