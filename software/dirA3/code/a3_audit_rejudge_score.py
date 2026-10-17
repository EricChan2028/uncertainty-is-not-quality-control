"""Inter-judge agreement between the registered 200-item audit of a run (audit_admitted.json) and an independent
second Opus audit of the same items (dirA3/results/audit_rejudge/<run>/parts/*.json, reversed order).
python3 a3_audit_rejudge_score.py <run>
"""
import glob, json, sys
from collections import Counter
R = "/home/tione/notebook/rsi_rl"
run = sys.argv[1]
A = json.load(open(f"{R}/dirB/runs/{run}/audit_admitted.json"))
v1 = {v["id"]: v["verdict"] for v in A["verdicts"]}
v2 = {}
for f in glob.glob(f"{R}/dirA3/results/audit_rejudge/{run}/parts/*.json"):
    for v in json.load(open(f))["verdicts"]:
        v2[v["id"]] = v["verdict"]
ids = [i for i in v1 if i in v2]
n = len(ids)
agree = sum(v1[i] == v2[i] for i in ids)
cats = ["SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"]
p1 = Counter(v1[i] for i in ids); p2 = Counter(v2[i] for i in ids)
pe = sum((p1[c] / n) * (p2[c] / n) for c in cats)
po = agree / n
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
b1 = {i: ("ILL" if v1[i] != "SOLVABLE" else "OK") for i in ids}
b2 = {i: ("ILL" if v2[i] != "SOLVABLE" else "OK") for i in ids}
ab = sum(b1[i] == b2[i] for i in ids) / n
q1 = Counter(b1.values()); q2 = Counter(b2.values())
peb = sum((q1[c] / n) * (q2[c] / n) for c in ("ILL", "OK"))
kb = (ab - peb) / (1 - peb)
conf = Counter((v1[i], v2[i]) for i in ids)
print(f"{run}: n={n} agreement {100*po:.1f}% kappa(3-class) {kappa:.3f}; ill-posed(uns+amb) vs solvable agreement {100*ab:.1f}% kappa {kb:.3f}")
print(f"  unsolvable share judge set 1 {100*p1['UNSOLVABLE']/n:.1f}%  set 2 {100*p2['UNSOLVABLE']/n:.1f}%;  ill-posed share {100*(n-p1['SOLVABLE'])/n:.1f}% vs {100*(n-p2['SOLVABLE'])/n:.1f}%")
print("  confusion (set1, set2):", dict(sorted(conf.items())))
json.dump({"run": run, "n": n, "agreement": po, "kappa3": kappa, "agreement_binary": ab, "kappa_binary": kb,
           "unsolvable_share": [p1["UNSOLVABLE"] / n, p2["UNSOLVABLE"] / n], "illposed_share": [(n - p1["SOLVABLE"]) / n, (n - p2["SOLVABLE"]) / n],
           "confusion": {f"{a}|{b}": c for (a, b), c in conf.items()}}, open(f"{R}/dirA3/results/audit_rejudge/{run}/result.json", "w"), indent=1)
