"""A5 J-P1 read: frozen Opus audit on 400 UMWP items vs UMWP human labels. Ill-posed = UNSOLVABLE or AMBIGUOUS.
J-P1 holds if >= 85% of unanswerable are ill-posed, <= 12% of answerable are ill-posed, and binary Cohen kappa >= 0.70.
Also: per-category detection, 3-class confusion.   python3 a5_j1_p1_read.py"""
import glob, json
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"
K = json.load(open(f"{R}/dirA5/results/j1_umwp400_key.json"))
V = {}
for f in glob.glob(f"{R}/dirB/runs/umwp400_j1/audit_parts/*.json"):
    for v in json.load(open(f))["verdicts"]:
        V[v["id"]] = v["verdict"]
ids = [i for i in K if i in V]
y = [0 if K[i]["answerable"] else 1 for i in ids]           # 1 = unanswerable (human label)
p = [0 if V[i] == "SOLVABLE" else 1 for i in ids]           # 1 = ill-posed (judge)
n = len(ids); po = sum(a == b for a, b in zip(y, p)) / n
py, pp = sum(y) / n, sum(p) / n; pe = py * pp + (1 - py) * (1 - pp)
kappa = (po - pe) / (1 - pe)
det = sum(a and b for a, b in zip(y, p)) / sum(y)
fp = sum((not a) and b for a, b in zip(y, p)) / (n - sum(y))
bycat = defaultdict(list)
for i in ids:
    if not K[i]["answerable"]: bycat[K[i]["category"]].append(V[i] != "SOLVABLE")
res = {"n": n, "detection_unanswerable": det, "false_illposed_answerable": fp, "kappa_binary": kappa, "agreement": po,
       "J_P1": det >= 0.85 and fp <= 0.12 and kappa >= 0.70,
       "detection_by_category": {str(k): round(sum(v) / len(v), 3) for k, v in sorted(bycat.items())},
       "confusion": {h: dict(Counter(V[i] for i in ids if K[i]["answerable"] == (h == "answerable"))) for h in ("answerable", "unanswerable")}}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{R}/dirA5/results/j1_p1_read.json", "w"), indent=1)
