"""A4a.1 registered reads NG1-NG3 on a pool (default: held-out s1 pool; 'qwen_s0' for the search pool, for reference).
  python3 a4_ng_read.py [qwen_s1|qwen_s0]
"""
import glob
import json
import sys

R = "/home/tione/notebook/rsi_rl"
POOL = sys.argv[1] if len(sys.argv) > 1 else "qwen_s1"
AUD = {"qwen_s1": "qwen_s1_r1", "qwen_s0": "qwen_s0_r1"}[POOL]
V = {}
for f in glob.glob(f"{R}/dirA3/results/pool_audit/{AUD}/parts/*.json"):
    for v in json.load(open(f))["verdicts"]:
        V[v["id"]] = v["verdict"]
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]


def load(d, ck):
    try:
        return {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{d}/{ck}/guard_records.jsonl")) if r["kind"] == "proposed"}
    except FileNotFoundError:
        return None


P = {c: load(POOL, c) for c in CKS}
A = {c: load(POOL + "_ag", c) for c in CKS}
RULES = {
    "R1 band [.25,.75]": lambda p, a: .25 <= p["p_mode"] <= .75,
    "R2 floor [.5,.875]": lambda p, a: .5 <= p["p_mode"] <= .875,
    "R3 floor [.5,.875] & no abstention": lambda p, a: .5 <= p["p_mode"] <= .875 and a["abstain"] == 0,
    "R4 band & abstention < 2/8": lambda p, a: .25 <= p["p_mode"] <= .75 and a["abstain"] < .25,
    "R5 band & no abst & same mode": lambda p, a: .25 <= p["p_mode"] <= .75 and a["abstain"] == 0 and p["mode"] is not None and p["mode"] == a["mode"],
    "R6 [.5, 1]": lambda p, a: p["p_mode"] >= .5,
}
print(f"pool {POOL}: {len(V)} audited candidates ({sum(v=='SOLVABLE' for v in V.values())} S)")
res = {}
for name, f in RULES.items():
    sh, ns = [], []
    for c in CKS:
        if P[c] is None or A[c] is None:
            sh.append(None); ns.append(None); continue
        ids = [i for i in V if i in P[c] and i in A[c]]
        adm = [i for i in ids if f(P[c][i], A[c][i])]
        s = sum(V[i] == "SOLVABLE" for i in adm)
        sh.append(100 * (len(adm) - s) / max(1, len(adm)))
        ns.append(sum(1 for i in adm if V[i] == "SOLVABLE" and P[c][i]["p_mode"] < 1))
    res[name] = {"share": sh, "solvable_gradient_bearing": ns}
    print(f"  {name:36s} ill-posed% " + " ".join("-" if x is None else f"{x:5.1f}" for x in sh) + "   solvable(p<1) " + " ".join("-" if x is None else f"{x:4d}" for x in ns))
ok = lambda x: x is not None
ng1 = all(ok(res[k]["share"][0]) and ok(res[k]["share"][5]) and res[k]["share"][5] - res[k]["share"][0] >= 15 for k in list(RULES)[:5])
r6 = res["R6 [.5, 1]"]
ng2 = ok(r6["share"][0]) and ok(r6["share"][5]) and r6["share"][5] - r6["share"][0] < 10 and r6["solvable_gradient_bearing"][5] <= 0.5 * r6["solvable_gradient_bearing"][0]
gb = []
for c in CKS:
    if P[c] is None:
        gb.append(None); continue
    ids = [i for i in V if i in P[c] and 0.125 <= P[c][i]["p_mode"] <= 0.875]
    gb.append(100 * sum(V[i] != "SOLVABLE" for i in ids) / max(1, len(ids)))
ng3 = all(ok(x) for x in gb) and all(gb[i + 1] > gb[i] for i in range(5))
print(f"  gradient-bearing (1/8 <= p <= 7/8) ill-posed%: " + " ".join("-" if x is None else f"{x:5.1f}" for x in gb))
print(f"NG1 {ng1} | NG2 {ng2} | NG3 {ng3}")
json.dump({"pool": POOL, "rules": res, "gradient_bearing": gb, "NG1": ng1, "NG2": ng2, "NG3": ng3}, open(f"{R}/dirA4/results/ng_read_{POOL}.json", "w"), indent=1)
