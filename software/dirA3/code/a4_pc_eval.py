"""Evaluate program-closure (PC) guards on an audited fixed pool: pass rates by audit class for several PC criteria, and the
admitted ill-posed share / solvable count per checkpoint for band & PC (and floor [.5,.875] & PC).
  python3 a4_pc_eval.py <pool_tag qwen_s0|qwen_s1> <audit_parts_dir> <pc_records.jsonl>
"""
import glob
import json
import sys

R = "/home/tione/notebook/rsi_rl"
TAG, AUD, PCF = sys.argv[1], sys.argv[2], sys.argv[3]
V = {}
for f in glob.glob(f"{AUD}/*.json"):
    for v in json.load(open(f))["verdicts"]:
        V[v["id"]] = v["verdict"]
PC = {json.loads(l)["id"]: json.loads(l) for l in open(PCF)}
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
P = {}
for c in CKS:
    try:
        P[c] = {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{TAG}/{c}/guard_records.jsonl")) if r["kind"] == "proposed"}
    except FileNotFoundError:
        P[c] = None
ids = [i for i in V if i in PC]
CRIT = {
    "any closed": lambda r: r["n_closed"] >= 1,
    ">=2 closed": lambda r: r["n_closed"] >= 2,
    ">=2 closed & agree": lambda r: r["agree_closed"] >= 2,
    "majority closed & agree (>=3)": lambda r: r["agree_closed"] >= 3,
    "no foreign literal in any running program": lambda r: r["n_ok"] >= 1 and r["any_foreign"] == 0,
}
print(f"pool {TAG}: {len(ids)} candidates with audit + programs")
print("pass rate by class (SOLVABLE / AMBIGUOUS / UNSOLVABLE) and ill-posed share of passers:")
for name, f in CRIT.items():
    rates = {cl: sum(f(PC[i]) for i in ids if V[i] == cl) / max(1, sum(V[i] == cl for i in ids)) for cl in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
    pas = [i for i in ids if f(PC[i])]
    print(f"  {name:42s} {100*rates['SOLVABLE']:5.1f} / {100*rates['AMBIGUOUS']:5.1f} / {100*rates['UNSOLVABLE']:5.1f} | ill-posed of passers {100*sum(V[i]!='SOLVABLE' for i in pas)/max(1,len(pas)):5.1f}% (n {len(pas)})")
print("per checkpoint: ill-posed% / solvable n of the admitted set")
for name, f in list(CRIT.items())[1:4]:
    for gname, g in (("band", lambda p: .25 <= p <= .75), ("floor [.5,.875]", lambda p: .5 <= p <= .875), ("band + (.75,.875]", lambda p: .25 <= p <= .875)):
        line = f"  {gname:18s} & {name:30s}"
        for c in CKS:
            if P[c] is None:
                line += " |    -"; continue
            adm = [i for i in ids if i in P[c] and g(P[c][i]["p_mode"]) and f(PC[i])]
            s = sum(V[i] == "SOLVABLE" for i in adm)
            line += f" | {100*(len(adm)-s)/max(1,len(adm)):4.1f} {s:3d}"
        print(line)
line = "  plain band (reference)                          "
for c in CKS:
    if P[c] is None:
        line += " |    -"; continue
    adm = [i for i in ids if i in P[c] and .25 <= P[c][i]["p_mode"] <= .75]
    s = sum(V[i] == "SOLVABLE" for i in adm)
    line += f" | {100*(len(adm)-s)/max(1,len(adm)):4.1f} {s:3d}"
print(line)
