"""Exploratory (NOT a registered read): which label-free per-candidate signal, available to the guard at round t, separates
the proposer's ill-posed candidates from well-posed ones on the fixed pool of A3 section 2d (1,158 audited candidates)?
Signals per checkpoint ck (solver of round ck): plain-prompt p_mode / mode (guard_diag/qwen_s0/<ck>), abstention-prompt
p_mode / mode / abstain share (guard_diag/qwen_s0_ag/<ck>), and the modes of EARLIER checkpoints (loop history).
Candidate admission rules are applied on top of the plain R-Zero band; the table reports admitted n, ill-posed share,
unsolvable share and the number of SOLVABLE candidates kept (retention vs the plain band).
  python3 a4_signal_search.py [pool_dir_plain] [pool_dir_abs] [audit_parts_dir]
"""
import glob
import json
import sys

R = "/home/tione/notebook/rsi_rl"
DP = sys.argv[1] if len(sys.argv) > 1 else f"{R}/dirA3/results/guard_diag/qwen_s0"
DA = sys.argv[2] if len(sys.argv) > 2 else f"{R}/dirA3/results/guard_diag/qwen_s0_ag"
AP = sys.argv[3] if len(sys.argv) > 3 else f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts"
V = {}
for f in glob.glob(f"{AP}/*.json"):
    for v in json.load(open(f))["verdicts"]:
        V[v["id"]] = v["verdict"]
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]


def load(d, ck):
    try:
        return {r["id"]: r for r in (json.loads(l) for l in open(f"{d}/{ck}/guard_records.jsonl")) if r["kind"] == "proposed"}
    except FileNotFoundError:
        return None


P = {ck: load(DP, ck) for ck in CKS}
A = {ck: load(DA, ck) for ck in CKS}
ids = [i for i in V if P["base"] and i in P["base"]]
print(f"candidates with verdicts: {len(ids)}  ({sum(V[i]=='SOLVABLE' for i in ids)} S / {sum(V[i]=='UNSOLVABLE' for i in ids)} U / {sum(V[i]=='AMBIGUOUS' for i in ids)} A)")


def band(p):
    return 0.25 <= p <= 0.75


def rules(ck):
    k = CKS.index(ck)
    prev = CKS[k - 1] if k > 0 else None
    p, a = P[ck], A[ck]
    out = {}
    out["plain band"] = lambda i: band(p[i]["p_mode"])
    if a:
        out["band & abst<2/8"] = lambda i: band(p[i]["p_mode"]) and a[i]["abstain"] < 0.25
        out["band & abst==0"] = lambda i: band(p[i]["p_mode"]) and a[i]["abstain"] == 0
        out["band & mode(plain)==mode(abs)"] = lambda i: band(p[i]["p_mode"]) and p[i]["mode"] is not None and p[i]["mode"] == a[i]["mode"]
        out["band & abst==0 & same mode"] = lambda i: band(p[i]["p_mode"]) and a[i]["abstain"] == 0 and p[i]["mode"] is not None and p[i]["mode"] == a[i]["mode"]
    if ck != "base":
        pb = P["base"]
        out["band & mode==base mode"] = lambda i: band(p[i]["p_mode"]) and p[i]["mode"] is not None and p[i]["mode"] == pb[i]["mode"]
    if prev:
        pp = P[prev]
        out["band & mode==prev mode"] = lambda i: band(p[i]["p_mode"]) and p[i]["mode"] is not None and p[i]["mode"] == pp[i]["mode"]
        if a and A[prev]:
            ap = A[prev]
            out["band & abst==0 & mode==prev"] = lambda i: band(p[i]["p_mode"]) and a[i]["abstain"] == 0 and p[i]["mode"] is not None and p[i]["mode"] == pp[i]["mode"]
    out["p_mode>=0.5 (no upper bound)"] = lambda i: p[i]["p_mode"] >= 0.5
    out["p_mode>=0.25 (no upper bound)"] = lambda i: p[i]["p_mode"] >= 0.25
    return out


for ck in CKS:
    if P[ck] is None:
        continue
    print(f"\n== checkpoint {ck}")
    base_keep = None
    for name, f in rules(ck).items():
        adm = [i for i in ids if i in P[ck] and (A[ck] is None or i in A[ck]) and f(i)]
        n = len(adm)
        s = sum(V[i] == "SOLVABLE" for i in adm); u = sum(V[i] == "UNSOLVABLE" for i in adm)
        if name == "plain band":
            base_keep = s
        print(f"  {name:32s} n {n:4d} | ill-posed {100*(n-s)/max(1,n):5.1f}% | unsolvable {100*u/max(1,n):5.1f}% | solvable kept {s:4d} ({100*s/max(1,base_keep):5.1f}% of plain band)")
# separability of single signals at each checkpoint (per class, among all candidates)
print("\n== per-class signal rates (all candidates): abstain>0 | mode(plain)!=mode(abs) | mode!=base mode | mode!=prev mode")
for ck in CKS:
    if P[ck] is None:
        continue
    k = CKS.index(ck); prev = CKS[k - 1] if k else None
    line = f"  {ck:4s}"
    for cls in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"):
        ii = [i for i in ids if V[i] == cls and i in P[ck] and (A[ck] is None or i in A[ck])]
        ab = sum(A[ck][i]["abstain"] > 0 for i in ii) / len(ii) if A[ck] else float("nan")
        mm = sum(A[ck][i]["mode"] != P[ck][i]["mode"] for i in ii) / len(ii) if A[ck] else float("nan")
        mb = sum(P[ck][i]["mode"] != P["base"][i]["mode"] for i in ii) / len(ii)
        mp = sum(P[ck][i]["mode"] != P[prev][i]["mode"] for i in ii) / len(ii) if prev else float("nan")
        line += f" | {cls[:4]} {100*ab:4.0f} {100*mm:4.0f} {100*mb:4.0f} {100*mp:4.0f}"
    print(line)
