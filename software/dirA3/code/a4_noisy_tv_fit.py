"""Noisy-TV model of curriculum drift (A4a): admitted ill-posed share s_t = sum_{c in ill} pi_c m_c(t) / sum_c pi_c m_c(t), where
pi_c is the pool share of audit class c (SOLVABLE / AMBIGUOUS / UNSOLVABLE) and m_c(t) the class's in-band mass under the
round-t solver. Per-class dynamics: m_c(t) = m_c,inf + (m_c(0) - m_c,inf) * rho_c^t (SOLVABLE: m_inf fitted too).
Fit on the s0 fixed pool; predict (i) the held-out s1 pool from its own t = 0 masses, (ii) every audited Qwen-4B own-loop
round (guard at round n uses the round n-1 solver, so round n <-> t = n - 1). Writes dirA4/results/noisy_tv_fit.json.
  python3 a4_noisy_tv_fit.py [lo hi]
"""
import glob
import json
import os
import sys

R = "/home/tione/notebook/rsi_rl"
LO, HI = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.25, 0.75)
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
CLS = ["SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"]


def verdicts(d):
    V = {}
    for f in glob.glob(f"{d}/*.json"):
        for v in json.load(open(f))["verdicts"]:
            V[v["id"]] = v["verdict"]
    return V


def masses(gd, V):
    out = []
    for ck in CKS:
        f = f"{gd}/{ck}/guard_records.jsonl"
        if not os.path.exists(f):
            out.append(None); continue
        recs = {r["id"]: r for r in (json.loads(l) for l in open(f)) if r["kind"] == "proposed"}
        m = {}
        for c in CLS:
            ids = [i for i in V if V[i] == c and i in recs]
            m[c] = sum(LO <= recs[i]["p_mode"] <= HI for i in ids) / max(1, len(ids))
        out.append(m)
    return out


def share(pi, m):
    num = sum(pi[c] * m[c] for c in ("AMBIGUOUS", "UNSOLVABLE")); den = num + pi["SOLVABLE"] * m["SOLVABLE"]
    return 100 * num / den if den else None


def fit_class(ms):
    """grid search m_inf in [0, m0], rho in (0,1) minimising squared error of m(t)."""
    m0 = ms[0]; best = None
    for i in range(0, 101):
        minf = m0 * i / 100
        for j in range(1, 100):
            rho = j / 100
            err = sum((minf + (m0 - minf) * rho ** t - ms[t]) ** 2 for t in range(len(ms)) if ms[t] is not None)
            if best is None or err < best[0]:
                best = (err, minf / m0 if m0 else 0, rho)
    return {"minf_frac": best[1], "rho": best[2], "sse": best[0]}


V0 = verdicts(f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts")
pi0 = {c: sum(v == c for v in V0.values()) / len(V0) for c in CLS}
M0 = masses(f"{R}/dirA3/results/guard_diag/qwen_s0", V0)
params = {c: fit_class([m[c] for m in M0]) for c in CLS}
print(f"band [{LO}, {HI}]  s0 pool shares {({c: round(pi0[c], 3) for c in CLS})}")
for c in CLS:
    print(f"  {c:10s} m(t) observed {[round(m[c], 3) for m in M0]} -> fit m_inf/m0 {params[c]['minf_frac']:.2f} rho {params[c]['rho']:.2f}")


def predict(pi, m0, T=6):
    out = []
    for t in range(T):
        m = {c: m0[c] * (params[c]["minf_frac"] + (1 - params[c]["minf_frac"]) * params[c]["rho"] ** t) for c in CLS}
        out.append(share(pi, m))
    return out


obs0 = [share(pi0, m) for m in M0]
fit0 = predict(pi0, M0[0])
print("s0 pool (in-sample): observed", [round(x, 1) for x in obs0], " model", [round(x, 1) for x in fit0])
res = {"band": [LO, HI], "params": params, "s0": {"observed": obs0, "model": fit0}}
V1 = verdicts(f"{R}/dirA3/results/pool_audit/qwen_s1_r1/parts")
if V1:
    pi1 = {c: sum(v == c for v in V1.values()) / len(V1) for c in CLS}
    M1 = masses(f"{R}/dirA3/results/guard_diag/qwen_s1", V1)
    if M1[0]:
        obs1 = [share(pi1, m) if m else None for m in M1]
        pred1 = predict(pi1, M1[0])
        err = [abs(a - b) for a, b in zip(obs1, pred1) if a is not None]
        print("s1 pool (HELD OUT): observed", [None if x is None else round(x, 1) for x in obs1], " predicted", [round(x, 1) for x in pred1], f" MAE {sum(err)/len(err):.1f} pts")
        res["s1"] = {"observed": obs1, "predicted": pred1, "mae": sum(err) / len(err)}
# loop rounds: audited ill-posed share (UNSOLVABLE + AMBIGUOUS of the 200-item audit) vs the s0-fixed-pool model at t = n - 1
rows = []
for f in sorted(glob.glob(f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s*_r*/audit_admitted.json")):
    run = f.split("/")[-2]; n = int(run.rsplit("_r", 1)[1])
    c = json.load(open(f))["counts"]; tot = sum(c.values())
    if n - 1 < len(fit0):
        rows.append((run, n, 100 * (tot - c.get("SOLVABLE", 0)) / tot, fit0[n - 1]))
if rows:
    mae = sum(abs(a - b) for _, _, a, b in rows) / len(rows)
    print(f"Qwen-4B own loop rounds (fresh pools each round, 5 seeds): {len(rows)} audited rounds, MAE vs fixed-pool model {mae:.1f} pts")
    for run, n, a, b in rows:
        print(f"  {run:32s} round {n}: audited ill-posed {a:5.1f}  model {b:5.1f}")
    res["loops"] = {"n": len(rows), "mae": mae, "rows": rows}
os.makedirs(f"{R}/dirA4/results", exist_ok=True)
json.dump(res, open(f"{R}/dirA4/results/noisy_tv_fit_{LO}_{HI}.json", "w"), indent=1)
