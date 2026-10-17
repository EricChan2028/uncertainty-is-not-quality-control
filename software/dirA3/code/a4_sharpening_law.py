"""Per-class sharpening law (A4, mechanism): the answer distribution of a question of audit class c evolves as
q_t(a) proportional to q_0(a)^(beta_c^t); q_0 approximated from the base-solver agreement (mode share q0, remaining mass
over r = ceil((1-q0)/q0) answers); in-band probability from Binomial(G = 8). beta_c is fitted per class on the TRAIN pool
(in-band mass trajectory base..r5); the law then predicts, from the TEST pool's base agreements alone, each class's in-band
mass and the admitted ill-posed share at every checkpoint. Also reports the fit of logit(share) vs round (linearity).
  python3 a4_sharpening_law.py [train=qwen_s0] [test=qwen_s1]
"""
import glob
import json
import math
import sys
from math import comb

import numpy as np

R = "/home/tione/notebook/rsi_rl"
TR = sys.argv[1] if len(sys.argv) > 1 else "qwen_s0"
TE = sys.argv[2] if len(sys.argv) > 2 else "qwen_s1"
AUD = {"qwen_s0": "qwen_s0_r1", "qwen_s1": "qwen_s1_r1"}
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
CL = ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")


def load_pool(tag):
    V = {}
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/{AUD[tag]}/parts/*.json"):
        for v in json.load(open(f))["verdicts"]:
            V[v["id"]] = v["verdict"]
    P = {}
    for c in CKS:
        try:
            P[c] = {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}/{c}/guard_records.jsonl")) if r["kind"] == "proposed"}
        except FileNotFoundError:
            P[c] = None
    ids = [i for i in V if P["base"] and i in P["base"]]
    return V, P, ids


def ib(q, G=8):
    return sum(comb(G, k) * q ** k * (1 - q) ** (G - k) for k in range(G + 1) if 0.25 <= k / G <= 0.75)


def sh(q0, g):
    q0 = min(max(q0, 1e-3), 1 - 1e-3); r = max(1, math.ceil((1 - q0) / q0 - 1e-9)); o = (1 - q0) / r
    a = q0 ** g; b = r * o ** g
    return a / (a + b)


def masses_obs(V, P, ids):
    return {cl: [float(np.mean([0.25 <= P[c][i]["p_mode"] <= 0.75 for i in ids if V[i] == cl and i in P[c]])) if P[c] else None for c in CKS] for cl in CL}


def masses_model(V, P, ids, beta):
    q0 = {i: (P["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in ids}
    return {cl: [float(np.mean([ib(sh(q0[i], beta[cl] ** t)) for i in ids if V[i] == cl])) for t in range(len(CKS))] for cl in CL}


def share(pi, m, t):
    num = sum(pi[c] * m[c][t] for c in ("AMBIGUOUS", "UNSOLVABLE")); den = num + pi["SOLVABLE"] * m["SOLVABLE"][t]
    return 100 * num / den


Vt, Pt, It = load_pool(TR)
obs_t = masses_obs(Vt, Pt, It)
beta = {}
for cl in CL:
    best = None
    q0 = {i: (Pt["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in It if Vt[i] == cl}
    for b in np.arange(1.0, 8.01, 0.05):
        m = [np.mean([ib(sh(q, b ** t)) for q in q0.values()]) for t in range(len(CKS))]
        e = sum((m[t] - obs_t[cl][t]) ** 2 for t in range(1, len(CKS)) if obs_t[cl][t] is not None)
        if best is None or e < best[0]:
            best = (e, float(b))
    beta[cl] = best[1]
print(f"train pool {TR}: per-class sharpening exponents {beta}")
res = {"train": TR, "test": TE, "beta": beta}
for tag in (TR, TE):
    V, P, ids = (Vt, Pt, It) if tag == TR else load_pool(tag)
    if not ids:
        print(f"{tag}: no data"); continue
    pi = {cl: sum(V[i] == cl for i in ids) / len(ids) for cl in CL}
    mo, mm = masses_obs(V, P, ids), masses_model(V, P, ids, beta)
    so = [share(pi, mo, t) if all(mo[c][t] is not None for c in CL) else None for t in range(len(CKS))]
    sp = [share(pi, mm, t) for t in range(len(CKS))]
    err = [abs(a - b) for a, b in zip(so, sp) if a is not None]
    kind = "in-sample" if tag == TR else "HELD-OUT prediction from base agreements only"
    print(f"{tag} ({kind}): admitted ill-posed share observed {[None if x is None else round(x, 1) for x in so]}  law {[round(x, 1) for x in sp]}  MAE {np.mean(err):.1f} pts")
    for cl in CL:
        print(f"   {cl:10s} in-band mass obs {[None if x is None else round(x, 3) for x in mo[cl]]}  law {[round(x, 3) for x in mm[cl]]}")
    ok = [(t, x) for t, x in enumerate(so) if x is not None]
    if len(ok) >= 3:
        tt = np.array([t for t, _ in ok]); lg = np.array([math.log(x / (100 - x)) for _, x in ok])
        A_ = np.vstack([tt, np.ones_like(tt)]).T; coef, resid, *_ = np.linalg.lstsq(A_, lg, rcond=None)
        r2 = 1 - (resid[0] if len(resid) else 0) / max(1e-9, ((lg - lg.mean()) ** 2).sum())
        print(f"   logit(share) vs t: slope {coef[0]:.3f} per round, R^2 {r2:.3f}")
    res[tag] = {"observed": so, "law": sp, "mae": float(np.mean(err)), "pi": pi}
json.dump(res, open(f"{R}/dirA4/results/sharpening_law_{TR}_to_{TE}.json", "w"), indent=1)
