"""[EXP correction, 2026-09-24 05:15] Numerically stable re-run of the sharpening-law fits. a4_sharpening_law.py and
a4_nt_analyses.py (3) compute a = q0**g, b = r*o**g in linear space; for large g both underflow to 0 and a/(a+b) is NaN, and
the grid search silently skips those beta values (NaN < best is False). Here the same model is evaluated in log space.
Outputs: per-class exponents (fit on qwen_s0, grid 1.00-8.00), held-out qwen_s1 prediction, and the single shared-rate model
(beta fitted on SOLVABLE only, grid 1.00-8.00) applied to all classes on both pools.   python3 a4_law_stable.py
"""
import glob, json, math
import numpy as np
R = "/home/tione/notebook/rsi_rl"
AUD = {"qwen_s0": "qwen_s0_r1", "qwen_s1": "qwen_s1_r1"}
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]; CL = ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")
GRID = np.round(np.arange(1.0, 8.0001, 0.05), 2)
K = np.arange(2, 7); BIN = np.array([math.comb(8, k) for k in K], dtype=float)
def load(tag):
    V = {}
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/{AUD[tag]}/parts/*.json"):
        for v in json.load(open(f))["verdicts"]: V[v["id"]] = v["verdict"]
    P = {c: {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}/{c}/guard_records.jsonl")) if r["kind"] == "proposed"} for c in CKS}
    ids = [i for i in V if i in P["base"] and V[i] in CL]
    out = {}
    for cl in CL:
        its = [i for i in ids if V[i] == cl]
        out[cl] = (np.array([(P["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in its]),
                   np.nanmean(np.array([[float(0.25 <= P[c][i]["p_mode"] <= 0.75) if i in P[c] else np.nan for c in CKS] for i in its]), 0))
    return out
def m_model(q0, g):
    q0 = np.clip(q0, 1e-3, 1 - 1e-3); r = np.maximum(1, np.ceil((1 - q0) / q0 - 1e-9)); o = (1 - q0) / r
    q = 1 / (1 + np.exp(np.log(r) + g * np.log(o) - g * np.log(q0)))
    return float((BIN[None, :] * q[:, None] ** K[None, :] * (1 - q[:, None]) ** (8 - K[None, :])).sum(1).mean())
def traj(q0, b): return np.array([m_model(q0, b ** t) for t in range(6)])
def fit(q0, obs):
    e = [((traj(q0, b)[1:] - obs[1:]) ** 2).sum() for b in GRID]; return float(GRID[int(np.argmin(e))]), e
def share(pool, m):
    n = {cl: len(pool[cl][0]) for cl in CL}; pi = {cl: n[cl] / sum(n.values()) for cl in CL}
    num = pi["AMBIGUOUS"] * m["AMBIGUOUS"] + pi["UNSOLVABLE"] * m["UNSOLVABLE"]; return 100 * num / (num + pi["SOLVABLE"] * m["SOLVABLE"])
tr, te = load("qwen_s0"), load("qwen_s1")
res = {}
beta = {}; errs = {}
for cl in CL:
    beta[cl], errs[cl] = fit(*tr[cl])
res["beta_per_class"] = beta
# flatness of the SOLVABLE fit: grid values within 10% of the minimum SSE
e = np.array(errs["SOLVABLE"]); res["SOLVABLE_beta_within_10pct_sse"] = [float(GRID[e <= 1.1 * e.min()].min()), float(GRID[e <= 1.1 * e.min()].max())]
res["SOLVABLE_sse_at_3.30_vs_min"] = [float(e[list(GRID).index(3.3)]), float(e.min())]
for tag, pool in (("qwen_s0", tr), ("qwen_s1", te)):
    obs = share(pool, {cl: pool[cl][1] for cl in CL}); pred = share(pool, {cl: traj(pool[cl][0], beta[cl]) for cl in CL})
    single = share(pool, {cl: traj(pool[cl][0], beta["SOLVABLE"]) for cl in CL})
    res[tag] = {"observed": obs.round(1).tolist(), "per_class_law": pred.round(1).tolist(), "mae_per_class": round(float(np.abs(pred - obs).mean()), 2),
                "single_rate_law": single.round(1).tolist(), "mae_single": round(float(np.abs(single - obs).mean()), 2),
                "in_band_obs": {cl: pool[cl][1].round(3).tolist() for cl in CL}, "in_band_law": {cl: traj(pool[cl][0], beta[cl]).round(3).tolist() for cl in CL}}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{R}/dirA4/results/law_stable_2026-09-24.json", "w"), indent=1)
