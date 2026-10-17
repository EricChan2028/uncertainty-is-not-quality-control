"""[EXP, 2026-09-24] Bootstrap uncertainty for the per-class sharpening law (a4_sharpening_law.py, vectorised, same model):
resample the TRAIN pool's items within class (B reps), refit beta per class on the grid 1.00-8.00 step 0.05, and predict the
TEST pool's admitted ill-posed share from its base agreements (test items also resampled within class). Reports 95%
percentile intervals for beta_c, for beta_S / beta_U, for the predicted test trajectory, and for the held-out MAE; also
the share of reps with beta_S > max(beta_A, beta_U) (the ordering the law needs).   python3 a4_law_bootstrap.py [B=500]
"""
import glob, json, math, sys
import numpy as np
R = "/home/tione/notebook/rsi_rl"
AUD = {"qwen_s0": "qwen_s0_r1", "qwen_s1": "qwen_s1_r1"}
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]; CL = ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
GRID = np.round(np.arange(1.0, 8.0001, 0.05), 2)
K = np.arange(2, 7)  # k/8 in [0.25, 0.75]
BIN = np.array([math.comb(8, k) for k in K], dtype=float)

def load(tag):
    V = {}
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/{AUD[tag]}/parts/*.json"):
        for v in json.load(open(f))["verdicts"]: V[v["id"]] = v["verdict"]
    P = {c: {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}/{c}/guard_records.jsonl")) if r["kind"] == "proposed"} for c in CKS}
    ids = [i for i in V if i in P["base"] and V[i] in CL]
    out = {}
    for cl in CL:
        its = [i for i in ids if V[i] == cl]
        q0 = np.array([(P["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in its])
        inb = np.array([[float(0.25 <= P[c][i]["p_mode"] <= 0.75) if i in P[c] else np.nan for c in CKS] for i in its])
        out[cl] = (q0, inb)
    return out

def inband_model(q0, g):
    q0 = np.clip(q0, 1e-3, 1 - 1e-3); r = np.maximum(1, np.ceil((1 - q0) / q0 - 1e-9)); o = (1 - q0) / r
    la = g * np.log(q0); lb = np.log(r) + g * np.log(o); q = 1 / (1 + np.exp(lb - la))
    q = q[:, None]; return (BIN[None, :] * q ** K[None, :] * (1 - q) ** (8 - K[None, :])).sum(1)

def table(q0):  # grid x t -> mean in-band mass over items
    return np.array([[inband_model(q0, b ** t).mean() for t in range(6)] for b in GRID])

def fit(q0, inb):
    obs = np.nanmean(inb, 0); T = table(q0)
    e = ((T[:, 1:] - obs[None, 1:]) ** 2).sum(1); return GRID[int(np.argmin(e))]

tr, te = load("qwen_s0"), load("qwen_s1")
rng = np.random.default_rng(0)
def share(pi, m):  # m: cl -> array over t
    num = pi["AMBIGUOUS"] * m["AMBIGUOUS"] + pi["UNSOLVABLE"] * m["UNSOLVABLE"]; return 100 * num / (num + pi["SOLVABLE"] * m["SOLVABLE"])
ntr = {cl: len(tr[cl][0]) for cl in CL}; nte = {cl: len(te[cl][0]) for cl in CL}
pi_te = {cl: nte[cl] / sum(nte.values()) for cl in CL}
obs_te_full = share(pi_te, {cl: np.nanmean(te[cl][1], 0) for cl in CL})
betas, preds, maes = [], [], []
for b in range(B + 1):
    be = {}
    for cl in CL:
        q0, inb = tr[cl]
        idx = np.arange(len(q0)) if b == 0 else rng.integers(0, len(q0), len(q0))
        be[cl] = fit(q0[idx], inb[idx])
    m, mo = {}, {}
    for cl in CL:
        q0, inb = te[cl]
        idx = np.arange(len(q0)) if b == 0 else rng.integers(0, len(q0), len(q0))
        m[cl] = np.array([inband_model(q0[idx], be[cl] ** t).mean() for t in range(6)])
        mo[cl] = np.nanmean(inb[idx], 0)
    p = share(pi_te, m); o = share(pi_te, mo)
    betas.append([be[c] for c in CL]); preds.append(p); maes.append(float(np.mean(np.abs(p - o))))
betas, preds, maes = np.array(betas), np.array(preds), np.array(maes)
ci = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
res = {"B": B, "point": {"beta": dict(zip(CL, betas[0].tolist())), "pred_test": preds[0].round(1).tolist(), "obs_test": obs_te_full.round(1).tolist(), "mae_test": round(maes[0], 2)},
       "ci95": {**{f"beta_{c}": ci(betas[1:, j]) for j, c in enumerate(CL)}, "beta_S_over_beta_U": ci(betas[1:, 0] / betas[1:, 2]),
                "pred_test_by_ck": [ci(preds[1:, t]) for t in range(6)], "mae_test": ci(maes[1:])},
       "share_reps_beta_S_gt_max_A_U": float(np.mean(betas[1:, 0] > np.maximum(betas[1:, 1], betas[1:, 2])))}
print(json.dumps(res, indent=1))
json.dump(res, open(f"{R}/dirA4/results/law_bootstrap.json", "w"), indent=1)
