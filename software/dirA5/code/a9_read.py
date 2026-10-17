"""A9 read (registered 00:00 25 Sep, dirA5/docs/A9_registration_2026-09-25.md). Per family (Qwen3-8B-Base s0, Phi-4-mini s0) on
the fixed audited pool qwen_s0_r1: the stable log-space per-class model of a4_law_stable.py / a4_law_bootstrap.py (same
functions, grid 1.00-8.00 step 0.05, squared error over t = 1..3 because 3 rounds exist), fitted per class.
L-P1: beta_SOLVABLE > beta_UNSOLVABLE in each family, and the ordering holds in >= 90% of 1,000 item-bootstrap resamples (items
      resampled within class; numpy default_rng(20260925)).
L-P2: point estimates ordered beta_S > beta_A > beta_U in each family.
Complete data only: every checkpoint (base, r1, r2, r3) must have guard_records.jsonl with all 1,158 proposed items, else the
family is not read (each checkpoint also needs guard_stats.json, and the proposed-id set must equal the audited set).
Ties on the grid count as not ordered. Files and items are read in sorted order (bootstrap reproducibility). Descriptive: betas with 95% bootstrap intervals; observed vs
predicted ill-posed share of the in-band set per checkpoint under own betas and under the Qwen3-4B betas (3.75 / 1.60 / 1.40);
the single shared rate (beta fitted on SOLVABLE only); MAE over r1..r3 and over all checkpoints; the share of bootstrap reps
with S > max(A, U) (the statistic quoted for Qwen3-4B) and at the grid bounds 1.00 / 8.00; per class the base-agreement
histogram, the frozen share (base p_mode in {0, 1/8, 4/8}) and the model floor; the share of records with no valid answer
per checkpoint; and a model-free check (misspecification alone can order fitted betas): in-band retention at r1..r3 of
SOLVABLE vs UNSOLVABLE items within the same base-agreement stratum k = 8 p_mode in 2..6, standardised to the UNSOLVABLE
stratum weights. python3 a9_read.py"""
import glob, json, math, os
import numpy as np
R = "/home/tione/notebook/rsi_rl"; CL = ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"); CKS = ["base", "r1", "r2", "r3"]
GRID = np.round(np.arange(1.0, 8.0001, 0.05), 2); K = np.arange(2, 7); BIN = np.array([math.comb(8, k) for k in K], dtype=float)
QWEN4B = {"SOLVABLE": 3.75, "AMBIGUOUS": 1.60, "UNSOLVABLE": 1.40}
V = {}
for f in sorted(glob.glob(f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts/*.json")):
    for v in json.load(open(f))["verdicts"]: V[v["id"]] = v["verdict"]
def load(fam):
    P = {}
    for c in CKS:
        f = f"{R}/dirA3/results/guard_diag/{fam}/{c}/guard_records.jsonl"
        if not os.path.exists(f) or not os.path.exists(f"{R}/dirA3/results/guard_diag/{fam}/{c}/guard_stats.json"): return None
        try: P[c] = {r["id"]: r for r in (json.loads(l) for l in open(f)) if r["kind"] == "proposed"}
        except ValueError: return None
        if set(P[c]) != set(V): return None
    ids = sorted(i for i in V if V[i] in CL); out = {}
    RAW[fam] = P
    for cl in CL:
        its = [i for i in ids if V[i] == cl]
        out[cl] = (np.array([(P["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in its]),
                   np.array([[float(0.25 <= P[c][i]["p_mode"] <= 0.75) for c in CKS] for i in its]))
    return out
RAW = {}
def inband_model(q0, g):
    q0 = np.clip(q0, 1e-3, 1 - 1e-3); r = np.maximum(1, np.ceil((1 - q0) / q0 - 1e-9)); o = (1 - q0) / r
    la = g * np.log(q0); lb = np.log(r) + g * np.log(o); q = 1 / (1 + np.exp(lb - la)); q = q[:, None]
    return (BIN[None, :] * q ** K[None, :] * (1 - q) ** (8 - K[None, :])).sum(1)
def table(q0): return np.array([[inband_model(q0, b ** t).mean() for t in range(len(CKS))] for b in GRID])
def fit(q0, inb):
    obs = inb.mean(0); T = table(q0); e = ((T[:, 1:] - obs[None, 1:]) ** 2).sum(1); return float(GRID[int(np.argmin(e))])
def traj(q0, b): return np.array([inband_model(q0, b ** t).mean() for t in range(len(CKS))])
def share(pool, m):
    n = {cl: len(pool[cl][0]) for cl in CL}; pi = {cl: n[cl] / sum(n.values()) for cl in CL}
    num = pi["AMBIGUOUS"] * m["AMBIGUOUS"] + pi["UNSOLVABLE"] * m["UNSOLVABLE"]; return 100 * num / (num + pi["SOLVABLE"] * m["SOLVABLE"])
res = {"families": {}}
for fam in ("fam_qwen3_8b_s0", "fam_phi4mini_s0"):
    pool = load(fam)
    if pool is None: res["families"][fam] = None; print(fam, "not read (incomplete)"); continue
    beta = {cl: fit(*pool[cl]) for cl in CL}
    rng = np.random.default_rng(20260925); bs = {cl: [] for cl in CL}
    for _ in range(1000):
        for cl in CL:
            q0, inb = pool[cl]; idx = rng.integers(0, len(q0), len(q0)); bs[cl].append(fit(q0[idx], inb[idx]))
    order_su = float(np.mean(np.array(bs["SOLVABLE"]) > np.array(bs["UNSOLVABLE"])))
    order_full = float(np.mean((np.array(bs["SOLVABLE"]) > np.array(bs["AMBIGUOUS"])) & (np.array(bs["AMBIGUOUS"]) > np.array(bs["UNSOLVABLE"]))))
    order_max = float(np.mean(np.array(bs["SOLVABLE"]) > np.maximum(np.array(bs["AMBIGUOUS"]), np.array(bs["UNSOLVABLE"]))))
    at_bounds = {cl: {"1.00": float(np.mean(np.array(bs[cl]) <= 1.0 + 1e-9)), "8.00": float(np.mean(np.array(bs[cl]) >= 8.0 - 1e-9))} for cl in CL}
    P = RAW[fam]; desc = {}
    for cl in CL:
        its = [i for i in sorted(V) if V[i] == cl]; kb = [round(8 * P["base"][i]["p_mode"]) for i in its]
        hist = {k: kb.count(k) for k in range(9)}; fz = {k: hist[k] / len(its) for k in (0, 1, 4)}
        desc[cl] = {"base_k_hist": hist, "frozen_share": sum(fz.values()), "model_floor_inband": 0.069 * fz[0] + 0.395 * fz[1] + 0.93 * fz[4],
                    "no_valid_answer_share": {c: sum(1 for i in its if P[c][i].get("mode") is None) / len(its) for c in CKS}}
    strat = {}
    for c in CKS[1:]:
        rs = {}
        for cl in ("SOLVABLE", "UNSOLVABLE"):
            by = {}
            for i in (i for i in sorted(V) if V[i] == cl):
                k = round(8 * P["base"][i]["p_mode"])
                if 2 <= k <= 6: by.setdefault(k, []).append(float(0.25 <= P[c][i]["p_mode"] <= 0.75))
            rs[cl] = by
        ks = [k for k in range(2, 7) if rs["SOLVABLE"].get(k) and rs["UNSOLVABLE"].get(k)]; nU = sum(len(rs["UNSOLVABLE"][k]) for k in ks)
        w = {k: len(rs["UNSOLVABLE"][k]) / nU for k in ks} if nU else {}
        strat[c] = {cl: (sum(w[k] * np.mean(rs[cl][k]) for k in ks) if ks else None) for cl in ("SOLVABLE", "UNSOLVABLE")}
        strat[c]["n_U_matched"] = nU
    obs = share(pool, {cl: pool[cl][1].mean(0) for cl in CL})
    own = share(pool, {cl: traj(pool[cl][0], beta[cl]) for cl in CL})
    tr4 = share(pool, {cl: traj(pool[cl][0], QWEN4B[cl]) for cl in CL})
    single = share(pool, {cl: traj(pool[cl][0], beta["SOLVABLE"]) for cl in CL})
    res["families"][fam] = {"n": {cl: len(pool[cl][0]) for cl in CL}, "beta": beta,
        "beta_ci95": {cl: [float(np.percentile(bs[cl], 2.5)), float(np.percentile(bs[cl], 97.5))] for cl in CL},
        "boot_share_S_gt_U": order_su, "boot_share_S_gt_A_gt_U": order_full, "boot_share_S_gt_max_A_U": order_max, "boot_at_grid_bounds": at_bounds,
        "class_descriptives": desc, "model_free_matched_retention": strat,
        "L-P1": bool(beta["SOLVABLE"] > beta["UNSOLVABLE"] and order_su >= 0.90),
        "L-P2": bool(beta["SOLVABLE"] > beta["AMBIGUOUS"] > beta["UNSOLVABLE"]),
        "in_band_obs": {cl: pool[cl][1].mean(0).round(3).tolist() for cl in CL},
        "illposed_share_obs": obs.round(1).tolist(), "own_law": own.round(1).tolist(), "mae_own_r1_r3": round(float(np.abs(own - obs)[1:].mean()), 2), "mae_own_all": round(float(np.abs(own - obs).mean()), 2),
        "qwen4b_law_transfer": tr4.round(1).tolist(), "mae_transfer_r1_r3": round(float(np.abs(tr4 - obs)[1:].mean()), 2), "mae_transfer_all": round(float(np.abs(tr4 - obs).mean()), 2),
        "single_rate": single.round(1).tolist(), "mae_single_r1_r3": round(float(np.abs(single - obs)[1:].mean()), 2), "mae_single_all": round(float(np.abs(single - obs).mean()), 2)}
    print(fam, json.dumps(res["families"][fam]))
fams = [v for v in res["families"].values()]
res["verdicts"] = {"L-P1": None if any(v is None for v in fams) else all(v["L-P1"] for v in fams),
                   "L-P2": None if any(v is None for v in fams) else all(v["L-P2"] for v in fams)}
print("verdicts", res["verdicts"])
json.dump(res, open(f"{R}/dirA5/results/a9_law_family.json", "w"), indent=1)
