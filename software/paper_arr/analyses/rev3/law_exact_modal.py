"""[EXP, robustness, 2026-10-04] Does the class-conditional sharpening model survive replacing the binomial proxy
P(2 <= Bin(8, q_mode) <= 6) by the exact probability that the MODAL COUNT of 8 samples lies in 2..6?

Same data, reconstruction, grid, fit window and bootstrap as dirA3/code/a4_law_stable.py / a4_law_bootstrap.py.
The base answer distribution is reconstructed as in the paper: mode q0 = (8 p0 + 0.5) / 9 and r = ceil((1 - q0) / q0)
competitors of equal mass; sharpening q_t(a) ~ q_0(a)^(beta^t) keeps that shape (one mode + r equal competitors).

Two ways of turning the sharpened distribution into an in-band probability bracket the truth:
  proxy  : P(2 <= K <= 6), K ~ Bin(8, q_mode). This is exact if the competing mass is spread over many answers that never
           repeat (diffuse errors), and it is what the paper reports.
  exact  : P(2 <= M <= 6), M = largest count among the 1 + r answers of a multinomial(8) draw. With the fewest competitors
           compatible with q0 being the mode, this is the most concentrated reconstruction.
Variants are reported on all problems and without problems whose reconstructed base mode is tied (they never sharpen).
python3 law_exact_modal.py [B=500]
"""
import glob
import json
import math
import sys

import numpy as np

R = "/home/tione/notebook/rsi_rl"
AUD = {"qwen_s0": "qwen_s0_r1", "qwen_s1": "qwen_s1_r1"}
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
CL = ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")
GRID = np.round(np.arange(1.0, 8.0001, 0.05), 2)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N = 8


def load(tag):
    V = {}
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/{AUD[tag]}/parts/*.json"):
        for v in json.load(open(f))["verdicts"]:
            V[v["id"]] = v["verdict"]
    P = {c: {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}/{c}/guard_records.jsonl"))
             if r["kind"] == "proposed"} for c in CKS}
    ids = [i for i in V if i in P["base"] and V[i] in CL]
    out = {}
    for cl in CL:
        its = [i for i in ids if V[i] == cl]
        q0 = np.array([(P["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in its])
        inb = np.array([[float(0.25 <= P[c][i]["p_mode"] <= 0.75) if i in P[c] else np.nan for c in CKS] for i in its])
        out[cl] = (q0, inb)
    return out


# U[m, r, c]: probability that m balls thrown uniformly into r bins leave every bin with at most c balls
RMAX = 40
U = np.zeros((N + 1, RMAX + 1, N + 1))
for r in range(1, RMAX + 1):
    for c in range(N + 1):
        poly = np.zeros(N + 1)
        poly[0] = 1.0
        base = np.array([1.0 / math.factorial(j) if j <= c else 0.0 for j in range(N + 1)])
        for _ in range(r):
            poly = np.convolve(poly, base)[: N + 1]
        for m in range(N + 1):
            U[m, r, c] = math.factorial(m) * poly[m] / (r ** m)
BINC = np.array([math.comb(N, k) for k in range(N + 1)], dtype=float)


def recon(q0):
    q0 = np.clip(q0, 1e-3, 1 - 1e-3)
    r = np.maximum(1, np.ceil((1 - q0) / q0 - 1e-9)).astype(int)
    o = (1 - q0) / r
    return q0, r, o


def sharp(q0, r, o, g):
    return 1 / (1 + np.exp(np.log(r) + g * np.log(o) - g * np.log(q0)))


def inband_proxy(q, r):
    k = np.arange(2, 7)
    return (BINC[k][None, :] * q[:, None] ** k[None, :] * (1 - q[:, None]) ** (N - k)[None, :]).sum(1)


def inband_exact(q, r):
    rr = np.minimum(r, RMAX)
    k = np.arange(N + 1)
    pk = BINC[None, :] * q[:, None] ** k[None, :] * (1 - q[:, None]) ** (N - k)[None, :]  # mode count distribution

    def le(c):  # P(M <= c)
        u = U[(N - k)[None, :], rr[:, None], c]  # competitors all <= c given N-k of them
        return (pk * (k[None, :] <= c) * u).sum(1)
    return le(6) - le(1)


IB = {"proxy": inband_proxy, "exact": inband_exact}


def traj(q0, g_beta, how):
    q0_, r, o = recon(q0)
    return np.array([IB[how](sharp(q0_, r, o, g_beta ** t), r).mean() for t in range(6)])


def table(q0, how):
    q0_, r, o = recon(q0)
    return np.array([[IB[how](sharp(q0_, r, o, b ** t), r).mean() for t in range(6)] for b in GRID])


def fit(q0, inb, how):
    obs = np.nanmean(inb, 0)
    T = table(q0, how)
    e = ((T[:, 1:] - obs[None, 1:]) ** 2).sum(1)
    return float(GRID[int(np.argmin(e))])


def share(pi, m):
    num = pi["AMBIGUOUS"] * m["AMBIGUOUS"] + pi["UNSOLVABLE"] * m["UNSOLVABLE"]
    return 100 * num / (num + pi["SOLVABLE"] * m["SOLVABLE"])


def untied(pool):
    out = {}
    for cl, (q0, inb) in pool.items():
        q0_, r, o = recon(q0)
        keep = o < q0_ - 1e-9
        out[cl] = (q0[keep], inb[keep])
    return out


def run(tr, te, how, label, boot):
    beta = {cl: fit(*tr[cl], how) for cl in CL}
    res = {"beta": beta}
    for tag, pool in (("train_s0", tr), ("heldout_s1", te)):
        n = {cl: len(pool[cl][0]) for cl in CL}
        pi = {cl: n[cl] / sum(n.values()) for cl in CL}
        obs = share(pi, {cl: np.nanmean(pool[cl][1], 0) for cl in CL})
        pred = share(pi, {cl: traj(pool[cl][0], beta[cl], how) for cl in CL})
        single = share(pi, {cl: traj(pool[cl][0], beta["SOLVABLE"], how) for cl in CL})
        res[tag] = {"observed": obs.round(1).tolist(), "per_class": pred.round(1).tolist(),
                    "mae_per_class": round(float(np.abs(pred - obs).mean()), 2),
                    "single_rate": single.round(1).tolist(), "mae_single": round(float(np.abs(single - obs).mean()), 2),
                    "n": n}
    if boot:
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(B):
            be = {}
            for cl in CL:
                q0, inb = tr[cl]
                idx = rng.integers(0, len(q0), len(q0))
                be[cl] = fit(q0[idx], inb[idx], how)
            bs.append([be[c] for c in CL])
        bs = np.array(bs)
        ci = lambda x: [round(float(np.percentile(x, 2.5)), 2), round(float(np.percentile(x, 97.5)), 2)]
        res["ci95"] = {f"beta_{c}": ci(bs[:, j]) for j, c in enumerate(CL)}
        res["share_reps_beta_S_gt_max_A_U"] = float(np.mean(bs[:, 0] > np.maximum(bs[:, 1], bs[:, 2])))
    print(label, json.dumps(res))
    return res


if __name__ == "__main__":
    tr, te = load("qwen_s0"), load("qwen_s1")
    out = {}
    out["proxy_all"] = run(tr, te, "proxy", "proxy_all", boot=False)          # must reproduce the paper
    out["exact_all"] = run(tr, te, "exact", "exact_all", boot=True)
    out["proxy_untied"] = run(untied(tr), untied(te), "proxy", "proxy_untied", boot=False)
    out["exact_untied"] = run(untied(tr), untied(te), "exact", "exact_untied", boot=True)
    # how far apart the two in-band probabilities are, by class, at base
    gap = {}
    for cl in CL:
        q0_, r, o = recon(tr[cl][0])
        gap[cl] = round(float(np.mean(inband_exact(q0_, r) - inband_proxy(q0_, r))), 4)
    out["mean_exact_minus_proxy_at_base_s0"] = gap
    out["check_example_0.4_0.3_0.3"] = {"exact": float(inband_exact(np.array([0.4]), np.array([2]))[0]),
                                       "proxy": float(inband_proxy(np.array([0.4]), np.array([2]))[0])}
    json.dump(out, open(f"{R}/paper_arr/analyses/rev3/law_exact_modal_2026-10-04.json", "w"), indent=1)
    print("saved")
