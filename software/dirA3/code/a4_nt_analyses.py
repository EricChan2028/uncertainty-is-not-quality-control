"""No-training analyses of the noisy-TV mechanism on an audited fixed pool (A4; exploratory, disclosed as such):
 (1) best label-free guard: 5-fold cross-validated logistic classifier P(SOLVABLE | label-free features) fitted per checkpoint on
     gradient-bearing candidates (1/8 <= p_mode <= 7/8); admit the top q of them by score; ill-posed share at q = 25 / 50 %.
     Features: plain p_mode, abstention-prompt p_mode and abstain share, mode agreement across prompts, question length,
     count of numbers in the question, program-closure stats if available.
 (2) learning progress: admit gradient-bearing candidates with the largest agreement change since the previous checkpoint
     (ALP-style guard); mean agreement gain per class (spurious consensus); what the modal answer on UNSOLVABLE items is.
 (3) one-parameter sharpening model: q_t(a) proportional to q_0(a)^(beta^t) with the base answer distribution approximated from
     the base p_mode (mode share q0, remaining mass split over r = ceil((1-q0)/q0) answers); beta fitted on SOLVABLE items only
     (in-band mass), then used to predict the AMBIGUOUS / UNSOLVABLE in-band masses and the admitted ill-posed share.
  python3 a4_nt_analyses.py [pool_tag=qwen_s0] [audit_dir] [program_closure.jsonl]
"""
import glob
import json
import math
import re
import sys
from math import comb

import numpy as np

R = "/home/tione/notebook/rsi_rl"
TAG = sys.argv[1] if len(sys.argv) > 1 else "qwen_s0"
AUD = sys.argv[2] if len(sys.argv) > 2 else f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts"
PCF = sys.argv[3] if len(sys.argv) > 3 else None
POOLF = {"qwen_s0": f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl", "qwen_s1": f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s1_r1/proposed.jsonl"}[TAG]
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
V = {}
for f in glob.glob(f"{AUD}/*.json"):
    for v in json.load(open(f))["verdicts"]:
        V[v["id"]] = v["verdict"]
TXT = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open(POOLF)}


def load(d, ck):
    try:
        return {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{d}/{ck}/guard_records.jsonl")) if r["kind"] == "proposed"}
    except FileNotFoundError:
        return None


P = {c: load(TAG, c) for c in CKS}
A = {c: load(TAG + "_ag", c) for c in CKS}
PC = {json.loads(l)["id"]: json.loads(l) for l in open(PCF)} if PCF else {}
ids = [i for i in V if i in TXT and P["base"] and i in P["base"]]
y_all = {i: 1.0 if V[i] == "SOLVABLE" else 0.0 for i in ids}
out = {"pool": TAG}


def feats(i, c):
    p, a = P[c][i], A[c][i]
    t = TXT[i]
    f = [p["p_mode"], a["p_mode"], a["abstain"], 1.0 if (p["mode"] is not None and p["mode"] == a["mode"]) else 0.0,
         len(t.split()) / 100, len(re.findall(r"\d[\d,]*\.?\d*", t)) / 10]
    if PC:
        r = PC.get(i, {})
        f += [r.get("n_closed", 0) / 4, r.get("n_ok", 0) / 4, r.get("agree_closed", 0) / 4, r.get("any_foreign", 0) / 4]
    return f


def logreg(X, y, l2=1.0, it=400, lr=0.5):
    X = np.hstack([X, np.ones((len(X), 1))]); w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) / len(y) + l2 * w / len(y); w -= lr * g
    return w


def cv_scores(X, y, k=5, seed=0):
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); s = np.zeros(len(y))
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xs = (X - mu) / sd
    for f in range(k):
        te = idx[f::k]; tr = np.setdiff1d(idx, te)
        w = logreg(Xs[tr], y[tr]); s[te] = np.hstack([Xs[te], np.ones((len(te), 1))]) @ w
    return s


print(f"pool {TAG}: {len(ids)} audited candidates; program-closure features: {'yes' if PC else 'no'}")
print("(1) best label-free guard (CV logistic, gradient-bearing candidates): ill-posed% of top 25% / top 50% by score | AUC | plain band")
res1 = {}
for c in CKS:
    if P[c] is None or A[c] is None:
        continue
    gb = [i for i in ids if i in A[c] and 0.125 <= P[c][i]["p_mode"] <= 0.875]
    if len(gb) < 50:
        continue
    X = np.array([feats(i, c) for i in gb]); y = np.array([y_all[i] for i in gb])
    s = cv_scores(X, y)
    order = np.argsort(-s)
    top = lambda q: 100 * (1 - y[order[:max(1, int(q * len(gb)))]].mean())
    pos, neg = s[y == 1], s[y == 0]
    auc = float(np.mean([(pp > nn) + 0.5 * (pp == nn) for pp in pos[:400] for nn in neg[:400]])) if len(pos) and len(neg) else float("nan")
    band = [i for i in ids if 0.25 <= P[c][i]["p_mode"] <= 0.75]
    res1[c] = {"n_gb": len(gb), "top25": top(.25), "top50": top(.5), "auc": auc, "band": 100 * sum(V[i] != "SOLVABLE" for i in band) / len(band), "solvable_gb": int(y.sum())}
    print(f"  {c:4s} n_gb {len(gb):4d} (solvable {int(y.sum()):3d}) | top25 {res1[c]['top25']:5.1f}  top50 {res1[c]['top50']:5.1f} | AUC {auc:.3f} | band {res1[c]['band']:5.1f}")
out["best_guard"] = res1

print("(2) learning progress / spurious consensus")
res2 = {}
for k in range(1, len(CKS)):
    c, pc = CKS[k], CKS[k - 1]
    if P[c] is None or P[pc] is None:
        continue
    gb = [i for i in ids if 0.125 <= P[c][i]["p_mode"] <= 0.875]
    dp = {i: P[c][i]["p_mode"] - P[pc][i]["p_mode"] for i in gb}
    top = sorted(gb, key=lambda i: -abs(dp[i]))[: max(1, len(gb) // 2)]
    pos = [i for i in gb if dp[i] > 0]
    res2[c] = {"alp_top50_illposed": 100 * sum(V[i] != "SOLVABLE" for i in top) / len(top),
               "progress_pos_illposed": 100 * sum(V[i] != "SOLVABLE" for i in pos) / max(1, len(pos))}
    print(f"  {c:4s} ALP guard (top half |dp|) ill-posed {res2[c]['alp_top50_illposed']:5.1f}% | positive-progress guard {res2[c]['progress_pos_illposed']:5.1f}%")
gain = {cl: np.mean([P["r5"][i]["p_mode"] - P["base"][i]["p_mode"] for i in ids if V[i] == cl and P["r5"] and i in P["r5"]]) for cl in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")} if P["r5"] else {}
print("  mean agreement gain base -> r5 by class:", {k: round(float(v), 3) for k, v in gain.items()})
unsolv = [i for i in ids if V[i] == "UNSOLVABLE" and P["r5"] and P["r5"][i]["p_mode"] >= 0.5 and P["r5"][i]["mode"] is not None]
def in_text(m, t):
    try:
        v = float(m)
    except Exception:
        return False
    return any(abs(v - float(x.replace(",", ""))) < 1e-9 for x in re.findall(r"\d[\d,]*\.?\d*", t) if x.replace(",", "").replace(".", "", 1).isdigit())
cat = {"copied from the question": 0, "zero": 0, "other invented number": 0, "non-numeric": 0}
for i in unsolv:
    m = P["r5"][i]["mode"]
    try:
        v = float(m)
        if v == 0: cat["zero"] += 1
        elif in_text(m, TXT[i]): cat["copied from the question"] += 1
        else: cat["other invented number"] += 1
    except Exception:
        cat["non-numeric"] += 1
stable = sum(1 for i in unsolv if P["base"][i]["mode"] == P["r5"][i]["mode"])
print(f"  confident (r5 p_mode>=0.5) UNSOLVABLE items: {len(unsolv)}; modal answer: {cat}; same mode as base: {stable} ({100*stable/max(1,len(unsolv)):.0f}%)")
out["progress"] = res2; out["gain"] = {k: float(v) for k, v in gain.items()}; out["modal_answers"] = cat; out["mode_stable_from_base"] = stable / max(1, len(unsolv))

print("(3) one-parameter sharpening model, beta fitted on SOLVABLE in-band mass only")
def inband_prob(q, G=8, lo=0.25, hi=0.75):
    return sum(comb(G, k) * q ** k * (1 - q) ** (G - k) for k in range(G + 1) if lo <= k / G <= hi)
def sharpen(q0, gam):
    q0 = min(max(q0, 1e-3), 1 - 1e-3); r = max(1, math.ceil((1 - q0) / q0 - 1e-9)); o = (1 - q0) / r
    a = q0 ** gam; b = r * o ** gam
    return a / (a + b)
q0 = {i: (P["base"][i]["p_mode"] * 8 + 0.5) / 9 for i in ids}
obs = {cl: [np.mean([0.25 <= P[c][i]["p_mode"] <= 0.75 for i in ids if V[i] == cl and i in P[c]]) if P[c] else None for c in CKS] for cl in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
def model_mass(cl, beta):
    return [np.mean([inband_prob(sharpen(q0[i], beta ** t)) for i in ids if V[i] == cl]) for t in range(len(CKS))]
best = None
for b in np.arange(1.0, 3.01, 0.02):
    m = model_mass("SOLVABLE", b)
    err = sum((m[t] - obs["SOLVABLE"][t]) ** 2 for t in range(1, len(CKS)) if obs["SOLVABLE"][t] is not None)
    if best is None or err < best[0]:
        best = (err, b)
beta = best[1]
pi = {cl: sum(V[i] == cl for i in ids) / len(ids) for cl in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
mm = {cl: model_mass(cl, beta) for cl in pi}
share = lambda m, t: 100 * sum(pi[c] * m[c][t] for c in ("AMBIGUOUS", "UNSOLVABLE")) / sum(pi[c] * m[c][t] for c in pi)
pred = [share(mm, t) for t in range(len(CKS))]
obs_share = [share(obs, t) if all(obs[c][t] is not None for c in pi) else None for t in range(len(CKS))]
print(f"  beta = {beta:.2f} (fit on SOLVABLE only)")
for cl in pi:
    print(f"  {cl:10s} in-band mass observed {[None if x is None else round(float(x), 3) for x in obs[cl]]}  predicted {[round(float(x), 3) for x in mm[cl]]}")
print(f"  admitted ill-posed share observed {[None if x is None else round(x, 1) for x in obs_share]}  predicted {[round(x, 1) for x in pred]}")
out["sharpening"] = {"beta": float(beta), "observed_share": obs_share, "predicted_share": pred, "obs_mass": {k: [None if x is None else float(x) for x in v] for k, v in obs.items()}, "pred_mass": {k: [float(x) for x in v] for k, v in mm.items()}}
json.dump(out, open(f"{R}/dirA4/results/nt_analyses_{TAG}{'_pc' if PC else ''}.json", "w"), indent=1)
