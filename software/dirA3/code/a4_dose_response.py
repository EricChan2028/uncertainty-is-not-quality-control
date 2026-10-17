"""Dose-response of the harm (A4, exploratory): for every Qwen3-4B lineage round of the A3 arms (own, own_clean, own_sub,
own_dirty, own_tg, own_azr), the ill-posed fraction of the round's TRAINING set is estimated from the audits (proposals:
audited ill-posed share of that run's admitted set, 0 for own_clean, 1 for own_dirty; deleted tracers count as ill-posed;
intact tracers as well-posed); the dose is the cumulative sum over rounds 1..n. Q1u (plain prompt, 412 validated unsolvable
twins) at round n is regressed on round n (sharpening) and dose: Q1u = a + b * n + c * dose. Lineage-bootstrap CI for c.
  python3 a4_dose_response.py
"""
import glob
import json
import os
import re

import numpy as np

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}


def q1u(run):
    fs = glob.glob(f"{B}/results/hp/hp_{run}_shard*.jsonl")
    if not fs:
        return None
    d = [json.loads(l) for f in fs for l in open(f)]
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(r["p_mode"] >= 0.5 for r in u) / len(u) if len(u) >= 400 else None


def audited_share(run):
    for name in ("audit_full.json", "audit_admitted.json"):
        f = f"{B}/runs/{run}/{name}"
        if os.path.exists(f):
            c = json.load(open(f))["counts"]; n = sum(c.values())
            return (n - c.get("SOLVABLE", 0)) / n
    return None


def train_frac(run, arm, rd, seed):
    d = f"{B}/runs/{run}"
    fn = {"own_sub": "admitted_sub.jsonl", "own_clean": "admitted_clean.jsonl", "own_dirty": "admitted_dirty.jsonl"}.get(arm, "admitted.jsonl")
    if not os.path.exists(f"{d}/{fn}"):
        return None
    rows = [json.loads(l) for l in open(f"{d}/{fn}")]
    npro = sum(r["kind"] == "proposed" for r in rows); ndel = sum(r["kind"] == "tracer_deleted" for r in rows)
    if arm == "own_clean":
        s = 0.0
    elif arm == "own_dirty":
        s = 1.0
    else:
        s = audited_share(run)
        if s is None and rd == 1:          # round-1 sets of sub reuse the own round-1 proposals
            s = audited_share(f"qwen3_4b_Base_own_a3_s{seed}_r1")
        if s is None:
            return None
    return (npro * s + ndel) / max(1, len(rows))


pts = []
for L in sorted({re.sub(r"_r\d+$", "", os.path.basename(p)) for p in glob.glob(f"{B}/runs/qwen3_4b_Base_*_a3_s*_r*") if not p.endswith(".claim")}):
    m = re.match(r"qwen3_4b_Base_(own(?:_\w+?)?)_a3_s(\d+)$", L)
    if not m:
        continue
    arm, seed = m.group(1), int(m.group(2))
    dose = 0.0
    for rd in range(1, 7):
        run = f"{L}_r{rd}"
        if not os.path.isdir(f"{B}/runs/{run}"):
            break
        f = train_frac(run, arm, rd, seed)
        if f is None:
            break
        dose += f
        q = q1u(run)
        if q is not None:
            pts.append({"lineage": L, "arm": arm, "seed": seed, "round": rd, "frac": f, "dose": dose, "q1u": q})
print(f"{len(pts)} lineage-rounds with dose and Q1u")
for p in pts:
    print(f"  {p['lineage']:36s} r{p['round']}  train ill-posed frac {p['frac']:.2f}  cumulative dose {p['dose']:.2f}  Q1u {p['q1u']:.3f}")
X = np.array([[1, p["round"], p["dose"]] for p in pts]); y = np.array([p["q1u"] for p in pts])
coef = np.linalg.lstsq(X, y, rcond=None)[0]
lins = sorted({p["lineage"] for p in pts}); rng = np.random.RandomState(0); bs = []
for _ in range(2000):
    pick = rng.choice(lins, len(lins), replace=True)
    rows = [p for L in pick for p in pts if p["lineage"] == L]
    Xb = np.array([[1, p["round"], p["dose"]] for p in rows]); yb = np.array([p["q1u"] for p in rows])
    if np.linalg.matrix_rank(Xb) == 3:
        bs.append(np.linalg.lstsq(Xb, yb, rcond=None)[0])
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
print(f"\nQ1u = {coef[0]:.3f} + {coef[1]:.3f} * round + {coef[2]:.3f} * dose")
print(f"  round coefficient 95% CI [{lo[1]:.3f}, {hi[1]:.3f}]; dose coefficient 95% CI [{lo[2]:.3f}, {hi[2]:.3f}] (lineage bootstrap, {len(lins)} lineages)")
json.dump({"points": pts, "coef": coef.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist()}, open(f"{R}/dirA4/results/dose_response.json", "w"), indent=1)
