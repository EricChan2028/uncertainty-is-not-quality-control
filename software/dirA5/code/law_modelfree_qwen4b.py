"""DESCRIPTIVE (not registered; 25 Sep): the A9 model-free check applied to the ORIGINAL Qwen3-4B law data (fixed pools qwen_s0_r1
and qwen_s1_r1 scored by base and own r1..r5). At matched base agreement (strata k = 8 p_mode in 2..6, standardised to the
UNSOLVABLE stratum weights), in-band retention of SOLVABLE vs UNSOLVABLE (and AMBIGUOUS) items per checkpoint. Motivation: a
reviewer showed that model misspecification alone can order fitted per-class betas; this check does not use the model.
python3 law_modelfree_qwen4b.py"""
import glob, json
import numpy as np
R = "/home/tione/notebook/rsi_rl"; CKS = ["base", "r1", "r2", "r3", "r4", "r5"]; CL = ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")
out = {}
for tag, pool in (("qwen_s0", "qwen_s0_r1"), ("qwen_s1", "qwen_s1_r1")):
    V = {}
    for f in sorted(glob.glob(f"{R}/dirA3/results/pool_audit/{pool}/parts/*.json")):
        for v in json.load(open(f))["verdicts"]: V[v["id"]] = v["verdict"]
    P = {c: {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}/{c}/guard_records.jsonl")) if r["kind"] == "proposed"} for c in CKS}
    ids = sorted(i for i in V if V[i] in CL and all(i in P[c] for c in CKS))
    by = {cl: {} for cl in CL}
    for i in ids:
        k = round(8 * P["base"][i]["p_mode"])
        if 2 <= k <= 6: by[V[i]].setdefault(k, []).append(i)
    ks = [k for k in range(2, 7) if all(by[cl].get(k) for cl in CL)]; nU = sum(len(by["UNSOLVABLE"][k]) for k in ks)
    w = {k: len(by["UNSOLVABLE"][k]) / nU for k in ks}
    res = {"n_matched": {cl: sum(len(by[cl][k]) for k in ks) for cl in CL}, "strata": ks, "retention": {}}
    for c in CKS[1:]:
        res["retention"][c] = {cl: round(float(sum(w[k] * np.mean([0.25 <= P[c][i]["p_mode"] <= 0.75 for i in by[cl][k]]) for k in ks)), 3) for cl in CL}
    out[tag] = res; print(tag, json.dumps(res))
json.dump(out, open(f"{R}/dirA5/results/law_modelfree_qwen4b.json", "w"), indent=1)
