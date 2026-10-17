"""A6 X read (registered 09:32). Q1u = share of the 412 validated unsolvable twins answered confidently (p_mode >= 0.5,
plain prompt; as a4_rescue_read.q1u), from complete probe files only (1,086 rows over shards).
X1: Q1u(CLN1) <= Q1u(RND1) - 0.04 in >= 4 of 5 Qwen3-4B seeds AND pooled paired McNemar p < 0.01 (items paired within seed,
discordant counts summed over seeds, exact two-sided binomial test). X3: |Q1u(RND1) - Q1u(own r3)| <= 0.04 in >= 3 of 5.
X4: lenient mean accuracy (MATH-500 / held-out-1000 / GSM8K) of CLN1 and RND1 within 2 points in every seed.
python3 a6_x_read.py"""
import glob, json, math, os
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]
def conf(run):
    d = [json.loads(l) for f in glob.glob(f"{B}/results/hp/hp_{run}_shard*.jsonl") for l in open(f)]
    if len(d) < 1086: return None
    u = {(str(r["fam"]), int(r["task"])): r["p_mode"] >= 0.5 for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS}
    return u if len(u) == 412 else None
def acc(run):
    f = f"{B}/results/evals/{run}_step60.summary.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); v = [d[k].get("greedy_acc_lenient", d[k]["greedy_acc"]) for k in DS if k in d]
    return 100 * sum(v) / 3 if len(v) == 3 else None
def binom_two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)
res = {"seeds": {}}; bsum = csum = 0; x1 = []; x3 = []; x4 = []
for s in range(5):
    c1, r1, ow = conf(f"qwen3_4b_Base_cln1_a6_s{s}_r3"), conf(f"qwen3_4b_Base_rnd1_a6_s{s}_r3"), conf(f"qwen3_4b_Base_own_a3_s{s}_r3")
    a1, a2 = acc(f"qwen3_4b_Base_cln1_a6_s{s}_r3"), acc(f"qwen3_4b_Base_rnd1_a6_s{s}_r3")
    rec = {}
    if c1 and r1:
        q1, q2 = sum(c1.values()) / 412, sum(r1.values()) / 412
        b = sum(1 for k in c1 if r1[k] and not c1[k]); c = sum(1 for k in c1 if c1[k] and not r1[k])
        bsum += b; csum += c
        rec.update({"Q1u_cln1": q1, "Q1u_rnd1": q2, "discordant_rnd_only": b, "discordant_cln_only": c}); x1.append(q1 <= q2 - 0.04 + 1e-9)
        if ow: qo = sum(ow.values()) / 412; rec["Q1u_own"] = qo; x3.append(abs(q2 - qo) <= 0.04 + 1e-9)
    if a1 is not None and a2 is not None:
        rec.update({"acc_cln1": a1, "acc_rnd1": a2}); x4.append(abs(a1 - a2) <= 2 + 1e-9)
    res["seeds"][s] = rec; print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()})
p = binom_two_sided(bsum, csum)
res["pooled_mcnemar"] = {"rnd_only": bsum, "cln_only": csum, "p": p}
res["verdicts"] = {"X1": None if len(x1) < 5 else (sum(x1) >= 4 and p < 0.01 and bsum > csum), "X3": None if len(x3) < 5 else sum(x3) >= 3,
                   "X4": None if len(x4) < 5 else all(x4), "X1_counts": [sum(x1), len(x1)], "X3_counts": [sum(x3), len(x3)]}
print("pooled McNemar", res["pooled_mcnemar"], "verdicts", res["verdicts"])
json.dump(res, open(f"{R}/dirA5/results/a6_x_read.json", "w"), indent=1)
