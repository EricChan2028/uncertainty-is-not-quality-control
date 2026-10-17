"""A4c reads: Q1u (confident majority answer, p_mode >= 0.5, on the 412 validated unsolvable GSM-HP twins, plain prompt) per
arm / seed / round for the Qwen3-4B attribution arms, with paired exact McNemar tests (same items) of own_clean and own_sub
against own; lenient accuracy for HC3.   python3 a4_q1u_attrib.py
"""
import glob
import json
import math
import os

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]


def conf(run):
    fs = glob.glob(f"{B}/results/hp/hp_{run}_shard*.jsonl") or glob.glob(f"{B}/results/hp/hpq_{run}_shard*.jsonl")
    if not fs:
        return None
    d = {}
    for f in fs:
        for l in open(f):
            r = json.loads(l)
            if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS:
                d[(str(r["fam"]), int(r["task"]))] = r["p_mode"] >= 0.5
    return d if len(d) >= 412 else None


def acc(run):
    f = f"{B}/results/evals/{run}_step60.summary.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    v = [d[k].get("greedy_acc_lenient", d[k].get("greedy_acc")) for k in DS if k in d]
    return 100 * sum(v) / 3 if len(v) == 3 else None


def mcnemar(a, b):
    keys = [k for k in a if k in b]
    n10 = sum(a[k] and not b[k] for k in keys); n01 = sum(b[k] and not a[k] for k in keys)
    n = n10 + n01
    if n == 0:
        return 1.0, n10, n01
    k = min(n10, n01)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), n10, n01


rows = {}
print(f"{'seed':4s} {'rd':3s} | {'own':>6s} {'clean':>6s} {'sub':>6s} | clean-own  p(McN)  | sub-own  p(McN) | acc own/clean/sub")
for s in range(0, 5):
    for rd in (1, 2, 3):
        own = conf(f"qwen3_4b_Base_own_a3_s{s}_r{rd}")
        cl = conf(f"qwen3_4b_Base_own_clean_a3_s{s}_r{rd}")
        sb = conf(f"qwen3_4b_Base_own_sub_a3_s{s}_r{rd}")
        if own is None or (cl is None and sb is None):
            continue
        q = lambda d: None if d is None else sum(d.values()) / len(d)
        line = f"s{s:<3d} r{rd:<2d} | {q(own):6.3f} " + (f"{q(cl):6.3f} " if cl else "     - ") + (f"{q(sb):6.3f}" if sb else "     -") + " | "
        rec = {"own": q(own), "clean": q(cl), "sub": q(sb)}
        if cl:
            p, n10, n01 = mcnemar(cl, own); rec["clean_minus_own"] = q(cl) - q(own); rec["p_clean"] = p
            line += f"{q(cl)-q(own):+7.3f}  {p:7.4f} | "
        else:
            line += "      -        - | "
        if sb:
            p, n10, n01 = mcnemar(sb, own); rec["sub_minus_own"] = q(sb) - q(own); rec["p_sub"] = p
            line += f"{q(sb)-q(own):+7.3f} {p:7.4f} | "
        else:
            line += "     -       - | "
        a = [acc(f"qwen3_4b_Base_{arm}_a3_s{s}_r{rd}") for arm in ("own", "own_clean", "own_sub")]
        rec["acc"] = a
        line += "/".join("-" if x is None else f"{x:.1f}" for x in a)
        print(line)
        rows[f"s{s}_r{rd}"] = rec
r3 = {k: v for k, v in rows.items() if k.endswith("_r3") and v.get("clean_minus_own") is not None}
hc1 = [v["clean_minus_own"] <= -0.05 for v in r3.values()]
print(f"\nround 3: clean-own <= -0.05 in {sum(hc1)}/{len(hc1)} seeds; McNemar p<0.05 in {sum(v['p_clean'] < 0.05 for v in r3.values())}/{len(r3)}")
os.makedirs(f"{R}/dirA4/results", exist_ok=True)
json.dump(rows, open(f"{R}/dirA4/results/q1u_attrib.json", "w"), indent=1)
