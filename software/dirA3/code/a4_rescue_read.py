"""A4f registered reads HA1-HA3: rescue_abs (own r2 solver + own r3 admitted set, abstention-aware reward) vs own r3 (same
start, same data, own reward) and own_clean r3, per seed. Q1u = confident majority (p_mode >= 0.5) on the 412 validated
unsolvable twins, plain prompt; abstention under the abstention prompt on unsolvable and intact twins (hpabs files);
lenient mean accuracy (MATH-500 / held-out-1000 / GSM8K).   python3 a4_rescue_read.py
"""
import glob
import json
import os

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]


def rows(tag, run):
    fs = glob.glob(f"{B}/results/hp/{tag}_{run}_shard*.jsonl")
    return [json.loads(l) for f in fs for l in open(f)] if fs else None


def q1u(run):
    d = rows("hp", run) or rows("hpq", run)
    if not d or len(d) < 1086:
        return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(r["p_mode"] >= 0.5 for r in u) / len(u) if len(u) >= 400 else None


def abst(run):
    d = rows("hpabs", run)
    if not d or len(d) < 1086:
        return None, None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    i = [r for r in d if r["form"] == "intact"]
    return (sum(r.get("abstain", 0) for r in u) / max(1, len(u)), sum(r.get("abstain", 0) for r in i) / max(1, len(i)))


def acc(run):
    f = f"{B}/results/evals/{run}_step60.summary.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    v = [d[k].get("greedy_acc_lenient", d[k].get("greedy_acc")) for k in DS if k in d]
    return 100 * sum(v) / 3 if len(v) == 3 else None


f1 = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
print("seed | arm        | acc   | Q1u   | abstain uns / intact (abs prompt)")
res = {}
for s in (0, 1, 2):
    arms = {"own": f"qwen3_4b_Base_own_a3_s{s}_r3", "rescue_abs": f"qwen3_4b_Base_rescue_abs_a4_s{s}_r3", "own_clean": f"qwen3_4b_Base_own_clean_a3_s{s}_r3"}
    r = {}
    for a, run in arms.items():
        au, ai = abst(run)
        r[a] = {"acc": acc(run), "q1u": q1u(run), "abs_uns": au, "abs_int": ai}
        print(f"s{s}   | {a:10s} | {f1(r[a]['acc'], 1):5s} | {f1(r[a]['q1u'])} | {f1(au)} / {f1(ai)}")
    o, x = r["own"], r["rescue_abs"]
    ha1 = None if x["q1u"] is None or o["q1u"] is None else x["q1u"] <= o["q1u"] - 0.05
    ha2 = None if x["acc"] is None or o["acc"] is None else x["acc"] >= o["acc"] - 2
    ha3 = None if x["abs_uns"] is None or o["abs_uns"] is None else (x["abs_uns"] >= o["abs_uns"] + 0.10 and x["abs_int"] <= 0.15)
    print(f"  -> seed {s}: HA1 {ha1} | HA2 {ha2} | HA3 {ha3}")
    res[s] = {"arms": r, "HA1": ha1, "HA2": ha2, "HA3": ha3}
json.dump(res, open(f"{R}/dirA4/results/rescue_read.json", "w"), indent=1)
