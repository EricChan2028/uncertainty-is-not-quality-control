"""A4k (Phi-4-mini replication of A4g) reads HU1-HU3 (+ HU5 descriptive). rescue_uv = own r2 solver + own r3 admitted set, reward own_uv ("unknown"
votes). Comparators: own r3 (same start, same data, own reward), rescue_abs r3 (A4f), own_clean r3. Probe files must be
complete (all 1,086 rows over shards) or the read is None.   python3 a4_uv_read.py
HU4: fully audited pools; abstention prompt; admission = band [0.25, 0.75] on the majority share counting "unknown" as an
answer (max(p_mode, abstain) in the guard records). Seed 0 -> pool s0 (own comparator qwen_s0_ag/r3), seed 1 -> pool s1
(qwen_s1_ag/r3).
"""
import glob, json, os
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB"; GD = f"{R}/dirA3/results/guard_diag"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]
def rows(tag, run):
    d = [json.loads(l) for f in glob.glob(f"{B}/results/hp/{tag}_{run}_shard*.jsonl") for l in open(f)]
    return d if len(d) >= 1086 else None
def q1u(run):
    d = rows("hp", run) or rows("hpq", run)
    if not d: return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(r["p_mode"] >= 0.5 for r in u) / len(u)
def abst(run):
    d = rows("hpabs", run)
    if not d: return None, None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]; i = [r for r in d if r["form"] == "intact"]
    return sum(r.get("abstain", 0) for r in u) / len(u), sum(r.get("abstain", 0) for r in i) / len(i)
def acc(run, strict=False):
    f = f"{B}/results/evals/{run}_step60.summary.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); v = [d[k]["greedy_acc"] if strict else d[k].get("greedy_acc_lenient", d[k]["greedy_acc"]) for k in DS if k in d]
    return 100 * sum(v) / 3 if len(v) == 3 else None
f1 = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
res = {"seeds": {}, "pools": {}}
print("seed | arm        | acc(len) acc(strict) | Q1u   | abstain uns / intact (abs prompt)")
for s in (0, 1):
    arms = {"own": f"phi4mini_own_a4_s{s}_r3", "rescue_uv": f"phi4mini_rescue_uv_a4_s{s}_r3",
            "rescue_abs": f"phi4mini_rescue_abs_a4_s{s}_r3", "own_clean": f"phi4mini_own_clean_a4_s{s}_r3"}
    r = {}
    for a, run in arms.items():
        au, ai = abst(run); r[a] = {"acc": acc(run), "acc_strict": acc(run, True), "q1u": q1u(run), "abs_uns": au, "abs_int": ai}
        print(f"s{s}   | {a:10s} | {f1(r[a]['acc'], 1):5s}   {f1(r[a]['acc_strict'], 1):5s}      | {f1(r[a]['q1u'])} | {f1(au)} / {f1(ai)}")
    o, x = r["own"], r["rescue_uv"]
    hu1 = None if x["q1u"] is None or o["q1u"] is None else x["q1u"] <= o["q1u"] - 0.05
    hu2 = None if x["acc"] is None or o["acc"] is None else x["acc"] >= o["acc"] - 2
    hu3 = None if x["abs_uns"] is None or o["abs_uns"] is None else (x["abs_uns"] >= o["abs_uns"] + 0.10 and x["abs_int"] <= 0.15)
    print(f"  -> seed {s}: HU1 {hu1} | HU2 {hu2} | HU3 {hu3}")
    res["seeds"][s] = {"arms": r, "HU1": hu1, "HU2": hu2, "HU3": hu3}
json.dump(res, open(f"{R}/dirA4/results/uv_read_phi.json", "w"), indent=1)
