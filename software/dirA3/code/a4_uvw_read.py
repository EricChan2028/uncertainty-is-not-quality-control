"""A4l reads HW1-HW4 (= HU1-HU4 thresholds) for rescue_uvw (w = 3); rescue_uv = own r2 solver + own r3 admitted set, reward own_uv ("unknown"
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
    arms = {"own": f"qwen3_4b_Base_own_a3_s{s}_r3", "rescue_uv": f"qwen3_4b_Base_rescue_uvw_a4_s{s}_r3", "rescue_uv_w1": f"qwen3_4b_Base_rescue_uv_a4_s{s}_r3",
            "rescue_abs": f"qwen3_4b_Base_rescue_abs_a4_s{s}_r3", "own_clean": f"qwen3_4b_Base_own_clean_a3_s{s}_r3"}
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
def verdicts(tag):
    Vd = {}
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/{tag}_r1/parts/*.json"):
        for v in json.load(open(f))["verdicts"]: Vd[v["id"]] = v["verdict"]
    return Vd
def comp(path, Vd):
    f = f"{path}/guard_records.jsonl"
    if not os.path.exists(f): return None
    recs = [json.loads(l) for l in open(f)]; recs = [r for r in recs if r["kind"] == "proposed" and r["id"] in Vd]
    if len(recs) < 1000: return None
    out = {}
    for name, key in (("count_unknown", lambda r: max(r["p_mode"], r["abstain"])), ("plain_p_mode", lambda r: r["p_mode"])):
        adm = [r for r in recs if 0.25 <= key(r) <= 0.75]; n = len(adm)
        ill = [r for r in adm if Vd[r["id"]] != "SOLVABLE"]
        unk = [r for r in ill if r["abstain"] > r["p_mode"]]
        m = {c: sum(0.25 <= key(r) <= 0.75 for r in recs if Vd[r["id"]] == c) / max(1, sum(Vd[r["id"]] == c for r in recs)) for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
        out[name] = {"n_admitted": n, "illposed_share": 100 * len(ill) / max(1, n), "unsolvable_share": 100 * sum(Vd[r["id"]] == "UNSOLVABLE" for r in adm) / max(1, n),
                     "illposed_with_unknown_majority_share_of_admitted": 100 * len(unk) / max(1, n),
                     "hallucination_target_share": 100 * (len(ill) - len(unk)) / max(1, n), "in_band_mass": m}
    out["unknown_plurality_by_class"] = {c: sum(r["abstain"] > r["p_mode"] for r in recs if Vd[r["id"]] == c) / max(1, sum(Vd[r["id"]] == c for r in recs)) for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
    return out
for s, pool in ((0, "qwen_s0"), (1, "qwen_s1")):
    Vd = verdicts(pool)
    P = {"own_r3": comp(f"{GD}/{pool}_ag/r3", Vd), "base": comp(f"{GD}/{pool}_ag/base", Vd),
         "rescue_uv": comp(f"{GD}/uv/pool_s{s}/rescue_uvw_s{s}", Vd), "rescue_uv_w1": comp(f"{GD}/uv/pool_s{s}/rescue_uv_s{s}", Vd), "rescue_abs": comp(f"{GD}/uv/pool_s{s}/rescue_abs_s{s}", Vd)}
    print(f"pool {pool} (seed {s}); admission on count-unknown majority share | plain p_mode band")
    for k, v in P.items():
        if v is None: print(f"  {k:10s} pending"); continue
        a, b = v["count_unknown"], v["plain_p_mode"]
        print(f"  {k:10s} n {a['n_admitted']:4d} ill {a['illposed_share']:5.1f} uns {a['unsolvable_share']:5.1f} ill&unknown-majority {a['illposed_with_unknown_majority_share_of_admitted']:5.1f} hallucination-target {a['hallucination_target_share']:5.1f} | plain ill {b['illposed_share']:5.1f} (n {b['n_admitted']})"
              f" | in-band S/A/U {a['in_band_mass']['SOLVABLE']:.2f}/{a['in_band_mass']['AMBIGUOUS']:.2f}/{a['in_band_mass']['UNSOLVABLE']:.2f} | unknown plurality S/A/U {v['unknown_plurality_by_class']['SOLVABLE']:.2f}/{v['unknown_plurality_by_class']['AMBIGUOUS']:.2f}/{v['unknown_plurality_by_class']['UNSOLVABLE']:.2f}")
    hu4 = None if P["rescue_uv"] is None or P["own_r3"] is None else P["rescue_uv"]["count_unknown"]["illposed_share"] <= P["own_r3"]["count_unknown"]["illposed_share"] - 10
    print(f"  -> seed {s}: HU4 {hu4}")
    res["pools"][pool] = {"comp": P, "HU4": hu4}
json.dump(res, open(f"{R}/dirA4/results/uvw_read.json", "w"), indent=1)
