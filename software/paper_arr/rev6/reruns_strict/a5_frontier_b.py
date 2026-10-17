# rev6 rerun copy of dirA5/code/a5_frontier_b.py (UMWP dir -> dumps_strict, 1 occurrence(s); output -> reruns_strict)
"""A5b (addendum hashed before any trained-arm frontier was computed or viewed): selective-answering frontier of trained arms
vs own r3 of the same seed, abstention-prompt dumps, the registered rule set and frontier construction of a5_frontier.py
(imported from it unchanged). CA_U at FA_A = 0.05 on UMWP answerable, and at FA_A = 0.10 on intact twins.
FB1: rescue_uv r3 <= own r3 - 0.03 in >= 3 of 4 seeds (0, 1, 2, 4) [UMWP]. FB2: loop_uv r4 <= own r3 - 0.03 in >= 2 of 3 [UMWP].
FB3: the same two on twins. Descriptive: guard r3, clean r3, rescue_abs r3, Phi rescue_uv r3, TG abs4/t4 r4.
python3 a5_frontier_b.py"""
import json
R = "/home/tione/notebook/rsi_rl"
src = open(f"{R}/dirA5/code/a5_frontier.py").read().replace("/paper_arr/rev6/dumps_strict", "/paper_arr/rev6/dumps_strict")
ns = {}
exec(src[:src.index("res = {")], ns)   # definitions only (load_umwp, load_twins, curve, frontier_at, subsample_draws, ...)
EPS = ns["EPS"]
def ca_at(tag, bench, fa):
    if bench == "umwp":
        d = ns["load_umwp"](tag, "abs")
        return None if d is None else ns["frontier_at"](ns["curve"](d), fa)
    d = ns["load_twins"](tag, "hpabs")
    return None if d is None else ns["frontier_at"](ns["curve"](d, draws=ns["subsample_draws"](d)), fa)
FA = {"umwp": 0.05, "twins": 0.10}
out = {"FB1": [], "FB2": [], "FB3_rescue": [], "FB3_loop": [], "descriptive": []}
own = lambda s: f"qwen3_4b_Base_own_a3_s{s}_r3"
base_cache = {}
def own_ca(s, bench):
    k = (s, bench)
    if k not in base_cache: base_cache[k] = ca_at(own(s), bench, FA[bench])
    return base_cache[k]
for key, arm, seeds, bench in (("FB1", "qwen3_4b_Base_rescue_uv_a4_s{}_r3", (0, 1, 2, 4), "umwp"), ("FB2", "qwen3_4b_Base_loop_uv_a4_s{}_r4", (0, 1, 2), "umwp"),
                               ("FB3_rescue", "qwen3_4b_Base_rescue_uv_a4_s{}_r3", (0, 1, 2, 4), "twins"), ("FB3_loop", "qwen3_4b_Base_loop_uv_a4_s{}_r4", (0, 1, 2), "twins")):
    for s in seeds:
        o, a = own_ca(s, bench), ca_at(arm.format(s), bench, FA[bench])
        ok = None if o is None or a is None else a <= o - 0.03 + EPS
        out[key].append({"seed": s, "own": o, "arm": a, "holds": ok})
        print(f"{key} s{s} {bench}: own r3 CA_U@{FA[bench]} {o} vs arm {a} -> {ok}")
def v(lst, need, of):
    rd = [x for x in lst if x["holds"] is not None]
    return None if len(rd) < of else sum(x["holds"] for x in rd) >= need
out["verdicts"] = {"FB1": v(out["FB1"], 3, 4), "FB2": v(out["FB2"], 2, 3),
                   "FB3": None if v(out["FB3_rescue"], 3, 4) is None or v(out["FB3_loop"], 2, 3) is None else (v(out["FB3_rescue"], 3, 4) and v(out["FB3_loop"], 2, 3))}
for tag, s in [(f"qwen3_4b_Base_own_ag_a4_s{s}_r3", s) for s in range(3)] + [(f"qwen3_4b_Base_own_clean_a3_s{s}_r3", s) for s in range(4)] + \
              [(f"qwen3_4b_Base_rescue_abs_a4_s{s}_r3", s) for s in range(3)] + [(f"qwen3_4b_Base_tg_{a}_a5_s{s}_r4", s) for a in ("abs4", "t4") for s in range(3)]:
    for bench in ("umwp", "twins"):
        out["descriptive"].append({"tag": tag, "bench": bench, "own_r3": own_ca(s, bench) if s < 5 else None, "arm": ca_at(tag, bench, FA[bench])})
print("verdicts:", out["verdicts"])
json.dump(out, open(f"{R}/paper_arr/rev6/reruns_strict/frontier_b_read.json", "w"), indent=1)
