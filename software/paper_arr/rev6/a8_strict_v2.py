"""rev6 (exploratory): the A8 numbers of paper_arr/analyses/rev5/strict_parse.py and of paper_arr/rev6/fig4_categories.py
recomputed with the amended STRICT of rev6_read.py (strict()'s validated status, the probe's number). On the 112 unanswerable
A8 questions per checkpoint: share whose most frequent numeric answer holds >= 4, >= 6 and 8 of 8 samples, under ORIGINAL,
STRICT-v1 (strict()'s own number, as in rev5) and STRICT (amended); and per unit, self-consistent (>= 4 of 8) with no sample
flagging a gap (LLM gap flags of A8). Output: paper_arr/rev6/a8_strict_v2.json.   python3 a8_strict_v2.py"""
import json
from collections import Counter
R = "/home/tione/notebook/rsi_rl"
src = open(f"{R}/paper_arr/analyses/rev5/strict_parse.py").read(); sp = {}
exec(compile(src[:src.index("# ---- validation against the human answer status")], "strict_parse_defs", "exec"), sp)
strict = sp["strict"]
import sys; sys.path.insert(0, f"{R}/paper_arr/rev6"); import strict_v3  # noqa: E402
TAGS = ("a8_base", "a8_own_s0_r3", "a8_own_s1_r3", "a8_own_s2_r3", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5")
K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{R}/dirA5/results/a8_key/units_main_key.jsonl")}
L = {x["key"]: x for x in json.load(open(f"{R}/dirA5/results/a8/labels_main.json")) if isinstance(x, dict) and "key" in x}
flags = {}
for k, m in K.items():
    if k in L: flags[("a8_base" if m["model"] == "base" else f"a8_own_s{m['seed']}_r{m['round']}", m["id"])] = L[k]["gap_flag"]
def top(ans):
    c = Counter(a for a in ans if a is not None); return c.most_common(1)[0][1] if c else 0
out = {}
for tag in TAGS:
    rows = [json.loads(l) for l in open(f"{R}/dirA5/results/a8/texts/{tag}.jsonl")]
    U = [x for x in rows if not x["answerable"]]
    res = {}
    for name in ("ORIGINAL", "STRICT_v1", "STRICT_v2", "STRICT"):
        tops, nf = [], 0
        for x in U:
            if name == "ORIGINAL": ans = x["answers"]
            elif name == "STRICT_v1": ans = [strict(t, f) for t, f in zip(x["texts"], x["finish"])]
            elif name == "STRICT_v2": ans = [None if strict(t, f) is None else (a if a is not None else strict(t, f)) for t, f, a in zip(x["texts"], x["finish"], x["answers"])]
            else: ans = [strict_v3.read(t, f) for t, f in zip(x["texts"], x["finish"])]
            tp = top(ans); tops.append(tp)
            fl = flags.get((tag, x["id"]))
            if fl is not None and tp >= 4 and not any(fl): nf += 1
        n = len(U)
        res[name] = {">=4": sum(t >= 4 for t in tops) / n, ">=6": sum(t >= 6 for t in tops) / n, "8": sum(t == 8 for t in tops) / n,
                     "sc_no_flag": nf / n if any((tag, x["id"]) in flags for x in U) else None, "n": n}
    out[tag] = res
    print(f"{tag:14s}", " | ".join(f"{k}: {v['>=4']:.3f} {v['>=6']:.3f} {v['8']:.3f}" + (f" nf {v['sc_no_flag']:.3f}" if v["sc_no_flag"] is not None else "") for k, v in res.items()))
json.dump(out, open(f"{R}/paper_arr/rev6/a8_strict_v2.json", "w"), indent=1)
