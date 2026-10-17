"""A6 E1/E2 read (registered 09:32). Per set (sum, mip_gsm8k, mip_math): CA = share of unanswerable items whose modal numeric
answer holds >= 0.5 of 8 samples (plain prompt); ABS_U = mean abstention share on unanswerable (abstention prompt).
E1: CA(own r3) >= CA(clean r3) + 0.05 in >= 3 of 4 seeds on >= 2 of 3 sets. E2: |ABS_U(own) - ABS_U(clean)| <= 0.05 in >= 3
of 4 seeds on >= 2 of 3 sets. A file counts only when complete (1552 rows).   python3 a6_e_read.py"""
import json, os
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA5/results/a6_sweep"; SETS = ("sum", "mip_gsm8k", "mip_math")
def load(tag):
    f = f"{D}/{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    return rows if len(rows) >= 1552 else None
def ns(ans):
    c = Counter(a for a in ans if a is not None and a != "unknown"); return (c.most_common(1)[0][1] if c else 0) / len(ans)
res = {"E1": {}, "E2": {}, "values": {}}
for st in SETS:
    e1, e2 = [], []
    for s in range(4):
        v = {}
        for arm in ("own_a3", "own_clean_a3"):
            p, a = load(f"plain_qwen3_4b_Base_{arm}_s{s}_r3"), load(f"abs_qwen3_4b_Base_{arm}_s{s}_r3")
            if p: U = [r for r in p.values() if r["source"] == st and not r["answerable"]]; v[f"CA_{arm}"] = sum(ns(r["answers"]) >= 0.5 for r in U) / len(U)
            if a: U = [r for r in a.values() if r["source"] == st and not r["answerable"]]; v[f"ABS_{arm}"] = sum(r["abstain"] for r in U) / len(U)
        res["values"][f"{st}_s{s}"] = v
        if "CA_own_a3" in v and "CA_own_clean_a3" in v: e1.append(v["CA_own_a3"] >= v["CA_own_clean_a3"] + 0.05 - 1e-9)
        if "ABS_own_a3" in v and "ABS_own_clean_a3" in v: e2.append(abs(v["ABS_own_a3"] - v["ABS_own_clean_a3"]) <= 0.05 + 1e-9)
        print(st, s, {k: round(x, 3) for k, x in v.items()})
    res["E1"][st] = None if len(e1) < 4 else sum(e1) >= 3
    res["E2"][st] = None if len(e2) < 4 else sum(e2) >= 3
def over(d):
    vals = [x for x in d.values() if x is not None]
    return None if len(vals) < 3 else sum(vals) >= 2
res["verdicts"] = {"E1": over(res["E1"]), "E2": over(res["E2"]), "per_set": {"E1": res["E1"], "E2": res["E2"]}}
print("verdicts:", res["verdicts"])
json.dump(res, open(f"{R}/dirA5/results/a6_e_read.json", "w"), indent=1)
