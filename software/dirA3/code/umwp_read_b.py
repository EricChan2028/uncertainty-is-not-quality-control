"""A4h addendum b read (HX2b): UMWP confident answering (CA) of own round 3 vs the base model for Qwen3-8B-Base and
Phi-4-mini-instruct; same metrics and completeness rule (1,200 rows) as umwp_read.py.   python3 umwp_read_b.py
"""
import json, os
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA4/results/umwp"


def load(tag):
    f = f"{D}/{tag}.jsonl"
    if not os.path.exists(f):
        return None
    rows = {}
    for l in open(f):
        try:
            r = json.loads(l); rows[r["id"]] = r
        except ValueError:
            pass
    return list(rows.values()) if len(rows) >= 1200 else None


def metrics(name):
    p, a = load(f"plain_{name}"), load(f"abs_{name}")
    m = {}
    if p:
        u = [r for r in p if not r["answerable"]]; s = [r for r in p if r["answerable"]]
        m["CA"] = sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u) / len(u)
        m["ACC_A"] = 100 * sum(r["correct"] for r in s) / len(s)
    if a:
        u = [r for r in a if not r["answerable"]]; s = [r for r in a if r["answerable"]]
        m["ABS_U"] = 100 * sum(r["abstain"] for r in u) / len(u)
        m["ABS_A"] = 100 * sum(r["abstain"] for r in s) / len(s)
    return m


fam = {"qwen3_8b": ("base_Qwen3-8B-Base", [f"qwen3_8b_Base_own_a3_s{s}_r3" for s in range(3)]),
       "phi4mini": ("base_Phi-4-mini-instruct", [f"phi4mini_own_a4_s{s}_r3" for s in range(2)] + [f"phi4mini_own_ag_a4_s{s}_r3" for s in range(2)])}
res = {}
for k, (base, runs) in fam.items():
    mb = metrics(base)
    print(f"{k} base: {mb}")
    res[k] = {"base": mb, "runs": {}, "HX2b": []}
    for r in runs:
        m = metrics(r); res[k]["runs"][r] = m
        print(f"  {r}: {m}")
        if "_own_a" in r and "CA" in m and "CA" in mb:
            res[k]["HX2b"].append((r, m["CA"] > mb["CA"]))
    print(f"  HX2b {k}: {res[k]['HX2b']}")
json.dump(res, open(f"{R}/dirA4/results/umwp_read_b.json", "w"), indent=1)
