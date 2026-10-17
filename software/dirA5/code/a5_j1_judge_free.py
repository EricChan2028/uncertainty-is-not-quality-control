"""A5 J1 (registered 08:04), judge-free drift reads J-P2 and J-P3.
In-band = modal NUMERIC answer share of G = 8 samples in [0.25, 0.75] (the guard's admission statistic; unparsed / truncated
samples count in the denominator, as in a3_guard.py). Ratio = in-band rate on unanswerable items / in-band rate on answerable
items. J-P2 (UMWP human labels, plain prompt, G = 8 dumps): ratio(r3) >= 1.25 x ratio(base) in >= 80% of powered own lineages;
a lineage is underpowered if < 30 answerable items are in band at base or at r3. J-P3 (412 validated premise-deletion twins
and their intact originals, 32-sample dumps subsampled to G = 8): same criterion in >= 8 of 10 own lineages. Deviation
(disclosed): the registered "exact hypergeometric subsampling" is computed by Monte Carlo, 2,000 draws per item, seed
20260924 (SE of an item's in-band probability < 0.012).   python3 a5_j1_judge_free.py"""
import glob, json, os, random
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; U = f"{R}/dirA4/results/umwp"; H = f"{R}/dirB/results/hp"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
def inband(ans):
    c = Counter(a for a in ans if a is not None and a != "unknown")
    k = c.most_common(1)[0][1] if c else 0
    return 0.25 <= k / len(ans) <= 0.75
def umwp(tag):
    f = f"{U}/plain_{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    if len(rows) < 1200: return None
    rs = list(rows.values())
    iu = [inband(r["answers"]) for r in rs if not r["answerable"]]; ia = [inband(r["answers"]) for r in rs if r["answerable"]]
    return {"rate_U": sum(iu) / len(iu), "rate_A": sum(ia) / len(ia), "n_inband_A": sum(ia)}
rng = random.Random(20260924)
def twins(run, pattern=None):
    fs = glob.glob(pattern) if pattern else glob.glob(f"{H}/hp_{run}_shard*.jsonl")
    d = [json.loads(l) for f in fs for l in open(f)] if fs else []
    if len(d) < 1086: return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    uk = {(str(r["fam"]), int(r["task"])) for r in u}
    a = [r for r in d if r["form"] == "intact" and (str(r["fam"]), int(r["task"])) in uk]
    def p_in(r):
        ans = r["answers"]; n = 0
        for _ in range(2000):
            n += inband(rng.sample(ans, 8))
        return n / 2000
    pu = [p_in(r) for r in u]; pa = [p_in(r) for r in a]
    return {"rate_U": sum(pu) / len(pu), "rate_A": sum(pa) / len(pa), "n_inband_A": sum(pa), "n_U": len(u), "n_A": len(a)}
LIN = [("qwen3_4b", "qwen3_4b_base", [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)]),
       ("qwen3_8b", "base_Qwen3-8B-Base", [f"qwen3_8b_Base_own_a3_s{s}_r3" for s in range(3)]),
       ("phi4mini", "base_Phi-4-mini-instruct", [f"phi4mini_own_a4_s{s}_r3" for s in range(2)])]
TW_BASE = {"qwen3_4b": f"{R}/dirA/results/qwen3_4b_base_shard*of8.jsonl", "qwen3_8b": f"{R}/dirA/results/qwen3_8b_base_shard*.jsonl", "phi4mini": f"{R}/dirA/results/phi4mini_shard*of8.jsonl"}
out = {"JP2": [], "JP3": []}
for fam, base, runs in LIN:
    b = umwp(base)
    print(f"UMWP {fam} base: {b}")
    for run in runs:
        m = umwp(run)
        if b is None or m is None:
            out["JP2"].append({"run": run, "read": None, "why": "file missing or incomplete"}); print(f"  {run}: not read"); continue
        powered = b["n_inband_A"] >= 30 and m["n_inband_A"] >= 30
        rb, rm = b["rate_U"] / max(1e-9, b["rate_A"]), m["rate_U"] / max(1e-9, m["rate_A"])
        ok = rm >= 1.25 * rb
        out["JP2"].append({"run": run, "ratio_base": rb, "ratio_r3": rm, "powered": powered, "holds": ok, "base": b, "r3": m})
        print(f"  {run}: ratio base {rb:.3f} -> r3 {rm:.3f} (x{rm / max(1e-9, rb):.2f}) powered {powered} -> {ok}")
pw = [x for x in out["JP2"] if x.get("powered")]
out["JP2_verdict"] = None if not pw else sum(x["holds"] for x in pw) / len(pw) >= 0.8
print("J-P2:", out["JP2_verdict"], f"({sum(x['holds'] for x in pw)}/{len(pw)} powered lineages)")
bases = {}
for fam, pat in TW_BASE.items():
    bases[fam] = twins(None, pat)
    print(f"twins {fam} base ({pat}): {bases[fam]}")
for fam, base, runs in LIN:
    for run in runs:
        m = twins(run); b = bases.get(fam)
        if b is None or m is None:
            out["JP3"].append({"run": run, "read": None}); print(f"  twins {run}: not read"); continue
        rb, rm = b["rate_U"] / max(1e-9, b["rate_A"]), m["rate_U"] / max(1e-9, m["rate_A"])
        out["JP3"].append({"run": run, "ratio_base": rb, "ratio_r3": rm, "holds": rm >= 1.25 * rb})
        print(f"  twins {run}: ratio base {rb:.3f} -> r3 {rm:.3f} (x{rm / max(1e-9, rb):.2f})")
rd = [x for x in out["JP3"] if x.get("holds") is not None]
out["JP3_verdict"] = None if len(rd) < 10 else sum(x["holds"] for x in rd) >= 8
print("J-P3:", out["JP3_verdict"], f"({sum(x['holds'] for x in rd)}/{len(rd)} lineages read)")
json.dump(out, open(f"{R}/dirA5/results/j1_judge_free.json", "w"), indent=1)
