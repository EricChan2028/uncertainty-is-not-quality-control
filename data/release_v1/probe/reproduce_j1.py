#!/usr/bin/env python3
"""Recompute the judge-free reads J-P2 (UMWP) and J-P3 (premise-deletion twins) of results/j1_judge_free.txt from the
released dumps alone: umwp_dumps/, twin_dumps/ and ../twins/probe543_validation.jsonl (no model, GPU, UMWP or GSM-Symbolic
text needed). Same definitions as code/a5_j1_judge_free.py, which reads the project tree instead.

  cd probe && python3 reproduce_j1.py [--mc] [--out FILE]

In-band = the modal numeric answer's share of G = 8 samples is in [0.25, 0.75] (unparsed / truncated / 'unknown' samples
count in the denominator and never vote; the admission statistic of a3_guard.py). Ratio = in-band rate on unanswerable
items / in-band rate on answerable items. J-P2: UMWP unanswerable questions vs their answerable originals, plain-prompt
G = 8 dumps; holds if ratio(r3) >= 1.25 x ratio(base) in >= 80% of the powered own lineages (>= 30 answerable items in
band at base and at r3). This part is exact. J-P3: the deleted forms of the twins that pass-1 validation judged
UNSOLVABLE vs the intact forms of the same tasks, 32-sample dumps subsampled to G = 8; holds in >= 8 of 10 own lineages.
By default the subsampling is the exact hypergeometric probability (the registered method); --mc repeats the original
Monte-Carlo approximation (2,000 draws per item, seed 20260924), whose values depend on the row order of the dumps and
therefore match the original read only approximately. The script ends by comparing its ratios with
results/j1_judge_free.json.
"""
import argparse, json, os, random
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--umwp_dir", default=os.path.join(HERE, "umwp_dumps"))
ap.add_argument("--twin_dir", default=os.path.join(HERE, "twin_dumps"))
ap.add_argument("--validation", default=os.path.join(HERE, "..", "twins", "probe543_validation.jsonl"))
ap.add_argument("--original", default=os.path.join(HERE, "results", "j1_judge_free.json"))
ap.add_argument("--mc", action="store_true", help="Monte-Carlo subsampling as in the original a5_j1_judge_free.py")
ap.add_argument("--out", default="")
a = ap.parse_args()
G, LO, HI = 8, 0.25, 0.75


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


UNS = {(str(v["fam"]), int(v["task"])) for v in jl(a.validation) if v["pass1_verdict"] == "UNSOLVABLE"}


def inband(ans):
    c = Counter(x for x in ans if x is not None and x != "unknown")
    k = c.most_common(1)[0][1] if c else 0
    return LO <= k / len(ans) <= HI


KS = [k for k in range(G + 1) if LO <= k / G <= HI]


def p_inband_exact(ans):
    """Probability that a uniformly random G-subset of ans is in band (multivariate hypergeometric, exact)."""
    n = len(ans)
    if n == G:
        return float(inband(ans))
    c = Counter(x for x in ans if x is not None and x != "unknown")
    n_null = n - sum(c.values())

    def ways_max_le(m):  # number of G-subsets in which no answer has more than m votes
        poly = [comb(n_null, j) for j in range(G + 1)]
        for nc in c.values():
            new = [0] * (G + 1)
            for i, pi in enumerate(poly):
                if pi:
                    for j in range(0, min(m, nc, G - i) + 1):
                        new[i + j] += pi * comb(nc, j)
            poly = new
        return poly[G]
    return (ways_max_le(max(KS)) - ways_max_le(min(KS) - 1)) / comb(n, G)


rng = random.Random(20260924)


def p_inband_mc(ans):
    return sum(inband(rng.sample(ans, G)) for _ in range(2000)) / 2000


def umwp(tag):
    f = os.path.join(a.umwp_dir, f"plain_{tag}.jsonl")
    if not os.path.exists(f):
        return None
    rows = {r["umwp_id"]: r for r in jl(f)}
    if len(rows) < 1200:
        return None
    iu = [inband(r["answers"]) for r in rows.values() if not r["answerable"]]
    ia = [inband(r["answers"]) for r in rows.values() if r["answerable"]]
    return {"rate_U": sum(iu) / len(iu), "rate_A": sum(ia) / len(ia), "n_inband_A": sum(ia)}


def twins(tag):
    f = os.path.join(a.twin_dir, f"{tag}.jsonl")
    d = jl(f) if os.path.exists(f) else []
    if len(d) < 1086:
        return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    uk = {(str(r["fam"]), int(r["task"])) for r in u}
    ia = [r for r in d if r["form"] == "intact" and (str(r["fam"]), int(r["task"])) in uk]
    p = p_inband_mc if a.mc else p_inband_exact
    pu = [p(r["answers"]) for r in u]; pa = [p(r["answers"]) for r in ia]
    return {"rate_U": sum(pu) / len(pu), "rate_A": sum(pa) / len(pa), "n_inband_A": sum(pa), "n_U": len(u), "n_A": len(ia)}


LIN = [("qwen3_4b", "qwen3_4b_base", "Qwen3-4B-Base", [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)]),
       ("qwen3_8b", "base_Qwen3-8B-Base", "Qwen3-8B-Base", [f"qwen3_8b_Base_own_a3_s{s}_r3" for s in range(3)]),
       ("phi4mini", "base_Phi-4-mini-instruct", "Phi-4-mini-instruct", [f"phi4mini_own_a4_s{s}_r3" for s in range(2)])]
out = {"JP2": [], "JP3": [], "subsampling": "monte_carlo" if a.mc else "exact_hypergeometric"}
for fam, base, _, runs in LIN:
    b = umwp(base)
    print(f"UMWP {fam} base: {b}")
    for run in runs:
        m = umwp(run)
        if b is None or m is None:
            out["JP2"].append({"run": run, "read": None}); print(f"  {run}: not read"); continue
        powered = b["n_inband_A"] >= 30 and m["n_inband_A"] >= 30
        rb, rm = b["rate_U"] / max(1e-9, b["rate_A"]), m["rate_U"] / max(1e-9, m["rate_A"])
        out["JP2"].append({"run": run, "ratio_base": rb, "ratio_r3": rm, "powered": powered, "holds": rm >= 1.25 * rb})
        print(f"  {run}: ratio base {rb:.3f} -> r3 {rm:.3f} (x{rm / max(1e-9, rb):.2f}) powered {powered} -> {rm >= 1.25 * rb}")
pw = [x for x in out["JP2"] if x.get("powered")]
out["JP2_verdict"] = None if not pw else sum(x["holds"] for x in pw) / len(pw) >= 0.8
print("J-P2:", out["JP2_verdict"], f"({sum(x['holds'] for x in pw)}/{len(pw)} powered lineages)")
bases = {}
for fam, _, tb, _ in LIN:
    bases[fam] = twins(tb)
    print(f"twins {fam} base ({tb}): {bases[fam]}")
for fam, _, _, runs in LIN:
    for run in runs:
        m, b = twins(run), bases.get(fam)
        if b is None or m is None:
            out["JP3"].append({"run": run, "read": None}); print(f"  twins {run}: not read"); continue
        rb, rm = b["rate_U"] / max(1e-9, b["rate_A"]), m["rate_U"] / max(1e-9, m["rate_A"])
        out["JP3"].append({"run": run, "ratio_base": rb, "ratio_r3": rm, "holds": rm >= 1.25 * rb})
        print(f"  twins {run}: ratio base {rb:.3f} -> r3 {rm:.3f} (x{rm / max(1e-9, rb):.2f})")
rd = [x for x in out["JP3"] if x.get("holds") is not None]
out["JP3_verdict"] = None if len(rd) < 10 else sum(x["holds"] for x in rd) >= 8
print("J-P3:", out["JP3_verdict"], f"({sum(x['holds'] for x in rd)}/{len(rd)} lineages read)")
if os.path.exists(a.original):
    o = json.load(open(a.original, encoding="utf-8"))
    for part in ("JP2", "JP3"):
        prev = {x["run"]: x for x in o.get(part, []) if x.get("ratio_r3") is not None}
        diffs = [max(abs(x["ratio_base"] - prev[x["run"]]["ratio_base"]), abs(x["ratio_r3"] - prev[x["run"]]["ratio_r3"]))
                 for x in out[part] if x.get("ratio_r3") is not None and x["run"] in prev]
        same = sum(x["holds"] == prev[x["run"]]["holds"] for x in out[part] if x.get("ratio_r3") is not None and x["run"] in prev)
        print(f"vs results/j1_judge_free.json {part}: {len(diffs)} lineages, max |ratio difference| {max(diffs) if diffs else float('nan'):.4f}, "
              f"same per-lineage outcome {same}/{len(diffs)}, verdict {out[part + '_verdict']} vs {o.get(part + '_verdict')}")
if a.out:
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
