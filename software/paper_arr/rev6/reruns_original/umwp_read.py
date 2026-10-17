# rev6 rerun copy of dirA3/code/umwp_read.py (UMWP dir -> dumps_original, 1 occurrence(s); output -> reruns_original)
"""A4h reads HX1-HX4 on UMWP (see A4 registration, amendment A4h). A file counts only when complete (1,200 rows).
CA = share of unanswerable questions whose modal answer is a number with p_mode >= 0.5 (plain prompt); ABS_U / ABS_A = mean
abstention share under the abstention prompt; ACC_A = modal-answer accuracy on answerable questions (plain prompt).
python3 umwp_read.py
"""
import json, os
from collections import defaultdict
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/paper_arr/rev6/dumps_original"


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
        cat = defaultdict(list)
        for r in u:
            cat[r["category"]].append(r["mode"] is not None and r["p_mode"] >= 0.5)
        m["CA_by_cat"] = {str(k): round(sum(v) / len(v), 3) for k, v in sorted(cat.items())}
    if a:
        u = [r for r in a if not r["answerable"]]; s = [r for r in a if r["answerable"]]
        m["ABS_U"] = 100 * sum(r["abstain"] for r in u) / len(u)
        m["ABS_A"] = 100 * sum(r["abstain"] for r in s) / len(s)
        m["ACC_A_abs"] = 100 * sum(r["correct"] for r in s) / len(s)
    return m


names = {"base": "qwen3_4b_base"}
for s in range(5):
    for arm in ("own_a3", "own_clean_a3", "rescue_abs_a4", "rescue_uv_a4"):
        names[f"{arm}_s{s}_r3"] = f"qwen3_4b_Base_{arm}_s{s}_r3"
    names[f"own_a3_s{s}_r5"] = f"qwen3_4b_Base_own_a3_s{s}_r5"
M = {k: metrics(v) for k, v in names.items()}


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


print("run                    | CA     ACC_A | ABS_U  ABS_A | CA by category (1 missing, 2 ambiguous, 3 unrealistic, 4 unrelated, 5 no question)")
for k, m in M.items():
    if m:
        print(f"{k:22s} | {f(m.get('CA'))}  {f(m.get('ACC_A'), 1)} | {f(m.get('ABS_U'), 1)}  {f(m.get('ABS_A'), 1)} | {m.get('CA_by_cat', '-')}")


def g(k, x):
    return M.get(k, {}).get(x)


hx1 = [(s, g(f"own_clean_a3_s{s}_r3", "CA") < g(f"own_a3_s{s}_r3", "CA")) for s in range(4)
       if g(f"own_clean_a3_s{s}_r3", "CA") is not None and g(f"own_a3_s{s}_r3", "CA") is not None]
hx2 = [(s, g(f"own_a3_s{s}_r3", "CA") > g("base", "CA")) for s in range(5)
       if g(f"own_a3_s{s}_r3", "CA") is not None and g("base", "CA") is not None]
hx3, hx4 = [], []
for s in (0, 1, 2, 4):
    x, o = M.get(f"rescue_uv_a4_s{s}_r3", {}), M.get(f"own_a3_s{s}_r3", {})
    if all(k in x and k in o for k in ("ABS_U", "ABS_A", "ACC_A")):
        hx3.append((s, x["ABS_U"] >= o["ABS_U"] + 10 and x["ABS_A"] <= o["ABS_A"] + 5 and x["ACC_A"] >= o["ACC_A"] - 2))
    if "CA" in x and "CA" in o:
        hx4.append((s, x["CA"] <= o["CA"] - 0.05))
print(f"HX1 (clean < own, >= 3 of 4 seeds): {hx1} -> {None if len(hx1) < 3 else sum(v for _, v in hx1) >= 3}")
print(f"HX2 (own r3 > base, every seed): {hx2}")
print(f"HX3 (rescue_uv abstains on unanswerable, not on answerable, no accuracy cost; every seed run): {hx3}")
print(f"HX4 (rescue_uv CA <= own - 0.05; every seed run): {hx4}")
json.dump({"metrics": M, "HX1": hx1, "HX2": hx2, "HX3": hx3, "HX4": hx4}, open(f"{R}/paper_arr/rev6/reruns_original/umwp_read.json", "w"), indent=1)
