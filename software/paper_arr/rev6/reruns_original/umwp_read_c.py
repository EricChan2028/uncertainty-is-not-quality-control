# rev6 rerun copy of dirA3/code/umwp_read_c.py (UMWP dir -> dumps_original, 1 occurrence(s); output -> reruns_original)
"""A4h addendum c read: UMWP CA / ABS_A / ACC_A of the abstention-guarded loop (own_ag r3) vs own r3, Qwen3-4B seeds 0-2."""
import json, os
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/paper_arr/rev6/dumps_original"
def load(tag):
    f = f"{D}/{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try:
            r = json.loads(l); rows[r["id"]] = r
        except ValueError:
            pass
    return list(rows.values()) if len(rows) >= 1200 else None
def metrics(name):
    p, a = load(f"plain_{name}"), load(f"abs_{name}"); m = {}
    if p:
        u = [r for r in p if not r["answerable"]]; s = [r for r in p if r["answerable"]]
        m["CA"] = sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u) / len(u); m["ACC_A"] = 100 * sum(r["correct"] for r in s) / len(s)
    if a:
        u = [r for r in a if not r["answerable"]]; s = [r for r in a if r["answerable"]]
        m["ABS_U"] = 100 * sum(r["abstain"] for r in u) / len(u); m["ABS_A"] = 100 * sum(r["abstain"] for r in s) / len(s)
    return m
res = {"HX5": [], "HX6": []}
for s in range(3):
    o, g = metrics(f"qwen3_4b_Base_own_a3_s{s}_r3"), metrics(f"qwen3_4b_Base_own_ag_a4_s{s}_r3")
    print(f"s{s} own {o}\n   own_ag {g}")
    if "CA" in o and "CA" in g: res["HX5"].append((s, g["CA"] < o["CA"]))
    if all(k in o and k in g for k in ("ABS_A", "ACC_A")): res["HX6"].append((s, g["ABS_A"] <= o["ABS_A"] + 5 and g["ACC_A"] >= o["ACC_A"] - 2))
for s in range(2):
    print(f"coev_ag s{s}", metrics(f"qwen3_4b_Base_coev_ag_a4_s{s}_r3"))
print("HX5", res["HX5"], "HX6", res["HX6"])
json.dump(res, open(f"{R}/paper_arr/rev6/reruns_original/umwp_read_c.json", "w"), indent=1)
