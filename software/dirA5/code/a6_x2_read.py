"""A6 X2 read (registered 09:32): UMWP CA (share of the 600 unanswerable items whose plain-prompt modal numeric answer holds
>= 0.5 of 8 samples; umwp_read CA) of Phi-4-mini CLN1 vs RND1; holds if CA(CLN1) <= CA(RND1) - 0.04 in 2 of 2 seeds.
python3 a6_x2_read.py"""
import json, os
D = "/home/tione/notebook/rsi_rl/dirA4/results/umwp"
def ca(tag):
    f = f"{D}/plain_{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    if len(rows) < 1200: return None
    U = [r for r in rows.values() if not r["answerable"]]
    return sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in U) / len(U)
res = []
for s in (0, 1):
    c, r = ca(f"phi4mini_cln1_a6_s{s}_r3"), ca(f"phi4mini_rnd1_a6_s{s}_r3")
    res.append({"seed": s, "CA_cln1": c, "CA_rnd1": r, "holds": None if c is None or r is None else c <= r - 0.04 + 1e-9})
    print(res[-1])
v = None if any(x["holds"] is None for x in res) else all(x["holds"] for x in res)
print("X2 (Phi):", v)
json.dump({"seeds": res, "X2_phi": v}, open("/home/tione/notebook/rsi_rl/dirA5/results/a6_x2_read.json", "w"), indent=1)
