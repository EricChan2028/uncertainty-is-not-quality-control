"""A4i HL1 read: among the 200 audited admitted proposals of loop_uv round 4, the share that is ill-posed (UNSOLVABLE or
AMBIGUOUS) with a majority answer that is not "unknown" (abstain share <= p_mode in the guard record: the problem trains an
invented answer); compared with own round 4's audited ill-posed share (own has no unknown vote). Also the audited class
mix, and unknown-majority rates by audited class.   python3 a4i_hl1.py"""
import json
B = "/home/tione/notebook/rsi_rl/dirB/runs"
OWN = {0: 80.5, 1: 80.0, 2: 78.5}
out = {}
for s in (0, 1, 2):
    if not __import__("os").path.exists(f"{B}/qwen3_4b_Base_loop_uv_a4_s{s}_r4/audit_admitted.json"):
        print(f"seed {s}: audit pending"); continue
    d = f"{B}/qwen3_4b_Base_loop_uv_a4_s{s}_r4"
    V = {v["id"]: v["verdict"] for v in json.load(open(f"{d}/audit_admitted.json"))["verdicts"]}
    R = {r["id"]: r for r in (json.loads(l) for l in open(f"{d}/guard_records.jsonl"))}
    ids = [i for i in V if i in R]
    n = len(ids)
    ill = [i for i in ids if V[i] != "SOLVABLE"]
    unk = lambda i: R[i]["abstain"] > R[i]["p_mode"]
    inv = [i for i in ill if not unk(i)]
    by = {c: (sum(unk(i) for i in ids if V[i] == c), sum(V[i] == c for i in ids)) for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
    share_inv = 100 * len(inv) / n
    hl1 = share_inv <= OWN[s] - 15
    out[s] = {"n": n, "illposed": 100 * len(ill) / n, "invented_target_share": share_inv, "own_r4_illposed": OWN[s], "HL1": hl1,
              "unknown_majority_by_class": {c: f"{a}/{b}" for c, (a, b) in by.items()}}
    print(f"seed {s}: audited {n}; ill-posed {100 * len(ill) / n:.1f}%; ill-posed with a non-unknown majority (invented-answer target) {share_inv:.1f}%"
          f" vs own r4 ill-posed {OWN[s]}% -> HL1 {hl1}; unknown-majority by class {out[s]['unknown_majority_by_class']}")
json.dump(out, open("/home/tione/notebook/rsi_rl/dirA4/results/a4i_hl1.json", "w"), indent=1)
