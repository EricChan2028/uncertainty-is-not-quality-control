"""A5 TG: build the tri-gate training subset T4 from an A4i loop_uv round-4 run (same start, proposals and samples).
Keep an admitted item when (abstain <= 1/8 and numeric p_mode in [0.25, 0.75]) -> numeric target, or abstain in {5/8, 6/8}
-> "unknown" majority; drop items with 2-4 abstentions of 8. The registered 50% cap on unknown-majority items is
non-binding and only reported. Tracers follow the same rule. Writes <out>/admitted.jsonl and <out>/trigate_stats.json.
python3 a5_trigate.py <src_run_dir> <out_dir>"""
import json, sys, os
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
R = {r["id"]: r for r in (json.loads(l) for l in open(f"{src}/guard_records.jsonl"))}
A = [json.loads(l) for l in open(f"{src}/admitted.jsonl")]
keep, st = [], {"admitted_in": len(A), "numeric": 0, "unknown": 0, "dropped_2to4": 0, "other_dropped": 0}
for a in A:
    r = R[a["id"]]
    k = round(r["abstain"] * 8)
    if k <= 1 and 0.25 <= r["p_mode"] <= 0.75:
        keep.append(a); st["numeric"] += 1
    elif k in (5, 6):
        keep.append(a); st["unknown"] += 1
    elif 2 <= k <= 4:
        st["dropped_2to4"] += 1
    else:
        st["other_dropped"] += 1
st["kept"] = len(keep)
st["unknown_share_of_kept"] = st["unknown"] / max(1, len(keep))
st["cap_50pct_binding"] = st["unknown_share_of_kept"] > 0.5
with open(f"{out}/admitted.jsonl", "w") as fh:
    for a in keep:
        fh.write(json.dumps(a, ensure_ascii=False) + "\n")
json.dump(st, open(f"{out}/trigate_stats.json", "w"), indent=1)
print(json.dumps(st))
