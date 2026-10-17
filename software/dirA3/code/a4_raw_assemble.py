"""A4 PC1: assemble the 200-item audit of a round's RAW proposals (proposed.jsonl, Random(0) shuffle, first 200) from
<run>/audit_raw_parts/*.json into <run>/audit_raw.json (same format as audit_admitted.json).
  python3 a4_raw_assemble.py --run dirB/runs/<run>
"""
import argparse
import collections
import glob
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
P = [json.loads(l) for l in open(f"{a.run}/proposed.jsonl")]
random.Random(0).shuffle(P)
want = {p["id"]: k for k, p in enumerate(P[:200])}
V = {}
for f in sorted(glob.glob(f"{a.run}/audit_raw_parts/*.json")):
    for v in json.load(open(f)).get("verdicts", []):
        if v.get("id") in want and v.get("verdict") in ("SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"):
            V[v["id"]] = v
missing = [i for i in want if i not in V]
c = collections.Counter(v["verdict"] for v in V.values())
out = {"run": a.run.rstrip("/").split("/")[-1], "protocol": "200 raw proposals (Random(0) shuffle of proposed.jsonl), Opus judges, A3 audit prompt",
       "counts": dict(c), "illposed_share": (len(V) - c.get("SOLVABLE", 0)) / max(1, len(V)), "unsolvable_share": c.get("UNSOLVABLE", 0) / max(1, len(V)),
       "verdicts": sorted(V.values(), key=lambda v: want[v["id"]])}
print(json.dumps({k: out[k] for k in ("run", "counts", "illposed_share", "unsolvable_share")}), "missing", len(missing))
if not missing:
    json.dump(out, open(f"{a.run}/audit_raw.json", "w"), indent=1)
    print("wrote", f"{a.run}/audit_raw.json")
