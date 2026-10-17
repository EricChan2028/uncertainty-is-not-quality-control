"""Assemble the 200-item audit (audit_admitted.json, same format as the earlier panel outputs) from judge part files
audit_parts/<a>-<b>.json that cover k in [0,200) of the Random(0)-shuffled admitted proposer questions.
python3 a3_audit200_assemble.py --run dirB/runs/<run>
"""
import argparse
import collections
import glob
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
A = [json.loads(l) for l in open(f"{a.run}/admitted.jsonl")]
P = [x for x in A if x["kind"] == "proposed"]
random.Random(0).shuffle(P)
S = {x["id"]: k for k, x in enumerate(P[:200])}
V = {}
for p in sorted(glob.glob(f"{a.run}/audit_parts/*.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        print("unreadable", p, e)
        continue
    for v in d.get("verdicts", []):
        if v.get("id") in S and v.get("verdict") in ("SOLVABLE", "UNSOLVABLE", "AMBIGUOUS") and v["id"] not in V:
            V[v["id"]] = dict(v, k=S[v["id"]])
missing = [i for i in S if i not in V]
verdicts = sorted(V.values(), key=lambda v: v["k"])
counts = collections.Counter(v["verdict"] for v in verdicts)
out = {"run": os.path.basename(a.run.rstrip("/")), "protocol": "200 admitted proposer questions, Random(0) shuffle, Opus judges x 25 questions, SOLVABLE/UNSOLVABLE/AMBIGUOUS (strict), assembled from audit_parts",
       "judge": "claude opus (workflow judges)", "counts": dict(counts), "illposed_share": counts["UNSOLVABLE"] / max(1, len(verdicts)), "verdicts": verdicts}
print(json.dumps({k: out[k] for k in ("run", "counts", "illposed_share")}), "n", len(verdicts), "missing", len(missing))
if missing:
    print("MISSING", missing[:20])
    raise SystemExit(1)
json.dump(out, open(f"{a.run}/audit_admitted.json.tmp", "w"), indent=0, ensure_ascii=False)
os.replace(f"{a.run}/audit_admitted.json.tmp", f"{a.run}/audit_admitted.json")
print("wrote", f"{a.run}/audit_admitted.json")
