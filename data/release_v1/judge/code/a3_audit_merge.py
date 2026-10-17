"""Merge the 200-item audit (audit_admitted.json) with the extension parts (audit_parts/*.json) into audit_full.json
covering every admitted proposer question of a run. Exits 1 and lists missing ids if coverage is incomplete.
python3 a3_audit_merge.py --run dirB/runs/<run>
"""
import argparse
import collections
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
a = ap.parse_args()
A = [json.loads(l) for l in open(f"{a.run}/admitted.jsonl")]
prop_ids = [x["id"] for x in A if x["kind"] == "proposed"]
V, src = {}, collections.Counter()
if os.path.exists(f"{a.run}/audit_admitted.json"):
    for v in json.load(open(f"{a.run}/audit_admitted.json"))["verdicts"]:
        if v["id"] not in V:
            V[v["id"]] = v
            src["sample200"] += 1
for p in sorted(glob.glob(f"{a.run}/audit_parts/*.json")):
    try:
        d = json.load(open(p))
    except Exception as e:
        print("unreadable part", p, e)
        continue
    for v in d.get("verdicts", []):
        if v.get("id") and v["id"] not in V and v.get("verdict") in ("SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"):
            V[v["id"]] = v
            src[os.path.basename(p)] += 1
pset = set(prop_ids)
missing = [i for i in prop_ids if i not in V]
extra = [i for i in V if i not in pset]
judged = [V[i] for i in prop_ids if i in V]
counts = collections.Counter(v["verdict"] for v in judged)
n = max(1, len(judged))
out = {"run": os.path.basename(a.run.rstrip("/")),
       "protocol": "Opus judges, 20-25 questions each, same prompt as the 200-item audit (SOLVABLE/UNSOLVABLE/AMBIGUOUS, strict); merged sample200 + audit_parts",
       "n_proposed_admitted": len(prop_ids), "n_judged": len(judged), "counts": dict(counts),
       "illposed_share": counts["UNSOLVABLE"] / n, "ambiguous_share": counts["AMBIGUOUS"] / n,
       "sources": dict(src), "missing_ids": missing, "n_extra_ids": len(extra), "verdicts": judged}
print(json.dumps({k: out[k] for k in ("run", "n_proposed_admitted", "n_judged", "counts", "illposed_share", "ambiguous_share")}), "missing", len(missing), "extra", len(extra))
if missing:
    print("MISSING", missing[:30])
    raise SystemExit(1)
tmp = f"{a.run}/audit_full.json.tmp"
json.dump(out, open(tmp, "w"), indent=0, ensure_ascii=False)
os.replace(tmp, f"{a.run}/audit_full.json")
print("wrote", f"{a.run}/audit_full.json")
