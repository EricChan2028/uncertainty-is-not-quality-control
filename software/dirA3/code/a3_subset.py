"""Training-subset selection for the A3 attribution arms (amendment A3c, 2026-09-22).
  sub:   uniformly random fraction of the admitted set (rng seed given)
  clean: admitted proposer questions judged SOLVABLE in audit_full.json + admitted intact tracers
  dirty: admitted proposer questions judged UNSOLVABLE/AMBIGUOUS + admitted deleted tracers
python3 a3_subset.py --admitted admitted.jsonl --mode sub --frac 0.5 --seed 1 --out admitted_sub.jsonl
python3 a3_subset.py --admitted admitted.jsonl --mode clean --audit audit_full.json --out admitted_clean.jsonl
"""
import argparse
import collections
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--admitted", required=True)
ap.add_argument("--mode", choices=["sub", "clean", "dirty"], required=True)
ap.add_argument("--frac", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--audit")
ap.add_argument("--out", required=True)
a = ap.parse_args()
A = [json.loads(l) for l in open(a.admitted)]
if a.mode == "sub":
    rng = random.Random(a.seed)
    keep = rng.sample(A, round(a.frac * len(A)))
else:
    aud = json.load(open(a.audit))
    V = {v["id"]: v["verdict"] for v in aud["verdicts"]}
    prop = [x for x in A if x["kind"] == "proposed"]
    missing = [x["id"] for x in prop if x["id"] not in V]
    if missing:
        raise SystemExit(f"audit missing {len(missing)} admitted proposer ids, e.g. {missing[:5]}")
    if a.mode == "clean":
        keep = [x for x in A if (x["kind"] == "proposed" and V[x["id"]] == "SOLVABLE") or x["kind"] == "tracer_intact"]
    else:
        keep = [x for x in A if (x["kind"] == "proposed" and V[x["id"]] in ("UNSOLVABLE", "AMBIGUOUS")) or x["kind"] == "tracer_deleted"]
with open(a.out, "w") as f:
    for x in keep:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
st = {"mode": a.mode, "n_admitted": len(A), "n_kept": len(keep), "kinds_kept": dict(collections.Counter(x["kind"] for x in keep)),
      "kinds_admitted": dict(collections.Counter(x["kind"] for x in A)), "frac": a.frac, "seed": a.seed, "audit": a.audit}
json.dump(st, open(a.out.rsplit(".", 1)[0] + "_stats.json", "w"), indent=1)
print(json.dumps(st))
