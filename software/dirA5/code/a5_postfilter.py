"""A5 CTRL-L post-guard filter (registered 08:04). Input: a plain R-Zero band guard output dir (admitted.jsonl +
guard_records.jsonl) and a target count N (the guard lineage's admitted count for the same seed and round).
  rnd : random subsample of N admitted items, RNG 20260924 + 100*seed + round
  lowp: drop the lowest-modal-share admitted items until N remain (ties broken at random with the same RNG)
Tracers are treated like proposals (the guard lineage's count also includes tracers). If fewer than N items are admitted,
all are kept and the shortfall is recorded. Writes <out>/admitted.jsonl and <out>/postfilter_stats.json.
python3 a5_postfilter.py <rnd|lowp> <band_dir> <out_dir> <N> <seed> <round>"""
import json, random, sys, os
rule, src, out, N, S, RD = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
os.makedirs(out, exist_ok=True)
A = [json.loads(l) for l in open(f"{src}/admitted_band.jsonl")]
Rr = {r["id"]: r for r in (json.loads(l) for l in open(f"{src}/guard_records.jsonl"))}
rng = random.Random(20260924 + 100 * S + RD)
if len(A) <= N:
    keep = list(A)
elif rule == "rnd":
    keep = rng.sample(A, N)
elif rule == "lowp":
    tie = {a["id"]: rng.random() for a in A}
    keep = sorted(A, key=lambda a: (-Rr[a["id"]]["p_mode"], tie[a["id"]]))[:N]
else:
    raise SystemExit("rule must be rnd or lowp")
rng.shuffle(keep)
with open(f"{out}/admitted.jsonl", "w") as fh:
    for a in keep:
        fh.write(json.dumps(a, ensure_ascii=False) + "\n")
st = {"rule": rule, "band_admitted": len(A), "target": N, "kept": len(keep), "shortfall": max(0, N - len(A)),
      "kept_kinds": {k: sum(1 for a in keep if a["kind"] == k) for k in ("proposed", "tracer_deleted", "tracer_intact")},
      "min_p_mode_kept": min(Rr[a["id"]]["p_mode"] for a in keep) if keep else None}
json.dump(st, open(f"{out}/postfilter_stats.json", "w"), indent=1)
print(json.dumps(st))
