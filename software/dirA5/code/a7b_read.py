"""A7b read (addendum registered 17:12 24 Sep, dirA5/docs/A7b_addendum_2026-09-24.md). Inputs: dirA5/results/a7b_blind/
key_full.jsonl (900 items: seed, round, key, original non-blind verdict) and blind_verdicts.json (the 36 blind judges' verdicts,
one per key). Ill-posed = UNSOLVABLE or AMBIGUOUS.
B-P1: blind share(r5) >= blind share(r1) + 10 points in >= 2 of 3 seeds.
B-P2: pooled over seeds, |D_blind - D_orig| <= 7.5 points, D = share(r5) - share(r1) on the same items (pooled = all 300
      items of the round across the 3 seeds).
B-P3: binary Cohen's kappa (ill-posed vs solvable) blind vs original over all 900 items >= 0.60.
All three need every one of the 900 keys to have a blind verdict (else None). Descriptive: shares per seed x round, r3,
three-class kappa, paired item bootstrap (4,000 replicates, seed 20260926, items resampled within seed x round, the same draw
used for blind and original) 95% CIs for D_blind and D_blind - D_orig. python3 a7b_read.py"""
import json, random
from collections import Counter
D = "/home/tione/notebook/rsi_rl/dirA5/results/a7b_blind"
K = [json.loads(l) for l in open(f"{D}/key_full.jsonl")]
VALID = {"SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"}; ILL = {"UNSOLVABLE", "AMBIGUOUS"}
assert all(it["orig_verdict"] in VALID for it in K)
KEYS = {it["key"] for it in K}
raw = json.load(open(f"{D}/blind_verdicts.json")); seen = {}; n_invalid = n_norm = 0; conflict = set()
for v in raw:
    lab0 = v.get("verdict"); lab = str(lab0 if lab0 is not None else "").strip().upper()
    if lab not in VALID: n_invalid += 1; continue
    n_norm += lab != lab0
    k = v.get("key")
    if k in seen and seen[k] != lab: conflict.add(k)
    seen.setdefault(k, lab)
BV = {k: l for k, l in seen.items() if k in KEYS and k not in conflict}
cellsize = {}
for it in K: cellsize[(it["seed"], it["round"])] = cellsize.get((it["seed"], it["round"]), 0) + 1
complete = (len(K) == 900 and len(KEYS) == 900 and all(k in BV for k in KEYS) and len(cellsize) == 9 and all(n == 100 for n in cellsize.values()))
out = {"n_items": len(K), "n_blind": sum(it["key"] in BV for it in K), "complete": complete, "n_invalid_labels": n_invalid,
       "n_normalised_labels": n_norm, "n_conflicting_duplicate_keys": len(conflict), "n_extra_keys": len(set(seen) - KEYS),
       "rule": "labels normalised by strip().upper(); invalid labels and conflicting duplicate keys count as missing; any missing -> verdicts None",
       "blindness_check": "all 36 judges ran exactly one command: sed -n <range> on dirA5/results/a7b_blind/items.jsonl (transcripts wf_4d05cd5c-73f); none read key_full.jsonl",
       "cells": {}}
cell = {}
for it in K:
    if it["key"] not in BV: continue
    cell.setdefault((it["seed"], it["round"]), []).append((it["orig_verdict"] in ILL, BV[it["key"]] in ILL))
def share(xs, j): return 100 * sum(x[j] for x in xs) / len(xs) if xs else None
for (s, rd), xs in sorted(cell.items()):
    out["cells"][f"s{s}_r{rd}"] = {"n": len(xs), "orig": share(xs, 0), "blind": share(xs, 1)}
bp1 = []
for s in range(3):
    a, b = cell.get((s, 1)), cell.get((s, 5))
    bp1.append(None if not a or not b else share(b, 1) >= share(a, 1) + 10 - 1e-9)
pool = lambda rd: [x for s in range(3) for x in cell.get((s, rd), [])]
p1, p5 = pool(1), pool(5)
D_blind = share(p5, 1) - share(p1, 1) if p1 and p5 else None; D_orig = share(p5, 0) - share(p1, 0) if p1 and p5 else None
def kappa(pairs):
    n = len(pairs)
    if n == 0: return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / n ** 2
    return ((po - pe) / (1 - pe) if pe < 1 else None), po
kb, pb = kappa([(it["orig_verdict"] in ILL, BV[it["key"]] in ILL) for it in K if it["key"] in BV])
k3, p3 = kappa([(it["orig_verdict"], BV[it["key"]]) for it in K if it["key"] in BV])
rng = random.Random(20260926); bs_d, bs_dd = [], []
for _ in range(4000):
    rs = {c: [xs[rng.randrange(len(xs))] for _ in xs] for c, xs in cell.items()}
    q1 = [x for s in range(3) for x in rs.get((s, 1), [])]; q5 = [x for s in range(3) for x in rs.get((s, 5), [])]
    db = share(q5, 1) - share(q1, 1); do = share(q5, 0) - share(q1, 0)
    bs_d.append(db); bs_dd.append(db - do)
ci = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]]
out.update({"D_blind": D_blind, "D_orig": D_orig, "D_blind_ci95": ci(bs_d), "D_diff_ci95": ci(bs_dd),
            "kappa_binary": kb, "raw_agreement_binary": pb, "kappa_3class": k3, "raw_agreement_3class": p3,
            "pooled_r3": {"orig": share(pool(3), 0), "blind": share(pool(3), 1)}})
out["verdicts"] = {"B-P1": None if not complete else sum(x is True for x in bp1) >= 2, "B-P1_per_seed": bp1,
                   "B-P2": None if not complete or D_blind is None else abs(D_blind - D_orig) <= 7.5 + 1e-9,
                   "B-P3": None if not complete or kb is None else kb >= 0.60 - 1e-9}
json.dump(out, open(f"{D}/a7b_read.json", "w"), indent=1)
print(json.dumps(out, indent=1))
