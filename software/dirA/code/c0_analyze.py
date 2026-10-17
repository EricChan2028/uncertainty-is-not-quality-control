"""Direction A, Cycle 0 readout: Q1-Q5 exactly as registered (C0 + amendment C0a).

Reads dirA/results/<tag>_shard*.jsonl for each model tag and prints, per model:
  Q1  blind-spot mass: P(p_mode >= 0.5 | deleted)                       K1: < 0.15 kills the direction
  Q2  guard admission: P(admitted | deleted) for R-Zero band, AZR learnability, consistency>=0.5,
      and the deleted/intact admission ratio within p_mode bins
  Q3  probe separation: AUC(deleted vs intact) for s(q), P2 (assumed), self-judge, within p_mode bins
      (matched confidence), pooled by bin weight                         K2: < 0.75; K3: judge within 0.05
  Q4  family generality (Q1 on non-Qwen models)
  Q5  stratification by inversion depth and split kind
Bootstrap intervals are family-clustered (families are the sampling unit here; there is no seed axis
because nothing is trained). Smoke records are ignored by construction (tag != smoke).
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

D = "/home/tione/notebook/rsi_rl/dirA/results"
BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]


def load(tag):
    recs = []
    for f in glob.glob(f"{D}/{tag}_shard*.jsonl"):
        for l in open(f):
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    return recs


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); None-scored items are dropped."""
    pos = [x for x in pos if x is not None]
    neg = [x for x in neg if x is not None]
    if not pos or not neg:
        return None
    neg_s = sorted(neg)
    import bisect
    tot = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg_s, p)
        hi = bisect.bisect_right(neg_s, p)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def binned_auc(recs, key):
    """AUC within p_mode bins, pooled with bin weights = number of deleted items in the bin."""
    num = den = 0.0
    parts = []
    for lo, hi in BINS:
        dele = [r[key] for r in recs if r["form"] == "deleted" and lo <= r["p_mode"] < hi]
        inta = [r[key] for r in recs if r["form"] == "intact" and lo <= r["p_mode"] < hi]
        a = auc(dele, inta)
        parts.append((lo, hi, len(dele), len(inta), a))
        if a is not None and dele:
            num += a * len(dele)
            den += len(dele)
    return (num / den if den else None), parts


def guards(r):
    p = r["p_mode"]
    return {"rzero_band": 0.25 <= p <= 0.75, "azr_learnable": 0.0 < p < 1.0 and r["nonanswer"] < 1.0,
            "consistent>=0.5": p >= 0.5}


def clustered(recs, fn, rng, B=2000):
    fams = defaultdict(list)
    for r in recs:
        fams[r["fam"]].append(r)
    keys = sorted(fams)
    pt = fn([r for k in keys for r in fams[k]])
    draws = []
    for _ in range(B):
        s = [keys[rng.randrange(len(keys))] for _ in keys]
        v = fn([r for k in s for r in fams[k]])
        if v is not None:
            draws.append(v)
    draws.sort()
    return pt, draws[int(0.025 * len(draws))] if draws else None, draws[int(0.975 * len(draws)) - 1] if draws else None


def main():
    rng = random.Random(0)
    tags = sys.argv[1:] or sorted({os.path.basename(f).split("_shard")[0] for f in glob.glob(f"{D}/*_shard*.jsonl")} - {"smoke"})
    for tag in tags:
        recs = load(tag)
        dele = [r for r in recs if r["form"] == "deleted"]
        inta = [r for r in recs if r["form"] == "intact"]
        if not dele or not inta:
            print(f"\n== {tag}: incomplete ({len(dele)} deleted, {len(inta)} intact)")
            continue
        print("=" * 96)
        print(f"{tag}: {len(dele)} deleted, {len(inta)} intact items, {len(set(r['fam'] for r in recs))} families")
        print("=" * 96)
        # sanity: truncation and non-answers
        print(f"  truncation rate: deleted {sum(r['trunc_rate'] for r in dele)/len(dele):.3f}  intact {sum(r['trunc_rate'] for r in inta)/len(inta):.3f}"
              f"   | intact accuracy (modal==ref) {sum(r['correct'] for r in inta)/len(inta):.3f}   deleted 'lucky' {sum(r['correct'] for r in dele)/len(dele):.3f}")
        # Q1
        q1, lo, hi = clustered(dele, lambda rs: sum(1 for r in rs if r["p_mode"] >= 0.5) / len(rs), rng)
        print(f"\n  Q1 blind-spot mass P(p_mode>=0.5 | deleted) = {q1:.3f} [{lo:.3f}, {hi:.3f}]   {'PASSES K1' if q1 >= 0.15 else 'K1 KILL (<0.15)'}")
        q1i = sum(1 for r in inta if r["p_mode"] >= 0.5) / len(inta)
        print(f"     (intact items with p_mode>=0.5: {q1i:.3f}; confident-and-WRONG among deleted: {sum(1 for r in dele if r['p_mode']>=0.5 and not r['correct'])/len(dele):.3f})")
        # Q2
        print("\n  Q2 guard admission of DELETED items (and deleted/intact ratio at matched p_mode):")
        for g in ("rzero_band", "azr_learnable", "consistent>=0.5"):
            ad = sum(1 for r in dele if guards(r)[g]) / len(dele)
            ai = sum(1 for r in inta if guards(r)[g]) / len(inta)
            print(f"     {g:16s} admitted-deleted {ad:.3f}   admitted-intact {ai:.3f}   {'blind (>=0.30)' if ad >= 0.30 else ''}")
        # Q3
        print("\n  Q3 separation of deleted vs intact at matched confidence (binned AUC; >=0.75 usable):")
        res = {}
        for key, name in (("s_probe", "s(q) number-doubling"), ("p2_assumed", "P2 assumption-declared"), ("judge_missing", "self-judge")):
            if not any(key in r for r in recs):
                continue
            a, parts = binned_auc(recs, key)
            res[key] = a
            print(f"     {name:24s} pooled AUC {a if a is None else round(a,3)}   bins: " + " ".join(f"[{lo:.2f},{hi:.2f}) n={nd}/{ni} auc={('%.2f' % x) if x is not None else '-'}" for lo, hi, nd, ni, x in parts))
        j = res.get("judge_missing")
        for key, name in (("s_probe", "s(q)"), ("p2_assumed", "P2")):
            a = res.get(key)
            if a is None:
                continue
            verdict = "K2 fails (<0.75)" if a < 0.75 else ("K3: judge matches, probe redundant" if (j is not None and j >= a - 0.05) else "USABLE, beats judge")
            print(f"     -> {name}: {verdict}")
        # Q5
        print("\n  Q5 by inversion depth (deleted items): depth: n, blind-spot mass, P2 AUC")
        for d in sorted({r.get("depth") for r in dele if r.get("depth") is not None}):
            sub = [r for r in recs if r.get("depth") == d]
            dd = [r for r in sub if r["form"] == "deleted"]
            a, _ = binned_auc(sub, "p2_assumed") if any("p2_assumed" in r for r in sub) else (None, None)
            print(f"     depth {d}: n={len(dd)}  mass={sum(1 for r in dd if r['p_mode']>=0.5)/max(1,len(dd)):.2f}  P2 auc={a if a is None else round(a,2)}")
        for sk in ("value", "template"):
            sub = [r for r in dele if r.get("split_kind") == sk]
            if sub:
                print(f"     split {sk:8s}: n={len(sub)} blind-spot mass {sum(1 for r in sub if r['p_mode']>=0.5)/len(sub):.3f}")
    print("\nQ4 (family generality): compare Q1 across the tags above; needs >=0.15 on a non-Qwen model.")


if __name__ == "__main__":
    main()
