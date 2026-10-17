"""A10 read (registered 01:13 25 Sep, dirA5/docs/A10_registration_2026-09-25.md), v2 after a two-reviewer check (v1 never run).
Inputs: dirA5/results/a10/verdicts.jsonl (open-weight judge), dirA5/results/a7b_blind/key_full.jsonl (seed, round, original
non-blind Opus verdict) and blind_verdicts.json (A7b blind Opus verdicts, normalised exactly as a7b_read.py: strip().upper(),
invalid labels and conflicting duplicate keys -> missing). Ill-posed = UNSOLVABLE or AMBIGUOUS.
CONFIRMATORY
J2-P1: pooled over seeds, share(r5) - share(r1) >= 10 points and the item-bootstrap 95% CI of the difference (4,000 replicates,
       random.Random(20260925), items resampled within seed x round) excludes 0.
J2-P2: binary Cohen's kappa (open-weight vs blind Opus) >= 0.40.
Rule: shares and kappa use the items with a verdict (the registration allows <= 5% missing = 45 of 900); more missing makes
both None. The judge's actual parse rule (disclosed): the last case-insensitive match of "VERDICT: <label>" anywhere in the
output (asterisks allowed); unparsable -> one retry at temperature 0.7, seed 1.
DESCRIPTIVE (no thresholds registered): per seed x round shares, per-seed r5 - r1, r3, three-class kappa and raw agreement vs
blind and vs original Opus, overall ill-posed rate, confusion table, retry counts, parse audits on the stored 300-character
tails (multiple distinct VERDICT labels; the template echo; strict last-line parse) with a strict-parse sensitivity of D and
kappa, bounds on D counting all missing as ill-posed / as solvable, and D for both judges on the same items with a paired
bootstrap of their difference. python3 a10_read.py"""
import hashlib, json, random, re
from collections import Counter
D = "/home/tione/notebook/rsi_rl/dirA5/results"
VALID = {"SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"}; ILL = {"UNSOLVABLE", "AMBIGUOUS"}
K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a7b_blind/key_full.jsonl")}
assert len(K) == 900 and all(it["orig_verdict"] in VALID for it in K.values())
csz = Counter((it["seed"], it["round"]) for it in K.values())
assert set(csz) == {(s, r) for s in (0, 1, 2) for r in (1, 3, 5)} and all(n == 100 for n in csz.values())
seen, conflict = {}, set()
for v in json.load(open(f"{D}/a7b_blind/blind_verdicts.json")):
    lab = str(v.get("verdict") or "").strip().upper()
    if lab not in VALID: continue
    k = v.get("key")
    if k in seen and seen[k] != lab: conflict.add(k)
    seen.setdefault(k, lab)
BO = {k: l for k, l in seen.items() if k in K and k not in conflict}
assert len(BO) == 900
raw = open(f"{D}/a10/verdicts.jsonl", "rb").read(); OWR = {}; dup = unknown = 0
for l in raw.decode().splitlines():
    r = json.loads(l)
    if r["key"] not in K: unknown += 1; continue
    if r["key"] in OWR: dup += 1
    OWR[r["key"]] = r
OW = {k: r["verdict"] for k, r in OWR.items() if r.get("verdict") in VALID}
n_miss = sum(k not in OW for k in K)
def kappa(pairs):
    n = len(pairs)
    if not n: return None
    po = sum(a == b for a, b in pairs) / n; ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else None
share = lambda xs: 100 * sum(xs) / len(xs) if xs else None
pool = lambda c, rd: [x for s in range(3) for x in c.get((s, rd), [])]
def cells(lab, keys):
    c = {}
    for k in sorted(keys, key=lambda k: list(K).index(k)):
        c.setdefault((K[k]["seed"], K[k]["round"]), []).append(lab(k))
    return c
def diff_ci(c, seed=20260925, B=4000):
    p1, p5 = pool(c, 1), pool(c, 5)
    if not p1 or not p5: return None, None
    d = share(p5) - share(p1); rng = random.Random(seed); bs = []
    for _ in range(B):
        rs = {cc: [xs[rng.randrange(len(xs))] for _ in xs] for cc, xs in c.items()}
        bs.append(share(pool(rs, 5)) - share(pool(rs, 1)))
    bs.sort(); return d, [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]
have = [k for k in K if k in OW]
c_ow = cells(lambda k: OW[k] in ILL, have)
d, ci = diff_ci(c_ow)
kb = kappa([(OW[k] in ILL, BO[k] in ILL) for k in have])
ok = n_miss <= 45
out = {"inputs": {"n_items": 900, "n_verdicts": len(OW), "n_missing": n_miss, "duplicate_keys": dup, "unknown_keys": unknown,
                  "verdicts_sha256": hashlib.sha256(raw).hexdigest(), "blind_opus_conflicts": len(conflict)},
       "rule": "J2-P1 shares and J2-P2 kappa use the items with an open-weight verdict (<= 45 missing allowed)",
       "confirmatory": {"J2-P1": {"diff_r5_r1": d, "ci95": ci, "threshold": 10, "n_r1": len(pool(c_ow, 1)), "n_r5": len(pool(c_ow, 5)),
                                  "holds": None if not ok or d is None else (d >= 10 - 1e-9 and ci[0] > 0)},
                        "J2-P2": {"kappa_binary_vs_blind_opus": kb, "n_pairs": len(have), "threshold": 0.40,
                                  "holds": None if not ok or kb is None else kb >= 0.40 - 1e-9}}}
# descriptive
strict = re.compile(r"^\W*VERDICT\W*:\W*(SOLVABLE|UNSOLVABLE|AMBIGUOUS)\W*$", re.I)
anyv = re.compile(r"VERDICT:\s*\**\s*(SOLVABLE|UNSOLVABLE|AMBIGUOUS)", re.I)
multi = echo = 0; SP = {}
for k, r in OWR.items():
    t = r.get("raw_tail") or ""; labs = {m.upper() for m in anyv.findall(t)}
    multi += len(labs) > 1; echo += all(x in t.upper() for x in ("VERDICT: SOLVABLE", "VERDICT: UNSOLVABLE", "VERDICT: AMBIGUOUS"))
    lines = [x for x in t.strip().splitlines() if x.strip()]
    m = strict.match(lines[-1]) if lines else None
    if m: SP[k] = m.group(1).upper()
c_sp = cells(lambda k: SP[k] in ILL, [k for k in K if k in SP]); d_sp, ci_sp = diff_ci(c_sp)
c_hi = cells(lambda k: (OW[k] in ILL) if k in OW else True, list(K)); c_lo = cells(lambda k: (OW[k] in ILL) if k in OW else False, list(K))
# both judges on the same items, paired bootstrap
c_pair = cells(lambda k: (OW[k] in ILL, BO[k] in ILL), have)
def dpair(c):
    p1, p5 = pool(c, 1), pool(c, 5)
    return (share([a for a, _ in p5]) - share([a for a, _ in p1])), (share([b for _, b in p5]) - share([b for _, b in p1]))
d_ow, d_bo = dpair(c_pair); rng = random.Random(20260925); bsd = []
for _ in range(4000):
    rs = {cc: [xs[rng.randrange(len(xs))] for _ in xs] for cc, xs in c_pair.items()}
    a, b = dpair(rs); bsd.append(a - b)
bsd.sort()
out["descriptive"] = {
    "note": "no thresholds registered for anything in this block",
    "cells": {f"s{s}_r{rd}": {"n": len(v), "share_illposed": share(v)} for (s, rd), v in sorted(c_ow.items())},
    "pooled": {"r1": share(pool(c_ow, 1)), "r3": share(pool(c_ow, 3)), "r5": share(pool(c_ow, 5))},
    "per_seed_diff_r5_r1": {s: share(c_ow[(s, 5)]) - share(c_ow[(s, 1)]) for s in range(3) if (s, 5) in c_ow and (s, 1) in c_ow},
    "kappa_3class_vs_blind_opus": kappa([(OW[k], BO[k]) for k in have]),
    "raw_agreement_binary_vs_blind_opus": sum((OW[k] in ILL) == (BO[k] in ILL) for k in have) / len(have) if have else None,
    "kappa_binary_vs_original_opus": kappa([(OW[k] in ILL, K[k]["orig_verdict"] in ILL) for k in have]),
    "kappa_3class_vs_original_opus": kappa([(OW[k], K[k]["orig_verdict"]) for k in have]),
    "raw_agreement_binary_vs_original_opus": sum((OW[k] in ILL) == (K[k]["orig_verdict"] in ILL) for k in have) / len(have) if have else None,
    "overall_illposed_rate": share([OW[k] in ILL for k in have]),
    "confusion_ow_vs_blind_opus": dict(Counter(f"{OW[k]}|{BO[k]}" for k in have)),
    "retries": {"n_retried": sum(1 for r in OWR.values() if r.get("retry")), "n_recovered": sum(1 for r in OWR.values() if r.get("retry") and r.get("verdict") in VALID)},
    "parse_audit": {"tails_with_multiple_distinct_verdicts": multi, "tails_with_template_echo": echo, "strict_last_line_parsed": len(SP),
                    "strict_parse_diff_r5_r1": d_sp, "strict_parse_ci95": ci_sp,
                    "strict_parse_kappa_vs_blind_opus": kappa([(SP[k] in ILL, BO[k] in ILL) for k in SP])},
    "missing_bounds_diff": {"all_missing_illposed": diff_ci(c_hi)[0], "all_missing_solvable": diff_ci(c_lo)[0]},
    "both_judges_same_items": {"D_open_weight": d_ow, "D_blind_opus": d_bo, "D_diff_open_minus_opus": d_ow - d_bo,
                               "paired_ci95": [bsd[100], bsd[3899]]}}
json.dump(out, open(f"{D}/a10/a10_read.json", "w"), indent=1)
print(json.dumps(out, indent=1))
