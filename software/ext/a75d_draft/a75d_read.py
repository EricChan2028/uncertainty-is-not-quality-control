"""A7-5d read (addendum dirA5/docs/A7_5d_addendum_2026-09-26.md). Inputs: dirA5/results/a75d_key/key_full.jsonl (1,200
items: seed, iteration, run, id, key, original non-blind verdict) and, in dirA5/results/a75d_blind/, pass 1
blind_verdicts.json + blindness_check.json, and re-judge passes 2 and 3 (if run) blind_verdicts_rejudge.json +
blindness_check_rejudge.json and blind_verdicts_rejudge2.json + blindness_check_rejudge2.json. Ill-posed = UNSOLVABLE or
AMBIGUOUS.
Valid verdict: label in {SOLVABLE, UNSOLVABLE, AMBIGUOUS} after strip().upper(); key among the 1,200; line(key) = int(key[1:])
+ 1 inside the verdict's tagged range; that batch passed the blindness check. A key takes the label of the first pass in
which all its valid verdicts agree; a later pass only fills keys still missing. Pass k (k = 2, 3) must cover exactly the
25-line batches that hold a key missing after passes 1..k-1 (the read prints them); no pass after 3.
RB-P1: blind share(it3) >= blind share(it1) + 10 points in both seeds (2/2).
RB-P2: pooled over seeds, |D_blind - D_orig| <= 7.5 points, D = share(it3) - share(it1) on the same items (pooled = all 400
       items of the iteration across the 2 seeds).
RB-P3: binary Cohen's kappa (ill-posed vs solvable) blind vs original over all 1,200 items >= 0.60.
All three need every key to have a blind label, the original verdicts to reproduce from audit_admitted.json item by item,
and the original drift to equal the registered RZ-P1 audits (s0 +23.0, s1 +27.0, pooled +25.0 points) (else None).
Aborts (each means NOT RUN, reported with its reason): a verdict file without its check file; a check whose batches differ
from the registered ones; a verdict tagged with a batch its check never saw; saved verdicts of a passed batch that differ
from every structured return its judges made; checks of different passes sharing a transcript directory; a port outside
{<HOST>, <HOST>}; a pass present while an earlier one is missing or not needed.
While incomplete, only counts and the next pass's ranges are output (no outcome statistics): a75d_read_pass<k>.json after
pass k < 3, a75d_read.json (all verdicts None) once pass 3 has run.
Descriptive: shares per seed x iteration, it2, three-class kappa, per-seed D, paired item bootstrap (4,000 replicates, seed
20260926, items resampled within seed x iteration, the same draw used for blind and original) 95% CIs for D_blind and
D_blind - D_orig, D_blind - D_orig within multi-line vs single-line and CJK vs non-CJK items, per-pass counts, blind
consistency over repeated problem texts, judge sessions with local computation. python3 a75d_read.py"""
import hashlib, json, os, random, re
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA5/results/a75d_blind"; KD = f"{R}/dirA5/results/a75d_key"
K = [json.loads(l) for l in open(f"{KD}/key_full.jsonl", encoding="utf-8")]
VALID = {"SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"}; ILL = {"UNSOLVABLE", "AMBIGUOUS"}
assert all(it["orig_verdict"] in VALID for it in K)
KEYS = {it["key"] for it in K}
line = lambda k: int(k[1:]) + 1
batch_of = lambda ln: [(ln - 1) // 25 * 25 + 1, (ln - 1) // 25 * 25 + 25]
PASSES = [("blind_verdicts.json", "blindness_check.json"), ("blind_verdicts_rejudge.json", "blindness_check_rejudge.json"),
          ("blind_verdicts_rejudge2.json", "blindness_check_rejudge2.json")]

# the original verdicts must reproduce item by item from the registered audits
orig_ok = True; A = {}
for it in K:
    if it["run"] not in A:
        A[it["run"]] = {v["id"]: v["verdict"] for v in json.load(open(f"{R}/dirB/runs/{it['run']}/audit_admitted.json", encoding="utf-8"))["verdicts"]}
    orig_ok &= A[it["run"]].get(it["id"]) == it["orig_verdict"]


def load_pass(vf, cf, expected):
    st = {"file": os.path.basename(vf), "n_expected_batches": len(expected), "n_raw": 0, "n_valid": 0, "n_invalid_label": 0,
          "n_unknown_key": 0, "n_out_of_range": 0, "n_dropped_batch": 0, "n_conflict_keys": 0, "n_malformed": 0}
    raw = json.load(open(vf, encoding="utf-8")); raw = raw["verdicts"] if isinstance(raw, dict) else raw
    assert os.path.exists(cf), f"{vf} exists without its blindness check {cf}"
    chk = json.load(open(cf, encoding="utf-8"))
    listed = chk.get("passed", []) + chk.get("dropped", []) + chk.get("absent", [])
    assert sorted(map(list, expected)) == sorted(map(list, listed)), f"{cf}: batches differ from the registered ones"
    assert chk.get("n_expected") == len(expected) and not chk.get("unexpected"), f"{cf} does not match the registered batches"
    passed = {tuple(x) for x in chk.get("passed", [])}; seen = {tuple(x) for x in chk.get("passed", []) + chk.get("dropped", [])}
    for k in ("dropped", "absent", "transcript_dirs", "port", "n_judge_sessions", "n_sessions_with_local_computation"):
        st[k] = chk.get(k)
    # the saved verdicts of each passed batch must equal one of that batch's structured returns (sorted key/verdict pairs)
    for rg in passed:
        pairs = sorted([str(v.get("key")), str(v.get("verdict"))] for v in raw if isinstance(v, dict) and v.get("lines") == list(rg))
        if not pairs:  # no saved verdicts for this batch: its keys stay missing
            continue
        sha = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
        ok_shas = {s for t in chk["per_batch"][f"{rg[0]}-{rg[1]}"]["transcripts"] for s in t.get("structured_output_pair_sha256", [])}
        assert sha in ok_shas, f"{vf}: verdicts of batch {rg} differ from the judges' structured returns"
    labs = defaultdict(set)
    for v in raw:
        st["n_raw"] += 1
        try:
            lab = str(v.get("verdict") if v.get("verdict") is not None else "").strip().upper(); k = v.get("key")
            rg = [int(x) for x in v.get("lines")] if isinstance(v.get("lines"), list) else None
        except (AttributeError, TypeError, ValueError):
            st["n_malformed"] += 1; continue
        if rg is not None and len(rg) == 2:
            assert tuple(rg) in seen, f"verdict tagged with a batch the blindness check never saw: {rg}"
        if lab not in VALID:
            st["n_invalid_label"] += 1; continue
        if not isinstance(k, str) or not re.fullmatch(r"z\d{4}", k) or k not in KEYS:
            st["n_unknown_key"] += 1; continue
        if not (rg is not None and len(rg) == 2 and rg[0] <= line(k) <= rg[1]):
            st["n_out_of_range"] += 1; continue
        if tuple(rg) not in passed:
            st["n_dropped_batch"] += 1; continue
        st["n_valid"] += 1; labs[k].add(lab)
    res = {k: next(iter(s)) for k, s in labs.items() if len(s) == 1}
    st["n_conflict_keys"] = sum(len(s) > 1 for s in labs.values()); st["n_keys_labelled"] = len(res)
    return res, st


present = [os.path.exists(f"{D}/{vf}") for vf, _ in PASSES]
assert all(present[i] or not any(present[i + 1:]) for i in range(3)), "a later pass exists while an earlier one is missing"
BV, stats, expected, n_run = {}, [], [[a, a + 24] for a in range(1, 1201, 25)], 0
for (vf, cf), pr in zip(PASSES, present):
    if pr and not expected:
        raise AssertionError(f"{vf} exists although no key was missing after the previous pass")
    if not pr:
        break
    res, st = load_pass(f"{D}/{vf}", f"{D}/{cf}", expected)
    filled = 0
    for k, lab in res.items():
        if k not in BV:
            BV[k] = lab; filled += 1
    st["n_keys_filled"] = filled; stats.append(st); n_run += 1
    expected = sorted({tuple(batch_of(line(k))) for k in KEYS - set(BV)}); expected = [list(x) for x in expected]
dirs = [d for st in stats for d in (st["transcript_dirs"] or [])]
assert len(dirs) == len(set(dirs)), "two passes share a transcript directory"
assert all(st["port"] in (<HOST>, <HOST>) for st in stats), "a pass used a port outside the registered set"
missing = sorted(KEYS - set(BV))
cellsize = Counter((it["seed"], it["iteration"]) for it in K)


def od(s, i):
    xs = [it for it in K if it["seed"] == s and it["iteration"] == i]
    return 100 * sum(it["orig_verdict"] in ILL for it in xs) / len(xs)


orig_drift = {"s0": od(0, 3) - od(0, 1), "s1": od(1, 3) - od(1, 1)}
orig_drift["pooled"] = (orig_drift["s0"] + orig_drift["s1"]) / 2
drift_ok = abs(orig_drift["s0"] - 23.0) < 1e-9 and abs(orig_drift["s1"] - 27.0) < 1e-9 and abs(orig_drift["pooled"] - 25.0) < 1e-9
complete = (n_run >= 1 and orig_ok and drift_ok and len(K) == 1200 and len(KEYS) == 1200 and not missing
            and len(cellsize) == 6 and all(n == 200 for n in cellsize.values()))
out = {"n_items": len(K), "n_blind": len(BV), "complete": complete, "orig_verdicts_reproduce": orig_ok,
       "orig_drift_matches_registered": drift_ok, "orig_drift": orig_drift, "passes_run": n_run, "passes": stats,
       "n_missing": len(missing), "next_pass_ranges": expected if (missing and n_run < 3) else []}
if not complete:
    out["verdicts"] = {"RB-P1": None, "RB-P2": None, "RB-P3": None}
    f = f"{D}/a75d_read.json" if n_run >= 3 else f"{D}/a75d_read_pass{n_run}.json"
    json.dump(out, open(f, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    raise SystemExit(0)

out["cells"] = {}
cell = {}
for it in K:
    cell.setdefault((it["seed"], it["iteration"]), []).append((it["orig_verdict"] in ILL, BV[it["key"]] in ILL))


def share(xs, j):
    return 100 * sum(x[j] for x in xs) / len(xs) if xs else None


for (s, i), xs in sorted(cell.items()):
    out["cells"][f"s{s}_it{i}"] = {"n": len(xs), "orig": share(xs, 0), "blind": share(xs, 1)}
rb1 = []; per_seed = {}
for s in (0, 1):
    a, b = cell[(s, 1)], cell[(s, 3)]
    rb1.append(share(b, 1) >= share(a, 1) + 10 - 1e-9)
    per_seed[f"s{s}"] = {"D_blind": share(b, 1) - share(a, 1), "D_orig": share(b, 0) - share(a, 0)}
pool = lambda i: [x for s in (0, 1) for x in cell.get((s, i), [])]
p1, p3 = pool(1), pool(3)
D_blind = share(p3, 1) - share(p1, 1); D_orig = share(p3, 0) - share(p1, 0)


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None, None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / n ** 2
    return ((po - pe) / (1 - pe) if pe < 1 else None), po


kb, pb = kappa([(it["orig_verdict"] in ILL, BV[it["key"]] in ILL) for it in K])
k3, p3a = kappa([(it["orig_verdict"], BV[it["key"]]) for it in K])
rng = random.Random(20260926); bs_d, bs_dd = [], []
for _ in range(4000):
    rs = {c: [xs[rng.randrange(len(xs))] for _ in xs] for c, xs in cell.items()}
    q1 = [x for s in (0, 1) for x in rs[(s, 1)]]; q3 = [x for s in (0, 1) for x in rs[(s, 3)]]
    db = share(q3, 1) - share(q1, 1); do = share(q3, 0) - share(q1, 0)
    bs_d.append(db); bs_dd.append(db - do)
ci = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]]
CJK = re.compile(r"[㐀-鿿豈-﫿]")


def dd(sel):
    xs = [it for it in K if sel(it)]
    g = lambda i, j: [((it["orig_verdict"] in ILL), (BV[it["key"]] in ILL))[j] for it in xs if it["iteration"] == i]
    f = lambda v: 100 * sum(v) / len(v)
    if not g(1, 0) or not g(3, 0):
        return None
    ob, bl = f(g(3, 0)) - f(g(1, 0)), f(g(3, 1)) - f(g(1, 1))
    return {"n_it1": len(g(1, 0)), "n_it3": len(g(3, 0)), "D_orig": ob, "D_blind": bl, "diff": bl - ob}


groups = defaultdict(list)
for it in K:
    groups[it["problem"]].append(it["key"])
rep = [g for g in groups.values() if len(g) > 1]
out.update({"D_blind": D_blind, "D_orig": D_orig, "per_seed": per_seed, "D_blind_ci95": ci(bs_d), "D_diff_ci95": ci(bs_dd),
            "kappa_binary": kb, "raw_agreement_binary": pb, "kappa_3class": k3, "raw_agreement_3class": p3a,
            "pooled_it2": {"orig": share(pool(2), 0), "blind": share(pool(2), 1)},
            "subsets_descriptive": {"multi_line": dd(lambda it: "\n" in it["problem"]),
                                    "single_line": dd(lambda it: "\n" not in it["problem"]),
                                    "cjk": dd(lambda it: bool(CJK.search(it["problem"]))),
                                    "non_cjk": dd(lambda it: not CJK.search(it["problem"]))},
            "repeated_texts": {"n_texts": len(rep), "n_items": sum(map(len, rep)),
                               "n_texts_consistent_binary": sum(len({BV[k] in ILL for k in g}) == 1 for g in rep)}})
out["verdicts"] = {"RB-P1": all(rb1), "RB-P1_per_seed": rb1, "RB-P2": abs(D_blind - D_orig) <= 7.5 + 1e-9,
                   "RB-P3": kb is not None and kb >= 0.60 - 1e-9}
json.dump(out, open(f"{D}/a75d_read.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
