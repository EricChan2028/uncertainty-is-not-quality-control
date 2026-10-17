"""A12 part-1 read (A12_registration_2026-09-26.md, section 1.6). Inputs: dirA5/results/a12_key/key_full.jsonl (1,200 raw items:
seed, iteration, run, raw_id, downstream status, key), dirA5/results/a12_key/stage_s<S>_meta.json, and in dirA5/results/a12_blind/
the items file items.jsonl (must decode to key_full's (key, problem) pairs in order) and the passes blind_verdicts.json +
blindness_check.json, blind_verdicts_rejudge.json + blindness_check_rejudge.json, blind_verdicts_rejudge2.json +
blindness_check_rejudge2.json. Comparison inputs (registered, hash-checked): the six
dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/audit_admitted.json (original admitted audits, restricted to the 200 ids of
admitted.jsonl) and dirA5/results/a75d_blind/a75d_read.json (A7-5d blind re-audit of the same admitted items); a hash mismatch
of any of these seven files aborts the read.
Merge rule (verbatim from a75d_read.py, keys y0000..y1199): a verdict is valid if its label is SOLVABLE / UNSOLVABLE / AMBIGUOUS
after strip().upper(), its key is one of the 1,200, line(key) = int(key[1:]) + 1 lies in its tagged range and that batch passed
the blindness check; a key takes the label of the first pass in which all its valid verdicts agree; pass k (k = 2, 3) must cover
exactly the 25-line batches holding a key still missing; no pass after 3. The check file must be an a12_blind_check.py check of
the registered items file (never the pilot). A batch that passed the check but has no verdict in the saved file aborts the
read, unless every structured return of that batch was rejected by the harness (a null return; its keys stay missing and it
is listed). While keys are missing only counts and the next pass's ranges are written (a12_p1_read_pass<k>.json); after pass 3
with keys missing the part is NOT RUN (a12_p1_read.json, reading "NOT RUN").
Ill-posed = UNSOLVABLE or AMBIGUOUS.
PRIMARY (descriptive, pre-stated reading): D_raw = raw ill-posed share(it3) - share(it1), pooled over seeds (400 items per
iteration), item-bootstrap 95% CI (4,000 replicates, random.Random(20260926 + 12), items resampled within seed x iteration).
  RISES: CI lower bound > 0 | FALLS: CI upper bound < 0 | FLAT: CI inside [-10, +10] points and containing 0 |
  INCONCLUSIVE: CI contains 0 but reaches beyond +-10 points.
DOWNSTREAM-SELECTION SHARE (pre-stated, second state): Delta log E = Delta logit(ill-posed share among the in-sample admitted
items) - Delta logit(raw share), it3 vs it1, pooled over seeds, with its CI from the same replicates; E is the enrichment of all
downstream stages, i.e. of downstream selection (pseudo-labelling and the 0.3-0.8 filter): POSITIVE (CI lower bound > 0) |
NEGATIVE (CI upper bound < 0) | NOT ESTABLISHED (CI contains 0, or undefined). The words "the drift sits in downstream
selection" appear only with POSITIVE (reading_text below). BAND SHARE (SECONDARY, a descriptive label): the same three states
from the CI of Delta log E(scored), E among scored items only (status != not_scored), which isolates the 0.3-0.8 filter from the
pseudo-labelling drops; reported beside the primary state (reading.band_share_secondary), with no wording attached.
DESCRIPTIVE: per seed x iteration shares with Wilson 95% CIs (ill-posed, UNSOLVABLE, AMBIGUOUS); per-seed D_raw; comparison with
the admitted shares (original audits; A7-5d blind) and D_adm - D_raw with a bootstrap CI (original audits, items independent);
D_raw / D_adm; the framing-free within-sample decomposition (status admitted / scored_rejected / not_scored; admission rate of
ill-posed vs solvable raw items, enrichment E = a_ill / a_solv, logit(share among in-sample admitted) = logit(raw share) +
log E exactly, E among scored items only, and the challenger's fraction of the admitted log-odds drift); the condition check
(per seed, then averaged over seeds: in-sample admitted share minus the A7-5d blind admitted share (primary reference) and
minus the original audit share (secondary), per iteration and pooled over iterations with a bootstrap CI; the flag is raised
when the pooled CI against A7-5d blind excludes 0); agreement on raw items whose text equals an audited admitted item of the
same run; covariates per cell (parse-failure rate, not-scored rate, CJK, multi-line, length, repeated text).
    python3 a12_p1_read.py"""
import hashlib, json, math, os, random, re
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA5/results/a12_blind"; KD = f"{R}/dirA5/results/a12_key"
REG_ITEMS = f"{D}/items.jsonl"
VALID = {"SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"}; ILL = {"UNSOLVABLE", "AMBIGUOUS"}
line = lambda k: int(k[1:]) + 1
batch_of = lambda ln: [(ln - 1) // 25 * 25 + 1, (ln - 1) // 25 * 25 + 25]
PASSES = [("blind_verdicts.json", "blindness_check.json"), ("blind_verdicts_rejudge.json", "blindness_check_rejudge.json"),
          ("blind_verdicts_rejudge2.json", "blindness_check_rejudge2.json")]
AUDIT_SHA = {(0, 1): "81ad37b312184dcb348ea99b2d2a070e62be129d47a9cbf9a22a3835975cbb68",
             (0, 2): "4cdc864798717369d741361c95135cd56dfd619d05f89bf784a7357eb6f7a29a",
             (0, 3): "957f294e40d0a3b9df3d44c5669a9ebe94c3a5c134551006d9275af02060962d",
             (1, 1): "301597a2607f4d5eab128529e8100084f3aaff19c16a28d05611e9dcc2c108b0",
             (1, 2): "685a9a1b54a4f798f7fd6385734adf164f3925e5d547bd31bc5b1d5f493d5669",
             (1, 3): "6700e71f2a9ca8f42e0b76438e80b719aad1e9768982860cd4d010f94a7d5a32"}
A75D_READ_SHA = "7c9bd8bd17d285c7b810450f2561d0c5566ceeb74e95d1e8bd7413e5deb87d7b"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def share_state(ci95):
    """Pre-stated state from the CI of Delta log E (primary: downstream selection) or of Delta log E(scored) (secondary: band)."""
    if ci95 is None:
        return "NOT ESTABLISHED"
    return "POSITIVE" if ci95[0] > 0 else ("NEGATIVE" if ci95[1] < 0 else "NOT ESTABLISHED")


DS = "downstream selection (pseudo-labelling and the 0.3-0.8 filter)"


def reading_text(label, fstate, ci95, d_raw, d_adm):
    """Registered wording of the part-1 reading (section 1.6). fstate = downstream-selection state (CI of Delta log E); ci95 =
    CI of D_raw; d_adm = registered admitted drift (pooled)."""
    if label == "INCONCLUSIVE":
        return "the direction of the raw challenger share is not determined at 400 items per iteration; numbers only"
    if label == "RISES":
        t = (f"the challenger's raw output drifts upward (D_raw {d_raw:+.1f} points, CI [{ci95[0]:+.1f}, {ci95[1]:+.1f}]); A12 does "
             "not separate its cause (the uncertainty reward, the repetition penalty or the format reward): consistent with, "
             "not proof of, the uncertainty reward")
        return t + {"POSITIVE": f"; {DS} adds drift on top (Delta log E CI > 0): the drift acts at both stages",
                    "NEGATIVE": f"; the enrichment by {DS} weakens over iterations (Delta log E CI < 0)",
                    "NOT ESTABLISHED": f"; the added share of {DS} is not established (Delta log E CI contains 0 or is undefined)"}[fstate]
    if label == "FLAT":
        y = f", i.e. < {100 * ci95[1] / d_adm:.0f}% of the admitted drift of {d_adm:+.1f}" if d_adm > 0 else ""
        t = f"no raw drift detected; raw drift within the CI [{ci95[0]:+.1f}, {ci95[1]:+.1f}] (< {ci95[1]:.1f} points{y})"
        return t + (f"; the drift sits in {DS} (Delta log E CI > 0)" if fstate == "POSITIVE"
                    else "; raw flat; share of downstream selection not established (Delta log E CI does not lie above 0)")
    if label == "FALLS":
        t = (f"the challenger writes fewer ill-posed questions over iterations (D_raw {d_raw:+.1f} points, CI "
             f"[{ci95[0]:+.1f}, {ci95[1]:+.1f}])")
        return t + (f"; the drift sits in {DS} (Delta log E CI > 0)" if fstate == "POSITIVE"
                    else "; raw falls; share of downstream selection not established (Delta log E CI does not lie above 0)")
    raise ValueError(label)


K = [json.loads(l) for l in open(f"{KD}/key_full.jsonl", encoding="utf-8")]
KEYS = {it["key"] for it in K}
ITEMS = [json.loads(l) for l in open(REG_ITEMS, encoding="utf-8")]
assert [(x["key"], x["problem"]) for x in ITEMS] == [(x["key"], x["problem"]) for x in K] and all(set(x) == {"key", "problem"} for x in ITEMS), \
    "items.jsonl does not decode to key_full's (key, problem) pairs in order"


def load_pass(vf, cf, expected):
    st = {"file": os.path.basename(vf), "n_expected_batches": len(expected), "n_raw": 0, "n_valid": 0, "n_invalid_label": 0,
          "n_unknown_key": 0, "n_out_of_range": 0, "n_dropped_batch": 0, "n_conflict_keys": 0, "n_malformed": 0,
          "passed_null_return": []}
    raw = json.load(open(vf, encoding="utf-8")); raw = raw["verdicts"] if isinstance(raw, dict) else raw
    assert os.path.exists(cf), f"{vf} exists without its blindness check {cf}"
    chk = json.load(open(cf, encoding="utf-8"))
    assert chk.get("checker") == "a12_blind_check.py" and chk.get("items_file") == REG_ITEMS and chk.get("pilot") is False, \
        f"{cf} is not an a12_blind_check.py check of the registered items file"
    listed = chk.get("passed", []) + chk.get("dropped", []) + chk.get("absent", [])
    assert sorted(map(list, expected)) == sorted(map(list, listed)), f"{cf}: batches differ from the registered ones"
    assert chk.get("n_expected") == len(expected) and not chk.get("unexpected"), f"{cf} does not match the registered batches"
    passed = {tuple(x) for x in chk.get("passed", [])}; seen = {tuple(x) for x in chk.get("passed", []) + chk.get("dropped", [])}
    for k in ("dropped", "absent", "transcript_dirs", "port", "n_judge_sessions", "n_sessions_with_local_computation"):
        st[k] = chk.get(k)
    for rg in sorted(passed):
        pairs = sorted([str(v.get("key")), str(v.get("verdict"))] for v in raw if isinstance(v, dict) and v.get("lines") == list(rg))
        trs = chk["per_batch"][f"{rg[0]}-{rg[1]}"]["transcripts"]
        if not pairs:
            # legitimate only for a null return: every structured return of the batch was rejected by the harness
            acc = [a for t in trs if not t.get("neutral") for a in (t.get("structured_output_accepted")
                                                                    or [True] * len(t.get("structured_output_pair_sha256", [])))]
            assert not any(acc), f"{vf}: batch {list(rg)} passed the check and its judge returned verdicts, but the saved file has none"
            st["passed_null_return"].append(list(rg)); continue
        s = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
        ok_shas = {x for t in trs for x in t.get("structured_output_pair_sha256", [])}
        assert s in ok_shas, f"{vf}: verdicts of batch {rg} differ from the judges' structured returns"
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
        if not isinstance(k, str) or not re.fullmatch(r"y\d{4}", k) or k not in KEYS:
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
    expected = [list(x) for x in sorted({tuple(batch_of(line(k))) for k in KEYS - set(BV)})]
dirs = [d for st in stats for d in (st["transcript_dirs"] or [])]
assert len(dirs) == len(set(dirs)), "two passes share a transcript directory"
assert all(st["port"] in (<HOST>, <HOST>) for st in stats), "a pass used a port outside the registered set"
missing = sorted(KEYS - set(BV))
cellsize = Counter((it["seed"], it["iteration"]) for it in K)
complete = (n_run >= 1 and len(K) == 1200 and len(KEYS) == 1200 and not missing and len(cellsize) == 6
            and all(n == 200 for n in cellsize.values()))
out = {"part": "A12 part 1 (raw R-Zero challenger audit)", "n_items": len(K), "n_blind": len(BV), "complete": complete,
       "passes_run": n_run, "passes": stats, "n_missing": len(missing),
       "next_pass_ranges": expected if (missing and n_run < 3) else [], "items_sha256": sha(REG_ITEMS),
       "key_sha256": sha(f"{KD}/key_full.jsonl")}
if not complete:
    out["reading"] = {"label": "NOT RUN" if n_run >= 3 or n_run == 0 else "INCOMPLETE (next pass mandatory)"}
    f = f"{D}/a12_p1_read.json" if n_run >= 3 else f"{D}/a12_p1_read_pass{n_run}.json"
    json.dump(out, open(f, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    raise SystemExit(0)

# ---------------- statistics
ill = {it["key"]: BV[it["key"]] in ILL for it in K}


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def share(xs):
    return 100 * sum(xs) / len(xs) if xs else None


def ci(v):
    v = sorted(x for x in v if x is not None)
    return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if len(v) >= 40 else None


mean = lambda xs: (sum(xs) / len(xs)) if xs and None not in xs else None
cell = defaultdict(list)
for it in K:
    cell[(it["seed"], it["iteration"])].append(it)
out["cells"] = {}
for (s, i), xs in sorted(cell.items()):
    n = len(xs); ki = sum(ill[x["key"]] for x in xs)
    ku = sum(BV[x["key"]] == "UNSOLVABLE" for x in xs); ka = sum(BV[x["key"]] == "AMBIGUOUS" for x in xs)
    out["cells"][f"s{s}_it{i}"] = {"n": n, "raw_illposed": 100 * ki / n, "raw_illposed_wilson95": wilson(ki, n),
                                   "raw_unsolvable": 100 * ku / n, "raw_unsolvable_wilson95": wilson(ku, n),
                                   "raw_ambiguous": 100 * ka / n, "raw_ambiguous_wilson95": wilson(ka, n)}
pool = lambda c, i: [x for s in (0, 1) for x in c.get((s, i), [])]
lab_ill = lambda xs: [ill[x["key"]] for x in xs]
lab_uns = lambda xs: [BV[x["key"]] == "UNSOLVABLE" for x in xs]


def d_of(c, f, s=None):
    a, b = (c[(s, 1)], c[(s, 3)]) if s is not None else (pool(c, 1), pool(c, 3))
    return share(f(b)) - share(f(a))


# registered admitted audits (original, non-blind), item level
adm = {}
for (s, i), h in AUDIT_SHA.items():
    run = f"rzero_qwen3_4b_Base_s{s}_it{i}"; af = f"{R}/dirB/runs/{run}/audit_admitted.json"
    assert sha(af) == h, f"{af} is not the registered audit"
    ids = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open(f"{R}/dirB/runs/{run}/admitted.jsonl", encoding="utf-8")}
    vs = [v for v in json.load(open(af, encoding="utf-8"))["verdicts"] if v["id"] in ids]
    assert len(vs) == 200 and len({v["id"] for v in vs}) == 200, (run, len(vs))
    adm[(s, i)] = [{"ill": v["verdict"] in ILL, "verdict": v["verdict"], "problem": ids[v["id"]]} for v in vs]
adm_ill = lambda xs: [x["ill"] for x in xs]
D_adm = {"s0": d_of(adm, adm_ill, 0), "s1": d_of(adm, adm_ill, 1), "pooled": d_of(adm, adm_ill)}
adm_ok = all(abs(D_adm[k] - v) < 1e-9 for k, v in (("s0", 23.0), ("s1", 27.0), ("pooled", 25.0)))
a75 = f"{R}/dirA5/results/a75d_blind/a75d_read.json"   # pinned like the six audits: a mismatch aborts the read
assert os.path.exists(a75) and sha(a75) == A75D_READ_SHA, f"{a75} is not the registered A7-5d blind read"
A75 = json.load(open(a75, encoding="utf-8"))
BLIND = {(s, i): (A75["cells"][f"s{s}_it{i}"]["blind"], A75["cells"][f"s{s}_it{i}"]["n"]) for s in (0, 1) for i in (1, 2, 3)}

st_of = lambda x: x["status"]
adm_rate = lambda xs: (sum(st_of(x) == "admitted" for x in xs) / len(xs)) if xs else None
ratio = lambda a, b: (a / b) if (a is not None and b) else None
logit = lambda p: math.log(p / (100 - p)) if p is not None and 0 < p < 100 else None
dlt = lambda a, b: (b - a) if (a is not None and b is not None) else None
lg = lambda e: math.log(e) if e else None


def decomp(c, i):
    xs = pool(c, i)
    I = [x for x in xs if ill[x["key"]]]; S = [x for x in xs if not ill[x["key"]]]
    Isc = [x for x in I if st_of(x) != "not_scored"]; Ssc = [x for x in S if st_of(x) != "not_scored"]
    aI, aS, aIs, aSs = adm_rate(I), adm_rate(S), adm_rate(Isc), adm_rate(Ssc)
    A_ = [x for x in xs if st_of(x) == "admitted"]
    by_seed = {}
    for s in (0, 1):
        As = [x for x in c.get((s, i), []) if st_of(x) == "admitted"]
        by_seed[f"s{s}"] = share(lab_ill(As)) if As else None
    return {"a_ill": aI, "a_solv": aS, "E": ratio(aI, aS), "a_ill_scored": aIs, "a_solv_scored": aSs, "E_scored": ratio(aIs, aSs),
            "s_raw": share(lab_ill(xs)), "s_adm_in": share(lab_ill(A_)) if A_ else None, "n_adm_in": len(A_),
            "s_adm_in_by_seed": by_seed, "n_adm_in_by_seed": {f"s{s}": sum(st_of(x) == "admitted" for x in c.get((s, i), [])) for s in (0, 1)},
            "status_shares": {k: share([st_of(x) == k for x in xs]) for k in ("admitted", "scored_rejected", "not_scored")}}


def logodds(d1, d3):
    lr = dlt(logit(d1["s_raw"]), logit(d3["s_raw"])); la = dlt(logit(d1["s_adm_in"]), logit(d3["s_adm_in"]))
    return {"delta_logit_raw": lr, "delta_logit_admitted_in_sample": la, "delta_log_E": dlt(lr, la),
            "delta_log_E_scored": dlt(lg(d1["E_scored"]), lg(d3["E_scored"])),
            "challenger_fraction": (lr / la) if (lr is not None and la) else None}


def cond(decs, adm_c, blind):
    """Condition check: per iteration, the in-sample admitted share averaged over seeds minus the reference averaged over
    seeds (blind: A7-5d blind cell shares; orig: the original audits); pooled = mean over the three iterations."""
    r = {"per_iteration": {}}
    for i in (1, 2, 3):
        si = mean([decs[i]["s_adm_in_by_seed"][f"s{s}"] for s in (0, 1)])
        ro = mean([share(adm_ill(adm_c[(s, i)])) for s in (0, 1)])
        rb = mean([blind[(s, i)] for s in (0, 1)]) if blind else None
        r["per_iteration"][f"it{i}"] = {"in_sample_admitted_seed_mean": si, "a75d_blind": rb, "registered_original": ro,
                                        "diff_vs_blind": dlt(rb, si), "diff_vs_original": dlt(ro, si)}
    for k in ("diff_vs_blind", "diff_vs_original"):
        r[f"pooled_{k}"] = mean([r["per_iteration"][f"it{i}"][k] for i in (1, 2, 3)])
    return r


# bootstrap (one stream; raw and admitted items resampled independently within seed x iteration; the A7-5d blind cell shares,
# available only as shares, are redrawn as binomial(n, share))
rng = random.Random(20260926 + 12)
bs = defaultdict(list)
for _ in range(4000):
    rc = {c: [xs[rng.randrange(len(xs))] for _ in xs] for c, xs in cell.items()}
    ra = {c: [xs[rng.randrange(len(xs))] for _ in xs] for c, xs in adm.items()}
    rb = {c: 100 * sum(rng.random() < p / 100 for _ in range(n)) / n for c, (p, n) in BLIND.items()}
    dr = d_of(rc, lab_ill); da = d_of(ra, adm_ill)
    bs["D_raw"].append(dr); bs["D_adm_minus_D_raw"].append(da - dr); bs["ratio"].append(dr / da if da > 0 else None)
    bs["D_raw_uns"].append(d_of(rc, lab_uns))
    for s in (0, 1):
        bs[f"D_raw_s{s}"].append(d_of(rc, lab_ill, s))
    decs = {i: decomp(rc, i) for i in (1, 2, 3)}
    bs["D_adm_in"].append(dlt(decs[1]["s_adm_in"], decs[3]["s_adm_in"]))
    lo = logodds(decs[1], decs[3])
    for k in ("delta_log_E", "delta_log_E_scored", "challenger_fraction"):
        bs[k].append(lo[k])
    for i in (1, 2, 3):
        bs[f"E_it{i}"].append(decs[i]["E"]); bs[f"E_scored_it{i}"].append(decs[i]["E_scored"])
    cc = cond(decs, ra, rb)
    for k in ("pooled_diff_vs_blind", "pooled_diff_vs_original"):
        bs[k].append(cc[k])

D_raw = d_of(cell, lab_ill); CI = ci(bs["D_raw"])
if CI[0] > 0:
    label = "RISES"
elif CI[1] < 0:
    label = "FALLS"
elif CI[0] >= -10 - 1e-9 and CI[1] <= 10 + 1e-9:
    label = "FLAT"
else:
    label = "INCONCLUSIVE"
dec = {i: decomp(cell, i) for i in (1, 2, 3)}
LO = logodds(dec[1], dec[3])
dle_ci = ci(bs["delta_log_E"]); fstate = share_state(dle_ci)
dles_ci = ci(bs["delta_log_E_scored"]); bstate = share_state(dles_ci)   # secondary: the band alone
out["primary"] = {"D_raw_pooled": D_raw, "ci95": CI, "n_it1": len(pool(cell, 1)), "n_it3": len(pool(cell, 3)),
                  "pooled_raw_share": {f"it{i}": share(lab_ill(pool(cell, i))) for i in (1, 2, 3)}}
out["selection_share"] = {"delta_log_E": LO["delta_log_E"], "ci95": dle_ci, "state": fstate,
                          "note": "Delta log E = Delta logit(in-sample admitted) - Delta logit(raw), it3 vs it1, pooled over seeds; "
                                  "E is the enrichment of downstream selection (pseudo-labelling and the 0.3-0.8 filter)"}
out["band_share_secondary"] = {"delta_log_E_scored": LO["delta_log_E_scored"], "ci95": dles_ci, "state": bstate,
                               "note": "SECONDARY, descriptive label: the same states for Delta log E among scored items only "
                                       "(status != not_scored), which isolates the 0.3-0.8 filter; no wording attached"}
out["reading"] = {"label": label, "selection_share": fstate, "band_share_secondary": bstate,
                  "text": reading_text(label, fstate, CI, D_raw, D_adm["pooled"])}
C0 = cond(dec, adm, {c: p for c, (p, _) in BLIND.items()})
cci = {k: ci(bs[k]) for k in ("pooled_diff_vs_blind", "pooled_diff_vs_original")}
flag = (None if cci["pooled_diff_vs_blind"] is None
        else (cci["pooled_diff_vs_blind"][0] > 0 or cci["pooled_diff_vs_blind"][1] < 0))
methods = Counter(it["status_method"] for it in K)
# raw items whose text equals an audited admitted item of the same run
ov = []
for it in K:
    for a in adm[(it["seed"], it["iteration"])]:
        if a["problem"] == it["problem"]:
            ov.append((ill[it["key"]], a["ill"], BV[it["key"]], a["verdict"])); break
CJK = re.compile(r"[㐀-鿿豈-﫿]")
meta = {S: json.load(open(f"{KD}/stage_s{S}_meta.json", encoding="utf-8")) for S in (0, 1)}
cov = {}
for (s, i), xs in sorted(cell.items()):
    L = sorted(len(x["problem"]) for x in xs); mc = meta[s]["cells"][f"it{i}"]
    cov[f"s{s}_it{i}"] = {"parse_failure_rate": 100 * (mc["n_generated"] - mc["n_parsed"]) / mc["n_generated"],
                          "not_scored_rate_all_parsed": 100 * mc["status_counts_all_parsed"].get("not_scored", 0) / mc["n_parsed"],
                          "cjk": share([bool(CJK.search(x["problem"])) for x in xs]),
                          "multi_line": share(["\n" in x["problem"] for x in xs]), "median_chars": L[len(L) // 2],
                          "repeated_text": share([x["text_multiplicity"] > 1 for x in xs]),
                          "status_counts": dict(Counter(x["status"] for x in xs)),
                          "illposed_by_status": {st: {"n": len(ys), "share": share(lab_ill(ys)),
                                                      "wilson95": wilson(sum(lab_ill(ys)), len(ys))}
                                                 for st in ("admitted", "scored_rejected", "not_scored")
                                                 for ys in [[x for x in xs if x["status"] == st]]}}
out["descriptive"] = {
    "note": "no thresholds registered in this block except the condition flag (pooled CI against A7-5d blind excludes 0)",
    "per_seed_D_raw": {f"s{s}": {"D": d_of(cell, lab_ill, s), "ci95": ci(bs[f"D_raw_s{s}"])} for s in (0, 1)},
    "D_raw_unsolvable_only": {"D": d_of(cell, lab_uns), "ci95": ci(bs["D_raw_uns"])},
    "admitted_original": {"cells": {f"s{s}_it{i}": share(adm_ill(adm[(s, i)])) for (s, i) in sorted(adm)}, "D": D_adm,
                          "matches_registered_RZ_P1": adm_ok},
    "admitted_a75d_blind": {"sha256": A75D_READ_SHA, "cells": {k: v.get("blind") for k, v in A75.get("cells", {}).items()},
                            "D_pooled": A75.get("D_blind"), "per_seed": A75.get("per_seed")},
    "D_adm_orig_minus_D_raw": {"point": D_adm["pooled"] - D_raw, "ci95": ci(bs["D_adm_minus_D_raw"])},
    "D_raw_over_D_adm_orig": {"point": D_raw / D_adm["pooled"] if D_adm["pooled"] > 0 else None, "ci95": ci(bs["ratio"])},
    "within_sample": {"status_method_counts": dict(methods), "by_iteration": {f"it{i}": dec[i] for i in (1, 2, 3)},
                      "E_ci95": {f"it{i}": ci(bs[f"E_it{i}"]) for i in (1, 2, 3)},
                      "E_scored_ci95": {f"it{i}": ci(bs[f"E_scored_it{i}"]) for i in (1, 2, 3)},
                      "D_adm_in_sample": {"point": dlt(dec[1]["s_adm_in"], dec[3]["s_adm_in"]), "ci95": ci(bs["D_adm_in"])},
                      "logodds": dict(LO, delta_log_E_ci95=dle_ci, delta_log_E_scored_ci95=dles_ci,
                                      challenger_fraction_ci95=ci(bs["challenger_fraction"]))},
    "condition_check": dict(C0, pooled_ci95=cci, primary_reference="A7-5d blind", flag=flag),
    "overlap_with_audited_admitted": {"n": len(ov), "binary_agreement": (sum(a == b for a, b, _, _ in ov) / len(ov)) if ov else None,
                                      "three_class_agreement": (sum(c == d for _, _, c, d in ov) / len(ov)) if ov else None},
    "covariates": cov,
    "populations": {f"s{S}": meta[S]["cells"] for S in (0, 1)}}
json.dump(out, open(f"{D}/a12_p1_read.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
