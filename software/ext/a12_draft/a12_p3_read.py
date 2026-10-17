"""A12 part-3 read (A12_registration_2026-09-26.md, section 3.4): an open-weight well-posedness verifier on top of the band.
Inputs (registered, pinned by sha256; any mismatch aborts the read, i.e. part 3 is NOT RUN): the 1,158 fixed-pool proposals
(dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl), the Opus labels (dirA3/results/pool_audit/qwen_s0_r1/parts/*.json, 47
files, pinned as one digest over "<name> <sha256>" lines in name order, read exactly as a3_guard_diag_score.py reads them), the
band decisions per checkpoint (dirA3/results/guard_diag/qwen_s0/<ck>/guard_records.jsonl, ck = base, r1..r5; R-Zero band
[0.25, 0.75], G = 8; records of kind "proposed"), their registered readout (result.json, which the band-alone numbers must
reproduce), and the open-weight verdicts (dirA5/results/a12_open/verdicts.jsonl; A10 parse rule; its sha256 is recorded).
Ill-posed = UNSOLVABLE or AMBIGUOUS (Opus). Verifier rule: admit an in-band proposal only if the open judge says SOLVABLE; a
missing open verdict counts as not SOLVABLE (rejected); sensitivity with missing = SOLVABLE. More than 58 missing (5%) makes
every reading None (NOT RUN), as in A10.
Per checkpoint: s_band = ill-posed share of the band-admitted proposals, s_ver = the same with the verifier on top, admitted
counts, Wilson 95% CIs, the in-band miss rate eps_t = P(open SOLVABLE | Opus ill-posed, in band) and keep rate k_t =
P(open SOLVABLE | Opus SOLVABLE, in band); exact identity odds(s_ver) = odds(s_band) x eps_t / k_t (Proposition 1 form), so
OR_ver / OR_band (odds ratios r5 vs base) = (eps_r5 / k_r5) / (eps_base / k_base).
Pool level (all 1,158): miss rate P(open SOLVABLE | Opus ill-posed) (also for A and U separately), false-reject rate
P(open not SOLVABLE | Opus SOLVABLE), confusion table, binary and three-class kappa.
Paired item bootstrap (the 1,158 proposals resampled with replacement, the same draw at every checkpoint; 4,000 replicates,
random.Random(20260926 + 13)) for D_band = s_band(r5) - s_band(base), D_ver likewise, the level gap s_band - s_ver per checkpoint,
and OR_ver / OR_band. Each CI uses the replicates in which its quantity is defined (D_band, D_ver: s_ver defined at base and r5;
the gaps: s_ver defined at that checkpoint; OR_ver / OR_band: in addition all four odds defined, every share strictly between 0
and 100%) and is undefined with fewer than 40 of them.
PRE-STATED READINGS: level LOWERED if the gap's CI excludes 0 upward at every checkpoint, NOT LOWERED if it never does, else
MIXED; direction (a statement about direction and power only) PERSISTS if D_ver's CI lower bound > 0, REVERSED if its upper
bound < 0, else NOT SHOWN (also when the D_ver CI is undefined; never by itself evidence against Corollary 1); Corollary 1
(mechanism note s.1; with eps / k fixed it predicts OR_ver = OR_band) is read from the CI of OR_ver / OR_band: CONSISTENT
(contains 1: consistent with a constant eps/k), WEAKENS (upper bound < 1: the verifier weakens the drift more than a fixed miss
rate allows), AMPLIFIES (lower bound > 1), NOT DETERMINED (CI undefined: fewer than 40 replicates with all four odds defined);
eps "near 0" (descriptive: the Corollary's escape condition) if the Wilson upper bound of the in-band miss rate at r5 is < 5%.
    python3 a12_p3_read.py"""
import glob, hashlib, json, math, os, random
from collections import Counter
R = "/home/tione/notebook/rsi_rl"
PARTS = f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts"; GD = f"{R}/dirA3/results/guard_diag/qwen_s0"
OW_F = f"{R}/dirA5/results/a12_open/verdicts.jsonl"; OUTF = f"{R}/dirA5/results/a12_open/a12_p3_read.json"
PROP = f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl"
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
VALID = {"SOLVABLE", "UNSOLVABLE", "AMBIGUOUS"}; ILL = {"UNSOLVABLE", "AMBIGUOUS"}
PIN = {PROP: "a97bf54ebae71447ae5d686222083745202f8516f090068d4386d8176b15acb9",
       f"{GD}/result.json": "a250ce8824a7fcaa86871d77b773d05a85996335de71546ea5648d48dd36c7ad",
       f"{GD}/base/guard_records.jsonl": "d7271ee75aa62713519f53f9344aa5a715c5f56fb08d64a45dc47dde99f781cb",
       f"{GD}/r1/guard_records.jsonl": "b9aa0601ec2755f943afd57e12558143d10735cb66a85b065eff23adc3cc78ac",
       f"{GD}/r2/guard_records.jsonl": "5da61cd028448aa6195be14b9755a5f9335f05647ae7ea28535ce6004d17cd03",
       f"{GD}/r3/guard_records.jsonl": "9ccddc3394df8074d90739e66c21b40dd0161684accddf319bf8c579c4269443",
       f"{GD}/r4/guard_records.jsonl": "ca5dc3d652a5fa559d25fb87dfa29ec28b4545cf8a99711f295e2182f0842228",
       f"{GD}/r5/guard_records.jsonl": "6677ec70cfeb37680d39cfeffcf57fe9d47f611f0b8d74523644b929e60ce4bb"}
PARTS_N, PARTS_DIGEST = 47, "d4718c01ee11de3cbeb624772ec507a003bdc67cb20788c265fc2aaf2aa194f7"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def corollary(orci, near0):
    """Registered Corollary-1 reading from the CI of OR_ver / OR_band (section 3.4)."""
    if orci is None:
        st, t = "NOT DETERMINED", ("fewer than 40 of the 4,000 replicates have all four odds defined (a share of 0 or 100%, or no "
                                   "verifier-admitted item); no Corollary reading")
    elif orci[1] < 1:
        st, t = "WEAKENS", ("OR_ver / OR_band CI lies below 1: the verifier weakens the drift more than a fixed miss rate allows "
                            "(eps_t / k_t falls from base to r5; see eps_t and k_t across checkpoints)")
    elif orci[0] > 1:
        st, t = "AMPLIFIES", ("OR_ver / OR_band CI lies above 1: the verifier amplifies the drift (eps_t / k_t rises from base to "
                              "r5; see eps_t and k_t across checkpoints)")
    else:
        st, t = "CONSISTENT", ("OR_ver / OR_band CI contains 1: consistent with Corollary 1, i.e. consistent with a constant eps/k "
                               "(eps_t and k_t are reported across checkpoints)")
    if near0:
        t += ("; eps is near 0 at r5 (the Corollary's escape condition): the verifier removes almost every in-band ill-posed "
              "item, and the odds-ratio test has little power")
    return st, t


DIRECTION_TEXT = {
    "PERSISTS": "the admitted ill-posed share still rises from base to r5 with the verifier on top (D_ver CI lower bound > 0)",
    "NOT SHOWN": ("the direction with the verifier on top is not established at this n (D_ver CI contains 0 or is undefined); "
                  "this alone is not evidence against Corollary 1, which is read from OR_ver / OR_band"),
    "REVERSED": "the admitted ill-posed share falls from base to r5 with the verifier on top (D_ver CI upper bound < 0)"}

for p, h in PIN.items():
    assert sha(p) == h, f"{p} is not the registered input"
pf = sorted(glob.glob(PARTS + "/*.json"))
dig = hashlib.sha256("".join(f"{os.path.basename(f)} {sha(f)}\n" for f in pf).encode()).hexdigest()
assert len(pf) == PARTS_N and dig == PARTS_DIGEST, "the Opus label parts are not the registered ones"
ids = [json.loads(l)["id"] for l in open(PROP, encoding="utf-8")]
assert len(ids) == 1158 and len(set(ids)) == 1158
op, dup_conflict = {}, 0
for f in glob.glob(PARTS + "/*.json"):  # same order and overwrite rule as a3_guard_diag_score.py
    for v in json.load(open(f))["verdicts"]:
        if v["id"] in op and op[v["id"]] != v["verdict"]:
            dup_conflict += 1
        op[v["id"]] = v["verdict"]
assert set(op) == set(ids) and all(v in VALID for v in op.values()), "Opus labels do not cover the 1,158 proposals"
band = {}
for ck in CKS:
    rows = [json.loads(l) for l in open(f"{GD}/{ck}/guard_records.jsonl")]
    band[ck] = {r["id"]: bool(r["admit"]) for r in rows if r["kind"] == "proposed"}
    assert set(band[ck]) == set(ids), ck
reg = json.load(open(f"{GD}/result.json"))
raw_ow = open(OW_F, "rb").read(); OW = {}
for l in raw_ow.decode().splitlines():
    r = json.loads(l)
    if r["key"] in op:
        OW[r["key"]] = r.get("verdict") if r.get("verdict") in VALID else None
n_missing = sum(OW.get(k) is None for k in ids)
ok = n_missing <= 58
solv_open = lambda k, miss_as_solvable=False: (OW.get(k) == "SOLVABLE") or (miss_as_solvable and OW.get(k) is None)


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def stats(sample, ck, mas=False):
    B = [k for k in sample if band[ck][k]]
    V = [k for k in B if solv_open(k, mas)]
    bi = sum(op[k] in ILL for k in B); vi = sum(op[k] in ILL for k in V)
    BI = [k for k in B if op[k] in ILL]; BS = [k for k in B if op[k] == "SOLVABLE"]
    eps = sum(solv_open(k, mas) for k in BI) / len(BI) if BI else None
    kp = sum(solv_open(k, mas) for k in BS) / len(BS) if BS else None
    return {"n_band": len(B), "s_band": 100 * bi / len(B) if B else None, "n_ver": len(V), "s_ver": 100 * vi / len(V) if V else None,
            "eps_inband": eps, "keep_inband": kp, "_bi": bi, "_vi": vi, "_nbi": len(BI)}


odds = lambda p: p / (100 - p) if p is not None and 0 < p < 100 else None
per = {ck: stats(ids, ck) for ck in CKS}
reprod = all(per[ck]["n_band"] == reg[ck]["admitted_candidates"]["n"]
             and abs(per[ck]["s_band"] - 100 * reg[ck]["admitted_candidates"]["illposed"]) < 1e-9 for ck in CKS)
rng = random.Random(20260926 + 13); bs = {"D_band": [], "D_ver": [], "or_ratio": []}; gap = {ck: [] for ck in CKS}
for _ in range(4000):
    smp = [ids[rng.randrange(len(ids))] for _ in ids]
    st = {ck: stats(smp, ck) for ck in CKS}
    for ck in CKS:
        if st[ck]["s_ver"] is not None:
            gap[ck].append(st[ck]["s_band"] - st[ck]["s_ver"])
    if st["r5"]["s_ver"] is not None and st["base"]["s_ver"] is not None:
        bs["D_band"].append(st["r5"]["s_band"] - st["base"]["s_band"]); bs["D_ver"].append(st["r5"]["s_ver"] - st["base"]["s_ver"])
        ob, ov = [odds(st[c]["s_band"]) for c in ("base", "r5")], [odds(st[c]["s_ver"]) for c in ("base", "r5")]
        if None not in ob + ov and ov[0] > 0 and ob[0] > 0:
            bs["or_ratio"].append((ov[1] / ov[0]) / (ob[1] / ob[0]))
ci = lambda v: [sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1]] if len(v) >= 40 else None
D_band = per["r5"]["s_band"] - per["base"]["s_band"]
D_ver = (per["r5"]["s_ver"] - per["base"]["s_ver"]) if None not in (per["r5"]["s_ver"], per["base"]["s_ver"]) else None
gci = {ck: ci(gap[ck]) for ck in CKS}; dci = ci(bs["D_ver"]); orci = ci(bs["or_ratio"])
ob0, ov0 = [odds(per[c]["s_band"]) for c in ("base", "r5")], [odds(per[c]["s_ver"]) for c in ("base", "r5")]
or_point = ((ov0[1] / ov0[0]) / (ob0[1] / ob0[0])) if None not in ob0 + ov0 and ov0[0] > 0 and ob0[0] > 0 else None
lowered = [g is not None and g[0] > 0 for g in gci.values()]
level = "LOWERED" if all(lowered) else ("NOT LOWERED" if not any(lowered) else "MIXED")
direction = "NOT SHOWN" if dci is None else ("PERSISTS" if dci[0] > 0 else ("REVERSED" if dci[1] < 0 else "NOT SHOWN"))
e5 = per["r5"]; eps_w = wilson(round(e5["eps_inband"] * e5["_nbi"]), e5["_nbi"]) if e5["_nbi"] else None
near0 = eps_w is not None and eps_w[1] < 5.0
cor_state, cor_text = corollary(orci, near0)
ILLp = [k for k in ids if op[k] in ILL]; SOLp = [k for k in ids if op[k] == "SOLVABLE"]


def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n; ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else None


have = [k for k in ids if OW.get(k) is not None]
run_ok = ok and reprod
why = "> 58 missing verdicts" if not ok else "band numbers do not reproduce result.json"
out = {"inputs": {"n_items": 1158, "n_open_verdicts": len(have), "n_missing": n_missing, "open_verdicts_sha256": hashlib.sha256(raw_ow).hexdigest(),
                  "pinned_inputs_sha256": {os.path.relpath(p, R): h for p, h in PIN.items()}, "opus_parts_digest": dig,
                  "opus_label_counts": dict(Counter(op.values())), "opus_duplicate_conflicts": dup_conflict,
                  "band_alone_reproduces_result_json": reprod},
       "rule": "admit an in-band proposal only if the open judge says SOLVABLE; missing counts as not SOLVABLE",
       "per_checkpoint": {ck: {"n_band": v["n_band"], "s_band": v["s_band"], "s_band_wilson95": wilson(v["_bi"], v["n_band"]),
                               "n_ver": v["n_ver"], "s_ver": v["s_ver"], "s_ver_wilson95": wilson(v["_vi"], v["n_ver"]),
                               "gap_band_minus_ver": (v["s_band"] - v["s_ver"]) if v["s_ver"] is not None else None, "gap_ci95": gci[ck],
                               "yield_ver_over_band": v["n_ver"] / v["n_band"] if v["n_band"] else None,
                               "eps_inband": v["eps_inband"], "keep_inband": v["keep_inband"],
                               "eps_over_k": (v["eps_inband"] / v["keep_inband"]) if v["eps_inband"] is not None and v["keep_inband"] else None,
                               "odds_identity_check": (abs(odds(v["s_ver"]) - odds(v["s_band"]) * v["eps_inband"] / v["keep_inband"]) < 1e-9)
                               if None not in (odds(v["s_ver"]), odds(v["s_band"]), v["eps_inband"]) and v["keep_inband"] else None}
                          for ck, v in per.items()},
       "drift": {"D_band": D_band, "D_band_ci95": ci(bs["D_band"]), "D_ver": D_ver, "D_ver_ci95": dci,
                 "OR_ver_over_OR_band": or_point, "OR_ver_over_OR_band_ci95": orci, "n_or_replicates": len(bs["or_ratio"]),
                 "n_d_replicates": len(bs["D_ver"]), "n_gap_replicates": {ck: len(gap[ck]) for ck in CKS}},
       "pool_level": {"miss_rate_on_opus_illposed": sum(solv_open(k) for k in ILLp) / len(ILLp),
                      "miss_rate_ambiguous": sum(solv_open(k) for k in ILLp if op[k] == "AMBIGUOUS") / max(1, sum(op[k] == "AMBIGUOUS" for k in ids)),
                      "miss_rate_unsolvable": sum(solv_open(k) for k in ILLp if op[k] == "UNSOLVABLE") / max(1, sum(op[k] == "UNSOLVABLE" for k in ids)),
                      "false_reject_rate_on_opus_solvable": sum(not solv_open(k) for k in SOLp) / len(SOLp),
                      "confusion_open_vs_opus": dict(Counter(f"{OW.get(k)}|{op[k]}" for k in ids)),
                      "kappa_binary": kappa([(OW[k] in ILL, op[k] in ILL) for k in have]) if have else None,
                      "kappa_3class": kappa([(OW[k], op[k]) for k in have]) if have else None},
       "sensitivity_missing_as_solvable": {ck: {k2: v[k2] for k2 in ("n_ver", "s_ver")} for ck, v in
                                           ((ck, stats(ids, ck, True)) for ck in CKS)},
       "readings": {"level": level if run_ok else None, "direction": direction if run_ok else None,
                    "direction_text": DIRECTION_TEXT[direction] if run_ok else None,
                    "corollary": cor_state if run_ok else "NOT RUN",
                    "corollary_text": cor_text if run_ok else f"NOT RUN ({why})",
                    "eps_r5_inband_wilson95": eps_w, "eps_near_0": near0 if run_ok else None}}
json.dump(out, open(OUTF, "w"), indent=1)
print(json.dumps(out, indent=1))
