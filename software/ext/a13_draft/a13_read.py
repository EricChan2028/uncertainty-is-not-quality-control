#!/usr/bin/env python3
"""A13 read (registration A13_registration_2026-09-27.md, section 6). CPU only. Reads the dumps; writes ONE file,
ext/a13_runs/a13_read.json, opened with mode "x": a second run refuses (exit 3) and changes nothing.

Arms (Qwen3-4B, one round of 60 steps from the round-2 solver of seed s, s = 0..4):
  S  = cln1_a6 (solvable admitted problems; no planted premise-deleted problem)      dumps of 2026-09-24
  R  = rnd1_a6 (random subset of the admitted band of size |S|; 20-39 planted deletions)  dumps of 2026-09-24
  R0 = r0_a13  (R with every planted deletion replaced in place by a random admitted proposal; a13_build.py)
Metric, copied from paper_arr/analyses/attribution.py (the Table 3 script): plain-prompt UMWP dump plain_<run>.jsonl, rows
de-duplicated by id (last row wins), exactly 600 unanswerable and 600 answerable rows required; CA = share of the 600
unanswerable rows with mode not None and p_mode >= 0.5; ACC_A = 100 * mean "correct" over the 600 answerable rows;
bench = 100 * mean greedy_acc_lenient over math500_train, math_heldout1000, gsm8k_test500 (b_eval summary).
Paired seed-level differences (same seed), mean +- t_{n-1,0.975} * sd / sqrt(n), sd = statistics.stdev (attribution.paired).
Check (gates every reading): the stored R-S contrast is recomputed from the dumps and must equal
attribution.json paired["cln1-rnd1:CA"] (mean, sd, ci95, diffs; |diff| < 1e-12) and round to the paper's -5.6 [-9.0, -2.1].
Registered quantities: E_prop = CA(R0) - CA(S); E_plant = CA(R) - CA(R0); identity E_prop + E_plant = CA(R) - CA(S) per seed.
Readings of E_prop (95% CI, points): ADDS if lower bound > 0; REDUCES if upper bound < 0; NOT DETECTED otherwise, with the
sub-case "estimate <= 0" (mean <= 0) that also qualifies the claims row and the conclusion.
A seed counts only if its R0 run passes every acceptance test (section 6.2); with fewer than 4 accepted seeds, or a failed
check, the result is NOT RUN. E_plant, benchmark accuracy, ACC_A and the corrected-abstention numbers are descriptive.
Usage: python3 a13_read.py                  registered read (once)
       python3 a13_read.py --check_only     recompute the stored R-S contrast from the existing dumps; writes nothing
       python3 a13_read.py --root DEV --constants DEV/c.json --out DEV/o.json     dev tests on synthetic trees only
"""
import argparse, hashlib, json, math, os, statistics, sys
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/home/tione/notebook/rsi_rl")
ap.add_argument("--out", default=None, help="default <root>/ext/a13_runs/a13_read.json")
ap.add_argument("--constants", default=None, help="dev only: JSON overriding the registered constants")
ap.add_argument("--check_only", action="store_true")
args = ap.parse_args()
REAL = "/home/tione/notebook/rsi_rl"
R = os.path.abspath(args.root)
if args.constants and R == REAL:
    sys.exit("--constants is for synthetic trees only")
UM, X = f"{R}/dirA4/results/umwp", f"{R}/ext/a13_runs"
RUNS_OLD, EV_OLD, EV_A13, RUNS_A13 = f"{R}/dirB/runs", f"{R}/dirB/results/evals", f"{X}/evals", f"{X}/runs"
ATTR = f"{R}/paper_arr/analyses/attribution.json"
OUT = args.out or f"{X}/a13_read.json"
BENCH = ("math500_train", "math_heldout1000", "gsm8k_test500")
SEEDS = range(5)
NAME = {"S": "qwen3_4b_Base_cln1_a6_s%d_r3", "R": "qwen3_4b_Base_rnd1_a6_s%d_r3", "R0": "qwen3_4b_Base_r0_a13_s%d_r3"}

K = {  # registered constants
    "umwp_probe_sha256": "32de59f95a296fa0c026b44cf8a6110172898b87b258179ab3a54da00ee1e56d",
    "absfix_probe_sha256": "11f796c2af5c56afe72c802cf34e444e391c1b53f10561e5c115bb0b7f6cfa68",
    "b_eval_sha256": "5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189",
    "b_train_sha256": "fb9156b0ef6a685029c14c2ca52c4f17b6f0302a97c657c43cffeec1945d97d7",
    "r0_sha256": {"0": "a4cd93eb55911a77b8adfd6fb5a0ac41d6b28718cc2e348807863fb285e0e64c",
                  "1": "8b9a2f5723628edc226a2756b9369ce1ae0e831e284595c4c546b7c5bbcaa9b3",
                  "2": "f3c672c53c1b33a2cb37a18e78b8f555113aff28995ea77eb6302135a8c9c27b",
                  "3": "ab04539c7d4f1d3b3423c2196d79362cb3acb85a4468974ea9d3e299d726338a",
                  "4": "52bb2cf1326833742a4f6a17c65c048cc705ae98e64bd0793d3404b0944e34f4"},
    "old_sha256": {  # the existing R/S inputs of the Table 3 contrast (2026-09-24 runs), as of 2026-09-27
        "dirA4/results/umwp/plain_qwen3_4b_Base_cln1_a6_s0_r3.jsonl": "f9141ce7816f5e0aa016f2b87d4304841ba6bb68554b40cef564dc978d311499",
        "dirA4/results/umwp/plain_qwen3_4b_Base_cln1_a6_s1_r3.jsonl": "f10cc95a3d9bdf187b5402cceb5fe4916bf07559785c0a3f31c96d0adf0ef076",
        "dirA4/results/umwp/plain_qwen3_4b_Base_cln1_a6_s2_r3.jsonl": "17ce0fd87f71fdfdc75982fb45f7f5ad2cf6baf37e74af2b09af04a34a77ab71",
        "dirA4/results/umwp/plain_qwen3_4b_Base_cln1_a6_s3_r3.jsonl": "bf1205eb16341f17a638c936545acceb0b6e186f1e8f0f2112bd97eb298bd76d",
        "dirA4/results/umwp/plain_qwen3_4b_Base_cln1_a6_s4_r3.jsonl": "f7940fbdb97f2dd9cfd2a0cabf101ca63788de1c5df601ded95624e3c68cba08",
        "dirA4/results/umwp/plain_qwen3_4b_Base_rnd1_a6_s0_r3.jsonl": "183841405d4b0d121130bc34c4f6ff8e4d4805a0078ab9bb02d616f67b90e48c",
        "dirA4/results/umwp/plain_qwen3_4b_Base_rnd1_a6_s1_r3.jsonl": "3cc917fb60113ea484e60e514f55a8dac83a9e7cf757583ef0017ea6e5c4a469",
        "dirA4/results/umwp/plain_qwen3_4b_Base_rnd1_a6_s2_r3.jsonl": "e59b7c12283f4736f8b87a4333f09998be6ec90bedfc12106e5b72eeefa288a0",
        "dirA4/results/umwp/plain_qwen3_4b_Base_rnd1_a6_s3_r3.jsonl": "3ffea605c7a034ec8841d772191148e194085111c5e9173c03f0efc3a019d616",
        "dirA4/results/umwp/plain_qwen3_4b_Base_rnd1_a6_s4_r3.jsonl": "365eced03a530ce85ef3256d306eae4a1b13b7112bac22d4d73ba0531c99b948",
        "dirB/results/evals/qwen3_4b_Base_cln1_a6_s0_r3_step60.summary.json": "87cf755d4133c561343b5d3378f551d9399971eea8183c33efade343e00c5e88",
        "dirB/results/evals/qwen3_4b_Base_cln1_a6_s1_r3_step60.summary.json": "5da742e715c564d479144372c39ab21c04983ebd904ca0a0fe7218447d67b485",
        "dirB/results/evals/qwen3_4b_Base_cln1_a6_s2_r3_step60.summary.json": "b82682efa79f8410950772ceb6782a3c743f418b53192d630b90a8325cab6499",
        "dirB/results/evals/qwen3_4b_Base_cln1_a6_s3_r3_step60.summary.json": "62c35618f38e5cfe498251901111e02d2c453014eecc9e99db25231d40455078",
        "dirB/results/evals/qwen3_4b_Base_cln1_a6_s4_r3_step60.summary.json": "6ae70bff97347c45ee0f5bf9a55dc397eed0c123e201783dfb7df6fa7615794e",
        "dirB/results/evals/qwen3_4b_Base_rnd1_a6_s0_r3_step60.summary.json": "23d27b6bd218672d728bb8a15c7a123e734a0f11e275f3b8ba5c7c8924f65359",
        "dirB/results/evals/qwen3_4b_Base_rnd1_a6_s1_r3_step60.summary.json": "380e6582453ef21c166d5e2bd4dc3a264c15a89a3b0210f8c8d930a8e22da5ca",
        "dirB/results/evals/qwen3_4b_Base_rnd1_a6_s2_r3_step60.summary.json": "b673599b55e7f853d88c6a09b432db222160132c810a0c25f15774e604083447",
        "dirB/results/evals/qwen3_4b_Base_rnd1_a6_s3_r3_step60.summary.json": "5d9eca59b0c4d7092c382b3019dcf889797f393a09536435ca2f2006100a8ea4",
        "dirB/results/evals/qwen3_4b_Base_rnd1_a6_s4_r3_step60.summary.json": "017e9f0c9ddf4a7811fa1050e067475bd0139c97041e612c95856c772b643f7d",
        "paper_arr/analyses/attribution.json": "6146256cad82dcaea2076fcd088aac0e8479429acedc5c6d2c7b0e99022c479f"},
    "paper_rs": [-5.6, -9.0, -2.1], "paper_rs_bench": 0.8, "min_seeds": 4,
    "config_ignore": ["out", "data"],
}
if args.constants:
    K.update(json.load(open(args.constants)))

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def jload(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- copied from attribution.py (Table 3 script) ---
def umwp(tag):
    f = os.path.join(UM, "%s.jsonl" % tag)
    rows = {}
    for l in open(f):
        try:
            r = json.loads(l)
            rows[r["id"]] = r
        except ValueError:
            pass
    u = [r for r in rows.values() if r["answerable"] is False]
    a = [r for r in rows.values() if r["answerable"] is True]
    assert len(u) == 600 and len(a) == 600, (tag, len(u), len(a))
    ca_k = sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u)
    return {"n_unanswerable": len(u), "n_answerable": len(a), "CA": ca_k / len(u), "CA_count": ca_k,
            "ACC_A": 100 * sum(r["correct"] for r in a) / len(a)}


def bench_file(p):
    d = json.load(open(p))
    per = {b: 100 * d[b]["greedy_acc_lenient"] for b in BENCH}
    return {"acc_mean": sum(per.values()) / 3, "per_set": per, "n_per_set": {b: d[b]["n"] for b in BENCH}}


def paired_lists(xa, xb, seeds, label, metric):
    d = [xa[s] - xb[s] for s in seeds]
    n = len(d)
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    t = float(stats.t.ppf(0.975, n - 1))
    tstat = m / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(tstat), n - 1)) if se > 0 else float("nan")
    return {"contrast": label, "metric": metric, "seeds": list(seeds), "n": n, "diffs": d,
            "mean": m, "sd": sd, "t_crit": t, "ci95": [m - t * se, m + t * se], "t": tstat, "p_two_sided": p,
            "n_negative": sum(x < 0 for x in d), "n_positive": sum(x > 0 for x in d)}


# ---------------------------------------------------------------- abstention-prompt metrics (dirA3/code/umwp_read.py) ---
def absfix_metrics(tag):
    f = os.path.join(UM, "%s.jsonl" % tag)
    rows = {}
    for l in open(f):
        try:
            r = json.loads(l)
            rows[r["id"]] = r
        except ValueError:
            pass
    u = [r for r in rows.values() if r["answerable"] is False]
    a = [r for r in rows.values() if r["answerable"] is True]
    assert len(u) == 600 and len(a) == 600, (tag, len(u), len(a))
    return {"ABS_U": 100 * sum(r["abstain"] for r in u) / len(u), "ABS_A": 100 * sum(r["abstain"] for r in a) / len(a),
            "CA_abs": sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u) / len(u),
            "ACC_A_abs": 100 * sum(r["correct"] for r in a) / len(a)}


def sidecar_ok(p, sha_key, model_suffix, output):
    """A13 sidecar of a finished cell: state complete, registered script, the cell's model, and the output file unchanged
    since the launcher marked it complete (sha256 recorded then)."""
    m = jload(p)
    if not m:
        return False, "no A13 sidecar"
    if m.get("registration") != "A13" or m.get("state") != "complete":
        return False, "sidecar state %r" % m.get("state")
    if m.get("script_sha256") != K[sha_key]:
        return False, "script sha256 is not the registered one"
    if not str(m.get("model", "")).rstrip("/").endswith(model_suffix):
        return False, "model %r" % m.get("model")
    if not os.path.exists(output) or m.get("output_sha256") != sha(output):
        return False, "output changed after completion or missing"
    return True, ""


# ---------------------------------------------------------------- 1. check: reproduce the stored R-S contrast ---
def check_rs():
    ca = {a: {s: umwp("plain_" + NAME[a] % s)["CA"] for s in SEEDS} for a in ("S", "R")}
    rs = paired_lists(ca["S"], ca["R"], list(SEEDS), "cln1 - rnd1", "CA")
    at = jload(ATTR) or {}
    st = at.get("paired", {}).get("cln1-rnd1:CA")
    eq = lambda a, b: abs(a - b) < 1e-12
    same = bool(st) and eq(rs["mean"], st["mean"]) and eq(rs["sd"], st["sd"]) and all(eq(a, b) for a, b in zip(rs["ci95"], st["ci95"])) \
        and len(st["diffs"]) == 5 and all(eq(a, b) for a, b in zip(rs["diffs"], st["diffs"]))
    rounded = [round(100 * rs["mean"], 1), round(100 * rs["ci95"][0], 1), round(100 * rs["ci95"][1], 1)]
    changed = [p for p, h in K["old_sha256"].items() if not os.path.exists(f"{R}/{p}") or sha(f"{R}/{p}") != h]
    ok = same and rounded == K["paper_rs"] and not changed
    bz = {a: {s: bench_file(f"{EV_OLD}/{NAME[a] % s}_step60.summary.json")["acc_mean"] for s in SEEDS} for a in ("S", "R")}
    rb = paired_lists(bz["S"], bz["R"], list(SEEDS), "cln1 - rnd1", "bench")
    sb = at.get("paired", {}).get("cln1-rnd1:bench")
    bench_ok = bool(sb) and eq(rb["mean"], sb["mean"]) and round(rb["mean"], 1) == K["paper_rs_bench"]
    return ok, {"recomputed": rs, "stored": st, "equal_to_stored": same, "rounded_points": rounded, "paper": K["paper_rs"],
                "inputs_changed_since_registration": changed,
                "attribution_json_sha256": sha(ATTR) if os.path.exists(ATTR) else None,
                "bench_recomputed_mean": rb["mean"], "bench_equal_to_stored_and_paper": bench_ok}, ca, bz, bench_ok


if args.check_only:
    ok, chk, _, _, _ = check_rs()
    print(json.dumps({"check_ok": ok, "rounded_points": chk["rounded_points"], "paper": chk["paper"],
                      "equal_to_stored": chk["equal_to_stored"], "inputs_changed": chk["inputs_changed_since_registration"],
                      "diffs": chk["recomputed"]["diffs"],
                      "ci95": chk["recomputed"]["ci95"], "bench_ok": chk["bench_equal_to_stored_and_paper"]}, indent=1))
    sys.exit(0 if ok else 1)

if os.path.exists(OUT):
    print(f"refusing: {OUT} exists (the A13 read runs once)")
    sys.exit(3)
check_ok, chk, CA, BZ, bench_check_ok = check_rs()


# ---------------------------------------------------------------- 2. acceptance of each R0 seed (registration 6.2) ---
def accept_r0(s):
    n = NAME["R0"] % s
    d = f"{RUNS_A13}/{n}"
    why = []
    if (open(f"{d}/.trained").read().strip() if os.path.exists(f"{d}/.trained") else "") != "60":
        why.append("no .trained = 60")
    steps, done = set(), False
    if os.path.exists(f"{d}/log.jsonl"):
        for l in open(f"{d}/log.jsonl"):
            try:
                x = json.loads(l)
            except ValueError:
                continue
            if "step" in x and "loss" in x:
                steps.add(x["step"])
            if x.get("done") and x.get("steps") == 60:
                done = True
    if not done or steps != set(range(1, 61)):
        why.append("log.jsonl: done=%s, %d of 60 steps logged" % (done, len(steps & set(range(1, 61)))))
    cfg, ref = jload(f"{d}/config.json"), jload(f"{RUNS_OLD}/{NAME['R'] % s}/config.json")
    if not cfg or not ref:
        why.append("config.json missing")
    else:
        diff = sorted(k for k in set(cfg) | set(ref) if k not in K["config_ignore"] and cfg.get(k) != ref.get(k))
        if diff:
            why.append("config differs from rnd1 in %s" % diff)
        if cfg.get("data") != f"{d}/admitted.jsonl" or cfg.get("out", "").rstrip("/") != d:
            why.append("config data/out not in the run directory")
        if not str(cfg.get("model", "")).endswith(f"/ckpt_b/qwen3_4b_Base_own_a3_s{s}_r2/final"):
            why.append("config model is not the seed's round-2 solver")
    if not os.path.exists(f"{d}/admitted.jsonl") or sha(f"{d}/admitted.jsonl") != K["r0_sha256"][str(s)]:
        why.append("training set is not the registered R0 set")
    tr = jload(f"{d}/a13_train.json") or {}
    if tr.get("registration") != "A13" or tr.get("state") != "complete":
        if tr.get("state") in ("not_run", "script_changed"):
            why.append("training not run: %s" % tr.get("reason"))
        else:
            why.append("train sidecar state %r" % tr.get("state"))
    if tr.get("script_sha256") != K["b_train_sha256"]:
        why.append("b_train sha256 not registered")
    if tr.get("start_ckpt_sha_ok") is not True:
        why.append("start checkpoint not verified")
    if os.path.exists(f"{d}/config.json") and tr.get("config_sha256") != sha(f"{d}/config.json"):
        why.append("config.json changed after the training finished")
    ok, w = sidecar_ok(f"{UM}/plain_{n}.a13.json", "umwp_probe_sha256", f"/{n}/final", f"{UM}/plain_{n}.jsonl")
    if not ok:
        why.append("plain dump: " + w)
    u = None
    if not why:
        try:
            u = umwp("plain_" + n)
        except (OSError, AssertionError) as e:
            why.append("plain dump incomplete: %s" % (e,))
    return (not why), why, u, tr


# reasons meaning "not finished by the read"; any other reason is a failed registered check
UNFINISHED = ("no .trained", "log.jsonl:", "config.json missing", "train sidecar state", "plain dump: no A13 sidecar",
              "plain dump: sidecar state", "plain dump incomplete")
per_seed, R0 = {}, {}
for s in SEEDS:
    ok, why, u, tr = accept_r0(s)
    per_seed[str(s)] = {"accepted": ok, "reasons": why, "umwp": u,
                        "failed_check": any(not r.startswith(UNFINISHED) for r in why),
                        "train": {k: tr.get(k) for k in ("host", "n_attempts", "n_yields", "n_resumes", "resumes", "start_ckpt")}}
    if ok:
        R0[s] = u
ok_seeds = sorted(R0)
man = jload(f"{RUNS_A13}/a13_build_manifest.json") or {}
man_ok = bool(man) and all(man["seeds"][str(s)]["R0_sha256"] == K["r0_sha256"][str(s)] for s in SEEDS)


def rng(v):
    v = [x for x in v if x is not None]
    return None if not v else ("%d" % v[0] if min(v) == max(v) else "%d--%d" % (min(v), max(v)))


comp = {}
if man_ok:
    c = lambda s, arm, f: f(man["seeds"][str(s)]["composition"][arm])
    sel = ok_seeds or list(SEEDS)
    comp = {"items": rng([c(s, "R0", lambda x: x["n"]) for s in sel]),
            "ill_R0": rng([c(s, "R0", lambda x: x["ill_posed_proposals"]) for s in sel]),
            "ill_R": rng([c(s, "R_rnd1", lambda x: x["ill_posed_proposals"]) for s in sel]),
            "intact_R0": rng([c(s, "R0", lambda x: x["kinds"].get("tracer_intact", 0)) for s in sel]),
            "intact_S": rng([c(s, "S_cln1", lambda x: x["kinds"].get("tracer_intact", 0)) for s in sel]),
            "k": rng([man["seeds"][str(s)]["k_replaced"] for s in sel])}

# ---------------------------------------------------------------- 3. registered quantities and reading ---
res = {"registration": "A13_registration_2026-09-27.md", "read_sha256": sha(os.path.abspath(__file__)), "root": R,
       "check_R_minus_S": chk, "check_ok": check_ok, "manifest_ok": man_ok, "per_seed": per_seed, "accepted_seeds": ok_seeds,
       "composition": comp}
reading, why_not, why_not_paper = None, None, None
W = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
if not check_ok and chk["equal_to_stored"] and chk["rounded_points"] == K["paper_rs"]:
    why_not = "an input of the stored R-S contrast changed after registration: %s" % chk["inputs_changed_since_registration"]
    why_not_paper = "an input of the stored same-size contrast changed after registration"
elif not check_ok:
    why_not = "the stored R-S contrast was not reproduced from the dumps"
    why_not_paper = "the stored same-size contrast could not be reproduced from its dumps"
elif not man_ok:
    why_not = "the build manifest does not carry the registered R0 sets"
    why_not_paper = "the registered training sets were not built"
elif len(ok_seeds) < K["min_seeds"]:
    why_not = "%d of 5 seeds accepted (at least %d required)" % (len(ok_seeds), K["min_seeds"])
    chk_fail = any(per_seed[str(s)]["failed_check"] for s in SEEDS if s not in ok_seeds)
    why_not_paper = ("%s of five runs finished in time%s, four were required"
                     % (W[len(ok_seeds)], " and passed the registered checks" if chk_fail else ""))
if why_not is None:
    ca0 = {s: R0[s]["CA"] for s in ok_seeds}
    E_prop = paired_lists(ca0, CA["S"], ok_seeds, "R0 - S (r0_a13 - cln1)", "CA")
    E_plant = paired_lists(CA["R"], ca0, ok_seeds, "R - R0 (rnd1 - r0_a13)", "CA")
    T3 = paired_lists(CA["S"], ca0, ok_seeds, "S - R0 (Table 3 convention)", "CA")
    RS = paired_lists(CA["R"], CA["S"], ok_seeds, "R - S (same seeds)", "CA")
    ident = max(abs(E_prop["diffs"][i] + E_plant["diffs"][i] - RS["diffs"][i]) for i in range(len(ok_seeds)))
    lo, hi = E_prop["ci95"]
    reading = "ADDS" if lo > 0 else ("REDUCES" if hi < 0 else "NOT DETECTED")
    res.update({"E_prop": E_prop, "E_plant": E_plant, "table3_S_minus_R0": T3, "R_minus_S_same_seeds": RS,
                "identity_max_abs_error": ident, "reading": reading,
                "E_prop_points": {"mean": 100 * E_prop["mean"], "ci95": [100 * lo, 100 * hi], "sd": 100 * E_prop["sd"],
                                  "half_width": 100 * (hi - E_prop["mean"])},
                "E_plant_points": {"mean": 100 * E_plant["mean"], "ci95": [100 * x for x in E_plant["ci95"]]}})
    # descriptive: benchmark accuracy, ACC_A
    bz0 = {}
    for s in ok_seeds:
        n = NAME["R0"] % s
        p = f"{EV_A13}/{n}_step60.summary.json"
        okb, w = sidecar_ok(f"{EV_A13}/{n}_step60.a13.json", "b_eval_sha256", f"/{n}/final", p)
        try:
            if okb:
                bz0[s] = bench_file(p)["acc_mean"]
        except (OSError, KeyError, ValueError):
            pass
    desc = {"bench_R0": bz0}
    bs = sorted(bz0)
    if bench_check_ok and len(bs) >= 2:
        desc["bench_S_minus_R0"] = paired_lists(BZ["S"], bz0, bs, "S - R0", "bench")
        desc["bench_R_minus_R0"] = paired_lists(BZ["R"], bz0, bs, "R - R0", "bench")
        desc["bench_complete"] = bs == ok_seeds
    acc0 = {s: R0[s]["ACC_A"] for s in ok_seeds}
    accS = {s: umwp("plain_" + NAME["S"] % s)["ACC_A"] for s in ok_seeds}
    desc["ACC_A_S_minus_R0"] = paired_lists(accS, acc0, ok_seeds, "S - R0", "ACC_A")
    # descriptive: corrected abstention prompt (A11 probe), seeds where all three arms have an accepted dump
    ab = {a: {} for a in NAME}
    for a in NAME:
        for s in SEEDS:
            n = NAME[a] % s
            okc, w = sidecar_ok(f"{UM}/absfix_{n}.a13.json", "absfix_probe_sha256", f"/{n}/final", f"{UM}/absfix_{n}.jsonl")
            if okc:
                try:
                    ab[a][s] = absfix_metrics("absfix_" + n)
                except (OSError, AssertionError):
                    pass
    both = [s for s in ok_seeds if all(s in ab[a] for a in NAME)]
    desc["absfix"] = {a: {str(s): v for s, v in ab[a].items()} for a in NAME}
    desc["absfix_seeds_all_arms"] = both
    if len(both) >= 2:
        for m in ("ABS_U", "ABS_A", "CA_abs"):
            g = lambda a: {s: ab[a][s][m] for s in both}
            desc[f"absfix_{m}_R0_minus_S"] = paired_lists(g("R0"), g("S"), both, "R0 - S", m)
            desc[f"absfix_{m}_R_minus_R0"] = paired_lists(g("R"), g("R0"), both, "R - R0", m)
            desc[f"absfix_{m}_seed_means"] = {a: statistics.mean(g(a).values()) for a in NAME}
    res["descriptive"] = desc
else:
    res["reading"] = "NOT RUN"
    res["not_run_reason"] = why_not
    res["not_run_reason_paper"] = why_not_paper

# ---------------------------------------------------------------- 4. paper edits for the reading (registration 7) ---
# Exact find/replace pairs on the paper sources (paper_arr_oct2026/sections; each find string occurs exactly once there,
# checked 2026-09-27 01:55). A find string that no longer matches is applied by hand to the same sentence; the replacement
# text is not reworded.
F = {
    "sec5": "The drifted curriculum thus adds about six points in one round to the sharpening any label-free curriculum "
            "shares, an estimate that does not separate proposer-written from planted ill-posed ones.",
    "head": "\\paragraph{Most of the rise is sharpening; the ill-posed curriculum adds to it.}",
    "abstract": "Most of this rise is sharpening itself; at equal training-set size the ill-posed curriculum adds about six points.",
    "intro": "Most of this rise is majority-vote sharpening, which any label-free curriculum shares; the drifted curriculum adds "
             "about six points in one round.",
    "concl": "Sharpening turns these readings into one stated assumption, mostly by itself and partly through the drift, and "
             "benchmark accuracy reveals neither.",
    "limit": "\\paragraph{Attribution.} The drift's own contribution to the solver's behaviour is estimated from one-round, "
             "same-size contrasts, and these do not separate proposer-written from planted ill-posed problems "
             "(\\S\\ref{sec:accommodation}).",
    "claims": "Most of the rise is sharpening; at equal size the ill-posed curriculum adds about six points &",
    "row": "Same size$^{\\dagger}$ (5) & 148--177 (0) & $-30$ & $-5.6$ {\\scriptsize$[-9.0, -2.1]$} & $+0.8$ \\\\",
    "caption": " Last row: a loop admitting a random subset of the band of the guard's size (\\S\\ref{sec:remedies}).}",
    "setup": "or on a random subset of that set of the same size (148--177 problems, of which 20--39 are premise-deleted).",
    "reg1": "Four analyses were registered with a pre-stated interpretation for each possible outcome instead of pass/fail "
            "predictions, three before any of their data existed",
    "reg2": " and a verifier on top of the band (the level falls, the drift persists, and the change in odds is consistent with "
            "a fixed miss rate; \\S\\ref{sec:remedies}).",
}


def _rd(x, d):
    v = round(float(x), d)
    return 0.0 if v == 0 else v                              # no "-0.0"


def _dec(x):
    """one decimal; two (or three) when the value is not 0 but would print as 0.0 with one"""
    for d in (1, 2, 3):
        if _rd(x, d) != 0:
            return d
    return 1


def _fmt(x, sign=False, math=False):
    d = _dec(x)
    v = _rd(x, d)
    if v < 0:
        return ("-%.*f" if math else "$-$%.*f") % (d, -v)
    return ("+%.*f" if sign else "%.*f") % (d, v)


f1 = lambda x: _fmt(x)                          # running text, points
fs = lambda x: _fmt(x, sign=True)               # running text, signed
m1 = lambda x: _fmt(x, math=True)               # inside $...$
sg = lambda x: _fmt(x, sign=True, math=True)    # inside $...$, signed (Table 3 style)


def seeds_phrase(k, n):
    if k == n:
        return "in all %s seeds" % W[n]
    if k == 0:
        return "in none of the %s seeds" % W[n]
    return "in %s of %s seeds" % (W[k], W[n])


def edit(f, key, new):
    return {"file": f, "find": F[key], "replace": new}


edits, extra = [], {}
reg1_new = (F["reg1"].replace("Four analyses", "Five analyses")
            + ", the same-size contrast without planted problems before any of its new runs existed")
reg2_new = lambda clause: (", a verifier on top of the band (the level falls, the drift persists, and the change in odds is "
                           "consistent with a fixed miss rate; \\S\\ref{sec:remedies}) and the same-size contrast without "
                           "planted problems (%s)." % clause)
if res["reading"] == "NOT RUN":
    edits += [edit("11_appendix.tex", "reg1", reg1_new),
              edit("11_appendix.tex", "reg2", reg2_new("not completed: %s" % why_not_paper))]
else:
    ep, pl, t3 = res["E_prop_points"], res["E_plant_points"], res["table3_S_minus_R0"]
    e, lo, hi = ep["mean"], ep["ci95"][0], ep["ci95"][1]
    n, kp, kn = E_prop["n"], E_prop["n_positive"], E_prop["n_negative"]
    rs = 100 * res["R_minus_S_same_seeds"]["mean"]         # the stored same-size contrast on the accepted seeds (5.6 with all five)
    same = "" if n == 5 else " on the same %s seeds" % W[n]
    lead = "With the random subset's planted premise-deleted problems replaced by other random admitted proposals, "
    back = ("putting the planted problems back changes it by %s points (95\\%% CI $[%s, %s]$)"
            % (fs(pl["mean"]), m1(pl["ci95"][0]), m1(pl["ci95"][1])))
    six = "adds about six points in one round to the sharpening any label-free curriculum shares. "
    intact = ("without planted premise-deleted problems the random subset also holds fewer intact planted problems than the "
              "solvable set (%s against %s; Appendix~\\ref{app:setup})" % (comp.get("intact_R0", "?"), comp.get("intact_S", "?")))
    lim_comp = ("\\paragraph{Attribution.} The drift's own contribution to the solver's behaviour is estimated from one-round, "
                "same-size contrasts over %s seeds; %s." % (W[n], intact))
    sub = None
    if reading == "ADDS":
        sec5 = ("The drifted curriculum thus " + six + lead + "one round on the subset still raises the rate over the solvable "
                "problems %s, by %s points on average (95\\%% CI $[%s, %s]$), against %s points with them%s; %s. "
                "Proposer-written ill-posed problems thus raise the rate by %s points without any planted premise-deleted "
                "problem." % (seeds_phrase(kp, n), f1(e), m1(lo), m1(hi), f1(rs), same, back, f1(e)))
        lim, clause = lim_comp, "without planted problems the rate still rises by %s points; \\S\\ref{sec:accommodation}" % f1(e)
        edits += [edit("01_introduction.tex", "intro", F["intro"][:-1] + ", and still %s points with the premise-deleted "
                       "problems we plant among its candidates replaced by model-written ones." % f1(e))]
    elif reading == "NOT DETECTED":
        sub = "estimate <= 0" if e <= 0 else "estimate > 0"
        sec5 = ("The drifted curriculum " + six + lead + "one round on the subset changes the rate relative to the solvable "
                "problems by %s points on average (95\\%% CI $[%s, %s]$, an interval that includes zero; higher %s), against "
                "%s points with them%s; %s." % (fs(e), m1(lo), m1(hi), seeds_phrase(kp, n), f1(rs), same, back))
        lim = ("\\paragraph{Attribution.} The drift's own contribution to the solver's behaviour is estimated from one-round, "
               "same-size contrasts over %s seeds, which do not resolve how much of it proposer-written rather than planted "
               "ill-posed problems contribute (\\S\\ref{sec:accommodation}); %s." % (W[n], intact))
        clause = "not resolved at %s seeds; \\S\\ref{sec:accommodation}" % W[n]
        if e <= 0:
            edits += [edit("11_appendix.tex", "claims", "Most of the rise is sharpening; at equal size the ill-posed curriculum "
                           "adds about six points, not detected without its planted problems &"),
                      edit("08_conclusion.tex", "concl", "Sharpening turns these readings into one stated assumption, mostly by "
                           "itself and partly through the ill-posed curriculum, and benchmark accuracy reveals neither.")]
    else:  # REDUCES
        sec5 = (lead + "one round on the subset lowers the rate below that of the solvable problems %s, by %s points on "
                "average (95\\%% CI of the difference $[%s, %s]$), against a %s-point rise with them%s; %s. The six points by "
                "which the drifted curriculum exceeds the solvable problems in one round thus come from its planted "
                "premise-deleted problems, not from its proposer-written ones."
                % (seeds_phrase(kn, n), f1(-e), m1(lo), m1(hi), f1(rs), same, back))
        lim, clause = lim_comp, "the same-size difference comes from the planted problems; \\S\\ref{sec:accommodation}"
        edits += [edit("05_accommodation.tex", "head",
                       "\\paragraph{Most of the rise is sharpening; planted premise-deleted problems add to it.}"),
                  edit("00_abstract.tex", "abstract", "Most of this rise is sharpening itself; the six points the ill-posed "
                       "curriculum adds at equal training-set size come from premise-deleted problems we plant among its "
                       "candidates, not from model-written ones."),
                  edit("01_introduction.tex", "intro", "Most of this rise is majority-vote sharpening, which any label-free "
                       "curriculum shares; the drifted curriculum adds about six points in one round, which come from the "
                       "premise-deleted problems we plant among its candidates rather than from model-written ones."),
                  edit("11_appendix.tex", "claims", "Most of the rise is sharpening; at equal size the ill-posed curriculum "
                       "adds about six points, from planted premise-deleted problems, not proposer-written ones &"),
                  edit("08_conclusion.tex", "concl", "Sharpening turns these readings into one stated assumption, mostly by "
                       "itself rather than through the drift, and benchmark accuracy reveals neither.")]
    res["not_detected_subcase"] = sub
    dacc = res["descriptive"].get("bench_S_minus_R0")
    dacc_txt = ("$%s$" % sg(dacc["mean"]) if dacc and res["descriptive"].get("bench_complete") else "--")
    row = ("Same size, no planted (%d) & %s (0) & $0$ & $%s$ {\\scriptsize$[%s, %s]$} & %s \\\\"
           % (n, comp.get("items", "?"), sg(100 * t3["mean"]), m1(100 * t3["ci95"][0]), m1(100 * t3["ci95"][1]), dacc_txt))
    setup = ("The \\emph{same-size contrast without planted problems}, registered before any of its new runs existed, replaces "
             "in place each premise-deleted problem of that random subset (%s per seed) with a random admitted proposal not "
             "already in it, leaving the size, the order of the other problems and every setting unchanged; its runs were "
             "trained three days after the others, and GPU sampling is not bitwise reproducible. Of its problems, %s are "
             "ill-posed proposals (%s in the random subset) and %s intact planted problems, against none and %s in the "
             "solvable set." % (comp.get("k", "?"), comp.get("ill_R0", "?"), comp.get("ill_R", "?"), comp.get("intact_R0", "?"),
                                comp.get("intact_S", "?")))
    tot = "" if n == 5 else " on these %s seeds (%s points)" % (W[n], f1(rs))
    plant = ("Putting the planted premise-deleted problems back in place of these proposals changes the rate by %s points on "
             "average (95\\%% CI $[%s, %s]$; higher %s); this and the difference without them add up to the same-size "
             "contrast%s by construction." % (fs(pl["mean"]), m1(pl["ci95"][0]), m1(pl["ci95"][1]),
                                              seeds_phrase(E_plant["n_positive"], E_plant["n"]), tot))
    edits = [edit("05_accommodation.tex", "sec5", sec5),
             edit("05_accommodation.tex", "row", F["row"] + "\n" + row),
             edit("05_accommodation.tex", "caption", " Same size, no planted: as Same size, with the random subset's "
                  "planted premise-deleted problems replaced by other random admitted proposals." + F["caption"]),
             edit("09_limitations.tex", "limit", lim),
             edit("11_appendix.tex", "setup", F["setup"] + " " + setup + " " + plant),
             edit("11_appendix.tex", "reg1", reg1_new),
             edit("11_appendix.tex", "reg2", reg2_new(clause))] + edits
    d = res["descriptive"]
    if "absfix_ABS_U_seed_means" in d:
        u, a = d["absfix_ABS_U_seed_means"], d["absfix_ABS_A_seed_means"]
        extra["appendix_absfix_sentence"] = (
            "With the abstention option, one round on the solvable problems, on the random subset and on the subset without "
            "planted problems abstains on %.0f, %.0f and %.0f\\%% of the unanswerable UMWP questions and on %.0f, %.0f and "
            "%.0f\\%% of the answerable ones (seed means, %s seeds)."
            % (u["S"], u["R"], u["R0"], a["S"], a["R"], a["R0"], W[len(d["absfix_seeds_all_arms"])]))
res["paper"] = {"edits": edits, **extra}

with open(OUT, "x") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({"reading": res["reading"], "subcase": res.get("not_detected_subcase"), "accepted_seeds": ok_seeds, "check_ok": check_ok,
                  "E_prop_points": res.get("E_prop_points"), "E_plant_points": res.get("E_plant_points"),
                  "not_run_reason": res.get("not_run_reason"), "n_edits": len(edits), "out": OUT}, indent=1))
