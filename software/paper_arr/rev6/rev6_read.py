"""rev6 read (plan: paper_arr/rev6/rev6_plan_2026-10-06.md). Written while the rev6 runs were in progress and hashed into
PREREG_LOG.md before any aggregate of a rev6 output was computed.

  python3 rev6_read.py --check   reproduction check of this script's metric code on the ORIGINAL dumps (dirA4/results/umwp;
                                 stored answers = ORIGINAL parser; no texts) -> paper_arr/rev6/rev6_read_check.json
  python3 rev6_read.py           the read on paper_arr/rev6/texts -> paper_arr/rev6/rev6_read.json (refuses to overwrite)

Parsers, applied to every sample of a rev6 dump:
  ORIGINAL  the probe's own parse, i.e. the stored "answers" (umwp_probe_text.py answers_of: plain prompt: last
            'Answer: <number>', else the last number; abstention prompt: last \\boxed{} first; 'length' -> None).
  STRICT    a sample is a non-answer when strict() of paper_arr/analyses/rev5/strict_parse.py (definitions executed from that
            file, unchanged) returns None, i.e. the status validated against the annotators; otherwise its number is the
            probe's own (the stored answer; strict()'s number only where the probe found none). Abstention prompt: the probe's
            boxed branch (abstain token -> None, else the boxed number) and its 'Answer: unknown|unanswerable|insufficient
            information' branch are kept; strict() decides only where the probe falls back to the plain parser.
            AMENDMENT (before any aggregate of a rev6 output was computed): the first version used strict()'s own number,
            the first number of the last 'Answer:' line, which on the A8 texts differs from the probe's number in 6.7% of the
            base model's and 0.8-3.0% of the trained solvers' numeric samples ('Half of 144 is 144 / 2 = 72. Answer: 72' ->
            144); the validated status is unchanged.
            AMENDMENT 2 (7 Oct, before this read was run, after interim looks at completed dumps on 6 Oct 23:26 and 23:59):
            STRICT is read() of paper_arr/rev6/strict_v3.py: a solution is first cut at its first line that starts a new
            question ('Question:'; the base models continue with new model-written problems, and strict() read their last
            'Answer:' line), then strict()'s status and the first parser's number on the cut text; 94.6% agreement with the
            annotators against 92.4% without the cut (strict_v3_check.py). The non-answer split below uses the cut text.
Metrics per checkpoint (a dump counts only with exactly 600 unanswerable U and 600 answerable A ids):
  SC        share of U with mode not None and p_mode >= 0.5, mode = Counter(8 parsed answers).most_common(1)[0] (None is a
            category; the probe's definition, used by attribution.py, fig1_drift_harm.py and convergence.py)
  SC_k      share of U whose most frequent numeric answer is held by >= k of 8 samples, k = 4..8 (ca_thresholds.py)
  NUM_U/A   share of samples with a numeric answer;  ACC_A  share of A whose mode equals the reference (probe definition)
  STRICT non-answers on U split into length / decline (DECL in the last 'Answer:' line, or in the last 400 characters
            without one) / no number / symbolic; the split must agree with strict() (mismatches reported)
  abstention prompt: ABS_U, ABS_A = mean of the probe's per-question "abstain" share (parser-independent); CA_abs = SC on
            that dump; frontier = a5_frontier.py curve() and pareto() (definitions executed from that file) on items built
            from the parsed answers; frontier_at() at FA_A = 0.05 and at the guard's FA_A; default point = (ABS_A, plain SC);
            held-out calibration as paper_arr/analyses/rev2/frontier_cal.py (its definitions executed; split 20260927 and
            200 splits 1000..1199)
Contrasts (attribution.py): per seed paired differences of SC in points, mean +- t(n-1, 0.975) sd / sqrt(n):
  solvable_only = clean - own (seeds 0-3); same_size = cln1 - rnd1 (0-4); same_size_no_planted = cln1 - r0 (0-4);
  planted_back = rnd1 - r0 (0-4); random_loop = rnd - own (0-2); guard = guard - own (0-2)
Cross-seed agreement (convergence.py / check_convergence.py): over the U answered self-consistently (SC) by all three
  round-5 seeds: identical modes; mode-level null (check_convergence.mode_null, executed from that file: three fresh 8-sample
  runs of the base distribution); common mode = base mode (Counter order) / a tied-top numeric base answer / among the base's
  8 answers; mean base count; split by base SC; each seed's mode = base mode; common mode = the answerable original's reference.
  Also on the A8 texts (112 U; dirA5/results/a8/texts), both parsers.
"""
import argparse, hashlib, json, math, os, re, statistics, sys
from collections import Counter
from scipy import stats
R = "/home/tione/notebook/rsi_rl"; D6 = f"{R}/paper_arr/rev6"; ORIG = f"{R}/dirA4/results/umwp"
sys.path.insert(0, f"{R}/code"); from loto_common import last_boxed  # noqa: E402
sys.path.insert(0, f"{R}/dirB/code"); from pdc_util import is_abstain  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); ARGS = ap.parse_args()
OUT = f"{D6}/rev6_read_check.json" if ARGS.check else f"{D6}/rev6_read.json"
if not ARGS.check and os.path.exists(OUT): sys.exit(f"refusing to run: {OUT} exists")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
SP = f"{R}/paper_arr/analyses/rev5/strict_parse.py"; FR = f"{R}/dirA5/code/a5_frontier.py"
CC = f"{R}/paper_arr/analyses/check_convergence.py"; FC = f"{R}/paper_arr/analyses/rev2/frontier_cal.py"
src = open(SP).read(); sp = {}; exec(compile(src[:src.index("# ---- validation against the human answer status")], SP, "exec"), sp)
strict, DECL, SYMB, UNIT_OK, norm_ans = sp["strict"], sp["DECL"], sp["SYMB"], sp["UNIT_OK"], sp["norm_ans"]
src = open(FR).read(); fr = {}; exec(compile(src[:src.index("res = {")], FR, "exec"), fr)
src = open(CC).read(); cc = {"math": math, "Counter": Counter}; exec(compile(src[src.index("def mode_null"):src.index("def top_set")], CC, "exec"), cc)
src = open(FC).read(); fc = {}; exec(compile(src[:src.index("tags = {s:")], FC, "exec"), fc)
mode_null = cc["mode_null"]
sys.path.insert(0, D6); import strict_v3  # noqa: E402  (amendment 2)
SD = {x["id"]: x for x in (json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl"))}
ABS_RE = re.compile(r"Answer:\s*(unknown|unanswerable|insufficient information)", re.I)


def strict_abs(t, f):
    if f == "length": return None
    b = last_boxed(t)
    if b is not None and is_abstain(b): return None
    if b is not None: return norm_ans(b)
    if ABS_RE.search(t): return None
    return strict(t, f)


def why_none(t, f):
    """Branch of strict() (plain prompt) that makes a sample a non-answer; 'numeric' otherwise."""
    if f == "length": return "length"
    lines = re.findall(r"Answer:\s*([^\n]*)", t)
    if lines:
        last = lines[-1].strip().strip("*").strip()
        if DECL.search(last): return "decline"
        if not re.search(r"\d", last): return "no_number"
        if SYMB.search(last) and not UNIT_OK.match(last): return "symbolic"
        return "numeric" if re.search(r"-?\d[\d,]*\.?\d*", last) else "no_number"
    if DECL.search(t[-400:]): return "decline"
    return "numeric" if sp["original"](t, f) is not None else "no_number"


def load(path):
    rows = {}
    if not os.path.exists(path): return None
    for l in open(path):
        try: r = json.loads(l)
        except ValueError: continue
        rows[r["id"]] = r
    U = sum(not r["answerable"] for r in rows.values()); A = sum(r["answerable"] for r in rows.values())
    if U != 600 or A != 600 or any(len(r["answers"]) != 8 for r in rows.values()): return None
    return rows


def strict_num(t, f, a, prompt):  # amendment 2: paper_arr/rev6/strict_v3.py
    return strict_v3.read(t, f, prompt)


def parsed(rows, parser, prompt):
    if parser == "ORIGINAL": return {i: list(r["answers"]) for i, r in rows.items()}
    return {i: [strict_num(t, f, a, prompt) for t, f, a in zip(r["texts"], r["finish"], r["answers"])] for i, r in rows.items()}


def mode_of(ans):
    m, n = Counter(ans).most_common(1)[0]; return m, n / len(ans)


def sc(m, p): return m is not None and p >= 0.5 - 1e-9


def metrics(rows, ans, prompt, parser):
    U = [i for i in rows if not rows[i]["answerable"]]; A = [i for i in rows if rows[i]["answerable"]]
    md = {i: mode_of(ans[i]) for i in rows}
    top = {i: max(Counter(a for a in ans[i] if a is not None).values(), default=0) for i in U}
    out = {"SC": sum(sc(*md[i]) for i in U) / 600}
    for k in range(4, 9): out[f"SC_{k}"] = sum(top[i] >= k for i in U) / 600
    out["NUM_U"] = sum(a is not None for i in U for a in ans[i]) / 4800
    out["NUM_A"] = sum(a is not None for i in A for a in ans[i]) / 4800
    out["ACC_A"] = sum(md[i][0] is not None and rows[i]["ref"] is not None and md[i][0] == rows[i]["ref"] for i in A) / 600
    if prompt == "absfix":
        out["ABS_U"] = sum(rows[i]["abstain"] for i in U) / 600; out["ABS_A"] = sum(rows[i]["abstain"] for i in A) / 600
    if parser == "STRICT" and prompt == "plain":
        c = Counter(why_none(strict_v3.cut(t), f) for i in U for t, f in zip(rows[i]["texts"], rows[i]["finish"]))
        out["U_samples_by_branch"] = {k: v / 4800 for k, v in c.items()}
        out["branch_vs_strict_mismatches"] = sum((why_none(strict_v3.cut(t), f) == "numeric") != (a is not None) for i in U
                                                 for t, f, a in zip(rows[i]["texts"], rows[i]["finish"], ans[i]))
    return out, md


def items_of(rows, ans):
    return {i: {"U": not r["answerable"], "samples": [("NUM", a) if a is not None else ("NONE", None) for a in ans[i]],
                "abst": r["abstain"], "m": fr["nshare"](ans[i]), "p_mode": mode_of(ans[i])[1], "mode": mode_of(ans[i])[0]}
            for i, r in rows.items()}


def cal_one(items, target, seed):  # frontier_cal.one() with the items passed in
    h = fc["split"](items, seed); out = []
    for c, e in ((0, 1), (1, 0)):
        cal = {k: v for k, v in items.items() if k in h[c]}; ev = {k: v for k, v in items.items() if k in h[e]}
        pc, pe = fr["curve"](cal), fr["curve"](ev); mix, det = fc["calibrate"](pc, target)
        out.append((fc["evaluate"](pe, mix), fc["evaluate"](pe, det)))
    return {"CA_mix": statistics.mean(o[0][1] for o in out), "FA_mix": statistics.mean(o[0][0] for o in out),
            "CA_det": statistics.mean(o[1][1] for o in out), "FA_det": statistics.mean(o[1][0] for o in out)}


def paired(vals, a, b, seeds):
    d = [100 * (vals[a][s] - vals[b][s]) for s in seeds]; n = len(d); m = statistics.mean(d); sd = statistics.stdev(d)
    t = float(stats.t.ppf(0.975, n - 1)); se = sd / math.sqrt(n)
    return {"diffs": d, "mean": m, "ci95": [m - t * se, m + t * se], "n_neg": sum(x < 0 for x in d), "n_pos": sum(x > 0 for x in d)}


def cross(base, seeds, Uids, ref=None):
    """base, seeds: id -> 8 parsed answers."""
    bm = {k: mode_of(base[k]) for k in Uids}; sm = [{k: mode_of(s[k]) for k in Uids} for s in seeds]
    S = [k for k in Uids if all(sc(*m[k]) for m in sm)]
    idn = [k for k in S if len({m[k][0] for m in sm}) == 1]; cm = {k: sm[0][k][0] for k in idn}
    numc = lambda a: Counter(x for x in a if x is not None)
    tied = [k for k in idn if numc(base[k]).get(cm[k], 0) > 0 and numc(base[k])[cm[k]] == max(numc(base[k]).values())]
    MN = [mode_null(base[k]) for k in S]
    o = {"n_U": len(Uids), "S": len(S), "identical": len(idn), "identical_share": len(idn) / len(S) if S else None,
         "mode_null_three_base_runs": statistics.mean(m[0] for m in MN if m[0] is not None) if S else None,
         "eq_base_mode": sum(bm[k][0] == cm[k] for k in idn), "tied_top_base": len(tied),
         "in_base_8": sum(cm[k] in base[k] for k in idn), "mean_base_count": statistics.mean(base[k].count(cm[k]) for k in idn) if idn else None,
         "per_seed_mode_eq_base_mode_over_S": [sum(m[k][0] == bm[k][0] for k in S) / len(S) for m in sm] if S else None}
    for lab, cond in (("base_SC", True), ("base_not_SC", False)):
        sub = [k for k in idn if sc(*bm[k]) == cond]
        o[lab] = {"identical": len(sub), "eq_base_mode": sum(bm[k][0] == cm[k] for k in sub)}
    if ref is not None:
        has = [k for k in idn if ref.get(k) is not None]; o["eq_answerable_original_ref"] = [sum(ref[k] == cm[k] for k in has), len(has)]
    return o


def tagname(kind, tag): return f"{D6}/texts/{kind}_{tag}.jsonl" if not ARGS.check else f"{ORIG}/{kind}_{tag}.jsonl"


Q4 = "qwen3_4b_Base_%s_s%d_r%d"
ARMS = {"own": ("own_a3", range(5)), "clean": ("own_clean_a3", range(4)), "rnd": ("ctrl_rnd_a5", range(3)),
        "guard": ("own_ag_a4", range(3)), "cln1": ("cln1_a6", range(5)), "rnd1": ("rnd1_a6", range(5)), "r0": ("r0_a13", range(5))}
TAGS = ["qwen3_4b_base"] + [Q4 % (inf, s, 3) for inf, ss in ARMS.values() for s in ss] + [Q4 % ("own_a3", s, 5) for s in range(3)] \
    + ["base_Qwen3-8B-Base"] + ["qwen3_8b_Base_own_a3_s%d_r3" % s for s in range(3)] \
    + ["base_Phi-4-mini-instruct"] + ["phi4mini_own_a4_s%d_r3" % s for s in range(2)]
ABSTAGS = ["qwen3_4b_base"] + [Q4 % (ARMS[a][0], s, 3) for a in ("own", "clean", "guard", "cln1") for s in ARMS[a][1]]
PARSERS = ("ORIGINAL",) if ARGS.check else ("ORIGINAL", "STRICT")
res = {"read": "rev6", "mode": "check (original dumps, ORIGINAL only)" if ARGS.check else "rev6",
       "sha256": {os.path.relpath(p, R): sha(p) for p in (os.path.abspath(__file__), SP, FR, CC, FC, f"{D6}/strict_v3.py")},
       "per_tag": {}, "not_complete": [], "contrasts": {}, "cross_seed": {}, "frontier": {}}
ANS = {}
for kind, tags in (("plain", TAGS), ("absfix", ABSTAGS)):
    for tag in tags:
        rows = load(tagname(kind, tag))
        if rows is None: res["not_complete"].append(f"{kind}_{tag}"); continue
        e = {}
        for P in PARSERS:
            ans = parsed(rows, P, kind); m, _ = metrics(rows, ans, kind, P); e[P] = m; ANS[(kind, tag, P)] = (rows, ans)
        if not ARGS.check:
            e["stored_vs_probe_mode_mismatches"] = sum(mode_of(r["answers"])[0] != r["mode"] for r in rows.values())
            e["strict_parse_original_vs_stored_mismatches"] = sum(sp["original"](t, f) != a for r in rows.values() if kind == "plain"
                                                                  for t, f, a in zip(r["texts"], r["finish"], r["answers"]))
        res["per_tag"][f"{kind}_{tag}"] = e
for P in PARSERS:
    vals = {a: {} for a in ARMS}
    for a, (inf, ss) in ARMS.items():
        for s in ss:
            k = f"plain_{Q4 % (inf, s, 3)}"
            if k in res["per_tag"]: vals[a][s] = res["per_tag"][k][P]["SC"]
    for name, (a, b, ss) in {"solvable_only": ("clean", "own", range(4)), "same_size": ("cln1", "rnd1", range(5)),
                             "same_size_no_planted": ("cln1", "r0", range(5)), "planted_back": ("rnd1", "r0", range(5)),
                             "random_loop": ("rnd", "own", range(3)), "guard": ("guard", "own", range(3))}.items():
        if all(s in vals[a] and s in vals[b] for s in ss): res["contrasts"][f"{P}|{name}"] = paired(vals, a, b, ss)
        else: res["contrasts"][f"{P}|{name}"] = "not complete"
    # cross-seed agreement of round-5 answers
    need = [("plain", "qwen3_4b_base", P)] + [("plain", Q4 % ("own_a3", s, 5), P) for s in range(3)]
    if all(n in ANS for n in need):
        base = ANS[need[0]][1]; seeds = [ANS[n][1] for n in need[1:]]
        rows = ANS[need[0]][0]; Uids = sorted(i for i in rows if not rows[i]["answerable"])
        ref = {u: rows[SD[u]["relevant_ids"][0]]["ref"] for u in Uids}
        res["cross_seed"][f"{P}|umwp_r5"] = cross(base, seeds, Uids, ref)
    # frontier on the abstention-prompt dumps
    F = {}
    for tag in ABSTAGS:
        if ("absfix", tag, P) not in ANS: continue
        rows, ans = ANS[("absfix", tag, P)]; it = items_of(rows, ans); cv = fr["curve"](it)
        F[tag] = {"pareto": fr["pareto"](cv), "CA_at_FA_0.05": fr["frontier_at"](cv, 0.05), "items": it, "curve": cv}
        if ("plain", tag, P) in ANS:
            prow, pans = ANS[("plain", tag, P)]; F[tag]["default_point"] = fr["umwp_default"](items_of(prow, pans), it)
    for s in range(3):
        g, o = Q4 % ("own_ag_a4", s, 3), Q4 % ("own_a3", s, 3)
        if g in F and o in F and "default_point" in F[g]:
            gfa = F[g]["default_point"][0]; F[o]["guard_FA"] = gfa; F[o]["CA_at_guard_FA"] = fr["frontier_at"](F[o]["curve"], gfa)
            for tname, target in (("guardFA", gfa), ("FA5", 0.05)):
                rep = [cal_one(F[o]["items"], target, 1000 + k) for k in range(200)]
                F[o][f"heldout_{tname}"] = {"split_20260927": cal_one(F[o]["items"], target, 20260927),
                                            "200_splits_mean": {k: statistics.mean(r[k] for r in rep) for k in rep[0]}}
    if "qwen3_4b_base" in F:
        b = F["qwen3_4b_base"]; b["heldout_FA5"] = {"split_20260927": cal_one(b["items"], 0.05, 20260927),
                                                    "200_splits_mean": {k: statistics.mean(r[k] for r in [cal_one(b["items"], 0.05, 1000 + j) for j in range(200)]) for k in ("CA_mix", "FA_mix", "CA_det", "FA_det")}}
    res["frontier"][P] = {t: {k: v for k, v in d.items() if k not in ("items", "curve")} for t, d in F.items()}
# A8 texts (112 U, fresh samples): cross-seed agreement under both parsers
if not ARGS.check:
    A8 = {}
    for tg in ("a8_base", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5"):
        A8[tg] = {x["id"]: x for x in (json.loads(l) for l in open(f"{R}/dirA5/results/a8/texts/{tg}.jsonl")) if not x["answerable"]}
    Uids = sorted(A8["a8_base"])
    for P in PARSERS:
        an = {tg: ({i: list(x["answers"]) for i, x in d.items()} if P == "ORIGINAL" else
                   {i: [strict_num(t, f, a, "plain") for t, f, a in zip(x["texts"], x["finish"], x["answers"])] for i, x in d.items()}) for tg, d in A8.items()}
        res["cross_seed"][f"{P}|a8_r5"] = cross(an["a8_base"], [an[f"a8_own_s{s}_r5"] for s in range(3)], Uids)
json.dump(res, open(OUT, "x" if not ARGS.check else "w"), indent=1, default=str)
# ---- print a summary
for k, e in res["per_tag"].items():
    print(f"{k:48s}", " | ".join(f"{P}: SC {e[P]['SC']:.3f} SC6 {e[P]['SC_6']:.3f} SC8 {e[P]['SC_8']:.3f} NUM_U {e[P]['NUM_U']:.3f} ACC_A {e[P]['ACC_A']:.3f}"
                                     + (f" ABS_U {e[P]['ABS_U']:.3f} ABS_A {e[P]['ABS_A']:.3f}" if "ABS_U" in e[P] else "") for P in PARSERS))
for k, c in res["contrasts"].items():
    print(k, c if isinstance(c, str) else f"mean {c['mean']:+.2f} ci [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] neg {c['n_neg']}/{len(c['diffs'])}")
for k, c in res["cross_seed"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in c.items()})
for P, F in res["frontier"].items():
    for t, d in F.items(): print(P, t, {k: v for k, v in d.items() if k != "pareto"})
print("not complete:", res["not_complete"])
