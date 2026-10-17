#!/usr/bin/env python3
"""Dev test for human_read.py on SYNTHETIC files only (it never opens the real label files or the key).
Builds synthetic annotator, adjudication, rationale, key, packet, A7b (900 items) and open-judge files under /tmp/hr_claude_h1/,
runs human_read.py on them with --synthetic and checks the output against hand-computed values (closed-form kappa, Wilson
bounds, logistic slope and SE, designed false-positive rates and drifts) and against an independent pure-Python recomputation
of every bootstrap interval it checks (including both exploratory estimators); then checks determinism (two runs, identical
bytes), the overwrite refusal, the runs that must succeed (full-word labels, U-vs-A disagreements without adjudication,
(quasi-)separated round patterns for the logistic trend) and the abort paths (S-vs-ill-posed disagreement without
adjudication, invalid label, invalid label on an adjudicated item, empty label, wrong row count, duplicate row, conflicting
adjudication, key/A7b mismatch, key/packet text mismatch, unparsable packet, malformed open-judge file, registered-hash
mismatch; no NOT RUN message may contain a hid). Unit checks of the reading rule and the logistic fit import the script.
Deletes /tmp/hr_claude_h1/ at the end.  python3 dev_human_read_test.py"""
import importlib.util, json, math, os, random, re, shutil, subprocess, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "human_read.py")
REAL_OUT = "/home/tione/notebook/rsi_rl/dirA5/human_validation_drift/human_read.json"
W = "/tmp/hr_claude_h1"
SEED, SEED9, B, Z = 20260947, 20260948, 4000, 1.959963984540054
WORD = {"S": "SOLVABLE", "U": "UNSOLVABLE", "A": "AMBIGUOUS"}
ROUNDS = (1, 3, 5)
CONF = [("S", "S", 100), ("S", "U", 8), ("S", "A", 4), ("U", "S", 6), ("U", "U", 60), ("U", "A", 5),
        ("A", "S", 3), ("A", "U", 4), ("A", "A", 50)]
FAILS, NCHK = [], [0]


def check(name, cond, detail=""):
    NCHK[0] += 1
    if not cond:
        FAILS.append(f"{name}: {detail}")


def close(name, got, want, tol=1e-9):
    check(name, got is not None and want is not None and abs(got - want) <= tol, f"got {got} want {want}")


def close_ci(name, got, want, tol=1e-9):
    ok = got is not None and want is not None and len(got) == 2 and all(abs(g - w) <= tol for g, w in zip(got, want))
    check(name, ok, f"got {got} want {want}")


# ------------------------------------------------------------------ synthetic data
def build():
    items, first_su = [], True
    for x, y, n in CONF:
        for _ in range(n):
            if x == y:
                adj, fin = None, x
            elif x != "S" and y != "S":
                adj = fin = x
            elif (x, y) == ("S", "U") and first_su:
                adj = fin = "U"; first_su = False
            else:
                adj = fin = "S"
            items.append({"a1": x, "a2": y, "adj": adj, "final": fin})
    ill = [it for it in items if it["final"] != "S"]; sol = [it for it in items if it["final"] == "S"]
    assert len(ill) == 120 and len(sol) == 120
    rng = random.Random(5); rng.shuffle(ill); rng.shuffle(sol)
    for r, ni, ns in ((1, 20, 60), (3, 40, 40), (5, 60, 20)):
        mem = [ill.pop() for _ in range(ni)] + [sol.pop() for _ in range(ns)]
        random.Random(6 + r).shuffle(mem)
        for j, it in enumerate(mem):
            it["round"], it["seed"] = r, (0 if j < 27 else 1 if j < 54 else 2)
    random.Random(7).shuffle(items)
    for j, it in enumerate(items):
        it["hid"] = f"D{j + 1:03d}"
    a7b = [{"key": None, "seed": s, "round": r} for s in (0, 1, 2) for r in ROUNDS for _ in range(100)]
    for j, x in enumerate(a7b):
        x["key"] = f"b{j:04d}"
    pool = {}
    for x in a7b:
        pool.setdefault((x["seed"], x["round"]), []).append(x["key"])
    rng8 = random.Random(8)
    for v in pool.values():
        rng8.shuffle(v)
    for it in items:
        it["a7b_key"] = pool[(it["seed"], it["round"])].pop()
    T = sorted(items, key=lambda it: it["hid"])
    for it in T:
        it["orig"] = it["blind"] = it["final"]
    FP, FN = {1: 6, 3: 8, 5: 10}, {1: 2, 3: 4, 5: 3}
    for r in ROUNDS:
        sol_r = [it for it in T if it["round"] == r and it["final"] == "S"]
        ill_r = [it for it in T if it["round"] == r and it["final"] != "S"]
        for it in sol_r[:FP[r]]:
            it["blind"] = "U"
        for it in ill_r[:FN[r]]:
            it["blind"] = "S"
        if r == 3:
            for it in ill_r[FN[r]:FN[r] + 5]:
                it["blind"] = "A" if it["final"] == "U" else "U"
    rng13 = random.Random(13)
    for it in T:
        it["open"] = it["final"] if rng13.random() > 0.2 else rng13.choice([c for c in "SUA" if c != it["final"]])
    null_hid = [it for it in T if it["round"] == 3][0]["hid"]
    for it in T:
        if it["hid"] == null_hid:
            it["open"] = None
    sampled = {it["a7b_key"]: it for it in T}
    rng21 = random.Random(21); pill = {1: 0.5, 3: 0.7, 5: 0.75}
    for x in a7b:
        if x["key"] in sampled:
            it = sampled[x["key"]]; x["orig"], x["blind"], x["open"] = it["orig"], it["blind"], it["open"]
        else:
            labs = []
            for _ in range(3):
                u = rng21.random(); v = rng21.random()
                labs.append(("U" if v < 2 / 3 else "A") if u < pill[x["round"]] else "S")
            x["orig"], x["blind"], x["open"] = labs
    return T, a7b, null_hid


def problem_text(k):  # distinct multi-line synthetic problem per A7b key (padded, as raw problem fields can be)
    n = int(k[1:])
    return f"  Synthetic problem {n}: a tank holds {n + 7} litres.\n\nIt loses {n % 9 + 1} litres per hour, starting at {n % 12}:00.\nHow long until it is empty?\n"


def write_files(d, T, a7b):
    os.makedirs(f"{d}/hv/_key_do_not_open"); os.makedirs(f"{d}/res")
    lines = ["hid,label,note"]
    for j, it in enumerate(T):  # ann1: UTF-8 BOM, CRLF, some padded lower-case labels, quoted notes with commas
        lab = f" {it['a1'].lower()} " if j % 17 == 0 else it["a1"]
        note = '"missing rate, ""per hour"""' if j % 11 == 0 else ""
        lines.append(f"{it['hid']},{lab},{note}")
    open(f"{d}/hv/labels_ann1.csv", "wb").write(("﻿" + "\r\n".join(lines) + "\r\n").encode("utf-8"))
    lines = ["hid,label,note"] + [f"{it['hid']},{it['a2']},{'ok' if it['a2'] == 'S' else 'needs x'}" for it in reversed(T)] + [",,"]
    open(f"{d}/hv/labels_ann2.csv", "w").write("\n".join(lines) + "\n")  # ann2: reversed order, LF, one blank row
    dis = [it for it in T if it["a1"] != it["a2"]]
    extra = [it for it in T if it["a1"] == it["a2"]][0]
    lines = ["hid,label"] + [f"{it['hid']},{it['adj']}" for it in dis] + [f"{extra['hid']},{extra['a1']}"]
    open(f"{d}/hv/labels_adjudicated.csv", "w").write("\n".join(lines) + "\n")
    lines = ["hid,ann1,ann2,adjudicated,rationale"] + [f"{it['hid']},{it['a1']},{it['a2']},{it['adj']},\"reason, synthetic\"" for it in dis]
    open(f"{d}/hv/adjudication_rationale.csv", "w").write("\n".join(lines) + "\n")
    json.dump([{"hid": it["hid"], "a7b_key": it["a7b_key"], "seed": it["seed"], "round": it["round"],
                "opus_original": WORD[it["orig"]], "opus_blind": WORD[it["blind"]]} for it in T],
              open(f"{d}/hv/_key_do_not_open/key.json", "w"), indent=1)
    with open(f"{d}/hv/packet.md", "w") as fh:  # same layout as dirA5/code/hv_drift_packet.py
        fh.write("# Synthetic packet\n\nLabel each question with exactly one of:\n- **S (SOLVABLE)**: ...\n\n")
        for it in T:
            fh.write(f"## {it['hid']}\n\n{problem_text(it['a7b_key']).strip()}\n\nLabel (S / A / U): ____   Note: ____\n\n")
    with open(f"{d}/res/key_full.jsonl", "w") as fh:
        for j, x in enumerate(a7b):
            fh.write(json.dumps({"seed": x["seed"], "round": x["round"], "run": "synthetic", "id": f"syn:{j}",
                                 "orig_verdict": WORD[x["orig"]], "problem": problem_text(x["key"]), "key": x["key"]}) + "\n")
    json.dump([{"key": x["key"], "verdict": WORD[x["blind"]], "confidence": 0.9, "note": "syn"} for x in a7b],
              open(f"{d}/res/blind_verdicts.json", "w"))
    with open(f"{d}/res/verdicts.jsonl", "w") as fh:
        for x in a7b:
            fh.write(json.dumps({"key": x["key"], "verdict": WORD[x["open"]] if x["open"] else None, "raw_tail": "", "retry": False}) + "\n")


def argv(d, out, synthetic=True):
    for p in (d, out):
        assert os.path.abspath(p).startswith(W + "/"), p
    a = [sys.executable, SCRIPT, "--hv-dir", f"{d}/hv", "--key", f"{d}/hv/_key_do_not_open/key.json",
         "--a7b-key-full", f"{d}/res/key_full.jsonl", "--a7b-blind", f"{d}/res/blind_verdicts.json",
         "--open-judge", f"{d}/res/verdicts.jsonl", "--out", out]
    return a + (["--synthetic"] if synthetic else [])


def run(d, out, synthetic=True):
    p = subprocess.run(argv(d, out, synthetic), capture_output=True, text=True, timeout=1500)
    return p.returncode, p.stdout + p.stderr


def edit_lines(path, fn):
    raw = open(path, "rb").read(); bom = raw.startswith(b"\xef\xbb\xbf"); t = raw.decode("utf-8-sig")
    nl = "\r\n" if "\r\n" in t else "\n"; lines = t.split(nl)
    lines = fn(lines)
    open(path, "wb").write((("﻿" if bom else "") + nl.join(lines)).encode("utf-8"))


# ------------------------------------------------------------------ independent recomputation
def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[t] * cb[t] for t in set(ca) | set(cb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else None


def wilson_alt(k, n):  # algebraically equivalent closed form, written independently of the script
    p = k / n; z2 = Z * Z; root = Z * math.sqrt(z2 + 4 * n * p * (1 - p))
    return [100 * max(0.0, (2 * n * p + z2 - root) / (2 * (n + z2))), 100 * min(1.0, (2 * n * p + z2 + root) / (2 * (n + z2)))]


def py_pct(vals):
    v = sorted(x for x in vals if x is not None and math.isfinite(x)); nv = len(v)
    return [v[int(0.025 * nv)], v[int(0.975 * nv) - 1]]


def item_reps(n):
    rng = random.Random(SEED)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(B)]


def strat_reps(cells, seed):
    rng = random.Random(seed); out = []
    for _ in range(B):
        row = []
        for c in cells:
            m = len(c)
            for _ in range(m):
                row.append(c[rng.randrange(m)])
        out.append(row)
    return out


def share(idx, pred, cond):
    sel = [i for i in idx if cond(i)]
    return 100 * sum(1 for i in sel if pred(i)) / len(sel) if sel else None


def dif(a, b):
    return None if a is None or b is None else a - b


def main():
    shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
    real_out_before = os.path.exists(REAL_OUT)
    try:
        T, a7b, null_hid = build()
        base = f"{W}/base"; write_files(base, T, a7b)
        out1, out2 = f"{W}/out1.json", f"{W}/out2.json"
        rc, log = run(base, out1)
        check("main run exit 0", rc == 0, log[-3000:])
        if rc != 0:
            raise SystemExit
        J = json.load(open(out1))
        check("summary printed", "PRIMARY D_h" in log and "EXPLORATORY" in log, log[-500:])
        N = 240
        fin = [it["final"] for it in T]; ill = [x != "S" for x in fin]
        rnd = [it["round"] for it in T]; sd = [it["seed"] for it in T]
        a1 = [it["a1"] for it in T]; a2 = [it["a2"] for it in T]
        IR = item_reps(N)
        cells = [[i for i in range(N) if sd[i] == s and rnd[i] == r] for s in (0, 1, 2) for r in ROUNDS]
        SR = strat_reps(cells, SEED)

        # ---- quality
        q = J["quality"]
        check("synthetic flag", J["synthetic"] is True)
        check("ann1 rows", q["ann1"]["n_data_rows"] == 240 and q["ann1"]["header_as_registered"])
        check("ann1 normalised labels", q["ann1"]["n_labels_normalised"] == 15 and q["ann1"]["n_labels_given_as_words"] == 0,
              (q["ann1"]["n_labels_normalised"], q["ann1"]["n_labels_given_as_words"]))
        check("ann2 rows/blank", q["ann2"]["n_data_rows"] == 240 and q["ann2"]["n_blank_rows_ignored"] == 1)
        g3, g2 = q["agreement"]["three_class"], q["agreement"]["binary_illposed_vs_solvable"]
        close("raw agreement 3-class", g3["raw_agreement"], 210 / 240)
        close("kappa 3-class (hand: 29717/36917)", g3["kappa"], 29717 / 36917)
        close("raw agreement binary", g2["raw_agreement"], 219 / 240)
        close("kappa binary (hand: 23584/28624)", g2["kappa"], 23584 / 28624)
        close("kappa 3-class via formula", g3["kappa"], kappa(list(zip(a1, a2))))
        check("confusion ann1 x ann2", all(q["agreement"]["confusion_ann1_rows_ann2_cols"][x][y] == n for x, y, n in CONF))
        check("n disagreements", q["agreement"]["n_disagree_three_class"] == 30 and q["agreement"]["n_disagree_binary"] == 21)
        kb3 = [kappa([(a1[i], a2[i]) for i in idx]) for idx in IR]
        kb2 = [kappa([(a1[i] != "S", a2[i] != "S") for i in idx]) for idx in IR]
        pb3 = [sum(a1[i] == a2[i] for i in idx) / N for idx in IR]
        close_ci("kappa 3-class bootstrap CI", g3["kappa_ci95"], py_pct(kb3))
        close_ci("kappa binary bootstrap CI", g2["kappa_ci95"], py_pct(kb2))
        close_ci("raw agreement bootstrap CI", g3["raw_agreement_ci95"], py_pct(pb3))
        check("kappa bootstrap replicates all valid", g3["boot_valid_replicates"] == B)
        ad = q["adjudication"]
        dis = [it for it in T if it["a1"] != it["a2"]]
        check("adjudication extra row reported", ad["n_extra_rows_on_agreed_items"] == 1 and ad["n_extra_rows_conflicting_with_common_label"] == 0)
        check("adjudicated equals ann1/ann2", ad["adjudicated_equals"] == {"ann1": sum(it["adj"] == it["a1"] for it in dis),
                                                                          "ann2": sum(it["adj"] == it["a2"] for it in dis), "neither": 0}, ad["adjudicated_equals"])
        rc_ = q["rationale_consistency"]
        check("rationale consistency", rc_["checked"] and rc_["hid_set_equals_disagreements"]
              and all(v == 0 for v in rc_["column_mismatches_vs_label_files"].values()), rc_)

        # ---- final label
        fl = J["final_label"]
        check("final label counts", fl["n_from_adjudication"] == 30 and fl["label_counts"] == {x: fin.count(x) for x in "SUA"}, fl)
        check("three-class final label available", fl["three_class_final_label_available"] is True
              and fl["n_u_vs_a_without_adjudication_binary_illposed"] == 0 and fl["label_counts_binary"] == {"S": 120, "U_or_A": 120}, fl)

        # ---- primary
        pr = J["primary"]; sh = pr["illposed_share_by_round"]
        for r, k, lit in ((1, 20, [16.81, 35.48]), (3, 40, [39.30, 60.70]), (5, 60, [64.52, 83.19])):
            v = sh[f"r{r}"]
            check(f"share r{r}", v["k"] == k and v["n"] == 80 and abs(v["share_pct"] - 100 * k / 80) < 1e-12, v)
            close_ci(f"wilson r{r} (alt formula)", v["wilson95_pct"], wilson_alt(k, 80), 1e-9)
            close_ci(f"wilson r{r} (hand, 2 dp)", v["wilson95_pct"], lit, 0.006)
        d = pr["D_h_r5_minus_r1"]
        close("D_h r5 - r1 = 50", d["d_pts"], 50.0)
        bs = [dif(share(idx, lambda i: ill[i], lambda i: rnd[i] == 5), share(idx, lambda i: ill[i], lambda i: rnd[i] == 1)) for idx in SR]
        ci = py_pct(bs); close_ci("D_h bootstrap CI (independent)", d["ci95_pts"], ci)
        want_read = "RISES" if ci[0] > 0 else "FALLS" if ci[1] < 0 else "FLAT" if (ci[0] >= -10 and ci[1] <= 10) else "INCONCLUSIVE"
        check("D_h reading", d["reading"] == want_read == "RISES", d)
        st = J["secondary_human_trend"]
        close("D_h r3 - r1 = 25", st["D_h_r3_minus_r1"]["d_pts"], 25.0)
        bs31 = [dif(share(idx, lambda i: ill[i], lambda i: rnd[i] == 3), share(idx, lambda i: ill[i], lambda i: rnd[i] == 1)) for idx in SR]
        close_ci("D_h r3 - r1 CI", st["D_h_r3_minus_r1"]["ci95_pts"], py_pct(bs31))
        for s in (0, 1, 2):
            want = share(range(N), lambda i: ill[i], lambda i: rnd[i] == 5 and sd[i] == s) - share(range(N), lambda i: ill[i], lambda i: rnd[i] == 1 and sd[i] == s)
            close(f"per-seed D_h s{s}", st["D_h_r5_minus_r1_per_seed"][f"seed{s}"]["d_pts"], want)
        bss = [dif(share(idx, lambda i: ill[i], lambda i: rnd[i] == 5 and sd[i] == 1), share(idx, lambda i: ill[i], lambda i: rnd[i] == 1 and sd[i] == 1)) for idx in SR]
        close_ci("per-seed D_h s1 CI", st["D_h_r5_minus_r1_per_seed"]["seed1"]["ci95_pts"], py_pct(bss))
        lt = st["logistic_trend_over_round"]
        close("logistic slope = ln(3)/2", lt["b1_per_round"], math.log(3) / 2, 1e-8)
        close("logistic intercept = -1.5 ln 3", lt["b0"], -1.5 * math.log(3), 1e-8)
        close("logistic SE = sqrt(1/120)", lt["se_b1"], math.sqrt(1 / 120), 1e-8)
        close_ci("logistic Wald CI", lt["wald_ci95_b1"], [math.log(3) / 2 - Z * math.sqrt(1 / 120), math.log(3) / 2 + Z * math.sqrt(1 / 120)], 1e-8)
        check("logistic converged", lt["status"] == "estimated" and lt["converged"] and not lt["separation_suspected"], lt)
        close("logistic odds ratio per round = sqrt(3)", lt["odds_ratio_per_round"], math.sqrt(3), 1e-8)
        close("logistic odds ratio r5 vs r1 = 9", lt["odds_ratio_r5_vs_r1"], 9.0, 1e-7)

        # ---- judges
        JJ = J["secondary_judge_validity"]["judges"]
        jo = JJ["opus_original"]
        close("orig kappa 3-class = 1", jo["agreement_three_class"]["kappa"], 1.0)
        close_ci("orig kappa CI = [1, 1]", jo["agreement_binary"]["kappa_ci95"], [1.0, 1.0])
        for r, ns, up in ((1, 60, 6.0172), (3, 40, 8.7621), (5, 20, 16.1124)):
            v = jo["false_positive_rate_by_round"][f"r{r}"]
            check(f"orig FPR r{r} = 0", v["k"] == 0 and v["n"] == ns and v["share_pct"] == 0, v)
            close_ci(f"orig FPR r{r} Wilson (hand)", v["wilson95_pct"], [0.0, up], 1e-3)
        a = jo["illposed_detection_vs_humans"]["all"]
        check("orig precision/recall/F1 = 1", a["precision"] == 1 and a["recall"] == 1 and a["f1"] == 1, a)
        dd = jo["drift_on_same_items"]
        close("orig judge D = D_h", dd["judge_D"]["d_pts"], 50.0)
        close_ci("orig paired diff CI = [0, 0]", dd["judge_minus_human_paired"]["ci95_pts"], [0.0, 0.0])

        jb = JJ["opus_blind"]; bl = [it["blind"] for it in T]; bill = [x != "S" for x in bl]
        check("blind joined 240", jb["n_joined"] == 240)
        close("blind kappa 3-class", jb["agreement_three_class"]["kappa"], kappa(list(zip(fin, bl))))
        close("blind kappa binary", jb["agreement_binary"]["kappa"], kappa(list(zip(ill, bill))))
        close_ci("blind kappa binary CI", jb["agreement_binary"]["kappa_ci95"], py_pct([kappa([(ill[i], bill[i]) for i in idx]) for idx in IR]))
        close_ci("blind kappa 3-class CI", jb["agreement_three_class"]["kappa_ci95"], py_pct([kappa([(fin[i], bl[i]) for i in idx]) for idx in IR]))
        close("blind accuracy binary", jb["agreement_binary"]["raw_agreement"], sum(a == b for a, b in zip(ill, bill)) / 240)
        a = jb["illposed_detection_vs_humans"]["all"]
        check("blind TP/FP/FN (hand 111/24/9)", (a["tp"], a["fp"], a["fn"]) == (111, 24, 9), a)
        close("blind precision 111/135", a["precision"], 111 / 135); close("blind recall 111/120", a["recall"], 111 / 120)
        close("blind F1 222/255", a["f1"], 222 / 255)
        for r, tp, fp, fn in ((1, 18, 6, 2), (3, 36, 8, 4), (5, 57, 10, 3)):
            v = jb["illposed_detection_vs_humans"][f"r{r}"]
            check(f"blind r{r} counts", (v["tp"], v["fp"], v["fn"]) == (tp, fp, fn), v)
            close(f"blind r{r} F1", v["f1"], 2 * tp / (2 * tp + fp + fn))
        for r, k, n, want in ((1, 6, 60, 10.0), (3, 8, 40, 20.0), (5, 10, 20, 50.0)):
            v = jb["false_positive_rate_by_round"][f"r{r}"]
            check(f"blind FPR r{r} = {want}", v["k"] == k and v["n"] == n and abs(v["share_pct"] - want) < 1e-12, v)
            close_ci(f"blind FPR r{r} Wilson", v["wilson95_pct"], wilson_alt(k, n))
        close("blind FPR r5 - r1 = 40", jb["fpr_r5_minus_r1"]["d_pts"], 40.0)
        bf = [dif(share(idx, lambda i: bill[i], lambda i: rnd[i] == 5 and not ill[i]), share(idx, lambda i: bill[i], lambda i: rnd[i] == 1 and not ill[i])) for idx in SR]
        close_ci("blind FPR diff CI", jb["fpr_r5_minus_r1"]["ci95_pts"], py_pct(bf))
        check("blind FPR diff valid replicates", jb["fpr_r5_minus_r1"]["boot_valid_replicates"] == sum(x is not None for x in bf))
        close("blind judge D = 53.75", jb["drift_on_same_items"]["judge_D"]["d_pts"], 53.75)
        bj = [dif(share(idx, lambda i: bill[i], lambda i: rnd[i] == 5), share(idx, lambda i: bill[i], lambda i: rnd[i] == 1)) for idx in SR]
        close_ci("blind judge D CI", jb["drift_on_same_items"]["judge_D"]["ci95_pts"], py_pct(bj))
        close_ci("blind judge - human paired CI", jb["drift_on_same_items"]["judge_minus_human_paired"]["ci95_pts"],
                 py_pct([dif(x, y) for x, y in zip(bj, bs)]))

        jw = JJ["open_weight_qwen2.5_32b"]; ow = [it["open"] for it in T]; have = [i for i in range(N) if ow[i] is not None]
        check("open judge joined 239", jw["n_joined"] == 239 and jw["n_not_joined"] == 1, (jw["n_joined"], null_hid))
        close("open kappa 3-class", jw["agreement_three_class"]["kappa"], kappa([(fin[i], ow[i]) for i in have]))
        close("open kappa binary", jw["agreement_binary"]["kappa"], kappa([(ill[i], ow[i] != "S") for i in have]))
        hs = set(have)
        close_ci("open kappa binary CI (masked)", jw["agreement_binary"]["kappa_ci95"],
                 py_pct([kappa([(ill[i], ow[i] != "S") for i in idx if i in hs]) for idx in IR]))
        oill = [x is not None and x != "S" for x in ow]
        wantD = share(have, lambda i: oill[i], lambda i: rnd[i] == 5) - share(have, lambda i: oill[i], lambda i: rnd[i] == 1)
        close("open judge D (239 items)", jw["drift_on_same_items"]["judge_D"]["d_pts"], wantD)
        wantH = share(have, lambda i: ill[i], lambda i: rnd[i] == 5) - share(have, lambda i: ill[i], lambda i: rnd[i] == 1)
        close("human D_h on the open judge's 239", jw["drift_on_same_items"]["human_D_h_same_items"]["d_pts"], wantH)
        bo = [dif(share(idx, lambda i: oill[i], lambda i: rnd[i] == 5 and i in hs), share(idx, lambda i: oill[i], lambda i: rnd[i] == 1 and i in hs)) for idx in SR]
        close_ci("open judge D CI (masked)", jw["drift_on_same_items"]["judge_D"]["ci95_pts"], py_pct(bo))
        conf = jw["confusion_human_rows_judge_cols"]
        check("open confusion", all(conf[x][y] == sum(1 for i in have if fin[i] == x and ow[i] == y) for x in "SUA" for y in "SUA"))
        v = jw["false_positive_rate_by_round"]["r3"]
        check("open FPR r3 denominator", v["n"] == sum(1 for i in have if rnd[i] == 3 and not ill[i]), v)

        # ---- robustness
        rb = J["robustness"]
        for nm, lab in (("annotator1_alone", [x != "S" for x in a1]), ("annotator2_alone", [x != "S" for x in a2]),
                        ("unsolvable_only_final", [x == "U" for x in fin])):
            want = share(range(N), lambda i: lab[i], lambda i: rnd[i] == 5) - share(range(N), lambda i: lab[i], lambda i: rnd[i] == 1)
            close(f"robustness {nm} D", rb[nm]["D_r5_minus_r1"]["d_pts"], want)
        lab1 = [x != "S" for x in a1]
        close_ci("robustness ann1 CI", rb["annotator1_alone"]["D_r5_minus_r1"]["ci95_pts"],
                 py_pct([dif(share(idx, lambda i: lab1[i], lambda i: rnd[i] == 5), share(idx, lambda i: lab1[i], lambda i: rnd[i] == 1)) for idx in SR]))

        # ---- exploratory (point estimate, recomputed)
        ex = J["exploratory"]; pis = {}
        for r in ROUNDS:
            p = sum(x["blind"] != "S" for x in a7b if x["round"] == r) / sum(x["round"] == r for x in a7b)
            se = sum(1 for i in range(N) if rnd[i] == r and ill[i] and bill[i]) / sum(1 for i in range(N) if rnd[i] == r and ill[i])
            sp = sum(1 for i in range(N) if rnd[i] == r and not ill[i] and not bill[i]) / sum(1 for i in range(N) if rnd[i] == r and not ill[i])
            pi = min(1.0, max(0.0, (p + sp - 1) / (se + sp - 1))); pis[r] = pi
            e = ex["per_round"][f"r{r}"]
            close(f"exploratory p_obs r{r}", e["p_obs"], p); close(f"exploratory Se r{r}", e["sensitivity"], se)
            close(f"exploratory Sp r{r}", e["specificity"], sp); close(f"exploratory pi r{r}", e["pi_clipped"], pi)
        close("exploratory corrected D", ex["D_corrected_900"]["d_pts"], 100 * (pis[5] - pis[1]))
        u = lambda r: 100 * sum(x["blind"] != "S" for x in a7b if x["round"] == r) / 300
        close("exploratory uncorrected D (900)", ex["D_uncorrected_900"]["d_pts"], u(5) - u(1))
        check("exploratory CI present", ex["D_corrected_900"]["ci95_pts"] is not None and ex["D_corrected_900"]["boot_valid_replicates"] > 3900)
        # independent recomputation of both exploratory CIs (900 stream: cells in (seed, round) order, items in key order)
        k9 = sorted(x["key"] for x in a7b); A9 = {x["key"]: x for x in a7b}
        rd9 = [A9[k]["round"] for k in k9]; ob9 = [A9[k]["blind"] != "S" for k in k9]
        hmap = {it["a7b_key"]: (it["final"] != "S") for it in T}
        hs9 = [k in hmap for k in k9]; hi9 = [hmap.get(k, False) for k in k9]
        cells9 = [[i for i in range(900) if A9[k9[i]]["seed"] == s and rd9[i] == r] for s in (0, 1, 2) for r in ROUNDS]
        SR9 = strat_reps(cells9, SEED9)

        def rg_rep(idx240, idx900):
            pis = {}
            for r in (1, 5):
                sel = [i for i in idx900 if rd9[i] == r]; p = sum(ob9[i] for i in sel) / len(sel)
                hi = [i for i in idx240 if rnd[i] == r and ill[i]]; so = [i for i in idx240 if rnd[i] == r and not ill[i]]
                if not hi or not so:
                    return None
                se = sum(bill[i] for i in hi) / len(hi); sp = sum(not bill[i] for i in so) / len(so)
                if se + sp - 1 <= 0:
                    return None
                pis[r] = min(1.0, max(0.0, (p + sp - 1) / (se + sp - 1)))
            return 100 * (pis[5] - pis[1])

        def tp_rep(idx900):
            pis = {}
            for r in (1, 5):
                sel = [i for i in idx900 if rd9[i] == r]; p = sum(ob9[i] for i in sel) / len(sel)
                jp = [i for i in sel if hs9[i] and ob9[i]]; jn = [i for i in sel if hs9[i] and not ob9[i]]
                if (p > 0 and not jp) or (p < 1 and not jn):
                    return None
                pis[r] = (sum(hi9[i] for i in jp) / len(jp) * p if p > 0 else 0.0) + (sum(hi9[i] for i in jn) / len(jn) * (1 - p) if p < 1 else 0.0)
            return 100 * (pis[5] - pis[1])
        rg_bs = [rg_rep(a, b) for a, b in zip(SR, SR9)]
        close_ci("exploratory Rogan-Gladen CI (independent)", ex["D_corrected_900"]["ci95_pts"], py_pct(rg_bs))
        check("exploratory Rogan-Gladen valid replicates", ex["D_corrected_900"]["boot_valid_replicates"] == sum(x is not None for x in rg_bs),
              (ex["D_corrected_900"]["boot_valid_replicates"], sum(x is not None for x in rg_bs)))
        close("exploratory two-phase D (independent)", ex["D_two_phase_900"]["d_pts"], tp_rep(list(range(900))))
        for r in ROUNDS:
            sel = [i for i in range(900) if rd9[i] == r]; jp = [i for i in sel if hs9[i] and ob9[i]]; jn = [i for i in sel if hs9[i] and not ob9[i]]
            e = ex["per_round_two_phase"][f"r{r}"]
            close(f"two-phase PPV r{r}", e["ppv"], sum(hi9[i] for i in jp) / len(jp))
            close(f"two-phase 1-NPV r{r}", e["one_minus_npv"], sum(hi9[i] for i in jn) / len(jn))
            check(f"two-phase n sampled r{r}", e["n_sampled_judge_illposed"] + e["n_sampled_judge_solvable"] == 80, e)
        tp_bs = [tp_rep(b) for b in SR9]
        close_ci("exploratory two-phase CI (independent)", ex["D_two_phase_900"]["ci95_pts"], py_pct(tp_bs))
        check("exploratory two-phase valid replicates", ex["D_two_phase_900"]["boot_valid_replicates"] == sum(x is not None for x in tp_bs))
        close_ci("exploratory uncorrected CI (independent)", ex["D_uncorrected_900"]["ci95_pts"],
                 py_pct([100 * (sum(ob9[i] for i in b if rd9[i] == 5) / sum(rd9[i] == 5 for i in b)
                                - sum(ob9[i] for i in b if rd9[i] == 1) / sum(rd9[i] == 1 for i in b)) for b in SR9]))

        # ---- descriptive / integrity
        dc = J["descriptive"]["final_label_counts_by_round"]
        for r in ROUNDS:
            check(f"counts r{r}", dc[f"r{r}"] == {"n": 80, "U_or_A": sum(1 for i in range(N) if rnd[i] == r and fin[i] != "S"),
                                                "U": sum(1 for i in range(N) if rnd[i] == r and fin[i] == "U"),
                                                "A": sum(1 for i in range(N) if rnd[i] == r and fin[i] == "A"),
                                                "S": sum(1 for i in range(N) if rnd[i] == r and fin[i] == "S")}, dc[f"r{r}"])
        ig = J["integrity"]
        check("integrity", ig["key_opus_original_differs_from_key_full"] == 0 and ig["key_opus_blind_differs_from_blind_verdicts"] == 0
              and ig["open_judge_file"]["n_lines"] == 900 and ig["open_judge_file"]["n_lines_without_valid_verdict"] == 1
              and ig["n_packet_text_differs_from_key_full_problem"] == 0, ig)
        check("packet input hashed", "packet" in J["inputs"], J["inputs"].keys())

        # ---- determinism and overwrite refusal
        rc2, log2 = run(base, out2)
        check("second run exit 0", rc2 == 0, log2[-1000:])
        check("bootstrap reproducibility: identical bytes", open(out1, "rb").read() == open(out2, "rb").read())
        before = open(out1, "rb").read()
        rc3, log3 = run(base, out1)
        check("overwrite refused (exit 3)", rc3 == 3 and "REFUSED" in log3, (rc3, log3[-300:]))
        check("overwrite left the file unchanged", open(out1, "rb").read() == before)

        # ---- runs that must succeed
        def ok_case(name, mutate):
            d = f"{W}/ok_{name}"; shutil.copytree(base, d); mutate(d)
            o = f"{W}/out_ok_{name}.json"; rcx, lg = run(d, o)
            check(f"run {name}: exit 0", rcx == 0 and os.path.exists(o), (rcx, lg[-600:]))
            return json.load(open(o)) if rcx == 0 and os.path.exists(o) else None

        def relabel(path, fn):  # rewrite the label column of a hid,label[,note] file
            def f_(L):
                out = [L[0]]
                for x in L[1:]:
                    p = x.split(",", 2)
                    if len(p) >= 2 and p[0].startswith("D"):
                        p[1] = fn(p[0], p[1])
                    out.append(",".join(p))
                return out
            edit_lines(path, f_)
        cyc = [str.title, str.upper, str.lower]

        def words(d):
            relabel(f"{d}/hv/labels_ann2.csv", lambda h, l: cyc[int(h[1:]) % 3](WORD[l.strip().upper()]))
            relabel(f"{d}/hv/labels_adjudicated.csv", lambda h, l: " " + WORD[l.strip().upper()].lower() + " ")
        Jw = ok_case("full_word_labels", words)
        if Jw:
            check("words: ann2 all mapped", Jw["quality"]["ann2"]["n_labels_given_as_words"] == 240
                  and Jw["quality"]["ann2"]["label_counts"] == q["ann2"]["label_counts"], Jw["quality"]["ann2"])
            check("words: adjudicated mapped", Jw["quality"]["adjudication"]["n_labels_given_as_words"] == 31, Jw["quality"]["adjudication"])
            check("words: same results as letters", Jw["primary"] == J["primary"] and Jw["quality"]["agreement"] == q["agreement"]
                  and Jw["final_label"] == J["final_label"] and Jw["secondary_judge_validity"] == J["secondary_judge_validity"])

        ua = {it["hid"] for it in T if it["a1"] != it["a2"] and it["a1"] != "S" and it["a2"] != "S"}
        Ju = ok_case("u_vs_a_without_adjudication", lambda d: edit_lines(f"{d}/hv/labels_adjudicated.csv",
                     lambda L: [x for x in L if x.split(",")[0] not in ua]))
        if Ju:
            fu = Ju["final_label"]
            check("U-vs-A fallback: flagged", len(ua) == 9 and fu["three_class_final_label_available"] is False
                  and fu["n_u_vs_a_without_adjudication_binary_illposed"] == 9 and fu["n_from_adjudication"] == 21
                  and "not available" in fu["label_counts"]["status"] and fu["label_counts_binary"] == {"S": 120, "U_or_A": 120}, fu)
            check("U-vs-A fallback: primary unchanged", Ju["primary"] == J["primary"])
            check("U-vs-A fallback: binary trend unchanged", Ju["secondary_human_trend"] == J["secondary_human_trend"])
            ok3 = True
            for nm, jj in Ju["secondary_judge_validity"]["judges"].items():
                j0 = J["secondary_judge_validity"]["judges"][nm]
                ok3 &= ("not available" in jj["agreement_three_class"]["status"] and "not available" in jj["confusion_human_rows_judge_cols"]["status"]
                        and all(jj[x] == j0[x] for x in ("agreement_binary", "illposed_detection_vs_humans", "false_positive_rate_by_round",
                                                         "fpr_r5_minus_r1", "drift_on_same_items")))
            check("U-vs-A fallback: judges binary unchanged, three-class not available", ok3)
            check("U-vs-A fallback: robustness", "not available" in Ju["robustness"]["unsolvable_only_final"]["status"]
                  and Ju["robustness"]["annotator1_alone"] == J["robustness"]["annotator1_alone"])
            check("U-vs-A fallback: exploratory unchanged", Ju["exploratory"] == J["exploratory"])
            check("U-vs-A fallback: descriptive without U/A", all("U" not in v and "U_or_A" in v for v in Ju["descriptive"]["final_label_counts_by_round"].values()))
            check("U-vs-A fallback: adjudication bookkeeping", Ju["quality"]["adjudication"]["n_used"] == 21
                  and Ju["quality"]["adjudication"]["n_u_vs_a_disagreements_without_adjudication"] == 9, Ju["quality"]["adjudication"])

        # (quasi-)separated round patterns: both annotators give the same label; the adjudication rows become ignored extras
        rk = {it["hid"]: (it["round"], j) for j, it in enumerate(T)}
        for name, rule, want_d in (("sep_r1r3_S_r5_mixed", lambda r, j: "S" if r < 5 else ("U" if j % 2 else "S"), None),
                                   ("sep_r1_S_r5_ill", lambda r, j: "S" if r == 1 else "A" if r == 5 else ("U" if j % 2 else "S"), 100.0),
                                   ("sep_r3r5_ill", lambda r, j: "U" if r > 1 else ("U" if j % 2 else "S"), None),
                                   ("sep_complete", lambda r, j: "S" if r < 5 else "U", 100.0),
                                   ("sep_reverse", lambda r, j: "U" if r == 1 else "S", -100.0)):
            def mut(d, rule=rule):
                for fn in ("labels_ann1.csv", "labels_ann2.csv"):
                    relabel(f"{d}/hv/{fn}", lambda h, l: rule(*rk[h]))
            Js = ok_case(name, mut)
            if Js:
                lts = Js["secondary_human_trend"]["logistic_trend_over_round"]
                check(f"{name}: logistic not estimable (separation)", "(quasi-)separation" in lts.get("status", "") and "b1_per_round" not in lts, lts)
                ds = Js["primary"]["D_h_r5_minus_r1"]
                ill_s = {h: rule(*rk[h]) != "S" for h in rk}
                want = share(range(N), lambda i: ill_s[T[i]["hid"]], lambda i: rnd[i] == 5) - share(range(N), lambda i: ill_s[T[i]["hid"]], lambda i: rnd[i] == 1)
                close(f"{name}: primary D_h computed", ds["d_pts"], want)
                if want_d is not None:
                    check(f"{name}: D_h {want_d} with a degenerate CI", ds["d_pts"] == want_d and ds["ci95_pts"] == [want_d, want_d], ds)
                check(f"{name}: primary reading present", ds["reading"] in ("RISES", "FALLS", "FLAT", "INCONCLUSIVE"), ds)

        # ---- abort paths
        HIDRX = re.compile(r"D\d{3}")

        def case(name, mutate, expect, synthetic=True):
            d = f"{W}/case_{name}"; shutil.copytree(base, d); mutate(d)
            o = f"{W}/out_{name}.json"; rcx, lg = run(d, o, synthetic)
            check(f"abort {name}: exit 2", rcx == 2, (rcx, lg[-400:]))
            check(f"abort {name}: NOT RUN + reason", "NOT RUN" in lg and expect in lg, lg[-400:])
            check(f"abort {name}: nothing written", not os.path.exists(o))
            check(f"abort {name}: no hid in the message", not HIDRX.search(lg), lg[-400:])
        sil = [it for it in dis if (it["a1"] == "S") != (it["a2"] == "S")]
        first_dis, first_sil = dis[0]["hid"], sil[0]["hid"]
        case("missing_adjudication_s_vs_ill", lambda d: edit_lines(f"{d}/hv/labels_adjudicated.csv",
             lambda L: [x for x in L if not x.startswith(first_sil + ",")]), "1 S-vs-ill-posed disagreement(s) without an adjudicated label")
        case("invalid_label", lambda d: edit_lines(f"{d}/hv/labels_ann1.csv",
             lambda L: [("D005,X," if x.startswith("D005,") else x) for x in L]), "labels_ann1.csv: 1 invalid or empty label(s)")
        case("invalid_label_on_adjudicated_item", lambda d: edit_lines(f"{d}/hv/labels_ann1.csv",
             lambda L: [(f"{first_sil},?," if x.startswith(first_sil + ",") else x) for x in L]), "labels_ann1.csv: 1 invalid or empty label(s)")
        case("empty_label", lambda d: edit_lines(f"{d}/hv/labels_ann2.csv",
             lambda L: [("D010,,note" if x.startswith("D010,") else x) for x in L]), "labels_ann2.csv: 1 invalid or empty label(s)")
        case("wrong_row_count", lambda d: edit_lines(f"{d}/hv/labels_ann2.csv",
             lambda L: [x for x in L if not x.startswith("D100,")]), "239 data rows, expected 240")
        case("duplicate_row", lambda d: edit_lines(f"{d}/hv/labels_ann1.csv",
             lambda L: L[:3] + [L[2]] + L[3:]), "241 data rows, expected 240")
        other = {"S": "U", "U": "A", "A": "S"}[dis[0]["adj"]]
        case("conflicting_adjudication", lambda d: edit_lines(f"{d}/hv/labels_adjudicated.csv",
             lambda L: [x for x in L if x] + [f"{first_dis},{other}", ""]), "1 hid(s) with conflicting duplicate rows")

        def swap_seed(d):
            p = f"{d}/hv/_key_do_not_open/key.json"; k = json.load(open(p))
            i = next(j for j, e in enumerate(k) if e["round"] == 1 and e["seed"] == 0)
            j = next(j for j, e in enumerate(k) if e["round"] == 1 and e["seed"] == 1)
            k[i]["seed"], k[j]["seed"] = 1, 0; json.dump(k, open(p, "w"))
        case("key_vs_a7b_mismatch", swap_seed, "2 hid(s): seed/round differ from key_full.jsonl")

        def swap_key_same_cell(d):  # seed/round still agree with key_full; only the packet text reveals the swap
            p = f"{d}/hv/_key_do_not_open/key.json"; k = json.load(open(p))
            i, j = [n for n, e in enumerate(k) if e["round"] == 3 and e["seed"] == 2][:2]
            k[i]["a7b_key"], k[j]["a7b_key"] = k[j]["a7b_key"], k[i]["a7b_key"]; json.dump(k, open(p, "w"))
        case("key_vs_packet_mismatch", swap_key_same_cell, "2 hid(s): packet.md text differs from the key_full.jsonl problem")
        case("packet_unparsable", lambda d: edit_lines(f"{d}/hv/packet.md",
             lambda L: [x for x in L if not x.startswith("## D017")]), "packet.md: 239 parsed sections")
        case("open_judge_malformed", lambda d: open(f"{d}/res/verdicts.jsonl", "a").write('{"key": "b0001", "verdict": \n'),
             "open-judge verdicts.jsonl malformed")
        case("headerless_annotator_file", lambda d: edit_lines(f"{d}/hv/labels_ann2.csv", lambda L: L[1:]),
             "labels_ann2.csv: header lacks a hid or label column (header not printed)")
        case("registered_hash_mismatch", lambda d: None, "differs from the registered", synthetic=False)

        # ---- unit checks (import the script; no data files involved)
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("human_read_under_test", SCRIPT)
        hr = importlib.util.module_from_spec(spec); spec.loader.exec_module(hr)
        for ci, want in (([0.0, 30.0], "INCONCLUSIVE"), ([0.0, 10.0], "FLAT"), ([0.0, 10.000001], "INCONCLUSIVE"),
                         ([1e-12, 50.0], "RISES"), ([1.25, 31.25], "RISES"), ([-10.0, -0.0], "FLAT"), ([-30.0, -1e-12], "FALLS"),
                         ([-10.000001, 5.0], "INCONCLUSIVE"), ([-5.0, 5.0], "FLAT"), ([0.0, 0.0], "FLAT"), (None, None)):
            check(f"reading {ci}", hr.reading(ci) == want, hr.reading(ci))
        np_ = hr.np

        def pattern(k1, k3, k5, n=80):
            y, x = [], []
            for r, k in ((1, k1), (3, k3), (5, k5)):
                y += [1] * k + [0] * (n - k); x += [r] * n
            return np_.array(y), np_.array(x)
        for ks, want in (((0, 0, 40), "(quasi-)separation"), ((0, 40, 80), "(quasi-)separation"), ((40, 80, 80), "(quasi-)separation"),
                         ((0, 0, 80), "(quasi-)separation"), ((80, 40, 0), "(quasi-)separation"), ((80, 80, 0), "(quasi-)separation"),
                         ((0, 0, 0), "all labels equal"), ((80, 80, 80), "all labels equal")):
            try:
                o = hr.logistic_trend(*pattern(*ks)); got = o.get("status", "")
            except Exception as e:  # the old code raised LinAlgError / OverflowError here
                got = f"raised {type(e).__name__}"
            check(f"logistic {ks}: {want}", want in got, got)
        for ks in ((0, 1, 79), (1, 0, 80), (20, 40, 60), (3, 70, 5)):
            y, x = pattern(*ks); o = hr.logistic_trend(y, x)
            ok = o.get("status") == "estimated" and o["converged"] and all(math.isfinite(o[v]) for v in ("b0", "b1_per_round", "se_b1"))
            if ok:
                p = 1 / (1 + np_.exp(-(o["b0"] + o["b1_per_round"] * x)))
                ok = abs(float((y - p).sum())) < 1e-6 and abs(float((x * (y - p)).sum())) < 1e-6
            check(f"logistic {ks}: estimated, score equations = 0 at the MLE", ok, o)
        real_solve = np_.linalg.solve
        try:
            def boom(*a, **k):
                raise np_.linalg.LinAlgError("Singular matrix")
            np_.linalg.solve = boom
            o = hr.logistic_trend(*pattern(20, 40, 60))
            check("logistic LinAlgError caught", "numerical failure" in o.get("status", ""), o)
        except Exception as e:
            check("logistic LinAlgError caught", False, repr(e))
        finally:
            np_.linalg.solve = real_solve
        check("sexp overflow -> None", hr.sexp(1000.0) is None and hr.sexp(0.0) == 1.0)
        check("human_label words", [hr.human_label(v) for v in (" Solvable", "UNSOLVABLE ", "ambiguous", "s", " a ", "?", "")] ==
              [("S", True), ("U", True), ("A", True), ("S", False), ("A", False), ("?", False), ("", False)])
    except SystemExit:
        pass
    finally:
        shutil.rmtree(W, ignore_errors=True)
    check("real human_read.json untouched", os.path.exists(REAL_OUT) == real_out_before)
    check("synthetic dir deleted", not os.path.exists(W))
    print(f"{NCHK[0]} checks, {len(FAILS)} failures")
    for x in FAILS:
        print("FAIL", x)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
