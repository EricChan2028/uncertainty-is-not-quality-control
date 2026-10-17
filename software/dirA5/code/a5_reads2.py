"""A5 reads for TG (T-P1..T-P4 + kill gate), HZN (H-P1..H-P5), SWAP (S-P1..S-P3), CTRL-L (C-P1..C-P5), registered 08:04 on
24 Sep in dirA5/docs/A5_registration_2026-09-24.md (text implemented verbatim). v1 written at 15:15; v2 after a two-reviewer
adversarial check (v1 was never run; no read of these statistics had been computed or viewed). Shared definitions (as the A4
reads): Q1u = share of the 412 validated unsolvable twins answered confidently (p_mode >= 0.5, plain prompt), complete probe
files only (all 1,086 (fam, task, form) keys present after de-duplication; malformed lines skipped); lenient mean accuracy
over MATH-500 / held-out-1000 / GSM8K; intact abstention = mean abstention share over the 543 intact twins (abstention prompt);
UMWP CA = share of the 600 unanswerable items whose plain-prompt numeric mode holds >= 0.5 of 8 samples (umwp_read).
Pooled item-paired bootstrap (registered "over the seeds"): PRIMARY = item-clustered, one draw of 412 item indices per
replicate applied to every seed (the seeds share the same twins); the within-seed independent draw is a disclosed secondary.
4,000 replicates, seed 20260924; one-sided p = (1 + #{pooled diff <= 0}) / (B + 1) for the registered direction.
k-of-n verdicts are three-state: True if hits >= k; False if hits + unread < k; None otherwise (unread seeds recorded).
Disclosed readings (fixed before any read): C-P3 validity is computed on the common r1 universe U = items in the LOWP plain band
AND in the guard's own (abstention-prompt) band 0.25 <= p_mode <= 0.75 in its r1 guard_records, so both rules act on the same
items: guard-rejected = U minus guard admitted (the abstention-rule rejections), LOWP-rejected = U minus LOWP admitted; the
per-seed Jaccard is averaged over the 3 seeds and all 3 are required; the plain-band-difference version and a version with
guard-rejected = band items with abstain >= 2/8 are descriptive; C-P3_raw is kept when declared uninformative. SWAP: at least 3 powered seeds must remain, else
S-P1..S-P3 are 'underpowered'; S-P1/S-P2 keep the absolute threshold 3, S-P3 uses min(4, n_powered). C-P5 = 3 of 3 (strict
reading). TG kill gate: the reference for "accuracy drop > 5" is not named in the registration; the gate fires if the T4 accuracy
is > 5 below EITHER loop_uv r4 or the shared rescue_uv r3 start (the stricter reading; which one fired is recorded);
T5_gate = T-P2 and T-P3 hold and the kill gate is not triggered. HZN/CTRL: the constants printed in the registration
(own r5 Q1u 0.478/0.345/0.571, accuracy 80.5/81.0/81.8; guard r3 0.257/0.216/0.192; own r3 0.347/0.316/0.410) are primary;
live recomputed values are stored and a verdict computed with them is reported as a sensitivity. Q1u follows the registered
C0 convention (a None mode counts; under the plain prompt a boxed 'unknown' is scored by the last number); for every TG run a
descriptive q1u_numeric (numeric mode only), the plain-prompt abstain share and the None-mode share are added, and T-P1/T-P2 are
flagged 'definition-sensitive' if their branch changes under q1u_numeric. Accuracy requires the lenient key on all 3 sets.
python3 a5_reads2.py"""
import glob, json, os, random, re
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB"; UM = f"{R}/dirA4/results/umwp"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = sorted({(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}); UNSS = set(UNS)
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]
Q = lambda n: f"qwen3_4b_Base_{n}"


def rows(tag, run):
    fs = glob.glob(f"{B}/results/hp/{tag}_{run}_shard*of*.jsonl")
    sets = {}
    for f in fs:
        m = re.search(r"_shard(\d+)of(\d+)\.jsonl$", f)
        if m: sets.setdefault(int(m.group(2)), set()).add(int(m.group(1)))
    if len(sets) != 1: return None
    n, ks = next(iter(sets.items()))
    if ks != set(range(n)): return None
    d = {}
    for k in range(n):
        for l in open(f"{B}/results/hp/{tag}_{run}_shard{k}of{n}.jsonl"):
            try: r = json.loads(l)
            except ValueError: continue
            d[(str(r["fam"]), int(r["task"]), r["form"])] = r
    return list(d.values()) if len(d) == 1086 else None


def conf(run):
    d = rows("hp", run)
    if not d: return None
    u = {(str(r["fam"]), int(r["task"])): r["p_mode"] >= 0.5 for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNSS}
    return u if len(u) == 412 else None


def q1u(run):
    c = conf(run); return None if c is None else sum(c.values()) / 412


def q1u_desc(run):
    d = rows("hp", run)
    if not d: return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNSS]
    if len(u) != 412: return None
    return {"q1u_numeric": sum(r.get("mode") is not None and r["p_mode"] >= 0.5 for r in u) / 412,
            "none_mode_confident": sum(r.get("mode") is None and r["p_mode"] >= 0.5 for r in u) / 412,
            "plain_abstain_mean": sum(r.get("abstain", 0) for r in u) / 412}


def intact_abs(run):
    d = rows("hpabs", run)
    if not d: return None
    i = [r for r in d if r["form"] == "intact"]
    return sum(r.get("abstain", 0) for r in i) / len(i) if len(i) == 543 else None


def acc(run):
    f = f"{B}/results/evals/{run}_step60.summary.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); v = [d[k]["greedy_acc_lenient"] for k in DS if k in d and "greedy_acc_lenient" in d[k]]
    return 100 * sum(v) / 3 if len(v) == 3 else None


def umwp_ca(tag):
    f = f"{UM}/plain_{tag}.jsonl"
    if not os.path.exists(f): return None
    rr = {}
    for l in open(f):
        try: r = json.loads(l); rr[r["id"]] = r
        except ValueError: pass
    if len(rr) < 1200: return None
    U = [r for r in rr.values() if not r["answerable"]]
    return sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in U) / len(U)


def paired_boot(pairs, B_=4000, seed=20260924):
    """pairs: list over seeds of (conf_lo, conf_hi) dicts; tests pooled mean(hi) - mean(lo) > 0.
    Returns (diff, p_item_clustered [primary], p_within_seed [secondary])."""
    keys = UNS
    def pooled(idx_per_seed):
        tot = n = 0
        for (lo, hi), idx in zip(pairs, idx_per_seed):
            for j in idx: tot += hi[keys[j]] - lo[keys[j]]; n += 1
        return tot / n
    full = pooled([range(412)] * len(pairs))
    rng = random.Random(seed); le_c = 0
    for _ in range(B_):
        one = [rng.randrange(412) for _ in range(412)]
        le_c += pooled([one] * len(pairs)) <= 0
    rng = random.Random(seed); le_w = 0
    for _ in range(B_):
        le_w += pooled([[rng.randrange(412) for _ in range(412)] for _ in pairs]) <= 0
    return full, (1 + le_c) / (B_ + 1), (1 + le_w) / (B_ + 1)


def kofn(lst, k, n):
    h = sum(x is True for x in lst); unread = n - sum(x is not None for x in lst)
    return True if h >= k else (False if h + unread < k else None)


def unread(lst): return [i for i, x in enumerate(lst) if x is None]


res = {}
# ---- TG
tg = {"seeds": {}}; p1, alt, p2, p3, p4, kills = [], [], [], [], [], []
for s in range(3):
    lu, a4, t4, st = Q(f"loop_uv_a4_s{s}_r4"), Q(f"tg_abs4_a5_s{s}_r4"), Q(f"tg_t4_a5_s{s}_r4"), Q(f"rescue_uv_a4_s{s}_r3")
    v = {k: f(r) for k, (f, r) in {"q_loop": (q1u, lu), "q_abs4": (q1u, a4), "q_t4": (q1u, t4), "acc_loop": (acc, lu), "acc_t4": (acc, t4),
                                    "acc_start": (acc, st), "int_t4": (intact_abs, t4), "ca_abs4": (umwp_ca, a4), "ca_t4": (umwp_ca, t4)}.items()}
    v["drop_vs_start_descriptive"] = None if v["acc_t4"] is None or v["acc_start"] is None else v["acc_start"] - v["acc_t4"]
    tg["seeds"][s] = v
    p1.append(None if v["q_loop"] is None or v["q_abs4"] is None else v["q_loop"] <= v["q_abs4"] - 0.05 + 1e-9)
    alt.append(None if v["q_loop"] is None or v["q_abs4"] is None else abs(v["q_loop"] - v["q_abs4"]) <= 0.02 + 1e-9)
    p2.append(None if None in (v["q_t4"], v["q_abs4"], v["q_loop"]) else (v["q_t4"] <= v["q_abs4"] - 0.05 + 1e-9 and v["q_t4"] <= v["q_loop"] + 1e-9))
    p3.append(None if None in (v["acc_t4"], v["acc_loop"], v["int_t4"]) else (v["acc_t4"] >= v["acc_loop"] - 2 - 1e-9 and v["int_t4"] <= 0.15 + 1e-9))
    p4.append(None if v["ca_t4"] is None or v["ca_abs4"] is None else v["ca_t4"] <= v["ca_abs4"] - 0.05 + 1e-9)
    fired = [n for n, c in (("intact_abstention", v["int_t4"] is not None and v["int_t4"] > 0.30),
                            ("drop_vs_loop_uv_r4", v["acc_t4"] is not None and v["acc_loop"] is not None and v["acc_t4"] < v["acc_loop"] - 5),
                            ("drop_vs_start", v["acc_t4"] is not None and v["acc_start"] is not None and v["acc_t4"] < v["acc_start"] - 5)) if c]
    v["kill_fired"] = fired
    kills.append(True if fired else (None if None in (v["int_t4"], v["acc_t4"], v["acc_loop"], v["acc_start"]) else False))
    for k, r in (("loop", lu), ("abs4", a4), ("t4", t4)): v[f"desc_{k}"] = q1u_desc(r)
kill = True if any(x is True for x in kills) else (None if any(x is None for x in kills) else False)
t_p1, t_alt = kofn(p1, 2, 3), kofn(alt, 2, 3)
ALT = "alternative: the unknown reward adds nothing beyond the abstention prompt and start at round 4"
tp1 = "holds" if t_p1 else (ALT if t_alt else ("inconclusive" if t_p1 is False and t_alt is False else None))
tp2, tp3 = kofn(p2, 2, 3), kofn(p3, 3, 3)
def qn(s, k):
    x = tg["seeds"][s][f"desc_{k}"]; return None if x is None else x["q1u_numeric"]
p1n = [None if None in (qn(s, "loop"), qn(s, "abs4")) else qn(s, "loop") <= qn(s, "abs4") - 0.05 + 1e-9 for s in range(3)]
altn = [None if None in (qn(s, "loop"), qn(s, "abs4")) else abs(qn(s, "loop") - qn(s, "abs4")) <= 0.02 + 1e-9 for s in range(3)]
p2n = [None if None in (qn(s, "t4"), qn(s, "abs4"), qn(s, "loop")) else (qn(s, "t4") <= qn(s, "abs4") - 0.05 + 1e-9 and qn(s, "t4") <= qn(s, "loop") + 1e-9) for s in range(3)]
tp1n = "holds" if kofn(p1n, 2, 3) else (ALT if kofn(altn, 2, 3) else ("inconclusive" if kofn(p1n, 2, 3) is False and kofn(altn, 2, 3) is False else None))
tp2n = kofn(p2n, 2, 3)
tg["verdicts"] = {"T-P1": tp1, "T-P2": tp2, "T-P3": tp3, "T-P4": kofn(p4, 2, 3), "kill_gate_triggered": kill,
                  "T5_gate": (tp2 is True and tp3 is True and kill is False) if None not in (tp2, tp3, kill) else None,
                  "unread_seeds": {"P1": unread(p1), "P2": unread(p2), "P3": unread(p3), "P4": unread(p4)},
                  "numeric_q1u_sensitivity": {"T-P1": tp1n, "T-P2": tp2n,
                                              "definition_sensitive": {"T-P1": None if None in (tp1, tp1n) else tp1 != tp1n,
                                                                       "T-P2": None if None in (tp2, tp2n) else tp2 != tp2n}}}
res["TG"] = tg
# ---- HZN
OWN_R5 = {0: 0.478, 1: 0.345, 2: 0.571}; G_R3 = {0: 0.257, 1: 0.216, 2: 0.192}; OWN_R5_ACC = {0: 80.5, 1: 81.0, 2: 81.8}   # as printed in the registration
def uns(run):
    f = f"{B}/runs/{run}/audit_admitted.json"
    if not os.path.exists(f): return None
    c = json.load(open(f))["counts"]
    if sum(c.values()) != 200 or not set(c) <= {"SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"}: return None
    return 100 * c.get("UNSOLVABLE", 0) / 200
hz = {"seeds": {}, "constant_check": {}}; h1, h2, h3, h4, h5 = [], [], [], [], []
for s in range(3):
    g5, o5 = Q(f"own_ag_a4_s{s}_r5"), Q(f"own_a3_s{s}_r5")
    q = q1u(g5); a = acc(g5); cg, co = umwp_ca(g5), umwp_ca(o5)
    qo5, ao5, qg3 = q1u(o5), acc(o5), q1u(Q(f"own_ag_a4_s{s}_r3"))
    hz["constant_check"][s] = {"own_r5_q1u_recomputed": qo5, "registered": OWN_R5[s], "own_r5_acc_recomputed": ao5, "registered_acc": OWN_R5_ACC[s],
                               "guard_r3_q1u_recomputed": qg3, "registered_guard_r3": G_R3[s],
                               "flag_q": None if qo5 is None else abs(qo5 - OWN_R5[s]) > 0.0005,
                               "flag_acc": None if ao5 is None else abs(ao5 - OWN_R5_ACC[s]) > 0.05,
                               "flag_g3": None if qg3 is None else abs(qg3 - G_R3[s]) > 0.0005}
    u5, u3 = uns(g5), uns(Q(f"own_ag_a4_s{s}_r3"))
    hz["seeds"][s] = {"q_guard_r5": q, "acc_guard_r5": a, "ca_guard_r5": cg, "ca_own_r5": co, "uns_r5": u5, "uns_r3": u3}
    h1.append(None if q is None else q <= OWN_R5[s] - 0.10 + 1e-9)
    h2.append(None if q is None else q <= G_R3[s] + 0.05 + 1e-9)
    h3.append(None if a is None else a >= OWN_R5_ACC[s] - 2.5 - 1e-9)
    h4.append(None if cg is None or co is None else cg <= co - 0.05 + 1e-9)
    h5.append(None if u5 is None or u3 is None else u5 >= u3 - 5 - 1e-9)
# sensitivity (descriptive): H-P1 / H-P3 with the recomputed own r5 values
h1r = [None if hz["seeds"][s]["q_guard_r5"] is None or hz["constant_check"][s]["own_r5_q1u_recomputed"] is None else
       hz["seeds"][s]["q_guard_r5"] <= hz["constant_check"][s]["own_r5_q1u_recomputed"] - 0.10 + 1e-9 for s in range(3)]
h3r = [None if hz["seeds"][s]["acc_guard_r5"] is None or hz["constant_check"][s]["own_r5_acc_recomputed"] is None else
       hz["seeds"][s]["acc_guard_r5"] >= hz["constant_check"][s]["own_r5_acc_recomputed"] - 2.5 - 1e-9 for s in range(3)]
h2r = [None if hz["seeds"][s]["q_guard_r5"] is None or hz["constant_check"][s]["guard_r3_q1u_recomputed"] is None else
       hz["seeds"][s]["q_guard_r5"] <= hz["constant_check"][s]["guard_r3_q1u_recomputed"] + 0.05 + 1e-9 for s in range(3)]
hz["verdicts"] = {"H-P1": kofn(h1, 2, 3), "H-P2": kofn(h2, 2, 3), "H-P3": kofn(h3, 3, 3), "H-P4": kofn(h4, 2, 3), "H-P5": kofn(h5, 2, 3),
                  "sensitivity_recomputed_constants": {"H-P1": kofn(h1r, 2, 3), "H-P2": kofn(h2r, 2, 3), "H-P3": kofn(h3r, 3, 3)},
                  "unread_seeds": {"P1": unread(h1), "P3": unread(h3), "P4": unread(h4), "P5": unread(h5)}}
res["HZN"] = hz
# ---- SWAP
sw = {"seeds": {}}; s1, s2, s2alt, s3, pairs = [], [], [], [], []
n_known = 0
for s in range(5):
    st_f = f"{B}/runs/swap_flag_a5_s{s}/swap_build_stats.json"
    st = json.load(open(st_f)) if os.path.exists(st_f) else None
    under = None if st is None else bool(st["underpowered"])
    n_known += st is not None
    g1, c1, F = Q(f"swap_g1_a5_s{s}_r3"), Q(f"swap_c1_a5_s{s}_r3"), Q(f"own_a3_s{s}_r3")
    cg, cc, cf = conf(g1), conf(c1), conf(F)
    qg, qc, qf = [None if x is None else sum(x.values()) / 412 for x in (cg, cc, cf)]
    ag, ac, af = acc(g1), acc(c1), acc(F)
    sw["seeds"][s] = {"underpowered": under, "flagged_share": None if st is None else st["flagged_share"], "q_g1": qg, "q_c1": qc, "q_F": qf, "acc": [ag, ac, af]}
    if under is not False: continue
    s1.append(None if qg is None or qf is None else qg <= qf - 0.04 + 1e-9)
    s2.append(None if qg is None or qc is None else qg <= qc - 0.03 + 1e-9)
    s2alt.append(None if qg is None or qc is None else abs(qg - qc) <= 0.02 + 1e-9)
    s3.append(None if None in (ag, ac, af) else (abs(ag - af) <= 2.5 + 1e-9 and abs(ac - af) <= 2.5 + 1e-9))
    if cg is not None and cc is not None: pairs.append((cg, cc))
n_ok = len(s1)
p = None
if n_ok >= 3 and len(pairs) == n_ok:
    d, p, pw = paired_boot(pairs); sw["pooled_bootstrap_c1_minus_g1"] = {"diff": d, "p_one_sided_item_clustered": p, "p_one_sided_within_seed_secondary": pw}
if n_known == 5 and n_ok < 3:
    sw["verdicts"] = {"seeds_powered": n_ok, "S-P1": "underpowered: fewer than 3 seeds with >= 10% flagged", "S-P2": "underpowered", "S-P3": "underpowered"}
elif n_known < 5:
    sw["verdicts"] = {"seeds_powered": n_ok, "S-P1": None, "S-P2": None, "S-P3": None, "note": "swap_build_stats missing for some seeds"}
else:
    k2, k2a = kofn(s2, 3, n_ok), kofn(s2alt, 3, n_ok)
    sp2 = None
    if k2 is True and p is not None: sp2 = "holds" if p < 0.05 else None
    if sp2 is None and (k2 is not None or k2a is not None):
        if k2a is True: sp2 = "alternative: composition account holds"
        elif (k2 is False or (k2 is True and p is not None and p >= 0.05)) and k2a is False: sp2 = "inconclusive"
    sw["verdicts"] = {"seeds_powered": n_ok, "S-P1": kofn(s1, 3, n_ok), "S-P2": sp2, "S-P3": kofn(s3, min(4, n_ok), n_ok),
                      "unread": {"S1": unread(s1), "S2": unread(s2), "S3": unread(s3)}}
res["SWAP"] = sw
# ---- CTRL-L
OWN_R3 = {0: 0.347, 1: 0.316, 2: 0.410}
ct = {"seeds": {}}; c1l, c2l, c3l, c4l, c5l, cp1 = [], [], [], [], [], []
jac, jac_desc = [], []
for s in range(3):
    gd, rn, lp, ow = Q(f"own_ag_a4_s{s}_r3"), Q(f"ctrl_rnd_a5_s{s}_r3"), Q(f"ctrl_lowp_a5_s{s}_r3"), Q(f"own_a3_s{s}_r3")
    cgd, crn, clp = conf(gd), conf(rn), conf(lp)
    qg, qr, ql = [None if x is None else sum(x.values()) / 412 for x in (cgd, crn, clp)]
    cag, car = umwp_ca(gd), umwp_ca(rn)
    ar, al, ao = acc(rn), acc(lp), acc(ow)
    L1, G1 = f"{B}/runs/{Q(f'ctrl_lowp_a5_s{s}_r1')}", f"{B}/runs/{Q(f'own_ag_a4_s{s}_r1')}"
    jv = jd = None; miss = None
    if all(os.path.exists(x) for x in (f"{L1}/admitted_band.jsonl", f"{L1}/admitted.jsonl", f"{G1}/guard_records.jsonl", f"{G1}/admitted.jsonl")):
        band = {json.loads(l)["id"] for l in open(f"{L1}/admitted_band.jsonl")}
        l_adm = {json.loads(l)["id"] for l in open(f"{L1}/admitted.jsonl")}
        g_adm = {json.loads(l)["id"] for l in open(f"{G1}/admitted.jsonl")}
        grec = {r["id"]: r for r in map(json.loads, open(f"{G1}/guard_records.jsonl"))}
        miss = len([i for i in band if i not in grec])
        J = lambda a, b: len(a & b) / len(a | b) if (a | b) else None
        U = {i for i in band if i in grec and 0.25 - 1e-9 <= grec[i]["p_mode"] <= 0.75 + 1e-9}
        jv = J(U - g_adm, U - l_adm)
        lr = band - l_adm
        jd = {"n_band": len(band), "n_common_universe": len(U), "band_ids_missing_in_guard_records": miss,
              "band_difference_jaccard": J(band - g_adm, lr),
              "abstain_rule_on_band_jaccard": J({i for i in band if i in grec and grec[i]["abstain"] >= 0.25 - 1e-9}, lr),
              "J_admitted": J(g_adm, l_adm),
              "guard_admit_field_vs_file_mismatch": len({i for i, r in grec.items() if r.get("admit")} ^ g_adm)}
    jac.append(jv); jac_desc.append(jd)
    ct["seeds"][s] = {"q_guard": qg, "q_rnd": qr, "q_lowp": ql, "ca_guard": cag, "ca_rnd": car, "acc_rnd": ar, "acc_lowp": al, "acc_own": ao,
                      "jaccard_r1": jv, "jaccard_descriptive": jd}
    c1l.append(None if qg is None or qr is None else qg <= qr - 0.05 + 1e-9)
    c2l.append(None if cag is None or car is None else cag <= car - 0.05 + 1e-9)
    c3l.append(None if qg is None or ql is None else qg <= ql - 0.05 + 1e-9)
    c4l.append(None if qr is None else abs(qr - OWN_R3[s]) <= 0.05 + 1e-9)
    c5l.append(None if None in (ar, al, ao) else (abs(ar - ao) <= 2 + 1e-9 and abs(al - ao) <= 2 + 1e-9))
    if cgd is not None and crn is not None: cp1.append((cgd, crn))
p = None
if len(cp1) == 3:
    d, p, pw = paired_boot(cp1); ct["pooled_bootstrap_rnd_minus_guard"] = {"diff": d, "p_one_sided_item_clustered": p, "p_one_sided_within_seed_secondary": pw}
jac_ok = [x for x in jac if x is not None]
k1 = kofn(c1l, 2, 3)
cp1v = None if p is None or k1 is None else (k1 and p < 0.01)
if k1 is False: cp1v = False
cp3 = None if len(jac_ok) < 3 else ("uninformative (Jaccard > 0.6)" if sum(jac_ok) / 3 > 0.6 else kofn(c3l, 2, 3))
ct["verdicts"] = {"C-P1": cp1v, "C-P2": kofn(c2l, 2, 3), "C-P3": cp3, "C-P3_raw": kofn(c3l, 2, 3), "C-P4": kofn(c4l, 2, 3), "C-P5": kofn(c5l, 3, 3),
                  "jaccard_mean": sum(jac_ok) / 3 if len(jac_ok) == 3 else None,
                  "unread_seeds": {"C1": unread(c1l), "C2": unread(c2l), "C3": unread(c3l), "C4": unread(c4l), "C5": unread(c5l)}}
res["CTRL"] = ct
for k in ("TG", "HZN", "SWAP", "CTRL"): print(k, json.dumps(res[k]["verdicts"]))
json.dump(res, open(f"{R}/dirA5/results/a5_reads2.json", "w"), indent=1)
