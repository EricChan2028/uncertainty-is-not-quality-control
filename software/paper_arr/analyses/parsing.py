"""EXPLORATORY (post hoc, not registered) parsing-robustness analysis for the paper.  Read-only on all inputs;
writes only paper_arr/analyses/parsing.json.   python3 parsing.py

(a) Fixed pool qwen_s0 (1,158 fully audited round-1 proposals of Qwen3-4B own s0 + 400 guard-seed-1 tracers), scored by the
    base model and own s0 r1..r5 (dirA3/results/guard_diag/qwen_s0/<ck>/guard_records.jsonl, R-Zero band [0.25, 0.75], G = 8).
    LIMITATION: a3_guard.py stores per item only p_mode = k/8 (k = count of the most frequent parsed, non-unknown answer;
    truncated / unparsable samples are None and stay in the denominator), the modal answer and the unknown share.  The
    number of None samples per item is NOT stored, so:
      * exact:   share of items with no voting answer (mode None, i.e. all 8 samples None/unknown), unknown share,
                 non-numeric modal answers, k histogram;
      * bounds:  per-sample None share per class, LB = only the None samples forced by mode-None items,
                 UB = every non-modal, non-unknown sample is None;
      * proxy:   per-sample None rate (1 - fmt_rate) and truncation rate of the SAME checkpoint under the SAME prompt,
                 parser, sampling and max_new, read from step 1 (exact checkpoint, 256 rollouts) and steps 1-10 of the
                 s0 training round that starts from it (dirB/runs/qwen3_4b_Base_own_a3_s0_r<n+1>/log.jsonl); these are
                 admitted-set items, not the fixed pool;
      * re-denominated band p2 = k / m, m = number of voting (parsed, non-unknown) samples, as an EXPECTATION under an
        imputation: each of the 8 - k - u non-modal, non-unknown samples is None independently with probability theta
        (u = unknown count).  theta scenarios: calibrated to the proxy None rate (uniform over classes), a uniform grid,
        and class-concentrated placements (same None volume, all on SOLVABLE or all on ill-posed items).  Ill-posed share of
        the admitted proposals = E[#UNSOLVABLE + #AMBIGUOUS] / E[#admitted].  Items with m = 0 are never admitted.
(b) UMWP (dirA4/results/umwp/plain_<tag>.jsonl; 600 unanswerable + 600 answerable, 8 samples; field answers stores the parsed
    number per sample, None = non-answer = truncated or no number): non-answer rate on answerable/unanswerable items, and
    CA under: registered (mode over all 8 with None as its own category, mode not None and p_mode >= 0.5);
    numeric/8 (numeric modal count / 8 >= 0.5: non-answers count as abstentions); parsed-only (numeric modal count / number
    of numeric answers >= 0.5, >= 1 numeric answer); parsed-only with >= 4 numeric answers.  Paired r3 - base differences with
    item-bootstrap 95% CIs (2,000 resamples, Random(0)).
"""
import glob
import json
import math
import random
import re
from collections import Counter

R = "/home/tione/notebook/rsi_rl"
OUT = f"{R}/paper_arr/analyses/parsing.json"
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
G, LO, HI = 8, 0.25, 0.75
EPS = 1e-9
IP = ("UNSOLVABLE", "AMBIGUOUS")
TH0 = "theta_0 (m = 8 - unknown; = registered unless unknown > 0)"


def rnd4(x, d=4):
    return None if x is None else round(x, d)


# ------------------------------------------------------------------ (a) fixed pool
verdict = {}
for f in glob.glob(f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts/*.json"):          # same loader as a3_guard_diag_score.py
    for v in json.load(open(f))["verdicts"]:
        verdict[v["id"]] = v["verdict"]


def cls(r):
    if r["kind"] != "proposed":
        return r["kind"]
    return verdict.get(r["id"], "UNJUDGED")


CLASSES = ["SOLVABLE", "AMBIGUOUS", "UNSOLVABLE", "tracer_intact", "tracer_deleted"]
NUMRE = re.compile(r"-?[\d.,]+(/\d+)?")

recs = {}
for ck in CKS:
    rows = [json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/qwen_s0/{ck}/guard_records.jsonl")]
    items = []
    for r in rows:
        k = int(round(r["p_mode"] * G)); u = int(round(r["abstain"] * G))
        items.append({"id": r["id"], "c": cls(r), "k": k, "u": u, "mode": r["mode"], "admit_rec": r["admit"]})
    recs[ck] = items

# proxy None/truncation rates from the training round that starts at each checkpoint
proxy = {}
for i, ck in enumerate(CKS):
    f = f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r{i + 1}/log.jsonl"
    st = [json.loads(l) for l in open(f)]
    st = [x for x in st if "trunc" in x and "fmt_rate" in x]
    s1 = [x for x in st if x["step"] == 1][0]
    s10 = [x for x in st if 1 <= x["step"] <= 10]
    proxy[ck] = {"source": f.replace(R + "/", ""), "step1_none_rate": rnd4(1 - s1["fmt_rate"]), "step1_trunc_rate": rnd4(s1["trunc"]),
                 "step1_n_rollouts": s1["n_seqs"], "steps1_10_none_rate": rnd4(sum(1 - x["fmt_rate"] for x in s10) / len(s10)),
                 "steps1_10_trunc_rate": rnd4(sum(x["trunc"] for x in s10) / len(s10)), "steps1_10_n_rollouts": sum(x["n_seqs"] for x in s10)}


def admit_prob(k, u, theta):
    """P(0.25 <= k/m <= 0.75) with m = 8 - u - j, j ~ Bin(8 - k - u, theta) None samples among the non-modal slots."""
    if k == 0:
        return 0.0
    nslot = G - k - u
    p = 0.0
    for j in range(nslot + 1):
        m = G - u - j
        if LO - EPS <= k / m <= HI + EPS:
            p += math.comb(nslot, j) * theta ** j * (1 - theta) ** (nslot - j)
    return p


def composition(items, theta_of):
    """Expected admitted proposals by class under the re-denominated band; theta_of(class) -> theta."""
    e = Counter()
    for it in items:
        if it["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"):
            e[it["c"]] += admit_prob(it["k"], it["u"], theta_of(it["c"]))
    n = sum(e.values())
    return {"n_admitted": rnd4(n, 2), "illposed_share": rnd4((e["UNSOLVABLE"] + e["AMBIGUOUS"]) / n if n else None),
            "unsolvable_share": rnd4(e["UNSOLVABLE"] / n if n else None), "by_class": {c: rnd4(e[c], 2) for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}}


def props(items):
    return [it for it in items if it["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")]


def none_volume(items):
    """A0 = None samples forced by mode-None items; slot counts of items with k >= 1 (by class group)."""
    a0 = sum(G - it["u"] for it in items if it["k"] == 0)
    sS = sum(G - it["k"] - it["u"] for it in items if it["k"] > 0 and it["c"] == "SOLVABLE")
    sI = sum(G - it["k"] - it["u"] for it in items if it["k"] > 0 and it["c"] in IP)
    return a0, sS, sI


A = {"descriptives": {}, "registered_band": {}, "proxy_none_rate": proxy, "redenominated": {}}
for ck in CKS:
    items = recs[ck]
    d = {}
    for c in CLASSES:
        its = [it for it in items if it["c"] == c]
        n = len(its)
        lb = sum(G - it["u"] for it in its if it["k"] == 0)
        ub = sum(G - it["k"] - it["u"] for it in its)
        d[c] = {"n": n, "no_voting_answer_items": sum(it["k"] == 0 for it in its),
                "no_voting_answer_share": rnd4(sum(it["k"] == 0 for it in its) / n),
                "nonnumeric_mode_items": sum(1 for it in its if it["mode"] is not None and not NUMRE.fullmatch(str(it["mode"]))),
                "unknown_sample_share": rnd4(sum(it["u"] for it in its) / (G * n)),
                "sample_none_share_LB": rnd4(lb / (G * n)), "sample_none_share_UB": rnd4(ub / (G * n)),
                "k_hist": [sum(it["k"] == j for it in its) for j in range(G + 1)],
                "in_band_registered": rnd4(sum(LO <= it["k"] / G <= HI for it in its) / n)}
    A["descriptives"][ck] = d
    P = props(items)
    adm = [it for it in P if it["admit_rec"]]
    chk = [it for it in P if LO <= it["k"] / G <= HI]
    A["registered_band"][ck] = {"n_admitted": len(adm), "illposed_share": rnd4(sum(it["c"] in IP for it in adm) / len(adm)),
                                "unsolvable_share": rnd4(sum(it["c"] == "UNSOLVABLE" for it in adm) / len(adm)),
                                "recomputed_from_k_matches_admit_field": len(chk) == len(adm) and {x["id"] for x in chk} == {x["id"] for x in adm}}
    a0, sS, sI = none_volume(P)
    N8 = G * len(P)
    sc = {TH0: composition(items, lambda c: 0.0)}
    for key in ("step1_none_rate", "steps1_10_none_rate"):
        q = proxy[ck][key]
        th = min(1.0, max(0.0, (q * N8 - a0) / (sS + sI)))
        x_ = composition(items, lambda c, th=th: th)
        x_["theta"] = rnd4(th); x_["implied_sample_none_rate"] = rnd4((a0 + th * (sS + sI)) / N8)
        sc[f"calibrated_uniform[{key}]"] = x_
        thS = min(1.0, max(0.0, (q * N8 - a0) / sS)); thI = min(1.0, max(0.0, (q * N8 - a0) / sI))
        x_ = composition(items, lambda c, t=thS: t if c == "SOLVABLE" else 0.0); x_["theta_S"] = rnd4(thS); x_["theta_IP"] = 0.0
        sc[f"calibrated_all_None_on_SOLVABLE[{key}]"] = x_
        x_ = composition(items, lambda c, t=thI: t if c in IP else 0.0); x_["theta_S"] = 0.0; x_["theta_IP"] = rnd4(thI)
        sc[f"calibrated_all_None_on_illposed[{key}]"] = x_
    for th in (0.10, 0.25, 0.50):
        x_ = composition(items, lambda c, th=th: th); x_["theta"] = th; x_["implied_sample_none_rate"] = rnd4((a0 + th * (sS + sI)) / N8)
        sc[f"uniform_theta_{th}"] = x_
    x_ = composition(items, lambda c: 0.5 if c == "SOLVABLE" else 0.0); x_["theta_S"] = 0.5; x_["theta_IP"] = 0.0
    sc["skew_theta_0.5_on_SOLVABLE_only"] = x_
    x_ = composition(items, lambda c: 0.5 if c in IP else 0.0); x_["theta_S"] = 0.0; x_["theta_IP"] = 0.5
    sc["skew_theta_0.5_on_illposed_only"] = x_
    A["redenominated"][ck] = sc

drift = {}
for key in A["redenominated"]["base"]:
    seq = [A["redenominated"][ck][key]["illposed_share"] for ck in CKS]
    drift[key] = {"illposed_share_base_to_r5": seq, "r5_minus_base_pts": rnd4(100 * (seq[-1] - seq[0]), 2),
                  "monotone_nondecreasing": all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))}
seq = [A["registered_band"][ck]["illposed_share"] for ck in CKS]
drift["registered_band"] = {"illposed_share_base_to_r5": seq, "r5_minus_base_pts": rnd4(100 * (seq[-1] - seq[0]), 2),
                            "monotone_nondecreasing": all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))}
adv = {}
for tag, kS, kI in (("calibrated_step1", "calibrated_all_None_on_SOLVABLE[step1_none_rate]", "calibrated_all_None_on_illposed[step1_none_rate]"),
                    ("calibrated_steps1_10", "calibrated_all_None_on_SOLVABLE[steps1_10_none_rate]", "calibrated_all_None_on_illposed[steps1_10_none_rate]"),
                    ("theta_0.5", "skew_theta_0.5_on_SOLVABLE_only", "skew_theta_0.5_on_illposed_only")):
    b = max(A["redenominated"]["base"][k]["illposed_share"] for k in (kS, kI, TH0))
    e = min(A["redenominated"]["r5"][k]["illposed_share"] for k in (kS, kI, TH0))
    adv[tag] = {"max_base_illposed": b, "min_r5_illposed": e, "worst_case_drift_pts": rnd4(100 * (e - b), 2)}
# non-numeric modal answers (boxed text / expressions vote as answers under the registered parser): drop them from admission
TEXTABS = re.compile(r"insufficient|cannot|can'?not|notenough|undetermin|notdetermin|notpossible|notprovided|unknown|noanswer|notgiven|missing", re.I)
nn = {}
for ck in CKS:
    P = props(recs[ck])
    adm = [it for it in P if it["admit_rec"]]
    keep = [it for it in adm if it["mode"] is not None and NUMRE.fullmatch(str(it["mode"]))]
    nn[ck] = {"n_admitted_nonnumeric_mode": len(adm) - len(keep), "n_admitted_numeric_only": len(keep),
              "illposed_share_numeric_only": rnd4(sum(it["c"] in IP for it in keep) / len(keep)),
              "textual_abstention_modes_all_proposals": sum(1 for it in P if it["mode"] is not None and TEXTABS.search(str(it["mode"]))),
              "textual_abstention_modes_tracer_deleted": sum(1 for it in recs[ck] if it["c"] == "tracer_deleted" and it["mode"] is not None and TEXTABS.search(str(it["mode"])))}
seq = [nn[ck]["illposed_share_numeric_only"] for ck in CKS]
drift["numeric_mode_only_admission"] = {"illposed_share_base_to_r5": seq, "r5_minus_base_pts": rnd4(100 * (seq[-1] - seq[0]), 2),
                                        "monotone_nondecreasing": all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))}
A["nonnumeric_mode_sensitivity"] = nn
A["drift_summary"] = drift
A["adversarial_placement_drift"] = adv
A["pool"] = {"n_proposals": len(props(recs["base"])), "audit_counts": dict(Counter(verdict.values())), "n_tracers": 400}

# ------------------------------------------------------------------ (b) UMWP
UD = f"{R}/dirA4/results/umwp"
TAGS = {"base": "qwen3_4b_base"}
for s in range(5):
    TAGS[f"own_s{s}_r3"] = f"qwen3_4b_Base_own_a3_s{s}_r3"
SUPP = {f"own_s{s}_r5": f"qwen3_4b_Base_own_a3_s{s}_r5" for s in range(3)}


def load(tag):
    rows = {}
    for l in open(f"{UD}/plain_{tag}.jsonl"):
        r = json.loads(l); rows[r["id"]] = r
    return rows


def feats(r):
    ans = r["answers"]
    nums = [a for a in ans if a is not None]
    cnt = Counter(nums)
    mnum, c1 = cnt.most_common(1)[0] if cnt else (None, 0)
    return {"n_num": len(nums), "c1": c1, "mnum": mnum,
            "reg": r["mode"] is not None and r["p_mode"] >= 0.5,
            "num8": c1 / G >= 0.5,
            "parsed": len(nums) >= 1 and c1 / len(nums) >= 0.5,
            "parsed_min4": len(nums) >= 4 and c1 / len(nums) >= 0.5}


VARS = ("reg", "num8", "parsed", "parsed_min4")


def umwp_metrics(rows):
    U = [r for r in rows.values() if not r["answerable"]]; S = [r for r in rows.values() if r["answerable"]]
    fu = {r["id"]: feats(r) for r in U}; fs = {r["id"]: feats(r) for r in S}
    consist = all(abs(r["nonanswer"] - (G - len([a for a in r["answers"] if a is not None])) / G) < 1e-9 for r in rows.values())
    m = {"n_unanswerable": len(U), "n_answerable": len(S), "nonanswer_field_consistent_with_answers": consist}
    for nm, R_, F_ in (("U", U, fu), ("A", S, fs)):
        m[f"nonanswer_rate_{nm}"] = rnd4(sum(r["nonanswer"] for r in R_) / len(R_))
        m[f"items_any_nonanswer_{nm}"] = rnd4(sum(r["nonanswer"] > 0 for r in R_) / len(R_))
        m[f"items_all_nonanswer_{nm}"] = rnd4(sum(r["nonanswer"] >= 1 - 1e-9 for r in R_) / len(R_))
        m[f"abstain_text_rate_{nm}"] = rnd4(sum(r["abstain"] for r in R_) / len(R_))
        for v in VARS:
            key = f"CA_{v}" if nm == "U" else f"conf_A_{v}"
            m[key] = rnd4(sum(F_[r["id"]][v] for r in R_) / len(R_))
    m["CA_reg_vs_num8_disagree_items"] = sum(fu[i]["reg"] != fu[i]["num8"] for i in fu)
    m["CA_reg_vs_parsed_disagree_items"] = sum(fu[i]["reg"] != fu[i]["parsed"] for i in fu)
    m["ACC_A_registered_pct"] = rnd4(100 * sum(r["correct"] for r in S) / len(S), 2)
    m["ACC_A_numeric_mode_pct"] = rnd4(100 * sum(fs[r["id"]]["mnum"] is not None and r["ref"] is not None and fs[r["id"]]["mnum"] == r["ref"] for r in S) / len(S), 2)
    m["n_answerable_ref_none"] = sum(r["ref"] is None for r in S)
    return m, fu


def boot_diff(fb, fo, v, nb=2000, seed=0):
    ids = sorted(set(fb) & set(fo))
    dd = [int(fo[i][v]) - int(fb[i][v]) for i in ids]
    rng = random.Random(seed)
    bs = sorted(sum(dd[rng.randrange(len(dd))] for _ in dd) / len(dd) for _ in range(nb))
    return {"n_paired": len(ids), "diff": rnd4(sum(dd) / len(dd)), "ci95": [rnd4(bs[int(0.025 * nb)]), rnd4(bs[int(0.975 * nb) - 1])]}


B = {"metrics": {}, "paired_r3_minus_base": {}, "supplementary_r5": {}}
F = {}
for k, t in TAGS.items():
    B["metrics"][k], F[k] = umwp_metrics(load(t))
for k, t in SUPP.items():
    B["supplementary_r5"][k], F[k] = umwp_metrics(load(t))
for k in TAGS:
    if k != "base":
        B["paired_r3_minus_base"][k] = {v: boot_diff(F["base"], F[k], v) for v in VARS}
B["direction_r3_gt_base_seeds"] = {v: sum(B["metrics"][k][f"CA_{v}"] > B["metrics"]["base"][f"CA_{v}"] for k in TAGS if k != "base") for v in VARS}
B["ci_excludes_zero_seeds"] = {v: sum(B["paired_r3_minus_base"][k][v]["ci95"][0] > 0 for k in B["paired_r3_minus_base"]) for v in VARS}
rng_ = {}
for v in VARS:
    vals = [B["metrics"][k][f"CA_{v}"] for k in TAGS if k != "base"]
    rng_[v] = {"base": B["metrics"]["base"][f"CA_{v}"], "r3_min": min(vals), "r3_max": max(vals)}
B["CA_ranges"] = rng_

out = {"analysis": "parsing", "status": "EXPLORATORY (post hoc, not registered)", "script": f"{R}/paper_arr/analyses/parsing.py",
       "a_fixed_pool_qwen_s0": A, "b_umwp_qwen3_4b": B}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps({"registered": drift["registered_band"], "adv": adv,
                  "calib": {k: drift[k] for k in drift if k.startswith("calibrated_uniform")},
                  "B_ranges": rng_, "B_dir": B["direction_r3_gt_base_seeds"], "B_ci": B["ci_excludes_zero_seeds"]}, indent=1))
