# rev6 rerun copy of paper_arr/analyses/convergence.py (UMWP dir -> dumps_strict, 1 occurrence(s); output -> reruns_strict)
"""EXPLORATORY (post hoc) analysis "convergence": cross-seed convergence of the accommodated answer on UMWP.

Inputs (read-only):
  dirA4/results/umwp/plain_qwen3_4b_Base_own_a3_s{0,1,2}_r5.jsonl  (own loop, round 5, seeds 0-2)
  dirA4/results/umwp/plain_qwen3_4b_Base_own_a3_s{0,1,2}_r3.jsonl  (round-3 reference, same seeds)
  dirA4/results/umwp/plain_qwen3_4b_base.jsonl                     (Qwen3-4B-Base)
  dirA5/results/umwp_category_map.json                             (category id -> name)
  data_ext/UMWP/data/StandardDataset.jsonl                         (relevant_ids: unanswerable -> answerable original)
Definitions:
  confident(row) = mode is not None (numeric) and p_mode >= 0.5 (same criterion as UMWP CA).
  S = questions answered confidently by all three seeds of a round.
  identical = the three seed modes are equal.
  base baseline (per question in S, from the base model's 8 answers, numeric answers only), averaged over S:
    q3_plugin_num   = sum_a q0(a)^3, q0 normalised over the numeric answers (conditional on a numeric answer)
    q3_plugin_all8  = sum_a (c_a/8)^3 over numeric a (non-answers never count as coinciding)
    q3_unbiased_num = sum_a c_a(c_a-1)(c_a-2) / (n(n-1)(n-2)), n = number of numeric answers (draws without replacement)
    q2_plugin_num   = sum_a q0(a)^2 (reference for the mean pairwise agreement)
  identical_and_equal_base_mode = share of identical questions whose common mode equals the base model's mode
    (the dump's mode; Counter breaks ties by first occurrence, so a tie-aware version is also reported).
  identical_and_equal_answerable_original_ref = common mode on the unanswerable question equals the reference answer of
    its answerable UMWP original (relevant_ids[0]).
  split_base_confident / split_base_not_confident = the same identity share within S, split by base confidence.
  coupling diagnostic: all dumps were sampled with vLLM seed=17, n=8, so sample i of two models shares random numbers;
    aligned_agree = P(answer_i(A) == answer_i(B)), misaligned_agree = P(answer_i(A) == answer_j(B)), i != j, both numeric.
Output: paper_arr/analyses/convergence.json
"""
import json, math, itertools
from collections import Counter, defaultdict

R = "/home/tione/notebook/rsi_rl"
U = f"{R}/paper_arr/rev6/dumps_strict"
CAT = {int(k): v for k, v in json.load(open(f"{R}/dirA5/results/umwp_category_map.json")).items() if k.isdigit()}


def load(tag):
    rows = [json.loads(l) for l in open(f"{U}/plain_{tag}.jsonl")]
    d = {(r["id"], r["answerable"]): r for r in rows}
    assert len(d) == len(rows) == 1200, (tag, len(d), len(rows))
    assert all(len(r["answers"]) == 8 for r in rows)
    return d


def conf(r):
    return r["mode"] is not None and r["p_mode"] >= 0.5


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def mean(x):
    x = list(x)
    return round(sum(x) / len(x), 4) if x else None


def frac(k, n, ci=True):
    d = {"k": k, "n": n, "share": round(k / n, 4) if n else None}
    if ci:
        d["wilson95"] = wilson(k, n)
    return d


def base_coincide(answers):
    num = [a for a in answers if a is not None]
    c = Counter(num); n = len(num)
    p_num = sum((v / n) ** 3 for v in c.values()) if n else None
    p_all8 = sum((v / len(answers)) ** 3 for v in c.values())
    p_unb = sum(v * (v - 1) * (v - 2) for v in c.values()) / (n * (n - 1) * (n - 2)) if n >= 3 else None
    p2_num = sum((v / n) ** 2 for v in c.values()) if n else None
    return p_num, p_all8, p_unb, p2_num


def block(keys, seeds, base, ref=None, pair_rows=None):
    """keys: question keys of one class; seeds: 3 dicts; base: dict; ref: key -> answerable original's ref (U only)."""
    S = [k for k in keys if all(conf(s[k]) for s in seeds)]
    out = {"n_questions": len(keys), "n_all_three_confident": len(S)}
    if not S:
        return out
    ident = [k for k in S if len({s[k]["mode"] for s in seeds}) == 1]
    out["identical_modes"] = frac(len(ident), len(S))
    out["mean_pairwise_mode_agreement"] = mean(
        mean(int(a[k]["mode"] == b[k]["mode"]) for a, b in itertools.combinations(seeds, 2)) for k in S)
    bc = [base_coincide(base[k]["answers"]) for k in S]
    out["base_baseline"] = {
        "q3_plugin_num_mean": mean(b[0] for b in bc if b[0] is not None), "n_q3_plugin_num": sum(b[0] is not None for b in bc),
        "q3_plugin_all8_mean": mean(b[1] for b in bc),
        "q3_unbiased_num_mean": mean(b[2] for b in bc if b[2] is not None), "n_q3_unbiased_num": sum(b[2] is not None for b in bc),
        "q2_plugin_num_mean": mean(b[3] for b in bc if b[3] is not None),
        "base_confident_share_in_S": round(sum(conf(base[k]) for k in S) / len(S), 4),
        "base_mean_p_mode_in_S": mean(base[k]["p_mode"] for k in S),
    }
    if ident:
        cm = {k: seeds[0][k]["mode"] for k in ident}
        out["identical_and_equal_base_mode"] = frac(sum(base[k]["mode"] == cm[k] for k in ident), len(ident))
        out["identical_and_equal_base_mode_and_base_confident"] = frac(
            sum(base[k]["mode"] == cm[k] and conf(base[k]) for k in ident), len(ident), ci=False)
        out["identical_common_mode_among_base_8_answers"] = frac(sum(cm[k] in base[k]["answers"] for k in ident), len(ident), ci=False)
        out["base_q0_of_common_mode_mean"] = mean(sum(a == cm[k] for a in base[k]["answers"]) / 8 for k in ident)
        # tie-aware: common mode has the maximal numeric count among the base's 8 answers (Counter picks the first of tied modes)
        def is_base_top(k):
            c = Counter(a for a in base[k]["answers"] if a is not None)
            return bool(c) and c.get(cm[k], 0) == max(c.values())
        out["identical_common_mode_is_a_tied_top_numeric_base_answer"] = frac(sum(is_base_top(k) for k in ident), len(ident), ci=False)
    # split S by whether the base model itself was confident on the question
    for lab, sub in (("base_confident", [k for k in S if conf(base[k])]), ("base_not_confident", [k for k in S if not conf(base[k])])):
        if sub:
            idn = [k for k in sub if len({s[k]["mode"] for s in seeds}) == 1]
            out[f"split_{lab}"] = {"identical_modes": frac(len(idn), len(sub)),
                                   "q3_plugin_num_mean": mean(base_coincide(base[k]["answers"])[0] for k in sub if base_coincide(base[k]["answers"])[0] is not None),
                                   "identical_and_equal_base_mode": frac(sum(base[k]["mode"] == seeds[0][k]["mode"] for k in idn), len(idn), ci=False)}
    if ident:
        if ref is not None:
            has = [k for k in ident if ref.get(k) is not None]
            out["identical_and_equal_answerable_original_ref"] = frac(sum(ref[k] == cm[k] for k in has), len(has))
            out["identical_and_equal_all3_r_modes_on_answerable_original"] = frac(
                sum(all(pair_rows[i][k]["mode"] == cm[k] for i in range(3)) for k in has), len(has), ci=False)
            hasS = [k for k in S if ref.get(k) is not None]
            out["base_mode_equals_answerable_original_ref_in_S"] = frac(sum(base[k]["mode"] == ref[k] for k in hasS), len(hasS), ci=False)
    return out


def coupling(keys, a, b):
    al, mis = [], []
    for k in keys:
        x, y = a[k]["answers"], b[k]["answers"]
        for i in range(8):
            for j in range(8):
                if x[i] is None or y[j] is None:
                    continue
                (al if i == j else mis).append(int(x[i] == y[j]))
    return {"aligned_agree": mean(al), "n_aligned": len(al), "misaligned_agree": mean(mis), "n_misaligned": len(mis)}


base = load("qwen3_4b_base")
rounds = {r: [load(f"qwen3_4b_Base_own_a3_s{s}_{r}") for s in range(3)] for r in ("r5", "r3")}
keysU = sorted(k for k in base if not k[1]); keysA = sorted(k for k in base if k[1])
assert len(keysU) == 600 and len(keysA) == 600
for ss in rounds.values():
    for s in ss:
        assert set(s) == set(base)

# pair each unanswerable question with its answerable original (the probe used relevant_ids[0])
D = {x["id"]: x for x in (json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl"))}
u2a = {k: (D[k[0]]["relevant_ids"][0], True) for k in keysU}
assert all(v in base for v in u2a.values()) and len(set(u2a.values())) == 600
catU = {k: base[k]["category"] for k in keysU}
catA = {u2a[k]: catU[k] for k in keysU}
ref_of_pair = {k: base[u2a[k]]["ref"] for k in keysU}

res = {"analysis": "convergence", "status": "EXPLORATORY post hoc; not preregistered",
       "definitions": __doc__.split("Definitions:")[1].split("Output:")[0].strip(),
       "category_names": CAT,
       "base_sanity": {"CA": round(sum(conf(base[k]) for k in keysU) / 600, 4),
                       "ACC_A": round(sum(base[k]["correct"] for k in keysA) / 600, 4)},
       "rounds": {}}
for r, ss in rounds.items():
    pair_rows = [{k: s[u2a[k]] for k in keysU} for s in ss]
    entry = {"sanity_CA_per_seed": [round(sum(conf(s[k]) for k in keysU) / 600, 4) for s in ss],
             "sanity_ACC_A_per_seed": [round(sum(s[k]["correct"] for k in keysA) / 600, 4) for s in ss],
             "unanswerable": block(keysU, ss, base, ref_of_pair, pair_rows),
             "answerable": block(keysA, ss, base),
             "by_category": {}}
    SA = [k for k in keysA if all(conf(s[k]) for s in ss) and len({s[k]["mode"] for s in ss}) == 1]
    entry["answerable"]["identical_and_correct"] = frac(sum(ss[0][k]["mode"] == base[k]["ref"] for k in SA), len(SA), ci=False)
    for c in sorted(CAT):
        kU = [k for k in keysU if catU[k] == c]; kA = [k for k in keysA if catA[k] == c]
        entry["by_category"][f"{c}:{CAT[c]}"] = {"unanswerable": block(kU, ss, base, ref_of_pair, pair_rows),
                                                 "answerable_originals": block(kA, ss, base)}
    entry["coupling_diagnostic"] = {
        "U_seed_pairs": {f"s{i}-s{j}": coupling(keysU, ss[i], ss[j]) for i, j in itertools.combinations(range(3), 2)},
        "U_base_vs_seed": {f"base-s{i}": coupling(keysU, base, ss[i]) for i in range(3)},
        "A_seed_pairs": {f"s{i}-s{j}": coupling(keysA, ss[i], ss[j]) for i, j in itertools.combinations(range(3), 2)}}
    res["rounds"][r] = entry
json.dump(res, open(f"{R}/paper_arr/rev6/reruns_strict/convergence.json", "w"), indent=1)
print(json.dumps(res, indent=1))
