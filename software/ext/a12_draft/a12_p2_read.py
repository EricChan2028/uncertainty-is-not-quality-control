"""A12 part-2 read (A12_registration_2026-09-26.md, section 2.4): benchmark accuracy of the R-Zero solvers.
Models: Qwen3-4B-Base (qwen3_4b_base) and the R-Zero solvers S_i of seeds 0 and 1 (rzero_s<S>_it<i>, i = 1, 2, 3). Conditions:
  plain  - b_eval.py as registered through a12_plain_eval.py (Base prompt format, greedy, --k 1, stop also at <|im_end|>;
           tags a12_plain_<model>): b_eval's scorer (greedy_ok / greedy_acc) and its lenient scorer (greedy_acc_lenient);
  rzchat - R-Zero's own solver prompt (a12_rzchat_eval.py; tags a12rz_<model>): b_eval's scorer (greedy_ok / greedy_acc) and
           the R-Zero scorer without its GPT-4o re-check (ok_rzero_style / greedy_acc_rzero_style).
Sets: math500_train (= MATH-500), math_heldout1000, gsm8k_test500. A cell counts if its summary holds the set with n rows and its
per-item file has n unique ids; missing cells are NOT RUN. Every summary must name the cell's own checkpoint as its model, every
rzchat summary the registered b_eval.py, and the rendered first prompt of each rzchat set must be identical across models
(else the read aborts).
PRIMARY cell: rzchat, iteration 3, math500_train, b_eval scorer. SECONDARY cell: plain, iteration 3, math500_train, b_eval scorer.
Per seed S: paired delta S_3 - base (same condition, same items) with a paired item bootstrap, 4,000 replicates, stream
random.Random(f"{20260926 + 14}:<comparison key>"); label GAIN if the 95% CI lower bound > 0, LOSS if its upper bound < 0,
else NO_GAIN (margin 0 points at the CI bound). The pre-stated reading maps (primary labels of the 2 seeds, secondary labels of
the 2 seeds) to one name (reading() below; NOT RUN if any of the four is missing):
  primary 2 GAIN, secondary 2 GAIN            -> REPRODUCES
  primary 2 GAIN, secondary 0-1 GAIN          -> REPRODUCES, FORMAT-DEPENDENT
  primary 1 GAIN (any secondary)              -> REPRODUCES IN ONE SEED
  primary 0 GAIN, secondary >= 1 GAIN         -> PLAIN-ONLY
  primary 0 GAIN, no LOSS, secondary 0 GAIN   -> NOT REPRODUCED
  primary 0 GAIN, >= 1 LOSS, secondary 0 GAIN -> NOT REPRODUCED, LOSS
FORMAT-DEPENDENT wording (reading_text below): the words "the gain depends on the format" only if the paired format contrast
delta(rzchat) - delta(plain) at iteration 3 on MATH-500 has its 95% CI lower bound > 0 in both seeds; a note that the reading is
not called format adoption if the plain condition truncates much more often, i.e. the plain greedy truncation rate exceeds the
rzchat truncation rate by more than 5 points for the base model or S_3 of either seed on MATH-500 (plain_truncates_more).
DESCRIPTIVE (no reading attached): every other (condition, set, iteration) with the same per-seed labels; paired deltas for the
R-Zero-style scorer; accuracies (strict, lenient, R-Zero-style); magnitude against the R-Zero paper (arXiv 2508.05004 v1 math
table: Qwen3-4B-Base MATH 68.20 -> 78.60 / 79.80 / 79.60, GSM8K 87.79 -> 91.28 / 91.66 / 92.12; v4 reports the final model only,
same MATH and GSM8K numbers): seed-mean delta / paper delta; the format contrast delta(rzchat) - delta(plain), paired; plain
diagnostics (greedy truncation rate, share of greedy outputs whose first and last boxed answers differ, <|im_end|> stops);
rzchat truncation and <|im_end|> rates; the base reproduction check (fresh plain base vs the stored qwen3_4b_base_step0
summary, sha-pinned; flagged if any set differs by more than 2 points; no other consequence).
    python3 a12_p2_read.py"""
import hashlib, json, os, random
R = "/home/tione/notebook/rsi_rl"; E = f"{R}/dirB/results/evals"; OUTF = f"{R}/ext/a12_runs/a12_p2_read.json"
SETS = {"math500_train": 500, "math_heldout1000": 1000, "gsm8k_test500": 500}
MODELS = ["qwen3_4b_base"] + [f"rzero_s{s}_it{i}" for s in (0, 1) for i in (1, 2, 3)]
PATH = {"qwen3_4b_base": "/root/models/Qwen3-4B-Base",
        **{f"rzero_s{s}_it{i}": f"/run/ti/rzero/s{s}/models/rzero_qwen3_4b_Base_s{s}_solver_v{i}/global_step_15/actor/huggingface"
           for s in (0, 1) for i in (1, 2, 3)}}
COND = {"plain": ("a12_plain_", ["greedy_acc", "greedy_acc_lenient"]), "rzchat": ("a12rz_", ["greedy_acc", "greedy_acc_rzero_style"])}
B_EVAL_SHA = "5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189"
OLD = f"{E}/qwen3_4b_base_step0.summary.json"; OLD_SHA = "975e7e9770ebcfc6c7d2835fdae8002e221bc9d8ff50caa55f2e28313612e178"
PRIMARY, SECONDARY = ("rzchat", 3, "math500_train"), ("plain", 3, "math500_train")
TOL_REPRO = 2.0
TRUNC_MARGIN = 5.0   # points: plain greedy truncation minus rzchat truncation above which the plain condition "truncates much more"
PAPER = {"source": "https://arxiv.org/abs/2508.05004 (v1 math table, Iter 1-3 rows; v4 Table 1, final row); ICLR 2026",
         "base": {"math500_train": 68.20, "gsm8k_test500": 87.79},
         "it": {1: {"math500_train": 78.60, "gsm8k_test500": 91.28}, 2: {"math500_train": 79.80, "gsm8k_test500": 91.66},
                3: {"math500_train": 79.60, "gsm8k_test500": 92.12}},
         "math_avg_7": {"base": 42.58, "it1": 48.06, "it2": 48.44, "it3": 49.07, "v4_base": 42.57, "v4_final": 49.93},
         "protocol": "greedy (mean@32 for AMC/AIME), chat template + boxed system prompt, max 4,096 tokens, GPT-4o answer re-check; "
                     "GSM8K full test (1,319) vs our 500-item subset"}


def seed_label(ci95):
    """Per-seed label, margin 0 points at the 95% CI bound."""
    return "GAIN" if ci95[0] > 0 else ("LOSS" if ci95[1] < 0 else "NO_GAIN")


def reading(P, S):
    """Registered part-2 reading from the per-seed labels [seed 0, seed 1] of the primary (P) and secondary (S) cells."""
    if P is None or S is None or None in P or None in S:
        return "NOT RUN"
    gP, lP, gS = P.count("GAIN"), P.count("LOSS"), S.count("GAIN")
    if gP == 2:
        return "REPRODUCES" if gS == 2 else "REPRODUCES, FORMAT-DEPENDENT"
    if gP == 1:
        return "REPRODUCES IN ONE SEED"
    if gS >= 1:
        return "PLAIN-ONLY"
    return "NOT REPRODUCED, LOSS" if lP >= 1 else "NOT REPRODUCED"


READING_TEXT = {
    "REPRODUCES": ("under R-Zero's own prompt the final solver beats the base model on MATH-500 in both seeds, and so it does "
                   "under the plain Base prompt: the replication reproduces the direction of R-Zero's reported gain (magnitude "
                   "ratio reported)"),
    "REPRODUCES, FORMAT-DEPENDENT": ("under R-Zero's own prompt the final solver beats the base model on MATH-500 in both "
                                     "seeds; the gain is not shown in both seeds under the plain Base prompt (format contrast and "
                                     "plain truncation reported)"),
    "REPRODUCES IN ONE SEED": ("under R-Zero's own prompt the final solver beats the base model on MATH-500 in one seed only "
                               "(the other seed's label is reported): a partial reproduction, weak either way with two seeds"),
    "PLAIN-ONLY": ("no seed gains under R-Zero's own prompt while at least one gains under the plain Base prompt: the replication "
                   "does not reproduce the reported gain under R-Zero's protocol; the plain-prompt gain is reported descriptively"),
    "NOT REPRODUCED": ("no seed gains under either prompt: the replication does not reproduce the reported gains at greedy "
                       "decoding with our scorers (no GPT-4o re-check)"),
    "NOT REPRODUCED, LOSS": ("no seed gains under either prompt, and under R-Zero's own prompt the final solver is worse than the "
                             "base model on MATH-500 in at least one seed"),
    "NOT RUN": "a primary or secondary cell of a seed is missing"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def reading_text(rd, fc_pos, trunc_more):
    """Registered part-2 wording (sections 2.4, 2.5). fc_pos: the paired format-contrast CI lower bound (iteration 3, MATH-500)
    is > 0 in both seeds; trunc_more: the plain-truncation flag. Both additions apply to REPRODUCES, FORMAT-DEPENDENT only."""
    t = READING_TEXT[rd]
    if rd == "REPRODUCES, FORMAT-DEPENDENT":
        if fc_pos is True:
            t += "; the gain depends on the format (paired format-contrast CI lower bound > 0 in both seeds)"
        if trunc_more is True:
            t += (f"; the plain condition truncates more than {TRUNC_MARGIN:.0f} points more often than rzchat (2,048- vs "
                  "4,096-token cap), so this is not called format adoption")
    return t


def boxes(t):
    """Contents of every \\boxed{...} in t, braces matched."""
    out, i = [], 0
    while True:
        j = t.find("\\boxed", i)
        if j < 0:
            return out
        k = j + 6
        while k < len(t) and t[k] == " ":
            k += 1
        if k >= len(t) or t[k] != "{":
            i = j + 6; continue
        depth, e = 0, k
        while e < len(t):
            depth += {"{": 1, "}": -1}.get(t[e], 0)
            if depth == 0:
                break
            e += 1
        if e >= len(t):
            return out
        out.append(t[k + 1:e].strip()); i = e + 1


def load(c, m, name):
    tag = COND[c][0] + m; sf, jf = f"{E}/{tag}.summary.json", f"{E}/{tag}.{name}.jsonl"
    if not (os.path.exists(sf) and os.path.exists(jf)):
        return None, None
    s = json.load(open(sf)).get(name)
    if s is None:
        return None, None
    assert s.get("model") == PATH[m], f"{sf}: set {name} was run on {s.get('model')}, not on {PATH[m]}"
    if c == "rzchat":
        assert s.get("b_eval_sha256") == B_EVAL_SHA, f"{sf}: not scored with the registered b_eval.py"
    rows = {}
    for l in open(jf, encoding="utf-8"):
        r = json.loads(l); rows[r["id"]] = r
    if s.get("n") != SETS[name] or len(rows) != SETS[name]:
        return None, None
    return s, rows


def paired(a, b, key, B=4000):
    ids = sorted(a); assert set(ids) == set(b), key
    dd = [b[i] - a[i] for i in ids]; n = len(dd); rng = random.Random(f"{20260926 + 14}:{key}")
    bs = sorted(100 * sum(rng.choices(dd, k=n)) / n for _ in range(B))
    return 100 * sum(dd) / n, [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]


out = {"paper": PAPER, "accuracy": {}, "delta_vs_base": {}, "delta_vs_base_rzero_style": {}, "labels": {}, "format_contrast": {},
       "plain_diagnostics": {}, "not_run": []}
data, summ = {}, {}
for c, (pre, metrics) in COND.items():
    for m in MODELS:
        for name in SETS:
            s, rows = load(c, m, name)
            if s is None:
                out["not_run"].append(f"{c}:{m}:{name}"); continue
            data[(c, m, name)] = rows; summ[(c, m, name)] = s
            out["accuracy"][f"{c}:{m}:{name}"] = {k: (100 * s[k] if s.get(k) is not None else None) for k in metrics} | {
                k: s.get(k) for k in ("trunc_rate", "im_end_rate", "sample_trunc", "mean_len", "k") if k in s}
for name in SETS:
    ps = {summ[k].get("rendered_prompt_0_sha256") for k in summ if k[0] == "rzchat" and k[2] == name}
    assert len(ps) <= 1, f"rzchat {name}: the rendered first prompt differs between models"
ok_ = lambda rows, f: {i: int(r[f] or 0) for i, r in rows.items()}
for c in COND:
    for name in SETS:
        base = data.get((c, "qwen3_4b_base", name))
        for i in (1, 2, 3):
            ds = {}
            for s in (0, 1):
                m = f"rzero_s{s}_it{i}"
                if base is not None and (c, m, name) in data:
                    d, ci = paired(ok_(base, "greedy_ok"), ok_(data[(c, m, name)], "greedy_ok"), f"{c}:{m}:{name}:greedy_ok")
                    ds[s] = (d, ci, seed_label(ci))
                    out["delta_vs_base"][f"{c}:{m}:{name}"] = {"delta": d, "ci95": ci, "label": seed_label(ci)}
                    if c == "rzchat" and all(r.get("ok_rzero_style") is not None for r in list(base.values()) + list(data[(c, m, name)].values())):
                        d2, ci2 = paired(ok_(base, "ok_rzero_style"), ok_(data[(c, m, name)], "ok_rzero_style"), f"{c}:{m}:{name}:ok_rzero_style")
                        out["delta_vs_base_rzero_style"][f"{c}:{m}:{name}"] = {"delta": d2, "ci95": ci2, "label": seed_label(ci2)}
            e = {"per_seed": [ds[s][2] if s in ds else None for s in (0, 1)]}
            if len(ds) == 2:
                mean = (ds[0][0] + ds[1][0]) / 2
                pd = (PAPER["it"][i][name] - PAPER["base"][name]) if name in PAPER["base"] else None
                e.update({"seed_mean_delta": mean, "paper_delta": pd, "ratio_to_paper": (mean / pd) if pd else None})
            out["labels"][f"{c}:it{i}:{name}"] = e
pk, sk = f"{PRIMARY[0]}:it{PRIMARY[1]}:{PRIMARY[2]}", f"{SECONDARY[0]}:it{SECONDARY[1]}:{SECONDARY[2]}"
P, S = out["labels"][pk]["per_seed"], out["labels"][sk]["per_seed"]
rd = reading(P, S)
out["primary"] = {"cell": pk, "scorer": "b_eval (greedy_ok)", "per_seed": {f"s{s}": out["delta_vs_base"].get(f"rzchat:rzero_s{s}_it3:math500_train") for s in (0, 1)}}
out["secondary"] = {"cell": sk, "scorer": "b_eval (greedy_ok)", "per_seed": {f"s{s}": out["delta_vs_base"].get(f"plain:rzero_s{s}_it3:math500_train") for s in (0, 1)}}
for name in SETS:
    for i in (1, 2, 3):
        for s in (0, 1):
            m = f"rzero_s{s}_it{i}"; k = [("plain", "qwen3_4b_base"), ("plain", m), ("rzchat", "qwen3_4b_base"), ("rzchat", m)]
            if all((c, mm, name) in data for c, mm in k):
                a0, a1, b0, b1 = (ok_(data[(c, mm, name)], "greedy_ok") for c, mm in k)
                dplain = {j: a1[j] - a0[j] for j in a0}; drz = {j: b1[j] - b0[j] for j in a0}
                d, ci = paired(dplain, drz, f"format:{m}:{name}")
                out["format_contrast"][f"{m}:{name}"] = {"delta_rzchat_minus_delta_plain": d, "ci95": ci}
for m in MODELS:
    tag = "a12_plain_" + m; side = f"{E}/{tag}.plain_stops.json"
    st = json.load(open(side)) if os.path.exists(side) else {}
    for name in SETS:
        rows = data.get(("plain", m, name))
        if rows is None:
            continue
        bx = [boxes(r.get("greedy_text") or "") for r in rows.values()]
        out["plain_diagnostics"][f"{m}:{name}"] = {
            "greedy_trunc_rate": sum(int(r.get("greedy_trunc") or 0) for r in rows.values()) / len(rows),
            "first_last_boxed_differ_rate": sum(len(b) >= 2 and b[0] != b[-1] for b in bx) / len(rows),
            "multi_boxed_rate": sum(len(b) >= 2 for b in bx) / len(rows),
            "im_end_stop_rate": (st[name]["n_im_end_stop"] / st[name]["n"]) if name in st and st[name].get("prompts_match") else None}
# the FORMAT-DEPENDENT wording (sections 2.4, 2.5): format contrast at iteration 3 on MATH-500 in both seeds; plain truncation
# minus rzchat truncation (points) for the base model and the two S_3 on MATH-500
fc = {f"s{s}": out["format_contrast"].get(f"rzero_s{s}_it3:math500_train") for s in (0, 1)}
fc_pos = None if None in fc.values() else all(v["ci95"][0] > 0 for v in fc.values())
tr = {}
for m in ("qwen3_4b_base", "rzero_s0_it3", "rzero_s1_it3"):
    pdg, acc = out["plain_diagnostics"].get(f"{m}:math500_train"), out["accuracy"].get(f"rzchat:{m}:math500_train", {})
    tr[m] = 100 * (pdg["greedy_trunc_rate"] - acc["trunc_rate"]) if pdg and acc.get("trunc_rate") is not None else None
trunc_more = True if any(v is not None and v > TRUNC_MARGIN for v in tr.values()) else (None if None in tr.values() else False)
out["reading"] = {"label": rd, "text": reading_text(rd, fc_pos, trunc_more), "primary_labels": P, "secondary_labels": S,
                  "magnitude_ratio_to_paper_primary": out["labels"][pk].get("ratio_to_paper"),
                  "format_contrast_it3_math500": fc, "format_contrast_ci_lower_gt0_both_seeds": fc_pos,
                  "plain_minus_rzchat_truncation_points_it3_math500": tr, "plain_truncates_more": trunc_more}
if os.path.exists(OLD):
    old_ok = sha(OLD) == OLD_SHA
    o = json.load(open(OLD)) if old_ok else {}
    br = {"stored_summary_registered": old_ok, "tolerance_points": TOL_REPRO, "sets": {}}
    for name in SETS:
        k = f"plain:qwen3_4b_base:{name}"
        if k in out["accuracy"] and name in o:
            f_, s_ = out["accuracy"][k]["greedy_acc"], 100 * o[name]["greedy_acc"]
            br["sets"][name] = {"fresh": f_, "stored": s_, "diff": f_ - s_, "fresh_lenient": out["accuracy"][k]["greedy_acc_lenient"],
                                "stored_lenient": 100 * o[name]["greedy_acc_lenient"] if "greedy_acc_lenient" in o[name] else None}
    br["flag"] = any(abs(v["diff"]) > TOL_REPRO for v in br["sets"].values()) if br["sets"] else None
    out["base_reproduction"] = br
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
json.dump(out, open(OUTF, "w"), indent=1)
print(json.dumps(out, indent=1))
