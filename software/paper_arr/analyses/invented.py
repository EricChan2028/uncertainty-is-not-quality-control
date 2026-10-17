"""EXPLORATORY (post hoc) baseline for the 'invented number' statistic of dirA3/code/a4_nt_analyses.py (L110-131).

a4_nt_analyses.py reports, on the fixed pool qwen_s0 (Qwen3-4B-Base own s0 r1 proposals, audited), for UNSOLVABLE items answered
confidently by the round-5 solver (mode not None and p_mode >= 0.5), the category of the modal answer:
zero / copied from the question / other invented number / non-numeric (zero tested first, then text match).
This script applies the SAME loader, audit labels, problem texts, candidate set (ids), confidence criterion and the
verbatim in_text / category code to every audit class (SOLVABLE, AMBIGUOUS, UNSOLVABLE, and ill-posed = AMBIGUOUS+UNSOLVABLE)
at every checkpoint base, r1..r5, so the 'invented' share on unsolvable items can be compared with the same statistic on
solvable items. Read-only except the output JSON in paper_arr/analyses/.
  python3 invented.py
"""
import glob
import json
import math
import re

R = "/home/tione/notebook/rsi_rl"
TAG = "qwen_s0"
AUD = f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts"
POOLF = f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl"
OUT = f"{R}/paper_arr/analyses/invented.json"
REF = f"{R}/dirA4/results/nt_analyses_qwen_s0.json"
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
CLASSES = ["SOLVABLE", "AMBIGUOUS", "UNSOLVABLE", "ILLPOSED"]

# ---- identical to a4_nt_analyses.py L28-45 (restricted to the plain-prompt records it uses for the modal-answer analysis)
V = {}
for f in glob.glob(f"{AUD}/*.json"):
    for v in json.load(open(f))["verdicts"]:
        V[v["id"]] = v["verdict"]
TXT = {json.loads(l)["id"]: json.loads(l)["problem"] for l in open(POOLF)}


def load(d, ck):
    try:
        return {r["id"]: r for r in (json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{d}/{ck}/guard_records.jsonl")) if r["kind"] == "proposed"}
    except FileNotFoundError:
        return None


P = {c: load(TAG, c) for c in CKS}
ids = [i for i in V if i in TXT and P["base"] and i in P["base"]]


# ---- verbatim from a4_nt_analyses.py L113-128
def in_text(m, t):
    try:
        v = float(m)
    except Exception:
        return False
    return any(abs(v - float(x.replace(",", ""))) < 1e-9 for x in re.findall(r"\d[\d,]*\.?\d*", t) if x.replace(",", "").replace(".", "", 1).isdigit())


def categorize(m, t):
    try:
        v = float(m)
        if v == 0:
            return "zero"
        elif in_text(m, t):
            return "copied from the question"
        else:
            return "other invented number"
    except Exception:
        return "non-numeric"


def in_class(i, cl):
    return V[i] in ("AMBIGUOUS", "UNSOLVABLE") if cl == "ILLPOSED" else V[i] == cl


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


def n_numbers(t):
    return len(re.findall(r"\d[\d,]*\.?\d*", t))


res = {}
for c in CKS:
    if P[c] is None:
        continue
    res[c] = {}
    for cl in CLASSES:
        members = [i for i in ids if in_class(i, cl) and i in P[c]]
        conf = [i for i in members if P[c][i]["p_mode"] >= 0.5 and P[c][i]["mode"] is not None]
        cat = {"copied from the question": 0, "zero": 0, "other invented number": 0, "non-numeric": 0}
        for i in conf:
            cat[categorize(P[c][i]["mode"], TXT[i])] += 1
        n = len(conf); k = cat["other invented number"]; num = n - cat["non-numeric"]
        row = {"n_class": len(members), "n_confident": n, "confident_share_pct": round(100 * n / max(1, len(members)), 1),
               "categories": cat,
               "invented_pct": round(100 * k / n, 1) if n else None, "invented_wilson95": wilson(k, n),
               "copied_pct": round(100 * cat["copied from the question"] / n, 1) if n else None,
               "zero_pct": round(100 * cat["zero"] / n, 1) if n else None,
               "nonnumeric_pct": round(100 * cat["non-numeric"] / n, 1) if n else None,
               "invented_pct_of_numeric": round(100 * k / num, 1) if num else None,
               "mean_numbers_in_text_confident": round(sum(n_numbers(TXT[i]) for i in conf) / n, 2) if n else None}
        if c != "base":
            st = sum(1 for i in conf if P["base"][i]["mode"] == P[c][i]["mode"])
            row["same_mode_as_base"] = st; row["same_mode_as_base_pct"] = round(100 * st / max(1, n), 1)
        res[c][cl] = row


def diff(c, a="UNSOLVABLE", b="SOLVABLE"):
    ra, rb = res[c][a], res[c][b]
    pa, pb = ra["categories"]["other invented number"] / ra["n_confident"], rb["categories"]["other invented number"] / rb["n_confident"]
    se = math.sqrt(pa * (1 - pa) / ra["n_confident"] + pb * (1 - pb) / rb["n_confident"])
    return {"a": a, "b": b, "diff_pp": round(100 * (pa - pb), 1), "normal_approx_95": [round(100 * (pa - pb - 1.96 * se), 1), round(100 * (pa - pb + 1.96 * se), 1)]}


out = {"pool": TAG, "exploratory_post_hoc": True, "n_audited_candidates": len(ids),
       "definition": "confident = plain-prompt modal answer not None and p_mode >= 0.5 (G=8); categories verbatim from a4_nt_analyses.py L113-128: zero tested first, then 'copied' if the numeric value equals any number in the problem text, else 'other invented number'; non-numeric if float() fails",
       "sources": {"records": f"{R}/dirA3/results/guard_diag/{TAG}/<ck>/guard_records.jsonl (kind == proposed)", "audit": AUD, "texts": POOLF, "code_path": f"{R}/dirA3/code/a4_nt_analyses.py L28-45, L112-131"},
       "class_sizes": {cl: sum(in_class(i, cl) for i in ids) for cl in CLASSES},
       "by_checkpoint": res,
       "contrasts": {"r5_U_minus_S": diff("r5"), "base_U_minus_S": diff("base"), "r5_illposed_minus_S": diff("r5", "ILLPOSED"), "base_illposed_minus_S": diff("base", "ILLPOSED")}}

# sanity check: reproduce the published UNSOLVABLE r5 numbers
ref = json.load(open(REF))
out["reproduces_nt_analyses_modal_answers"] = res["r5"]["UNSOLVABLE"]["categories"] == ref["modal_answers"]
out["reproduces_mode_stable_from_base"] = abs(res["r5"]["UNSOLVABLE"]["same_mode_as_base"] / res["r5"]["UNSOLVABLE"]["n_confident"] - ref["mode_stable_from_base"]) < 1e-12

print(f"pool {TAG}: {len(ids)} audited candidates; classes {out['class_sizes']}")
print("reproduces nt_analyses r5 UNSOLVABLE:", out["reproduces_nt_analyses_modal_answers"], out["reproduces_mode_stable_from_base"])
for c in res:
    for cl in CLASSES:
        r = res[c][cl]
        print(f"  {c:4s} {cl:10s} conf {r['n_confident']:3d}/{r['n_class']:3d} | invented {r['invented_pct']}% {r['invented_wilson95']} | copied {r['copied_pct']} zero {r['zero_pct']} nonnum {r['nonnumeric_pct']} | "
              f"#nums {r['mean_numbers_in_text_confident']} | same-as-base {r.get('same_mode_as_base_pct')}")
print("contrasts:", json.dumps(out["contrasts"]))
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
