"""Independent read-only check of paper_arr/analyses/invented.py (EXPLORATORY, post hoc).
Re-implements loading, class membership, confidence rule and the invented/copied/zero/non-numeric classification
with own code (Decimal-based matching), compares item-level with the verbatim a4_nt_analyses.py rule, and adds
sensitivity variants. Writes nothing except its own JSON next to it.  python3 check_invented.py"""
import collections, glob, json, math, re
from decimal import Decimal, InvalidOperation

R = "/home/tione/notebook/rsi_rl"
OUT = f"{R}/paper_arr/analyses/check_invented.json"
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
out = {"exploratory_post_hoc": True, "integrity": {}}

# ---- audit verdicts (check duplicates / conflicts)
verd, dup, conflict = {}, 0, 0
for f in sorted(glob.glob(f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts/*.json")):
    for v in json.load(open(f))["verdicts"]:
        if v["id"] in verd:
            dup += 1
            conflict += verd[v["id"]] != v["verdict"]
        verd[v["id"]] = v["verdict"]
out["integrity"]["audit"] = {"n": len(verd), "dup": dup, "conflict": conflict, "labels": dict(collections.Counter(verd.values()))}

# ---- pool texts
rows = [json.loads(l) for l in open(f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl")]
text = {r["id"]: r["problem"] for r in rows}
out["integrity"]["pool"] = {"rows": len(rows), "unique_ids": len(text)}

# ---- guard records
rec, integ = {}, {}
for ck in CKS:
    allr = [json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/qwen_s0/{ck}/guard_records.jsonl")]
    prop = [r for r in allr if r["kind"] == "proposed"]
    ids_ = [r["id"] for r in prop]
    rec[ck] = {r["id"]: r for r in prop}
    integ[ck] = {"records": len(allr), "kinds": dict(collections.Counter(r["kind"] for r in allr)), "proposed": len(prop),
                 "dup_ids": len(ids_) - len(set(ids_)), "p_mode_not_multiple_of_1/8": sum(abs(r["p_mode"] * 8 - round(r["p_mode"] * 8)) > 1e-9 for r in prop),
                 "mode_none": sum(r["mode"] is None for r in prop), "missing_audited": sum(1 for i in verd if i not in rec[ck])}
out["integrity"]["guard_records"] = integ
cand = [i for i in verd if i in text and i in rec["base"]]
out["integrity"]["n_candidates"] = len(cand)

# ---- own classification
NUM_RE = re.compile(r"\d[\d,]*\.?\d*")
def text_values(t):
    vals = set()
    for tok in NUM_RE.findall(t):
        s = tok.replace(",", "")
        if s.endswith("."):
            s = s[:-1]
        if s and s.count(".") <= 1 and s.replace(".", "").isdigit():
            vals.add(Decimal(s))
    return vals

def as_number(m):
    try:
        f = float(m)
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(f):
        return None, True
    try:
        return Decimal(str(m).strip().replace("_", "")), False
    except InvalidOperation:
        return Decimal(repr(f)), False

def mine(m, t):
    d, nonfinite = as_number(m)
    if nonfinite:
        return "invented_nonfinite"
    if d is None:
        return "non-numeric"
    if d == 0:
        return "zero"
    if any(abs(d - v) < Decimal("1e-9") for v in text_values(t)):
        return "copied"
    return "invented"

def a4_in_text(m, t):
    try:
        v = float(m)
    except Exception:
        return False
    return any(abs(v - float(x.replace(",", ""))) < 1e-9 for x in re.findall(r"\d[\d,]*\.?\d*", t) if x.replace(",", "").replace(".", "", 1).isdigit())
def a4_cat(m, t):
    try:
        v = float(m)
        if v == 0: return "zero"
        elif a4_in_text(m, t): return "copied"
        else: return "invented"
    except Exception:
        return "non-numeric"

WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000, "half": Decimal("0.5"), "twice": 2, "double": 2, "triple": 3, "dozen": 12, "quarter": Decimal("0.25")})
def text_values_broad(t):
    vals = set(text_values(t))
    for w in re.findall(r"[a-z]+", t.lower()):
        if w in WORDS:
            vals.add(Decimal(WORDS[w]))
    for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", t):
        vals.add(Decimal(p) / 100)
    for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", t):
        if int(b):
            vals.add(Decimal(a) / Decimal(b))
    return vals
def broad(m, t):
    d, nonfinite = as_number(m)
    if d is None:
        return "non-numeric" if not nonfinite else "invented_nonfinite"
    if d == 0:
        return "zero"
    if any(abs(d - v) < Decimal("1e-6") for v in text_values_broad(t)):
        return "copied"
    return "invented"

def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

CL = {"S": ("SOLVABLE",), "A": ("AMBIGUOUS",), "U": ("UNSOLVABLE",), "ILL": ("AMBIGUOUS", "UNSOLVABLE")}
tab, disagree, nonfin = {}, [], []
for ck in CKS:
    tab[ck] = {}
    for cl, labs in CL.items():
        mem = [i for i in cand if verd[i] in labs]
        conf = [i for i in mem if rec[ck][i]["mode"] is not None and rec[ck][i]["p_mode"] >= 0.5]
        c_m = collections.Counter(mine(rec[ck][i]["mode"], text[i]) for i in conf)
        c_b = collections.Counter(broad(rec[ck][i]["mode"], text[i]) for i in conf)
        for i in conf:
            a, b = a4_cat(rec[ck][i]["mode"], text[i]), mine(rec[ck][i]["mode"], text[i])
            if a != b.replace("_nonfinite", "") and cl != "ILL":
                disagree.append([ck, cl, i, rec[ck][i]["mode"], a, b])
            if b == "invented_nonfinite" and cl != "ILL":
                nonfin.append([ck, cl, i, rec[ck][i]["mode"]])
        n = len(conf); k = c_m["invented"] + c_m["invented_nonfinite"]
        nnum = n - c_m["non-numeric"]
        tab[ck][cl] = {"n_class": len(mem), "n_conf": n, "conf_pct": round(100 * n / len(mem), 1),
                       "copied": c_m["copied"], "zero": c_m["zero"], "invented": k, "nonnumeric": c_m["non-numeric"],
                       "invented_pct": round(100 * k / n, 1), "wilson": wilson(k, n),
                       "zero_pct": round(100 * c_m["zero"] / n, 1), "copied_pct": round(100 * c_m["copied"] / n, 1),
                       "invented_pct_of_numeric": round(100 * k / nnum, 1),
                       "broad_textmatch_invented_pct": round(100 * (c_b["invented"] + c_b["invented_nonfinite"]) / n, 1),
                       "mean_nums_in_text": round(sum(len(NUM_RE.findall(text[i])) for i in conf) / n, 2),
                       "mean_distinct_values_in_text": round(sum(len(text_values(text[i])) for i in conf) / n, 2)}
        if ck != "base":
            tab[ck][cl]["same_mode_as_base"] = sum(rec["base"][i]["mode"] == rec[ck][i]["mode"] for i in conf)
out["table"] = tab
out["item_disagreements_vs_a4_rule"] = disagree[:50]
out["n_item_disagreements_vs_a4_rule"] = len(disagree)
out["nonfinite_modes"] = nonfin

def dif(ck, a, b="S"):
    x, y = tab[ck][a], tab[ck][b]
    pa, pb = x["invented"] / x["n_conf"], y["invented"] / y["n_conf"]
    se = math.sqrt(pa * (1 - pa) / x["n_conf"] + pb * (1 - pb) / y["n_conf"])
    return [round(100 * (pa - pb), 1), round(100 * (pa - pb - 1.96 * se), 1), round(100 * (pa - pb + 1.96 * se), 1)]
out["contrasts_pp_ci"] = {f"{ck}_{a}-S": dif(ck, a) for ck in ("base", "r5") for a in ("U", "ILL", "A")}

strata = lambda t: min(len(text_values(t)), 9)
st = {}
for ck in ("base", "r5"):
    per = collections.defaultdict(lambda: {"S": [0, 0], "U": [0, 0]})
    for cl in ("S", "U"):
        for i in cand:
            if verd[i] != CL[cl][0]:
                continue
            r = rec[ck][i]
            if r["mode"] is None or r["p_mode"] < 0.5:
                continue
            k = strata(text[i]); per[k][cl][1] += 1; per[k][cl][0] += mine(r["mode"], text[i]).startswith("invented")
    rows_ = {str(k): {"S": f"{v['S'][0]}/{v['S'][1]}", "U": f"{v['U'][0]}/{v['U'][1]}"} for k, v in sorted(per.items())}
    wS = {k: v["S"][1] for k, v in per.items() if v["U"][1] > 0 and v["S"][1] > 0}
    tot = sum(wS.values())
    stdU = sum(wS[k] * per[k]["U"][0] / per[k]["U"][1] for k in wS) / tot
    crudeS_sub = sum(per[k]["S"][0] for k in wS) / tot
    st[ck] = {"by_n_distinct_values(9=9+)": rows_, "U_invented_std_to_S_mix_pct": round(100 * stdU, 1), "S_invented_same_strata_pct": round(100 * crudeS_sub, 1), "S_items_in_shared_strata": tot}
out["stratified"] = st

ref = json.load(open(f"{R}/dirA4/results/nt_analyses_qwen_s0.json"))
out["ref_nt_analyses"] = {"modal_answers": ref["modal_answers"], "mode_stable_from_base": ref["mode_stable_from_base"]}
ana = json.load(open(f"{R}/paper_arr/analyses/invented.json"))
mism = []
for ck in CKS:
    for cl, cn in (("S", "SOLVABLE"), ("A", "AMBIGUOUS"), ("U", "UNSOLVABLE"), ("ILL", "ILLPOSED")):
        a = ana["by_checkpoint"][ck][cn]; m = tab[ck][cl]
        pairs = [(a["n_class"], m["n_class"]), (a["n_confident"], m["n_conf"]), (a["categories"]["other invented number"], m["invented"]),
                 (a["categories"]["copied from the question"], m["copied"]), (a["categories"]["zero"], m["zero"]), (a["categories"]["non-numeric"], m["nonnumeric"]),
                 (a["invented_wilson95"], m["wilson"]), (a["mean_numbers_in_text_confident"], m["mean_nums_in_text"]), (a.get("same_mode_as_base"), m.get("same_mode_as_base"))]
        if any(x != y for x, y in pairs):
            mism.append([ck, cl, pairs])
out["mismatches_vs_analyst_json"] = mism
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out["integrity"], indent=0))
print("disagreements vs a4 rule:", len(disagree), disagree[:10]); print("nonfinite:", nonfin)
for ck in CKS:
    for cl in CL:
        print(ck, cl, tab[ck][cl])
print("contrasts", out["contrasts_pp_ci"])
print("stratified", json.dumps(st, indent=0))
print("mismatches vs analyst:", mism)
print("ref", out["ref_nt_analyses"])
