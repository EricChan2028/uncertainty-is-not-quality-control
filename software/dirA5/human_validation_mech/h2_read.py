"""H2 read (registration dirA5/human_validation_mech/H2_registration_2026-10-04.md): human check of the A8 mechanism labels.
Inputs: _key_do_not_open/key.json (h2_packet.py); dirA5/results/a8/labels_main.json and a8_key/units_main_key.jsonl (LLM labels
and parsed answers); human labels labels_A_ann1.csv, labels_A_ann2.csv, labels_B_ann1.csv, labels_B_ann2.csv, optionally
labels_A_adjudicated.csv and labels_B_adjudicated.csv (columns as in the templates; the adjudicated files need only the
disagreement items).
Normalisation: NFKC, strip, upper case. Part A label N/R/E/M (also NONE/READING(S)/ERROR(S)/MIXED); part B flag Y/N (also
YES/NO/TRUE/FALSE/1/0) and answer NUM/DECLINE/NONE (also NUMBER/NUMERIC/DECLINES/DECLINED/NO ANSWER). Anything else is missing.
An annotator missing more than 10% of a part's items is left out of that part's primary estimates (reported descriptively).
Primary (each included annotator's binary label, averaged per item, so an item scores 0, 0.5 or 1):
  H2a  part A: D_h = share(R or M | U) - share(R or M | A); 95% CI from an item bootstrap within class.
  H2b  part B, the 48 unanswerable questions: flag rate of the round-5 solution minus that of the base solution; 95% CI from a
       bootstrap over questions (pairs resampled).
  H2c  Cohen's kappa of each annotator with the LLM label (part A: R-or-M binary and four-way; part B: flag) and between the
       annotators, with item-bootstrap 95% CIs.
  All bootstraps: 4,000 replicates, random.Random(20261005); the same draws give the LLM value on the same items and the
  human-minus-LLM difference.
Secondary: per-annotator estimates; consensus labels (agreeing annotators, else the adjudicated label, else missing);
Figure-3 categories (flagged answer = NUM and Y; silent answer = NUM and N; no numeric answer = DECLINE or NONE) at base and
round 5, with the LLM analogue on the same samples (gap_flag; numeric if the parsed answer exists); the 12 answerable pairs.
Interpretation bands (registration section 6): H2a CONFIRMED if the CI's lower end > 0.20, DIRECTION_CONFIRMED_WEAKER if it is in
(0, 0.20], NOT_CONFIRMED otherwise; H2b the same with 0.10; kappa: > 0.60 substantial, 0.41-0.60 moderate, 0.21-0.40 fair,
<= 0.20 slight. python3 h2_read.py [--dev DIR]  (DIR: simulated label files; output DIR/h2_read_dev.json)"""
import argparse, csv, json, os, random, unicodedata
from collections import Counter

R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA5/results"; OUT = f"{R}/dirA5/human_validation_mech"
ap = argparse.ArgumentParser(); ap.add_argument("--dev", default=None); args = ap.parse_args()
LAB = args.dev or OUT
OUTFILE = os.path.join(args.dev, "h2_read_dev.json") if args.dev else f"{OUT}/h2_read.json"
B_REPS, SEED = 4000, 20261005

KEYH = {x["hid"]: x for x in json.load(open(f"{OUT}/_key_do_not_open/key.json"))}
A8K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8_key/units_main_key.jsonl")}
SRC = {"NONE", "READINGS", "ERRORS", "MIXED"}
LLM = {}
for x in json.load(open(f"{D}/a8/labels_main.json")):
    if (isinstance(x, dict) and isinstance(x.get("key"), str) and x.get("disagreement_source") in SRC
            and isinstance(x.get("gap_flag"), list) and len(x["gap_flag"]) == 8 and all(isinstance(b, bool) for b in x["gap_flag"])):
        if x["key"] in LLM and LLM[x["key"]] != x: LLM[x["key"]] = None  # conflicting duplicates count as missing
        else: LLM.setdefault(x["key"], x)
LLM = {k: v for k, v in LLM.items() if v is not None}
A_ITEMS = sorted(h for h, x in KEYH.items() if x["part"] == "A")
B_ITEMS = sorted(h for h, x in KEYH.items() if x["part"] == "B")

def norm(s):
    return unicodedata.normalize("NFKC", s or "").strip().upper()
MAP_A = {"N": "N", "NONE": "N", "R": "R", "READING": "R", "READINGS": "R", "E": "E", "ERROR": "E", "ERRORS": "E", "M": "M", "MIXED": "M"}
MAP_F = {"Y": 1, "YES": 1, "TRUE": 1, "1": 1, "N": 0, "NO": 0, "FALSE": 0, "0": 0}
MAP_ANS = {"NUM": "NUM", "NUMBER": "NUM", "NUMERIC": "NUM", "DECLINE": "DECLINE", "DECLINES": "DECLINE", "DECLINED": "DECLINE",
           "NONE": "NONE", "NO ANSWER": "NONE"}

def read_csv(name, cols):
    p = os.path.join(LAB, name)
    if not os.path.exists(p): return None
    raw = open(p, "rb").read()
    for enc in ("utf-8-sig", "gb18030"):  # a spreadsheet program may save the file in the local code page
        try: text = raw.decode(enc); break
        except UnicodeDecodeError: continue
    out = {}
    for r in csv.DictReader(text.splitlines()):
        r = {norm(k).lower(): v for k, v in r.items() if k is not None}
        h = norm(r.get("hid"))
        if h in KEYH: out[h] = {c: norm(r.get(c)) for c in cols}
    return out

def ann_A(name):
    raw = read_csv(name, ["label"])
    return None if raw is None else {h: MAP_A.get(v["label"]) for h, v in raw.items() if MAP_A.get(v["label"])}
def ann_B(name):
    raw = read_csv(name, ["flag", "answer"])
    if raw is None: return None
    return {h: {"flag": MAP_F.get(v["flag"]), "answer": MAP_ANS.get(v["answer"])} for h, v in raw.items()}

HA = {a: ann_A(f"labels_A_{a}.csv") for a in ("ann1", "ann2")}
HB = {a: ann_B(f"labels_B_{a}.csv") for a in ("ann1", "ann2")}
ADJ_A, ADJ_B = ann_A("labels_A_adjudicated.csv") or {}, ann_B("labels_B_adjudicated.csv") or {}
rd = lambda lab: None if lab is None else int(lab in ("R", "M"))
llm_rd = lambda h: (lambda l: None if l is None else int(l["disagreement_source"] in ("READINGS", "MIXED")))(LLM.get(KEYH[h]["a8_key"]))
llm_src = lambda h: (lambda l: None if l is None else l["disagreement_source"][0])(LLM.get(KEYH[h]["a8_key"]))
llm_flag = lambda h: (lambda l: None if l is None else int(l["gap_flag"][KEYH[h]["sample"]]))(LLM.get(KEYH[h]["a8_key"]))
llm_num = lambda h: int(A8K[KEYH[h]["a8_key"]]["answers"][KEYH[h]["sample"]] is not None)

def coverage(lbl, items, field=None):
    if lbl is None: return 0.0
    got = [h for h in items if h in lbl and (field is None or lbl[h].get(field) is not None)]
    return len(got) / len(items)
incA = [a for a in ("ann1", "ann2") if coverage(HA[a], A_ITEMS) >= 0.90]
incB = [a for a in ("ann1", "ann2") if coverage(HB[a], B_ITEMS, "flag") >= 0.90]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v: return None
    i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1))))); return v[i]
def ci(v): return [pct(v, 0.025), pct(v, 0.975)]
def band(lo, cut):
    if lo is None: return None
    return "CONFIRMED" if lo > cut else ("DIRECTION_CONFIRMED_WEAKER" if lo > 0 else "NOT_CONFIRMED")
def kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0: return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / n ** 2
    return None if pe == 1 else (po - pe) / (1 - pe)
def kband(k):
    if k is None: return None
    return "substantial" if k > 0.60 else "moderate" if k > 0.40 else "fair" if k > 0.20 else "slight"

# ---------- H2a ----------
def item_rd(h, anns, labels):  # mean over annotators of the binary label
    return mean([rd(labels[a].get(h)) for a in anns if labels[a] is not None])
def contrast(vals, uitems, aitems):
    mu, ma = mean([vals[h] for h in uitems]), mean([vals[h] for h in aitems])
    return None if mu is None or ma is None else mu - ma
U_A = [h for h in A_ITEMS if KEYH[h]["class"] == "U"]; A_A = [h for h in A_ITEMS if KEYH[h]["class"] == "A"]
def h2a(anns, labels):
    if not anns: return None
    hv = {h: item_rd(h, anns, labels) for h in A_ITEMS}; lv = {h: llm_rd(h) for h in A_ITEMS}
    rng = random.Random(SEED); bh, bl, bd = [], [], []
    for _ in range(B_REPS):
        us = [rng.choice(U_A) for _ in U_A]; as_ = [rng.choice(A_A) for _ in A_A]
        x, y = contrast(hv, us, as_), contrast(lv, us, as_)
        bh.append(x); bl.append(y); bd.append(None if x is None or y is None else x - y)
    est = contrast(hv, U_A, A_A); c = ci(bh)
    return {"annotators": anns, "share_U": mean([hv[h] for h in U_A]), "share_A": mean([hv[h] for h in A_A]), "D_h": est, "ci95": c,
            "llm_share_U": mean([lv[h] for h in U_A]), "llm_share_A": mean([lv[h] for h in A_A]), "D_llm_same_items": contrast(lv, U_A, A_A),
            "ci95_llm": ci(bl), "D_h_minus_D_llm": None if est is None else est - contrast(lv, U_A, A_A), "ci95_diff": ci(bd),
            "n_U": len(U_A), "n_A": len(A_A), "interpretation": band(c[0], 0.20)}

# ---------- H2b ----------
def pairs_of(cls):
    by = {}
    for h in B_ITEMS:
        x = KEYH[h]
        if x["class"] == cls: by.setdefault(x["id"], {})["base" if x["model"] == "base" else "r5"] = h
    return [v for v in by.values() if set(v) == {"base", "r5"}]
PU, PA = pairs_of("U"), pairs_of("A")
def item_flag(h, anns, labels): return mean([labels[a].get(h, {}).get("flag") for a in anns if labels[a] is not None])
def rise(fl, prs):
    d = [None if fl[p["r5"]] is None or fl[p["base"]] is None else fl[p["r5"]] - fl[p["base"]] for p in prs]
    return mean(d)
def h2b(anns, labels, prs):
    if not anns: return None
    hv = {h: item_flag(h, anns, labels) for h in B_ITEMS}; lv = {h: llm_flag(h) for h in B_ITEMS}
    rng = random.Random(SEED); bh, bl, bd = [], [], []
    for _ in range(B_REPS):
        s = [rng.choice(prs) for _ in prs]; x, y = rise(hv, s), rise(lv, s)
        bh.append(x); bl.append(y); bd.append(None if x is None or y is None else x - y)
    est = rise(hv, prs); c = ci(bh)
    return {"annotators": anns, "flag_base": mean([hv[p["base"]] for p in prs]), "flag_r5": mean([hv[p["r5"]] for p in prs]), "rise_h": est,
            "ci95": c, "llm_flag_base": mean([lv[p["base"]] for p in prs]), "llm_flag_r5": mean([lv[p["r5"]] for p in prs]),
            "rise_llm_same_items": rise(lv, prs), "ci95_llm": ci(bl), "rise_h_minus_llm": None if est is None else est - rise(lv, prs),
            "ci95_diff": ci(bd), "n_pairs": len(prs), "interpretation": band(c[0], 0.10)}

# ---------- H2c ----------
def kappa_ci(f, items):
    rng = random.Random(SEED); est = kappa([f(h) for h in items]); bs = []
    for _ in range(B_REPS):
        s = [rng.choice(items) for _ in items]; bs.append(kappa([f(h) for h in s]))
    return {"kappa": est, "ci95": ci(bs), "n": sum(1 for h in items if None not in f(h)), "band": kband(est)}
def lab_A(a, h): return None if HA[a] is None else HA[a].get(h)
def lab_B(a, h, fld): return None if HB[a] is None else (HB[a].get(h) or {}).get(fld)
agree = {}
for a in ("ann1", "ann2"):
    agree[f"A_binary_{a}_vs_llm"] = kappa_ci(lambda h, a=a: (rd(lab_A(a, h)), llm_rd(h)), A_ITEMS)
    agree[f"A_fourway_{a}_vs_llm"] = kappa_ci(lambda h, a=a: (lab_A(a, h), llm_src(h)), A_ITEMS)
    agree[f"B_flag_{a}_vs_llm"] = kappa_ci(lambda h, a=a: (lab_B(a, h, "flag"), llm_flag(h)), B_ITEMS)
agree["A_binary_ann1_vs_ann2"] = kappa_ci(lambda h: (rd(lab_A("ann1", h)), rd(lab_A("ann2", h))), A_ITEMS)
agree["A_fourway_ann1_vs_ann2"] = kappa_ci(lambda h: (lab_A("ann1", h), lab_A("ann2", h)), A_ITEMS)
agree["B_flag_ann1_vs_ann2"] = kappa_ci(lambda h: (lab_B("ann1", h, "flag"), lab_B("ann2", h, "flag")), B_ITEMS)
agree["B_answer_ann1_vs_ann2"] = kappa_ci(lambda h: (lab_B("ann1", h, "answer"), lab_B("ann2", h, "answer")), B_ITEMS)

# ---------- consensus (secondary) ----------
def consensus_A():
    out = {}
    for h in A_ITEMS:
        l1, l2 = lab_A("ann1", h), lab_A("ann2", h)
        out[h] = l1 if (l1 is not None and l1 == l2) else ADJ_A.get(h)
    return {"cons": out}
def consensus_B():
    out = {}
    for h in B_ITEMS:
        f1, f2 = lab_B("ann1", h, "flag"), lab_B("ann2", h, "flag")
        a1, a2 = lab_B("ann1", h, "answer"), lab_B("ann2", h, "answer")
        adj = ADJ_B.get(h) or {}
        out[h] = {"flag": f1 if (f1 is not None and f1 == f2) else adj.get("flag"),
                  "answer": a1 if (a1 is not None and a1 == a2) else adj.get("answer")}
    return {"cons": out}

# ---------- Figure-3 categories (secondary) ----------
def fig3(anns, labels, cls):
    res = {}
    for when in ("base", "r5"):
        hs = [h for h in B_ITEMS if KEYH[h]["class"] == cls and (KEYH[h]["model"] == "base") == (when == "base")]
        cat = Counter(); n = 0
        for h in hs:
            for a in anns:
                x = (labels[a] or {}).get(h) or {}
                if x.get("flag") is None or x.get("answer") is None: continue
                n += 1; cat["flagged_answer" if x["answer"] == "NUM" and x["flag"] else "silent_answer" if x["answer"] == "NUM" else "no_numeric"] += 1
        lc = Counter("flagged_answer" if llm_num(h) and llm_flag(h) else "silent_answer" if llm_num(h) else "no_numeric" for h in hs if llm_flag(h) is not None)
        res[when] = {"human": {k: cat[k] / n for k in ("flagged_answer", "silent_answer", "no_numeric")} if n else None, "n_labels": n,
                     "llm": {k: lc[k] / sum(lc.values()) for k in ("flagged_answer", "silent_answer", "no_numeric")} if lc else None, "n_items": len(hs)}
    return res

out = {"inputs": {"labels_dir": LAB, "dev": bool(args.dev),
                  "coverage_A": {a: coverage(HA[a], A_ITEMS) for a in ("ann1", "ann2")},
                  "coverage_B_flag": {a: coverage(HB[a], B_ITEMS, "flag") for a in ("ann1", "ann2")},
                  "coverage_B_answer": {a: coverage(HB[a], B_ITEMS, "answer") for a in ("ann1", "ann2")},
                  "included_A": incA, "included_B": incB, "llm_labels_found": {"A": sum(llm_rd(h) is not None for h in A_ITEMS),
                  "B": sum(llm_flag(h) is not None for h in B_ITEMS)}, "adjudicated_A": len(ADJ_A), "adjudicated_B": len(ADJ_B)},
       "H2a_primary": h2a(incA, HA),
       "H2a_per_annotator": {a: h2a([a], HA) for a in ("ann1", "ann2") if HA[a] is not None},
       "H2b_primary_U": h2b(incB, HB, PU),
       "H2b_per_annotator_U": {a: h2b([a], HB, PU) for a in ("ann1", "ann2") if HB[a] is not None},
       "H2b_answerable_pairs": h2b(incB, HB, PA),
       "H2c_agreement": agree}
cA, cB = consensus_A(), consensus_B()
out["H2a_consensus"] = h2a(["cons"], {"cons": cA["cons"]})
out["H2b_consensus_U"] = h2b(["cons"], {"cons": cB["cons"]}, PU)
out["H2c_consensus_vs_llm"] = {"A_binary": kappa_ci(lambda h: (rd(cA["cons"].get(h)), llm_rd(h)), A_ITEMS),
                               "B_flag": kappa_ci(lambda h: (cB["cons"].get(h, {}).get("flag"), llm_flag(h)), B_ITEMS)}
out["n_disagreements"] = {"A_label": sum(1 for h in A_ITEMS if lab_A("ann1", h) and lab_A("ann2", h) and lab_A("ann1", h) != lab_A("ann2", h)),
                          "A_binary": sum(1 for h in A_ITEMS if lab_A("ann1", h) and lab_A("ann2", h) and rd(lab_A("ann1", h)) != rd(lab_A("ann2", h))),
                          "B_flag": sum(1 for h in B_ITEMS if lab_B("ann1", h, "flag") is not None and lab_B("ann2", h, "flag") is not None
                                        and lab_B("ann1", h, "flag") != lab_B("ann2", h, "flag")),
                          "B_answer": sum(1 for h in B_ITEMS if lab_B("ann1", h, "answer") and lab_B("ann2", h, "answer")
                                          and lab_B("ann1", h, "answer") != lab_B("ann2", h, "answer"))}
out["fig3_U"] = fig3(incB, HB, "U"); out["fig3_A"] = fig3(incB, HB, "A")
json.dump(out, open(OUTFILE, "w"), indent=1)
pa, pb = out["H2a_primary"], out["H2b_primary_U"]
print("H2a", None if pa is None else (round(pa["D_h"], 3), pa["ci95"], pa["interpretation"], "LLM same items", round(pa["D_llm_same_items"], 3)))
print("H2b", None if pb is None else (round(pb["rise_h"], 3), pb["ci95"], pb["interpretation"], "LLM same items", round(pb["rise_llm_same_items"], 3)))
print("kappa", {k: (None if v["kappa"] is None else round(v["kappa"], 2)) for k, v in agree.items()})
print("written", OUTFILE)
