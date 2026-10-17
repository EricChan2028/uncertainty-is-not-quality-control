"""H2 amendment 1 read (registration dirA5/human_validation_mech/H2_amendment1_2026-10-05.md): human check of the A8 mechanism
labels on the v2 packets. Inputs: _key_do_not_open/key_v2.json (h21_packet.py); dirA5/results/a8/labels_main.json and
a8_key/units_main_key.jsonl; human labels in v2/: labels_A_ann1.csv, labels_A_ann2.csv, labels_B_ann1.csv, labels_B_ann2.csv
(each against its own version's ids), optionally labels_A_adjudicated.csv and labels_B_adjudicated.csv (keyed by item, see below).
Normalisation as in h2_read.py (NFKC, upper case; N/R/E/M; Y/N; NUM/DECLINE/NONE; else missing).
Integrity checks (pre-registered; a file failing one is not analysed and the failure is reported):
  I1  every filled hid of labels_X_annN.csv belongs to version annN of part X;
  I2  copy check: if, in a part, the two annotators' label columns agree position by position (row i of one file against row i
      of the other, i.e. by raw id number, which maps to different items in the two versions) on more than 90% of the rows
      both filled, the two files are treated as copies of one another and neither is analysed in that part.
Inclusion: an annotator enters a part if they have at least 40 valid part-A labels or 120 valid part-B flags (the sizes of
the first H2 packet); items are randomly ordered, so a partial set is a random subset.
Primary (per item, the mean of the included annotators who labelled it; items are a8_key for part A and (a8_key, sample) for
part B):
  H2a  part A: D_h = share(R or M | U) - share(R or M | A); 95% CI from an item bootstrap within class.
  H2b  part B, U questions: per question the mean flag of its round-5 solutions minus that of its base solutions, averaged
       over questions with both; 95% CI from a bootstrap over questions.
  H2c  Cohen's kappa of each annotator with the LLM label (part A: R-or-M binary and four-way; part B: flag) and between the
       annotators on items both labelled, with item-bootstrap 95% CIs.
  Bootstraps: 4,000 replicates, random.Random(20261006); the same draws give the LLM value on the same items.
Secondary: per-annotator estimates; consensus (agreeing annotators, else adjudicated); Figure-3 categories at base and round 5;
the 38 answerable questions. Interpretation bands as in H2 (H2a: CI low > 0.20 CONFIRMED, in (0, 0.20] WEAKER, else NOT;
H2b: same with 0.10; kappa bands). python3 h21_read.py [--dev DIR]"""
import argparse, csv, json, os, random, unicodedata
from collections import Counter, defaultdict

R = "/home/tione/notebook/rsi_rl"; D = f"{R}/dirA5/results"; HV = f"{R}/dirA5/human_validation_mech"
ap = argparse.ArgumentParser(); ap.add_argument("--dev", default=None); args = ap.parse_args()
LAB = args.dev or f"{HV}/v2"
OUTFILE = os.path.join(args.dev, "h21_read_dev.json") if args.dev else f"{HV}/v2/h21_read.json"
B_REPS, SEED, MIN_A, MIN_B = 4000, 20261006, 40, 120

KEYV = json.load(open(f"{HV}/_key_do_not_open/key_v2.json"))
VMAP = {(x["version"], x["hid"]): x for x in KEYV}
A8K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8_key/units_main_key.jsonl")}
SRC = {"NONE", "READINGS", "ERRORS", "MIXED"}
LLM = {}
for x in json.load(open(f"{D}/a8/labels_main.json")):
    if (isinstance(x, dict) and isinstance(x.get("key"), str) and x.get("disagreement_source") in SRC
            and isinstance(x.get("gap_flag"), list) and len(x["gap_flag"]) == 8 and all(isinstance(b, bool) for b in x["gap_flag"])):
        if x["key"] in LLM and LLM[x["key"]] != x: LLM[x["key"]] = None
        else: LLM.setdefault(x["key"], x)
LLM = {k: v for k, v in LLM.items() if v is not None}
ITEM = {}  # canonical item -> metadata
for x in KEYV:
    it = x["a8_key"] if x["part"] == "A" else (x["a8_key"], x["sample"])
    ITEM[(x["part"], it)] = x
A_ITEMS = sorted(it for p, it in ITEM if p == "A")
B_ITEMS = sorted((it for p, it in ITEM if p == "B"), key=str)
meta = lambda part, it: ITEM[(part, it)]

def norm(s): return unicodedata.normalize("NFKC", s or "").strip().upper()
MAP_A = {"N": "N", "NONE": "N", "R": "R", "READING": "R", "READINGS": "R", "E": "E", "ERROR": "E", "ERRORS": "E", "M": "M", "MIXED": "M"}
MAP_F = {"Y": 1, "YES": 1, "TRUE": 1, "1": 1, "N": 0, "NO": 0, "FALSE": 0, "0": 0}
MAP_ANS = {"NUM": "NUM", "NUMBER": "NUM", "NUMERIC": "NUM", "DECLINE": "DECLINE", "DECLINES": "DECLINE", "DECLINED": "DECLINE",
           "NONE": "NONE", "NO ANSWER": "NONE"}

def read_rows(name):
    p = os.path.join(LAB, name)
    if not os.path.exists(p): return None
    raw = open(p, "rb").read()
    for enc in ("utf-8-sig", "gb18030"):
        try: text = raw.decode(enc); break
        except UnicodeDecodeError: continue
    return [{norm(k).lower(): norm(v) for k, v in r.items() if k is not None} for r in csv.DictReader(text.splitlines())]

integrity = {}
def load(part, n):
    rows = read_rows(f"labels_{part}_ann{n}.csv")
    if rows is None: return None, None
    ver = f"ann{n}"; out, raw_seq, bad_ids = {}, {}, 0
    for r in rows:
        h = r.get("hid", "")
        if part == "A": val = MAP_A.get(r.get("label", ""))
        else: val = {"flag": MAP_F.get(r.get("flag", "")), "answer": MAP_ANS.get(r.get("answer", ""))}
        filled = (val is not None) if part == "A" else (val["flag"] is not None or val["answer"] is not None)
        if not filled: continue
        if (ver, h) not in VMAP or VMAP[(ver, h)]["part"] != part: bad_ids += 1; continue
        x = VMAP[(ver, h)]; it = x["a8_key"] if part == "A" else (x["a8_key"], x["sample"])
        out[it] = val; raw_seq[x["order"]] = val
    integrity[f"I1_{part}_ann{n}"] = {"foreign_or_unknown_ids": bad_ids, "ok": bad_ids == 0}
    return (out if bad_ids == 0 else None), raw_seq

H, RAW = {}, {}
for part in ("A", "B"):
    for n in (1, 2):
        H[(part, n)], RAW[(part, n)] = load(part, n)
for part in ("A", "B"):
    s1, s2 = RAW[(part, 1)], RAW[(part, 2)]
    if s1 and s2:
        common = [o for o in s1 if o in s2]
        same = sum(1 for o in common if s1[o] == s2[o])
        frac = same / len(common) if common else None
        integrity[f"I2_{part}"] = {"positional_agreement": frac, "n": len(common), "copies": bool(frac is not None and frac > 0.90)}
        if frac is not None and frac > 0.90: H[(part, 1)] = H[(part, 2)] = None
def adj(part):
    rows = read_rows(f"labels_{part}_adjudicated.csv") or []
    out = {}
    for r in rows:  # adjudicated files use annotator 1's ids
        x = VMAP.get(("ann1", r.get("hid", "")))
        if not x or x["part"] != part: continue
        it = x["a8_key"] if part == "A" else (x["a8_key"], x["sample"])
        out[it] = MAP_A.get(r.get("label", "")) if part == "A" else {"flag": MAP_F.get(r.get("flag", "")), "answer": MAP_ANS.get(r.get("answer", ""))}
    return out
ADJ_A, ADJ_B = adj("A"), adj("B")

incA = [n for n in (1, 2) if H[("A", n)] is not None and sum(v is not None for v in H[("A", n)].values()) >= MIN_A]
incB = [n for n in (1, 2) if H[("B", n)] is not None and sum(v["flag"] is not None for v in H[("B", n)].values()) >= MIN_B]
rd = lambda lab: None if lab is None else int(lab in ("R", "M"))
llm_rd = lambda it: (lambda l: None if l is None else int(l["disagreement_source"] in ("READINGS", "MIXED")))(LLM.get(it))
llm_src = lambda it: (lambda l: None if l is None else l["disagreement_source"][0])(LLM.get(it))
llm_flag = lambda it: (lambda l: None if l is None else int(l["gap_flag"][it[1]]))(LLM.get(it[0]))
llm_num = lambda it: int(A8K[it[0]]["answers"][it[1]] is not None)

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
def pct(v, q):
    v = sorted(x for x in v if x is not None)
    if not v: return None
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]
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
def kappa_ci(f, items):
    rng = random.Random(SEED); bs = []
    for _ in range(B_REPS):
        s = [rng.choice(items) for _ in items]; bs.append(kappa([f(i) for i in s]))
    est = kappa([f(i) for i in items])
    return {"kappa": est, "ci95": ci(bs), "n": sum(1 for i in items if None not in f(i)), "band": kband(est)}

# ---------- H2a ----------
U_A = [it for it in A_ITEMS if meta("A", it)["class"] == "U"]; A_A = [it for it in A_ITEMS if meta("A", it)["class"] == "A"]
def contrast(vals, us, as_):
    mu, ma = mean([vals[i] for i in us]), mean([vals[i] for i in as_])
    return None if mu is None or ma is None else mu - ma
def h2a(labels):  # labels: list of dicts item -> label
    if not labels: return None
    hv = {it: mean([rd(l.get(it)) for l in labels]) for it in A_ITEMS}; lv = {it: llm_rd(it) for it in A_ITEMS}
    us = [i for i in U_A if hv[i] is not None]; as_ = [i for i in A_A if hv[i] is not None]
    if not us or not as_: return None
    rng = random.Random(SEED); bh, bl, bd = [], [], []
    for _ in range(B_REPS):
        su = [rng.choice(us) for _ in us]; sa = [rng.choice(as_) for _ in as_]
        x, y = contrast(hv, su, sa), contrast(lv, su, sa); bh.append(x); bl.append(y); bd.append(None if x is None or y is None else x - y)
    est, el = contrast(hv, us, as_), contrast(lv, us, as_); c = ci(bh)
    return {"share_U": mean([hv[i] for i in us]), "share_A": mean([hv[i] for i in as_]), "D_h": est, "ci95": c,
            "llm_share_U": mean([lv[i] for i in us]), "llm_share_A": mean([lv[i] for i in as_]), "D_llm_same_items": el, "ci95_llm": ci(bl),
            "D_h_minus_D_llm": None if est is None or el is None else est - el, "ci95_diff": ci(bd), "n_U": len(us), "n_A": len(as_),
            "interpretation": band(c[0], 0.20)}

# ---------- H2b ----------
def questions(cls):
    q = defaultdict(lambda: {"base": [], "r5": []})
    for it in B_ITEMS:
        x = meta("B", it)
        if x["class"] == cls: q[x["id"]]["base" if x["model"] == "base" else "r5"].append(it)
    return list(q.values())
QU, QA = questions("U"), questions("A")
def qrise(fl, q):
    b, r = mean([fl[i] for i in q["base"]]), mean([fl[i] for i in q["r5"]])
    return None if b is None or r is None else r - b
def h2b(labels, qs):
    if not labels: return None
    hv = {it: mean([(l.get(it) or {}).get("flag") for l in labels]) for it in B_ITEMS}; lv = {it: llm_flag(it) for it in B_ITEMS}
    qs = [q for q in qs if qrise(hv, q) is not None]
    if not qs: return None
    rng = random.Random(SEED); bh, bl, bd = [], [], []
    for _ in range(B_REPS):
        s = [rng.choice(qs) for _ in qs]; x, y = mean([qrise(hv, q) for q in s]), mean([qrise(lv, q) for q in s])
        bh.append(x); bl.append(y); bd.append(None if x is None or y is None else x - y)
    est, el = mean([qrise(hv, q) for q in qs]), mean([qrise(lv, q) for q in qs]); c = ci(bh)
    return {"flag_base": mean([hv[i] for q in qs for i in q["base"]]), "flag_r5": mean([hv[i] for q in qs for i in q["r5"]]),
            "rise_h": est, "ci95": c, "llm_flag_base": mean([lv[i] for q in qs for i in q["base"]]),
            "llm_flag_r5": mean([lv[i] for q in qs for i in q["r5"]]), "rise_llm_same_items": el, "ci95_llm": ci(bl),
            "rise_h_minus_llm": None if est is None or el is None else est - el, "ci95_diff": ci(bd), "n_questions": len(qs),
            "interpretation": band(c[0], 0.10)}

# ---------- agreement ----------
def gA(n, it): return None if H[("A", n)] is None else H[("A", n)].get(it)
def gB(n, it, f): return None if H[("B", n)] is None else (H[("B", n)].get(it) or {}).get(f)
agree = {}
for n in (1, 2):
    agree[f"A_binary_ann{n}_vs_llm"] = kappa_ci(lambda it, n=n: (rd(gA(n, it)), llm_rd(it)), A_ITEMS)
    agree[f"A_fourway_ann{n}_vs_llm"] = kappa_ci(lambda it, n=n: (gA(n, it), llm_src(it)), A_ITEMS)
    agree[f"B_flag_ann{n}_vs_llm"] = kappa_ci(lambda it, n=n: (gB(n, it, "flag"), llm_flag(it)), B_ITEMS)
agree["A_binary_ann1_vs_ann2"] = kappa_ci(lambda it: (rd(gA(1, it)), rd(gA(2, it))), A_ITEMS)
agree["A_fourway_ann1_vs_ann2"] = kappa_ci(lambda it: (gA(1, it), gA(2, it)), A_ITEMS)
agree["B_flag_ann1_vs_ann2"] = kappa_ci(lambda it: (gB(1, it, "flag"), gB(2, it, "flag")), B_ITEMS)
agree["B_answer_ann1_vs_ann2"] = kappa_ci(lambda it: (gB(1, it, "answer"), gB(2, it, "answer")), B_ITEMS)

consA = {it: (gA(1, it) if gA(1, it) is not None and gA(1, it) == gA(2, it) else ADJ_A.get(it)) for it in A_ITEMS}
consB = {}
for it in B_ITEMS:
    adjv = ADJ_B.get(it) or {}
    f1, f2, a1, a2 = gB(1, it, "flag"), gB(2, it, "flag"), gB(1, it, "answer"), gB(2, it, "answer")
    consB[it] = {"flag": f1 if f1 is not None and f1 == f2 else adjv.get("flag"), "answer": a1 if a1 is not None and a1 == a2 else adjv.get("answer")}

def fig3(labels, cls):
    res = {}
    for when in ("base", "r5"):
        its = [it for it in B_ITEMS if meta("B", it)["class"] == cls and (meta("B", it)["model"] == "base") == (when == "base")]
        cat, n = Counter(), 0
        for it in its:
            for l in labels:
                v = l.get(it) or {}
                if v.get("flag") is None or v.get("answer") is None: continue
                n += 1; cat["flagged_answer" if v["answer"] == "NUM" and v["flag"] else "silent_answer" if v["answer"] == "NUM" else "no_numeric"] += 1
        lc = Counter("flagged_answer" if llm_num(it) and llm_flag(it) else "silent_answer" if llm_num(it) else "no_numeric" for it in its if llm_flag(it) is not None)
        res[when] = {"human": {k: cat[k] / n for k in ("flagged_answer", "silent_answer", "no_numeric")} if n else None, "n_labels": n,
                     "llm": {k: lc[k] / sum(lc.values()) for k in ("flagged_answer", "silent_answer", "no_numeric")} if lc else None, "n_items": len(its)}
    return res

LA = [H[("A", n)] for n in incA]; LB = [H[("B", n)] for n in incB]
out = {"inputs": {"labels_dir": LAB, "dev": bool(args.dev), "integrity": integrity, "included_A": [f"ann{n}" for n in incA],
                  "included_B": [f"ann{n}" for n in incB],
                  "n_valid": {f"{p}_ann{n}": (None if H[(p, n)] is None else (sum(v is not None for v in H[(p, n)].values()) if p == "A"
                              else sum(v["flag"] is not None for v in H[(p, n)].values()))) for p in ("A", "B") for n in (1, 2)},
                  "adjudicated": {"A": len(ADJ_A), "B": len(ADJ_B)}},
       "H2a_primary": h2a(LA), "H2a_per_annotator": {f"ann{n}": h2a([H[("A", n)]]) for n in incA},
       "H2b_primary_U": h2b(LB, QU), "H2b_per_annotator_U": {f"ann{n}": h2b([H[("B", n)]], QU) for n in incB},
       "H2b_answerable": h2b(LB, QA), "H2c_agreement": agree,
       "H2a_consensus": h2a([consA]), "H2b_consensus_U": h2b([consB], QU),
       "fig3_U": fig3(LB, "U"), "fig3_A": fig3(LB, "A")}
json.dump(out, open(OUTFILE, "w"), indent=1)
pa, pb = out["H2a_primary"], out["H2b_primary_U"]
print("integrity", integrity)
print("H2a", None if pa is None else (round(pa["D_h"], 3), pa["ci95"], pa["interpretation"], "LLM", round(pa["D_llm_same_items"], 3), pa["n_U"], pa["n_A"]))
print("H2b", None if pb is None else (round(pb["rise_h"], 3), pb["ci95"], pb["interpretation"], "LLM", round(pb["rise_llm_same_items"], 3), pb["n_questions"]))
print("kappa", {k: (None if v["kappa"] is None else round(v["kappa"], 2)) for k, v in agree.items()})
print("written", OUTFILE)
