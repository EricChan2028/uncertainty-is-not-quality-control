"""Exploratory (not registered), after the H2 amendment-1 read: does the plain-prompt answer parser count verbal declines as
numeric answers, and how much does that change the self-consistent-answer rate?

Inputs (read only): dirA5/results/a8/texts/a8_*.jsonl (8 plain-prompt solutions per UMWP item, same prompt, parser and
sampling as the UMWP probe; texts saved), dirA5/human_validation_mech/v2/labels_B_ann{1,2}.csv and _key_do_not_open/key_v2.json
(human answer status of 600 single solutions).
Parsers. ORIGINAL = the probe's: last 'Answer: <number>' match, else the last number in the text; 'length' finishes are
non-answers. STRICT = ORIGINAL, except that a solution is a non-answer when its last 'Answer:' line contains no number, a
decline ('N/A', 'unknown', 'cannot be determined', 'insufficient', '<number>' ...) or an expression in a variable
('10S', '4x + 150'), or, without an 'Answer:' line, when its last 400 characters decline.
Outputs: agreement of each parser with the two annotators' answer status; per checkpoint, the share of the 112 unanswerable
items whose most frequent numeric answer holds >= 4 of 8 (self-consistent), and >= 6 and 8 of 8, under both parsers.
python3 strict_parse.py  ->  paper_arr/analyses/rev5/strict_parse_2026-10-06.json"""
import csv, json, re, unicodedata
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; HV = f"{R}/dirA5/human_validation_mech"
OUT = f"{R}/paper_arr/analyses/rev5/strict_parse_2026-10-06.json"
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
def norm_ans(a):
    if a is None: return None
    try:
        v = float(str(a).replace(",", "")); return int(v) if abs(v - round(v)) < 1e-9 else round(v, 4)
    except Exception: return None
def original(t, fin):
    if fin == "length": return None
    m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", t)
    if m: return norm_ans(m[-1])
    m = NUM_RE.findall(t)
    return norm_ans(m[-1].rstrip(".")) if m else None
DECL = re.compile(r"n/?a\b|unknown|undetermin|indetermin|insufficient|not enough|cannot|can't|can not|unable|impossible|not possible|"
                  r"not (be )?(determin|calculat|found|comput|known|given|provided|specified)|no (definite|definitive|numeric|numerical|specific|single|exact) (answer|value|number)|<number>|depends on", re.I)
SYMB = re.compile(r"\d\s*[a-zA-Z]\b(?<!\d[kmKM]\b)|[a-zA-Z]\s*[+\-*/^]|[+\-*/^]\s*[a-zA-Z]\b|\([a-zA-Z]\)")
UNIT_OK = re.compile(r"^\$?\s*-?\d[\d,]*\.?\d*\s*(%|[a-zA-Z][a-zA-Z]+.*)?$")
def strict(t, fin):
    if fin == "length": return None
    lines = re.findall(r"Answer:\s*([^\n]*)", t)
    if lines:
        last = lines[-1].strip().strip("*").strip()
        if not re.search(r"\d", last) or DECL.search(last): return None
        if SYMB.search(last) and not UNIT_OK.match(last): return None
        m = re.search(r"-?\d[\d,]*\.?\d*", last)
        return norm_ans(m.group(0).rstrip(".")) if m else None
    if DECL.search(t[-400:]): return None
    return original(t, fin)

# ---- validation against the human answer status
key = json.load(open(f"{HV}/_key_do_not_open/key_v2.json")); VM = {(x["version"], x["hid"]): x for x in key}
TX = {}
for tag in ("a8_base", "a8_own_s0_r3", "a8_own_s1_r3", "a8_own_s2_r3", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5"):
    for l in open(f"{R}/dirA5/results/a8/texts/{tag}.jsonl"):
        x = json.loads(l); TX[(tag, x["id"])] = x
A8K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{R}/dirA5/results/a8_key/units_main_key.jsonl")}
def tag_of(m): return "a8_base" if m["model"] == "base" else f"a8_own_s{m['seed']}_r{m['round']}"
n = lambda s: unicodedata.normalize("NFKC", s or "").strip().upper()
hum = {}
for v in ("ann1", "ann2"):
    for r in csv.DictReader(open(f"{HV}/v2/labels_B_{v}.csv", encoding="utf-8-sig")):
        x = VM[(v, r["hid"].strip())]; hum.setdefault((x["a8_key"], x["sample"]), {})[v] = n(r["answer"])
conf = {"original": Counter(), "strict": Counter()}
for (k, s), d in hum.items():
    if len(set(d.values())) != 1: continue  # both annotators agree
    h_num = d["ann1"] == "NUM"
    m = A8K[k]; rec = TX[(tag_of(m), m["id"])]
    t, f = rec["texts"][s], rec["finish"][s]
    for name, fn in (("original", original), ("strict", strict)):
        conf[name][(h_num, fn(t, f) is not None)] += 1
val = {}
for name, c in conf.items():
    tot = sum(c.values()); acc = (c[(True, True)] + c[(False, False)]) / tot
    val[name] = {"n": tot, "accuracy": acc, "human_no_answer_parsed_as_number": c[(False, True)], "human_number_parsed_as_none": c[(True, False)],
                 "confusion": {f"human_num={a},parsed_num={b}": v for (a, b), v in c.items()}}

# ---- self-consistency on the 112 unanswerable items, per checkpoint
def sc_rates(fn, tag):
    rows = [x for (tg, _), x in TX.items() if tg == tag and not x["answerable"]]
    out = {">=4": 0, ">=6": 0, "8": 0, "n": len(rows)}
    for x in rows:
        ans = [fn(t, f) for t, f in zip(x["texts"], x["finish"])]
        c = Counter(a for a in ans if a is not None)
        top = c.most_common(1)[0][1] if c else 0
        out[">=4"] += top >= 4; out[">=6"] += top >= 6; out["8"] += top == 8
    return {k: (v / out["n"] if k != "n" else v) for k, v in out.items()}
sc = {tag: {"original": sc_rates(original, tag), "strict": sc_rates(strict, tag)}
      for tag in ("a8_base", "a8_own_s0_r3", "a8_own_s1_r3", "a8_own_s2_r3", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5")}
# sanity: ORIGINAL reproduces the stored parsed answers
mism = sum(1 for x in TX.values() for t, f, a in zip(x["texts"], x["finish"], x["answers"]) if original(t, f) != a)
json.dump({"validation": val, "self_consistency": sc, "original_vs_stored_mismatches": mism}, open(OUT, "w"), indent=1)
print("ORIGINAL parser vs stored answers, mismatches:", mism)
for name, v in val.items(): print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if k != "confusion"})
for tag, d in sc.items():
    print(f"{tag:14s} SC>=4 original {d['original']['>=4']:.3f} strict {d['strict']['>=4']:.3f} | >=6 {d['original']['>=6']:.3f} {d['strict']['>=6']:.3f} | 8/8 {d['original']['8']:.3f} {d['strict']['8']:.3f}")
