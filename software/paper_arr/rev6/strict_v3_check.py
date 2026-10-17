"""rev6 diagnostic (before the registered read): the base models continue a solution with new "Question: ... Answer: ..."
blocks, so strict() (which reads the last 'Answer:' line) reads an unrelated, model-written question. Candidate fix (v3): cut the
generated text at its first line starting with 'Question:' before reading it, then strict()'s status and the first parser's
number on the cut text. Checks: (1) agreement with the two annotators' answer status (592 solutions both agree on), for the
first parser, strict v1 and v3; (2) per rev6 dump: accuracy on the answerable originals and SC under ORIGINAL, v2 (frozen) and
v3, and the share of samples that contain a new question.   python3 strict_v3_check.py"""
import csv, json, os, re, sys, unicodedata
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; HV = f"{R}/dirA5/human_validation_mech"; D6 = f"{R}/paper_arr/rev6"
src = open(f"{R}/paper_arr/analyses/rev5/strict_parse.py").read(); sp = {}
exec(compile(src[:src.index("# ---- validation against the human answer status")], "strict_parse_defs", "exec"), sp)
strict, original = sp["strict"], sp["original"]
QRE = re.compile(r"\nQuestion:")
def cut(t):
    m = QRE.search(t); return t[:m.start()] if m else t
def v2(t, f, a):
    s = strict(t, f); return None if s is None else (a if a is not None else s)
def v3(t, f, a):
    tt = cut(t); s = strict(tt, f)
    if s is None: return None
    o = original(tt, f); return o if o is not None else s
# (1) human validation
key = json.load(open(f"{HV}/_key_do_not_open/key_v2.json")); VM = {(x["version"], x["hid"]): x for x in key}
A8K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{R}/dirA5/results/a8_key/units_main_key.jsonl")}
TX = {}
for tg in ("a8_base", "a8_own_s0_r3", "a8_own_s1_r3", "a8_own_s2_r3", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5"):
    for l in open(f"{R}/dirA5/results/a8/texts/{tg}.jsonl"):
        x = json.loads(l); TX[(tg, x["id"])] = x
tag_of = lambda m: "a8_base" if m["model"] == "base" else f"a8_own_s{m['seed']}_r{m['round']}"
nz = lambda s: unicodedata.normalize("NFKC", s or "").strip().upper()
hum = {}
for v in ("ann1", "ann2"):
    for r in csv.DictReader(open(f"{HV}/v2/labels_B_{v}.csv", encoding="utf-8-sig")):
        x = VM[(v, r["hid"].strip())]; hum.setdefault((x["a8_key"], x["sample"]), {})[v] = nz(r["answer"])
conf = {k: Counter() for k in ("original", "strict_v1", "v3")}
for (k, s), d in hum.items():
    if len(set(d.values())) != 1: continue
    h = d["ann1"] == "NUM"; m = A8K[k]; rec = TX[(tag_of(m), m["id"])]; t, f, a = rec["texts"][s], rec["finish"][s], rec["answers"][s]
    for name, val in (("original", a), ("strict_v1", strict(t, f)), ("v3", v3(t, f, a))):
        conf[name][(h, val is not None)] += 1
for name, c in conf.items():
    n = sum(c.values()); print(f"human check {name:9s}: agreement {(c[(True, True)] + c[(False, False)]) / n:.3f} (n {n}); numeric->none {c[(True, False)]}, none->numeric {c[(False, True)]}")
# (2) rev6 dumps
for f in sorted(os.listdir(f"{D6}/texts")):
    if not f.startswith("plain_") or "chat" in f: continue
    rows = {}
    for l in open(f"{D6}/texts/{f}"):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    if len(rows) < 1200: continue
    out = {}; nq = 0; ns_ = 0
    for name in ("ORIG", "v2", "v3"):
        acc = sc = 0
        for r in rows.values():
            if name == "ORIG": ans = r["answers"]
            else: ans = [(v2 if name == "v2" else v3)(t, fi, a) for t, fi, a in zip(r["texts"], r["finish"], r["answers"])]
            mode, n = Counter(ans).most_common(1)[0]
            if r["answerable"]: acc += mode is not None and r["ref"] is not None and mode == r["ref"]
            else: sc += mode is not None and n / 8 >= 0.5
            if name == "ORIG":
                nq += sum(bool(QRE.search(t)) for t in r["texts"]); ns_ += len(r["texts"])
        out[name] = (100 * sc / 600, 100 * acc / 600)
    print(f"{f[6:-6]:36s} newQ {100 * nq / ns_:5.1f}% | " + " | ".join(f"{k} SC {v[0]:5.1f} ACC {v[1]:5.1f}" for k, v in out.items()))
