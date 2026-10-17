"""rev6 (exploratory): four response categories for Figure 4, as the review asked. For each A8 sample (112 unanswerable and 112
hard answerable UMWP questions; base and round-5 solvers, seeds 0-2; eight samples each):
  silent answer      numeric answer under the STRICT parser, no gap flag
  conditional answer numeric answer, gap flag (a stated assumption or a flagged gap)
  flag and decline   no numeric answer, gap flag
  other non-answer   no numeric answer, no gap flag (cut off, symbolic, unrelated, declines without flagging)
Gap flags are the blind LLM labels of A8 (dirA5/results/a8/labels_main.json); numeric status is the STRICT parser of
paper_arr/analyses/rev5/strict_parse.py (validated against the H2 annotators). The same four categories from the two H2
annotators' labels on the 600 single solutions (flag Y/N x answer NUM vs DECLINE/NONE) are reported for comparison.
Also per unit: self-consistent under STRICT (modal numeric answer >= 4 of 8) and, of these, with no sample flagging a gap.
Output: paper_arr/rev6/fig4_categories.json.  python3 fig4_categories.py"""
import csv, importlib.util, json, unicodedata
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"
spec = importlib.util.spec_from_file_location("sp", f"{R}/paper_arr/analyses/rev5/strict_parse.py")
src = open(f"{R}/paper_arr/analyses/rev5/strict_parse.py").read()
src = src.split("# ---- validation against the human answer status")[0]  # parser definitions only
ns = {}; exec(compile(src, "strict_parse_defs", "exec"), ns)
strict = ns["strict"]
import sys; sys.path.insert(0, f"{R}/paper_arr/rev6"); import strict_v3  # noqa: E402  (amendment 2 of rev6_read.py)
K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{R}/dirA5/results/a8_key/units_main_key.jsonl")}
L = {x["key"]: x for x in json.load(open(f"{R}/dirA5/results/a8/labels_main.json")) if isinstance(x, dict) and "key" in x}
TX = {}
for tag in ("a8_base", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5"):
    for l in open(f"{R}/dirA5/results/a8/texts/{tag}.jsonl"):
        x = json.loads(l); TX[(tag, x["id"])] = x
def tag_of(m): return "a8_base" if m["model"] == "base" else f"a8_own_s{m['seed']}_r{m['round']}"
CATS = ("silent_answer", "conditional_answer", "flag_and_decline", "other_non_answer")
def cat(numeric, flag): return CATS[0] if numeric and not flag else CATS[1] if numeric else CATS[2] if flag else CATS[3]
res = defaultdict(Counter); unit = defaultdict(Counter)
for k, m in K.items():
    if m["model"] != "base" and m["round"] != 5: continue
    tg = tag_of(m)
    if (tg, m["id"]) not in TX or k not in L: continue
    x = TX[(tg, m["id"])]; fl = L[k]["gap_flag"]
    cell = f"{'base' if m['model'] == 'base' else 'r5_s' + str(m['seed'])}|{'A' if m['answerable'] else 'U'}"
    sa = [strict_v3.read(t, f) for t, f in zip(x["texts"], x["finish"])]
    for a, g in zip(sa, fl):
        res[cell][cat(a is not None, bool(g))] += 1
    c = Counter(a for a in sa if a is not None); top = c.most_common(1)[0][1] if c else 0
    unit[cell]["n"] += 1; unit[cell]["sc"] += top >= 4; unit[cell]["sc_no_flag"] += top >= 4 and not any(fl)
out = {"llm_flags_strict_parser": {c: {k: v / sum(cnt.values()) for k, v in cnt.items()} | {"n": sum(cnt.values())} for c, cnt in res.items()},
       "units_strict": {c: {"n": u["n"], "self_consistent": u["sc"] / u["n"], "self_consistent_no_sample_flagged": u["sc_no_flag"] / u["n"]} for c, u in unit.items()}}
# human reference (H2 part B): both annotators pooled
HV = f"{R}/dirA5/human_validation_mech"
key = json.load(open(f"{HV}/_key_do_not_open/key_v2.json")); VM = {(x["version"], x["hid"]): x for x in key}
n = lambda s: unicodedata.normalize("NFKC", s or "").strip().upper()
hres = defaultdict(Counter)
for v in ("ann1", "ann2"):
    for r in csv.DictReader(open(f"{HV}/v2/labels_B_{v}.csv", encoding="utf-8-sig")):
        x = VM[(v, r["hid"].strip())]
        cell = f"{'base' if x['model'] == 'base' else 'r5'}|{x['class']}"
        hres[cell][cat(n(r["answer"]) == "NUM", n(r["flag"]) == "Y")] += 1
out["human_part_b"] = {c: {k: v / sum(cnt.values()) for k, v in cnt.items()} | {"n": sum(cnt.values())} for c, cnt in hres.items()}
json.dump(out, open(f"{R}/paper_arr/rev6/fig4_categories.json", "w"), indent=1)
for src_name, d in out.items():
    print("==", src_name)
    for c in sorted(d):
        print(f"  {c:12s}", {k: round(v, 3) for k, v in d[c].items()})
