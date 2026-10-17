"""rev6 (exploratory): the 'Exact match' paragraph of Appendix G under both parsers. Round-5 unanswerable A8 units (3 seeds x 112):
share the LLM labellers mark as without disagreement (disagreement_source NONE), share whose eight answers all coincide, and
the modal count among the further units the label covers. Output printed.   python3 a8_exact.py"""
import json, statistics
from collections import Counter
R = "/home/tione/notebook/rsi_rl"
src = open(f"{R}/paper_arr/analyses/rev5/strict_parse.py").read(); sp = {}
exec(compile(src[:src.index("# ---- validation against the human answer status")], "strict_parse_defs", "exec"), sp)
strict = sp["strict"]
import sys; sys.path.insert(0, f"{R}/paper_arr/rev6"); import strict_v3  # noqa: E402
K = {json.loads(l)["key"]: json.loads(l) for l in open(f"{R}/dirA5/results/a8_key/units_main_key.jsonl")}
L = {x["key"]: x for x in json.load(open(f"{R}/dirA5/results/a8/labels_main.json")) if isinstance(x, dict) and "key" in x}
TX = {}
for tg in ("a8_base", "a8_own_s0_r5", "a8_own_s1_r5", "a8_own_s2_r5"):
    for l in open(f"{R}/dirA5/results/a8/texts/{tg}.jsonl"):
        x = json.loads(l); TX[(tg, x["id"])] = x
for P in ("ORIGINAL", "STRICT"):
    for model in ("base", "r5"):
        per = {}
        for k, m in K.items():
            if m["answerable"] or k not in L: continue
            if (model == "base") != (m["model"] == "base"): continue
            tg = "a8_base" if m["model"] == "base" else f"a8_own_s{m['seed']}_r{m['round']}"
            x = TX[(tg, m["id"])]
            ans = x["answers"] if P == "ORIGINAL" else [strict_v3.read(t, f) for t, f in zip(x["texts"], x["finish"])]
            c = Counter(a for a in ans if a is not None); top = c.most_common(1)[0][1] if c else 0
            s = per.setdefault(m.get("seed", -1), {"n": 0, "none": 0, "exact": 0, "further": []})
            s["n"] += 1; lab_none = L[k]["disagreement_source"] == "NONE"; s["none"] += lab_none; s["exact"] += top == 8
            if lab_none and top < 8: s["further"].append(top); s.setdefault("decl5", 0); s["decl5"] += sum(a is None for a in ans) >= 5
        fur = [t for s in per.values() for t in s["further"]]
        print(P, model, {sd: (round(100 * s["none"] / s["n"], 1), round(100 * s["exact"] / s["n"], 1)) for sd, s in sorted(per.items())},
              "further units", len(fur), "modal count range", (min(fur), max(fur)) if fur else None, "median", statistics.median(fur) if fur else None, "further with >=5 non-answers", sum(s.get("decl5", 0) for s in per.values()))
