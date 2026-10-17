"""Score the blind audit-validation parts (dirA3/results/audit_validation/parts/*.json) against the known status of the
200 tracer twins (labels.json): per kind the verdict distribution; on intact twins judged SOLVABLE, whether the judge's
note contains the gold answer; on deleted twins, how often the judge still finds a unique answer and whether it equals
the intact gold answer (a 'solvable-different' reading) or not. Writes audit_validation/result.json.
"""
import glob, json, re
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"
D = f"{R}/dirA3/results/audit_validation"
labels = json.load(open(f"{D}/labels.json"))
verd = {}
for f in sorted(glob.glob(f"{D}/parts/*.json")):
    for v in json.load(open(f))["verdicts"]:
        verd[v["id"]] = v
print("judged", len(verd), "of", len(labels))
def nums(s):
    return {x.replace(",", "") for x in re.findall(r"-?\d[\d,]*\.?\d*", s or "")}
def norm(x):
    try:
        return f"{float(str(x).replace(',', '')):g}"
    except Exception:
        return str(x)
res = {}
for kind in ("tracer_intact", "tracer_deleted"):
    ids = [i for i, l in labels.items() if l["kind"] == kind and i in verd]
    c = Counter(verd[i]["verdict"] for i in ids)
    conf = defaultdict(list)
    for i in ids:
        conf[verd[i]["verdict"]].append(float(verd[i].get("confidence", 0)))
    gold_hit = gold_miss = 0
    for i in ids:
        if verd[i]["verdict"] == "SOLVABLE":
            gold = norm(labels[i]["twin"]["answer"])
            if gold in {norm(x) for x in nums(verd[i].get("note", ""))}:
                gold_hit += 1
            else:
                gold_miss += 1
    n = len(ids)
    res[kind] = {"n": n, "counts": dict(c), "share": {k: round(v / n, 3) for k, v in c.items()},
                 "mean_conf": {k: round(sum(v) / len(v), 3) for k, v in conf.items()},
                 "solvable_note_matches_intact_gold": gold_hit, "solvable_note_other_answer": gold_miss}
    print(f"{kind:15s} n={n} " + " ".join(f"{k} {v} ({100*v/n:.1f}%)" for k, v in sorted(c.items())) +
          f" | SOLVABLE notes: gold answer stated {gold_hit}, other/none {gold_miss}")
ni = res["tracer_intact"]; nd = res["tracer_deleted"]
spec = ni["share"].get("SOLVABLE", 0)
sens = nd["share"].get("UNSOLVABLE", 0) + nd["share"].get("AMBIGUOUS", 0)
print(f"intact judged SOLVABLE (specificity of the ill-posed call): {100*spec:.1f}%")
print(f"deleted judged UNSOLVABLE or AMBIGUOUS (sensitivity, upper bound < 100 because ~24% of deletions leave a solvable text): {100*sens:.1f}%")
res["summary"] = {"intact_judged_solvable": spec, "deleted_judged_illposed": sens}
json.dump(res, open(f"{D}/result.json", "w"), indent=1)
print("wrote", f"{D}/result.json")
