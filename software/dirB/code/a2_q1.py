"""A2 E2 read for the verdict: on the 412 judge-validated UNSOLVABLE GSM-HP twins under the abstention prompt,
  Q1_conf = P(majority answer is a real answer (not an abstention) with p_mode >= 0.5)   <- confident ill-posedness
  Q1_raw  = P(p_mode >= 0.5) as printed by a2_analyze (abstention counted as the mode)
  abstention rates on unsolvable / solvable-different / intact, intact accuracy.
Per model, arm, seed (from dirB/results/hp/hpabs_<model>_<arm>_mx_s<seed>_shard0of1.jsonl) and for the base models
(hpabs_<basetag>_shard*.jsonl).  python3 a2_q1.py
"""
import glob
import json
import os
import re
from collections import defaultdict

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
VER = {(str(v["fam"]), int(v["task"])): v["verdict"] for v in V["verdicts"]}


def read(files):
    d = [json.loads(l) for f in files for l in open(f)]
    uns = [r for r in d if r["form"] == "deleted" and VER.get((str(r["fam"]), int(r["task"]))) == "UNSOLVABLE"]
    sdf = [r for r in d if r["form"] == "deleted" and VER.get((str(r["fam"]), int(r["task"]))) == "SOLVABLE_DIFFERENT"]
    inta = [r for r in d if r["form"] == "intact"]
    if not uns:
        return None
    conf = sum(1 for r in uns if r["p_mode"] >= 0.5 and str(r.get("mode")).lower() not in ("unknown", "none", "null")) / len(uns)
    raw = sum(1 for r in uns if r["p_mode"] >= 0.5) / len(uns)
    ab = lambda rows: sum(r.get("abstain", 0) for r in rows) / max(1, len(rows))
    acc = sum(r["correct"] for r in inta) / max(1, len(inta))
    return {"n_uns": len(uns), "q1_conf": conf, "q1_raw": raw, "abs_uns": ab(uns), "abs_sdf": ab(sdf), "abs_int": ab(inta), "int_acc": acc}


rows = {}
for tag, name in (("qwen3_4b_base", "qwen3_4b_Base"), ("llama3p2_3b", "llama3p2_3b")):
    fs = sorted(glob.glob(f"{B}/results/hp/hpabs_{tag}_shard*.jsonl"))
    if fs:
        rows[(name, "base", -1)] = read(fs)
for f in sorted(glob.glob(f"{B}/results/hp/hpabs_*_mx_s*_shard0of1.jsonl")):
    m = re.search(r"hpabs_(qwen3_4b_Base|llama3p2_3b)_(own_abs|own|pdc)_mx_s(\d+)_shard", f)
    if m:
        rows[(m.group(1), m.group(2), int(m.group(3)))] = read([f])
print(f"{'model':14s} {'arm':8s} seed | Q1_conf  Q1_raw | abstain uns / sdf / int | intact acc   (412 unsolvable twins, abstention prompt)")
for (mdl, arm, seed), r in sorted(rows.items()):
    if r is None:
        continue
    print(f"{mdl:14s} {arm:8s} {seed:4d} | {100*r['q1_conf']:6.1f} {100*r['q1_raw']:6.1f} | {100*r['abs_uns']:5.1f} / {100*r['abs_sdf']:5.1f} / {100*r['abs_int']:5.1f} | {100*r['int_acc']:5.1f}")
print()
agg = defaultdict(list)
for (mdl, arm, seed), r in rows.items():
    if r is not None and seed >= 0:
        agg[(mdl, arm)].append(r)
for (mdl, arm), rs in sorted(agg.items()):
    n = len(rs)
    print(f"{mdl:14s} {arm:8s} mean(n={n}) | Q1_conf {100*sum(r['q1_conf'] for r in rs)/n:5.1f}  Q1_raw {100*sum(r['q1_raw'] for r in rs)/n:5.1f} | abs {100*sum(r['abs_uns'] for r in rs)/n:5.1f} / {100*sum(r['abs_sdf'] for r in rs)/n:5.1f} / {100*sum(r['abs_int'] for r in rs)/n:5.1f} | int acc {100*sum(r['int_acc'] for r in rs)/n:5.1f}")
