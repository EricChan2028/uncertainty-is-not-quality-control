"""Direction B readout: per model x arm x seed table of greedy / maj@8 accuracy on the three datasets and
the GSM-HP blind-spot mass, then the registered contrasts (own-perm, own-fmt, gt-own, own-base) with a
seed-level exact randomisation test (all 2^k sign flips... no: all C(2k,k) label assignments between the
two arms' seeds, as in code/seed_level.py) and the P1-P5 / KB1-KB3 verdicts.

  python3 b_analyze.py            # everything on disk
"""
import glob
import itertools
import json
import os
import re
from collections import defaultdict

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]
MODELS = {"qwen3_4b_Base": "qwen3_4b_base", "llama3p2_3b": "llama3p2_3b"}
ARMS = ["own", "perm", "fmt", "gt"]


def load_summary(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


_VPATH = f"{R}/dirA/results/twin_validation_probe543.json"
VERDICT = {(str(v["fam"]), int(v["task"])): v["verdict"] for v in json.load(open(_VPATH))["verdicts"]} if os.path.exists(_VPATH) else {}


def hp_mass(path):
    """Q1 = P(p_mode >= 0.5 | deleted) from a c0_probe shard file, on the judge-validated UNSOLVABLE twins when
    the verdict file exists (twin_validation_note_2026-09-22.md); None if absent."""
    if not os.path.exists(path):
        return None
    d = [json.loads(l) for l in open(path)]
    dele = [r for r in d if r["form"] == "deleted" and (not VERDICT or VERDICT.get((str(r["fam"]), int(r["task"]))) == "UNSOLVABLE")]
    inta = [r for r in d if r["form"] == "intact"]
    if not dele:
        return None
    return {"q1": sum(1 for r in dele if r["p_mode"] >= 0.5) / len(dele), "n": len(dele), "restricted": bool(VERDICT),
            "intact_acc": sum(r["correct"] for r in inta) / max(1, len(inta)),
            "intact_conf": sum(1 for r in inta if r["p_mode"] >= 0.5) / max(1, len(inta))}


# ---- collect
rows = {}   # (model, arm, seed) -> {ds: summary}
for m, base_tag in MODELS.items():
    base = {ds: load_summary(f"{B}/results/evals/{ds}/{base_tag}_step0.summary.json").get(ds) for ds in DS}
    hp0 = glob.glob(f"{R}/dirA/results/{base_tag}_shard*of8.jsonl")
    if hp0:
        d = [json.loads(l) for f in hp0 for l in open(f)]
        dele = [r for r in d if r["form"] == "deleted" and (not VERDICT or VERDICT.get((str(r["fam"]), int(r["task"]))) == "UNSOLVABLE")]
        base["hp"] = {"q1": sum(1 for r in dele if r["p_mode"] >= 0.5) / len(dele), "n": len(dele), "restricted": bool(VERDICT)}
    rows[(m, "base", -1)] = base
    for arm in ARMS:
        for f in sorted(glob.glob(f"{B}/results/evals/{m}_{arm}_s*_step60.summary.json")):
            m_ = re.search(r"_s(\d+)_step60", f)
            if m_ is None or "_a3_" in f or "_mx_" in f:
                continue
            seed = int(m_.group(1))
            s = load_summary(f)
            s["hp"] = hp_mass(f"{B}/results/hp/hp_{m}_{arm}_s{seed}_shard0of1.jsonl")
            rows[(m, arm, seed)] = s


def val(s, ds, key="greedy_acc"):
    if ds == "hp":
        return None if not s.get("hp") else s["hp"]["q1"]
    return None if not s.get(ds) else s[ds][key]


print("=" * 110)
print("Direction B readout   (greedy acc / maj@8 acc per dataset; hp = Q1 blind-spot mass on GSM-HP deleted items)")
print("=" * 110)
for m in MODELS:
    print(f"\n## {m}")
    print(f"{'arm':6} {'seed':>4} | " + " | ".join(f"{ds[:14]:>16}" for ds in DS) + " | {:>7}".format("hp Q1"))
    for (mm, arm, seed), s in sorted(rows.items(), key=lambda kv: (kv[0][0], ARMS.index(kv[0][1]) if kv[0][1] in ARMS else -1, kv[0][2])):
        if mm != m:
            continue
        cells = []
        for ds in DS:
            g, mj = val(s, ds), val(s, ds, "maj_acc")
            cells.append(f"{100 * g:6.1f} / {100 * mj:6.1f}" if g is not None else f"{'-':>16}")
        h = val(s, "hp")
        print(f"{arm:6} {seed:>4} | " + " | ".join(cells) + f" | {h:7.3f}" if h is not None else f"{arm:6} {seed:>4} | " + " | ".join(cells) + " |       -")


def exact_test(a, b):
    """Two-sided seed-level exact randomisation test on the mean difference between two lists of seeds."""
    obs = sum(a) / len(a) - sum(b) / len(b)
    allv = a + b
    k = len(a)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), k):
        s_a = [allv[i] for i in idx]
        s_b = [allv[i] for i in range(len(allv)) if i not in idx]
        d = sum(s_a) / k - sum(s_b) / len(s_b)
        tot += 1
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, cnt / tot, tot


print("\n" + "=" * 110)
print("Registered contrasts (points, greedy; seed-level exact test; p-floor = 2/C(2k,k))")
print("=" * 110)
verd = {}
for m in MODELS:
    base = rows.get((m, "base", -1), {})
    per = defaultdict(dict)
    for (mm, arm, seed), s in rows.items():
        if mm == m and arm in ARMS:
            for ds in DS + ["hp"]:
                v = val(s, ds)
                if v is not None:
                    per[(arm, ds)][seed] = v
    for ds in DS + ["hp"]:
        b0 = val(base, ds)
        line = [f"{m:14} {ds:16}", f"base {100 * b0:6.1f}" if b0 is not None else "base      -"]
        for arm in ARMS:
            xs = per[(arm, ds)]
            if xs:
                mean = sum(xs.values()) / len(xs)
                line.append(f"{arm} {100 * mean:6.1f}(n={len(xs)})" + (f" d={100 * (mean - b0):+5.1f}" if b0 is not None else ""))
        print("  ".join(line))
        for a1, a2 in [("own", "perm"), ("own", "fmt"), ("gt", "own"), ("perm", "fmt")]:
            x, y = per[(a1, ds)], per[(a2, ds)]
            common = sorted(set(x) & set(y))
            if len(common) >= 2:
                obs, p, tot = exact_test([x[s] for s in common], [y[s] for s in common])
                print(f"      {a1}-{a2}: {100 * obs:+6.2f} pts  p={p:.3f} (seeds {common}, {tot} assignments)")
                verd[(m, ds, a1, a2)] = (100 * obs, p, len(common))

print("\n" + "=" * 110)
print("Predictions / rules (greedy MATH-500 unless stated)")
print("=" * 110)


def get(m, ds, a1, a2):
    return verd.get((m, ds, a1, a2))


q = get("qwen3_4b_Base", "math500_train", "own", "perm")
l = get("llama3p2_3b", "math500_train", "own", "perm")
if q:
    print(f"P1 Qwen own-perm = {q[0]:+.2f} (predicted within [-2,+2]) -> {'as predicted' if abs(q[0]) <= 2 else 'NOT as predicted'} (p={q[1]:.3f}, {q[2]} seeds)")
if l:
    print(f"P2 Llama own-perm = {l[0]:+.2f} (predicted >= +3) -> {'as predicted' if l[0] >= 3 else 'NOT as predicted'} (p={l[1]:.3f}, {l[2]} seeds)")
for m in MODELS:
    g = get(m, "math500_train", "gt", "own")
    if g:
        print(f"P3 {m} gt-own = {g[0]:+.2f} -> {'as predicted' if g[0] >= 0 else 'NOT as predicted'}")
    a, b = get(m, "math500_train", "own", "perm"), get(m, "math_heldout1000", "own", "perm")
    if a and b:
        print(f"P4 {m} sign(own-perm) train {a[0]:+.2f} vs heldout {b[0]:+.2f} -> {'agree' if (a[0] >= 0) == (b[0] >= 0) else 'DISAGREE'}")
hq = rows.get(("qwen3_4b_Base", "base", -1), {}).get("hp")
if hq:
    own_hp = [val(s, "hp") for (mm, arm, sd), s in rows.items() if mm == "qwen3_4b_Base" and arm == "own" and val(s, "hp") is not None]
    perm_hp = [val(s, "hp") for (mm, arm, sd), s in rows.items() if mm == "qwen3_4b_Base" and arm == "perm" and val(s, "hp") is not None]
    if own_hp:
        print(f"P5 Qwen Q1: base {hq['q1']:.3f} -> own {sum(own_hp) / len(own_hp):.3f}" + (f", perm {sum(perm_hp) / len(perm_hp):.3f}" if perm_hp else "")
              + f" -> {'as predicted' if sum(own_hp) / len(own_hp) - hq['q1'] >= 0.05 else 'NOT as predicted'}")
print("KB1/KB2/KB3 are read from the contrasts above once >= 3 seeds are in for both models.")
