"""Direction A2 readout (registration A2 + amendment A2a). Per model x arm x seed:
  E1  greedy accuracy on MATH-500 / MATH-heldout / GSM8K, plain prompt and with the abstention option
  E2  GSM-HP (abstention prompt): Q1 on the 412 judge-validated UNSOLVABLE twins, abstention rate on
      unsolvable / solvable-different / intact, intact accuracy; plus the no-op insertion control
Then the pre-registered contrasts (pdc vs own, pdc vs own_abs) with the seed-level exact test.
  python3 a2_analyze.py
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
ARMS = ["own", "own_abs", "pdc"]
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
verdict = {(str(v["fam"]), int(v["task"])): v["verdict"] for v in V["verdicts"]}


def summ(tag):
    """Merge <evals>/<tag>.summary.json with the per-dataset <evals>/<ds>/<tag>.summary.json files (base evals)."""
    out = {}
    for p in [f"{B}/results/evals/{tag}.summary.json"] + glob.glob(f"{B}/results/evals/*/{tag}.summary.json"):
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def hp(path):
    if not os.path.exists(path):
        return None
    d = [json.loads(l) for l in open(path)]
    dele = [r for r in d if r["form"] == "deleted"]
    inta = [r for r in d if r["form"] == "intact"]
    uns = [r for r in dele if verdict.get((str(r["fam"]), int(r["task"]))) == "UNSOLVABLE"]
    sdf = [r for r in dele if verdict.get((str(r["fam"]), int(r["task"]))) == "SOLVABLE_DIFFERENT"]
    f = lambda xs, k: (sum(r.get(k, 0.0) for r in xs) / len(xs)) if xs else float("nan")
    return {"q1_uns": sum(1 for r in uns if r["p_mode"] >= 0.5) / max(1, len(uns)), "n_uns": len(uns),
            "abs_uns": f(uns, "abstain"), "abs_sdf": f(sdf, "abstain"), "abs_int": f(inta, "abstain"),
            "acc_int": f(inta, "correct"), "acc_sdf_answer": f(sdf, "correct")}


rows = {}
for m, bt in MODELS.items():
    base = {"plain": summ(f"{bt}_step0"), "abs": summ(f"{bt}_step0_abs"), "hp": hp(f"{B}/results/hp/hpabs_{bt}_shard0of1.jsonl"),
            "hp_plain": hp(f"{R}/dirA/results/{bt}_shard0of8.jsonl")}
    # base plain hp from the 8 shards
    files = glob.glob(f"{R}/dirA/results/{bt}_shard*of8.jsonl")
    if files:
        tmp = f"/tmp/_base_{bt}.jsonl"
        with open(tmp, "w") as fh:
            for f_ in files:
                fh.write(open(f_).read())
        base["hp_plain"] = hp(tmp)
    rows[(m, "base", -1)] = base
    for arm in ARMS:
        for f in sorted(glob.glob(f"{B}/results/evals/{m}_{arm}_mx_s*_step60.summary.json")):
            seed = int(re.search(r"_s(\d+)_step60", f).group(1))
            run = f"{m}_{arm}_mx_s{seed}"
            rows[(m, arm, seed)] = {"plain": summ(f"{run}_step60"), "abs": summ(f"{run}_step60_abs"),
                                    "hp": hp(f"{B}/results/hp/hpabs_{run}_shard0of1.jsonl"), "hp_plain": hp(f"{B}/results/hp/hp_{run}_shard0of1.jsonl")}


def g(s, ds, key="greedy_acc"):
    return s.get(ds, {}).get(key)


def fmt(x, mul=100, w=6, d=1):
    return f"{x * mul:{w}.{d}f}" if isinstance(x, (int, float)) and x == x else " " * (w - 1) + "-"


print("=" * 150)
print("A2 readout.  E1: greedy acc plain | with abstention option (abstain% greedy).  E2 (abstention prompt): Q1 on 412 unsolvable | abstain% unsolvable / solvable-diff / intact | intact acc | no-op: acc, abstain%")
print("=" * 150)
for m in MODELS:
    print(f"\n## {m}")
    for (mm, arm, seed), s in sorted(rows.items(), key=lambda kv: (kv[0][0], (ARMS.index(kv[0][1]) if kv[0][1] in ARMS else -1), kv[0][2])):
        if mm != m:
            continue
        e1 = " | ".join(f"{ds[:8]} {fmt(g(s['plain'], ds))}/{fmt(g(s['abs'], ds))}({fmt(g(s['abs'], ds, 'abstain_greedy'), w=4, d=0)})" for ds in DS)
        h = s["hp"] or {}
        noop = s["abs"].get("hp_noop500", {})
        e2 = (f"Q1u {fmt(h.get('q1_uns'), w=5)} | abs {fmt(h.get('abs_uns'), w=5)}/{fmt(h.get('abs_sdf'), w=5)}/{fmt(h.get('abs_int'), w=5)} | intacc {fmt(h.get('acc_int'), w=5)}"
              f" | noop {fmt(noop.get('greedy_acc'), w=5)},{fmt(noop.get('abstain_greedy'), w=4, d=0)}") if h or noop else "(no hp/noop yet)"
        print(f"{arm:8}{seed:>3} | {e1} || {e2}")


def exact(a, b):
    obs = sum(a) / len(a) - sum(b) / len(b)
    allv, k = a + b, len(a)
    cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), k):
        sa = [allv[i] for i in idx]
        sb = [allv[i] for i in range(len(allv)) if i not in idx]
        tot += 1
        cnt += abs(sum(sa) / k - sum(sb) / len(sb)) >= abs(obs) - 1e-12
    return obs, cnt / tot


print("\n" + "=" * 150)
print("Registered contrasts (mean over seeds; seed-level exact test)")
print("=" * 150)
for m in MODELS:
    per = defaultdict(dict)
    for (mm, arm, seed), s in rows.items():
        if mm != m or arm not in ARMS:
            continue
        for ds in DS:
            v = g(s["plain"], ds)
            if v is not None:
                per[(arm, f"{ds}:plain")][seed] = v
            v = g(s["abs"], ds)
            if v is not None:
                per[(arm, f"{ds}:abs")][seed] = v
        if s["hp"]:
            for k in ("q1_uns", "abs_uns", "abs_sdf", "abs_int", "acc_int"):
                per[(arm, f"hp:{k}")][seed] = s["hp"][k]
        noop = s["abs"].get("hp_noop500", {})
        if noop:
            per[(arm, "noop:acc")][seed] = noop["greedy_acc"]
            per[(arm, "noop:abstain")][seed] = noop["abstain_greedy"]
    keys = sorted({k for (_, k) in per})
    for k in keys:
        line = [f"{m:14} {k:22}"]
        for arm in ARMS:
            xs = per[(arm, k)]
            if xs:
                line.append(f"{arm:7} {100 * sum(xs.values()) / len(xs):6.1f}(n={len(xs)})")
        for a1, a2 in (("pdc", "own"), ("pdc", "own_abs")):
            x, y = per[(a1, k)], per[(a2, k)]
            common = sorted(set(x) & set(y))
            if len(common) >= 2:
                obs, p = exact([x[s] for s in common], [y[s] for s in common])
                line.append(f"{a1}-{a2} {100 * obs:+6.2f} p={p:.3f}")
        print("  ".join(line))
