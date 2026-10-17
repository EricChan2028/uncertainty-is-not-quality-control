"""A3 readout per lineage and round (registration A3 + amendments A3a-A3d):
  guard: proposals admitted, deleted/intact tracer admission (ratio), confident shares
  audit: ill-posed / ambiguous share of admitted proposer questions (Opus panel)
  solver after the round: greedy MATH-500 / held-out / GSM8K (lenient extraction = primary per A3d; strict in brackets),
  boxed-format rate on MATH-500, no-op control acc + abstention, GSM-HP Q1 on the 412 judge-validated unsolvable twins,
  abstention under the abstention prompt.
  python3 a3_analyze.py
"""
import glob
import json
import os
import re

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
BASE_TAG = {"qwen3_4b_Base": "qwen3_4b_base", "llama3p2_3b": "llama3p2_3b", "qwen3_8b_Base": "qwen3_8b_base"}


def summ(tag):
    """Per-dataset summary dirs first, then the flat file (the v4 re-evaluation writes the flat file, so it wins)."""
    out = {}
    for p in sorted(glob.glob(f"{B}/results/evals/*/{tag}.summary.json")) + [f"{B}/results/evals/{tag}.summary.json"]:
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def acc(s, ds):
    d = s.get(ds, {})
    return d.get("greedy_acc_lenient", d.get("greedy_acc"))


def q1u(path):
    if not os.path.exists(path):
        return None
    d = [json.loads(l) for l in open(path)]
    dele = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    inta = [r for r in d if r["form"] == "intact"]
    if not dele:
        return None
    return {"q1": sum(1 for r in dele if r["p_mode"] >= 0.5) / len(dele), "abs_uns": sum(r.get("abstain", 0) for r in dele) / len(dele),
            "abs_int": sum(r.get("abstain", 0) for r in inta) / max(1, len(inta)), "int_acc": sum(r["correct"] for r in inta) / max(1, len(inta))}


def base_q1(tag):
    files = glob.glob(f"{R}/dirA/results/{tag}_shard*of8.jsonl") or glob.glob(f"{R}/dirA/results/{tag}_shard*.jsonl")
    if not files:
        return None
    d = [json.loads(l) for f in files for l in open(f)]
    dele = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(1 for r in dele if r["p_mode"] >= 0.5) / max(1, len(dele))


f1 = lambda x, m=100, w=5, d=1: (f"{x * m:{w}.{d}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-")


def acc3(s):
    """'MATH lenient(strict) | held | GSM | boxed%' for a summary dict."""
    parts = []
    for ds, lab in (("math500_train", "MATH"), ("math_heldout1000", "held"), ("gsm8k_test500", "GSM")):
        d = s.get(ds, {})
        le, st = d.get("greedy_acc_lenient"), d.get("greedy_acc")
        parts.append(f"{lab} {f1(le if le is not None else st)}({f1(st, 100, 4, 0) if le is not None else ' v3 '})")
    bx = s.get("math500_train", {}).get("greedy_boxed_rate")
    parts.append(f"boxed {f1(bx, 100, 3, 0)}%")
    return " | ".join(parts)


print("=" * 170)
print("A3 loop readout (A3d: accuracies are lenient-extraction greedy; strict boxed-only in brackets; 'v3' = not yet re-evaluated).")
print("guard: admit% of proposals | deleted vs intact tracer admit% (ratio) | confident% del/int || audit: unsolvable% + ambiguous% of admitted proposals")
print("solver: MATH-500 | held-out | GSM8K | boxed rate on MATH | no-op acc, abstain% (abs prompt) | Q1u plain | Q1u abs-prompt, abstain% uns/int")
print("=" * 170)
lineages = sorted({re.sub(r"_r\d+$", "", os.path.basename(d.rstrip("/"))) for d in glob.glob(f"{B}/runs/*_a3_s*_r*/") if not d.rstrip("/").endswith(".claim")})
for L in lineages:
    model = L.split("_own_")[0].split("_pdc_")[0]
    bt = BASE_TAG.get(model, model.lower())
    b = summ(f"{bt}_step0")
    print(f"\n## {L}   base: {acc3(b)}  Q1u {f1(base_q1(bt), 1, 5, 3)}")
    for rd in (1, 2, 3, 4, 5, 6):
        d = f"{B}/runs/{L}_r{rd}"
        if not os.path.isdir(d):
            continue
        g = json.load(open(f"{d}/guard_stats.json")) if os.path.exists(f"{d}/guard_stats.json") else {}
        a = json.load(open(f"{d}/audit_admitted.json")) if os.path.exists(f"{d}/audit_admitted.json") else {}
        ac = a.get("counts", {})
        an = max(1, sum(ac.values())) if ac else 0
        s = summ(f"{L}_r{rd}_step60")
        sa = summ(f"{L}_r{rd}_step60_abs")
        h = q1u(f"{B}/results/hp/hp_{L}_r{rd}_shard0of1.jsonl")
        ha = q1u(f"{B}/results/hp/hpabs_{L}_r{rd}_shard0of1.jsonl")
        ratio = (g.get("tracer_deleted_admit_rate", 0) / g["tracer_intact_admit_rate"]) if g.get("tracer_intact_admit_rate") else None
        gl = (f"admit {f1(g.get('proposed_admit_rate'))}% | del {f1(g.get('tracer_deleted_admit_rate'))} vs int {f1(g.get('tracer_intact_admit_rate'))} ({f1(ratio, 1, 4, 2)}) | conf {f1(g.get('tracer_deleted_confident_rate'))}/{f1(g.get('tracer_intact_confident_rate'))}") if g else "guard pending"
        al = (f"audit uns {100 * ac.get('UNSOLVABLE', 0) / an:4.1f}% amb {100 * ac.get('AMBIGUOUS', 0) / an:4.1f}% (n={an})") if ac else "audit pending"
        st = {}
        for sub in ("admitted_sub_stats.json", "admitted_clean_stats.json", "admitted_dirty_stats.json"):
            if os.path.exists(f"{d}/{sub}"):
                st = json.load(open(f"{d}/{sub}"))
        tl = f" | train set {st['n_kept']}/{st['n_admitted']} ({st['mode']})" if st else ""
        noop = sa.get("hp_noop500", {})
        sl = (f"{acc3(s)} | noop {f1(noop.get('greedy_acc'))},{f1(noop.get('abstain_greedy'), 100, 4, 0)}%"
              f" | Q1u {f1(h['q1'], 1, 5, 3) if h else '    -'} | Q1u-abs {f1(ha['q1'], 1, 5, 3) if ha else '    -'} abs {f1(ha['abs_uns']) if ha else '   -'}/{f1(ha['abs_int']) if ha else '   -'}") if (s or h) else "solver eval pending"
        print(f"  r{rd}: {gl} || {al}{tl}\n      {sl}")
