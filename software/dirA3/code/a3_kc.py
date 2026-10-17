"""Amendment A3c readout: per model and seed, the own / own_clean / own_sub / own_dirty lineages round by round
(mean of greedy MATH-500, held-out-1000, GSM8K; MATH-500 sample truncation), the collapse rule, and the KC1-KC3 reads.
Collapse (registered): end-of-round accuracy below half the base mean, or truncation > 0.5 (sample_trunc on MATH-500);
a collapsed lineage is read at the previous round (round 0 = base).  python3 a3_kc.py
"""
import glob
import json
import os

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
MODELS = {"qwen3_4b_Base": "qwen3_4b_base", "llama3p2_3b": "llama3p2_3b", "qwen3_8b_Base": "qwen3_8b_base"}
DS = ("math500_train", "math_heldout1000", "gsm8k_test500")
ARMS = ("own", "own_clean", "own_sub", "own_dirty", "own_tg", "own_azr")
ROUNDS = (1, 2, 3, 4, 5, 6)


def summ(tag):
    """Per-dataset summary dirs first, then the flat file (the A3d v4 re-evaluation writes the flat file, so it wins)."""
    out = {}
    for p in sorted(glob.glob(f"{B}/results/evals/*/{tag}.summary.json")) + [f"{B}/results/evals/{tag}.summary.json"]:
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


def acc(s, strict=False):
    """Mean greedy accuracy over the three sets; A3d: lenient extraction is primary (falls back to strict if not re-evaluated)."""
    if not s or any(d not in s for d in DS):
        return None
    key = "greedy_acc" if strict else "greedy_acc_lenient"
    return 100 * sum(s[d].get(key, s[d]["greedy_acc"]) for d in DS) / len(DS)


def lineage(model, arm, seed):
    """rounds -> (mean acc, trunc, collapsed); read value per collapse rule."""
    base = acc(summ(f"{MODELS[model]}_step0"))
    rows, read, read_rd = [], base, 0
    for rd in ROUNDS:
        s = summ(f"{model}_{arm}_a3_s{seed}_r{rd}_step60")
        a = acc(s)
        if a is None:
            rows.append((rd, None, None, None))
            break
        tr = s["math500_train"].get("sample_trunc")
        col = (base is not None and a < 0.5 * base) or (tr is not None and tr > 0.5)
        rows.append((rd, a, tr, col))
        if col:
            break
        if rd <= 3:
            read, read_rd = a, rd      # the registered KC read is round 3 (or the last non-collapsed round before it)
    r3 = [r for r in rows if r[0] == 3]
    done3 = bool(r3) and r3[0][1] is not None and not r3[0][3]
    collapsed = any(r[3] for r in rows if r[0] <= 3)
    return base, rows, read, read_rd, done3 or collapsed


def f(x, w=5, d=1):
    return f"{x:{w}.{d}f}" if isinstance(x, (int, float)) else " " * (w - 1) + "-"


for model in MODELS:
    if not glob.glob(f"{B}/runs/{model}_own_a3_s*_r1/"):
        continue
    print("=" * 120)
    print(f"{model}: mean greedy acc (MATH-500, held-out, GSM8K) per round; trunc = MATH-500 sample truncation; C = collapsed")
    reads = {}
    for seed in (0, 1, 2, 3, 4):
        for arm in ARMS:
            if not os.path.isdir(f"{B}/runs/{model}_{arm}_a3_s{seed}_r1"):
                continue
            base, rows, read, read_rd, final = lineage(model, arm, seed)
            cells = " | ".join(f"r{rd} {f(a)} t{f(tr, 4, 2)}{'C' if col else ' '}" for rd, a, tr, col in rows)
            print(f"  s{seed} {arm:10s} base {f(base)} | {cells}   -> read r{read_rd} {f(read)}{' (final)' if final else ' (pending)'}")
            reads.setdefault(seed, {})[arm] = (read, read_rd, final, base)
    # KC reads on seeds 0,1 once own, own_clean and own_sub are final
    verdict = []
    for seed in (0, 1):
        r = reads.get(seed, {})
        if all(a in r and r[a][2] for a in ("own", "own_clean", "own_sub")):
            dc, ds = r["own_clean"][0] - r["own"][0], r["own_sub"][0] - r["own"][0]
            verdict.append((seed, dc, ds, r["own"][1], r["own_clean"][1], r["own_sub"][1]))
    if len(verdict) == 2:
        kc1 = all(dc >= 2 and abs(ds) < 2 for _, dc, ds, *_ in verdict)
        kc2 = all(abs(dc) < 2 and abs(ds) < 2 for _, dc, ds, *_ in verdict)
        kc3 = any(abs(ds) >= 2 and abs(ds) >= abs(dc) for _, dc, ds, *_ in verdict)
        own_col = [s for s, *_, ro, rc, rs in verdict if ro < 3 and rc == 3]
        for seed, dc, ds, ro, rc, rs in verdict:
            print(f"  seed {seed}: clean-own {dc:+.1f}  sub-own {ds:+.1f}  (read rounds own r{ro}, clean r{rc}, sub r{rs})")
        print(f"  KC1 (clean>own by >=2 both seeds, |sub-own|<2): {kc1}   KC2 (all within 2): {kc2}   KC3 (sub moves as much as clean): {kc3}")
        if own_col:
            print(f"  own collapsed but own_clean reached round 3 at seed(s) {own_col}: evidence for KC1 regardless of margins")
    else:
        print(f"  KC reads pending ({len(verdict)}/2 seeds final)")
    for seed in (0, 1):
        r = reads.get(seed, {})
        if "own_dirty" in r and "own" in r and r["own_dirty"][2] is not None:
            d = summ(f"{model}_own_dirty_a3_s{seed}_r1_step60")
            o = summ(f"{model}_own_a3_s{seed}_r1_step60")
            if acc(d) is not None and acc(o) is not None:
                print(f"  seed {seed} round 1 descriptive: dirty {acc(d):.1f} vs own {acc(o):.1f} (base {f(r['own'][3])})")
