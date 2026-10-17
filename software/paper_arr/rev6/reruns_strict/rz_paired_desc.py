# rev6 rerun copy of dirA5/code/rz_paired_desc.py (UMWP dir -> dumps_strict, 1 occurrence(s); output -> reruns_strict)
"""POST-HOC DESCRIPTIVE (not registered; cannot change RZ-P1 / RZ-P2). Paired comparison of the R-Zero final solver and the
base model on the same UMWP items, with the CA and ACC_A definitions of rz_read.py / umwp_read.py:
  confident answer on an unanswerable item = mode is not None and p_mode >= 0.5;  correct on an answerable item = correct.
Exact two-sided McNemar on discordant items and a 10,000-draw item bootstrap 95% CI of the difference (seed 0).
v2 (26 Sep): adds, per dump, the mean per-item non-answer rate ("nonanswer" field), the share of items with no modal answer
and the mean modal-answer share, separately for answerable (A) and unanswerable (U) items; the paired results are unchanged.
    python3 dirA5/code/rz_paired_desc.py"""
import json, math, random

U = "/home/tione/notebook/rsi_rl/paper_arr/rev6/dumps_strict"


def load(tag):
    rows = {}
    for l in open(f"{U}/{tag}.jsonl", errors="replace"):
        try:
            r = json.loads(l); rows[r["id"]] = r
        except (ValueError, KeyError, TypeError):
            pass
    assert len(rows) >= 1200, (tag, len(rows))
    return rows


def conf(r):
    return int(r["mode"] is not None and r["p_mode"] >= 0.5)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(base, final, answerable, f):
    ids = sorted(i for i in base if base[i]["answerable"] == answerable and i in final)
    x = [f(base[i]) for i in ids]; y = [f(final[i]) for i in ids]
    n = len(ids); d = (sum(y) - sum(x)) / n
    b = sum(1 for a, z in zip(x, y) if a == 0 and z == 1); c = sum(1 for a, z in zip(x, y) if a == 1 and z == 0)
    rng = random.Random(0); bs = []
    for _ in range(10000):
        s = [rng.randrange(n) for _ in range(n)]
        bs.append(sum(y[j] - x[j] for j in s) / n)
    bs.sort()
    return {"n": n, "base": round(sum(x) / n, 4), "final": round(sum(y) / n, 4), "diff": round(d, 4),
            "ci95": [round(bs[249], 4), round(bs[9749], 4)], "base0_final1": b, "base1_final0": c,
            "mcnemar_p": float(f"{mcnemar(b, c):.3g}")}


out = {"note": "post-hoc descriptive, not registered; cannot change RZ-P1 or RZ-P2"}
for cond, btag in (("plain", "plain_qwen3_4b_base"), ("plainchat", "plainchat_qwen3_4b_base")):
    base = load(btag)
    for S in (0, 1):
        final = load(f"{cond}_rzero_s{S}_it3")
        out[f"{cond}_s{S}"] = {"CA_unanswerable": paired(base, final, False, conf),
                               "ACC_answerable": paired(base, final, True, lambda r: int(r["correct"]))}


def rates(rows):
    o = {}
    for lab, a in (("A", True), ("U", False)):
        s = [r for r in rows.values() if r["answerable"] == a]; m = [r for r in s if r["mode"] is not None]
        o[lab] = {"n": len(s), "nonanswer": round(sum(r["nonanswer"] for r in s) / len(s), 4),
                  "no_mode": round((len(s) - len(m)) / len(s), 4), "mean_p_mode": round(sum(r["p_mode"] for r in m) / max(1, len(m)), 4)}
    return o


out["response_rates"] = {t: rates(load(t)) for t in ("plain_qwen3_4b_base", "plain_rzero_s0_it3", "plain_rzero_s1_it3",
                                                      "plainchat_qwen3_4b_base", "plainchat_rzero_s0_it3", "plainchat_rzero_s1_it3")}
print(json.dumps(out, indent=1))
