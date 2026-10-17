# rev6 rerun copy of paper_arr/analyses/rev3/ca_thresholds.py (UMWP dir -> dumps_original, 1 occurrence(s); output -> reruns_original)
"""[EXP, robustness, 2026-10-04] Self-consistent answering on UMWP unanswerable questions at stricter thresholds.

The paper's rate (CA) counts a question when the most frequent of 8 answers is a number held by >= 4 samples, so a 4-4 tie
counts. Here: the same rate, the rate without ties at the top, and modal counts >= 5, 6, 7 and 8 (unanimous).
Inputs: dirA4/results/umwp/plain_<tag>.jsonl (field "answers": the 8 parsed answers, None if unparseable).
python3 ca_thresholds.py
"""
import json
import os
from collections import Counter

R = "/home/tione/notebook/rsi_rl"
D = f"{R}/paper_arr/rev6/dumps_original"
GROUPS = {
    "base": ["qwen3_4b_base"],
    "own_r3": [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)],
    "solvable_only_r3": [f"qwen3_4b_Base_own_clean_a3_s{s}_r3" for s in range(4)],
    "own_r5": [f"qwen3_4b_Base_own_a3_s{s}_r5" for s in range(3)],
}


def load(tag):
    f = f"{D}/plain_{tag}.jsonl"
    rows = {}
    for line in open(f):
        try:
            r = json.loads(line)
            rows[r["id"]] = r
        except ValueError:
            pass
    assert len(rows) >= 1200, (tag, len(rows))
    return [r for r in rows.values() if not r["answerable"]]


def rates(rows):
    n = len(rows)
    out = {"paper_CA": 0, "no_tie_at_top": 0, ">=5": 0, ">=6": 0, ">=7": 0, "8of8": 0, "ties_among_CA": 0}
    for r in rows:
        ans = r["answers"]
        c = Counter(ans)
        mode, k = c.most_common(1)[0]
        num = Counter(a for a in ans if a is not None)
        top = num.most_common(2)
        n1 = top[0][1] if top else 0
        n2 = top[1][1] if len(top) > 1 else 0
        ca = mode is not None and k >= 4
        out["paper_CA"] += ca
        if ca and n1 == 4 and (n2 == 4 or c.get(None, 0) == 4):
            out["ties_among_CA"] += 1
        out["no_tie_at_top"] += ca and not (n1 == 4 and (n2 == 4 or c.get(None, 0) == 4))
        out[">=5"] += n1 >= 5
        out[">=6"] += n1 >= 6
        out[">=7"] += n1 >= 7
        out["8of8"] += n1 == 8
    return {k: round(100.0 * v / n, 1) for k, v in out.items()}


res = {}
for g, tags in GROUPS.items():
    per = {}
    for t in tags:
        if os.path.exists(f"{D}/plain_{t}.jsonl"):
            per[t] = rates(load(t))
    res[g] = per
    keys = list(next(iter(per.values())).keys())
    rng = {k: [min(v[k] for v in per.values()), max(v[k] for v in per.values())] for k in keys}
    print(g, json.dumps(rng))
json.dump(res, open(f"{R}/paper_arr/rev6/reruns_original/ca_thresholds.json", "w"), indent=1)
print("saved")
