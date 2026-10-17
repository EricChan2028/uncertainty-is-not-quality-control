"""rev6 stop-sequence check (exploratory): per checkpoint, SC / numeric share on U / accuracy on A under
  ORIG-rev6   the probe's own answers on the rev6 texts (no stop)
  V3-rev6     strict_v3.read on the rev6 texts (the registered strict reading: cut at a new Question: line)
  V4-rev6     the same cut, but a cut solution counts as finished ('stop'), as generation with the stop would finish it
  STOP        strict() on the texts generated with the stop sequence (paper_arr/rev6/texts_stop), and the probe's own answers
python3 stop_compare.py [--pred]  (--pred: rev6 columns only, before the stop runs exist)"""
import json, os, sys
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; D6 = f"{R}/paper_arr/rev6"
sys.path.insert(0, D6); import strict_v3  # noqa: E402
TAGS = ["qwen3_4b_base", "qwen3_4b_Base_own_a3_s0_r3", "qwen3_4b_Base_own_a3_s1_r3", "base_Qwen3-8B-Base",
        "qwen3_8b_Base_own_a3_s0_r3", "qwen3_8b_Base_own_a3_s1_r3", "qwen3_8b_Base_own_a3_s2_r3", "base_Phi-4-mini-instruct"]
def load(p):
    if not os.path.exists(p): return None
    rows = {}
    for l in open(p):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    return rows if len(rows) >= 1200 else None
def v4(t, f):
    tt = strict_v3.cut(t)
    if tt != t: f = "stop"
    s = strict_v3.strict(tt, f)
    if s is None: return None
    o = strict_v3.original(tt, f); return o if o is not None else s
def stats(rows, fn):
    sc = acc = num = n = 0
    for r in rows.values():
        ans = [fn(r, i) for i in range(len(r["texts"]))]
        mode, k = Counter(ans).most_common(1)[0]
        if r["answerable"]: acc += mode is not None and r["ref"] is not None and mode == r["ref"]
        else:
            sc += mode is not None and k / len(ans) >= 0.5; num += sum(a is not None for a in ans); n += len(ans)
    return f"SC {100 * sc / 600:5.1f} NUM_U {100 * num / n:5.1f} ACC {100 * acc / 600:5.1f}"
pred = "--pred" in sys.argv
for tag in TAGS:
    a = load(f"{D6}/texts/plain_{tag}.jsonl")
    out = [f"ORIG-rev6 {stats(a, lambda r, i: r['answers'][i])}",
           f"V3-rev6 {stats(a, lambda r, i: strict_v3.read(r['texts'][i], r['finish'][i]))}",
           f"V4-rev6 {stats(a, lambda r, i: v4(r['texts'][i], r['finish'][i]))}"]
    if not pred:
        b = load(f"{D6}/texts_stop/stopq_{tag}.jsonl")
        out += ["STOP not complete"] if b is None else [f"STOP-strict {stats(b, lambda r, i: strict_v3.strict(r['texts'][i], r['finish'][i]))}",
                                                        f"STOP-probe {stats(b, lambda r, i: r['answers'][i])}"]
    print(f"{tag:30s} | " + " | ".join(out))
