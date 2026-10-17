"""Diagnostic: why does the strict reading reject Qwen3-8B-Base answers? Print the last Answer: line of samples that the probe
parses as a number but strict() rejects, by branch, for answerable questions (where the reference is known)."""
import json, os, re, sys
from collections import Counter
D6 = "/home/tione/notebook/rsi_rl/paper_arr/rev6"; os.chdir(D6)
sys.argv = ["rev6_read.py"]
src = open("rev6_read.py").read(); ns = {"__file__": os.path.abspath("rev6_read.py")}
exec(compile(src[:src.index('Q4 = "qwen3_4b_Base_%s_s%d_r%d"')], "rev6_read_defs", "exec"), ns)
tag = sys.argv[1] if len(sys.argv) > 1 else "plain_base_Qwen3-8B-Base"
for tag in ("plain_base_Qwen3-8B-Base", "plain_qwen3_8b_Base_own_a3_s0_r3", "plain_qwen3_4b_base"):
    rows = [json.loads(l) for l in open(f"texts/{tag}.jsonl")]
    br = Counter(); ex = {}
    for r in rows:
        if not r["answerable"]: continue
        for t, f, a in zip(r["texts"], r["finish"], r["answers"]):
            if a is None: continue
            w = ns["why_none"](t, f)
            if w != "numeric":
                br[w] += 1
                lines = re.findall(r"Answer:\s*([^\n]*)", t)
                ex.setdefault(w, []).append((a, r["ref"], (lines[-1] if lines else t[-150:])[:110]))
    print("==", tag, "answerable samples parsed numeric by the probe but rejected by strict():", dict(br))
    for w, lst in ex.items():
        for a, ref, line in lst[:6]: print(f"   [{w}] probe={a} ref={ref} :: {line!r}")
