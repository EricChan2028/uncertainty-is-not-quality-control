"""rev6 interim look (descriptive, not the read): per completed plain dump, the frozen rev6_read.py metric code under both
parsers. The registered read (rev6_read.py, run once when all runs are done) supersedes these numbers."""
import os, sys
D6 = "/home/tione/notebook/rsi_rl/paper_arr/rev6"; os.chdir(D6)
sys.argv = ["rev6_read.py"]
src = open("rev6_read.py").read(); ns = {"__file__": os.path.abspath("rev6_read.py")}
exec(compile(src[:src.index('Q4 = "qwen3_4b_Base_%s_s%d_r%d"')], "rev6_read_defs", "exec"), ns)
for kind in ("plain", "absfix"):
    for f in sorted(os.listdir("texts")):
        if not f.startswith(kind + "_") or f.startswith("plain_chat_"): continue
        rows = ns["load"]("texts/" + f)
        if rows is None: continue
        out = []
        for P in ("ORIGINAL", "STRICT"):
            ans = ns["parsed"](rows, P, kind); m, _ = ns["metrics"](rows, ans, kind, P)
            s = f"{P[:4]} SC {100 * m['SC']:5.1f} SC6 {100 * m['SC_6']:5.1f} SC8 {100 * m['SC_8']:4.1f} NUM_U {100 * m['NUM_U']:5.1f} ACC {100 * m['ACC_A']:5.1f}"
            if kind == "absfix": s += f" ABS_U {100 * m['ABS_U']:5.1f} ABS_A {100 * m['ABS_A']:4.1f}"
            if P == "STRICT" and kind == "plain":
                b = m["U_samples_by_branch"]; s += f" decl {100 * b.get('decline', 0):4.1f} len {100 * b.get('length', 0):4.1f} sym {100 * b.get('symbolic', 0):3.1f} nonum {100 * b.get('no_number', 0):3.1f}"
            out.append(s)
        print(f"{f[:-6]:44s}", " | ".join(out))
