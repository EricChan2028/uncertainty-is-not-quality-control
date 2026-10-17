import os
p = "/home/tione/notebook/rsi_rl/dirA3/code/a3_guard_diag_score.py"
s = open(p).read()
old = 'D = f"{R}/dirA3/results/guard_diag/qwen_s0"'
assert old in s
s = s.replace(old, 'SUB = sys.argv[1] if len(sys.argv) > 1 else "qwen_s0"\nD = f"{R}/dirA3/results/guard_diag/{SUB}"')
s = s.replace("import glob, json, os\n", "import glob, json, os, sys\n")
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("scorer patched")
