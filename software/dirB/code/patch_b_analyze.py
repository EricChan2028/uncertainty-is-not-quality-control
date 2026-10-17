"""Skip non-Direction-B summaries (A3 rounds, A2 mixed runs) in b_analyze.py's per-arm glob."""
p = "/home/tione/notebook/rsi_rl/dirB/code/b_analyze.py"
s = open(p).read()
old = '            seed = int(re.search(r"_s(\\d+)_step60", f).group(1))\n'
assert old in s, "anchor not found"
new = ('            m_ = re.search(r"_s(\\d+)_step60", f)\n'
       '            if m_ is None or "_a3_" in f or "_mx_" in f:\n'
       '                continue\n'
       '            seed = int(m_.group(1))\n')
s = s.replace(old, new, 1)
open(p, "w", newline="\n").write(s)
print("patched b_analyze.py")
