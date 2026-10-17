"""Patch a3_guard.py: add --guard abstain (A4 addition 1). Writes .new and os.replace (never rewrites a running script in place)."""
import os
p = "/home/tione/notebook/rsi_rl/dirA3/code/a3_guard.py"
s = open(p).read()
old1 = 'ap.add_argument("--guard", choices=["rzero", "azr", "tracer"], default="rzero",'
new1 = 'ap.add_argument("--guard", choices=["rzero", "azr", "tracer", "abstain"], default="rzero",'
assert old1 in s, "guard choices line not found"
s = s.replace(old1, new1)
old2 = 'ap.add_argument("--abstain_prompt", type=int, default=0,'
new2 = ('ap.add_argument("--abs_max", type=float, default=0.25, help="A4 abstention guard: reject a candidate when >= this share of its G samples abstain (needs --abstain_prompt 1)")\n'
        'ap.add_argument("--abstain_prompt", type=int, default=0,')
assert old2 in s
s = s.replace(old2, new2)
old3 = '    admit = lo <= p_mode <= hi\n'
new3 = '    admit = (lo <= p_mode <= hi) and (args.guard != "abstain" or abst < args.abs_max)   # A4: abstention guard\n'
assert old3 in s
s = s.replace(old3, new3)
old4 = '    stats[f"{x[\'kind\']}_confident"] += int(p_mode >= 0.5)\n'
new4 = old4 + '    stats[f"{x[\'kind\']}_abstained"] += int(abst >= args.abs_max)\n'
assert old4 in s
s = s.replace(old4, new4)
# abstain guard requires the abstention prompt
old5 = 'lo, hi = args.lo, args.hi\n'
new5 = 'lo, hi = args.lo, args.hi\nif args.guard == "abstain" and not args.abstain_prompt:\n    raise SystemExit("--guard abstain requires --abstain_prompt 1")\n'
assert old5 in s
s = s.replace(old5, new5, 1)
open(p + ".new", "w").write(s)
os.replace(p + ".new", p)
print("patched", p)
