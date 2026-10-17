"""Amendment A3e: add --guard {rzero,azr,tracer} to a3_guard.py.
  rzero : the registered band [lo, hi] on p_mode (default 0.25-0.75)
  azr   : learnability band (0, 1) exclusive - admit anything the solver neither always nor never solves
  tracer: tracer-calibrated band - choose [lo, hi] on a grid to maximise intact-tracer admission minus
          deleted-tracer admission (intact admission >= 0.30), then apply that band to the proposals.
"""
p = "/home/tione/notebook/rsi_rl/dirA3/code/a3_guard.py"
s = open(p).read()
assert "--guard" not in s
old = 'ap.add_argument("--lo", type=float, default=0.25)\n'
assert old in s
s = s.replace(old, 'ap.add_argument("--guard", choices=["rzero", "azr", "tracer"], default="rzero", help="A3e: rzero band | azr learnability (0,1) | tracer-calibrated band")\n' + old, 1)
old = '''outs = llm.generate([wrap(x["prompt"]) for x in pool], SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, stop=STOP, seed=args.seed))
stats = Counter()
admitted, records = [], []
for x, o in zip(pool, outs):
    ans = [extract(c.text) if c.finish_reason != "length" else None for c in o.outputs]
    cnt = Counter(a for a in ans if a is not None and a != "unknown")
    mode, k = cnt.most_common(1)[0] if cnt else (None, 0)
    p_mode = k / len(ans)
    abst = sum(1 for a in ans if a == "unknown") / len(ans)
    admit = args.lo <= p_mode <= args.hi
'''
assert old in s
new = '''outs = llm.generate([wrap(x["prompt"]) for x in pool], SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, stop=STOP, seed=args.seed))
stats = Counter()
admitted, records = [], []
pm = {}
for x, o in zip(pool, outs):
    ans = [extract(c.text) if c.finish_reason != "length" else None for c in o.outputs]
    cnt = Counter(a for a in ans if a is not None and a != "unknown")
    mode, k = cnt.most_common(1)[0] if cnt else (None, 0)
    pm[x["id"]] = (k / len(ans), mode, sum(1 for a in ans if a == "unknown") / len(ans))
lo, hi = args.lo, args.hi
if args.guard == "azr":
    lo, hi = 1e-9, 1 - 1e-9
elif args.guard == "tracer":
    def rate(kind, a, b):
        v = [pm[x["id"]][0] for x in pool if x["kind"] == kind]
        return sum(1 for q in v if a <= q <= b) / max(1, len(v))
    grid = [i / args.G for i in range(args.G + 1)]
    best = None
    for a in grid:
        for b in grid:
            if b < a:
                continue
            ri, rd = rate("tracer_intact", a, b), rate("tracer_deleted", a, b)
            if ri < 0.30:
                continue
            key = (ri - rd, ri)
            if best is None or key > best[0]:
                best = (key, a, b, ri, rd)
    if best is None:
        best = ((0, 0), args.lo, args.hi, rate("tracer_intact", args.lo, args.hi), rate("tracer_deleted", args.lo, args.hi))
    lo, hi = best[1], best[2]
    print("TRACER_BAND", lo, hi, "intact", best[3], "deleted", best[4], flush=True)
for x in pool:
    p_mode, mode, abst = pm[x["id"]]
    admit = lo <= p_mode <= hi
'''
s = s.replace(old, new, 1)
old = 'summary["n_admitted"] = len(admitted)\n'
assert old in s
s = s.replace(old, old + 'summary["guard"] = args.guard\nsummary["band_lo"] = lo\nsummary["band_hi"] = hi\n', 1)
open(p, "w", newline="\n").write(s)
print("patched a3_guard.py")
