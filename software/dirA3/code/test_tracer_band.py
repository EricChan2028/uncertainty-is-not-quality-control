"""Offline check of the tracer-calibrated band (amendment A3e) on existing guard records: which band would have been chosen,
and what it would have admitted, versus the registered R-Zero band."""
import json
import sys

B = "/home/tione/notebook/rsi_rl/dirB"
G = 8
grid = [i / G for i in range(G + 1)]
for r in sys.argv[1:]:
    recs = [json.loads(l) for l in open(f"{B}/runs/{r}/guard_records.jsonl")]

    def rate(kind, a, b):
        v = [x["p_mode"] for x in recs if x["kind"] == kind]
        return sum(1 for q in v if a <= q <= b) / max(1, len(v))

    best = None
    for a in grid:
        for b in grid[:-1]:
            if b < a:
                continue
            ri, rd = rate("tracer_intact", a, b), rate("tracer_deleted", a, b)
            if ri < 0.30:
                continue
            key = (ri - rd, ri)
            if best is None or key > best[0]:
                best = (key, a, b, ri, rd)
    key, a, b, ri, rd = best
    print("{}: tracer band [{:.3f},{:.3f}] intact {:.2f} deleted {:.2f} ratio {:.2f} proposals {:.2f} || R-Zero: intact {:.2f} deleted {:.2f} proposals {:.2f}".format(
        r, a, b, ri, rd, rd / ri, rate("proposed", a, b), rate("tracer_intact", .25, .75), rate("tracer_deleted", .25, .75), rate("proposed", .25, .75)))
