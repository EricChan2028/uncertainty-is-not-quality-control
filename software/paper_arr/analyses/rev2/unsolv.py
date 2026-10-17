import json, os
R = "/home/tione/notebook/rsi_rl/dirB/runs"
for arm, seeds, rounds in (("own_a3", range(5), (1, 3, 5)), ("own_ag_a4", range(3), (3, 5)), ("coev_a4", range(4), (3,)), ("coev_ag_a4", range(2), (3,)), ("ctrl_rnd_a5", range(3), (3,))):
    for r in rounds:
        row = []
        for s in seeds:
            f = "%s/qwen3_4b_Base_%s_s%d_r%d/audit_admitted.json" % (R, arm, s, r)
            if not os.path.exists(f):
                row.append((s, None)); continue
            c = json.load(open(f))["counts"]; n = sum(c.values())
            row.append((s, n, round(100 * c.get("UNSOLVABLE", 0) / n, 1), round(100 * c.get("AMBIGUOUS", 0) / n, 1), round(100 * (c.get("UNSOLVABLE", 0) + c.get("AMBIGUOUS", 0)) / n, 1)))
        print(arm, "r%d" % r, "(seed, n, U%, A%, U+A%):", row)
