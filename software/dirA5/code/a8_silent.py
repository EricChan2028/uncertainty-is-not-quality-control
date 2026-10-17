"""EXPLORATORY (post hoc, not registered; written 03:10 25 Sep after the A8 read): split each sample on the A8 unanswerable (U)
and hard-answerable (A) units into (i) answers with a flagged gap/assumption (numeric answer and gap_flag), (ii) SILENT
commitments (numeric answer, no gap flag) and (iii) no numeric answer. Uses the A8 blind labels and the extracted answers only.
Also the unit-level rate of 'confident silent' items (modal numeric answer share >= 0.5 of 8 AND no sample flags a gap).
python3 a8_silent.py"""
import json
from collections import Counter, defaultdict
D = "/home/tione/notebook/rsi_rl/dirA5/results"
KEY = {json.loads(l)["key"]: json.loads(l) for l in open(f"{D}/a8_key/units_main_key.jsonl")}
L = {x["key"]: x for x in json.load(open(f"{D}/a8/labels_main.json"))}
cnt = defaultdict(Counter); unit = defaultdict(Counter)
for k, u in KEY.items():
    if k not in L: continue
    cell = ("base" if u["model"] == "base" else f"r5_s{u['seed']}", "U" if not u["answerable"] else "A")
    g = L[k]["gap_flag"]; ans = u["answers"]
    for a, f in zip(ans, g):
        cnt[cell]["flagged_answer" if (a is not None and f) else ("silent_answer" if a is not None else "no_answer")] += 1
    num = [a for a in ans if a is not None]
    c = Counter(num); pm = (c.most_common(1)[0][1] / 8) if c else 0
    unit[cell]["n"] += 1; unit[cell]["confident"] += pm >= 0.5
    unit[cell]["confident_silent"] += (pm >= 0.5 and not any(g))
out = {}
for cell in sorted(cnt):
    n = sum(cnt[cell].values())
    out[f"{cell[0]}|{cell[1]}"] = {"samples": {k: round(v / n, 3) for k, v in cnt[cell].items()},
                                   "units_confident": round(unit[cell]["confident"] / unit[cell]["n"], 3),
                                   "units_confident_and_silent": round(unit[cell]["confident_silent"] / unit[cell]["n"], 3)}
    print(f"{cell[0]:7s} {cell[1]}", out[f"{cell[0]}|{cell[1]}"])
json.dump(out, open(f"{D}/a8/a8_silent_exploratory.json", "w"), indent=1)
