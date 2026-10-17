"""Fixed-pool guard diagnostic readout. Pool = own s0 round-1 proposals (1,158 candidates, every one judged by Opus:
dirA3/results/pool_audit/qwen_s0_r1/parts/*.json) + the round-1 planted tracers (200 premise-deleted, 200 intact; status
known by construction). Guard passes = dirA3/results/guard_diag/qwen_s0/<ck>/guard_records.jsonl for ck in base, r1..r5
(R-Zero band [0.25, 0.75], G = 8, same seed). For every checkpoint: mean p_mode, share confident (p_mode >= 0.5), share
in band (admitted), by item class; ill-posed share of the admitted candidates; and the p_mode histogram by class.
python3 a3_guard_diag_score.py   -> prints the table and writes dirA3/results/guard_diag/qwen_s0/result.json
"""
import glob, json, os, sys
from collections import Counter, defaultdict
R = "/home/tione/notebook/rsi_rl"
SUB = sys.argv[1] if len(sys.argv) > 1 else "qwen_s0"
D = f"{R}/dirA3/results/guard_diag/{SUB}"
verdict = {}
for f in glob.glob((sys.argv[2] if len(sys.argv) > 2 else f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts") + "/*.json"):
    for v in json.load(open(f))["verdicts"]:
        verdict[v["id"]] = v["verdict"]
print("pool audit verdicts:", len(verdict), dict(Counter(verdict.values())))
def cls(r):
    if r["kind"] == "tracer_deleted": return "tracer_deleted"
    if r["kind"] == "tracer_intact": return "tracer_intact"
    return "cand_" + verdict.get(r["id"], "UNJUDGED")
CLASSES = ["cand_SOLVABLE", "cand_AMBIGUOUS", "cand_UNSOLVABLE", "tracer_intact", "tracer_deleted"]
BINS = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
out = {}
hdr = f"{'ck':5s} " + " ".join(f"{c[:14]:>14s}" for c in CLASSES) + " | admitted: n  ill-posed%  (uns% amb%)  del/int tracer ratio"
print("mean p_mode | confident% (p_mode>=0.5) | admitted% per class")
print(hdr)
for ck in ["base", "r1", "r2", "r3", "r4", "r5"]:
    f = f"{D}/{ck}/guard_records.jsonl"
    if not os.path.exists(f):
        print(f"{ck:5s} pending"); continue
    rows = [json.loads(l) for l in open(f)]
    by = defaultdict(list)
    for r in rows:
        by[cls(r)].append(r)
    stat = {}
    for c in CLASSES:
        rs = by.get(c, [])
        n = len(rs)
        stat[c] = {"n": n,
                   "mean_p": sum(r["p_mode"] for r in rs) / n if n else None,
                   "confident": sum(r["p_mode"] >= 0.5 for r in rs) / n if n else None,
                   "admitted": sum(bool(r["admit"]) for r in rs) / n if n else None,
                   "hist": [sum(1 for r in rs if (b == 0 and r["p_mode"] <= 0) or (b > 0 and BINS[i-1] < r["p_mode"] <= b)) for i, b in enumerate(BINS)]}
    adm = [r for r in rows if r["admit"] and r["kind"] == "proposed" and r["id"] in verdict]
    na = len(adm)
    uns = sum(verdict[r["id"]] == "UNSOLVABLE" for r in adm) / na if na else 0
    amb = sum(verdict[r["id"]] == "AMBIGUOUS" for r in adm) / na if na else 0
    ti = stat["tracer_intact"]["admitted"] or 0; td = stat["tracer_deleted"]["admitted"] or 0
    stat["admitted_candidates"] = {"n": na, "unsolvable": uns, "ambiguous": amb, "illposed": uns + amb, "tracer_ratio": (td / ti) if ti else None}
    out[ck] = stat
    line = f"{ck:5s} " + " ".join(f"{100*stat[c]['mean_p']:4.0f}/{100*stat[c]['confident']:3.0f}/{100*stat[c]['admitted']:3.0f}".rjust(14) for c in CLASSES)
    line += f" | {na:4d}  {100*(uns+amb):5.1f}  ({100*uns:4.1f} {100*amb:4.1f})  {td/ti if ti else float('nan'):.2f}"
    print(line)
print("cells: mean p_mode / confident% / admitted%  (candidates by Opus verdict; tracers by construction)")
json.dump(out, open(f"{D}/result.json", "w"), indent=1)
print("wrote", f"{D}/result.json")
