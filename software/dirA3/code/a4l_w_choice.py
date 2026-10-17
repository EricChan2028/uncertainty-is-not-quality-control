"""Choose the abstention vote weight w for arm own_uvw from existing guard records only (no new data): on the two fully
audited pools, with the own r2 solver of the matching seed under the abstention prompt (qwen_s0_ag/r2, qwen_s1_ag/r2),
the share of items per audited class where "unknown" would win a vote in which each abstention counts w times:
w * abstain > p_mode (p_mode = share of the modal numeric answer). python3 w_choice.py"""
import glob, json
R = "/home/tione/notebook/rsi_rl"
for tag in ("qwen_s0", "qwen_s1"):
    V = {}
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/{tag}_r1/parts/*.json"):
        for v in json.load(open(f))["verdicts"]:
            V[v["id"]] = v["verdict"]
    recs = [json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}_ag/r2/guard_records.jsonl")]
    recs = [r for r in recs if r["kind"] == "proposed" and r["id"] in V]
    ti = [json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/{tag}_ag/r2/guard_records.jsonl")]
    for w in (1, 1.5, 2, 3):
        row = []
        for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"):
            rs = [r for r in recs if V[r["id"]] == c]
            row.append(f"{c[:4]} {100 * sum(w * r['abstain'] > r['p_mode'] for r in rs) / len(rs):5.1f}%")
        trd = [r for r in ti if r["kind"] == "tracer_deleted"]; tin = [r for r in ti if r["kind"] == "tracer_intact"]
        row.append(f"tracer del {100 * sum(w * r['abstain'] > r['p_mode'] for r in trd) / len(trd):5.1f}% int {100 * sum(w * r['abstain'] > r['p_mode'] for r in tin) / len(tin):5.1f}%")
        print(tag, f"w={w}:", " | ".join(row))
