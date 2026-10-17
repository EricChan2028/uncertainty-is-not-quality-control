"""A4 registered reads (dirA4/docs/A4_registration_2026-09-23.md): PF2-PF4 (abstention guard), PC1-PC3 (co-evolving
proposer), PG1-PG2 (Phi-4-mini family), printed per seed with the booleans the decision rules KF / KH / KG use.
Inputs: dirB/runs/<lineage>_r<n>/{guard_stats.json, audit_admitted.json, audit_raw.json}, dirB/results/evals/<run>_step60.summary.json
(lenient greedy accuracy, A3d), dirB/results/hp/hp_<run>_shard0of1.jsonl (Q1u on the 412 validated unsolvable twins),
dirB/results/hp/hpabs_<run>_shard0of1.jsonl (abstention under the abstention prompt).   python3 a4_reads.py
"""
import glob
import json
import os

R = "/home/tione/notebook/rsi_rl"
B = f"{R}/dirB"
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
DS = ["math500_train", "math_heldout1000", "gsm8k_test500"]


def acc(tag):
    f = f"{B}/results/evals/{tag}.summary.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    vals = [d[k].get("greedy_acc_lenient", d[k].get("greedy_acc")) for k in DS if k in d]
    return 100 * sum(vals) / len(vals) if len(vals) == 3 else None


def q1u(run):
    fs = glob.glob(f"{B}/results/hp/hp_{run}_shard*.jsonl")
    if not fs:
        return None
    d = [json.loads(l) for f in fs for l in open(f)]
    if len(d) < 1086:
        return None
    dele = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(1 for r in dele if r["p_mode"] >= 0.5) / max(1, len(dele)) if dele else None


def abst_uns(run):
    fs = glob.glob(f"{B}/results/hp/hpabs_{run}_shard*.jsonl")
    if not fs:
        return None
    d = [json.loads(l) for f in fs for l in open(f)]
    if len(d) < 1086:
        return None
    dele = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    return sum(r.get("abstain", 0) for r in dele) / max(1, len(dele)) if dele else None


def audit(run, name="audit_admitted.json"):
    f = f"{B}/runs/{run}/{name}"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    c = d["counts"]; n = sum(c.values())
    return {"uns": 100 * c.get("UNSOLVABLE", 0) / n, "ill": 100 * (n - c.get("SOLVABLE", 0)) / n, "n": n}


def guard(run):
    f = f"{B}/runs/{run}/guard_stats.json"
    if not os.path.exists(f):
        return None
    g = json.load(open(f))
    ti, td = g.get("tracer_intact_admit_rate"), g.get("tracer_deleted_admit_rate")
    return {"ratio": (td / ti) if ti else None, "admit": g.get("proposed_admit_rate")}


def row(L, rd):
    run = f"{L}_r{rd}"
    return {"acc": acc(f"{run}_step60"), "q1u": q1u(run), "abs": abst_uns(run), "audit": audit(run), "raw": audit(run, "audit_raw.json"), "guard": guard(run),
            "done": os.path.exists(f"{B}/runs/{run}/.done")}


def fmt(x, d=1):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def lineage_table(L, rounds=(1, 2, 3)):
    rows = {rd: row(L, rd) for rd in rounds}
    for rd, r in rows.items():
        a, g, ra = r["audit"], r["guard"], r["raw"]
        print(f"  {L} r{rd}: acc {fmt(r['acc'])} | Q1u {fmt(r['q1u'], 3)} | abstain(uns) {fmt(r['abs'], 3)} | ratio {fmt(g['ratio'] if g else None, 2)} admit {fmt(100*g['admit'] if g and g['admit'] is not None else None)}% "
              f"| audit uns {fmt(a['uns'] if a else None)} ill {fmt(a['ill'] if a else None)} | raw uns {fmt(ra['uns'] if ra else None)}{'' if r['done'] else '  (running)'}")
    return rows


OWN_R3 = {"qwen3_4b_Base": {0: "qwen3_4b_Base_own_a3_s0_r3", 1: "qwen3_4b_Base_own_a3_s1_r3", 2: "qwen3_4b_Base_own_a3_s2_r3"},
          "qwen3_8b_Base": {1: "qwen3_8b_Base_own_a3_s1_r3", 2: "qwen3_8b_Base_own_a3_s2_r3"},
          "phi4mini": {0: "phi4mini_own_a4_s0_r3", 1: "phi4mini_own_a4_s1_r3"}}
OWN_R1_RAW = 29.3   # UNSOLVABLE share of the frozen proposer's raw candidates (A3 section 2d pool audit: 339 / 1158)

print("=== PF (abstention guard, arm own_ag): PF2 uns <= 30 every round & rise r1->r3 < 10 & ratio <= 0.6 every round; PF3 acc_r3 >= own_r3 - 2; PF4 Q1u_r3 <= own Q1u_r3")
for model, seeds in (("qwen3_4b_Base", (0, 1, 2)), ("qwen3_8b_Base", (1, 2)), ("phi4mini", (0, 1))):
    for s in seeds:
        L = f"{model}_own_ag_a4_s{s}"
        if not glob.glob(f"{B}/runs/{L}_r1"):
            continue
        rows = lineage_table(L)
        a = {rd: rows[rd]["audit"] for rd in rows}; g = {rd: rows[rd]["guard"] for rd in rows}
        pf2 = all(a[rd] and a[rd]["uns"] <= 30 for rd in rows) and (a[1] and a[3] and a[3]["uns"] - a[1]["uns"] < 10) and all(g[rd] and g[rd]["ratio"] is not None and g[rd]["ratio"] <= 0.6 for rd in rows)
        own = OWN_R3[model].get(s)
        own_acc = acc(f"{own}_step60") if own else None
        own_q = q1u(own) if own else None
        pf3 = (rows[3]["acc"] is not None and own_acc is not None and rows[3]["acc"] >= own_acc - 2)
        pf4 = (rows[3]["q1u"] is not None and own_q is not None and rows[3]["q1u"] <= own_q)
        print(f"  -> seed {s}: PF2 {pf2} | PF3 {pf3} (own r3 {fmt(own_acc)}) | PF4 {pf4} (own Q1u r3 {fmt(own_q, 3)})")

print("=== PC (co-evolving proposer): PC1 raw uns rise r1->r3 >= 15 & admitted r3 >= own r3; PC2 coev acc_r3 <= own_r3 - 2; PC3 coev_ag raw rise < 10 & acc_r3 >= coev_r3 + 2 & >= own_r3 - 2")
for s in (0, 1):
    Lc, La = f"qwen3_4b_Base_coev_a4_s{s}", f"qwen3_4b_Base_coev_ag_a4_s{s}"
    if not glob.glob(f"{B}/runs/{Lc}_r1"):
        continue
    rc = lineage_table(Lc); ra_ = lineage_table(La)
    own_acc = acc(f"{OWN_R3['qwen3_4b_Base'][s]}_step60"); own_aud = audit(OWN_R3["qwen3_4b_Base"][s])
    raw1 = rc[1]["raw"]["uns"] if rc[1]["raw"] else OWN_R1_RAW
    raw3 = rc[3]["raw"]["uns"] if rc[3]["raw"] else None
    pc1 = raw3 is not None and raw3 - raw1 >= 15 and rc[3]["audit"] and own_aud and rc[3]["audit"]["uns"] >= own_aud["uns"]
    pc2 = rc[3]["acc"] is not None and own_acc is not None and rc[3]["acc"] <= own_acc - 2
    raw1a = ra_[1]["raw"]["uns"] if ra_[1]["raw"] else None; raw3a = ra_[3]["raw"]["uns"] if ra_[3]["raw"] else None
    pc3 = (raw1a is not None and raw3a is not None and raw3a - raw1a < 10 and ra_[3]["acc"] is not None and rc[3]["acc"] is not None
           and ra_[3]["acc"] >= rc[3]["acc"] + 2 and own_acc is not None and ra_[3]["acc"] >= own_acc - 2)
    print(f"  -> seed {s}: PC1 {pc1} (raw uns r1 {fmt(raw1)} -> r3 {fmt(raw3)}) | PC2 {pc2} (coev r3 {fmt(rc[3]['acc'])} vs own {fmt(own_acc)}) | PC3 {pc3} (coev_ag r3 {fmt(ra_[3]['acc'])})")

print("=== PG (Phi-4-mini family): PG1 own ratio rises r1->r2 & audited uns rises >= 10 r1->r3 without collapse; PG2 = PF2 & PF3 on Phi own_ag")
for s in (0, 1):
    L = f"phi4mini_own_a4_s{s}"
    if not glob.glob(f"{B}/runs/{L}_r1"):
        continue
    rows = lineage_table(L)
    g, a = {rd: rows[rd]["guard"] for rd in rows}, {rd: rows[rd]["audit"] for rd in rows}
    base_acc = acc("phi4mini_step0")
    collapsed = any(rows[rd]["acc"] is not None and base_acc is not None and rows[rd]["acc"] < 0.5 * base_acc for rd in rows)
    pg1 = (g[1] and g[2] and g[1]["ratio"] and g[2]["ratio"] and g[2]["ratio"] > g[1]["ratio"]) and (a[1] and a[3] and a[3]["uns"] - a[1]["uns"] >= 10) and not collapsed
    print(f"  -> seed {s}: PG1 {pg1} (base acc {fmt(base_acc)}, collapsed {collapsed})")
