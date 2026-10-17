"""A5 SWAP build (registered 08:04). Seed S: own r3 admitted set A; flags from the own r2 solver under the abstention prompt
(flag = abstain share >= 2/8, from dirB/runs/swap_flag_a5_s<S>/guard_records.jsonl); audited class per item (existing
200-item audit + the swap_audit_a5_s<S> parts for the rest; tracers by construction: deleted = UNSOLVABLE, intact = SOLVABLE).
  G1 = A minus the flagged items.
  C1 = A minus the same number of UNFLAGGED items with the same audited S/A/U mix as the flagged set (stratified random,
       RNG 20260924 + S); if a class has too few unflagged items, all of them are dropped and the gap is recorded.
Underpowered (excluded) if fewer than 10% of A are flagged. Writes the two admitted sets and swap_build_stats.json.
python3 a5_swap_build.py <seed>"""
import glob, json, os, random, sys
S = int(sys.argv[1])
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB/runs"
A = [json.loads(l) for l in open(f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/admitted.jsonl")]
flag_rec = {r["id"]: r for r in (json.loads(l) for l in open(f"{B}/swap_flag_a5_s{S}/guard_records.jsonl"))}
lab = {v["id"]: v["verdict"] for v in json.load(open(f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/audit_admitted.json"))["verdicts"]}
for f in glob.glob(f"{B}/swap_audit_a5_s{S}/audit_parts/*.json"):
    for v in json.load(open(f))["verdicts"]:
        lab[v["id"]] = v["verdict"]
def cls(a):
    if a["kind"] == "tracer_deleted": return "UNSOLVABLE"
    if a["kind"] == "tracer_intact": return "SOLVABLE"
    return lab.get(a["id"])
missing_flag = [a["id"] for a in A if a["id"] not in flag_rec]
missing_lab = [a["id"] for a in A if cls(a) is None]
if missing_flag or missing_lab:
    raise SystemExit(f"incomplete inputs: {len(missing_flag)} unflagged-scored, {len(missing_lab)} unlabelled")
flagged = [a for a in A if flag_rec[a["id"]]["abstain"] >= 0.25 - 1e-9]
unfl = [a for a in A if flag_rec[a["id"]]["abstain"] < 0.25 - 1e-9]
st = {"seed": S, "n": len(A), "flagged": len(flagged), "flagged_share": len(flagged) / len(A), "underpowered": len(flagged) < 0.10 * len(A)}
mix = {c: sum(cls(a) == c for a in flagged) for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")}
rng = random.Random(20260924 + S)
drop_c1, gap = [], {}
for c, n in mix.items():
    pool = sorted([a for a in unfl if cls(a) == c], key=lambda a: a["id"])
    take = rng.sample(pool, min(n, len(pool)))
    drop_c1 += take
    if len(pool) < n: gap[c] = n - len(pool)
drop_ids = {a["id"] for a in drop_c1}
G1 = [a for a in A if flag_rec[a["id"]]["abstain"] < 0.25 - 1e-9]
C1 = [a for a in A if a["id"] not in drop_ids]
st.update({"flagged_mix": mix, "c1_dropped": len(drop_c1), "c1_gap": gap, "g1_n": len(G1), "c1_n": len(C1),
           "g1_mix": {c: sum(cls(a) == c for a in G1) for c in mix}, "c1_mix": {c: sum(cls(a) == c for a in C1) for c in mix}})
for arm, items in (("g1", G1), ("c1", C1)):
    d = f"{B}/qwen3_4b_Base_swap_{arm}_a5_s{S}_r3"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/admitted.jsonl", "w") as fh:
        for a in items: fh.write(json.dumps(a, ensure_ascii=False) + "\n")
json.dump(st, open(f"{B}/swap_flag_a5_s{S}/swap_build_stats.json", "w"), indent=1)
print(json.dumps(st))
