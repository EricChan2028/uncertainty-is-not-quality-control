"""A6 X build (registered 09:32): for a Qwen3-4B seed, CLN1 = audited-SOLVABLE proposals of own r3's admitted set + its intact
tracers (as A3 own_clean); RND1 = a random subset of the same admitted set of exactly |CLN1| items (RNG 20260924 + seed).
Labels: the existing 200-item audit + the A5 SWAP full-audit parts. Writes dirB/runs/qwen3_4b_Base_{cln1,rnd1}_a6_s<S>_r3/
admitted.jsonl and prints counts.   python3 a6_cln_build.py <seed>"""
import glob, json, os, random, sys
S = int(sys.argv[1]); B = "/home/tione/notebook/rsi_rl/dirB/runs"
A = [json.loads(l) for l in open(f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/admitted.jsonl")]
lab = {v["id"]: v["verdict"] for v in json.load(open(f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/audit_admitted.json"))["verdicts"]}
for f in glob.glob(f"{B}/swap_audit_a5_s{S}/audit_parts/*.json"):
    for v in json.load(open(f))["verdicts"]: lab[v["id"]] = v["verdict"]
miss = [a["id"] for a in A if a["kind"] == "proposed" and a["id"] not in lab]
if miss: raise SystemExit(f"{len(miss)} proposals unlabelled")
cln = [a for a in A if (a["kind"] == "proposed" and lab[a["id"]] == "SOLVABLE") or a["kind"] == "tracer_intact"]
rnd = random.Random(20260924 + S).sample(A, len(cln))
for arm, items in (("cln1", cln), ("rnd1", rnd)):
    d = f"{B}/qwen3_4b_Base_{arm}_a6_s{S}_r3"; os.makedirs(d, exist_ok=True)
    with open(f"{d}/admitted.jsonl", "w") as fh:
        for a in items: fh.write(json.dumps(a, ensure_ascii=False) + "\n")
kinds = lambda items: {k: sum(a["kind"] == k for a in items) for k in ("proposed", "tracer_deleted", "tracer_intact")}
print(json.dumps({"seed": S, "admitted": len(A), "cln1": len(cln), "cln1_kinds": kinds(cln), "rnd1_kinds": kinds(rnd)}))
