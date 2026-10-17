"""A5 SWAP prep (registered 08:04): for each Qwen3-4B seed 0-4, collect the own r3 admitted PROPOSED items that the existing
200-item audit did not label, into a pseudo-run dirB/runs/swap_audit_a5_s<S>/admitted.jsonl for the frozen audit workflow
(tracers need no audit: deleted = UNSOLVABLE, intact = SOLVABLE by construction). Prints the counts per seed.
python3 a5_swap_prep.py"""
import json, os
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB/runs"
out = {}
for s in range(5):
    run = f"{B}/qwen3_4b_Base_own_a3_s{s}_r3"
    A = [json.loads(l) for l in open(f"{run}/admitted.jsonl")]
    done = {v["id"] for v in json.load(open(f"{run}/audit_admitted.json"))["verdicts"]}
    todo = [a for a in A if a["kind"] == "proposed" and a["id"] not in done]
    d = f"{B}/swap_audit_a5_s{s}"; os.makedirs(f"{d}/audit_parts", exist_ok=True)
    with open(f"{d}/admitted.jsonl", "w") as fh:
        for a in todo:
            fh.write(json.dumps({"id": a["id"], "problem": a.get("problem") or a.get("prompt"), "prompt": a.get("prompt") or a.get("problem"), "kind": "proposed"}, ensure_ascii=False) + "\n")
    out[s] = {"admitted": len(A), "proposed": sum(a["kind"] == "proposed" for a in A), "audited": len(done), "to_audit": len(todo)}
    print(s, out[s])
json.dump(out, open(f"{R}/dirA5/results/swap_prep.json", "w"), indent=1)
