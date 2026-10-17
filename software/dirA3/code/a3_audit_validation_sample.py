"""Audit-instrument validation sample: 100 intact GSM-HP twins (solvable by construction, gold answer known) and 100
premise-deleted twins (a number-bearing sentence removed) from the tracer pool (outside the 543 probe set), shuffled
with Random(0). sample.jsonl holds only k/id/problem (what the judges see); labels.json holds kind + full twin record.
"""
import json, os, random
R = "/home/tione/notebook/rsi_rl"
out = f"{R}/dirA3/results/audit_validation"; os.makedirs(out, exist_ok=True)
tw = [json.loads(l) for l in open(f"{R}/data/hp_twins.jsonl")]
print("fields:", sorted(tw[0].keys()))
pool = [t for t in tw if not (t["task"] == 0 or t["split_kind"] == "template")]
rng = random.Random(0); rng.shuffle(pool)
items = []
for t in pool[:100]:
    items.append({"id": f"trd:{t['fam']}:{t['task']}", "kind": "tracer_deleted", "problem": t["deleted"].strip(), "twin": t})
    items.append({"id": f"tri:{t['fam']}:{t['task']}", "kind": "tracer_intact", "problem": t["intact"].strip(), "twin": t})
rng.shuffle(items)
with open(f"{out}/sample.jsonl", "w") as f:
    for k, it in enumerate(items):
        f.write(json.dumps({"k": k, "id": it["id"], "problem": it["problem"]}) + "\n")
json.dump({it["id"]: {"k": k, "kind": it["kind"], "twin": it["twin"]} for k, it in enumerate(items)}, open(f"{out}/labels.json", "w"))
print("n", len(items), "deleted", sum(i["kind"] == "tracer_deleted" for i in items), "intact", sum(i["kind"] == "tracer_intact" for i in items))
print("example intact:", [i for i in items if i["kind"] == "tracer_intact"][0]["problem"][:150])
