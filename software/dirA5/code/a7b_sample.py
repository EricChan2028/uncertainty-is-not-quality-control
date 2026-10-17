"""A7b (registered 17:12 24 Sep): blind re-audit sample. 100 items per (Qwen3-4B own seed 0-2, round 1/3/5) from each
200-item audit sample, Random(20260926 + 100*seed + round); 900 items shuffled with Random(20260926); neutral keys; the
blind file holds only key + problem. python3 a7b_sample.py"""
import json, os, random
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB/runs"; OUT = f"{R}/dirA5/results/a7b_blind"
os.makedirs(OUT, exist_ok=True)
items = []
for s in range(3):
    for rd in (1, 3, 5):
        run = f"qwen3_4b_Base_own_a3_s{s}_r{rd}"
        a = json.load(open(f"{B}/{run}/audit_admitted.json"))
        tx = {}
        for l in open(f"{B}/{run}/admitted.jsonl"):
            r = json.loads(l); tx[r["id"]] = r["problem"]
        vs = sorted([v for v in a["verdicts"] if v["id"] in tx], key=lambda v: v["k"])
        assert len(vs) == 200, (run, len(vs))
        pick = random.Random(20260926 + 100 * s + rd).sample(vs, 100)
        items += [{"seed": s, "round": rd, "run": run, "id": v["id"], "orig_verdict": v["verdict"], "problem": tx[v["id"]]} for v in pick]
random.Random(20260926).shuffle(items)
for k, it in enumerate(items): it["key"] = f"b{k:04d}"
with open(f"{OUT}/key_full.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
with open(f"{OUT}/items.jsonl", "w") as f:
    for it in items: f.write(json.dumps({"key": it["key"], "problem": it["problem"]}) + "\n")
print(len(items))
