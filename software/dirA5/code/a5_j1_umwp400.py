"""A5 J-P1 item sample: 200 unanswerable + 200 answerable UMWP questions, stratified by source x category (answerable: by
source), RNG 20260924. Writes a pseudo-run dirB/runs/umwp400_j1/admitted.jsonl (kind "proposed", id "umwp:<id>") for the
frozen 200-item audit workflow (run over all 400) and dirA5/results/j1_umwp400_key.json (UMWP labels)."""
import json, os, random
from collections import defaultdict
R = "/home/tione/notebook/rsi_rl"
D = [json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl")]
rng = random.Random(20260924)
def strat(items, key, n):
    g = defaultdict(list)
    for x in sorted(items, key=lambda x: x["id"]): g[key(x)].append(x)
    tot = len(items); out = []
    alloc = {k: max(1, round(n * len(v) / tot)) for k, v in g.items()}
    while sum(alloc.values()) > n: alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < n: alloc[max(g, key=lambda k: len(g[k]) - alloc[k])] += 1
    for k, v in g.items(): out += rng.sample(v, min(alloc[k], len(v)))
    return out
U = strat([x for x in D if not x["answerable"]], lambda x: (x["source"], x["category"]), 200)
A = strat([x for x in D if x["answerable"]], lambda x: x["source"], 200)
items = U + A; rng.shuffle(items)
d = f"{R}/dirB/runs/umwp400_j1"; os.makedirs(f"{d}/audit_parts", exist_ok=True)
with open(f"{d}/admitted.jsonl", "w") as fh:
    for x in items:
        fh.write(json.dumps({"id": f"umwp:{x['id']}", "problem": x["question"].strip(), "prompt": x["question"].strip(), "kind": "proposed"}) + "\n")
json.dump({f"umwp:{x['id']}": {"answerable": x["answerable"], "category": x["category"], "source": x["source"]} for x in items},
          open(f"{R}/dirA5/results/j1_umwp400_key.json", "w"), indent=0)
print(len(U), len(A), len(items))
