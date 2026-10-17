"""A7 item 1 (registered 15:10, descriptive): sample for the ill-posedness typing pass. Up to 60 audited ill-posed
(UNSOLVABLE or AMBIGUOUS) admitted problems per (Qwen3-4B own seed 0-2, round 1/3/5) from the 200-item admitted audits, and 60
ill-posed items from each seed's raw round-1 proposals (fixed-pool full audits; these exist for seeds 0 and 1 only - seed 2's
raw pool was never fully audited, disclosed). Random(20260924) per stratum (seeded 20260924 + 100*seed + round; raw: + 50).
All items are shuffled together (Random(20260924)) and given blind keys; typers see only the problem text. A 25% subsample
(Random(20260925)) is typed a second time by independent typers for agreement (Cohen's kappa).
python3 a7_tax_sample.py"""
import glob, json, os, random
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB/runs"; PA = f"{R}/dirA3/results/pool_audit"; OUT = f"{R}/dirA5/results/a7_taxonomy"
os.makedirs(OUT, exist_ok=True)
ILL = {"UNSOLVABLE", "AMBIGUOUS"}
def texts(f):
    t = {}
    for l in open(f):
        try: r = json.loads(l)
        except ValueError: continue
        t[r["id"]] = r.get("problem") or r.get("prompt")
    return t
items, strata = [], {}
for s in range(3):
    for rd in (1, 3, 5):
        run = f"qwen3_4b_Base_own_a3_s{s}_r{rd}"
        a = json.load(open(f"{B}/{run}/audit_admitted.json"))
        tx = texts(f"{B}/{run}/admitted.jsonl")
        ill = sorted([v for v in a["verdicts"] if v["verdict"] in ILL and v["id"] in tx], key=lambda v: v["k"])
        pick = random.Random(20260924 + 100 * s + rd).sample(ill, min(60, len(ill)))
        strata[f"s{s}_r{rd}"] = {"n_illposed_audited": len(ill), "n_sampled": len(pick)}
        items += [{"stratum": f"s{s}_r{rd}", "seed": s, "round": rd, "source": "admitted", "id": v["id"], "audit": v["verdict"], "problem": tx[v["id"]]} for v in pick]
    pd = f"{PA}/qwen_s{s}_r1/parts"
    if os.path.isdir(pd):
        vs = {}
        for f in glob.glob(f"{pd}/*.json"):
            for v in json.load(open(f))["verdicts"]: vs[v["id"]] = v
        tx = texts(f"{B}/qwen3_4b_Base_own_a3_s{s}_r1/proposed.jsonl")
        ill = sorted([v for v in vs.values() if v["verdict"] in ILL and v["id"] in tx], key=lambda v: v["id"])
        pick = random.Random(20260924 + 100 * s + 50).sample(ill, min(60, len(ill)))
        strata[f"s{s}_raw"] = {"n_illposed_audited": len(ill), "n_pool_judged": len(vs), "n_sampled": len(pick)}
        items += [{"stratum": f"s{s}_raw", "seed": s, "round": 0, "source": "raw_r1_pool", "id": v["id"], "audit": v["verdict"], "problem": tx[v["id"]]} for v in pick]
    else:
        strata[f"s{s}_raw"] = {"n_sampled": 0, "note": "no fixed-pool full audit for this seed"}
random.Random(20260924).shuffle(items)
for k, it in enumerate(items): it["key"] = f"t{k:04d}"
second = sorted(random.Random(20260925).sample(range(len(items)), round(0.25 * len(items))))
json.dump({"strata": strata, "n": len(items), "second_pass_keys": [items[i]["key"] for i in second]}, open(f"{OUT}/sample_meta.json", "w"), indent=1)
with open(f"{OUT}/sample_full.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
with open(f"{OUT}/sample_blind.jsonl", "w") as f:        # what typers see: key + problem only
    for it in items: f.write(json.dumps({"key": it["key"], "problem": it["problem"]}) + "\n")
print(json.dumps(strata), len(items), len(second))
