"""A7-5d (registered 26 Sep): blind re-audit sample. All 200 items of each R-Zero audit sample (seeds 0-1, iterations 1-3),
1,200 items shuffled with Random(20260926 + 75); neutral keys z0000.. (key zNNNN = line NNNN + 1). The blind file holds only
key + problem, as UTF-8 (ensure_ascii=False); the key file is in a separate directory. python3 a75d_sample.py"""
import json, os, random
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB/runs"
OUT = f"{R}/dirA5/results/a75d_blind"; KEY = f"{R}/dirA5/results/a75d_key"
assert not os.path.exists(OUT) and not os.path.exists(KEY), "sample already built"
os.makedirs(OUT); os.makedirs(KEY)
items = []
for s in (0, 1):
    for it in (1, 2, 3):
        run = f"rzero_qwen3_4b_Base_s{s}_it{it}"
        a = json.load(open(f"{B}/{run}/audit_admitted.json", encoding="utf-8"))
        tx = {}
        for l in open(f"{B}/{run}/admitted.jsonl", encoding="utf-8"):
            r = json.loads(l); tx[r["id"]] = r["problem"]
        vs = sorted([v for v in a["verdicts"] if v["id"] in tx], key=lambda v: v["k"])
        assert len(vs) == 200 and len({v["id"] for v in vs}) == 200, (run, len(vs))
        items += [{"seed": s, "iteration": it, "run": run, "id": v["id"], "orig_verdict": v["verdict"], "problem": tx[v["id"]]}
                  for v in vs]
random.Random(20260926 + 75).shuffle(items)
for k, x in enumerate(items):
    x["key"] = f"z{k:04d}"
for x in items:
    assert not any(c in x["problem"] for c in "  \u0085\r"), x["key"]
with open(f"{KEY}/key_full.jsonl", "w", encoding="utf-8") as f:
    for x in items:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(f"{OUT}/items.jsonl", "w", encoding="utf-8") as f:
    for x in items:
        f.write(json.dumps({"key": x["key"], "problem": x["problem"]}, ensure_ascii=False) + "\n")
print(len(items))
