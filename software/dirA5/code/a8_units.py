"""A8 (registered 23:16 24 Sep; clarification A8a): build the blind labelling units from the generated texts. Units = (item,
model) for base and own r5 s0-2 (registered) and own r3 s0-2 (descriptive; separate file). Each unit shows the problem and its
8 solutions; each solution is cut at its first "\\nQuestion:" and then truncated to its last 1,800 characters (A8a.1). Units are
shuffled with Random(20260926) and given neutral keys; the key map (model, seed, round, id, answerable, category, answers, cut
and truncation flags) and the meta file (second-pass keys, reproducibility check) are written to a8_key/, apart from the unit
files. Batches of 5 (line numbers of the unit file) never hold two units of the same item family (an item across models, or an
unanswerable item and its answerable original) (A8a.4). python3 a8_units.py"""
import json, os, random
R = "/home/tione/notebook/rsi_rl"; A = f"{R}/dirA5/results/a8"; T = f"{A}/texts"; K = f"{R}/dirA5/results/a8_key"
os.makedirs(K, exist_ok=True)
sel = json.load(open(f"{A}/items.json")); HA = set(sel["hard_A"]); ids = HA | set(sel["U"])
Q, FAM = {}, {}
for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl"):
    x = json.loads(l); Q[x["id"]] = x["question"].strip()
    if x["id"] in ids: FAM[x["id"]] = x["id"] if x["answerable"] else (x.get("relevant_ids") or [x["id"]])[0]
MODELS = [("base", None, 0, "a8_base", "plain_qwen3_4b_base")] + [("own", s, 5, f"a8_own_s{s}_r5", f"plain_qwen3_4b_Base_own_a3_s{s}_r5") for s in range(3)]
MODELS_R3 = [("own", s, 3, f"a8_own_s{s}_r3", f"plain_qwen3_4b_Base_own_a3_s{s}_r3") for s in range(3)]
def load(path):
    rows = {}
    for l in open(path):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: pass
    return rows
def show(t):
    c = t.split("\nQuestion:")[0]; return c[-1800:], c != t, len(c) > 1800
repro, cut_stats = {}, {}
def batches(units, size=5):
    left = list(range(len(units))); out = []
    while left:
        b, fams = [], set()
        for j in list(left):
            f = FAM[units[j]["id"]]
            if f in fams: continue
            b.append(j); fams.add(f); left.remove(j)
            if len(b) == size: break
        out.append([j + 1 for j in b])       # 1-based line numbers of the unit file
    return out
def build(models, fname, keyname):
    units = []
    for kind, s, rd, tag, otag in models:
        rows = load(f"{T}/{tag}.jsonl"); o = load(f"{R}/dirA4/results/umwp/{otag}.jsonl")
        assert set(rows) >= ids, (tag, len(ids - set(rows)))
        repro[tag] = {"identical_answer_vectors": sum(1 for i in ids if rows[i]["answers"] == o[i]["answers"]), "n": len(ids)}
        nc = nt = 0
        for i in sorted(ids):
            r = rows[i]; assert len(r["texts"]) == 8 and r["answerable"] == (i in HA), (tag, i)
            sh = [show(t) for t in r["texts"]]; nc += sum(x[1] for x in sh); nt += sum(x[2] for x in sh)
            units.append({"model": kind, "seed": s, "round": rd, "id": i, "answerable": r["answerable"], "category": r.get("category"),
                          "answers": r["answers"], "cut": [x[1] for x in sh], "trunc": [x[2] for x in sh], "finish": r.get("finish"),
                          "problem": Q[i], "solutions": [x[0] for x in sh]})
        cut_stats[tag] = {"texts_cut_at_Question": nc, "texts_truncated": nt, "n_texts": 8 * len(ids)}
    random.Random(20260926).shuffle(units)
    for k, u in enumerate(units): u["key"] = f"{keyname}{k:04d}"
    with open(f"{A}/{fname}", "w") as f:
        for u in units: f.write(json.dumps({"key": u["key"], "problem": u["problem"], "solutions": u["solutions"]}) + "\n")
    with open(f"{K}/{fname.replace('.jsonl', '_key.jsonl')}", "w") as f:
        for u in units: f.write(json.dumps({k: u[k] for k in ("key", "model", "seed", "round", "id", "answerable", "category", "answers", "cut", "trunc", "finish")}) + "\n")
    return units
u = build(MODELS, "units_main.jsonl", "m")
json.dump(batches(u), open(f"{A}/batches_main.json", "w"))
line = {x["key"]: n + 1 for n, x in enumerate(u)}
base_keys = sorted(x["key"] for x in u if x["model"] == "base")
second = sorted(random.Random(20260926).sample(base_keys, round(0.25 * len(base_keys))))
su = [x for x in u if x["key"] in set(second)]; su.sort(key=lambda x: x["key"])
json.dump([[line[su[j - 1]["key"]] for j in b] for b in batches(su)], open(f"{A}/batches_second.json", "w"))
meta = {"second_pass_keys": second, "n_main": len(u), "repro": repro, "cut_stats": cut_stats}
try:
    u3 = build(MODELS_R3, "units_r3.jsonl", "t"); json.dump(batches(u3), open(f"{A}/batches_r3.json", "w")); meta["n_r3"] = len(u3)
except (AssertionError, FileNotFoundError) as e: meta["r3"] = f"not built: {e}"
meta["repro"], meta["cut_stats"] = repro, cut_stats
json.dump(meta, open(f"{K}/units_meta.json", "w"), indent=1)
print(len(u), "main units;", len(second), "second-pass;", json.dumps(repro), json.dumps(cut_stats))
