"""Blinded human-validation packet for the Opus audit instrument (A4, 2026-09-24). 150 questions stratified by Opus verdict
(50 SOLVABLE / 50 AMBIGUOUS / 50 UNSOLVABLE), each third drawn from the two fully audited round-1 pools (qwen_s0, qwen_s1) and
from the audited admitted sets of Qwen3-4B own loops at rounds 3-5 (drifted curricula). Random(2026). Writes
dirA4/human_validation/{packet.md, labels_template.csv, key.json}; the key (Opus verdicts, sources) is hashed so it is fixed
before any human label exists. A human annotator labels packet.md without opening key.json.  python3 a4_human_packet.py
"""
import glob, json, os, random
R = "/home/tione/notebook/rsi_rl"; B = f"{R}/dirB/runs"; OUT = f"{R}/dirA4/human_validation"; os.makedirs(OUT, exist_ok=True)
rng = random.Random(2026)
def texts(run):
    return {x["id"]: x.get("problem") or x.get("prompt") for x in (json.loads(l) for l in open(f"{B}/{run}/proposed.jsonl"))}
src = {"pool_s0": [], "pool_s1": [], "loop_r3_r5": []}
for tag, run in (("pool_s0", "qwen3_4b_Base_own_a3_s0_r1"), ("pool_s1", "qwen3_4b_Base_own_a3_s1_r1")):
    T = texts(run)
    for f in glob.glob(f"{R}/dirA3/results/pool_audit/qwen_{tag[-2:]}_r1/parts/*.json"):
        for v in json.load(open(f))["verdicts"]:
            if v["id"] in T: src[tag].append({"src": tag, "run": run, "id": v["id"], "opus": v["verdict"], "text": T[v["id"]]})
for f in sorted(glob.glob(f"{B}/qwen3_4b_Base_own_a3_s*_r[345]/audit_admitted.json")):
    run = f.split("/")[-2]; T = texts(run)
    for v in json.load(open(f))["verdicts"]:
        if v["id"] in T: src["loop_r3_r5"].append({"src": "loop_r3_r5", "run": run, "id": v["id"], "opus": v["verdict"], "text": T[v["id"]]})
items = []
for cl, quota in (("SOLVABLE", 50), ("AMBIGUOUS", 50), ("UNSOLVABLE", 50)):
    per = [17, 17, 16]
    for (tag, lst), q in zip(src.items(), per):
        cand = sorted([x for x in lst if x["opus"] == cl], key=lambda x: (x["run"], x["id"]))
        seen = {(x["run"], x["id"]) for x in items}; cand = [x for x in cand if (x["run"], x["id"]) not in seen]
        items += rng.sample(cand, min(q, len(cand)))
rng.shuffle(items)
for k, x in enumerate(items, 1): x["hid"] = f"H{k:03d}"
rubric = ("Label each question with exactly one of:\n"
          "- **S (SOLVABLE)**: the text determines a unique answer without inventing any assumption.\n"
          "- **U (UNSOLVABLE)**: a needed quantity or relation is missing, or the text is self-contradictory, so no unique answer exists.\n"
          "- **A (AMBIGUOUS)**: two reasonable readings give different answers.\n"
          "Be strict: if you would have to invent a default assumption to get a number, label U. Work each question yourself; "
          "do not look at key.json. Record labels in labels_template.csv (columns hid,label,note).\n")
with open(f"{OUT}/packet.md", "w") as fp:
    fp.write("# Human validation packet: are self-play questions well-posed? (150 items, blinded)\n\n" + rubric + "\n")
    for x in items: fp.write(f"## {x['hid']}\n\n{x['text'].strip()}\n\nLabel (S / A / U): ____   Note: ____\n\n")
with open(f"{OUT}/labels_template.csv", "w") as fp:
    fp.write("hid,label,note\n" + "".join(f"{x['hid']},,\n" for x in items))
json.dump([{k: x[k] for k in ("hid", "src", "run", "id", "opus")} for x in items], open(f"{OUT}/key.json", "w"), indent=0)
from collections import Counter
print(len(items), Counter((x["src"], x["opus"]) for x in items))
