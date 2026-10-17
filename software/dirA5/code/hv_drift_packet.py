"""Human-validation packet 2 (drift): 240 of the 900 A7b blind items (80 per round r1/r3/r5, 26-27 per seed and round),
random.Random(20260927), shuffled under neutral ids D001..D240; same instructions and format as dirA4/human_validation
(packet 1). The key (seed, round, A7b key, Opus verdicts) is written to a separate folder _key_do_not_open/.
Prepared only; nothing is annotated until the author decides. python3 hv_drift_packet.py"""
import csv, json, os, random
R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA5/human_validation_drift"; KD = f"{OUT}/_key_do_not_open"
os.makedirs(KD, exist_ok=True)
K = [json.loads(l) for l in open(f"{R}/dirA5/results/a7b_blind/key_full.jsonl")]
BO = {v["key"]: v["verdict"] for v in json.load(open(f"{R}/dirA5/results/a7b_blind/blind_verdicts.json"))}
rng = random.Random(20260927); pick = []
for rd in (1, 3, 5):
    for s, n in zip((0, 1, 2), (27, 27, 26)):
        cell = sorted([x for x in K if x["seed"] == s and x["round"] == rd], key=lambda x: x["key"])
        pick += rng.sample(cell, n)
rng.shuffle(pick)
key = []
with open(f"{OUT}/packet.md", "w") as f:
    f.write("# Human validation packet 2: are self-play questions well-posed? (240 items, blinded)\n\n"
            "Label each question with exactly one of:\n"
            "- **S (SOLVABLE)**: the text determines a unique answer without inventing any assumption.\n"
            "- **U (UNSOLVABLE)**: a needed quantity or relation is missing, or the text is self-contradictory, so no unique answer exists.\n"
            "- **A (AMBIGUOUS)**: two reasonable readings give different answers.\n"
            "Be strict: if you would have to invent a default assumption to get a number, label U. Work each question yourself; "
            "do not open the folder _key_do_not_open. Record labels in labels_template.csv (columns hid,label,note). "
            "If several people annotate, each fills their own copy of the template independently.\n\n")
    for j, x in enumerate(pick, 1):
        hid = f"D{j:03d}"
        f.write(f"## {hid}\n\n{x['problem'].strip()}\n\nLabel (S / A / U): ____   Note: ____\n\n")
        key.append({"hid": hid, "a7b_key": x["key"], "seed": x["seed"], "round": x["round"], "opus_original": x["orig_verdict"], "opus_blind": BO[x["key"]]})
with open(f"{OUT}/labels_template.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["hid", "label", "note"]); [w.writerow([k["hid"], "", ""]) for k in key]
json.dump(key, open(f"{KD}/key.json", "w"), indent=1)
print(len(key), {rd: sum(k["round"] == rd for k in key) for rd in (1, 3, 5)})
