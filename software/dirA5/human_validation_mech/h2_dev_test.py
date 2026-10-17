"""Development test of h2_read.py on SIMULATED labels only (no human labels exist when this runs). Writes simulated label files
to /tmp/h2_dev (LLM labels with 15% of entries replaced at random, plus formatting variants: lower case, full-width letters,
long forms, a byte-order mark, blanks and one unknown code), runs h2_read.py --dev /tmp/h2_dev and checks that the output has
every section and that the human and LLM estimates are close when the simulated noise is small. python3 h2_dev_test.py"""
import csv, json, os, random, subprocess, sys

R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA5/human_validation_mech"; DEV = "/tmp/h2_dev"
os.makedirs(DEV, exist_ok=True)
key = json.load(open(f"{OUT}/_key_do_not_open/key.json"))
LLM = {x["key"]: x for x in json.load(open(f"{R}/dirA5/results/a8/labels_main.json")) if isinstance(x, dict) and "key" in x}
rng = random.Random(1)
VAR_A = {"N": ["N", "n", "none"], "R": ["R", "r", "Ｒ", "readings"], "E": ["E", "e", "error"], "M": ["M", "m", "mixed"]}
VAR_F = {1: ["Y", "y", "yes", "Ｙ", "1"], 0: ["N", "n", "no", "0"]}

def write(name, rows, header, bom=False, enc=None):
    with open(f"{DEV}/{name}", "w", newline="", encoding=enc or ("utf-8-sig" if bom else "utf-8")) as f:
        w = csv.writer(f); w.writerow(header); [w.writerow(r) for r in rows]

for a, noise, bom in (("ann1", 0.15, True), ("ann2", 0.15, False)):
    ra, rb = [], []
    for x in key:
        l = LLM[x["a8_key"]]
        if x["part"] == "A":
            lab = l["disagreement_source"][0]
            if rng.random() < noise: lab = rng.choice("NREM")
            v = rng.choice(VAR_A[lab]) if rng.random() > 0.02 else ""
            ra.append([x["hid"], v, "", ""])
        else:
            fl = int(l["gap_flag"][x["sample"]])
            if rng.random() < noise: fl = 1 - fl
            ans = rng.choice(["NUM", "num", "DECLINE", "NONE", "number"])
            rb.append([x["hid"], rng.choice(VAR_F[fl]) if rng.random() > 0.02 else "maybe", ans, "备注" if rng.random() < 0.1 else ""])
    write(f"labels_A_{a}.csv", ra, ["hid", "label", "reading_sols", "note"], bom)
    write(f"labels_B_{a}.csv", rb, ["hid", "flag", "answer", "note"], bom, enc=None if a == "ann1" else "gb18030")
write("labels_A_adjudicated.csv", [[key[0]["hid"], "R", "", ""]], ["hid", "label", "reading_sols", "note"])
r = subprocess.run([sys.executable, f"{OUT}/h2_read.py", "--dev", DEV], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
o = json.load(open(f"{DEV}/h2_read_dev.json"))
need = ["inputs", "H2a_primary", "H2a_per_annotator", "H2b_primary_U", "H2b_per_annotator_U", "H2b_answerable_pairs", "H2c_agreement",
        "H2a_consensus", "H2b_consensus_U", "H2c_consensus_vs_llm", "n_disagreements", "fig3_U", "fig3_A"]
assert all(k in o for k in need), [k for k in need if k not in o]
assert o["inputs"]["included_A"] == ["ann1", "ann2"] and o["inputs"]["included_B"] == ["ann1", "ann2"], o["inputs"]
assert abs(o["H2a_primary"]["D_h"] - o["H2a_primary"]["D_llm_same_items"]) < 0.35
assert abs(o["H2b_primary_U"]["rise_h"] - o["H2b_primary_U"]["rise_llm_same_items"]) < 0.25
print("DEV TEST OK (simulated labels; not a result)")
