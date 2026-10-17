"""Development test of h21_read.py on SIMULATED labels only (run before any v2 human label exists). Three scenarios in
/tmp/h21_dev_{ok,copy,wrongver}: (ok) each annotator's file built from the LLM labels of its own version's items with 15% of
entries replaced at random, plus formatting variants and a GB18030-encoded file; (copy) annotator 2's files are annotator 1's
labels row by row under annotator 2's ids, which the I2 copy check must catch; (wrongver) annotator 2's part-A file uses
annotator 1's ids, which the I1 check must catch. python3 h21_dev_test.py"""
import csv, json, os, random, subprocess, sys

R = "/home/tione/notebook/rsi_rl"; HV = f"{R}/dirA5/human_validation_mech"
key = json.load(open(f"{HV}/_key_do_not_open/key_v2.json"))
LLM = {x["key"]: x for x in json.load(open(f"{R}/dirA5/results/a8/labels_main.json")) if isinstance(x, dict) and "key" in x}
VAR_A = {"N": ["N", "n", "none"], "R": ["R", "r", "Ｒ", "readings"], "E": ["E", "e", "error"], "M": ["M", "m", "mixed"]}
VAR_F = {1: ["Y", "y", "yes", "Ｙ"], 0: ["N", "n", "no"]}

def sim(n, rng):
    rows = {"A": [], "B": []}
    for x in sorted((x for x in key if x["version"] == f"ann{n}"), key=lambda x: (x["part"], x["order"])):
        l = LLM[x["a8_key"]]
        if x["part"] == "A":
            lab = l["disagreement_source"][0]
            if rng.random() < 0.15: lab = rng.choice("NREM")
            rows["A"].append([x["hid"], rng.choice(VAR_A[lab]), "", ""])
        else:
            fl = int(l["gap_flag"][x["sample"]])
            if rng.random() < 0.15: fl = 1 - fl
            rows["B"].append([x["hid"], rng.choice(VAR_F[fl]), rng.choice(["NUM", "num", "DECLINE", "NONE"]), "备注" if rng.random() < 0.05 else ""])
    return rows

def write(d, name, rows, header, enc="utf-8-sig"):
    with open(f"{d}/{name}", "w", newline="", encoding=enc) as f:
        w = csv.writer(f); w.writerow(header); [w.writerow(r) for r in rows]

HA, HB = ["hid", "label", "reading_sols", "note"], ["hid", "flag", "answer", "note"]
res = {}
for scen in ("ok", "copy", "wrongver"):
    d = f"/tmp/h21_dev_{scen}"; os.makedirs(d, exist_ok=True)
    r1, r2 = sim(1, random.Random(11)), sim(2, random.Random(22))
    if scen == "copy":
        r2 = {p: [[f"{p}2-{row[0].split('-')[1]}"] + row[1:] for row in r1[p]] for p in ("A", "B")}
    write(d, "labels_A_ann1.csv", r1["A"], HA); write(d, "labels_B_ann1.csv", r1["B"], HB)
    write(d, "labels_A_ann2.csv", r1["A"] if scen == "wrongver" else r2["A"], HA)
    write(d, "labels_B_ann2.csv", r2["B"], HB, enc="gb18030")
    p = subprocess.run([sys.executable, f"{HV}/h21_read.py", "--dev", d], capture_output=True, text=True)
    print(f"--- {scen}\n", p.stdout[-1500:], p.stderr[-1500:])
    res[scen] = json.load(open(f"{d}/h21_read_dev.json"))
ok, cp, wv = res["ok"], res["copy"], res["wrongver"]
assert ok["inputs"]["included_A"] == ["ann1", "ann2"] and ok["inputs"]["included_B"] == ["ann1", "ann2"]
assert ok["H2a_primary"]["n_U"] == 92 and ok["H2a_primary"]["n_A"] == 92 and ok["H2b_primary_U"]["n_questions"] == 112
assert abs(ok["H2a_primary"]["D_h"] - ok["H2a_primary"]["D_llm_same_items"]) < 0.25
assert abs(ok["H2b_primary_U"]["rise_h"] - ok["H2b_primary_U"]["rise_llm_same_items"]) < 0.20
assert not ok["inputs"]["integrity"]["I2_A"]["copies"] and not ok["inputs"]["integrity"]["I2_B"]["copies"]
assert cp["inputs"]["integrity"]["I2_A"]["copies"] and cp["inputs"]["integrity"]["I2_B"]["copies"] and cp["H2a_primary"] is None
assert not wv["inputs"]["integrity"]["I1_A_ann2"]["ok"] and wv["inputs"]["included_A"] == ["ann1"]
print("DEV TEST OK (simulated labels; not a result)")
