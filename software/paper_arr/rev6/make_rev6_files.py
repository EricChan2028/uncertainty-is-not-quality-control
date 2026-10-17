"""Create the rev6 id list and the text-saving probe copy with the A11-corrected abstention instruction."""
import json
R = "/home/tione/notebook/rsi_rl"
D = [json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl")]
json.dump({"U": [x["id"] for x in D], "hard_A": []}, open(f"{R}/paper_arr/rev6/ids_all.json", "w"))
src = open(f"{R}/dirA3/code/umwp_probe_text.py").read()
lines = src.split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("SOLVE_ABS = ")]
assert len(idx) == 1, idx
old = lines[idx[0]]
assert old.startswith('SOLVE_ABS = "') and "boxed" in old, old
lines[idx[0]] = "SOLVE_ABS = r" + old[len("SOLVE_ABS = "):] + "  # A11 correction: raw string"
assert lines[0].startswith('"""A8 text-saving copy')
lines[0] = '"""rev6 copy of umwp_probe_text.py with the A11-corrected abstention instruction (raw string); otherwise identical.\n' + lines[0][3:]
open(f"{R}/paper_arr/rev6/umwp_probe_text_absfix.py", "w").write("\n".join(lines))
abs_ref = [l for l in open(f"{R}/ext/a11_draft/umwp_probe_absfix.py").read().split("\n") if l.startswith("SOLVE_ABS = ")][0]
print(len(D), "ids")
print("new :", lines[idx[0]])
print("A11 :", abs_ref)
