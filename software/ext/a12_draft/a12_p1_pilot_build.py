"""A12 part-1 pilot (before hashing; registration section 1.5): 50 dummy items with NO R-Zero or project content (templated
arithmetic questions written here, every fifth with a needed quantity removed), keys q0000..q0049 (the letter q, not the
real items' y, so that no pilot file ever holds A12 keys and the pre-launch scan for y-keys is not tripped), in the format of
the registered items file, at dirA5/results/a12_pilot/items.jsonl (pre-hash, non-study output). The pilot runs
a12_p1_blind_wf.js with {pass: 1, pilot: true} (2 judges) and a12_blind_check.py with "pilot", to test the chain on real
transcripts. Its verdicts are never read. After the pilot, its items file, check file and workflow output are deleted and
the deletion is recorded in the hash-log entry of the package.
    python3 a12_p1_pilot_build.py [out_dir]"""
import json, os, random, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/tione/notebook/rsi_rl/dirA5/results/a12_pilot"
os.makedirs(OUT, exist_ok=True)
rng = random.Random(7)
names = ["Ana", "Ben", "Chloe", "Dev", "Eli", "Fay", "Gus", "Hana"]
items = []
for k in range(50):
    a, b, c = rng.randint(3, 40), rng.randint(2, 9), rng.randint(2, 30)
    n = rng.choice(names)
    if k % 5 == 4:  # a needed quantity is missing
        q = f"{n} buys some boxes of pencils with {b} pencils in each box and then gives away {c} pencils. How many pencils are left?"
    elif k % 5 == 3:
        q = f"A rectangle has perimeter {2 * (a + b)} and one side of length {a}. What is its area?"
    else:
        q = f"{n} has {a} marbles, buys {b} bags of {c} marbles each, and loses {min(a, c)} marbles. How many marbles does {n} have now?"
    items.append({"key": f"q{k:04d}", "problem": q})
with open(f"{OUT}/items.jsonl", "w", encoding="utf-8") as f:
    for x in items:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(len(items), f"{OUT}/items.jsonl")
