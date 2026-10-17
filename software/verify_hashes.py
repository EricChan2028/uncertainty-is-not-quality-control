#!/usr/bin/env python3
"""Recompute the SHA-256 of every packaged file that the registration log registers.

Unpack software.zip and data.zip side by side, then, from the software folder:
    python3 verify_hashes.py            # looks for the data folder next to it
    python3 verify_hashes.py --data /path/to/data
Inputs: docs/registration_hashes.txt (the anonymised copy of the log; hashes, paths and timestamps are unchanged
except where REDACTIONS.tsv says otherwise) and preregistration/PATHMAP.tsv.
"""
import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
args = ap.parse_args()

REG = defaultdict(list)
PAIR = re.compile(r"\b([0-9a-f]{64})\s{1,4}(/?[\w.-]+(?:/[\w.()-]+)*)")
for line in open(os.path.join(HERE, "docs", "registration_hashes.txt"), encoding="utf-8", errors="replace"):
    for m in PAIR.finditer(line.strip()):  # "hash  path" pairs, at the start of a line or inline after a header
        p = m.group(2).strip()
        if "/" in p or "." in p:
            REG[re.sub(r"^/home/tione/notebook/rsi_rl/", "", p)].append(m.group(1))

counts = Counter()
differs = []
with open(os.path.join(HERE, "preregistration", "PATHMAP.tsv"), encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        base = HERE if r["package"] == "software" else args.data
        p = os.path.join(base, r["package_path"])
        if not os.path.exists(p):
            counts["not found (is the other archive unpacked next to this one?)"] += 1
            continue
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if h != r["packaged_sha256"]:
            counts["CHANGED since packaging"] += 1
            differs.append(r["package_path"])
            continue
        hs = REG.get(r["registered_path"], [])
        if hs and h == hs[-1]:
            counts["verbatim, matches its last registration"] += 1
        elif h in hs:
            counts["verbatim, matches an earlier registration"] += 1
        elif r["status"].startswith("anonymised"):
            counts["anonymised copy (differs by design; see REDACTIONS.tsv)"] += 1
        else:
            counts["verbatim, edited after its registration"] += 1
            differs.append(r["registered_path"])

for k, v in counts.most_common():
    print(f"{v:6d}  {k}")
if differs:
    print("\nfiles that match no registered hash and are not anonymised copies:")
    for d in differs:
        print("   ", d)
