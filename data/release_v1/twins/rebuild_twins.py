#!/usr/bin/env python3
"""Rebuild the GSM-HP premise-deletion twins (intact + deleted texts) from your own copy of GSM-Symbolic.

The twins are adaptations of GSM-Symbolic (Apple; CC BY-NC-ND 4.0), which does not permit sharing adapted material, so this
release ships only patches: each text is a list of copy operations [start, end] into the GSM-Symbolic question it was built
from, plus short literals (numbers, number words, names; about 2% of the characters). Every rebuilt text is checked
against the SHA-256 stored in twin_patches.jsonl.

  pip install datasets   (or download main/p1/p2 test.jsonl of apple/GSM-Symbolic at the pinned revision)
  python3 rebuild_twins.py --patches twin_patches.jsonl --out twins_rebuilt.jsonl [--gsm_symbolic_dir DIR]

Output: one row per twin, in the order of the project's data/hp_twins.jsonl, with the fields c0_probe.py reads: fam, task,
split, split_kind, family_id_deleted, family_id_intact, answer (the intact problem's answer), intact, deleted, meta; plus
answer_intact (= answer), in_probe543, probe543_idx and canary (copied from GSM-Symbolic). The file is a drop-in
--twins_file for probe/code/c0_probe.py (its probe set = task 0 of every family + every template-split task = the 543
registered items, in probe543_idx order). Not rebuilt: the project file's deleted_sentence and meta.H_sentence (text).
Keep the canary: like GSM-Symbolic itself, the rebuilt twins should not appear in training corpora.
"""
import argparse, hashlib, json, os

REV = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
ap = argparse.ArgumentParser()
ap.add_argument("--patches", default="twin_patches.jsonl")
ap.add_argument("--out", default="twins_rebuilt.jsonl")
ap.add_argument("--gsm_symbolic_dir", default="", help="directory with main/test.jsonl, p1/test.jsonl, p2/test.jsonl")
a = ap.parse_args()
G = {}
for cfg in ("main", "p1", "p2"):
    if a.gsm_symbolic_dir:
        rows = [json.loads(l) for l in open(os.path.join(a.gsm_symbolic_dir, cfg, "test.jsonl"), encoding="utf-8")]
    else:
        from datasets import load_dataset
        rows = load_dataset("apple/GSM-Symbolic", cfg, revision=REV)["test"]
    for x in rows:
        G[(cfg, int(x["id"]), int(x["instance"]))] = x
bad = 0
with open(a.out, "w", encoding="utf-8") as fh:
    for l in open(a.patches, encoding="utf-8"):
        t = json.loads(l)
        s = t["source"]; x = G[(s["config"], s["id"], s["instance"])]
        src = x["question"]
        txt = {}
        for form in ("intact", "deleted"):
            txt[form] = "".join(src[o[0]:o[1]] if isinstance(o, list) else o for o in t[form + "_ops"])
            bad += hashlib.sha256(txt[form].encode("utf-8")).hexdigest() != t[form + "_sha256"]
        out = {"fam": t["fam"], "task": t["task"], "split": t["split"], "split_kind": t["split_kind"],
               "family_id_deleted": t["family_id_deleted"], "family_id_intact": t["family_id_intact"],
               "answer": t["answer_intact"], "intact": txt["intact"], "deleted": txt["deleted"], "meta": t["meta"],
               "answer_intact": t["answer_intact"], "in_probe543": t["in_probe543"], "probe543_idx": t["probe543_idx"],
               "canary": x.get("canary")}
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
print("rebuilt; sha256 mismatches:", bad)
raise SystemExit(1 if bad else 0)
