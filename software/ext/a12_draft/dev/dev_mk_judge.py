"""A12 provenance: derives a12_open_judge.py from the registered A10 judge (dirA5/code/a10_judge.py, sha256
68b7b3154d42d808521c1bf08cdd359ac14ddd2e0d961c3c2f5a59d9aba92664). Every replacement must match exactly once. The prompt, the
model, tensor parallelism, decoding, parsing and the single retry are byte-identical; only the docstring, the input (the 1,158
fixed-pool proposals, key = proposal id), the output directory, a refusal to overwrite and an atomic final write change.
    python3 dev_mk_judge.py <a10_judge.py> <out a12_open_judge.py>"""
import hashlib, sys

SRC, DST = sys.argv[1:3]
b = open(SRC, "rb").read()
assert hashlib.sha256(b).hexdigest() == "68b7b3154d42d808521c1bf08cdd359ac14ddd2e0d961c3c2f5a59d9aba92664", "not the registered A10 judge"
s = b.decode("utf-8")
first_doc_end = s.index('"""', 3) + 3
REPL = [
    (s[:first_doc_end],
     '"""A12 part 3 (A12_registration_2026-09-26.md, section 3): the registered A10 open-weight judge (a10_judge.py; Qwen2.5-32B-Instruct,\n'
     'vLLM TP 2, greedy, max 400 new tokens, one item per request, same prompt) on the 1,158 fixed-pool proposals\n'
     '(dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl; key = proposal id, text = "problem"). The last "VERDICT: X" line is parsed;\n'
     'unparsable items are re-asked once (temperature 0.7, seed 1); still unparsable -> missing. Output\n'
     'dirA5/results/a12_open/verdicts.jsonl (key, verdict or null, raw tail, retry), written once, atomically; refuses to overwrite.\n'
     'Derived from a10_judge.py by dev_mk_judge.py.  CUDA_VISIBLE_DEVICES=a,b /usr/bin/python3 a12_open_judge.py"""'),
    ('OUT = f"{R}/dirA5/results/a10"; os.makedirs(OUT, exist_ok=True)',
     'OUT = f"{R}/dirA5/results/a12_open"; os.makedirs(OUT, exist_ok=True)\n'
     'assert not os.path.exists(f"{OUT}/verdicts.jsonl"), "A12 part-3 verdicts exist: never overwritten"'),
    ('items = [json.loads(l) for l in open(f"{R}/dirA5/results/a7b_blind/items.jsonl")]',
     'items = [{"key": r["id"], "problem": r["problem"]} for r in (json.loads(l) for l in\n'
     '         open(f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl", encoding="utf-8"))]\n'
     'assert len(items) == 1158 and len({x["key"] for x in items}) == 1158'),
    ('with open(f"{OUT}/verdicts.jsonl", "w") as f:\n    for x in items: f.write(json.dumps(res[x["key"]]) + "\\n")',
     'with open(f"{OUT}/verdicts.jsonl.tmp", "w") as f:\n    for x in items: f.write(json.dumps(res[x["key"]]) + "\\n")\n'
     'os.replace(f"{OUT}/verdicts.jsonl.tmp", f"{OUT}/verdicts.jsonl")'),
]
for old, new in REPL:
    assert s.count(old) == 1, ("replacement must match exactly once", old[:80], s.count(old))
    s = s.replace(old, new, 1)
open(DST, "wb").write(s.encode("utf-8"))
print("wrote", DST, hashlib.sha256(s.encode("utf-8")).hexdigest())
