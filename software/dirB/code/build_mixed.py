"""A2 training set: MATH-500 (dirB) + GSM8K-train-500 (dirC), both unlabeled at training time, both with
the boxed instruction so one prompt format serves the whole set. Verified for control characters."""
import hashlib
import json

R = "/home/tione/notebook/rsi_rl"
SUF = " Let's think step by step and output the final answer within \\boxed{}."
assert "\x08" not in SUF and "\\boxed" in SUF
rows = []
for r in (json.loads(l) for l in open(f"{R}/dirB/data/math500_train.jsonl")):
    rows.append({"id": r["id"], "problem": r["problem"], "prompt": r["prompt"], "answer": r["answer"], "src": "math500"})
for r in (json.loads(l) for l in open(f"{R}/dirC/data/gsm8k_train500.jsonl")):
    rows.append({"id": r["id"], "problem": r["problem"], "prompt": r["problem"].strip() + SUF, "answer": r["answer"], "src": "gsm8k_train"})
path = f"{R}/dirB/data/mixed1000_train.jsonl"
with open(path, "w") as fh:
    for r in rows:
        assert r["prompt"].startswith(r["problem"].strip()), r["id"]
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
back = [json.loads(l) for l in open(path)]
bad = sum(1 for r in back for v in r.values() if isinstance(v, str) and any(ord(c) < 32 and c not in "\n\t" for c in v))
assert bad == 0 and len(back) == 1000
print("mixed1000_train", len(back), hashlib.sha256(open(path, "rb").read()).hexdigest()[:16], repr(back[600]["prompt"][-70:]))
