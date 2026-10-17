"""A6 E staging: MiP-GSM8K (582 pairs), MiP-MATH (52 pairs) from github.com/tianyi-lab/MiP-Overthinking, and SUM test (284
unanswerable; lime-nlp/Synthetic_Unanswerable_Math via hf-mirror; its built-in "I don't know" suffix is stripped so that our
own plain / abstention prompts apply). Writes data_ext/a6/sweep.jsonl with id, set, question, answerable, answer.
python3 a6_stage.py"""
import json, os, re
import pandas as pd
D = "/home/tione/notebook/rsi_rl/data_ext"; os.makedirs(f"{D}/a6", exist_ok=True)
rows = []
for k, x in enumerate(json.load(open(f"{D}/MiP/gsm8k.json"))):
    ans = x["answer"].rsplit("####", 1)[-1].strip().replace(",", "")
    rows.append({"id": f"mipg:{k}:u", "set": "mip_gsm8k", "question": x["insufficient_question"].strip(), "answerable": False, "answer": None})
    rows.append({"id": f"mipg:{k}:a", "set": "mip_gsm8k", "question": x["question"].strip(), "answerable": True, "answer": [ans]})
for k, x in enumerate(json.load(open(f"{D}/MiP/math.json"))):
    rows.append({"id": f"mipm:{k}:u", "set": "mip_math", "question": x["insufficient_question"].strip(), "answerable": False, "answer": None})
    rows.append({"id": f"mipm:{k}:a", "set": "mip_math", "question": x["question"].strip(), "answerable": True, "answer": [x["answer"]]})
d = pd.read_parquet(f"{D}/SUM/test.parquet")
for k, p in enumerate(d["prompt"]):
    q = p[: p.find("Let's think step by step")].strip()
    rows.append({"id": f"sum:{k}:u", "set": "sum", "question": q, "answerable": False, "answer": None})
with open(f"{D}/a6/sweep.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), Counter((r["set"], r["answerable"]) for r in rows))
