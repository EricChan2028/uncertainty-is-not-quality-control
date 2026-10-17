"""Zero-data proposer, R-Zero style: the base model writes grade-school math word problems from stems.

Output: JSONL of {problem, stem} for c0_probe.py --items_file. No answers and no verification - the
point is to measure what a real self-play proposer emits and which of its items the published guards
would admit, alongside the planted premise-deleted items that carry ground truth.
"""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu_util", type=float, default=0.85)
args = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402

STEMS = ["oranges", "train tickets", "a bakery", "a swimming pool", "a farm with chickens", "a school library",
         "a painter", "a road trip", "a lemonade stand", "a bookshelf", "a marathon", "a garden", "a parking lot",
         "a pizza order", "a piggy bank", "a classroom", "a factory shift", "a concert", "a bus route", "a recipe",
         "a bicycle shop", "a movie theater", "a zoo", "a coin collection"]
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1536, dtype="bfloat16")
tok = llm.get_tokenizer()
chat = tok.chat_template is not None and "Base" not in args.model


def wrap(t):
    if chat:
        return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
    return f"Task: {t}\nProblem:"


prompts = []
for i in range(args.n):
    st = STEMS[i % len(STEMS)]
    prompts.append(wrap(f"Write one new, challenging grade-school math word problem about {st}. It must require "
                        f"several arithmetic steps and have a single numeric answer. Output only the problem text, "
                        f"no solution."))
outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=0.95, max_tokens=300, n=1, seed=11))
n = 0
with open(args.out + ".tmp", "w") as fh:
    for i, o in enumerate(outs):
        t = o.outputs[0].text.strip()
        t = re.split(r"\n\s*(Solution|Answer|Step 1)\b", t)[0].strip()
        if len(t) < 40 or "?" not in t:
            continue
        fh.write(json.dumps({"problem": t, "stem": STEMS[i % len(STEMS)]}, ensure_ascii=False) + "\n")
        n += 1
os.replace(args.out + ".tmp", args.out)  # atomic: a killed proposer leaves no half file
print("PROPOSED", n, args.out)
