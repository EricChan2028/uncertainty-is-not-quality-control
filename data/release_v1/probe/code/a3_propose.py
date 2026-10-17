"""A3 loop, step 1: a frozen proposer writes N grade-school math word problems (R-Zero-style stems), one
round at a time (the sampling seed encodes the round so every round sees fresh problems). Output JSONL
{id, problem, prompt, stem} in the training-set format of b_train.py (no answer field: the loop is
label-free). Planted tracers are added later by a3_guard.py.

  python3 a3_propose.py --model <path> --out <round_dir>/proposed.jsonl --n 1200 --seed <seed*100+round>
"""
import argparse
import json
import os
import re

R = __import__("os").environ.get("RSI_REPO", "<REPO>")
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=1200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--no_think", type=int, default=0)
ap.add_argument("--tp", type=int, default=1)
args = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402

STEMS = ["oranges", "train tickets", "a bakery", "a swimming pool", "a farm with chickens", "a school library", "a painter",
         "a road trip", "a lemonade stand", "a bookshelf", "a marathon", "a garden", "a parking lot", "a pizza order",
         "a piggy bank", "a classroom", "a factory shift", "a concert", "a bus route", "a recipe", "a bicycle shop",
         "a movie theater", "a zoo", "a coin collection", "a phone plan", "a fish tank", "a car wash", "a summer camp",
         "a bookstore sale", "a laundromat", "a football season", "a carpenter", "a hotel", "a museum visit",
         "a paper route", "a bake sale", "a taxi ride", "a science fair", "a comic shop", "a music lesson"]
SUF = " Let's think step by step and output the final answer within \\boxed{}."
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1536, dtype="bfloat16", tensor_parallel_size=args.tp)
tok = llm.get_tokenizer()
base = "Base" in args.model


def wrap(t):
    if base:
        return f"Task: {t}\nProblem:"
    return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, **({"enable_thinking": False} if args.no_think else {}))


prompts = [wrap(f"Write one new, challenging grade-school math word problem about {STEMS[i % len(STEMS)]}. It must require "
                f"several arithmetic steps and have a single numeric answer. Output only the problem text, no solution.")
           for i in range(args.n)]
# one distinct seed per request: with a single seed, the 30 identical prompts per stem produce identical
# problems and the round collapses to ~126 unique items (found 2026-09-22 10:05)
outs = llm.generate(prompts, [SamplingParams(temperature=1.0, top_p=0.95, max_tokens=300, n=1, seed=args.seed * 10000 + i) for i in range(len(prompts))])
os.makedirs(os.path.dirname(args.out), exist_ok=True)
seen, n = set(), 0
with open(args.out + ".tmp", "w") as fh:
    for i, o in enumerate(outs):
        t = o.outputs[0].text.strip()
        t = re.split(r"\n\s*(Solution|Answer|Step 1|Problem)\b", t)[0].strip().strip('"')
        t = re.sub(r"\s+", " ", t)
        if len(t) < 40 or "?" not in t or t in seen or not re.search(r"\d", t):
            continue
        seen.add(t)
        fh.write(json.dumps({"id": f"prop{args.seed}:{n}", "problem": t, "prompt": t + SUF, "stem": STEMS[i % len(STEMS)]}, ensure_ascii=False) + "\n")
        n += 1
os.replace(args.out + ".tmp", args.out)
print("PROPOSED", n, args.out, flush=True)
