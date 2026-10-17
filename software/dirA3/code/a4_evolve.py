"""A4e frontier refill: the frozen proposer rewrites the questions a solver answers unanimously (p_mode == 1 in its guard
records) into harder variants. Output: a candidate pool in a3_propose.py format (id, problem, prompt, stem).
  python3 a4_evolve.py --model <proposer> --pool <proposed.jsonl> --records <guard_records.jsonl> --out <pool.jsonl> [--k 2]
"""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pool", required=True)
ap.add_argument("--records", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="evo")
ap.add_argument("--gpu_util", type=float, default=0.85)
args = ap.parse_args()
SUF = " Let's think step by step and output the final answer within \\boxed{}."
pool = {json.loads(l)["id"]: json.loads(l) for l in open(args.pool)}
rec = [json.loads(l) for l in open(args.records)]
seeds = [pool[r["id"]] for r in rec if r["kind"] == "proposed" and r["p_mode"] >= 1.0 and r["id"] in pool]
print("seeds (unanimous):", len(seeds), flush=True)
FEW = ("Original problem: A baker makes 24 muffins in the morning and 18 in the afternoon. He sells 30 of them. How many muffins are left?\n"
       "Harder problem (adds one or two more steps, keeps every number needed, single numeric answer): A baker makes 24 muffins in the "
       "morning and 18 in the afternoon. He sells 30 of them at $2 each and gives half of the remaining muffins to a shelter. How much money "
       "did he make, and how many muffins does he have left? Give the number of muffins left.\n\n"
       "Original problem: Sara reads 15 pages a day for 6 days. How many pages does she read?\n"
       "Harder problem (adds one or two more steps, keeps every number needed, single numeric answer): Sara reads 15 pages a day for 6 days, "
       "then 20 pages a day for the next 4 days. Her book has 250 pages. How many pages does she still have to read?\n\n")
from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048, dtype="bfloat16")
prompts = [FEW + f"Original problem: {s['problem']}\nHarder problem (adds one or two more steps, keeps every number needed, single numeric answer):" for s in seeds]
outs = llm.generate(prompts, SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300, n=args.k, stop=["\n\n", "\nOriginal problem:"], seed=args.seed))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
seen, n = set(), 0
with open(args.out, "w") as fh:
    for s, o in zip(seeds, outs):
        for c in o.outputs:
            t = re.sub(r"\s+", " ", c.text.strip().strip('"'))
            if len(t) < 40 or "?" not in t or t in seen or not re.search(r"\d", t) or t == s["problem"]:
                continue
            seen.add(t)
            fh.write(json.dumps({"id": f"{args.tag}:{n}", "problem": t, "prompt": t + SUF, "stem": s.get("stem", ""), "seed_id": s["id"]}, ensure_ascii=False) + "\n")
            n += 1
print("EVOLVED", n, args.out, flush=True)
