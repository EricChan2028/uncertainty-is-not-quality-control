"""A10 (registered 01:13 25 Sep): open-weight judge (Qwen2.5-32B-Instruct, vLLM TP 2, greedy, max 400 new tokens, one item per
request) on the 900 A7b blind items (dirA5/results/a7b_blind/items.jsonl: key + problem only). Frozen audit definitions; the
last "VERDICT: X" line is parsed; unparsable items are re-asked once (temperature 0.7, seed 1); still unparsable -> missing.
Output dirA5/results/a10/verdicts.jsonl (key, verdict or null, note, raw tail).  CUDA_VISIBLE_DEVICES=a,b python3 a10_judge.py"""
import json, os, re
from vllm import LLM, SamplingParams
R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA5/results/a10"; os.makedirs(OUT, exist_ok=True)
items = [json.loads(l) for l in open(f"{R}/dirA5/results/a7b_blind/items.jsonl")]
INSTR = ("You are a careful mathematician auditing self-generated training questions.\n"
         "This question was written by a Qwen3-4B-Base language model and ADMITTED by a self-play filter. Decide:\n"
         "  SOLVABLE    - the text determines a unique numeric answer (state it briefly in the note);\n"
         "  UNSOLVABLE  - a needed quantity/relation is missing, or the text is self-contradictory, so no unique answer exists;\n"
         "  AMBIGUOUS   - two reasonable readings give different answers.\n"
         "Be strict: a default assumption you would have to invent makes it UNSOLVABLE.\n"
         "Reply with a note of at most 20 words, then a final line exactly of the form\nVERDICT: SOLVABLE\nor\nVERDICT: UNSOLVABLE\nor\nVERDICT: AMBIGUOUS\n\n"
         "Question:\n")
llm = LLM(model="/dev/shm/models/Qwen2.5-32B-Instruct", tensor_parallel_size=2, gpu_memory_utilization=0.85, max_model_len=4096, dtype="bfloat16")
tok = llm.get_tokenizer()
def prompt(p): return tok.apply_chat_template([{"role": "user", "content": INSTR + p.strip()}], tokenize=False, add_generation_prompt=True)
RE = re.compile(r"VERDICT:\s*\**\s*(SOLVABLE|UNSOLVABLE|AMBIGUOUS)", re.I)
def parse(t):
    m = RE.findall(t); return m[-1].upper() if m else None
outs = llm.generate([prompt(x["problem"]) for x in items], SamplingParams(temperature=0.0, max_tokens=400))
res = {}
for x, o in zip(items, outs):
    t = o.outputs[0].text; res[x["key"]] = {"key": x["key"], "verdict": parse(t), "raw_tail": t[-300:], "retry": False}
miss = [x for x in items if res[x["key"]]["verdict"] is None]
if miss:
    outs2 = llm.generate([prompt(x["problem"]) for x in miss], SamplingParams(temperature=0.7, max_tokens=400, seed=1))
    for x, o in zip(miss, outs2):
        t = o.outputs[0].text; res[x["key"]] = {"key": x["key"], "verdict": parse(t), "raw_tail": t[-300:], "retry": True}
with open(f"{OUT}/verdicts.jsonl", "w") as f:
    for x in items: f.write(json.dumps(res[x["key"]]) + "\n")
print("done", len(res), "missing", sum(v["verdict"] is None for v in res.values()), "retried", len(miss))
