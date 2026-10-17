"""A4h external validity probe on UMWP (Sun et al., LREC-COLING 2024; CC-BY-SA-4.0): 600 unanswerable questions (Random(0)
sample of the 2,600) and their answerable originals (relevant_ids). Prompt, answer parsing and aggregation are copied from
dirA/code/c0_probe.py (Base models: "Question: ...\nAnswer: Let's think step by step."; plain SOLVE instruction, or the
abstention suffix + boxed instruction with --abstain_prompt 1). G = 8 samples, T = 1.0, max_new 1024, seed 17.
Output: dirA4/results/umwp/<tag>.jsonl (one row per question: id, answerable, category, source, ref, mode, p_mode,
nonanswer, abstain, correct).   python3 umwp_probe.py --model M --tag T [--abstain_prompt 1]
"""
import argparse, json, math, os, random, re, sys
from collections import Counter
sys.path.insert(0, "/home/tione/notebook/rsi_rl/code")
from loto_common import last_boxed, norm_num  # noqa: E402
sys.path.insert(0, "/home/tione/notebook/rsi_rl/dirB/code")
from pdc_util import ABSTAIN_SUFFIX, is_abstain  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--abstain_prompt", type=int, default=0); ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--n_uns", type=int, default=600)
args = ap.parse_args()
R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA4/results/umwp"; os.makedirs(OUT, exist_ok=True)
D = [json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl")]
byid = {x["id"]: x for x in D}
uns = [x for x in D if not x["answerable"] and x.get("relevant_ids") and all(i in byid and byid[i]["answerable"] for i in x["relevant_ids"])]
uns = random.Random(0).sample(sorted(uns, key=lambda x: x["id"]), args.n_uns)
ans_ids = sorted({x["relevant_ids"][0] for x in uns})
items = uns + [byid[i] for i in ans_ids]
outp = f"{OUT}/{args.tag}.jsonl"
done = set()
if os.path.exists(outp):
    for l in open(outp):
        try: done.add(json.loads(l)["id"])
        except ValueError: pass
items = [x for x in items if x["id"] not in done]
print(f"{len(uns)} unanswerable + {len(ans_ids)} answerable; {len(items)} to do", flush=True)
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
def _extract(text):
    m = _NUM_RE.findall(text); return norm_num(m[-1]) if m else None
from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048 + args.max_new, dtype="bfloat16", enable_prefix_caching=True)
tok = llm.get_tokenizer(); is_chat = tok.chat_template is not None and "Base" not in args.model
SOLVE = "Solve the problem. Show your reasoning briefly, then give the final numeric answer on the last line as 'Answer: <number>'."
SOLVE_ABS = "Solve the problem step by step and output the final answer within \boxed{}."
if args.abstain_prompt: SOLVE = SOLVE_ABS
def wrap(problem, instruction):
    if args.abstain_prompt: problem = problem + ABSTAIN_SUFFIX
    text = f"{problem}\n\n{instruction}"
    if is_chat: return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return f"Question: {text}\nAnswer: Let's think step by step."
def norm_ans(a):
    if a is None: return None
    try:
        v = float(str(a).replace(",", "")); return int(v) if abs(v - round(v)) < 1e-9 else round(v, 4)
    except Exception: return None
def answers_of(texts, finish):
    out = []
    for t, f in zip(texts, finish):
        if f == "length": out.append(None); continue
        if args.abstain_prompt:
            b = last_boxed(t)
            if b is not None and is_abstain(b): out.append(None); continue
            if b is not None: out.append(norm_ans(b)); continue
            if re.search(r"Answer:\s*(unknown|unanswerable|insufficient information)", t, re.I): out.append(None); continue
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", t)
        out.append(norm_ans(m[-1] if m else _extract(t)))
    return out
sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, seed=17)
with open(outp, "a") as fh:
    for bi in range(0, len(items), 128):
        batch = items[bi:bi + 128]
        outs = llm.generate([wrap(x["question"].strip(), SOLVE) for x in batch], sp)
        for x, o in zip(batch, outs):
            texts = [c.text for c in o.outputs]; fin = [c.finish_reason for c in o.outputs]
            ans = answers_of(texts, fin); c = Counter(ans); mode, n = c.most_common(1)[0]
            ref = norm_ans(x["answer"][0]) if x.get("answer") else None
            fh.write(json.dumps({"id": x["id"], "answerable": x["answerable"], "category": x.get("category"), "source": x.get("source"),
                                 "ref": ref, "mode": mode, "p_mode": n / len(ans), "nonanswer": sum(a is None for a in ans) / len(ans),
                                 "abstain": sum(1 for t in texts if is_abstain(last_boxed(t) or "") or re.search(r"Answer:\s*(unknown|unanswerable|insufficient information)", t, re.I)) / len(texts),
                                 "correct": int(mode is not None and ref is not None and mode == ref), "answers": ans}) + "\n")
        fh.flush()
print("done", flush=True)
