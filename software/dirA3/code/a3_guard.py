"""A3 loop, step 2: the published admission guard, applied by the CURRENT solver to the proposer's
candidates plus planted tracers. For every candidate the solver samples G answers (vLLM, T=1); the
R-Zero band admits p_mode in [lo, hi] (majority agreement of the solver's own samples). Tracers are
GSM-HP twins OUTSIDE the registered 543-item probe set (so the evaluation set stays clean): 200
premise-deleted problems (expected ~76% ill-posed) and their 200 intact twins, unlabeled, mixed into
the pool. Output: <round_dir>/admitted.jsonl (training set in b_train.py format, tracers included when
admitted, answer field omitted) and <round_dir>/guard_stats.json.

  python3 a3_guard.py --model <current solver> --candidates <proposed.jsonl> --out_dir <round_dir> --seed <s>
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

R = "/home/tione/notebook/rsi_rl"
sys.path.insert(0, f"{R}/code")
sys.path.insert(0, f"{R}/dirB/code")
from loto_common import _strip_math, last_boxed, norm_num  # noqa: E402
from pdc_util import ABSTAIN_SUFFIX, is_abstain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--candidates", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--guard", choices=["rzero", "azr", "tracer", "abstain"], default="rzero", help="A3e: rzero band | azr learnability (0,1) | tracer-calibrated band")
ap.add_argument("--lo", type=float, default=0.25)
ap.add_argument("--hi", type=float, default=0.75)
ap.add_argument("--n_tracer", type=int, default=200)
ap.add_argument("--abs_max", type=float, default=0.25, help="A4 abstention guard: reject a candidate when >= this share of its G samples abstain (needs --abstain_prompt 1)")
ap.add_argument("--abstain_prompt", type=int, default=0, help="1: the solver of an abstention-aware arm sees the abstention option")
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--tp", type=int, default=1, help="A4a: tensor parallel size")
ap.add_argument("--count_unknown", type=int, default=0, help="A4i: admit on the majority share counting unknown as an answer, max(p_mode, abstain)")
ap.add_argument("--no_think", type=int, default=0, help="A4a: chat template with enable_thinking=False (Qwen3 post-trained)")
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)
SUF = " Let's think step by step and output the final answer within \\boxed{}."

# ---- pool: proposer candidates + tracers (deterministic per seed)
cands = [json.loads(l) for l in open(args.candidates)]
for c in cands:
    c["kind"] = "proposed"
tw = [json.loads(l) for l in open(f"{R}/data/hp_twins.jsonl")]
pool_tw = [t for t in tw if not (t["task"] == 0 or t["split_kind"] == "template")]      # outside the probe set
rng = random.Random(1000 + args.seed)
rng.shuffle(pool_tw)
tracers = []
for t in pool_tw[: args.n_tracer]:
    tracers.append({"id": f"trd:{t['fam']}:{t['task']}", "problem": t["deleted"].strip(), "prompt": t["deleted"].strip() + SUF, "kind": "tracer_deleted", "fam": t["fam"], "task": t["task"]})
    tracers.append({"id": f"tri:{t['fam']}:{t['task']}", "problem": t["intact"].strip(), "prompt": t["intact"].strip() + SUF, "kind": "tracer_intact", "fam": t["fam"], "task": t["task"]})
pool = cands + tracers
rng.shuffle(pool)

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1024 + args.max_new, dtype="bfloat16", enable_prefix_caching=True, tensor_parallel_size=args.tp)
tok = llm.get_tokenizer()
base = "Base" in args.model
STOP = ["\nProblem:", "\nQuestion:", "\n\n\n\n"] if base else None


def wrap(p):
    if args.abstain_prompt:
        p = p + ABSTAIN_SUFFIX
    return p + "\n" if base else (tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) if args.no_think else tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))


def extract(text):
    """Boxed answer if present; otherwise the '####' or last number in the text (instruct models such as
    Llama-3.2-3B box only ~15% of GSM-style answers - amendment A3b). Abstentions never vote."""
    b = last_boxed(text)
    if b is not None:
        if is_abstain(b):
            return "unknown"
        s = _strip_math(b)
        n = norm_num(s)
        return n if (n is not None and re.fullmatch(r"-?[\d.,]+(/\d+)?", s)) else (s or None)
    if re.search(r"Answer:\s*(unknown|unanswerable|insufficient information)", text, re.I):
        return "unknown"
    if "####" in text:
        return norm_num(text.rsplit("####", 1)[1].strip().split("\n")[0])
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return norm_num(nums[-1]) if nums else None


outs = llm.generate([wrap(x["prompt"]) for x in pool], SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, stop=STOP, seed=args.seed))
stats = Counter()
admitted, records = [], []
pm = {}
for x, o in zip(pool, outs):
    ans = [extract(c.text) if c.finish_reason != "length" else None for c in o.outputs]
    cnt = Counter(a for a in ans if a is not None and a != "unknown")
    mode, k = cnt.most_common(1)[0] if cnt else (None, 0)
    pm[x["id"]] = (k / len(ans), mode, sum(1 for a in ans if a == "unknown") / len(ans))
lo, hi = args.lo, args.hi
if args.guard == "abstain" and not args.abstain_prompt:
    raise SystemExit("--guard abstain requires --abstain_prompt 1")
if args.guard == "azr":
    lo, hi = 1e-9, 1 - 1e-9
elif args.guard == "tracer":
    def rate(kind, a, b):
        v = [pm[x["id"]][0] for x in pool if x["kind"] == kind]
        return sum(1 for q in v if a <= q <= b) / max(1, len(v))
    grid = [i / args.G for i in range(args.G + 1)]
    best = None
    for a in grid:
        for b in grid[:-1]:   # hi <= 7/8: unanimous items excluded (A3e)
            if b < a:
                continue
            ri, rd = rate("tracer_intact", a, b), rate("tracer_deleted", a, b)
            if ri < 0.30:
                continue
            key = (ri - rd, ri)
            if best is None or key > best[0]:
                best = (key, a, b, ri, rd)
    if best is None:
        best = ((0, 0), args.lo, args.hi, rate("tracer_intact", args.lo, args.hi), rate("tracer_deleted", args.lo, args.hi))
    lo, hi = best[1], best[2]
    print("TRACER_BAND", lo, hi, "intact", best[3], "deleted", best[4], flush=True)
for x in pool:
    p_mode, mode, abst = pm[x["id"]]
    pa = max(p_mode, abst) if args.count_unknown else p_mode   # A4i: own_uv-consistent admission
    admit = (lo <= pa <= hi) and (args.guard != "abstain" or abst < args.abs_max)   # A4: abstention guard
    records.append({"id": x["id"], "kind": x["kind"], "p_mode": p_mode, "mode": mode, "abstain": abst, "admit": admit})
    stats[f"{x['kind']}_n"] += 1
    stats[f"{x['kind']}_admit"] += int(admit)
    stats[f"{x['kind']}_confident"] += int(p_mode >= 0.5)
    stats[f"{x['kind']}_abstained"] += int(abst >= args.abs_max)
    if admit:
        admitted.append({k_: v for k_, v in x.items() if k_ in ("id", "problem", "prompt", "kind", "fam", "task")})
rng.shuffle(admitted)
with open(f"{args.out_dir}/admitted.jsonl", "w") as fh:
    for a in admitted:
        fh.write(json.dumps(a, ensure_ascii=False) + "\n")
with open(f"{args.out_dir}/guard_records.jsonl", "w") as fh:
    for r_ in records:
        fh.write(json.dumps(r_) + "\n")
summary = {k: v for k, v in stats.items()}
for kind in ("proposed", "tracer_deleted", "tracer_intact"):
    n = stats[f"{kind}_n"]
    summary[f"{kind}_admit_rate"] = stats[f"{kind}_admit"] / n if n else None
    summary[f"{kind}_confident_rate"] = stats[f"{kind}_confident"] / n if n else None
summary["n_admitted"] = len(admitted)
summary["guard"] = args.guard
summary["band_lo"] = lo
summary["band_hi"] = hi
summary["admitted_tracer_deleted"] = sum(1 for a in admitted if a["kind"] == "tracer_deleted")
json.dump(summary, open(f"{args.out_dir}/guard_stats.json", "w"), indent=1)
print("GUARD_DONE", json.dumps(summary), flush=True)
