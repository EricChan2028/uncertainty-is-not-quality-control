"""Direction A, Cycle 0: sample the twin items, simulate the published guards, run the two probes.

Registered in dirA/docs/C0_registration_2026-09-21.md (hashed before this ran). One process per GPU
shard; each shard writes its own JSONL so a preemption costs one shard. No training anywhere.

Per item (deleted and intact forms of the same task):
  main     G=32 samples, T=1.0, max_new 1024 -> answers, truncation flags, p_mode, modal answer, correct
  probe    for each number in the problem text, x2 it and sample G'=8 -> D_j = modal answer changed
  judge    self-judge prompt, 8 samples -> P(model says a needed quantity is missing)
"""
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, "/home/tione/notebook/rsi_rl/code")
from loto_common import last_boxed, norm_num  # noqa: E402  (the programme number normaliser)

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract(text):
    """Last number in the text, the programme's convention for a final answer without an 'Answer:' line."""
    m = _NUM_RE.findall(text)
    return norm_num(m[-1]) if m else None

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--G", type=int, default=32)
ap.add_argument("--Gp", type=int, default=8)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--out", default="/home/tione/notebook/rsi_rl/dirA/results")
ap.add_argument("--limit", type=int, default=0, help="debug: first N items only")
ap.add_argument("--items_file", default="", help="JSONL of {problem,...}; scored as form=proposed, no reference")
ap.add_argument("--all_tasks", type=int, default=0, help="1 = every twin task, not just task 0 per family")
ap.add_argument("--twins_file", default="/home/tione/notebook/rsi_rl/data/hp_twins.jsonl", help="twins JSONL (fam, task, split_kind, intact, deleted, answer, meta)")
ap.add_argument("--abstain_prompt", type=int, default=0, help="1: append the abstention option to every solve prompt and record the abstention rate")
args = ap.parse_args()

R = "/home/tione/notebook/rsi_rl"
if args.items_file:
    rows = []
    for l in open(args.items_file):
        try:
            rows.append(json.loads(l))
        except ValueError:
            pass  # a truncated last line must not kill every shard
    for i, r in enumerate(rows):
        r.setdefault("fam", i); r.setdefault("task", 0); r.setdefault("split", "proposed"); r.setdefault("split_kind", "proposed")
        r.setdefault("answer", None); r["proposed"] = r["problem"]
    items = rows
    FORMS = ("proposed",)
else:
    rows = [json.loads(l) for l in open(args.twins_file)]
    # probe set: one task per family (task 0) + every template-split task; --all_tasks 1 takes everything
    items = rows if args.all_tasks else [r for r in rows if r["task"] == 0 or r["split_kind"] == "template"]
    FORMS = ("deleted", "intact")
items = items[args.shard::args.nshards]
if args.limit:
    items = items[:args.limit]
os.makedirs(args.out, exist_ok=True)
outp = f"{args.out}/{args.tag}_shard{args.shard}of{args.nshards}.jsonl"
done = set()
if os.path.exists(outp):
    for l in open(outp):
        try:
            d = json.loads(l)
            done.add((d["fam"], d["task"], d["form"]))
        except Exception:
            pass

from vllm import LLM, SamplingParams  # noqa: E402
sys.path.insert(0, '/home/tione/notebook/rsi_rl/dirB/code')
from pdc_util import ABSTAIN_SUFFIX, is_abstain  # noqa: E402

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048 + args.max_new,
          dtype="bfloat16", enable_prefix_caching=True)
tok = llm.get_tokenizer()
is_chat = tok.chat_template is not None and "Base" not in args.model


def wrap(problem, instruction):
    if args.abstain_prompt:
        problem = problem + ABSTAIN_SUFFIX
    text = f"{problem}\n\n{instruction}"
    if is_chat:
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return f"Question: {text}\nAnswer: Let's think step by step."


SOLVE = "Solve the problem. Show your reasoning briefly, then give the final numeric answer on the last line as 'Answer: <number>'."
SOLVE_ABS = "Solve the problem step by step and output the final answer within \\boxed{}."
# under the abstention prompt the solve instruction is the boxed one the abstention-aware arms were trained with
if args.abstain_prompt:
    SOLVE = SOLVE_ABS
SOLVE_P2 = ("Solve the problem. If any quantity you need is not stated in the text, say so explicitly on a "
            "line beginning 'ASSUMED:' before using a value for it. Then give the final numeric answer on the "
            "last line as 'Answer: <number>'.")
JUDGE = ("Do NOT solve this problem. Instead, decide whether every quantity needed to compute the answer is "
         "given in the text. Reply with exactly one word on the last line: 'COMPLETE' if it is solvable as "
         "stated, or 'MISSING' if a needed quantity is not given.")

NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?![\w.]|\.\d)")


def norm_ans(a):
    if a is None:
        return None
    try:
        v = float(str(a).replace(",", ""))
        return int(v) if abs(v - round(v)) < 1e-9 else round(v, 4)
    except Exception:
        return None


def answers_of(texts, finish):
    out = []
    for t, f in zip(texts, finish):
        if f == "length":
            out.append(None)          # truncated before an answer is a NON-answer, never dropped
            continue
        if args.abstain_prompt:
            b = last_boxed(t)
            if b is not None and is_abstain(b):
                out.append(None)            # an abstention is a non-answer: it never votes
                continue
            if b is not None:
                out.append(norm_ans(b))
                continue
            if re.search(r"Answer:\s*(unknown|unanswerable|insufficient information)", t, re.I):
                out.append(None)
                continue
        m = re.findall(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", t)
        a = m[-1] if m else _extract(t)
        out.append(norm_ans(a))
    return out


def summarize(ans):
    c = Counter(a for a in ans)
    mode, n = c.most_common(1)[0]
    p_mode = n / len(ans)
    probs = [v / len(ans) for v in c.values()]
    H = -sum(p * math.log(p) for p in probs)
    return mode, p_mode, H, sum(1 for a in ans if a is None) / len(ans)


def perturb(problem):
    """One variant per distinct number in the text: that number doubled, others untouched."""
    nums = []
    for m in NUM.finditer(problem):
        s = m.group(1)
        if s not in [x[0] for x in nums]:
            nums.append((s, m.start(), m.end()))
    variants = []
    for s, a, b in nums:
        v = int(s.replace(",", "")) * 2
        variants.append((s, problem[:a] + str(v) + problem[b:]))
    return variants


sp_main = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, seed=17)
sp_probe = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.Gp, seed=19)
sp_judge = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=256, n=8, seed=23)

BATCH = 64
with open(outp, "a") as fh:
    for bi in range(0, len(items), BATCH):
        batch = []
        for r in items[bi:bi + BATCH]:
            for form in FORMS:
                if (r["fam"], r["task"], form) in done:
                    continue
                batch.append((r, form, r[form]))
        if not batch:
            continue
        # 1. main samples
        outs = llm.generate([wrap(p, SOLVE) for _, _, p in batch], sp_main)
        recs = []
        for (r, form, prob), o in zip(batch, outs):
            texts = [c.text for c in o.outputs]
            fin = [c.finish_reason for c in o.outputs]
            ans = answers_of(texts, fin)
            mode, p_mode, H, non = summarize(ans)
            ref = norm_ans(r["answer"]) if r.get("answer") is not None else None
            recs.append({"fam": r["fam"], "task": r["task"], "form": form, "split": r["split"],
                         "split_kind": r["split_kind"], "depth": (r.get("meta") or {}).get("depth"),
                         "template": (r.get("meta") or {}).get("gsm_template"),
                         "ref": ref, "mode": mode, "p_mode": p_mode, "H_ans": H, "nonanswer": non,
                         "correct": int(mode is not None and mode == ref),
                         "trunc_rate": sum(1 for f in fin if f == "length") / len(fin),
                         "answers": ans, "problem": prob,
                         "abstain": sum(1 for t in texts if is_abstain(last_boxed(t) or "") or re.search(r"Answer:\s*(unknown|unanswerable|insufficient information)", t, re.I)) / len(texts)})
        # 2. probe: doubled numbers
        pp, pidx = [], []
        for k, (r, form, prob) in enumerate(batch):
            for s, var in perturb(prob):
                pp.append(wrap(var, SOLVE))
                pidx.append((k, s))
        if pp:
            pout = llm.generate(pp, sp_probe)
            per = {}
            for (k, s), o in zip(pidx, pout):
                texts = [c.text for c in o.outputs]
                fin = [c.finish_reason for c in o.outputs]
                m2, _, _, _ = summarize(answers_of(texts, fin))
                per.setdefault(k, []).append({"num": s, "mode": m2, "changed": int(m2 != recs[k]["mode"])})
            for k, rec in enumerate(recs):
                d = per.get(k, [])
                rec["probe"] = d
                rec["s_probe"] = (sum(x["changed"] for x in d) / len(d)) if d else None
        # 2b. probe P2 (amendment C0a): assumption declaration inside the solve
        p2out = llm.generate([wrap(p, SOLVE_P2) for _, _, p in batch], sp_probe)
        for rec, o in zip(recs, p2out):
            rec["p2_assumed"] = sum(1 for c in o.outputs if "ASSUMED:" in c.text.upper()) / len(o.outputs)
        # 3. self-judge
        jout = llm.generate([wrap(p, JUDGE) for _, _, p in batch], sp_judge)
        for rec, o in zip(recs, jout):
            votes = [("MISSING" in c.text.upper().split("\n")[-1]) or (c.text.upper().rstrip().endswith("MISSING")) for c in o.outputs]
            rec["judge_missing"] = sum(votes) / len(votes)
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[{args.tag} shard {args.shard}] {bi + len(batch)}/{2 * len(items)} items", flush=True)
print("SHARD_DONE", outp, flush=True)
