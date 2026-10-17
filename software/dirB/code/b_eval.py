"""Direction B evaluator (vLLM). One checkpoint, one or more datasets: greedy accuracy (T=0) plus maj@k /
mean p_mode / pass@k at T=1.0. Identical for every checkpoint of every arm; writes per-item JSONL and a
summary JSON. Resumable at the dataset level (a finished dataset summary is not recomputed).

  python3 b_eval.py --model <path> --tag <run>_step<N> --data math500_train,math_heldout1000,gsm8k_test500
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

R = "/home/tione/notebook/rsi_rl"
sys.path.insert(0, f"{R}/code")
from loto_common import _strip_math, last_boxed, math_reward, norm_num  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="math500_train,math_heldout1000,gsm8k_test500")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max_new", type=int, default=2048)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--chat", type=int, default=-1, help="-1: chat template unless the model path contains 'Base'")
ap.add_argument("--out", default=f"{R}/dirB/results/evals")
ap.add_argument("--abstain_prompt", type=int, default=0, help="1: offer \\boxed{unknown} in every prompt; abstentions count as wrong and are reported")
ap.add_argument("--redo", type=int, default=0, help="1: recompute datasets already in the summary (amendment A3d re-evaluation)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
sys.path.insert(0, f"{R}/dirB/code")
from pdc_util import ABSTAIN_SUFFIX, is_abstain  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1024 + args.max_new, dtype="bfloat16",
          enable_prefix_caching=True)
tok = llm.get_tokenizer()
base = ("Base" in args.model) if args.chat < 0 else (args.chat == 0)
STOP = ["\nProblem:", "\nQuestion:", "\n\n\n\n"] if base else None


def wrap(prompt):
    if args.abstain_prompt:
        prompt = prompt + ABSTAIN_SUFFIX
    if base:
        return prompt + "\n"
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def extract(text, kind):
    """Final answer string for voting (normalised); None if absent; 'unknown' for an abstention (never correct here)."""
    b = last_boxed(text)
    if b is not None and is_abstain(b):
        return "unknown"
    if kind == "gsm":
        if b is None and "####" in text:
            b = text.rsplit("####", 1)[1].strip().split("\n")[0]
        if b is None:
            nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
            b = nums[-1] if nums else None
        return norm_num(b) if b is not None else None
    if b is None:
        return None
    s = _strip_math(b)
    n = norm_num(s)
    return n if (n is not None and re.fullmatch(r"-?[\d.,]+(/\d+)?", s)) else (s or None)


def correct(text, gt, kind):
    if kind == "gsm":
        p = extract(text, "gsm")
        return int(p is not None and p == norm_num(gt))
    return int(math_reward(text, gt) > 0)


LENIENT_PATS = [r"####\s*([^\n]+)", r"answer is[:\s]*\$?([^\n$]+?)\$?\s*\.?\s*(?:\n|$)", r"[Aa]nswer\s*[:=]\s*\$?([^\n$]+?)\$?\s*\.?\s*(?:\n|$)"]


def extract_lenient(text, kind):
    """Amendment A3d: boxed if present; else '####' / 'answer is' / 'Answer:' patterns; else the last number. None if nothing."""
    if last_boxed(text) is not None:
        return extract(text, kind)
    cand = None
    for pat in LENIENT_PATS:
        m = re.findall(pat, text)
        if m:
            cand = m[-1].strip().strip("$").strip()
            break
    if cand is None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    if kind == "gsm":
        n = norm_num(cand)
        if n is None:
            nums = re.findall(r"-?\d+(?:\.\d+)?", cand)
            n = norm_num(nums[-1]) if nums else None
        return n
    return cand


def correct_lenient(text, gt, kind):
    if extract(text, kind) == "unknown":
        return 0
    if kind == "gsm":
        p = extract_lenient(text, "gsm")
        return int(p is not None and p == norm_num(gt))
    if math_reward(text, gt) > 0:
        return 1
    c = extract_lenient(text, "math")
    return int(c is not None and math_reward("\\boxed{" + c + "}", gt) > 0)


summary_path = f"{args.out}/{args.tag}.summary.json"
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
for name in args.data.split(","):
    if name in summary and not args.redo and "greedy_acc_lenient" in summary[name]:
        continue
    kind = "gsm" if name.startswith("gsm") else "math"
    rows = [json.loads(l) for l in open(f"{R}/dirB/data/{name}.jsonl")]
    prompts = [wrap(r["prompt"]) for r in rows]
    g = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_new, n=1, stop=STOP, seed=0))
    s = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.k, stop=STOP, seed=1))
    recs, acc_g, acc_maj, acc_pass, pm, trunc, lens = [], 0, 0, 0, [], 0, []
    acc_gl, boxed_g = 0, 0
    abs_g = abs_s = 0.0
    with open(f"{args.out}/{args.tag}.{name}.jsonl", "w") as fh:
        for r, og, os_ in zip(rows, g, s):
            gt = r["answer"]
            gtext = og.outputs[0].text
            cg = correct(gtext, gt, kind) if extract(gtext, kind) != "unknown" else 0
            texts = [o.text for o in os_.outputs]
            fins = [o.finish_reason for o in os_.outputs]
            ans = [extract(t, kind) if f != "length" else None for t, f in zip(texts, fins)]
            cs = [correct(t, gt, kind) if (f != "length" and a != "unknown") else 0 for t, f, a in zip(texts, fins, ans)]
            abs_g += float(extract(gtext, kind) == "unknown")
            abs_s += sum(1 for a in ans if a == "unknown") / len(ans)
            cnt = Counter(a for a in ans if a is not None and a != "unknown")
            mode, nmode = (cnt.most_common(1)[0] if cnt else (None, 0))
            p_mode = nmode / len(ans)
            maj_ok = int(mode is not None and any(c and a == mode for a, c in zip(ans, cs)))
            acc_g += cg; acc_maj += maj_ok; acc_pass += int(any(cs)); pm.append(p_mode)
            cgl = correct_lenient(gtext, gt, kind); acc_gl += cgl; boxed_g += int(last_boxed(gtext) is not None)
            trunc += sum(1 for f in fins if f == "length") / len(fins)
            lens.append(sum(len(o.token_ids) for o in os_.outputs) / len(os_.outputs))
            fh.write(json.dumps({"id": r["id"], "gt": gt, "greedy_ok": cg, "greedy_ans": extract(gtext, kind),
                                 "greedy_ok_lenient": cgl, "greedy_ans_lenient": extract_lenient(gtext, kind), "greedy_boxed": int(last_boxed(gtext) is not None), "greedy_text": gtext,
                                 "greedy_len": len(og.outputs[0].token_ids), "greedy_trunc": int(og.outputs[0].finish_reason == "length"),
                                 "mode": mode, "p_mode": p_mode, "maj_ok": maj_ok, "pass_k": int(any(cs)),
                                 "sample_acc": sum(cs) / len(cs), "answers": ans}, ensure_ascii=False) + "\n")
    n = len(rows)
    summary[name] = {"n": n, "greedy_acc": acc_g / n, "maj_acc": acc_maj / n, "pass_k": acc_pass / n, "k": args.k,
                     "mean_p_mode": sum(pm) / n, "sample_trunc": trunc / n, "mean_len": sum(lens) / n, "model": args.model,
                     "abstain_prompt": args.abstain_prompt, "abstain_greedy": abs_g / n, "abstain_samples": abs_s / n,
                     "greedy_acc_lenient": acc_gl / n, "greedy_boxed_rate": boxed_g / n}
    json.dump(summary, open(summary_path + ".tmp", "w"), indent=1)
    os.replace(summary_path + ".tmp", summary_path)
    print(f"[{args.tag}] {name}: greedy {acc_g / n:.3f} lenient {acc_gl / n:.3f} boxed {boxed_g / n:.2f} maj@{args.k} {acc_maj / n:.3f} pass@{args.k} {acc_pass / n:.3f} "
          f"p_mode {sum(pm) / n:.3f} trunc {trunc / n:.3f}", flush=True)
print("EVAL_DONE", args.tag, flush=True)
