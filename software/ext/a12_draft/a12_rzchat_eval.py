"""A12 part 2 (registration section 2.2): greedy accuracy with R-Zero's own solver prompt, as in R-Zero's evaluation/generate.py
(official code 5699329): chat template, system message "Please reason step by step, and put your final answer within
\\boxed{}.", the bare question as the user message, temperature 0, max 4,096 new tokens. Deviation (registered): generation
also stops at <|im_end|> (the end of the assistant turn), so the scored text is the assistant's answer; under greedy decoding
the kept text is the prefix of what R-Zero's call (stop at <|endoftext|> only) produces.
Data: the three b_eval sets (dirB/data/math500_train = MATH-500, math_heldout1000, gsm8k_test500); the question is the row's
"problem" (GSM8K: the b_eval "####" instruction removed; it must end the problem exactly). Scoring:
  primary   - b_eval.py's own extract() / correct() (their source is taken from the registered b_eval.py at run time, so the
              scorer is identical; the sha256 of b_eval.py and of the two project modules they call, code/loto_common.py and
              dirB/code/pdc_util.py, are checked);
  secondary - R-Zero's scorer without its GPT-4o re-check: first "\\boxed{...}" match of (?i)\\boxed\\s*{([^\\n]+)} and
              math_verify verify(parse(gold), parse(answer)) (evaluation/datasets_loader.py).
Outputs dirB/results/evals/<tag>.<set>.jsonl and <tag>.summary.json, resumable per set (a finished set is not recomputed).
    CUDA_VISIBLE_DEVICES=g /usr/bin/python3 a12_rzchat_eval.py --model <path> --tag a12rz_<name>"""
import argparse, ast, hashlib, json, os, re, sys

R = "/home/tione/notebook/rsi_rl"
B_EVAL = f"{R}/dirB/code/b_eval.py"; B_EVAL_SHA = "5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189"
SYS = "Please reason step by step, and put your final answer within \\boxed{}."
GSM_SUFFIX = " Let's think step by step and output the final answer after \"####\"."
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="math500_train,math_heldout1000,gsm8k_test500")
ap.add_argument("--out", default=f"{R}/dirB/results/evals")
args = ap.parse_args()
assert args.tag.startswith("a12rz_"), "A12: tags of this wrapper start with a12rz_"
src = open(B_EVAL, "rb").read()
assert hashlib.sha256(src).hexdigest() == B_EVAL_SHA, "dirB/code/b_eval.py is not the registered version"
DEP_SHA = {f"{R}/code/loto_common.py": "901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11",
           f"{R}/dirB/code/pdc_util.py": "f1ad5fe2992dfb59d086a239920c7a3dc4cb71dc9b199d2027f80f92f002d059"}
for _p, _h in DEP_SHA.items():
    assert hashlib.sha256(open(_p, "rb").read()).hexdigest() == _h, f"{_p} is not the registered version"
sys.path.insert(0, f"{R}/code"); sys.path.insert(0, f"{R}/dirB/code")
from loto_common import _strip_math, last_boxed, math_reward, norm_num  # noqa: E402,F401
from pdc_util import is_abstain  # noqa: E402,F401
tree = ast.parse(src.decode("utf-8"))
ns = {"re": re, "last_boxed": last_boxed, "is_abstain": is_abstain, "_strip_math": _strip_math, "norm_num": norm_num,
      "math_reward": math_reward}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("extract", "correct"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), B_EVAL, "exec"), ns)
extract, correct = ns["extract"], ns["correct"]
try:
    from math_verify import parse as mv_parse, verify as mv_verify
except Exception:  # secondary score unavailable
    mv_parse = mv_verify = None
RZ_BOXED = re.compile(r"(?i)\\boxed\s*{([^\n]+)}")


def rz_score(response, gold):
    if mv_parse is None:
        return None
    m = RZ_BOXED.search(response)
    if not m:
        return 0
    try:
        return int(bool(mv_verify(mv_parse(str(gold)), mv_parse(str(m.group(1))))))
    except Exception:
        return 0


from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=args.model, gpu_memory_utilization=0.85, max_model_len=8192, dtype="bfloat16", seed=0)
tok = llm.get_tokenizer()
assert tok.chat_template, "the model has no chat template"
IM_END = tok.convert_tokens_to_ids("<|im_end|>")
stops = sorted({tok.eos_token_id, IM_END})


def question(r, name):
    if name.startswith("gsm"):
        assert r["problem"].endswith(GSM_SUFFIX), r["id"]
        return r["problem"][:-len(GSM_SUFFIX)]
    return r["problem"]


def render(q):
    return tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True)


os.makedirs(args.out, exist_ok=True)
sp = f"{args.out}/{args.tag}.summary.json"
summary = json.load(open(sp)) if os.path.exists(sp) else {}
for name in args.data.split(","):
    if name in summary and summary[name].get("complete"):
        continue
    kind = "gsm" if name.startswith("gsm") else "math"
    rows = [json.loads(l) for l in open(f"{R}/dirB/data/{name}.jsonl", encoding="utf-8")]
    prompts = [render(question(r, name)) for r in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4096, stop_token_ids=stops, seed=0))
    n_ok = n_rz = n_trunc = n_im = n_rz_na = 0; lens = []
    with open(f"{args.out}/{args.tag}.{name}.jsonl.tmp", "w", encoding="utf-8") as fh:
        for r, o in zip(rows, outs):
            c = o.outputs[0]; ids = list(c.token_ids)
            if IM_END in ids:
                ids = ids[:ids.index(IM_END)]
            t = tok.decode(ids, skip_special_tokens=True)
            ok = correct(t, r["answer"], kind) if extract(t, kind) != "unknown" else 0
            rz = rz_score(t, r["answer"])
            n_ok += ok; n_rz += rz or 0; n_rz_na += rz is None; n_trunc += c.finish_reason == "length"
            n_im += c.stop_reason == IM_END or IM_END in c.token_ids; lens.append(len(c.token_ids))
            fh.write(json.dumps({"id": r["id"], "gt": r["answer"], "greedy_ok": ok, "greedy_ans": extract(t, kind),
                                 "ok_rzero_style": rz, "finish": c.finish_reason, "stop_token": c.stop_reason,
                                 "n_tokens": len(c.token_ids), "text": t}, ensure_ascii=False) + "\n")
    os.replace(f"{args.out}/{args.tag}.{name}.jsonl.tmp", f"{args.out}/{args.tag}.{name}.jsonl")
    n = len(rows)
    summary[name] = {"n": n, "greedy_acc": n_ok / n, "greedy_acc_rzero_style": (n_rz / n) if not n_rz_na else None,
                     "trunc_rate": n_trunc / n, "im_end_rate": n_im / n, "mean_len": sum(lens) / n, "model": args.model,
                     "prompt_format": "rzero_chat", "system": SYS, "max_tokens": 4096, "stop_token_ids": stops,
                     "rendered_prompt_0_sha256": hashlib.sha256(prompts[0].encode("utf-8")).hexdigest(),
                     "b_eval_sha256": B_EVAL_SHA, "complete": True}
    json.dump(summary, open(sp + ".tmp", "w"), indent=1); os.replace(sp + ".tmp", sp)
    print(f"[{args.tag}] {name}: greedy {n_ok / n:.3f} rzero-style {summary[name]['greedy_acc_rzero_style']} trunc {n_trunc / n:.3f}",
          flush=True)
print("EVAL_DONE", args.tag, flush=True)
