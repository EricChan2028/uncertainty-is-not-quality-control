"""A12 part 2, plain condition (registration section 2.1): runs the registered dirB/code/b_eval.py (sha256 5bf7c94a..., executed
unchanged from its own file) with two registered settings:
  --k 1          b_eval's T = 1 samples are not read in A12; greedy decoding is its own generate call (T = 0, seed 0) before
                 the sampling call, so greedy_acc does not depend on k;
  <|im_end|>     added to the stop tokens of every SamplingParams. The R-Zero solvers were trained in the chat format; in b_eval's
                 Base-format prompt a solver that ends its turn could otherwise continue into an invented next turn whose last
                 box b_eval would score. The stop applies to every model of the condition (the base model included), so the
                 paired comparison treats all models alike; under greedy decoding the kept text is the prefix of the unstopped
                 output, and for a model that never emits <|im_end|> nothing changes.
Before anything runs, b_eval.py and the two project modules it imports (code/loto_common.py, dirB/code/pdc_util.py) are checked
against their registered sha256. The runner writes <tag>.plain_stops.json next to b_eval's outputs: per set, the number of
greedy outputs that ended at <|im_end|> and at the length cap (sets matched to b_eval's greedy calls in b_eval's own order,
with a check that the call carried that set's prompts).
    CUDA_VISIBLE_DEVICES=g /usr/bin/python3 a12_plain_eval.py --model <path> --tag a12_plain_<name> --data <sets> --k 1"""
import hashlib, json, os, runpy, sys

R = "/home/tione/notebook/rsi_rl"
B_EVAL = f"{R}/dirB/code/b_eval.py"
PIN = {B_EVAL: "5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189",
       f"{R}/code/loto_common.py": "901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11",
       f"{R}/dirB/code/pdc_util.py": "f1ad5fe2992dfb59d086a239920c7a3dc4cb71dc9b199d2027f80f92f002d059"}
for p, h in PIN.items():
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == h, f"{p} is not the registered version"
argv = sys.argv[1:]
opt = lambda n: argv[argv.index(n) + 1] if n in argv and argv.index(n) + 1 < len(argv) else None
model, tag, data, k = opt("--model"), opt("--tag"), opt("--data"), opt("--k")
assert model and "Base" in model, "A12 plain: --model must be a Base-format path (b_eval's Base prompt)"
assert tag and tag.startswith("a12_plain_") and data and k == "1", "A12 plain: --tag a12_plain_<m>, --data and --k 1 are required"
assert not any(a.split("=")[0] in ("--chat", "--abstain_prompt", "--redo", "--max_new", "--gpu_util") for a in argv), \
    "A12 plain: b_eval's other options stay at their registered defaults"
OUT = opt("--out") or f"{R}/dirB/results/evals"
SIDE = f"{OUT}/{tag}.plain_stops.json"
psha = lambda ps: hashlib.sha256(json.dumps(ps, ensure_ascii=False).encode("utf-8")).hexdigest()
SETP = {}  # set name -> sha256 of b_eval's Base-format prompts of the set (row prompt + "\n", in file order)
for name in data.split(","):
    with open(f"{R}/dirB/data/{name}.jsonl", encoding="utf-8") as fh:
        SETP[name] = psha([json.loads(l)["prompt"] + "\n" for l in fh])
_sp = f"{OUT}/{tag}.summary.json"; _s0 = json.load(open(_sp)) if os.path.exists(_sp) else {}
TODO = [n for n in data.split(",") if not (n in _s0 and "greedy_acc_lenient" in _s0[n])]  # b_eval's own skip rule: its greedy calls, in order
CALLS = []

import vllm  # noqa: E402

_LLM, _SP = vllm.LLM, vllm.SamplingParams
IM_END = []


class LLM(_LLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        t = self.get_tokenizer(); i = t.convert_tokens_to_ids("<|im_end|>")
        assert isinstance(i, int) and i != t.unk_token_id and t.convert_ids_to_tokens(i) == "<|im_end|>", "no <|im_end|> token"
        IM_END.append(i)

    def generate(self, prompts, sp, *a, **kw):
        outs = super().generate(prompts, sp, *a, **kw)
        if getattr(sp, "temperature", None) == 0.0:
            name = TODO[len(CALLS)] if len(CALLS) < len(TODO) else f"call{len(CALLS)}"; CALLS.append(name)
            side = json.load(open(SIDE)) if os.path.exists(SIDE) else {}
            side[name] = {"n": len(outs), "n_im_end_stop": sum(o.outputs[0].stop_reason == IM_END[0] for o in outs),
                          "n_length": sum(o.outputs[0].finish_reason == "length" for o in outs), "im_end_id": IM_END[0],
                          "prompts_match": psha(list(prompts)) == SETP.get(name)}   # descriptive record only; never stops the run
            json.dump(side, open(SIDE + ".tmp", "w"), indent=1); os.replace(SIDE + ".tmp", SIDE)
        return outs


def SamplingParams(*a, **kw):
    assert IM_END, "SamplingParams built before the model"
    kw["stop_token_ids"] = sorted(set(kw.get("stop_token_ids") or []) | {IM_END[0]})
    return _SP(*a, **kw)


vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
sys.argv = [B_EVAL] + argv
runpy.run_path(B_EVAL, run_name="__main__")
