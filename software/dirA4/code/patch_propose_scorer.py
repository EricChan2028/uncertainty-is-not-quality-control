"""A4b: a3_propose.py gains --no_think / --tp; a3_guard_diag_score.py takes the audit-parts dir as argv[2]. .new + os.replace."""
import os
R = "/home/tione/notebook/rsi_rl"
p = f"{R}/dirA3/code/a3_propose.py"
s = open(p).read()
if "--no_think" not in s:
    old = 'ap.add_argument("--gpu_util", type=float, default=0.85)\n'
    assert old in s
    s = s.replace(old, old + 'ap.add_argument("--no_think", type=int, default=0)\nap.add_argument("--tp", type=int, default=1)\n')
    old2 = 'llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1536, dtype="bfloat16")'
    assert old2 in s
    s = s.replace(old2, 'llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1536, dtype="bfloat16", tensor_parallel_size=args.tp)')
    old3 = 'return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)'
    assert old3 in s
    s = s.replace(old3, 'return tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, **({"enable_thinking": False} if args.no_think else {}))')
    open(p + ".new", "w").write(s); os.replace(p + ".new", p)
q = f"{R}/dirA3/code/a3_guard_diag_score.py"
t = open(q).read()
old = 'glob.glob(f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts/*.json")'
if old in t:
    t = t.replace(old, 'glob.glob((sys.argv[2] if len(sys.argv) > 2 else f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts") + "/*.json")')
    open(q + ".new", "w").write(t); os.replace(q + ".new", q)
print("ok")
