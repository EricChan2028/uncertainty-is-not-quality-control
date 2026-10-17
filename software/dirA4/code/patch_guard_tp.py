"""A4a.4: a3_guard.py gains --tp and --no_think (defaults unchanged). .new + os.replace (safe for running loops)."""
import os
p = "/home/tione/notebook/rsi_rl/dirA3/code/a3_guard.py"
s = open(p).read()
if "--no_think" not in s:
    old = 'ap.add_argument("--gpu_util", type=float, default=0.85)\n'
    assert old in s
    s = s.replace(old, old + 'ap.add_argument("--tp", type=int, default=1, help="A4a: tensor parallel size")\n'
                            'ap.add_argument("--no_think", type=int, default=0, help="A4a: chat template with enable_thinking=False (Qwen3 post-trained)")\n')
    old2 = 'llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1024 + args.max_new, dtype="bfloat16", enable_prefix_caching=True)'
    assert old2 in s
    s = s.replace(old2, 'llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=1024 + args.max_new, dtype="bfloat16", enable_prefix_caching=True, tensor_parallel_size=args.tp)')
    old3 = 'tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)'
    assert old3 in s
    s = s.replace(old3, '(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) if args.no_think else ' + old3 + ')')
    open(p + ".new", "w").write(s)
    os.replace(p + ".new", p)
print("ok")
