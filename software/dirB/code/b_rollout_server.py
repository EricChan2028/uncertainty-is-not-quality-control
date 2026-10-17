"""vLLM rollout server for one trainer (paired GPU). File handshake on a RAM-disk directory:

  <dir>/live/                bf16 weights the trainer publishes (atomic dir swap) + <dir>/version
  <dir>/req.json             request written by the trainer: {"id", "version", "prompts", "n", "max_tokens",
                             "temperature", "stop", "seed"}   (written as req.json.tmp then renamed)
  <dir>/resp_<id>.json       response: per prompt, n x {"token_ids", "text", "finish_reason"}

Weights are reloaded in place (vllm worker.reload_weights) whenever the request's version differs from
the loaded one. Exits when <dir>/stop exists. Preemption-safe: killing it only stalls the trainer's
generate(), which times out, checkpoints and exits 0.
"""
import argparse
import json
import os
import time

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--max_model_len", type=int, default=2560)
ap.add_argument("--gpu_util", type=float, default=0.85)
args = ap.parse_args()
D = args.dir
LIVE = os.path.join(D, "live")
while not os.path.exists(os.path.join(LIVE, "config.json")):
    time.sleep(2)

from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model=LIVE, gpu_memory_utilization=args.gpu_util, max_model_len=args.max_model_len, dtype="bfloat16",
          enable_prefix_caching=True)
loaded = open(os.path.join(D, "version")).read().strip() if os.path.exists(os.path.join(D, "version")) else "0"
print("ROLLOUT_SERVER_READY", D, "version", loaded, flush=True)
open(os.path.join(D, "ready"), "w").write(loaded)
req_path = os.path.join(D, "req.json")
while not os.path.exists(os.path.join(D, "stop")):
    if not os.path.exists(req_path):
        time.sleep(0.5)
        continue
    try:
        req = json.load(open(req_path))
    except Exception:
        time.sleep(0.5)
        continue
    os.remove(req_path)
    if str(req.get("version", loaded)) != loaded:
        t0 = time.time()
        llm.collective_rpc("reload_weights")
        llm.reset_prefix_cache()
        loaded = str(req["version"])
        print(f"reloaded weights v{loaded} in {time.time() - t0:.1f}s", flush=True)
    sp = SamplingParams(temperature=req.get("temperature", 1.0), top_p=1.0, max_tokens=req["max_tokens"], n=req.get("n", 1),
                        stop=req.get("stop"), seed=req.get("seed"))
    t0 = time.time()
    outs = llm.generate(req["prompts"], sp, use_tqdm=False)
    resp = [[{"token_ids": list(o.token_ids), "text": o.text, "finish_reason": o.finish_reason} for o in out.outputs] for out in outs]
    tmp = os.path.join(D, f"resp_{req['id']}.json.tmp")
    json.dump(resp, open(tmp, "w"))
    os.replace(tmp, os.path.join(D, f"resp_{req['id']}.json"))
    print(f"req {req['id']}: {len(req['prompts'])}x{req.get('n', 1)} in {time.time() - t0:.1f}s", flush=True)
print("ROLLOUT_SERVER_EXIT", flush=True)
