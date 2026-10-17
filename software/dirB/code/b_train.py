"""Direction B trainer: label-free GRPO-style RL on MATH-500 questions with one of four reward arms
computed on the SAME rollouts (registration dirB/docs/B0_registration_2026-09-21.md):

  own   TTRL: r_i = 1 if rollout i's answer equals the group's majority-vote answer
  perm  same number of rewarded rollouts per group as `own`, chosen uniformly at random (keeps the
        per-group reward histogram, destroys which rollout agrees with the majority)
  fmt   r_i = 1 if the rollout parses to a boxed answer
  gt    r_i = 1 if the answer matches the ground truth (oracle ceiling)

Full-parameter bf16 weights + fp32 master, AdamW-8bit, group-mean baseline, sequence-mean token
normalisation, no KL. Preemptible: SIGTERM/SIGUSR1/SIGINT -> resume checkpoint at the next safe point,
exit 0; a restart with the same --out resumes (weights, master, optimizer, RNG, data cursor).
Every step logs the statistics of ALL four rewards on the batch, whichever arm is trained.
"""
import argparse
import json
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter

import torch                                       # (expandable_segments is incompatible with vLLM's sleep-mode memory pool)

R = "/home/tione/notebook/rsi_rl"
sys.path.insert(0, f"{R}/code")
from loto_common import _strip_math, last_boxed, math_reward, norm_num  # noqa: E402
sys.path.insert(0, f"{R}/dirB/code")
from pdc_util import ABSTAIN_SUFFIX, deleted_variant, is_abstain  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--uv_weight", type=float, default=3.0, help="A4l own_uvw: weight of an abstention vote")
ap.add_argument("--arm", required=True, choices=["own", "perm", "fmt", "gt", "own_abs", "pdc", "own_uv", "own_uvw"],
                help="own_abs: TTRL with the abstention option in the prompt; pdc: own_abs + label-free premise-deletion contrast")
ap.add_argument("--n_del", type=int, default=16, help="pdc: deleted variants per step (drawn from the step's own questions)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True, help="run dir on the shared volume (logs, markers)")
ap.add_argument("--ckpt_dir", default="/dev/shm/ckpt_b", help="weights: resume/ and final/ (RAM disk)")
ap.add_argument("--data", default=f"{R}/dirB/data/math500_train.jsonl")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--qps", type=int, default=32, help="questions per step")
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--gen_bs", type=int, default=64)
ap.add_argument("--micro_bs", type=int, default=4)
ap.add_argument("--ckpt_every", type=int, default=10)
ap.add_argument("--save_at", default="20,40", help="weights-only snapshots at these steps")
ap.add_argument("--optimizer", default="adamw8bit", choices=["adamw", "adamw8bit"])
ap.add_argument("--grad_in_param", type=int, default=0, help="A4d: accumulate gradients in bf16 p.grad (no separate fp32 buffer); 14B with --optimizer_device cpu")
ap.add_argument("--optimizer_device", default="cuda", choices=["cuda", "cpu"], help="cpu: fp32 master + AdamW on host RAM (8B on a 72 GB card); fp32 grad buffer stays on the GPU")
ap.add_argument("--attn", default="flash_attention_2")
ap.add_argument("--chat", type=int, default=-1, help="-1: chat template unless the model path contains 'Base'")
ap.add_argument("--rollout_dir", default="", help="if set, rollouts come from b_rollout_server.py (paired GPU) via this RAM-disk dir")
ap.add_argument("--rollout_timeout", type=int, default=1500, help="seconds to wait for the rollout server before checkpointing and exiting")
ap.add_argument("--answer_fallback", type=int, default=0, help="1: an unboxed rollout's answer is its '####' or last number (A3 loops; matches a3_guard.py). 0 = boxed only (B/A2 protocol)")
ap.add_argument("--vllm_inproc", type=int, default=0, help="1: rollouts from an in-process vLLM engine on the SAME GPU (sleep mode during the update; weights reloaded in place each step) - one GPU per run")
ap.add_argument("--vllm_util", type=float, default=0.25, help="gpu_memory_utilization of the in-process engine (fraction of the card)")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
RUN = os.path.basename(os.path.normpath(args.out))
CK = os.path.join(args.ckpt_dir, RUN)
RESUME_DIR = os.path.join(CK, "resume")
os.makedirs(CK, exist_ok=True)
with open(os.path.join(args.out, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=1)

# ------------------------------------------------------------ preemption ---
STOP = {"flag": False, "sig": None}


class Preempted(Exception):
    pass


def _on_signal(signum, frame):
    STOP["flag"] = True
    STOP["sig"] = int(signum)
    print(f"[preempt] signal {signum}; checkpointing at the next safe point", flush=True)


for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
    signal.signal(s, _on_signal)

LOGF = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")


def log(d):
    d["time"] = time.time()
    LOGF.write(json.dumps(d, ensure_ascii=False) + "\n")
    LOGF.flush()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "time"}), flush=True)


# ------------------------------------------------------------------ model ---
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList  # noqa: E402


class _StopOnSignal(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return bool(STOP["flag"])


RESUME_STATE = None
if os.path.exists(os.path.join(RESUME_DIR, "state.pt")) and os.path.exists(os.path.join(RESUME_DIR, "config.json")):
    RESUME_STATE = torch.load(os.path.join(RESUME_DIR, "state.pt"), map_location="cpu", weights_only=False)
    print(f"[resume] checkpoint at step {RESUME_STATE['step']} in {RESUME_DIR}", flush=True)

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
LOAD_FROM = RESUME_DIR if RESUME_STATE is not None else args.model
try:
    model = AutoModelForCausalLM.from_pretrained(LOAD_FROM, dtype=torch.bfloat16, attn_implementation=args.attn).cuda()
except Exception as e:  # pragma: no cover
    print("attn fallback to sdpa:", e, flush=True)
    model = AutoModelForCausalLM.from_pretrained(LOAD_FROM, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable()
model.config.use_cache = True
EOS_IDS = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])
EOS_IDS.add(tok.eos_token_id)
PAD_ID = tok.pad_token_id
BASE = ("Base" in args.model) if args.chat < 0 else (args.chat == 0)
STOP_STRINGS = ["\nProblem:", "\nQuestion:", "\n\n\n\n"] if BASE else None

params = [p for p in model.parameters() if p.requires_grad]
master, gbuf, opt = [], [], None
OPT_RESTORED = False


def setup_optimizer():
    """fp32 master + grad buffer + optimizer (+ resume of their state). Called after the in-process vLLM
    engine is up when --vllm_inproc 1, so the engine profiles the card before the optimizer takes 40 GB."""
    global master, gbuf, opt, OPT_RESTORED
    master = [p.detach().to(device=args.optimizer_device, dtype=torch.float32).clone() for p in params]
    for m in master:
        m.requires_grad_(True)
    gbuf = [] if args.grad_in_param else [torch.zeros_like(p, dtype=torch.float32, device="cuda") for p in params]     # fp32 accumulation on the GPU (A4d: none with --grad_in_param)
    if args.optimizer == "adamw8bit" and args.optimizer_device == "cuda":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(master, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    else:
        if args.optimizer == "adamw8bit":
            print("[optimizer] adamw8bit needs CUDA; using torch AdamW on the CPU master", flush=True)
            args.optimizer = "adamw"
        opt = torch.optim.AdamW(master, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8, foreach=True)
    if RESUME_STATE is not None and os.path.exists(os.path.join(RESUME_DIR, "optim.pt")):
        _o = torch.load(os.path.join(RESUME_DIR, "optim.pt"), map_location="cpu", weights_only=False)
        if _o.get("optimizer") == args.optimizer and len(_o["master"]) == len(master):
            for m, c in zip(master, _o["master"]):
                m.data.copy_(c.to(device=m.device, dtype=torch.float32))
            _sd = _o["opt"]
            for k, st in _sd["state"].items():
                for kk, v in list(st.items()):
                    if torch.is_tensor(v):
                        st[kk] = v.to(device=master[0].device)
            opt.load_state_dict(_sd)
            for mp, p in zip(master, params):
                p.data.copy_(mp.data.to(dtype=torch.bfloat16))
            OPT_RESTORED = True
            print("[resume] fp32 master and optimizer state restored", flush=True)
        del _o


if not args.vllm_inproc:
    setup_optimizer()


ABSTAIN = args.arm in ("own_abs", "pdc", "own_uv", "own_uvw")   # A4g own_uv: abstentions vote


def wrap(prompt):
    if ABSTAIN:
        prompt = prompt + ABSTAIN_SUFFIX
    if BASE:
        return prompt + "\n"
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


WEIGHTS_VERSION = "0"


def publish_weights():
    """Atomically replace <rollout_dir>/live with the current bf16 weights and bump the version the
    server compares against (a fresh timestamp, so a restarted trainer is never mistaken for the loaded one)."""
    global WEIGHTS_VERSION
    import shutil
    D = args.rollout_dir
    live, tmp, old = os.path.join(D, "live"), os.path.join(D, "live.tmp"), os.path.join(D, "live.old")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(live):
        os.rename(live, old)
    os.rename(tmp, live)
    shutil.rmtree(old, ignore_errors=True)
    WEIGHTS_VERSION = str(int(time.time() * 1000))
    open(os.path.join(D, "version"), "w").write(WEIGHTS_VERSION)


def generate_remote(prompts):
    """Same contract as generate(), served by b_rollout_server.py. The trainer repeats each question G
    times consecutively, so one prompt with n=G is sent per question (prefix caching on the server)."""
    D = args.rollout_dir
    uniq = prompts[:: args.G]
    assert uniq * 1 and all(prompts[i] == uniq[i // args.G] for i in range(len(prompts)))
    rid = f"{PHASE['step']}_{int(time.time())}"
    req = {"id": rid, "version": WEIGHTS_VERSION, "prompts": [wrap(p) for p in uniq], "n": args.G, "max_tokens": args.max_new,
           "temperature": args.temp, "stop": STOP_STRINGS, "seed": args.seed * 100000 + PHASE["step"]}
    tmp = os.path.join(D, "req.json.tmp")
    json.dump(req, open(tmp, "w"))
    os.replace(tmp, os.path.join(D, "req.json"))
    rp = os.path.join(D, f"resp_{rid}.json")
    t0 = time.time()
    while not os.path.exists(rp):
        if STOP["flag"]:
            raise Preempted()
        if time.time() - t0 > args.rollout_timeout:
            print("[rollout] server timeout; checkpointing and exiting", flush=True)
            STOP["flag"], STOP["sig"] = True, -1
            raise Preempted()
        time.sleep(0.5)
    resp = json.load(open(rp))
    os.remove(rp)
    p_ids, r_ids, r_txt, cap = [], [], [], []
    for p, outs in zip(uniq, resp):
        enc = tok(wrap(p), add_special_tokens=False)["input_ids"]
        for o in outs:
            p_ids.append(enc)
            r_ids.append(list(o["token_ids"]))
            r_txt.append(o["text"])
            cap.append(o["finish_reason"] == "length")
    return p_ids, r_ids, r_txt, cap


VLLM = None                      # in-process engine (--vllm_inproc 1); asleep during the policy update


def vllm_setup():
    """Publish the current weights to <ckpt>/live and start a sleep-capable vLLM engine on this GPU."""
    global VLLM
    args.rollout_dir = args.rollout_dir or os.path.join(CK, "rollout")
    os.makedirs(args.rollout_dir, exist_ok=True)
    publish_weights()
    # CUDA is already initialised in this process, so vLLM would spawn its EngineCore and re-import this
    # script; run the V1 engine core in-process instead (sleep mode and reload_weights work the same way)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM
    VLLM = LLM(model=os.path.join(args.rollout_dir, "live"), gpu_memory_utilization=args.vllm_util, max_model_len=1024 + args.max_new,
               dtype="bfloat16", enable_prefix_caching=True, enable_sleep_mode=True)
    log({"step": 0, "vllm_inproc": True, "vllm_util": args.vllm_util})


def vllm_refresh():
    """After an update: wake the engine and load the freshly published weights in place."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()                          # the update's cached activations must be returned before the engine re-allocates
    free, total = torch.cuda.mem_get_info()
    print(f"[vllm] before wake: torch allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB, free {free / 2**30:.1f}/{total / 2**30:.1f} GiB", flush=True)
    VLLM.wake_up()
    VLLM.collective_rpc("reload_weights")
    VLLM.reset_prefix_cache()


def generate_inproc(prompts):
    from vllm import SamplingParams
    torch.cuda.empty_cache()                          # release the update's cached activations before the engine samples
    uniq = prompts[:: args.G]
    outs = VLLM.generate([wrap(p) for p in uniq], SamplingParams(temperature=args.temp, top_p=1.0, max_tokens=args.max_new, n=args.G,
                                                                    stop=STOP_STRINGS, seed=args.seed * 100000 + PHASE["step"]), use_tqdm=False)
    p_ids, r_ids, r_txt, cap = [], [], [], []
    for p, out in zip(uniq, outs):
        enc = tok(wrap(p), add_special_tokens=False)["input_ids"]
        for o in out.outputs:
            p_ids.append(enc)
            r_ids.append(list(o.token_ids))
            r_txt.append(o.text)
            cap.append(o.finish_reason == "length")
    return p_ids, r_ids, r_txt, cap


@torch.no_grad()
def generate(prompts):
    """(prompt_ids, response_ids, response_text, hit_cap) aligned with prompts."""
    if args.vllm_inproc:
        return generate_inproc(prompts)
    if args.rollout_dir:
        return generate_remote(prompts)
    model.eval()
    torch.cuda.empty_cache()
    enc_all = [tok(wrap(p), add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(enc_all[i]))
    p_ids, r_ids, r_txt, cap = [None] * len(prompts), [None] * len(prompts), [None] * len(prompts), [False] * len(prompts)
    for s in range(0, len(order), args.gen_bs):
        if STOP["flag"]:
            raise Preempted()
        idx = order[s: s + args.gen_bs]
        batch = tok.pad({"input_ids": [enc_all[i] for i in idx]}, padding=True, return_tensors="pt")
        ids, am = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
        kw = dict(stop_strings=STOP_STRINGS, tokenizer=tok) if STOP_STRINGS else {}
        out = model.generate(input_ids=ids, attention_mask=am, do_sample=args.temp > 0, temperature=args.temp if args.temp > 0 else None,
                             top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=args.max_new, pad_token_id=PAD_ID, use_cache=True,
                             stopping_criteria=StoppingCriteriaList([_StopOnSignal()]), **kw)
        if STOP["flag"]:
            raise Preempted()
        gen = out[:, ids.shape[1]:].tolist()
        for k, i in enumerate(idx):
            resp, ended = [], False
            for t_ in gen[k]:
                resp.append(t_)
                if t_ in EOS_IDS:
                    ended = True
                    break
            while resp and resp[-1] == PAD_ID and PAD_ID not in EOS_IDS:
                resp.pop()
            txt = tok.decode(resp, skip_special_tokens=True)
            if STOP_STRINGS:
                for ss in STOP_STRINGS:
                    if txt.endswith(ss):
                        txt, ended = txt[: -len(ss)], True
            p_ids[i], r_ids[i], r_txt[i], cap[i] = enc_all[i], resp, txt, (not ended and len(resp) >= args.max_new)
    return p_ids, r_ids, r_txt, cap


def policy_gradient_step(seqs):
    """seqs: (prompt_ids, response_ids, advantage). One update, seq-mean-token-mean, constant denominator N."""
    model.train()
    model.config.use_cache = False
    N = len(seqs)
    total_loss = 0.0
    for g in gbuf:
        g.zero_()
    if args.grad_in_param:
        for p in params:
            p.grad = None
    for s in range(0, N, args.micro_bs):
        if STOP["flag"]:
            for g in gbuf:
                g.zero_()
            for p in params:
                p.grad = None
            model.config.use_cache = True
            raise Preempted()
        chunk = seqs[s: s + args.micro_bs]
        if all(abs(a) < 1e-8 for _, _, a in chunk):
            continue                                        # zero-advantage group: no gradient
        L = max(len(p) + len(r) for p, r, _ in chunk)
        ids = torch.full((len(chunk), L), PAD_ID, dtype=torch.long)
        am = torch.zeros((len(chunk), L), dtype=torch.long)
        rmask = torch.zeros((len(chunk), L), dtype=torch.float32)
        adv = torch.tensor([a for _, _, a in chunk], dtype=torch.float32)
        for k, (p, r, a) in enumerate(chunk):
            seq = p + r
            ids[k, : len(seq)] = torch.tensor(seq)
            am[k, : len(seq)] = 1
            rmask[k, len(p): len(seq)] = 1.0
        ids, am, rmask, adv = ids.cuda(), am.cuda(), rmask.cuda(), adv.cuda()
        logits = model(input_ids=ids, attention_mask=am).logits[:, :-1].float()
        logp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        del logits
        m = rmask[:, 1:]
        seq_logp = (logp * m).sum(1) / m.sum(1).clamp_min(1.0)
        loss = -(adv * seq_logp).sum() / N
        loss.backward()
        total_loss += loss.item()
        if not args.grad_in_param:
            for g, p in zip(gbuf, params):
                if p.grad is not None:
                    g.add_(p.grad.detach().to(dtype=torch.float32))
                    p.grad = None
        del logp
    gb = gbuf if not args.grad_in_param else [q.grad if q.grad is not None else torch.zeros_like(q) for q in params]
    total_norm = math.sqrt(sum(float(torch.linalg.vector_norm(g.float() if args.grad_in_param else g)) ** 2 for g in gb))
    clip = min(1.0, 1.0 / (total_norm + 1e-6))
    for mp, g in zip(master, gb):
        if args.grad_in_param:
            mp.grad = (g.float() * clip).to(device=mp.device)
        elif mp.device.type == "cuda":
            g.mul_(clip)
            mp.grad = g                                    # no second 16 GB buffer on the card
        else:
            mp.grad = (g * clip).to(device=mp.device)      # a host copy when the master lives on the CPU
    opt.step()
    opt.zero_grad(set_to_none=True)
    if args.grad_in_param:
        for q in params:
            q.grad = None
        del gb
    for mp, p in zip(master, params):
        p.data.copy_(mp.data.to(device="cuda", dtype=torch.bfloat16))
    model.config.use_cache = True
    torch.cuda.empty_cache()
    return {"loss": total_loss, "grad_norm": total_norm, "clipped": int(total_norm > 1.0)}


# ------------------------------------------------------------------- data ---
ROWS = [json.loads(l) for l in open(args.data)]
CUR = {"epoch": 0, "pos": 0}


def _order(epoch):
    o = list(range(len(ROWS)))
    random.Random(args.seed * 1000 + epoch).shuffle(o)
    return o


def next_batch():
    out = []
    while len(out) < args.qps:
        o = _order(CUR["epoch"])
        if CUR["pos"] >= len(o):
            CUR["epoch"] += 1
            CUR["pos"] = 0
            continue
        out.append(ROWS[o[CUR["pos"]]])
        CUR["pos"] += 1
    return out


def extract(text):
    b = last_boxed(text)
    if b is None:
        if not args.answer_fallback:
            return None
        if re.search(r"Answer:\s*(unknown|unanswerable|insufficient information)", text, re.I):
            return "unknown"
        if "####" in text:
            return norm_num(text.rsplit("####", 1)[1].strip().split("\n")[0])
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return norm_num(nums[-1]) if nums else None
    if is_abstain(b):
        return "unknown"
    s = _strip_math(b)
    n = norm_num(s)
    return n if (n is not None and re.fullmatch(r"-?[\d.,]+(/\d+)?", s)) else (s or None)


# ------------------------------------------------------------- checkpoint ---
def state_snapshot():
    return {"cur": dict(CUR), "py_random": random.getstate(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}


def save_resume(step, snapshot=None, with_optimizer=True):
    import shutil
    st = snapshot if snapshot is not None else state_snapshot()
    tmp = RESUME_DIR + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    torch.save({"step": step, "n_resumes": N_RESUMES, **st}, os.path.join(tmp, "state.pt"))
    if os.path.exists(RESUME_DIR):
        shutil.rmtree(RESUME_DIR)
    os.rename(tmp, RESUME_DIR)
    if with_optimizer:
        host_master = [m.detach().to("cpu", dtype=torch.float32, copy=True) for m in master]
        sd = opt.state_dict()
        sd_host = {"state": {k: {kk: (v.detach().cpu() if torch.is_tensor(v) else v) for kk, v in st_.items()} for k, st_ in sd["state"].items()},
                   "param_groups": sd["param_groups"]}
        torch.save({"master": host_master, "opt": sd_host, "optimizer": args.optimizer}, os.path.join(RESUME_DIR, "optim.pt.tmp"))
        os.replace(os.path.join(RESUME_DIR, "optim.pt.tmp"), os.path.join(RESUME_DIR, "optim.pt"))
    globals()["LAST_SAVED"] = step


def save_weights(name):
    d = os.path.join(CK, name)
    tmp = d + ".tmp"
    import shutil
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    tok.save_pretrained(tmp)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.rename(tmp, d)
    link = os.path.join(args.out, name)
    if not os.path.lexists(link):
        os.symlink(d, link)


if args.vllm_inproc:                  # engine first (it profiles the card), optimizer state second
    vllm_setup()
    setup_optimizer()
START_STEP, N_RESUMES, LAST_SAVED = 1, 0, -1
random.seed(args.seed)
torch.manual_seed(args.seed)
if RESUME_STATE is not None:
    N_RESUMES = int(RESUME_STATE.get("n_resumes", 0)) + 1
    CUR.update(RESUME_STATE["cur"])
    random.setstate(RESUME_STATE["py_random"])
    torch.set_rng_state(RESUME_STATE["torch_rng"])
    torch.cuda.set_rng_state(RESUME_STATE["cuda_rng"])
    START_STEP = int(RESUME_STATE["step"]) + 1
    LAST_SAVED = int(RESUME_STATE["step"])
    log({"resumed_from_step": int(RESUME_STATE["step"]), "n_resumes": N_RESUMES, "optimizer_restored": OPT_RESTORED})
SAVE_AT = {int(x) for x in args.save_at.split(",") if x.strip()}
PHASE = {"step": 0}
snap = None
try:
    if RESUME_STATE is None and args.ckpt_every > 0:
        save_resume(0, with_optimizer=False)
        log({"step": 0, "resume_ckpt_saved": True})
    if args.vllm_inproc:
        pass                                          # engine and optimizer were set up before the resume block
    elif args.rollout_dir:
        os.makedirs(args.rollout_dir, exist_ok=True)
        t0 = time.time()
        publish_weights()
        log({"step": START_STEP - 1, "weights_published": WEIGHTS_VERSION, "t_publish": time.time() - t0})
    for step in range(START_STEP, args.steps + 1):
        PHASE["step"] = step
        snap = state_snapshot()
        t_step = time.time()
        qs = next_batch()
        items = [{"prompt": q["prompt"], "answer": q.get("answer"), "kind": "intact", "src": i} for i, q in enumerate(qs)]
        if args.arm == "pdc":                                   # deleted variants of this step's own questions
            pool = list(range(len(qs)))
            random.shuffle(pool)
            for i in pool:
                var, dropped = deleted_variant(qs[i]["problem"], random)
                if var is None:
                    continue
                suf = qs[i]["prompt"][len(qs[i]["problem"].strip()):]
                items.append({"prompt": var + suf, "answer": qs[i].get("answer"), "kind": "deleted", "src": i, "dropped": dropped})
                if sum(1 for it in items if it["kind"] == "deleted") >= args.n_del:
                    break
        prompts = [it["prompt"] for it in items for _ in range(args.G)]
        t0 = time.time()
        pp, rr, tt, cap = generate(prompts)
        t_gen = time.time() - t0
        seqs = []
        gt_acc = own_mean = fmt_mean = pl_correct = unanimous = 0.0
        abs_int = abs_del = flagged = 0.0
        lens = []
        modes = {}
        per = []
        for qi, it in enumerate(items):                        # first pass: answers and majorities (abstentions never vote)
            sl = slice(qi * args.G, (qi + 1) * args.G)
            texts, caps = tt[sl], cap[sl]
            ans = [extract(t) if not c else None for t, c in zip(texts, caps)]
            cnt = Counter(a for a in ans if a is not None and (a != "unknown" or args.arm == "own_uv"))   # A4g: under own_uv "unknown" votes
            if args.arm == "own_uvw":   # A4l: "unknown" votes with weight uv_weight
                nu = sum(1 for a in ans if a == "unknown")
                top = cnt.most_common(1)[0][1] if cnt else 0
                if nu > 0 and args.uv_weight * nu > top:
                    cnt = Counter({"unknown": nu})
            mode = cnt.most_common(1)[0][0] if cnt else None
            if it["kind"] == "intact":
                modes[it["src"]] = mode
            per.append((sl, texts, caps, ans, mode))
        n_int = sum(1 for it in items if it["kind"] == "intact")
        n_del = len(items) - n_int
        for it, (sl, texts, caps, ans, mode) in zip(items, per):
            r_own = [1.0 if (a is not None and a == mode) else 0.0 for a in ans]
            r_fmt = [1.0 if a is not None else 0.0 for a in ans]
            r_gt = [math_reward(t, it["answer"]) if (not c and it.get("answer") is not None) else 0.0 for t, c in zip(texts, caps)]   # label-free sets have no answer: gt stats read 0
            r_abs = [1.0 if a == "unknown" else 0.0 for a in ans]
            k = int(sum(r_own))
            r_perm = [0.0] * args.G
            for j in random.sample(range(args.G), k):
                r_perm[j] = 1.0
            if it["kind"] == "deleted":
                ill = mode is None or mode != modes.get(it["src"])   # label-free: the dropped sentence changed the majority
                flagged += float(ill)
                r = r_abs if ill else r_own
                abs_del += sum(r_abs) / args.G
            else:
                r = {"own": r_own, "perm": r_perm, "fmt": r_fmt, "gt": r_gt, "own_abs": r_own, "pdc": r_own, "own_uv": r_own, "own_uvw": r_own}[args.arm]
                abs_int += sum(r_abs) / args.G
                gt_acc += sum(r_gt) / args.G
                own_mean += sum(r_own) / args.G
                fmt_mean += sum(r_fmt) / args.G
                pl_correct += float(mode is not None and any(g > 0 and a == mode for g, a in zip(r_gt, ans)))
                unanimous += float(k == args.G)
            mu = sum(r) / args.G
            for j in range(args.G):
                seqs.append((pp[sl][j], rr[sl][j], r[j] - mu))
            lens.extend(len(x) for x in rr[sl])
        nq = n_int
        t0 = time.time()
        if VLLM is not None:
            VLLM.sleep(level=1)                       # weights to host RAM, KV cache freed: the card belongs to the update
        upd = policy_gradient_step(seqs)
        t_upd = time.time() - t0
        if args.rollout_dir:
            t0 = time.time()
            publish_weights()
            if VLLM is not None:
                vllm_refresh()
            upd["t_publish"] = time.time() - t0
        log({"step": step, "epoch": CUR["epoch"], "n_seqs": len(seqs), "gt_acc": gt_acc / nq, "own_reward": own_mean / nq,
             "fmt_rate": fmt_mean / nq, "pl_acc": pl_correct / nq, "unanimous": unanimous / nq,
             "abstain_intact": abs_int / nq, "n_del": n_del, "abstain_deleted": (abs_del / n_del) if n_del else None,
             "flagged_illposed": (flagged / n_del) if n_del else None,
             "mean_len": sum(lens) / len(lens), "trunc": sum(cap) / len(cap), **upd, "t_gen": t_gen, "t_update": t_upd,
             "t_step": time.time() - t_step})
        if step in SAVE_AT:
            save_weights(f"step{step}")
            log({"step": step, "weights_saved": f"step{step}"})
        if args.ckpt_every > 0 and step % args.ckpt_every == 0 and step < args.steps:
            t0 = time.time()
            save_resume(step, with_optimizer=False)
            log({"step": step, "resume_ckpt_saved": True, "t_ckpt": time.time() - t0})
except Preempted:
    step = PHASE["step"] - 1                 # the aborted step is replayed on resume
    if step > LAST_SAVED:
        save_resume(step, snapshot=snap, with_optimizer=True)
        log({"step": step, "preempted": True, "sig": STOP["sig"], "ckpt": "with_optimizer"})
    else:
        log({"step": step, "preempted": True, "sig": STOP["sig"], "ckpt": "none_needed"})
    LOGF.close()
    if VLLM is not None:
        os._exit(0)
    sys.exit(0)

save_weights("final")
open(os.path.join(args.out, ".trained"), "w").write(f"{args.steps}\n")
import shutil  # noqa: E402

shutil.rmtree(RESUME_DIR, ignore_errors=True)
log({"done": True, "steps": args.steps})
if VLLM is not None:
    LOGF.close()
    os._exit(0)          # the in-process engine's allocator aborts during interpreter teardown; everything is on disk already
