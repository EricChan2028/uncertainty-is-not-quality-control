"""A4 addition 2: co-evolving proposer by filtered supervised fine-tuning. The proposer (a base LM used with the
a3_propose.py prompt "Task: <stem>\nProblem:") is fine-tuned on the proposals the guard ADMITTED this round, so the next
round's proposer writes more of what the current solver's guard accepts (R-Zero's selection pressure, without a separate
RL trainer). Loss on the problem tokens only. Output: an HF checkpoint dir whose name contains "Base" so a3_propose.py
keeps the base-style prompt.

  python3 a4_proposer_sft.py --model <proposer> --data <admitted.jsonl> --proposed <proposed.jsonl> --out <dir>/proposer_Base
"""
import argparse
import json
import math
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="admitted.jsonl of the round (kind == proposed items are used)")
ap.add_argument("--proposed", required=True, help="proposed.jsonl of the round (stems)")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max_len", type=int, default=384)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed); torch.manual_seed(args.seed)

stems = {}
for l in open(args.proposed):
    d = json.loads(l)
    stems[d["id"]] = d.get("stem", "")
items = [json.loads(l) for l in open(args.data)]
items = [x for x in items if x.get("kind") == "proposed" and x["id"] in stems]
print(f"admitted proposals for SFT: {len(items)}", flush=True)
if len(items) < 20:
    raise SystemExit("too few admitted proposals for SFT")

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda()
model.gradient_checkpointing_enable()
model.config.use_cache = False

def encode(x):
    prompt = f"Task: {stems[x['id']]}\nProblem:"
    target = " " + x["problem"].strip() + "\n\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + t_ids)[: args.max_len]
    labels = ([-100] * len(p_ids) + t_ids)[: args.max_len]
    return ids, labels

data = [encode(x) for x in items]
try:
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.0)
except Exception:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(data) / args.bs)
total = steps_per_epoch * args.epochs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(1, total)))
model.train()
step = 0
for ep in range(args.epochs):
    random.shuffle(data)
    for i in range(0, len(data), args.bs):
        batch = data[i:i + args.bs]
        L = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, (a, b) in enumerate(batch):
            ids[j, :len(a)] = torch.tensor(a); lab[j, :len(b)] = torch.tensor(b); att[j, :len(a)] = 1
        out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total:
            print(f"epoch {ep} step {step}/{total} loss {out.loss.item():.4f}", flush=True)
os.makedirs(args.out, exist_ok=True)
model.config.use_cache = True
model.save_pretrained(args.out, safe_serialization=True)
tok.save_pretrained(args.out)
json.dump({"n_items": len(items), "epochs": args.epochs, "lr": args.lr, "steps": total, "source": args.model}, open(os.path.join(args.out, "sft_meta.json"), "w"))
print("saved", args.out, flush=True)
