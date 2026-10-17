"""A7-5 (d): per iteration, the admitted (filtered) questions -> a shared copy, a pool summary, and the 200-item audit sample
for the frozen Opus audit workflow.
The admitted set of iteration i is the file upload.py wrote for <abbr>_solver_v<i>: questions of questioner Q_i that parsed,
were pseudo-labelled by solver S_(i-1) (9 samples, majority) and passed 0.3 <= score <= 0.8. It is the dataset passed to the
solver trainer of iteration i, which additionally drops prompts over 2,048 tokens (filter_overlong_prompts) and the last
incomplete batch of 512 (drop_last); the summary counts the rows over 2,048 tokens with the trainer's tokenizer, messages and
chat template. Audit sample: random.Random(20260924 + i).sample(range(n), min(200, n)) over its n rows (a repeated question
counts by multiplicity, as in training). It is written as the pseudo-run dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/admitted.jsonl
(kind "proposed", the format the frozen workflow reads); judges see only id and problem text. Idempotent: an existing sample
is never redrawn. Runs only on the host that owns the seed (ext/rzero_runs/s<S>.host, fix 4). CPU only. Concurrent calls (the
watcher, the driver and its exit handler, the monitor, by hand) are serialized by a lock on the seed's local storage (review
item 9). May be run by hand at any time, also after T0 + 71 h for the admitted sets that existed by then (review item 5).
    python3 rz_extract.py --seed S [--iters 1 2 3]"""
import argparse, collections, fcntl, hashlib, json, os, random, re, shutil, socket, sys, time
import pyarrow.parquet as ppq

R = "/home/tione/notebook/rsi_rl"; RZ = "/run/ti/rzero"
TOK = "/root/models/Qwen3-4B-Base"  # the tokenizer of every solver checkpoint (merged checkpoints keep it)
SYS = r"Please reason step by step, and put your final answer within \boxed{}."  # solver system message, verl/utils/dataset.py
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2, 3])
ap.add_argument("--iters", type=int, nargs="+", default=[1, 2, 3])
a = ap.parse_args()
abbr = f"rzero_qwen3_4b_Base_s{a.seed}"; gq = f"{RZ}/s{a.seed}/generated_question"
hf = f"{R}/ext/rzero_runs/s{a.seed}.host"
if not os.path.exists(hf):
    sys.exit(f"s{a.seed}: this seed has not started on any host ({hf} missing)")
owner = open(hf).read().split()[0]
if owner != socket.gethostname():
    sys.exit(f"refusing: seed {a.seed} belongs to host {owner}; this is {socket.gethostname()}")
X = f"{R}/ext/rzero_runs/s{a.seed}"; os.makedirs(X, exist_ok=True)
# review item 9: one extraction at a time per seed (a lock on the seed's local storage); a call that cannot get it within
# 15 min gives up, and the next call extracts
os.makedirs(f"{RZ}/s{a.seed}", exist_ok=True)
_lock = open(f"{RZ}/s{a.seed}/extract.lock", "a"); _t = time.time()
while True:
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except OSError:
        if time.time() - _t > 900:
            sys.exit(f"s{a.seed}: another extraction has held {RZ}/s{a.seed}/extract.lock for 15 min; not run")
        time.sleep(2)


def is_numeric(label):
    t = re.sub(r"\s|\$|\\[,!;]", "", str(label)).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", t)
    return bool(re.fullmatch(r"-?\d+(\.\d+)?|-?\d+/\d+|-?\\frac\{\d+\}\{\d+\}", t))


_tok = []


def n_overlong(rows):
    """rows the trainer's filter_overlong_prompts drops (same messages and length test as verl/utils/dataset.py)"""
    try:
        if not _tok:
            from transformers import AutoTokenizer
            _tok.append(AutoTokenizer.from_pretrained(TOK))
        t = _tok[0]
        if t.chat_template:
            return sum(len(t.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["problem"]}],
                                                 add_generation_prompt=True)) > 2048 for r in rows)
        return sum(len("system: " + SYS + "\n" + "user: " + r["problem"]) > 2048 for r in rows)
    except Exception as e:  # descriptive only
        print("overlong count unavailable:", repr(e)[:200])
        return None


for it in a.iters:
    name = f"{abbr}_solver_v{it}"
    pq, js, pool_f = f"{gq}/{name}_admitted.parquet", f"{gq}/{name}_admitted.jsonl", f"{gq}/{name}_all_scored.json"
    if not os.path.exists(pq):
        print(f"s{a.seed} it{it}: no admitted set yet"); continue
    rows = [json.loads(l) for l in open(js)]
    assert ppq.read_metadata(pq).num_rows == len(rows), "admitted jsonl and parquet row counts differ"
    run = f"rzero_qwen3_4b_Base_s{a.seed}_it{it}"; d = f"{R}/dirB/runs/{run}"
    if os.path.exists(f"{d}/admitted.jsonl"):
        print(f"s{a.seed} it{it}: audit sample exists ({d}), not redrawn")
    else:
        idx = random.Random(20260924 + it).sample(range(len(rows)), min(200, len(rows)))
        os.makedirs(f"{d}/audit_parts", exist_ok=True)
        with open(f"{d}/admitted.jsonl.tmp", "w") as fh:
            for rank, j in enumerate(idx):
                r = rows[j]
                fh.write(json.dumps({"id": f"rz{a.seed}i{it}r{j}", "problem": r["problem"], "prompt": r["problem"], "kind": "proposed",
                                     "rz_row": j, "rz_rank": rank, "rz_label": r["answer"], "rz_score": r["score"]}, ensure_ascii=False) + "\n")
        os.replace(f"{d}/admitted.jsonl.tmp", f"{d}/admitted.jsonl")
        print(f"s{a.seed} it{it}: audit sample of {len(idx)} / {len(rows)} admitted written to {d}")
    shutil.copy(js, f"{X}/{name}_admitted.jsonl")
    raw, shards = [], 0
    for k in range(8):
        f = f"{gq}/{name}_{k}.json.raw"
        if os.path.exists(f):
            raw += json.load(open(f)); shards += 1
    pool = json.load(open(pool_f)) if os.path.exists(pool_f) else []
    sha = hashlib.sha256(open(js, "rb").read()).hexdigest()
    sf = f"{X}/{name}_summary.json"
    old = json.load(open(sf)) if os.path.exists(sf) else {}
    over = old.get("n_admitted_over_2048_tokens") if old.get("sha256_admitted_jsonl") == sha else None
    if over is None:
        over = n_overlong(rows)
    summ = {"seed": a.seed, "iteration": it, "sample_seed": 20260924 + it, "audit_run": run, "audit_sample_n": min(200, len(rows)),
            "n_raw_shards": shards, "n_generated": len(raw), "n_parsed": sum(r.get("score") == 0 for r in raw),
            "n_scored": len(pool), "n_admitted": len(rows), "n_admitted_unique": len({r["problem"] for r in rows}),
            "n_admitted_over_2048_tokens": over,
            "n_solver_batches_of_512": (len(rows) - over) // 512 if over is not None else None,
            "admitted_share_of_scored": len(rows) / len(pool) if pool else None,
            "share_numeric_label_admitted": sum(is_numeric(r["answer"]) for r in rows) / len(rows) if rows else None,
            "pool_score_hist": dict(sorted(collections.Counter(round(r["score"], 3) for r in pool).items())),
            "admitted_parquet_mtime": os.path.getmtime(pq), "extract_host": socket.gethostname(),
            "sha256_admitted_jsonl": sha}
    json.dump(summ, open(sf + ".tmp", "w"), indent=1); os.replace(sf + ".tmp", sf)
    print(json.dumps({k: summ[k] for k in ("iteration", "n_generated", "n_parsed", "n_scored", "n_admitted", "n_admitted_unique",
                                           "n_admitted_over_2048_tokens")}))
