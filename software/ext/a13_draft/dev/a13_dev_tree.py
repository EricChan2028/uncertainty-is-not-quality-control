#!/usr/bin/env python3
"""A13 dev tests: builds a SYNTHETIC tree (no real data, no GPU) under DEV for a13_build.py, a13_launch.sh and a13_read.py.
  DEV/rsi_rl/...            fake repo: bands, R (rnd1) and S (cln1) sets built with the A6 rule, audit labels, old dumps and
                            summaries, attribution.json (computed here independently), stubs of b_train / umwp_probe /
                            umwp_probe_absfix / b_eval, pinned dummy files, a copy of the A13 files, registration hashes
  DEV/shm_<host>, DEV/arch_<host>   host-local RAM disk and archive of two simulated hosts hostA / hostB
  DEV/fakegpu/<host>/<g>.mem|.apps  what the nvidia-smi stub (DEV/bin/nvidia-smi) reports
  DEV/targets.json          CA / ACC_A / ABS_U / ABS_A / bench per run name (the stubs write dumps that realise them)
  DEV/dev_constants.sh      sourced by a13_launch.sh when A13_DEV_ROOT=DEV;  DEV/read_constants.json for a13_read.py
Usage: python3 a13_dev_tree.py DEV SRC      (SRC = the a13_draft directory; DEV must not exist or be empty)
"""
import hashlib, json, math, os, random, shutil, statistics, subprocess, sys
from scipy import stats

DEV, SRC = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
assert not DEV.startswith("/home/tione/notebook") and not DEV.startswith("/dev/shm"), DEV
R = f"{DEV}/rsi_rl"
B = f"{R}/dirB/runs"
UM = f"{R}/dirA4/results/umwp"
EV = f"{R}/dirB/results/evals"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
line = lambda r: json.dumps(r, ensure_ascii=False) + "\n"


def w(p, s, mode="w"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as fh:
        fh.write(s)


# ---------------------------------------------------------------- bands, labels, S and R (A6 rule) ---
for S in range(5):
    rnd = random.Random(1000 + S)
    rows, lab = [], {}
    for j in range(36):
        v = ["SOLVABLE"] * 12 + ["UNSOLVABLE"] * 18 + ["AMBIGUOUS"] * 6
        pid = f"p{S}_{j}"
        rows.append({"id": pid, "kind": "proposed", "problem": f"Proposal {S}-{j}: how many apples é?", "prompt": f"Q{S}-{j}"})
        lab[pid] = v[j]
    for j in range(12):
        rows.append({"id": f"ti{S}_{j}", "kind": "tracer_intact", "problem": f"Intact {S}-{j}", "prompt": f"I{S}-{j}"})
        rows.append({"id": f"td{S}_{j}", "kind": "tracer_deleted", "problem": f"Deleted {S}-{j}", "prompt": f"D{S}-{j}"})
    rnd.shuffle(rows)
    w(f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/admitted.jsonl", "".join(line(r) for r in rows))
    ids = sorted(lab)
    first = ids[:20]
    json.dump({"verdicts": [{"id": i, "verdict": lab[i]} for i in first]}, open(f"{B}/qwen3_4b_Base_own_a3_s{S}_r3/audit_admitted.json", "w"))
    os.makedirs(f"{B}/swap_audit_a5_s{S}/audit_parts", exist_ok=True)
    rest = ids[20:]
    for k, chunk in enumerate((rest[:8], rest[8:])):
        json.dump({"verdicts": [{"id": i, "verdict": lab[i]} for i in chunk]}, open(f"{B}/swap_audit_a5_s{S}/audit_parts/{k * 25}-{k * 25 + 25}.json", "w"))
    cln = [a for a in rows if (a["kind"] == "proposed" and lab[a["id"]] == "SOLVABLE") or a["kind"] == "tracer_intact"]
    rs = random.Random(20260924 + S).sample(rows, len(cln))
    for arm, items in (("cln1", cln), ("rnd1", rs)):
        d = f"{B}/qwen3_4b_Base_{arm}_a6_s{S}_r3"
        w(f"{d}/admitted.jsonl", "".join(line(a) for a in items))
        cfg = {"model": f"/dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s{S}_r2/final", "uv_weight": 3.0, "arm": "own", "n_del": 16,
               "seed": S, "out": d, "ckpt_dir": "/dev/shm/ckpt_b", "data": f"{d}/admitted.jsonl", "steps": 60, "qps": 32,
               "G": 8, "lr": 1e-06, "max_new": 1024, "temp": 1.0, "gen_bs": 64, "micro_bs": 4, "ckpt_every": 10,
               "save_at": "20,40", "optimizer": "adamw8bit", "grad_in_param": 0, "optimizer_device": "cuda",
               "attn": "flash_attention_2", "chat": -1, "rollout_dir": "", "rollout_timeout": 1500, "answer_fallback": 1,
               "vllm_inproc": 1, "vllm_util": 0.25}
        json.dump(cfg, open(f"{d}/config.json", "w"), indent=1)

# ---------------------------------------------------------------- targets and old dumps ---
S_CA = [0.600, 0.450, 0.620, 0.700, 0.660]
RS_D = [0.065, 0.085, 0.070, 0.040, 0.020]
R0_D = [0.040, 0.050, 0.030, 0.035, 0.045]            # ADDS by design (E_prop ~ +4 points, small spread)
T = {}
for s in range(5):
    T[f"qwen3_4b_Base_cln1_a6_s{s}_r3"] = {"CA": S_CA[s], "ACC_A": 95.0 + s * 0.2, "ABS_U": 66.0 + s, "ABS_A": 2.0, "bench": 82.0 + 0.1 * s}
    T[f"qwen3_4b_Base_rnd1_a6_s{s}_r3"] = {"CA": S_CA[s] + RS_D[s], "ACC_A": 95.2, "ABS_U": 68.0 + s, "ABS_A": 2.5, "bench": 81.2 + 0.1 * s}
    T[f"qwen3_4b_Base_r0_a13_s{s}_r3"] = {"CA": S_CA[s] + R0_D[s], "ACC_A": 95.1, "ABS_U": 67.0 + s, "ABS_A": 2.2, "bench": 81.6 + 0.1 * s}
json.dump(T, open(f"{DEV}/targets.json", "w"), indent=1)

ROWGEN = r'''
def umwp_rows(t, abstain_prompt):
    """1,200 rows realising the targets: CA = round(CA*600)/600, ACC_A = round(ACC_A*6)/6, ABS_* likewise."""
    k = round(t["CA"] * 600); ka = round(t["ACC_A"] / 100 * 600)
    ku = round(t["ABS_U"] / 100 * 600); kA = round(t["ABS_A"] / 100 * 600)
    out = []
    for i in range(600):
        if i < k: mode, p = 7, (0.5 if i % 3 == 0 else 0.875)          # confident (p_mode >= 0.5; the boundary included)
        elif i % 2: mode, p = 7, 0.375                                  # a mode, but not confident
        else: mode, p = None, 0.625                                     # no numeric mode
        ab = (1.0 if i < ku else 0.0) if abstain_prompt else 0.0
        out.append({"id": "u%d" % i, "answerable": False, "category": 1 + i % 5, "source": "dev", "ref": None, "mode": mode,
                    "p_mode": p, "nonanswer": 0.0, "abstain": ab, "correct": 0, "answers": []})
    for i in range(600):
        ab = (1.0 if i < kA else 0.0) if abstain_prompt else 0.0
        out.append({"id": "a%d" % i, "answerable": True, "category": None, "source": "dev", "ref": 3, "mode": 3 if i < ka else 4,
                    "p_mode": 1.0, "nonanswer": 0.0, "abstain": ab, "correct": int(i < ka), "answers": []})
    return out
'''
exec(ROWGEN)
os.makedirs(UM, exist_ok=True)
for a in ("cln1", "rnd1"):
    for s in range(5):
        n = f"qwen3_4b_Base_{a}_a6_s{s}_r3"
        rows = umwp_rows(T[n], 0)
        with open(f"{UM}/plain_{n}.jsonl", "w") as fh:
            for r in rows[:5]:                     # a duplicated id first (the read de-duplicates, last row wins)
                fh.write(json.dumps(dict(r, mode=None)) + "\n")
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        acc = T[n]["bench"] / 100
        os.makedirs(EV, exist_ok=True)
        json.dump({b: {"greedy_acc_lenient": acc, "n": 500} for b in ("math500_train", "math_heldout1000", "gsm8k_test500")},
                  open(f"{EV}/{n}_step60.summary.json", "w"))


def ca_of(n):
    k = round(T[n]["CA"] * 600)
    return k / 600


def paired(xa, xb):
    d = [a - b for a, b in zip(xa, xb)]
    m, sd = statistics.mean(d), statistics.stdev(d)
    t = float(stats.t.ppf(0.975, len(d) - 1)); se = sd / math.sqrt(len(d))
    return {"diffs": d, "mean": m, "sd": sd, "ci95": [m - t * se, m + t * se]}


pc = paired([ca_of(f"qwen3_4b_Base_cln1_a6_s{s}_r3") for s in range(5)], [ca_of(f"qwen3_4b_Base_rnd1_a6_s{s}_r3") for s in range(5)])
pb = paired([T[f"qwen3_4b_Base_cln1_a6_s{s}_r3"]["bench"] for s in range(5)], [T[f"qwen3_4b_Base_rnd1_a6_s{s}_r3"]["bench"] for s in range(5)])
w(f"{R}/paper_arr/analyses/attribution.json", json.dumps({"paired": {"cln1-rnd1:CA": pc, "cln1-rnd1:bench": pb}}, indent=1))

# ---------------------------------------------------------------- stubs and pinned dummies ---
STUB_TRAIN = r'''#!/usr/bin/env python3
"""DEV STUB of dirB/code/b_train.py: same flags and defaults, config.json = vars(args); 60 fake steps; resume; TERM."""
import argparse, json, os, signal, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--uv_weight", type=float, default=3.0)
ap.add_argument("--arm", required=True); ap.add_argument("--n_del", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--ckpt_dir", default="/dev/shm/ckpt_b"); ap.add_argument("--data", default="x")
ap.add_argument("--steps", type=int, default=60); ap.add_argument("--qps", type=int, default=32); ap.add_argument("--G", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-6); ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--gen_bs", type=int, default=64); ap.add_argument("--micro_bs", type=int, default=4); ap.add_argument("--ckpt_every", type=int, default=10)
ap.add_argument("--save_at", default="20,40"); ap.add_argument("--optimizer", default="adamw8bit"); ap.add_argument("--grad_in_param", type=int, default=0)
ap.add_argument("--optimizer_device", default="cuda"); ap.add_argument("--attn", default="flash_attention_2"); ap.add_argument("--chat", type=int, default=-1)
ap.add_argument("--rollout_dir", default=""); ap.add_argument("--rollout_timeout", type=int, default=1500)
ap.add_argument("--answer_fallback", type=int, default=0); ap.add_argument("--vllm_inproc", type=int, default=0); ap.add_argument("--vllm_util", type=float, default=0.25)
args = ap.parse_args()
DEV = os.environ["A13_DEV_ROOT"]
os.makedirs(args.out, exist_ok=True)
RUN = os.path.basename(os.path.normpath(args.out))
CK = os.path.join(os.path.dirname(os.path.dirname(args.model)), RUN)      # <host shm>/ckpt_b/<run>
os.makedirs(CK, exist_ok=True)
json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
LOG = open(os.path.join(args.out, "log.jsonl"), "a")
def log(d): d["time"] = time.time(); LOG.write(json.dumps(d) + "\n"); LOG.flush()
STOP = {"f": False}
signal.signal(signal.SIGTERM, lambda *a: STOP.update(f=True))
RES = os.path.join(CK, "resume", "state.json")
def save(step, opt=False):          # as b_train: the preemption save carries the optimizer state, the periodic ones do not
    os.makedirs(os.path.dirname(RES), exist_ok=True); json.dump({"step": step, "opt": opt}, open(RES + ".tmp", "w")); os.replace(RES + ".tmp", RES)
start = 1
if os.path.exists(RES):
    st = json.load(open(RES)); start = st["step"] + 1
    log({"resumed_from_step": st["step"], "n_resumes": 1, "optimizer_restored": bool(st.get("opt"))})
else:
    save(0); log({"step": 0, "resume_ckpt_saved": True})
delay = 0.5 if os.path.exists(f"{DEV}/inject/slow_{RUN}") else 0.03
crash = f"{DEV}/inject/crash_{RUN}"
for step in range(start, args.steps + 1):
    if STOP["f"]:
        save(step - 1, True); log({"step": step - 1, "preempted": True}); sys.exit(0)
    time.sleep(delay)
    if os.path.exists(crash) and step == 5:
        os.remove(crash); print("injected crash", flush=True); os._exit(1)
    log({"step": step, "epoch": 0, "loss": 0.1})
    if step in (20, 40):
        os.makedirs(os.path.join(CK, "step%d" % step), exist_ok=True); open(os.path.join(CK, "step%d" % step, "w.bin"), "wb").write(b"x" * 1000)
    if step % args.ckpt_every == 0 and step < args.steps:
        save(step)
os.makedirs(os.path.join(CK, "rollout", "live"), exist_ok=True)
f = os.path.join(CK, "final"); os.makedirs(f, exist_ok=True)
json.dump({"model_type": "qwen3"}, open(os.path.join(f, "config.json"), "w"))
json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}, open(os.path.join(f, "model.safetensors.index.json"), "w"))
for n in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json"):
    open(os.path.join(f, n), "wb").write(RUN.encode() * 10)
open(os.path.join(args.out, ".trained"), "w").write("%d\n" % args.steps)
import shutil; shutil.rmtree(os.path.join(CK, "resume"), ignore_errors=True)
log({"done": True, "steps": args.steps})
'''
STUB_PROBE = r'''#!/usr/bin/env python3
"""DEV STUB of the UMWP probe (plain or corrected abstention): writes rows realising DEV/targets.json; resumes by id."""
import argparse, json, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--abstain_prompt", type=int, default=0); ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--gpu_util", type=float, default=0.85); ap.add_argument("--n_uns", type=int, default=600)
args = ap.parse_args()
ABSFIX = __ABSFIX__
if ABSFIX:
    assert args.tag.startswith("absfix_") and args.abstain_prompt == 1
DEV = os.environ["A13_DEV_ROOT"]
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{R}/dirA4/results/umwp"; os.makedirs(OUT, exist_ok=True)
run = os.path.basename(os.path.dirname(args.model.rstrip("/")))
T = json.load(open(f"{DEV}/targets.json"))[run]
__ROWGEN__
rows = umwp_rows(T, args.abstain_prompt)
outp = f"{OUT}/{args.tag}.jsonl"
done = set()
if os.path.exists(outp):
    for l in open(outp):
        try: done.add(json.loads(l)["id"])
        except ValueError: pass
crash = f"{DEV}/inject/crash_{args.tag}"
with open(outp, "a") as fh:
    for k, r in enumerate([r for r in rows if r["id"] not in done]):
        if os.path.exists(crash) and k == 300:
            os.remove(crash); fh.write('{"id": "partial'); fh.flush(); print("injected crash", flush=True); os._exit(1)
        fh.write(json.dumps(r) + "\n")
print("done", flush=True)
'''
STUB_BEVAL = r'''#!/usr/bin/env python3
"""DEV STUB of dirB/code/b_eval.py: writes <out>/<tag>.summary.json from DEV/targets.json."""
import argparse, json, os
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data", default="math500_train,math_heldout1000,gsm8k_test500"); ap.add_argument("--k", type=int, default=8)
args = ap.parse_args()
DEV = os.environ["A13_DEV_ROOT"]
run = os.path.basename(os.path.dirname(args.model.rstrip("/")))
acc = json.load(open(f"{DEV}/targets.json"))[run]["bench"] / 100
os.makedirs(args.out, exist_ok=True)
json.dump({b: {"greedy_acc_lenient": acc, "n": 500} for b in args.data.split(",")}, open(f"{args.out}/{args.tag}.summary.json", "w"))
'''
w(f"{R}/dirB/code/b_train.py", STUB_TRAIN)
w(f"{R}/dirA3/code/umwp_probe.py", STUB_PROBE.replace("__ABSFIX__", "False").replace("__ROWGEN__", ROWGEN))
w(f"{R}/ext/a11_draft/umwp_probe_absfix.py", STUB_PROBE.replace("__ABSFIX__", "True").replace("__ROWGEN__", ROWGEN))
w(f"{R}/dirB/code/b_eval.py", STUB_BEVAL)
pins = ["dirB/code/pdc_util.py", "code/loto_common.py", "data_ext/UMWP/data/StandardDataset.jsonl",
        "dirB/data/math500_train.jsonl", "dirB/data/math_heldout1000.jsonl", "dirB/data/gsm8k_test500.jsonl"]
for p in pins:
    w(f"{R}/{p}", "dev dummy %s\n" % p)

# ---------------------------------------------------------------- hosts: RAM disks, archives, checkpoints, fake GPUs ---
def fake_model(d, tag):
    os.makedirs(d, exist_ok=True)
    json.dump({"model_type": "qwen3", "dev": True}, open(f"{d}/config.json", "w"))
    json.dump({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}, open(f"{d}/model.safetensors.index.json", "w"))
    rr = random.Random(tag)
    for n in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        open(f"{d}/{n}", "wb").write(bytes(rr.getrandbits(8) for _ in range(4096)))
    open(f"{d}/tokenizer.json", "w").write("{}")


HOSTS = {"hostA": {"start": [0, 1, 3, 4], "old_shm": [0, 3], "old_arch": []},
         "hostB": {"start": [0, 1, 2, 4], "old_shm": [], "old_arch": [1, 2, 4]}}
for h, spec in HOSTS.items():
    shm, arch = f"{DEV}/shm_{h}", f"{DEV}/arch_{h}"
    os.makedirs(f"{shm}/ckpt_b", exist_ok=True); os.makedirs(arch, exist_ok=True)
    for s in spec["start"]:           # the real layout: /dev/shm/ckpt_b/<prev>/final -> archive (symlink)
        a = f"{arch}/qwen3_4b_Base_own_a3_s{s}_r2/final"; fake_model(a, "start%d" % s)
        os.makedirs(f"{shm}/ckpt_b/qwen3_4b_Base_own_a3_s{s}_r2", exist_ok=True)
        os.symlink(a, f"{shm}/ckpt_b/qwen3_4b_Base_own_a3_s{s}_r2/final")
    for s in spec["old_shm"]:
        for a in ("cln1", "rnd1"):
            fake_model(f"{shm}/ckpt_b/qwen3_4b_Base_{a}_a6_s{s}_r3/final", f"{a}{s}")
    for s in spec["old_arch"]:
        for a in ("cln1", "rnd1"):
            fake_model(f"{arch}/qwen3_4b_Base_{a}_a6_s{s}_r3/final", f"{a}{s}")
    os.makedirs(f"{DEV}/fakegpu/{h}", exist_ok=True)
os.makedirs(f"{DEV}/inject", exist_ok=True)
w(f"{DEV}/bin/nvidia-smi", r'''#!/bin/bash
D=$A13_DEV_ROOT/fakegpu/$A13_DEV_HOST
case "$*" in
  *--query-compute-apps*) cat $D/*.apps 2>/dev/null; exit 0;;
  *uuid*) echo "GPU-dev-$A13_DEV_HOST-$2";;
  *memory.used*) cat $D/$2.mem 2>/dev/null || echo 0;;
esac
''')
os.chmod(f"{DEV}/bin/nvidia-smi", 0o755)

# ---------------------------------------------------------------- A13 files (copies), registration hashes, build ---
dst = f"{R}/ext/a13_draft"
shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
files = ["A13_registration_2026-09-27.md", "a13_build.py", "a13_launch.sh", "a13_read.py", "dev/a13_dev_test.sh", "dev/a13_dev_tree.py"]
w(f"{R}/docs/registration_hashes.txt", "".join("%s  ext/a13_draft/%s\n" % (sha(f"{dst}/{f}"), f) for f in files if os.path.exists(f"{dst}/{f}")))
os.makedirs(f"{R}/ext/a13_runs/runs", exist_ok=True)
b = subprocess.run([sys.executable, f"{dst}/a13_build.py", "--out_root", f"{R}/ext/a13_runs/runs", "--root", R, "--dev"],
                   capture_output=True, text=True)
assert b.returncode == 0, b.stdout + b.stderr
sets = json.loads(b.stdout)["sets"]
r0 = {str(s): sets[str(s)]["sha256"] for s in range(5)}
ck = lambda s, n: sha(f"{DEV}/arch_{'hostA' if s != 2 else 'hostB'}/qwen3_4b_Base_own_a3_s{s}_r2/final/{n}")
cst = f'''# dev constants for a13_launch.sh (sourced when A13_DEV_ROOT={DEV})
R={R}; SHM={DEV}/shm_$A13_DEV_HOST; ARCH={DEV}/arch_$A13_DEV_HOST; NVSMI={DEV}/bin/nvidia-smi
SHA_B_TRAIN={sha(f"{R}/dirB/code/b_train.py")}
SHA_UMWP={sha(f"{R}/dirA3/code/umwp_probe.py")}
SHA_ABSFIX={sha(f"{R}/ext/a11_draft/umwp_probe_absfix.py")}
SHA_B_EVAL={sha(f"{R}/dirB/code/b_eval.py")}
PINS="{" ".join("%s:%s" % (p, sha(f"{R}/{p}")) for p in pins)}"
R0SHA=({" ".join(r0[str(s)] for s in range(5))})
CKSHA=({" ".join('"%s %s"' % (ck(s, "model-00001-of-00002.safetensors"), ck(s, "model-00002-of-00002.safetensors")) for s in range(5))})
CKCFG={ck(0, "config.json")}
IDLE_GAP=1; SCAN=1; MON=1; CONFIRM=1; COOLDOWN=4; YIELD_WAIT=2; KILL_GRACE=2; TRAIN_GRACE=5; STOP_GRACE=20
PREF_DELAY=${{A13_DEV_PREF_DELAY:-15}}; T_TRAIN_LAST=${{A13_DEV_T_TRAIN_LAST:-600}}; T_EVAL_LAST=${{A13_DEV_T_EVAL_LAST:-900}}
TMO_TRAIN=120; TMO_UMWP=60; TMO_BEVAL=60; MAXPAR=${{A13_MAXPAR:-6}}
'''
w(f"{DEV}/dev_constants.sh", cst)
rc = {"umwp_probe_sha256": sha(f"{R}/dirA3/code/umwp_probe.py"), "absfix_probe_sha256": sha(f"{R}/ext/a11_draft/umwp_probe_absfix.py"),
      "b_eval_sha256": sha(f"{R}/dirB/code/b_eval.py"), "b_train_sha256": sha(f"{R}/dirB/code/b_train.py"), "r0_sha256": r0,
      "paper_rs": [round(100 * pc["mean"], 1), round(100 * pc["ci95"][0], 1), round(100 * pc["ci95"][1], 1)],
      "paper_rs_bench": round(pb["mean"], 1), "config_ignore": ["out", "data", "model"], "old_sha256": {}}
json.dump(rc, open(f"{DEV}/read_constants.json", "w"), indent=1)
print(json.dumps({"dev_tree": DEV, "r0_sha256": r0, "paper_rs": rc["paper_rs"]}))
