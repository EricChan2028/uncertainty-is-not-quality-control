#!/bin/bash
# A13 launcher (registration A13_registration_2026-09-27.md in this directory, sections 4-5). Per Qwen3-4B seed S (0-4): one
# round of b_train (60 steps, arm own, the flags of dirA5/code/a6_cln.sh) from the round-2 solver on the registered R0 set
# (a13_build.py), then on the same host the UMWP plain probe (primary), the A11-corrected abstention probe and b_eval. Tier 2
# (descriptive): the corrected abstention probe on the existing R (rnd1_a6) and S (cln1_a6) checkpoints. One instance per host;
# the hosts share the NFS repo; checkpoints are host-local, so a seed's evaluations run on the host that trained it.
#   bash a13_launch.sh build          once, before the first run (either host): the five R0 sets (a13_build.py, idempotent)
#   bash a13_launch.sh locate         read-only: where each cell's model is on THIS host
#   bash a13_launch.sh check [full]   read-only preflight (pinned code/data/sets/packages, registration; full: start-checkpoint sha256)
#   bash a13_launch.sh run            the registered run: every cell whose model is on this host
#   bash a13_launch.sh stop           safe stop of this host's launcher (stop flag, TERM its jobs, wait, cleanup)
#   bash a13_launch.sh status         the status files and the launch log tail
# Cells: train:S, plain:S, absfix:S, beval:S (S = 0..4) and oabs:<cln1|rnd1>_S. Train cells prefer a host (the host of port
# <HOST> takes seed 3, whose start checkpoint is only there; the host of port <PORT> takes seeds 2, 1, 0, 4; seed 2's start
# checkpoint is only there, seeds 0, 1, 4 are on both); a non-preferred seed is taken only if still unclaimed PREF_DELAY (300 s)
# after T0. A train cell takes its cross-host claim only once it holds a GPU (if the claim fails, it releases the GPU), and
# gives the claim up again when it has waited PREF_DELAY s for a GPU between two attempts: a host without a free GPU never
# holds a seed that the other host could train. A seed's evaluation cells become startable once its training finished on
# this host.
# GPUs (as A12): a GPU is FREE only if it has no marker (/dev/shm/gpu_single_<g>), uses < 1500 MiB, carries no foreign compute
# process and is not cooling down; a GPU cools down for COOLDOWN (600) s after a yield of this host and after every check that
# found a foreign compute process or >= 1500 MiB on it without a marker (so a short gap in a co-tenant's use is not taken). It
# must be free at two checks IDLE_GAP (60) s apart. Taken in random order, atomically with ( set -o noclobber; : >
# /dev/shm/gpu_single_<g> ), under a lock of this launcher; one job per GPU; at most A13_MAXPAR (default 6) GPUs per host;
# markers released after every attempt and by the exit handler; at start, orphan jobs and markers of a DEAD A13 launcher of
# this host are removed. Priority: train and plain cells (level 0), then absfix and beval (1), then oabs (2). A cell takes a
# GPU only if more GPUs are free than cells of a lower level wait for one; a cell of level >= 1 is started only if a GPU is
# free for it and fewer than A13_MAXPAR cells run, an oabs cell only if no train or plain cell of this host waits for a GPU;
# level-0 cells may start while up to A13_MAXPAR + 2 cells run (the GPU cap still holds). Cross-host de-duplication: an atomic
# mkdir claim per cell under ext/a13_runs/claims (NFS); a claim of another host is never taken over.
# Jobs: each starts with setsid (own session and process group) and carries A13_TAG; killed by signalling the group and every
# listed descendant (A12's kill_job). Co-tenant priority: every MON (60) s a job's GPU is checked; a compute process there that
# is not ours (confirmed CONFIRM (30) s later) makes the job YIELD: TERM (b_train then writes its resume checkpoint), KILL
# after KILL_GRACE (20 s; 120 s for training; a job's own timeout sends its KILL only STOP_GRACE + 30 s after a TERM, so it never
# cuts this grace short), GPU released and cooled down COOLDOWN (600) s for this host, the cell waits
# YIELD_WAIT (600) s and claims again (a yield is not a failed attempt; at most 12 per cell). Yield by hand: touch
# ext/a13_runs/STOP (all hosts) or ext/a13_runs/STOP_<host>: no new attempt starts; running jobs finish.
# Resume: b_train resumes from /dev/shm/ckpt_b/<run>/resume (same host); the probes skip finished ids after a partial last line
# is cut; b_eval resumes per set. Up to 3 attempts per cell per launch, each on a freshly claimed GPU, each capped by timeout;
# at every attempt the script's sha256 is checked against its pin and recorded in the sidecar. After a training its final is
# also copied to the host-local disk archive (sizes verified, atomic rename); the evaluations fall back to that copy.
# Clock: T0 = ext/a13_runs/launch_epoch (written once by the first launcher past the preflight). No train attempt starts after
# T0 + 2 h, no evaluation attempt after T0 + 3.5 h; the read (a13_read.py) runs after both launchers have exited, at the latest
# at T0 + 4.5 h; unfinished cells are NOT RUN.
# Log: ext/a13_runs/a13_launch.log; status per host: ext/a13_runs/status_<host>.tsv; job logs: ext/a13_runs/logs/.
# Dev tests only: A13_DEV_ROOT=<dir> redirects every path (repo, /dev/shm, archive, nvidia-smi, host name, constants) into a
# synthetic tree under /tmp; it refuses to run if that tree is the real repo.
set -u
MODE=${1:-locate}; OPT=${2:-}
R=/home/tione/notebook/rsi_rl; SHM=/dev/shm; ARCH=/run/ti/rsi_ckpt_archive; NVSMI=nvidia-smi; PY=/usr/bin/python3
H=$(hostname | cut -c1-22)
C=$(cd "$(dirname "$0")" && pwd)
MAXPAR=${A13_MAXPAR:-6}
PRIORITY=${A13_PRIORITY:-1}   # lead author, 2026-09-27: A13 has top priority; 1 = never yield, kill foreign processes on our GPUs
IDLE_GAP=60; SCAN=30; MON=60; CONFIRM=30; COOLDOWN=600; YIELD_WAIT=600; KILL_GRACE=20; TRAIN_GRACE=120; STOP_GRACE=180
PREF_DELAY=300; T_TRAIN_LAST=7200; T_EVAL_LAST=12600
TMO_TRAIN=6000; TMO_UMWP=1800; TMO_BEVAL=3600
HOST_A=<NODE>; HOST_B=<NODE>   # ports <PORT> / <HOST>
case $H in $HOST_A) PREF="3";; $HOST_B) PREF="2 1 0 4";; *) PREF="";; esac
# registered pins (section 4 of the registration)
SHA_B_TRAIN=fb9156b0ef6a685029c14c2ca52c4f17b6f0302a97c657c43cffeec1945d97d7
SHA_UMWP=32de59f95a296fa0c026b44cf8a6110172898b87b258179ab3a54da00ee1e56d
SHA_ABSFIX=11f796c2af5c56afe72c802cf34e444e391c1b53f10561e5c115bb0b7f6cfa68
SHA_B_EVAL=5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189
PINS="dirB/code/pdc_util.py:f1ad5fe2992dfb59d086a239920c7a3dc4cb71dc9b199d2027f80f92f002d059 code/loto_common.py:901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11 data_ext/UMWP/data/StandardDataset.jsonl:e8840e8383357238a08e9c5028e4758ceceb369e1db31c64678b0f851c9c9e73 dirB/data/math500_train.jsonl:0862774dedd8fb0ccca3c9c2d17aa6c61e6a7259fe90a11fea7918f27cffa8f6 dirB/data/math_heldout1000.jsonl:d4bde3127b4ef11cd3f6ec6ed1650a2b0ebe9296ccc9c3246e43ee4bc1c1d890 dirB/data/gsm8k_test500.jsonl:58a521c14bb56a26fa4f0e3a4481eed137c1080af5eb3bc157c26a23a3bab31b"
R0SHA=(a4cd93eb55911a77b8adfd6fb5a0ac41d6b28718cc2e348807863fb285e0e64c 8b9a2f5723628edc226a2756b9369ce1ae0e831e284595c4c546b7c5bbcaa9b3 f3c672c53c1b33a2cb37a18e78b8f555113aff28995ea77eb6302135a8c9c27b ab04539c7d4f1d3b3423c2196d79362cb3acb85a4468974ea9d3e299d726338a 52bb2cf1326833742a4f6a17c65c048cc705ae98e64bd0793d3404b0944e34f4)
# start checkpoints (round-2 finals of own_a3): sha256 of shard 1, shard 2 and config.json
CKSHA=("4e9746b85a6afa3036deaf288b851b5a86d8089d154a36e174b471d29cb86add abd1f8b3d4f30dc5a11d949256bc5e44285baca559cbf00ebe619bd14e44fda3"
       "79f96c0e57045619369b25ee079c8ee2e00e0f04507e1e50e42fc7792c0299bb 2fdf751eb566ca8541f102d5ef764f32765dc4ec078a94506349e836ec6cf7c1"
       "b99d62e64ab2201c597202c21e1162626aa591899cfb94ba59033bd8944013be fee99b8a605fa819e87f23a22e99857a48c3721d3377f273546203ba6a225e1d"
       "35599ddb0fc6cb786536ee3f64c02628cfd4447dd1af84de5fca76f1a7191fa9 637590e8dd94e35437ab02b6f4e667c86c2763e34cbca7df3f06238995fc2a1a"
       "32f011dc44ae99a8363dbde27c212beebee5a2c31ba7cea3b8bff8d683a5f394 25f4016c1d84390dc3e7ecbddc3a43765ece6a20e7d73bb5742a5af3f2f8ab4d")
CKCFG=0a7cb83af085ee2147ad20ad7e302df9f843263b32c13d47462033f5b6ddc79e
CKSHARDS="model-00001-of-00002.safetensors model-00002-of-00002.safetensors"
PKGS="torch=2.8.0+cu128 transformers=4.57.1 vllm=0.11.0 bitsandbytes=0.48.1 flash_attn=2.8.1"
FILES="A13_registration_2026-09-27.md a13_build.py a13_launch.sh a13_read.py dev/a13_dev_test.sh dev/a13_dev_tree.py"
DEV=""
if [ -n "${A13_DEV_ROOT:-}" ]; then       # dev tests: every path inside the synthetic tree
  DEV=$A13_DEV_ROOT; . "$DEV/dev_constants.sh" || exit 9
  [ "$(cd "$R" && pwd -P)" = /home/tione/notebook/rsi_rl ] && { echo "dev mode must not point at the real repo"; exit 9; }
  case $SHM in /dev/shm|/dev/shm/) echo "dev mode must not use /dev/shm"; exit 9;; esac
  [ -n "${A13_DEV_HOST:-}" ] && H=$A13_DEV_HOST; PREF=${A13_DEV_PREF-$PREF}
fi
X=$R/ext/a13_runs; UM=$R/dirA4/results/umwp; EVX=$X/evals; CK=$SHM/ckpt_b; REG=$R/docs/registration_hashes.txt
GM=$SHM/gpu_single_; ME="a13 $$"; TAG="a13_${H}_$$"; PD=$SHM/a13_pids_$$; LOCK=$SHM/a13_launch.lock; PIDF=$SHM/a13_launch.pid
nm(){ echo qwen3_4b_Base_r0_a13_s$1_r3; }

L(){ echo "$(date '+%F %T %Z') $H a13[$$]: $*" >> $X/a13_launch.log; }
ST(){ printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$H" "$1" "$2" "${3:-}" >> $X/status_$H.tsv; }
sha(){ sha256sum "$1" | cut -c1-64; }
registered(){ grep -q "^$(sha "$1")  " $REG 2>/dev/null; }
stopping(){ [ -e $PD/STOPPING ] || [ -e $X/STOP ] || [ -e $X/STOP_$H ]; }
nap(){ local t=$1; while [ "$t" -gt 0 ] && ! stopping; do sleep $(( t < 5 ? t : 5 )); t=$((t - 5)); done; }
now(){ date +%s; }
past(){ [ -n "${T0:-}" ] && [ "$(now)" -ge $((T0 + $1)) ]; }
trim_partial(){ $PY -c '
import os, sys
f = sys.argv[1]
if os.path.exists(f) and os.path.getsize(f):
    b = open(f, "rb").read()
    if not b.endswith(b"\n"):
        k = b.rfind(b"\n") + 1
        open(f, "r+b").truncate(k)
        print("cut a partial last line of %d bytes from %s" % (len(b) - k, f))' "$1"; }
n_ids(){ $PY -c '
import json, os, sys
ids = set()
if os.path.exists(sys.argv[1]):
    for l in open(sys.argv[1], errors="replace"):
        try:
            ids.add(json.loads(l)["id"])
        except (ValueError, KeyError, TypeError):
            pass
print(len(ids))' "$1"; }
bench_done(){ $PY -c 'import json,os,sys; f=sys.argv[1]; s=json.load(open(f)) if os.path.exists(f) else {}
sys.exit(0 if all(n in s and "greedy_acc_lenient" in s[n] for n in ("math500_train","math_heldout1000","gsm8k_test500")) else 1)' "$1"; }

# ---- models: config, every weight shard, tokenizer, 'Base' in the path (probes and b_train pick the base-model prompt by it)
ck_ok(){ [ -f "$1/config.json" ] || return 1; case "$1" in *Base*) ;; *) return 1;; esac
  $PY - "$1" <<'EOF'
import glob, json, os, sys
m = sys.argv[1]; ix = os.path.join(m, "model.safetensors.index.json")
need = set(json.load(open(ix))["weight_map"].values()) if os.path.exists(ix) else {os.path.basename(f) for f in glob.glob(os.path.join(m, "*.safetensors"))}
ok = bool(need) and all(os.path.isfile(os.path.join(m, f)) and os.path.getsize(os.path.join(m, f)) > 0 for f in need)
sys.exit(0 if ok and any(os.path.isfile(os.path.join(m, t)) for t in ("tokenizer.json", "tokenizer_config.json")) else 1)
EOF
}
start_ck(){ echo $CK/qwen3_4b_Base_own_a3_s$1_r2/final; }    # the path string R and S were trained from (config "model")
start_ck_verify(){ local S=$1 d f i=0 want; d=$(start_ck $S)
  [ "$(sha $d/config.json)" = $CKCFG ] || return 1
  for f in $CKSHARDS; do want=$(echo ${CKSHA[$S]} | cut -d' ' -f$((i + 1))); [ "$(sha $d/$f)" = "$want" ] || return 1; i=$((i + 1)); done; }
# prints the model directory of a cell on this host, or returns 1
locate(){ local k=$1 i=$2 n d
  case $k in
    train) d=$(start_ck $i); ck_ok $d && { echo $d; return 0; };;
    plain|absfix|beval) n=$(nm $i); [ "$(cat $X/runs/$n/.trained 2>/dev/null)" = 60 ] || return 1
       for d in $CK/$n/final $ARCH/$n/final; do ck_ok $d && { echo $d; return 0; }; done;;
    oabs) n=qwen3_4b_Base_${i%_*}_a6_s${i##*_}_r3
       for d in $CK/$n/final $ARCH/$n/final; do ck_ok $d && { echo $d; return 0; }; done;;
  esac; return 1; }
# probe output, completion size, script, tag
probe_of(){ local k=$1 i=$2
  case $k in
    plain)  echo "$UM plain_$(nm $i) $R/dirA3/code/umwp_probe.py $SHA_UMWP";;
    absfix) echo "$UM absfix_$(nm $i) $R/ext/a11_draft/umwp_probe_absfix.py $SHA_ABSFIX";;
    oabs)   echo "$UM absfix_qwen3_4b_Base_${i%_*}_a6_s${i##*_}_r3 $R/ext/a11_draft/umwp_probe_absfix.py $SHA_ABSFIX";;
  esac; }
sidecar_path(){ local k=$1 i=$2 o t p s
  case $k in
    train) echo $X/runs/$(nm $i)/a13_train.json;;
    beval) echo $EVX/$(nm $i)_step60.a13.json;;
    *) read o t p s <<< "$(probe_of $k $i)"; echo $o/$t.a13.json;;
  esac; }
sidecar_field(){ $PY -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
except Exception: print("")' "$1" "$2"; }
# sidecar <kind> <id> <state> <model> <gpu> <script_sha> [key=value ...]: history appended, top-level fields updated
# (script_sha: sha256 of the script computed at that attempt)
sidecar(){ local f; f=$(sidecar_path $1 $2)
  $PY - "$f" "$H" "$@" <<'EOF'
import json, os, sys, time
f, host, kind, cid, state, model, gpu, ssha = sys.argv[1:9]
kv = dict(a.split("=", 1) for a in sys.argv[9:])
for k, v in list(kv.items()):
    if v in ("true", "false"): kv[k] = (v == "true")
    elif v.lstrip("-").isdigit(): kv[k] = int(v)
    elif k in ("fingerprint", "resumes"): kv[k] = json.loads(v)
m = {}
if os.path.exists(f):
    try: m = json.load(open(f))
    except ValueError: m = {"note": "previous sidecar unreadable"}
ev = {"time": time.strftime("%F %T %Z"), "host": host, "state": state, "gpu": gpu, "model": model, "script_sha256": ssha, **kv}
m.setdefault("history", []).append(ev)
m.update({"registration": "A13", "cell": "%s:%s" % (kind, cid), "state": state, "host": host, "model": model,
          "script_sha256": ssha, **kv})
os.makedirs(os.path.dirname(f), exist_ok=True)
tmp = f + ".tmp.%d" % os.getpid(); json.dump(m, open(tmp, "w"), indent=1); os.replace(tmp, f)
EOF
}
fingerprint(){ $PY - "$1" <<'EOF'
import hashlib, json, os, sys
m = os.path.realpath(sys.argv[1]); out = {"realpath": m, "files": {}}
for n in sorted(os.listdir(m)):
    p = os.path.join(m, n)
    if os.path.isfile(p):
        with open(p, "rb") as fh:
            out["files"][n] = [os.path.getsize(p), hashlib.sha256(fh.read(64 << 20)).hexdigest()[:16]]
print(json.dumps(out, separators=(",", ":")))
EOF
}
# the resumes b_train logged (step resumed from, optimizer state restored or not), as compact JSON
resumes_of(){ $PY -c 'import json, sys
r = []
for l in open(sys.argv[1], errors="replace"):
    try:
        x = json.loads(l)
    except ValueError:
        continue
    if isinstance(x, dict) and "resumed_from_step" in x:
        r.append({"from_step": x["resumed_from_step"], "optimizer_restored": x.get("optimizer_restored")})
print(json.dumps(r, separators=(",", ":")))' "$1" 2>/dev/null || echo '[]'; }
# raw completion of a cell's output (the sidecar says 'complete' only after the launcher saw this)
out_done(){ local k=$1 i=$2 o t p s
  case $k in
    train) [ "$(cat $X/runs/$(nm $i)/.trained 2>/dev/null)" = 60 ];;
    beval) bench_done $EVX/$(nm $i)_step60.summary.json;;
    *) read o t p s <<< "$(probe_of $k $i)"; [ "$(n_ids $o/$t.jsonl)" -ge 1200 ];;
  esac; }
done_cell(){ [ "$(sidecar_field $(sidecar_path $1 $2) state)" = complete ] && out_done $1 $2; }

# ---- claims (NFS mkdir); owner line: host, launcher PID, epoch
cell_claim(){ local d=$X/claims/$1__$2 o
  if mkdir $d 2>/dev/null; then echo "$H $$ $(now)" > $d/owner; return 0; fi
  o=$(cat $d/owner 2>/dev/null)
  if [ "${o%% *}" = "$H" ] && ! kill -0 "$(echo $o | cut -d' ' -f2)" 2>/dev/null; then
    echo "$H $$ $(now)" > $d/owner; L "took over a stale claim of this host for $1:$2 ($o)"; return 0; fi
  return 1; }
claimed_other(){ local d=$X/claims/$1__$2 o; [ -d $d ] || return 1; o=$(cat $d/owner 2>/dev/null)
  [ "$(echo $o | cut -d' ' -f1-2)" = "$H $$" ] && return 1
  [ "${o%% *}" = "$H" ] && ! kill -0 "$(echo $o | cut -d' ' -f2)" 2>/dev/null && return 1   # stale claim of this host
  return 0; }
cell_release(){ local d=$X/claims/$1__$2; [ "$(cut -d' ' -f1-2 $d/owner 2>/dev/null)" = "$H $$" ] && rm -rf $d; }

# ---- processes (A12): every job is its own session and process group and carries A13_TAG
desc(){ local p=$1 c; for c in $(ps -o pid= --ppid $p 2>/dev/null); do echo $c; desc $c; done; }
pstart(){ local s; s=$(cat /proc/$1/stat 2>/dev/null) || return 1; s=${s##*) }; set -- $s; echo ${20}; }
kill_job(){ local p=$1 s=$2 q t d=""
  for q in $(desc $p; ps -o pid= --sid $p 2>/dev/null); do t=$(pstart $q) && echo "$q $t"; done >> $PD/snap_$p
  while read -r q t; do [ "$(pstart $q)" = "$t" ] && d="$d $q"; done < <(sort -u $PD/snap_$p 2>/dev/null)
  kill -$s -- -$p 2>/dev/null; kill -$s $p $d 2>/dev/null; }
sweep(){ local e q
  for e in /proc/[0-9]*/environ; do q=${e#/proc/}; q=${q%/environ}
    [ "$q" != $$ ] && grep -qzx "A13_TAG=$TAG" $e 2>/dev/null && kill -$1 $q 2>/dev/null; done; }
gpu_uuid(){ $NVSMI -i $1 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' '; }
ours(){ local q=$1 n=0 pg
  pg=$(ps -o pgid= -p $q 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && [ -e $PD/job_$pg ] && return 0
  while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null && [ $n -lt 30 ]; do
    grep -qzx "A13_TAG=$TAG" /proc/$q/environ 2>/dev/null && return 0; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); n=$((n + 1)); done
  [ -d /proc/$1 ] && ! grep -q '^VmRSS:' /proc/$1/status 2>/dev/null; }
foreign_on(){ local g=$1 u pid uu; u=$(gpu_uuid $g)
  while IFS=', ' read -r pid uu; do [ "$uu" = "$u" ] && ! ours $pid && echo "pid $pid on gpu $g"; done \
    < <(timeout 60 $NVSMI --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+, '); }
# kill_foreign <gpu> <foreign_on output>: log pid, command line, GPU and time to killed_foreign.log, TERM, then KILL after 10 s
kill_foreign(){ local g=$1 f=$2 pid cmd
  for pid in $(awk '{print $2}' <<< "$f"); do
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-300)
    echo "$(date '+%F %T %Z') host $H gpu $g pid $pid cmd: ${cmd:-<not visible from this container>}" >> $X/killed_foreign.log
    L "PRIORITY: killing foreign pid $pid on gpu $g (${cmd:0:80})"; kill -TERM $pid 2>/dev/null; done
  sleep 10
  for pid in $(awk '{print $2}' <<< "$f"); do kill -0 $pid 2>/dev/null && kill -KILL $pid 2>/dev/null; done; return 0; }
mem(){ $NVSMI -i $1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
cooling(){ local t; t=$(cat $X/cooldown/${H}_gpu$1 2>/dev/null) || return 1; [ $(( $(now) - t )) -lt $COOLDOWN ] 2>/dev/null; }
free_now(){ local m    # a foreign compute process or >= 1500 MiB without a marker also starts a cooldown of that GPU
  [ -e ${GM}$1 ] && return 1; cooling $1 && return 1
  m=$(mem $1); case $m in ''|*[!0-9]*) return 1;; esac
  if [ "$m" -ge 1500 ] || [ -n "$(foreign_on $1)" ]; then now > $X/cooldown/${H}_gpu$1; return 1; fi
  return 0; }
take(){ ( set -o noclobber; : > ${GM}$1 ) 2>/dev/null && echo "$ME $(now)" > ${GM}$1; }
release_gpu(){ [ -n "$1" ] && [ "$(cut -d' ' -f1-2 ${GM}$1 2>/dev/null)" = "$ME" ] && rm -f ${GM}$1; }
prio(){ case $1 in train|plain) echo 0;; absfix|beval) echo 1;; *) echo 2;; esac; }
n_waiting_below(){ local n=0 f k     # cells of this launcher of a priority level below $1 that wait for a GPU
  for f in $PD/waiting_*; do [ -e "$f" ] || continue; k=${f##*/waiting_}; k=${k%%_*}; [ "$(prio $k)" -lt $1 ] && n=$((n + 1)); done
  echo $n; }
n_free(){ local g n=0; for g in $(seq 0 7); do free_now $g && n=$((n + 1)); done; echo $n; }
n_held(){ local f n=0; for f in ${GM}[0-7]; do [ "$(cut -d' ' -f1-2 $f 2>/dev/null)" = "$ME" ] && n=$((n + 1)); done; echo $n; }
owns(){ [ -e $PD/own_${1}_$2 ]; }
# claim_gpu <deadline> <kind> <id>: prints a GPU index; returns 1 once the stop flag is raised or the deadline passed, 2 if the
# train cell is held or was finished by another host. A train cell that holds its cross-host claim gives it up after waiting
# PREF_DELAY s here without a GPU (the other host may then take the seed).
claim_gpu(){ local dl=$1 k=$2 i=$3 lvl g c t1
  lvl=$(prio $k); t1=$(now); : > $PD/waiting_${k}_$i
  while ! stopping && ! past $dl; do
    if [ $k = train ]; then
      if owns $k $i; then
        if [ $(( $(now) - t1 )) -ge $PREF_DELAY ]; then cell_release $k $i; rm -f $PD/own_${k}_$i
          L "train:$i: no GPU for $PREF_DELAY s; its cross-host claim is released"; fi
      elif claimed_other $k $i || out_done $k $i; then rm -f $PD/waiting_${k}_$i; return 2; fi
    fi
    c=$(for g in $(seq 0 7); do free_now $g && echo $g; done | shuf | tr '\n' ' ')
    if [ "$(wc -w <<< "$c")" -gt "$(n_waiting_below $lvl)" ]; then
      nap $IDLE_GAP; { stopping || past $dl; } && break
      exec 8> $PD/take.lock; flock 8
      c=$(for g in $c; do free_now $g && echo $g; done | tr '\n' ' ')
      if [ "$(wc -w <<< "$c")" -gt "$(n_waiting_below $lvl)" ] && [ "$(n_held)" -lt $MAXPAR ]; then
        for g in $c; do take $g && { rm -f $PD/waiting_${k}_$i; exec 8>&-; echo $g; return 0; }; done; fi
      exec 8>&-
    fi
    nap $SCAN
  done; rm -f $PD/waiting_${k}_$i; return 1; }
# one attempt: prints rc (0 ok, 75 yielded to a foreign process, 143 not started or stopped, other = failure)
attempt(){ local g=$1 tmo=$2 grace=$3 logf=$4; shift 4; local pp f rc x i
  stopping && { echo 143; return; }
  ( ulimit -c 0; exec setsid -w env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE A13_TAG=$TAG \
      CUDA_VISIBLE_DEVICES=$g HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false timeout -k $((STOP_GRACE + 30)) $tmo "$@" >> $logf 2>&1 < /dev/null 9>&- ) &
  pp=$!; echo $pp > $PD/job_$pp
  sleep 1; x=$(ps -o pgid= -p $pp 2>/dev/null | tr -d ' ')
  [ -n "$x" ] && [ "$x" != "$pp" ] && L "WARNING: job $pp is not its own process group ($x)"
  while kill -0 $pp 2>/dev/null; do
    sleep $MON & wait $!
    kill -0 $pp 2>/dev/null || break
    [ -e $PD/STOPPING ] && break
    f=$(foreign_on $g)
    if [ -n "$f" ]; then sleep $CONFIRM; f=$(foreign_on $g)
      if [ -n "$f" ] && kill -0 $pp 2>/dev/null && [ "$PRIORITY" = 1 ]; then kill_foreign $g "$f"; continue; fi
      if [ -n "$f" ] && kill -0 $pp 2>/dev/null; then L "YIELD on gpu $g: $f"; date +%s > $X/cooldown/${H}_gpu$g
        kill_job $pp TERM; for i in $(seq $grace); do kill -0 $pp 2>/dev/null || break; sleep 1; done; kill_job $pp KILL
        wait $pp 2>/dev/null; kill -KILL -- -$pp 2>/dev/null; rm -f $PD/job_$pp $PD/snap_$pp; echo 75; return; fi; fi
  done
  wait $pp; rc=$?; kill -KILL -- -$pp 2>/dev/null; rm -f $PD/job_$pp; echo $rc; }

# after a finished training: move step20/step40/rollout/resume to the host-local disk archive (copy, verify size, remove),
# as dirB's ckpt_janitor2.sh does for dirB runs; the final stays on the RAM disk for the evaluations, and a copy of it goes to
# the archive (file sizes verified, then renamed from final.tmp), so that a lost RAM disk does not lose the seed
files_sizes(){ ( cd "$1" 2>/dev/null && find . -type f -printf '%P %s\n' | sort ); }
tidy(){ local n=$1 sub
  for sub in step20 step40 rollout resume resume.tmp; do [ -e $CK/$n/$sub ] || continue; mkdir -p $ARCH/$n
    if cp -a $CK/$n/$sub $ARCH/$n/ && [ "$(du -s $ARCH/$n/$sub | cut -f1)" -ge "$(du -s $CK/$n/$sub | cut -f1)" ]; then
      rm -rf $CK/$n/$sub; L "tidy: $n/$sub moved to $ARCH"; fi; done
  if [ -d $CK/$n/final ] && [ ! -e $ARCH/$n/final ]; then mkdir -p $ARCH/$n; rm -rf $ARCH/$n/final.tmp
    if cp -a $CK/$n/final $ARCH/$n/final.tmp && [ -n "$(files_sizes $CK/$n/final)" ] \
       && [ "$(files_sizes $CK/$n/final)" = "$(files_sizes $ARCH/$n/final.tmp)" ] && mv $ARCH/$n/final.tmp $ARCH/$n/final; then
      L "tidy: $n/final copied to $ARCH (kept on the RAM disk)"
    else rm -rf $ARCH/$n/final.tmp; L "tidy: copy of $n/final to $ARCH failed"; fi; fi; }

run_cell(){ local k=$1 i=$2 M=$3 a=0 y=0 g rc st why="" tmo grace deadline logf ssha="" pin scr o t p s n extra="" okck lost=0 cg
  echo $BASHPID > $PD/running_${k}_$i
  case $k in
    train) n=$(nm $i); tmo=$TMO_TRAIN; grace=$TRAIN_GRACE; deadline=$T_TRAIN_LAST; logf=$X/runs/$n/train.log
           scr=$R/dirB/code/b_train.py; pin=$SHA_B_TRAIN
           if [ "$(sha $X/runs/$n/admitted.jsonl 2>/dev/null)" != "${R0SHA[$i]}" ]; then
             why="R0 set missing or not the registered one"; st=not_run
           elif start_ck_verify $i; then okck=true; else why="start checkpoint sha256 differs from the registered one"; st=not_run; fi
           extra="start_ckpt=$M start_ckpt_sha_ok=${okck:-false} set_sha256=${R0SHA[$i]}";;
    beval) n=$(nm $i); tmo=$TMO_BEVAL; grace=$KILL_GRACE; deadline=$T_EVAL_LAST; logf=$X/logs/beval_$n.log
           scr=$R/dirB/code/b_eval.py; pin=$SHA_B_EVAL; : > $PD/own_${k}_$i;;                # claimed by the main loop
    *) read o t p s <<< "$(probe_of $k $i)"; tmo=$TMO_UMWP; grace=$KILL_GRACE; deadline=$T_EVAL_LAST; logf=$o/$t.a13.log
       scr=$p; pin=$s; : > $PD/own_${k}_$i                                                        # claimed by the main loop
       if [ ! -f $o/$t.a13.json ] && [ "$(n_ids $o/$t.jsonl)" -gt 0 ]; then why="rows without an A13 sidecar"; st=unknown_rows; fi
       if pgrep -f -- "--tag $t " > /dev/null; then why="a probe for $t is alive on this host"; st=live_writer; fi;;
  esac
  if [ -z "$why" ]; then
    while [ $a -lt 3 ] && ! out_done $k $i && ! stopping && ! past $deadline; do
      case $k in train|beval) ;; *) trim_partial $o/$t.jsonl >> $X/a13_launch.log;; esac
      out_done $k $i && break
      g=$(claim_gpu $deadline $k $i); cg=$?
      [ $cg = 2 ] && { lost=1; break; }
      [ $cg = 0 ] || break
      if ! owns $k $i; then                        # a train cell takes its cross-host claim only together with a GPU
        if cell_claim $k $i; then : > $PD/own_${k}_$i; else release_gpu $g; lost=1; break; fi
      fi
      if stopping || past $deadline || out_done $k $i; then release_gpu $g; break; fi
      ssha=$(sha $scr)
      if [ "$ssha" != "$pin" ]; then release_gpu $g; why="$scr is not the pinned version (sha256 $ssha)"; st=script_changed; break; fi
      if [ $k = train ] && [ "$(sha $X/runs/$n/admitted.jsonl 2>/dev/null)" != "${R0SHA[$i]}" ]; then
        release_gpu $g; why="R0 set changed after the start of the cell"; st=not_run; break; fi
      sidecar $k $i running "$M" $g $ssha $extra attempt=$((a + 1))
      L "start $k:$i attempt $((a + 1)) on gpu $g ($M)"; ST $k:$i start "gpu $g"
      case $k in
        train) rc=$(attempt $g $tmo $grace $logf python3 $scr --model $M --arm own --seed $i --out $X/runs/$n \
                     --data $X/runs/$n/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1);;
        beval) rc=$(attempt $g $tmo $grace $logf python3 $scr --model $M --tag ${n}_step60 --out $EVX);;
        plain) rc=$(attempt $g $tmo $grace $logf python3 $scr --model $M --tag $t --abstain_prompt 0);;
        *)     rc=$(attempt $g $tmo $grace $logf $PY $scr --model $M --tag $t --abstain_prompt 1);;
      esac
      release_gpu $g
      L "end $k:$i on gpu $g rc=$rc"
      if [ "$rc" = 75 ]; then y=$((y + 1)); ST $k:$i yielded "$y"; sidecar $k $i yielded "$M" $g $ssha $extra n_yields=$y
        [ $y -ge 12 ] && break; nap $YIELD_WAIT
      elif stopping; then break
      elif ! out_done $k $i; then a=$((a + 1)); fi
    done
  fi
  # a training finished on this host by an earlier launcher that died before marking it complete
  if [ $k = train ] && [ -z "$why" ] && [ $lost = 0 ] && ! owns $k $i && out_done $k $i \
     && [ "$(sidecar_field $(sidecar_path $k $i) host)" = "$H" ] && cell_claim $k $i; then : > $PD/own_${k}_$i; fi
  if [ $lost = 1 ] || { ! owns $k $i && [ -z "$why" ] && out_done $k $i; }; then   # train cell held or finished by the other host
    L "train:$i is held or was finished by another host; this host leaves it"; ST $k:$i left_to_other_host ""
    rm -f $PD/waiting_${k}_$i $PD/running_${k}_$i; return 0; fi
  [ -n "$ssha" ] || ssha=$(sha $scr)
  if [ -n "$why" ]; then ST $k:$i ${st:-not_run} "$why"; L "NOT RUN $k:$i: $why"
    [ $k = train ] && owns $k $i && sidecar $k $i ${st:-not_run} "$M" - $ssha $extra reason="$why"
  elif ! owns $k $i; then                          # a train cell that never held its claim here (no GPU before stop/deadline)
    if stopping; then st=stopped; elif past $deadline; then st=not_run_deadline; else st=incomplete; fi
    ST $k:$i $st "never claimed on this host (no free GPU)"; L "${st^^} $k:$i (never claimed on this host)"
  elif out_done $k $i; then
    case $k in
      train) extra="$extra n_resumes=$(grep -c resumed_from_step $X/runs/$n/log.jsonl 2>/dev/null) resumes=$(resumes_of $X/runs/$n/log.jsonl) config_sha256=$(sha $X/runs/$n/config.json)";;
      beval) extra="fingerprint=$(fingerprint "$M") output_sha256=$(sha $EVX/${n}_step60.summary.json)";;
      *) extra="fingerprint=$(fingerprint "$M") output_sha256=$(sha $o/$t.jsonl)";;
    esac
    set -f; sidecar $k $i complete "$M" - $ssha $extra n_attempts=$a n_yields=$y; set +f
    ST $k:$i complete "$M"; L "COMPLETE $k:$i"
    [ $k = train ] && tidy $n
  else
    if stopping; then st=stopped; elif past $deadline; then st=not_run_deadline; else st=incomplete; fi
    sidecar $k $i $st "$M" - $ssha $extra n_attempts=$a n_yields=$y; ST $k:$i $st "attempts $a yields $y"; L "${st^^} $k:$i"
  fi
  cell_release $k $i; rm -f $PD/own_${k}_$i $PD/waiting_${k}_$i; : > $PD/ended_${k}_$i; rm -f $PD/running_${k}_$i; }

# ---- scheduling: state of a cell on this host = done | start | wait | never
preferred(){ local s; for s in $PREF; do [ "$s" = "$1" ] && return 0; done; return 1; }
cell_state(){ local k=$1 i=$2
  [ -e $PD/ended_${k}_$i ] && { echo never; return; }
  [ -e $PD/running_${k}_$i ] && { echo wait; return; }
  done_cell $k $i && { echo done; return; }
  case $k in
    train) past $T_TRAIN_LAST && { echo never; return; }
           locate train $i > /dev/null || { echo never; return; }
           claimed_other train $i && { echo wait; return; }        # may be released by the other host after a failure
           if ! preferred $i && [ $(( $(now) - T0 )) -lt $PREF_DELAY ]; then echo wait; return; fi
           echo start;;
    plain|absfix|beval) past $T_EVAL_LAST && { echo never; return; }
           if locate $k $i > /dev/null; then claimed_other $k $i && { echo never; return; }; echo start; return; fi
           case $(cell_state train $i) in start|wait) echo wait;; *) echo never;; esac;;
    oabs)  past $T_EVAL_LAST && { echo never; return; }
           locate oabs $i > /dev/null || { echo never; return; }
           claimed_other oabs $i && { echo never; return; }
           echo start;;
  esac; }
CELLS=""; for s in $PREF; do CELLS="$CELLS train:$s"; done
for s in 0 1 2 3 4; do preferred $s || CELLS="$CELLS train:$s"; done
for k in plain absfix beval; do for s in 0 1 2 3 4; do CELLS="$CELLS $k:$s"; done; done
for s in 0 1 2 3 4; do CELLS="$CELLS oabs:cln1_$s oabs:rnd1_$s"; done

reap_dead(){ local e q t p pg f o
  for e in /proc/[0-9]*/environ; do q=${e#/proc/}; q=${q%/environ}
    t=$( { tr '\0' '\n' < $e; } 2>/dev/null | grep -m1 '^A13_TAG=a13_') || continue
    t=${t#A13_TAG=}; p=${t##*_}
    [ "$t" = "a13_${H}_$p" ] && [ "$p" != $$ ] && ! kill -0 "$p" 2>/dev/null || continue
    pg=$(ps -o pgid= -p $q 2>/dev/null | tr -d ' ')
    L "killing orphan job $q (group ${pg:-?}) of dead A13 launcher $p"; [ -n "$pg" ] && kill -KILL -- -$pg 2>/dev/null; kill -KILL $q 2>/dev/null
  done
  for f in ${GM}[0-7]; do [ -e "$f" ] || continue
    o=$(cat $f 2>/dev/null); set -- $o
    [ "${1:-}" = a13 ] && [ -n "${2:-}" ] && [ "$2" != $$ ] && ! kill -0 "$2" 2>/dev/null && { rm -f $f; L "released marker $f ($o) of dead A13 launcher $2"; }
  done
  for f in $SHM/a13_pids_*; do [ -d "$f" ] && [ "${f##*_}" != $$ ] && ! kill -0 "${f##*_}" 2>/dev/null && rm -rf "$f"; done; return 0; }
cleanup(){ local g d; for g in $(seq 0 7); do release_gpu $g; done
  for d in $X/claims/*/; do [ "$(cut -d' ' -f1-2 ${d}owner 2>/dev/null)" = "$H $$" ] && rm -rf $d; done
  rm -rf $PD; [ "$(cat $PIDF 2>/dev/null)" = "$$" ] && rm -f $PIDF; }
on_signal(){ local f i
  trap '' INT TERM HUP
  : > $PD/STOPPING; L "signal: stop flag set; stopping running jobs (TERM, KILL after $STOP_GRACE s)"
  for f in $PD/job_*; do [ -e "$f" ] && kill_job "$(cat $f)" TERM; done
  for i in $(seq 1 $STOP_GRACE); do [ -z "$(ls $PD/job_* 2>/dev/null)" ] && [ -z "$(jobs -rp)" ] && break; sleep 1; done
  for f in $(ls $PD 2>/dev/null | sed -n 's/^\(job\|snap\)_\([0-9][0-9]*\)$/\2/p' | sort -u); do kill_job $f KILL; done
  sweep KILL
  wait; cleanup; L "stopped"; exit 143; }

preflight(){ local f bad=0 kv s d
  [ "$(sha $R/dirB/code/b_train.py)" = $SHA_B_TRAIN ] || { echo "b_train.py is not the pinned version"; bad=1; }
  [ "$(sha $R/dirA3/code/umwp_probe.py)" = $SHA_UMWP ] || { echo "umwp_probe.py is not the pinned version"; bad=1; }
  [ "$(sha $R/ext/a11_draft/umwp_probe_absfix.py)" = $SHA_ABSFIX ] || { echo "umwp_probe_absfix.py is not the pinned version"; bad=1; }
  [ "$(sha $R/dirB/code/b_eval.py)" = $SHA_B_EVAL ] || { echo "b_eval.py is not the pinned version"; bad=1; }
  for kv in $PINS; do [ "$(sha $R/${kv%%:*})" = "${kv#*:}" ] || { echo "${kv%%:*} is not the pinned version"; bad=1; }; done
  for s in 0 1 2 3 4; do d=$X/runs/$(nm $s)
    [ "$(sha $d/admitted.jsonl 2>/dev/null)" = "${R0SHA[$s]}" ] || { echo "R0 set of seed $s missing or not registered (run: bash a13_launch.sh build)"; bad=1; }
    [ -e $d/.trained ] && echo "seed $s: already trained ($d/.trained)"; done
  if [ -z "$DEV" ]; then
    [ "$(command -v python3)" = /usr/bin/python3 ] || { echo "python3 is not /usr/bin/python3"; bad=1; }
    $PY - $PKGS <<'EOF' || bad=1
import importlib.metadata as m, sys
bad = [a for a in sys.argv[1:] if m.version(a.split("=")[0]) != a.split("=", 1)[1]]
print("packages as registered" if not bad else "package versions differ: %s" % bad); sys.exit(1 if bad else 0)
EOF
  fi
  for s in 0 1 2 3 4; do d=$(start_ck $s)
    if ck_ok $d; then
      if [ "$1" = full ]; then if start_ck_verify $s; then echo "start checkpoint s$s: here, sha256 as registered"; else echo "start checkpoint s$s: here, sha256 DIFFERS"; bad=1; fi
      else echo "start checkpoint s$s: here ($d)"; fi
    else echo "start checkpoint s$s: not on this host"; fi; done
  for f in $FILES; do if registered $C/$f; then echo "registered: $f"; else echo "NOT REGISTERED: $f ($(sha $C/$f 2>/dev/null))"; bad=1; fi; done
  [ "$1" = strict ] || [ "$1" = full ] || return 0
  [ $bad = 0 ] || { echo "refusing to run: fix the items above (register every A13 file in $REG first)"; return 1; }; }

case $MODE in
  build) mkdir -p $X/runs
         if [ -n "$DEV" ]; then $PY $C/a13_build.py --out_root $X/runs --root $R --dev; else $PY $C/a13_build.py --out_root $X/runs; fi
         exit $?;;
  locate) for c in $CELLS; do k=${c%%:*}; i=${c#*:}
            if M=$(locate $k $i); then printf '%-18s %s:%s\n' $c $H $M; else printf '%-18s %s:MISSING\n' $c $H; fi; done; exit 0;;
  check) preflight "${OPT:-plain}"; exit $?;;
  status) cat $X/status_*.tsv 2>/dev/null; tail -n 30 $X/a13_launch.log 2>/dev/null; exit 0;;
  stop) p=$(cat $PIDF 2>/dev/null)
        if [ -n "$p" ] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'a13_launch\.sh run'; then kill -TERM $p && echo "sent TERM to $p"
        else echo "no A13 launcher on this host"; fi; exit 0;;
  run) ;;
  *) echo "usage: a13_launch.sh build | locate | check [full] | run | stop | status"; exit 2;;
esac

mkdir -p $X/claims $X/logs $X/cooldown $EVX $PD
exec 9> $LOCK; flock -n 9 || { echo "another A13 launcher runs on this host"; rm -rf $PD; exit 1; }
echo $$ > $PIDF
trap on_signal INT TERM HUP
trap cleanup EXIT
preflight strict >> $X/a13_launch.log 2>&1 || { L "preflight failed (see above); nothing started"; exit 1; }
( set -o noclobber; now > $X/launch_epoch ) 2>/dev/null
T0=""; for k in $(seq 30); do T0=$(cat $X/launch_epoch 2>/dev/null); case $T0 in ''|*[!0-9]*) T0=""; sleep 1;; *) break;; esac; done
[ -n "$T0" ] || { L "cannot read $X/launch_epoch; nothing started"; exit 1; }
reap_dead
L "launch (MAXPAR $MAXPAR, preferred seeds '$PREF', T0 $(date -d @$T0 '+%F %T %Z')), launcher sha $(sha $C/a13_launch.sh)"
while true; do
  pending=0; nf=""                                  # nf: free GPUs, counted at most once per pass
  for c in $CELLS; do
    k=${c%%:*}; i=${c#*:}
    stopping && break
    case $(cell_state $k $i) in
      done) [ -e $PD/seen_done_${k}_$i ] || { ST $c skipped_complete; : > $PD/seen_done_${k}_$i; };;
      wait) pending=1;;
      start) pending=1; lv=$(prio $k); nr=$(ls $PD/running_* 2>/dev/null | wc -l)
             if [ $lv = 0 ]; then [ $nr -ge $((MAXPAR + 2)) ] && continue
             else
               [ $nr -ge $MAXPAR ] && continue
               [ $lv = 2 ] && [ "$(n_waiting_below 1)" -gt 0 ] && continue      # oabs: no train or plain cell waits for a GPU
               [ -n "$nf" ] || nf=$(n_free)
               [ "$nf" -gt "$(n_waiting_below $lv)" ] || continue             # and a GPU is free for it now
             fi
             M=$(locate $k $i) || continue
             if [ $k != train ]; then cell_claim $k $i || continue; fi          # a train cell claims once it holds a GPU
             : > $PD/running_${k}_$i
             if [ $lv = 0 ]; then : > $PD/waiting_${k}_$i; else nf=$((nf - 1)); fi
             run_cell $k $i "$M" &
             sleep 2;;
    esac
  done
  if stopping; then L "stop flag or STOP file: no new cells"; break; fi
  [ $pending = 0 ] && [ -z "$(ls $PD/running_* 2>/dev/null)" ] && break
  sleep $SCAN & wait $!
done
while [ -n "$(jobs -rp)" ]; do sleep 5 & wait $!; done
L "launcher done"
