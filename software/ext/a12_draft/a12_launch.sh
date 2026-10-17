#!/bin/bash
# A12 launcher, GPU parts (registration A12_registration_2026-09-26.md in this directory, sections 2.3, 3.3 and 4).
# Part 2: a12_plain_eval.py (registered b_eval.py, --k 1, stop also at <|im_end|>; tag a12_plain_<m>) and a12_rzchat_eval.py
# (R-Zero prompt, tag a12rz_<m>) on Qwen3-4B-Base and the R-Zero solvers S_1..S_3 of seeds 0 and 1 (host-local: seed 0 on the
# host with -p <PORT>, seed 1 on -p <PORT>). One cell = one model x one condition on ONE GPU. Part 3: a12_open_judge.py
# (Qwen2.5-32B-Instruct, TP 2) on a claimed PAIR of GPUs, on the host holding /dev/shm/models/Qwen2.5-32B-Instruct. One
# instance per host; the hosts share the NFS repo.
#   bash a12_launch.sh locate              read-only: where each cell's model lives on THIS host
#   bash a12_launch.sh check               read-only preflight (registration hashes, pinned inputs, rendered R-Zero prompt)
#   bash a12_launch.sh run [p2|p3|all]     run every cell whose model is on this host (default all)
#   bash a12_launch.sh stop                safe stop of this host's launcher (stop flag, kill its jobs, wait, cleanup)
# GPUs: a GPU is FREE only if it has no marker (/dev/shm/gpu_single_<g>), uses < 1500 MiB, carries no foreign compute process
# (the yield rule's test, foreign_on) and is not cooling down after a yield of this host; it must be free at two checks
# IDLE_GAP (60) s apart (idle > 1 minute). Free GPUs are taken in random order, atomically with
# ( set -o noclobber; : > /dev/shm/gpu_single_<g> ); one job per GPU (a pair for part 3); at most A12_MAXPAR (default 4) jobs
# per host; markers released after every attempt and by the exit handler; at start, orphan jobs and markers of a DEAD A12
# launcher of this host are removed. Cross-host de-duplication: an atomic mkdir claim per cell under ext/a12_runs/claims (NFS).
# Jobs: each starts with setsid (its own session and process group, so that reparented vLLM workers are killed with it) and
# carries A12_TAG. A job is killed (stop, yield) by first listing its descendants and its session's members, with their start
# times, then signalling its whole process group and every listed process; the KILL pass reuses the TERM pass's list (entries
# whose start time still matches) together with a fresh one, so a descendant that left the group (setpgid, or setsid while a
# descendant) is killed even after its parent has died. Resume: a finished set is never recomputed (b_eval and the wrapper
# resume per set; the judge writes its verdicts once, atomically); up to 3 attempts per cell per launch, each on a freshly
# claimed GPU, each capped by timeout; the stop flag is checked before every attempt.
# Co-tenant priority: a monitor checks the job's GPUs every 60 s, and a compute process there that is not ours (confirmed 30 s
# later) makes the job YIELD: it is killed, its GPUs are released and not claimed again by this host for COOLDOWN (600) s, and
# the cell waits YIELD_WAIT (600) s before claiming again (a yield is not a failed attempt; at most 12 yields per cell per
# launch). Yield by hand: touch ext/a12_runs/STOP (all hosts) or ext/a12_runs/STOP_<host>: no new job starts.
# Log: ext/a12_runs/a12_launch.log; status per host: ext/a12_runs/status_<host>.tsv.
set -u
MODE=${1:-locate}; PART=${2:-all}
R=/home/tione/notebook/rsi_rl; C=$(cd "$(dirname "$0")" && pwd); X=$R/ext/a12_runs; EV=$R/dirB/results/evals
REG=$R/docs/registration_hashes.txt; PY=/usr/bin/python3; H=$(hostname | cut -c1-22)
MAXPAR=${A12_MAXPAR:-4}; ME="a12 $$"; TAG="a12_${H}_$$"; PD=/dev/shm/a12_pids_$$
IDLE_GAP=60
COOLDOWN=600
YIELD_WAIT=600
MON=60
CONFIRM=30
KILL_GRACE=20
STOP_GRACE=120
SCAN=30
JM=/dev/shm/models/Qwen2.5-32B-Instruct; OPEN_OUT=$R/dirA5/results/a12_open/verdicts.jsonl
BEVAL=$R/dirB/code/b_eval.py; BEVAL_SHA=5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189
DEP_SHA="code/loto_common.py:901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11 dirB/code/pdc_util.py:f1ad5fe2992dfb59d086a239920c7a3dc4cb71dc9b199d2027f80f92f002d059"
DATA_SHA="math500_train:0862774dedd8fb0ccca3c9c2d17aa6c61e6a7259fe90a11fea7918f27cffa8f6 math_heldout1000:d4bde3127b4ef11cd3f6ec6ed1650a2b0ebe9296ccc9c3246e43ee4bc1c1d890 gsm8k_test500:58a521c14bb56a26fa4f0e3a4481eed137c1080af5eb3bc157c26a23a3bab31b"
FILES="A12_registration_2026-09-26.md a12_p1_sample.py a12_p1_blind_wf.js a12_blind_check.py a12_p1_read.py a12_p1_pilot_build.py a12_rzchat_eval.py a12_plain_eval.py a12_open_judge.py a12_launch.sh a12_p2_read.py a12_p3_read.py dev/dev_mk_check.py dev/dev_mk_judge.py dev/dev_a12_selftest.py dev/dev_a12_launch_test.sh dev/wf_stub_test.mjs"

# ---- cells (kind:model), the final solvers and the base first, then iterations 1-2 ("if cheap")
P2=""; for m in qwen3_4b_base rzero_s0_it3 rzero_s1_it3 rzero_s0_it1 rzero_s1_it1 rzero_s0_it2 rzero_s1_it2; do P2="$P2 plain:$m rzchat:$m"; done
P3="open:pool"
case $PART in p2) CELLS=$P2;; p3) CELLS=$P3;; all) CELLS="$P3 $P2";; *) echo "part must be p2, p3 or all"; exit 2;; esac

L(){ echo "$(date '+%F %T %Z') $H a12[$$]: $*" >> $X/a12_launch.log; }
ST(){ printf '%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$H" "$1" "$2" "${3:-}" >> $X/status_$H.tsv; }
sha(){ sha256sum "$1" | cut -c1-64; }
registered(){ grep -q "^$(sha "$1")  " $REG; }
stopping(){ [ -e $PD/STOPPING ] || [ -e $X/STOP ] || [ -e $X/STOP_$H ]; }
nap(){ local t=$1; while [ "$t" -gt 0 ] && ! stopping; do sleep $(( t < 5 ? t : 5 )); t=$((t - 5)); done; }   # sleep, cut short by the stop flag

# ---- models
ck_ok(){ [ -f "$1/config.json" ] || return 1
  $PY - "$1" <<'EOF'
import glob, json, os, sys
m = sys.argv[1]; ix = os.path.join(m, "model.safetensors.index.json")
need = set(json.load(open(ix))["weight_map"].values()) if os.path.exists(ix) else {os.path.basename(f) for f in glob.glob(os.path.join(m, "*.safetensors"))}
ok = bool(need) and all(os.path.isfile(os.path.join(m, f)) and os.path.getsize(os.path.join(m, f)) > 0 for f in need)
sys.exit(0 if ok and any(os.path.isfile(os.path.join(m, t)) for t in ("tokenizer.json", "tokenizer_config.json")) else 1)
EOF
}
locate(){ local m=$1 c="" S I F
  case $m in
    qwen3_4b_base) c=/root/models/Qwen3-4B-Base;;
    rzero_s*_it*) S=${m#rzero_s}; S=${S%%_*}; I=${m##*_it}; F=/run/ti/rzero/s$S/models/rzero_qwen3_4b_Base_s${S}_solver_v$I
                  [ -f $F/A75_DONE ] && c=$F/global_step_15/actor/huggingface;;
    pool) [ -f $JM.ok ] && c=$JM;;
  esac
  [ -n "$c" ] && ck_ok "$c" && { echo "$c"; return 0; }; return 1; }
# ---- completion (a cell is done when every set is in its summary, or the judge's verdicts exist)
done_cell(){ local k=$1 m=$2
  case $k in
    plain)  $PY -c 'import json,os,sys; f=sys.argv[1]; s=json.load(open(f)) if os.path.exists(f) else {}; sys.exit(0 if all(n in s and "greedy_acc_lenient" in s[n] for n in ("math500_train","math_heldout1000","gsm8k_test500")) else 1)' $EV/a12_plain_$m.summary.json;;
    rzchat) $PY -c 'import json,os,sys; f=sys.argv[1]; s=json.load(open(f)) if os.path.exists(f) else {}; sys.exit(0 if all(s.get(n, {}).get("complete") for n in ("math500_train","math_heldout1000","gsm8k_test500")) else 1)' $EV/a12rz_$m.summary.json;;
    open)   [ -f $OPEN_OUT ] && [ "$(wc -l < $OPEN_OUT)" = 1158 ];;
  esac; }

# ---- claims
cell_claim(){ local d=$X/claims/$1__$2 o
  if mkdir $d 2>/dev/null; then echo "$H $$ $(date +%s)" > $d/owner; return 0; fi
  o=$(cat $d/owner 2>/dev/null)
  if [ "${o%% *}" = "$H" ] && ! kill -0 "$(echo $o | cut -d' ' -f2)" 2>/dev/null; then
    echo "$H $$ $(date +%s)" > $d/owner; L "took over a stale claim of this host for $1:$2 ($o)"; return 0; fi
  return 1; }
cell_release(){ local d=$X/claims/$1__$2; [ "$(cut -d' ' -f1-2 $d/owner 2>/dev/null)" = "$H $$" ] && rm -rf $d; }
# ---- processes: every job is its own session and process group (setsid; both ids are the pid in $PD/job_<pid>) and carries
# A12_TAG
desc(){ local p=$1 c; for c in $(ps -o pid= --ppid $p 2>/dev/null); do echo $c; desc $c; done; }
pstart(){ local s; s=$(cat /proc/$1/stat 2>/dev/null) || return 1; s=${s##*) }; set -- $s; echo ${20}; }   # start time (field 22)
# kill_job <pid> <signal>: list the job's descendants and its session's members BEFORE signalling (a descendant that left the
# group is reached through parentage only while its parent lives), append them as "pid start-time" to $PD/snap_<pid>, then
# signal the group and every listed process whose start time still matches (so the KILL pass reuses the TERM pass's list)
kill_job(){ local p=$1 s=$2 q t d=""
  for q in $(desc $p; ps -o pid= --sid $p 2>/dev/null); do t=$(pstart $q) && echo "$q $t"; done >> $PD/snap_$p
  while read -r q t; do [ "$(pstart $q)" = "$t" ] && d="$d $q"; done < <(sort -u $PD/snap_$p 2>/dev/null)
  kill -$s -- -$p 2>/dev/null; kill -$s $p $d 2>/dev/null; }
sweep(){ local e q   # every process still carrying this launcher's tag (stop path only)
  for e in /proc/[0-9]*/environ; do q=${e#/proc/}; q=${q%/environ}
    [ "$q" != $$ ] && grep -qzx "A12_TAG=$TAG" $e 2>/dev/null && kill -$1 $q 2>/dev/null; done; }
gpu_uuid(){ nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' '; }
ours(){ local q=$1 n=0 pg
  pg=$(ps -o pgid= -p $q 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && [ -e $PD/job_$pg ] && return 0                   # a member of one of our jobs' process groups
  while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null && [ $n -lt 30 ]; do
    grep -qzx "A12_TAG=$TAG" /proc/$q/environ 2>/dev/null && return 0; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); n=$((n + 1)); done
  [ -d /proc/$1 ] && ! grep -q '^VmRSS:' /proc/$1/status 2>/dev/null; }   # an exiting process of this container
foreign_on(){ local g u pid uu
  for g in ${1//,/ }; do u=$(gpu_uuid $g)
    while IFS=', ' read -r pid uu; do [ "$uu" = "$u" ] && ! ours $pid && echo "pid $pid on gpu $g"; done \
      < <(timeout 60 nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+, ')
  done; }
# ---- GPUs
mem(){ nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
cooling(){ local t; t=$(cat $X/cooldown/${H}_gpu$1 2>/dev/null) || return 1; [ $(( $(date +%s) - t )) -lt $COOLDOWN ] 2>/dev/null; }
free_now(){ local m
  [ -e /dev/shm/gpu_single_$1 ] && return 1
  cooling $1 && return 1
  m=$(mem $1); [ "$m" -lt 1500 ] 2>/dev/null || return 1
  [ -z "$(foreign_on $1)" ]; }
take(){ ( set -o noclobber; : > /dev/shm/gpu_single_$1 ) 2>/dev/null && echo "$ME $(date +%s)" > /dev/shm/gpu_single_$1; }
release_gpu(){ [ "$(cut -d' ' -f1-2 /dev/shm/gpu_single_$1 2>/dev/null)" = "$ME" ] && rm -f /dev/shm/gpu_single_$1; }
# prints "g" (n=1) or "g,h" (n=2), or nothing if the stop flag was raised while waiting
claim_gpus(){ local n=$1 g h c ok
  while ! stopping; do
    c=$(for g in $(seq 0 7); do free_now $g && echo $g; done | shuf | tr '\n' ' ')
    if [ $(echo $c | wc -w) -ge $n ]; then
      nap $IDLE_GAP; stopping && return 1
      ok=$(for g in $c; do free_now $g && echo $g; done | tr '\n' ' ')
      if [ $(echo $ok | wc -w) -ge $n ]; then
        g=$(echo $ok | cut -d' ' -f1)
        if take $g; then
          [ $n = 1 ] && { echo $g; return 0; }
          for h in $(echo $ok | cut -d' ' -f2-); do take $h && { echo "$g,$h"; return 0; }; done
          release_gpu $g
        fi
      fi
    fi
    nap $SCAN
  done; return 1; }
# run one attempt: prints rc (0 ok, 75 yielded to a foreign process, 143 not started or stopped, other = failure)
attempt(){ local gpus=$1 tmo=$2; shift 2; local pp f rc x
  stopping && { echo 143; return; }
  setsid -w env -u PYTHONPATH A12_TAG=$TAG CUDA_VISIBLE_DEVICES=$gpus HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 timeout -k 60 $tmo "$@" >> $LOGF 2>&1 < /dev/null 9>&- &   # jobs do not hold the launcher lock
  pp=$!; echo $pp > $PD/job_$pp
  sleep 1; x=$(ps -o pgid= -p $pp 2>/dev/null | tr -d ' ')
  [ -n "$x" ] && [ "$x" != "$pp" ] && L "WARNING: job $pp is not its own process group ($x)"
  while kill -0 $pp 2>/dev/null; do
    sleep $MON & wait $!
    kill -0 $pp 2>/dev/null || break
    [ -e $PD/STOPPING ] && break
    f=$(foreign_on $gpus)
    if [ -n "$f" ]; then sleep $CONFIRM; f=$(foreign_on $gpus)
      if [ -n "$f" ] && kill -0 $pp 2>/dev/null; then L "YIELD on gpus $gpus: $f"
        for x in ${gpus//,/ }; do date +%s > $X/cooldown/${H}_gpu$x; done
        kill_job $pp TERM; sleep $KILL_GRACE; kill_job $pp KILL
        wait $pp 2>/dev/null; kill -KILL -- -$pp 2>/dev/null; rm -f $PD/job_$pp $PD/snap_$pp; echo 75; return; fi; fi
  done
  wait $pp; rc=$?; kill -KILL -- -$pp 2>/dev/null; rm -f $PD/job_$pp; echo $rc; }   # strays left in the job's group

run_cell(){ local k=$1 m=$2 M=$3 n=1 tmo cmd a=0 y=0 g rc x
  LOGF=$X/logs/${k}_$m.log
  case $k in
    plain)  tmo=7200; cmd="$PY $C/a12_plain_eval.py --model $M --tag a12_plain_$m --data math500_train,math_heldout1000,gsm8k_test500 --k 1";;
    rzchat) tmo=5400; cmd="$PY $C/a12_rzchat_eval.py --model $M --tag a12rz_$m";;
    open)   tmo=3600; n=2; cmd="$PY $C/a12_open_judge.py";;
  esac
  while [ $a -lt 3 ] && ! done_cell $k $m && ! stopping; do
    g=$(claim_gpus $n) || break
    if stopping; then for x in ${g//,/ }; do release_gpu $x; done; break; fi
    L "start $k:$m attempt $((a + 1)) on gpus $g ($M)"; ST $k:$m start "gpus $g"
    rc=$(attempt $g $tmo $cmd)
    for x in ${g//,/ }; do release_gpu $x; done
    L "end $k:$m on gpus $g rc=$rc"
    if [ "$rc" = 75 ]; then y=$((y + 1)); ST $k:$m yielded "$y"; [ $y -ge 12 ] && break; nap $YIELD_WAIT
    elif stopping; then break
    elif [ "$rc" != 0 ] || ! done_cell $k $m; then a=$((a + 1)); fi
  done
  if done_cell $k $m; then ST $k:$m complete "$M"; L "COMPLETE $k:$m"; else ST $k:$m incomplete "attempts $a yields $y"; L "INCOMPLETE $k:$m"; fi
  cell_release $k $m; }

# at start: kill orphan jobs of a DEAD A12 launcher of this host (their tag names its pid), then release its markers and
# remove its pid directory (its cell claims are taken over by cell_claim)
reap_dead(){ local e q t p pg f o
  for e in /proc/[0-9]*/environ; do q=${e#/proc/}; q=${q%/environ}
    t=$( { tr '\0' '\n' < $e; } 2>/dev/null | grep -m1 '^A12_TAG=a12_') || continue
    t=${t#A12_TAG=}; p=${t##*_}
    [ "$t" = "a12_${H}_$p" ] && [ "$p" != $$ ] && ! kill -0 "$p" 2>/dev/null || continue
    pg=$(ps -o pgid= -p $q 2>/dev/null | tr -d ' ')
    L "killing orphan job $q (group ${pg:-?}) of dead A12 launcher $p"; [ -n "$pg" ] && kill -KILL -- -$pg 2>/dev/null; kill -KILL $q 2>/dev/null
  done
  for f in /dev/shm/gpu_single_[0-7]; do [ -e "$f" ] || continue
    o=$(cat $f 2>/dev/null); set -- $o
    [ "${1:-}" = a12 ] && [ -n "${2:-}" ] && [ "$2" != $$ ] && ! kill -0 "$2" 2>/dev/null && { rm -f $f; L "released marker $f ($o) of dead A12 launcher $2"; }
  done
  for f in /dev/shm/a12_pids_*; do [ -d "$f" ] && [ "${f##*_}" != $$ ] && ! kill -0 "${f##*_}" 2>/dev/null && rm -rf "$f"; done; return 0; }
cleanup(){ local g d; for g in $(seq 0 7); do release_gpu $g; done
  for d in $X/claims/*/; do [ "$(cut -d' ' -f1-2 ${d}owner 2>/dev/null)" = "$H $$" ] && rm -rf $d; done
  rm -rf $PD; [ "$(cat /dev/shm/a12_launch.pid 2>/dev/null)" = "$$" ] && rm -f /dev/shm/a12_launch.pid; }
# safe signal handling: stop flag, kill our jobs (groups, sessions, listed descendants), wait, then cleanup (no second signal
# interrupts it); the KILL pass also covers jobs that ended meanwhile, through the lists ($PD/snap_<pid>) of the TERM pass
on_signal(){ local f i
  trap '' INT TERM HUP
  : > $PD/STOPPING; L "signal: stop flag set; stopping running jobs"
  for f in $PD/job_*; do [ -e "$f" ] && kill_job "$(cat $f)" TERM; done
  for i in $(seq 1 $STOP_GRACE); do [ -z "$(jobs -rp)" ] && break; sleep 1; done
  for f in $(ls $PD 2>/dev/null | sed -n 's/^\(job\|snap\)_\([0-9][0-9]*\)$/\2/p' | sort -u); do kill_job $f KILL; done
  sweep KILL
  wait; cleanup; L "stopped"; exit 143; }

preflight(){ local f bad=0 kv
  [ "$(sha $BEVAL)" = $BEVAL_SHA ] || { echo "b_eval.py is not the registered version"; bad=1; }
  for kv in $DEP_SHA; do [ "$(sha $R/${kv%%:*})" = "${kv#*:}" ] || { echo "${kv%%:*} is not the registered version"; bad=1; }; done
  for kv in $DATA_SHA; do [ "$(sha $R/dirB/data/${kv%%:*}.jsonl)" = "${kv#*:}" ] || { echo "data ${kv%%:*} changed"; bad=1; }; done
  for f in $FILES; do if registered $C/$f; then echo "registered: $f"; else echo "NOT REGISTERED: $f ($(sha $C/$f))"; bad=1; fi; done
  CUDA_VISIBLE_DEVICES= $PY - /root/models/Qwen3-4B-Base <<'EOF' || bad=1
import sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1])
p = t.apply_chat_template([{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                           {"role": "user", "content": "What is 1+1?"}], tokenize=False, add_generation_prompt=True)
print("R-Zero prompt as rendered:", repr(p)); print("eos", t.eos_token_id, "im_end", t.convert_tokens_to_ids("<|im_end|>"))
EOF
  [ "${1:-}" = strict ] && [ $bad = 1 ] && { echo "refusing to run: register every A12 file in $REG first"; return 1; }
  return 0; }

case $MODE in
  locate) for c in $P3 $P2; do m=${c#*:}; echo $m; done | awk '!s[$0]++' | while read -r m; do
            if M=$(locate $m); then printf '%-16s %s:%s\n' $m $H $M; else printf '%-16s %s:MISSING\n' $m $H; fi; done; exit 0;;
  check) preflight; exit $?;;
  stop) p=$(cat /dev/shm/a12_launch.pid 2>/dev/null)
        if [ -n "$p" ] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q 'a12_launch\.sh run'; then kill -TERM $p && echo "sent TERM to $p"
        else echo "no A12 launcher on this host"; fi; exit 0;;
  run) ;;
  *) echo "usage: a12_launch.sh locate | check | run [p2|p3|all] | stop"; exit 2;;
esac

mkdir -p $X/claims $X/logs $X/cooldown $PD
exec 9> /dev/shm/a12_launch.lock; flock -n 9 || { echo "another A12 launcher runs on this host"; rm -rf $PD; exit 1; }
echo $$ > /dev/shm/a12_launch.pid
trap on_signal INT TERM HUP
trap cleanup EXIT
preflight strict >> $X/a12_launch.log 2>&1 || { L "preflight failed (see above); nothing started"; exit 1; }
reap_dead
L "launch $PART (MAXPAR $MAXPAR), launcher sha $(sha $C/a12_launch.sh)"
for c in $CELLS; do
  k=${c%%:*}; m=${c#*:}
  stopping && { L "stop flag or STOP file: no new cells"; break; }
  if done_cell $k $m; then ST $c skipped_complete; continue; fi
  if ! M=$(locate $m); then ST $c missing_here; L "$c: model not on this host"; continue; fi
  cell_claim $k $m || { ST $c claimed_elsewhere "$(cat $X/claims/${k}__$m/owner 2>/dev/null)"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ] && ! stopping; do sleep 5 & wait $!; done
  stopping && { cell_release $k $m; break; }
  run_cell $k $m "$M" &
  sleep 5
done
while [ -n "$(jobs -rp)" ]; do sleep 5 & wait $!; done
L "launcher done ($PART)"
