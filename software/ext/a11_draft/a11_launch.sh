#!/bin/bash
# A11 launcher (registration A11_registration_2026-09-26.md in this directory): corrected abstention-prompt re-run on UMWP and
# on the A6 external sets (SUM, MiP-GSM8K, MiP-MATH). One instance per host; both hosts share the NFS repo, checkpoints are
# host-local. Only the abstention prompt is run (--abstain_prompt 1); plain-prompt dumps are unaffected by the defect.
#   bash a11_launch.sh locate      read-only: where each listed checkpoint lives on THIS host (writes nothing)
#   bash a11_launch.sh check       read-only: preflight (hashes, one-line diff, imports, rendered prompt, registration state)
#   bash a11_launch.sh run [all]   the registered run: every cell (Tier 1 first, then Tier 2; both required) whose checkpoint
#                                  is on this host. There is no partial-tier mode.
# A cell is one probe on one GPU: output absfix_<tag>.jsonl next to the old abs_<tag>.jsonl (the old file is never opened for
# writing; the probe refuses any tag without the absfix_ prefix), probe log absfix_<tag>.log, sidecar absfix_<tag>.a11.json
# (state, host, model path + fingerprint, probe sha256, GPU of every attempt; the read accepts a dump only with state
# "complete" and the registered probe sha). Cross-host de-duplication: an atomic mkdir claim per cell under
# ext/a11_runs/claims (NFS); the claim names the launcher PID and the PID of the cell's subshell, and a claim of this host is
# taken over only when both are gone. After a claim succeeds the cell is re-checked and skipped if it became complete.
# GPUs: claimed atomically with ( set -o noclobber; : > /dev/shm/gpu_single_<g> ) only while nvidia-smi shows < 1500 MiB used
# (checked twice, 15 s apart), one probe per GPU, at most A11_MAXPAR (default 4) probes per host, marker released after every
# attempt. Resume: a complete output (1,200 unique ids UMWP, 1,552 A6) is skipped; a partial one is resumed by the probe after
# its partial last line is cut; up to 3 attempts per cell per launch, each on a freshly claimed GPU (so a co-tenant that kills
# a probe gets its GPU back). Each attempt is capped at 30 min (timeout -k 60 1800; the hang watchdog does not match these
# probes). The probe runs with the host lock fd 9 closed (9>&-).
# A missing checkpoint is logged and the cell is NOT RUN on this host (status file); nothing waits for it.
# Clock: T0 = ext/a11_runs/launch_epoch, written once by the first launcher that passes the preflight. No GPU claim and no new
# attempt after T0 + 11 h; the confirmatory read runs at the latest at T0 + 12 h, and unfinished cells are NOT RUN.
# Yield: touch ext/a11_runs/STOP (all hosts) or ext/a11_runs/STOP_<host>: no new cell, no new GPU claim and no new attempt
# starts (checked before every claim and every attempt); running probes finish.
# Signal (kill -TERM <launcher pid>): the launcher sets its stop flag first, sends TERM to every running probe (timeout
# forwards it to the probe's process group and sends KILL 60 s later) and to every cell subshell, then waits (no fixed sleep)
# until every child has exited; each subshell waits for its own probe, releases its GPU, writes state "stopped" and releases
# its claim; only then does the launcher run its cleanup and exit 143. No attempt is started after the flag is set.
# Log: ext/a11_runs/a11_launch.log (new file, shared, one line per event with the host name); status_<host>.tsv per host.
set -u
MODE=${1:-locate}; ARG=${2:-all}
R=/home/tione/notebook/rsi_rl; C=$(cd "$(dirname "$0")" && pwd); X=$R/ext/a11_runs
UM=$R/dirA4/results/umwp; A6=$R/dirA5/results/a6_sweep; REG=$R/docs/registration_hashes.txt
H=$(hostname | cut -c1-22); PY=/usr/bin/python3; MAXPAR=${A11_MAXPAR:-4}; ME="a11_absfix $$"
GM=/dev/shm/gpu_single_; LOCK=/dev/shm/a11_launch.lock; PD=/dev/shm/a11_pids_$$
T_ATTEMPT=1800; T_NEWSTART=39600      # 30 min per attempt; no claim or attempt after T0 + 11 h
Q=qwen3_4b_Base

# ---- cells (bench:tag). Tier 1: the two R-Zero cells first (host-local checkpoints under /run/ti/rzero), then the
# confirmatory cells and the named descriptives. Tier 2: the remaining descriptive cells. Both tiers are required.
T1="umwp:rzero_s0_it3 umwp:rzero_s1_it3"
for s in 0 1 2 3; do T1="$T1 umwp:${Q}_own_a3_s${s}_r3 umwp:${Q}_own_clean_a3_s${s}_r3"; done
for s in 0 1 2; do T1="$T1 umwp:${Q}_own_ag_a4_s${s}_r3"; done
for s in 0 1 2 3; do T1="$T1 a6:${Q}_own_a3_s${s}_r3 a6:${Q}_own_clean_a3_s${s}_r3"; done
T1="$T1 umwp:qwen3_4b_base umwp:${Q}_own_a3_s4_r3"
T2=""; for s in 0 1 2 4; do T2="$T2 umwp:${Q}_rescue_uv_a4_s${s}_r3"; done
for s in 0 1 2; do T2="$T2 umwp:${Q}_rescue_abs_a4_s${s}_r3"; done
T2="$T2 umwp:${Q}_loop_uv_a4_s0_r4 umwp:${Q}_loop_uv_a4_s2_r4 umwp:${Q}_coev_ag_a4_s0_r3 umwp:${Q}_coev_ag_a4_s1_r3"
for s in 0 1 2; do T2="$T2 umwp:${Q}_tg_abs4_a5_s${s}_r4"; done
T2="$T2 umwp:${Q}_tg_t4_a5_s1_r4"
CELLS="$T1 $T2"

L(){ echo "$(date '+%F %T %Z') $H a11[$$]: $*" >> $X/a11_launch.log; }
ST(){ printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$H" "$1" "$2" "$3" "${4:-}" >> $X/status_$H.tsv; }
bench_out(){ if [ $1 = umwp ]; then echo "$UM 1200 $C/umwp_probe_absfix.py"; else echo "$A6 1552 $C/a6_probe_absfix.py"; fi; }
sha(){ sha256sum "$1" | cut -c1-64; }
registered(){ grep -q "^$(sha "$1")  " $REG; }
nap(){ sleep "$1" 9>&- & wait $! 2>/dev/null; }     # a sleep that a trapped signal interrupts at once

# ---- stop conditions: aborting = a signal reached this launcher (its stop flag); no_new_work = aborting, a STOP file, or
# the T0 + 11 h clock
aborting(){ [ -n "${ABORT:-}" ] || [ -e $PD/STOPPING ]; }
past_deadline(){ [ -n "${T0:-}" ] && [ "$(date +%s)" -ge $((T0 + T_NEWSTART)) ]; }
no_new_work(){ aborting || [ -e $X/STOP ] || [ -e $X/STOP_$H ] || past_deadline; }
why_stop(){ if aborting; then echo signal; elif [ -e $X/STOP ]; then echo "STOP file"; elif [ -e $X/STOP_$H ]; then echo "STOP_$H file"
  else echo "T0 + 11 h clock"; fi; }

# ---- dump files (verbatim copies of trim_partial / n_ids in ext/rzero_design/rz_common.sh)
trim_partial(){ /usr/bin/python3 -c '
import os, sys
f = sys.argv[1]
if os.path.exists(f) and os.path.getsize(f):
    b = open(f, "rb").read()
    if not b.endswith(b"\n"):
        k = b.rfind(b"\n") + 1
        open(f, "r+b").truncate(k)
        print("cut a partial last line of %d bytes from %s" % (len(b) - k, f))' "$1"; }
n_ids(){ /usr/bin/python3 -c '
import json, os, sys
ids = set()
if os.path.exists(sys.argv[1]):
    for l in open(sys.argv[1], errors="replace"):
        try:
            ids.add(json.loads(l)["id"])
        except (ValueError, KeyError, TypeError):
            pass
print(len(ids))' "$1"; }

# ---- checkpoints: config, every weight shard, tokenizer, and 'Base' in the path (the probe picks the base-model prompt
# from the path, as in every old dump)
ck_ok(){ [ -f "$1/config.json" ] || return 1; case "$1" in *Base*) ;; *) return 1;; esac
  $PY - "$1" <<'EOF'
import glob, json, os, sys
m = sys.argv[1]; ix = os.path.join(m, "model.safetensors.index.json")
need = set(json.load(open(ix))["weight_map"].values()) if os.path.exists(ix) else {os.path.basename(f) for f in glob.glob(os.path.join(m, "*.safetensors"))}
ok = bool(need) and all(os.path.isfile(os.path.join(m, f)) and os.path.getsize(os.path.join(m, f)) > 0 for f in need)
ok = ok and any(os.path.isfile(os.path.join(m, t)) for t in ("tokenizer.json", "tokenizer_config.json"))
sys.exit(0 if ok else 1)
EOF
}
# prints the model directory for a tag on this host, or nothing (return 1)
locate(){ local t=$1 d c="" S F
  case $t in
    qwen3_4b_base) c=/root/models/Qwen3-4B-Base;;
    rzero_s*_it3) S=${t#rzero_s}; S=${S%_it3}; F=/run/ti/rzero/s$S/models/rzero_qwen3_4b_Base_s${S}_solver_v3
                  [ -f $F/A75_DONE ] && c=$F/global_step_15/actor/huggingface;;
    *) for d in /dev/shm/ckpt_b/$t/final /run/ti/rsi_ckpt_archive/$t/final; do ck_ok $d && { echo $d; return 0; }; done
       # fallback: /run/ti/rsi_ckpt_archive/**/<run>/final (exact run name, symlinks followed), used only when it is unique
       c=$(find -L /run/ti/rsi_ckpt_archive -maxdepth 4 -type d -name final -path "*/${t}/final" 2>/dev/null | sort)
       [ "$(echo "$c" | grep -c .)" = 1 ] || c="";;
  esac
  [ -n "$c" ] && ck_ok "$c" && { echo "$c"; return 0; }; return 1; }
fingerprint(){ $PY - "$1" <<'EOF'
import hashlib, json, os, sys
m = os.path.realpath(sys.argv[1]); out = {"path": sys.argv[1], "realpath": m, "files": {}}
for n in sorted(os.listdir(m)):
    p = os.path.join(m, n)
    if os.path.isfile(p):
        with open(p, "rb") as fh:
            out["files"][n] = [os.path.getsize(p), hashlib.sha256(fh.read(64 << 20)).hexdigest()[:16]]  # first 64 MiB
print(json.dumps(out, separators=(",", ":")))
EOF
}
# sidecar absfix_<tag>.a11.json: bench tag state model gpu n_ids probe_sha fingerprint_json
sidecar(){ local o; o=$(bench_out $1 | cut -d' ' -f1)
  $PY - "$o/absfix_$2.a11.json" "$@" "$H" <<'EOF'
import json, os, sys, time
f, bench, tag, state, model, gpu, n, psha, fp, host = sys.argv[1:11]
m = {}
if os.path.exists(f):
    try: m = json.load(open(f))
    except ValueError: m = {"note": "previous sidecar unreadable"}
ev = {"time": time.strftime("%F %T %Z"), "host": host, "state": state, "gpu": gpu, "n_ids": int(n), "model": model, "probe_sha256": psha}
if fp: ev["model_fingerprint"] = json.loads(fp)
m.setdefault("history", []).append(ev)
m.update({"registration": "A11", "bench": bench, "tag": "absfix_" + tag, "state": state, "host": host, "model": model,
          "probe_sha256": psha, "n_ids": int(n)})
tmp = f + ".tmp"; json.dump(m, open(tmp, "w"), indent=1); os.replace(tmp, f)
EOF
}
sidecar_field(){ $PY -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
except Exception: print("")' "$1" "$2"; }

# ---- claims. Owner line: host, launcher PID, epoch[, PID of the cell subshell]
CLAIMING=""
live_writer(){ pgrep -f -- "--tag absfix_$1 --abstain_prompt" > /dev/null; }   # a probe for this tag runs on this host
cell_claim(){ local d=$X/claims/$1__$2 o
  if mkdir $d 2>/dev/null; then CLAIMING=$d; echo "$H $$ $(date +%s)" > $d/owner; CLAIMING=""; return 0; fi
  o=$(cat $d/owner 2>/dev/null)
  if live_writer $2; then L "claim of $1:$2 not taken over: a probe for absfix_$2 is still alive on this host"; return 1; fi
  if [ "${o%% *}" = "$H" ] && ! kill -0 "$(echo $o | cut -d' ' -f2)" 2>/dev/null \
     && { [ -z "$(echo $o | cut -d' ' -f4)" ] || ! kill -0 "$(echo $o | cut -d' ' -f4)" 2>/dev/null; }; then
    echo "$H $$ $(date +%s)" > $d/owner; L "took over a stale claim of this host for $1:$2 ($o)"; return 0; fi
  return 1; }
cell_release(){ local d=$X/claims/$1__$2; [ "$(cut -d' ' -f1-2 $d/owner 2>/dev/null)" = "$H $$" ] && rm -rf $d; }
# claims a free GPU into G and returns 0, or returns 1 as soon as no new work may start
G=""
claim_gpu(){ local g m; G=""
  while true; do
    for g in $(seq 0 7); do
      no_new_work && return 1
      [ -e ${GM}$g ] && continue
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [ "$m" -lt 1500 ] 2>/dev/null || continue
      nap 15; no_new_work && return 1
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      if [ "$m" -lt 1500 ] 2>/dev/null && ( set -o noclobber; : > ${GM}$g ) 2>/dev/null; then
        G=$g; echo "$ME $(date +%s) $BASHPID" > ${GM}$g; return 0; fi
    done
    nap 30
  done; }
release_gpu(){ [ -n "$1" ] && [ "$(cut -d' ' -f1-2 ${GM}$1 2>/dev/null)" = "$ME" ] && rm -f ${GM}$1; }
cleanup(){ local g d; for g in $(seq 0 7); do release_gpu $g; done
  [ -n "$CLAIMING" ] && [ -d "$CLAIMING" ] && [ ! -s "$CLAIMING/owner" ] && rm -rf "$CLAIMING"
  for d in $X/claims/*/; do [ "$(cut -d' ' -f1-2 ${d}owner 2>/dev/null)" = "$H $$" ] && rm -rf $d; done; rm -rf $PD; }

# ---- preflight (no GPU): originals, data, imported helpers, one-line fix, rendered prompt; with 'strict' every A11 file must
# be registered
preflight(){ local f bad=0
  $PY - "$R" "$C" <<'EOF' || return 1
import ast, hashlib, os, sys
sys.dont_write_bytecode = True
R, C = sys.argv[1:3]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
REGISTERED_INPUTS = {"dirA3/code/umwp_probe.py": "32de59f95a296fa0c026b44cf8a6110172898b87b258179ab3a54da00ee1e56d",
                     "dirA5/code/a6_probe.py": "af94764de247aa2c762b8b7bad67fe835a89fab4e814e3547f4e9b60741ec9f9",
                     "dirB/code/pdc_util.py": "f1ad5fe2992dfb59d086a239920c7a3dc4cb71dc9b199d2027f80f92f002d059",
                     "code/loto_common.py": "901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11",
                     "data_ext/UMWP/data/StandardDataset.jsonl": "e8840e8383357238a08e9c5028e4758ceceb369e1db31c64678b0f851c9c9e73",
                     "data_ext/a6/sweep.jsonl": "eb78c0c5e4409119ea4a8b3b832dbb49e15414f7095957d1aa6535584fad951b"}
for p, s in REGISTERED_INPUTS.items():
    assert h(f"{R}/{p}") == s, f"{p} is not the pinned version"
BS = b"\\"
old = b'SOLVE_ABS = "Solve the problem step by step and output the final answer within ' + BS + b'boxed{}."\n'
new = b'SOLVE_ABS = r"Solve the problem step by step and output the final answer within ' + BS + b'boxed{}."\n'
anchor = b"args = ap.parse_args()\n"
guard = (b'assert args.tag.startswith("absfix_") and args.abstain_prompt == 1, '
         b'"A11: this probe writes only corrected abstention-prompt dumps, tagged absfix_<tag>"\n')
for fix, src in (("umwp_probe_absfix.py", "dirA3/code/umwp_probe.py"), ("a6_probe_absfix.py", "dirA5/code/a6_probe.py")):
    b = open(f"{R}/{src}", "rb").read()
    assert b.count(old) == 1 and b.count(anchor) == 1
    assert open(f"{C}/{fix}", "rb").read() == b.replace(old, new, 1).replace(anchor, anchor + guard, 1), f"{fix} is not the original plus the registered fix"
    v = [n.value.value for n in ast.walk(ast.parse(open(f"{C}/{fix}").read())) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SOLVE_ABS"][0]
    assert "\x08" not in v and v.endswith("within \\boxed{}."), repr(v)
# the helpers resolve exactly as in the probe: script directory first, then code/ (loto_common), then dirB/code (pdc_util)
sys.path[0] = C
sys.path.insert(0, f"{R}/code"); import loto_common
sys.path.insert(0, f"{R}/dirB/code"); import pdc_util
from pdc_util import ABSTAIN_SUFFIX
assert os.path.realpath(loto_common.__file__) == f"{R}/code/loto_common.py", loto_common.__file__
assert os.path.realpath(pdc_util.__file__) == f"{R}/dirB/code/pdc_util.py", pdc_util.__file__
print("preflight ok; base-model abstention prompt:", repr(f"Question: <problem>{ABSTAIN_SUFFIX}\n\n{v}\nAnswer: Let's think step by step."))
EOF
  for f in $C/A11_registration_2026-09-26.md $C/umwp_probe_absfix.py $C/a6_probe_absfix.py $C/a11_launch.sh $C/a11_read.py; do
    if registered $f; then echo "registered: $f"; else echo "NOT REGISTERED: $f ($(sha $f))"; bad=1; fi
  done
  [ "${1:-}" = strict ] && [ $bad = 1 ] && { echo "refusing to run: hash every A11 file into $REG first"; return 1; }
  return 0; }

# ---- one cell, in a background subshell (its PID is recorded as $PD/cell_<pid>, in the claim owner line and in its GPU
# markers). TERM sets ABORT; the subshell then stops its probe, waits for it and ends cleanly.
run_cell(){ local b=$1 t=$2 M=$3 o N P f a rc n pp fp psha st="" why="" lastg=- d=$X/claims/$1__$2
  ABORT=""; trap 'ABORT=1' TERM
  echo "$b:$t" > $PD/cell_$BASHPID; echo "$H $$ $(date +%s) $BASHPID" > $d/owner
  read o N P <<< "$(bench_out $b)"; f=$o/absfix_$t.jsonl; psha=$(sha $P); fp=$(fingerprint "$M")
  if [ ! -f $o/absfix_$t.a11.json ] && [ "$(n_ids $f)" -gt 0 ]; then   # rows not written under this launcher: never resumed
    ST $b $t unknown_rows "$(n_ids $f) rows without a sidecar"; L "WARNING $b:$t has rows but no sidecar: not resumed"
    cell_release $b $t; rm -f $PD/cell_$BASHPID; return; fi
  if live_writer $t; then   # an orphaned probe (e.g. after kill -9 of every launcher process) still writes this dump
    ST $b $t live_writer "a probe for absfix_$t is alive"; L "WARNING $b:$t: a probe for absfix_$t is alive; not started"
    cell_release $b $t; rm -f $PD/cell_$BASHPID; return; fi
  for a in 1 2 3; do
    trim_partial $f >> $X/a11_launch.log; [ "$(n_ids $f)" -ge $N ] && break
    if no_new_work || ! claim_gpu; then st=stopped; why=$(why_stop); break; fi
    lastg=$G
    if aborting; then release_gpu $G; G=""; st=stopped; why=signal; break; fi
    sidecar $b $t running "$M" $G "$(n_ids $f)" "$psha" "$fp"
    L "start $b:$t attempt $a on gpu $G, model $M"
    env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=$G HF_HOME=$R/hf_home \
      TOKENIZERS_PARALLELISM=false timeout -k 60 $T_ATTEMPT $PY $P --model "$M" --tag absfix_$t --abstain_prompt 1 \
      >> $o/absfix_$t.log 2>&1 < /dev/null 9>&- &
    pp=$!; echo $G > $PD/probe_$pp
    aborting && kill -TERM $pp 2>/dev/null           # the signal came while the probe was being started
    while true; do wait $pp; rc=$?; kill -0 $pp 2>/dev/null || break; kill -TERM $pp 2>/dev/null; done
    rm -f $PD/probe_$pp; release_gpu $G; G=""
    L "end $b:$t attempt $a rc=$rc unique ids $(n_ids $f)"
    if aborting; then st=stopped; why=signal; break; fi
  done
  trim_partial $f >> $X/a11_launch.log; n=$(n_ids $f)
  if [ "$n" -ge $N ]; then sidecar $b $t complete "$M" $lastg "$n" "$psha" ""; ST $b $t complete "$M"; L "COMPLETE $b:$t ($n ids)"
  elif [ "$st" = stopped ]; then sidecar $b $t stopped "$M" $lastg "$n" "$psha" ""; ST $b $t stopped "$n ids; $why"; L "STOPPED $b:$t ($n ids; $why)"
  else sidecar $b $t incomplete "$M" $lastg "$n" "$psha" ""; ST $b $t incomplete "$n ids after 3 attempts"; L "INCOMPLETE $b:$t ($n ids)"; fi
  cell_release $b $t; rm -f $PD/cell_$BASHPID; }

case $MODE in
  locate)
    for c in $CELLS; do t=${c#*:}; echo "$t"; done | awk '!s[$0]++' | while read -r t; do
      if M=$(locate $t); then printf '%-44s %s:%s\n' "$t" "$H" "$M"; else printf '%-44s %s:MISSING\n' "$t" "$H"; fi
    done; exit 0;;
  check) preflight; exit $?;;
  run) [ "$ARG" = all ] || { echo "A11 registers only 'run' (= 'run all': Tier 1 and Tier 2)"; exit 2; };;
  *) echo "usage: a11_launch.sh locate | check | run [all]"; exit 2;;
esac

# ---- run
mkdir -p $X/claims
exec 9> $LOCK; flock -n 9 || { echo "another A11 launcher runs on this host"; exit 1; }
mkdir -p $PD
preflight strict >> $X/a11_launch.log 2>&1 || { L "preflight failed (see above); nothing started"; rm -rf $PD; exit 1; }
( set -o noclobber; date +%s > $X/launch_epoch ) 2>/dev/null   # T0, written once by the first launcher past the preflight
T0=""; for k in $(seq 30); do T0=$(cat $X/launch_epoch 2>/dev/null); case $T0 in ''|*[!0-9]*) T0=""; sleep 1;; *) break;; esac; done
[ -n "$T0" ] || { L "cannot read $X/launch_epoch; nothing started"; rm -rf $PD; exit 1; }
# a recorded PID is signalled only if it is still what was recorded: an A11 probe (timeout ... *_absfix.py --model ...) or a
# cell subshell of this launcher
is_probe(){ tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -q "_absfix.py --model"; }
is_child(){ [ "$(ps -o ppid= -p $1 2>/dev/null | tr -d ' ')" = "$$" ]; }
on_signal(){ local p k=0 left
  trap "" INT TERM; : > $PD/STOPPING
  L "signal: stop flag set; TERM to running probes and cell subshells, waiting until every child has exited"
  for p in $PD/probe_*; do [ -e "$p" ] || continue; p=${p##*_}; is_probe $p && kill -TERM $p 2>/dev/null; done
  for p in $PD/cell_*; do [ -e "$p" ] || continue; p=${p##*_}; is_child $p && kill -TERM $p 2>/dev/null; done
  for p in $(jobs -p); do is_child $p && kill -TERM $p 2>/dev/null; done
  wait
  while true; do   # only if a cell subshell died without waiting for its probe: KILL the probe's process group after 90 s
    left=""; for p in $PD/probe_*; do [ -e "$p" ] || continue; p=${p##*_}; is_probe $p && left="$left $p"; done
    [ -z "$left" ] && break
    [ $k = 90 ] && for p in $left; do kill -KILL -- -$p 2>/dev/null; done
    [ $k -ge 120 ] && { L "WARNING probe(s)$left still alive 120 s after the signal"; break; }
    sleep 1; k=$((k + 1))
  done
  cleanup; L "signal: every child has exited; GPU markers and cell claims released"; trap - EXIT; exit 143; }
trap on_signal TERM INT
trap cleanup EXIT
L "launch run all (MAXPAR $MAXPAR, T0 $(date -d @$T0 '+%F %T %Z')), launcher sha $(sha $C/a11_launch.sh)"
for c in $CELLS; do
  b=${c%%:*}; t=${c#*:}; read o N P <<< "$(bench_out $b)"; f=$o/absfix_$t.jsonl
  if no_new_work; then L "no new cells: $(why_stop)"; break; fi
  if [ "$(n_ids $f)" -ge $N ]; then
    s=$(sidecar_field $o/absfix_$t.a11.json state)
    if [ "$s" = complete ]; then ST $b $t skipped_complete; continue; fi
    if [ "$s" = running ] && [ "$(sidecar_field $o/absfix_$t.a11.json probe_sha256)" = "$(sha $P)" ]; then
      if cell_claim $b $t; then   # the launcher that ran it died between the last row and the sidecar update
        sidecar $b $t complete "$(sidecar_field $o/absfix_$t.a11.json model)" - "$(n_ids $f)" "$(sha $P)" ""; cell_release $b $t
        ST $b $t complete "finalised at launch"; L "finalised $b:$t (complete dump, sidecar was 'running')"
      else ST $b $t claimed_elsewhere "$(cat $X/claims/${b}__$t/owner 2>/dev/null)"; fi
      continue; fi
    ST $b $t complete_unverified "sidecar state '$s'"; L "WARNING $b:$t is complete but its sidecar is '$s': not accepted by the read"; continue
  fi
  if ! M=$(locate $t); then ST $b $t missing_here; L "$b:$t: no checkpoint on this host"; continue; fi
  if ! cell_claim $b $t; then ST $b $t claimed_elsewhere "$(cat $X/claims/${b}__$t/owner 2>/dev/null)"; continue; fi
  if [ "$(n_ids $f)" -ge $N ]; then   # completed by another launcher between the check above and this claim
    cell_release $b $t; ST $b $t skipped_complete "completed elsewhere before this claim (sidecar '$(sidecar_field $o/absfix_$t.a11.json state)')"
    continue; fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ] && ! no_new_work; do nap 20; done
  if no_new_work; then cell_release $b $t; L "no new cells: $(why_stop)"; break; fi
  run_cell $b $t "$M" &
  nap 5
done
wait
L "launcher done"
