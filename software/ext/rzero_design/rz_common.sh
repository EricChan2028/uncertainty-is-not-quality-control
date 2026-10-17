# A7-5 shared constants and helpers, sourced by rz_setup.sh, rz_wait_and_run.sh, rz_smoke.sh, rz_run.sh, rz_stop.sh and
# rz_umwp.sh. Defines constants and functions only. Callers that log set TAG, STORAGE_PATH and LEDGER first.
: ${RZ:=/run/ti/rzero} ${D:=/home/tione/notebook/rsi_rl/ext/rzero_design} ${X:=/home/tione/notebook/rsi_rl/ext/rzero_runs}
MK=/dev/shm                          # GPU markers $MK/gpu_single_<g>: the claim protocol all our launchers respect
A75_T0=1790233800                    # T0 = 2026-09-24 15:10 CST (A7 registration)
A75_STOP=$((A75_T0 + 71 * 3600))     # 2026-09-27 14:10 CST: all A7-5 compute stops (enforced by the monitor)
A75_TREE=eb84092160b556ad9c4382f5795dd4b519414c3dbff66c949a18796fe37b0a1b   # patched working tree (C-locale file order); any other hash is fatal
L(){ echo "$(date '+%F %T %Z') $(hostname | cut -c1-22) $TAG: $*" | tee -a $STORAGE_PATH/driver.log >> $LEDGER; }
fail(){ L "preflight failed: $*"; exit 3; }
port_open(){ (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }

# ---- our own processes: exactly those whose exec-time environment contains RZ_A75_TAG=<tag> (a driver exports the tag
# after its own exec, so its /proc environ does not carry it)
tagged_pids(){ grep -lzx "RZ_A75_TAG=$1" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3; }
# A7-5a: vLLM engine processes blank their environ view (setproctitle), so they are also recognised by working directory
# (every R-Zero process runs with its cwd inside $RZ) or by a tagged ancestor.
rz_cwd(){ local c; c=$(readlink /proc/$1/cwd 2>/dev/null) || return 1; case "$c" in "$RZ"|"$RZ"/*) return 0;; esac; return 1; }
anc_tagged(){ local q=$1 n=0; while [ -n "$q" ] && [ "$q" -gt 1 ] 2>/dev/null && [ $n -lt 30 ]; do
    grep -qzx "RZ_A75_TAG=$TAG" /proc/$q/environ 2>/dev/null && return 0; q=$(ps -o ppid= -p $q 2>/dev/null | tr -d ' '); n=$((n + 1)); done; return 1; }
rz_engine_pids(){ local q; { gpu_apps | cut -d, -f1; pgrep '^VLLM::'; } 2>/dev/null | sort -u | while read -r q; do [ -n "$q" ] && rz_cwd $q && echo $q; done; }
# TERM, then KILL, then wait up to 120 s until none is left (a dying CUDA process can stay listed for a while; fix 5b)
kill_tag(){ local sig p w=0
  for sig in TERM KILL; do p=$(tagged_pids $1; rz_engine_pids); [ -z "$p" ] || kill -$sig $p 2>/dev/null; sleep 10; done
  while p=$(tagged_pids $1; rz_engine_pids); [ -n "$p" ] && [ $w -lt 120 ]; do kill -KILL $p 2>/dev/null; sleep 5; w=$((w + 5)); done
  [ -z "$p" ] || L "warning: tagged processes still present 120 s after KILL: $(echo $p | cut -c1-100)"; }
# the same from inside a tagged run: an untagged child does the killing, so it cannot hit itself
kill_own(){ env -u RZ_A75_TAG bash -c "TAG=$TAG STORAGE_PATH=$STORAGE_PATH LEDGER=$LEDGER; source $D/rz_common.sh; kill_tag $TAG"; }

# ---- GPUs (nvidia-smi queries only)
gpu_apps(){ timeout 60 nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+, '; }
gpu_mem(){ timeout 60 nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+, '; }
gpu_names(){ timeout 60 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sort -u; }
# GPU use that is not ours (fix 5a): a compute process whose /proc/<pid>/environ lacks RZ_A75_TAG=$TAG (a process of another
# container has no /proc entry here and counts as foreign), or > 1500 MiB in use on a GPU none of our processes is on.
# A process of this container that is exiting (no address space left, so no environment to read, while nvidia-smi still
# lists it) is not counted as foreign, and its GPU is not flagged by the memory rule (review item 1).
# One line per finding; nothing when the GPUs are ours alone.
foreign_gpu(){ local pid uuid idx m ours=" "
  while IFS=', ' read -r pid uuid; do
    if grep -qzx "RZ_A75_TAG=$TAG" /proc/$pid/environ 2>/dev/null; then ours="$ours$uuid "
    elif rz_cwd $pid || anc_tagged $pid; then ours="$ours$uuid "
    elif [ -d /proc/$pid ] && ! grep -q '^VmRSS:' /proc/$pid/status 2>/dev/null; then ours="$ours$uuid "
    else echo "pid $pid ($(tr '\0' ' ' 2>/dev/null < /proc/$pid/cmdline | cut -c1-60)) on $uuid"; fi
  done < <(gpu_apps)
  while IFS=', ' read -r idx uuid m; do
    [ "$m" -gt 1500 ] 2>/dev/null && case "$ours" in *" $uuid "*) ;; *) echo "gpu $idx: $m MiB in use, none of it ours";; esac
  done < <(gpu_mem); }
# a finding counts only if it is still there 30 s later (a process of ours that just ended can stay listed for seconds)
foreign_confirmed(){ local f; f=$(foreign_gpu); [ -n "$f" ] || return 0; sleep 30; f=$(foreign_gpu); [ -z "$f" ] || echo "$f" | head -3 | tr '\n' ';'; }

# ---- memory: the host's 1400 GiB memory cgroup counts /dev/shm (tmpfs); headroom = limit - (rss + shmem), in GiB (D-memory)
headroom_gib(){ local c=/sys/fs/cgroup lim used
  if [ -r $c/memory/memory.limit_in_bytes ]; then lim=$(cat $c/memory/memory.limit_in_bytes)
    used=$(awk '$1=="total_rss"||$1=="total_shmem"{s+=$2} END{printf "%.0f", s}' $c/memory/memory.stat)
  else lim=$(cat $c/memory.max); used=$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{printf "%.0f", s}' $c/memory.stat); fi
  [ "$lim" = max ] && lim=$(awk '/^MemTotal/{printf "%.0f", $2 * 1024}' /proc/meminfo)
  echo $(( (lim - used) / 1073741824 )); }

# ---- our other jobs on this host: queue runners, launchers, chain scripts and /root job scripts (the passive daemons
# hang_watchdog.sh and ckpt_janitor[2].sh excepted; a4_pair_watchdog.sh starts jobs and counts); rz_* scripts excluded
launchers_alive(){ { pgrep -af '(runner[0-9]*|umwp_launch[a-z_]*|a[3-7][a-z]?_[a-z0-9_]+)\.sh'; pgrep -af '^(/usr)?(/bin/)?bash /root/[A-Za-z0-9_.-]+\.sh'; } \
  | grep -v 'rz_' | grep -vE 'bash /root/(hang_watchdog|ckpt_janitor2?)\.sh( |$)' | sort -u; }

# ---- GPU markers (fixes 6 and 8, D-watchdogs)
m_field(){ cut -d' ' -f$2 $MK/gpu_single_$1 2>/dev/null; }
pid_is_rz(){ [ -n "$1" ] && grep -q 'rz_run\.sh\|rz_smoke\.sh' /proc/$1/cmdline 2>/dev/null; }
# a PASS of the smoke test on this host within the last hour (the smoke keeps its markers for the run that follows)
smoke_pass_recent(){ local e; e=$(grep 'smoke PASSED' $X/ledger_smoke.log 2>/dev/null | grep -- " host=$(hostname) " | tail -1 | grep -o ' epoch=[0-9]*' | cut -d= -f2)
  [ -n "$e" ] && [ $(( $(date +%s) - e )) -le 3600 ]; }
# may this run take over marker g? own leftovers (our tag, that driver dead) and the markers of a smoke test that PASSED here
# in the last hour
takeover_ok(){ local t p; t=$(m_field $1 1); p=$(m_field $1 2)
  [ "$t" = "$TAG" ] && ! pid_is_rz "$p" && return 0
  [ "$t" = rzA75_smoke ] && [ "$TAG" != rzA75_smoke ] && ! pid_is_rz "$p" && smoke_pass_recent && return 0
  return 1; }
# marker g is free for us: absent, ours to take over, or stale. Stale (fix 6) = older than 10 min and not naming a live A7-5
# driver, while the host-wide checks run before it found no GPU compute process, < 1500 MiB on every GPU and none of our
# launchers alive. A stale marker is logged and removed.
marker_ok(){ local f=$MK/gpu_single_$1
  [ -e $f ] || return 0
  takeover_ok $1 && return 0
  if [ -z "$(find $f -mmin -10 2>/dev/null)" ] && [ -e $f ] && ! pid_is_rz "$(m_field $1 2)"; then
    L "stale marker $f removed (content '$(head -c 60 $f 2>/dev/null)', age $(( $(date +%s) - $(stat -c %Y $f) )) s)"; rm -f $f; return 0; fi
  return 1; }
release(){ local g; [ "${KEEP_MARKERS:-0}" = 1 ] && return 0
  for g in 0 1 2 3 4 5 6 7; do [ "$(m_field $g 1)" = "$TAG" ] && rm -f $MK/gpu_single_$g; done; rm -f $STORAGE_PATH/driver.pid; }
# markers a PASSED smoke test kept (first field rzA75_smoke) that no live A7-5 driver holds any more (review item 3):
# removed when a waiter ends and by rz_stop.sh <S>, so that no leftover blocks our other launchers
drop_smoke_markers(){ local g n=0
  for g in 0 1 2 3 4 5 6 7; do [ "$(m_field $g 1)" = rzA75_smoke ] && ! pid_is_rz "$(m_field $g 2)" && rm -f "${MK:?}"/gpu_single_$g && n=$((n + 1)); done
  [ $n = 0 ] || L "removed $n GPU markers that a passed smoke test kept and no run took over"; }
# a stop requested by the monitor (stop_code = "<exit code> <reason>") ends the run with that code; any other signal with 143
on_signal(){ local c; if [ -f $STORAGE_PATH/stop_code ]; then c=$(cut -d' ' -f1 $STORAGE_PATH/stop_code)
  L "stopped by the monitor: $(cut -d' ' -f2- $STORAGE_PATH/stop_code)"; exit $c; fi; L "stopped by signal"; exit 143; }
stop_check(){ [ ! -f $STORAGE_PATH/stop_code ] || on_signal; }
# exit handler of rz_run.sh and rz_smoke.sh: no second signal can cut it short; watcher and monitor stopped, our processes
# gone, markers released (unless a PASSED smoke test keeps them)
cleanup(){ trap '' TERM INT HUP; kill $XW $MON 2>/dev/null; kill_own; release; }
# claim all 8 GPUs: each marker is created atomically (noclobber); own leftovers and a PASSED smoke test's kept markers are
# taken over by an atomic rename; released on exit
claim(){ local g; trap release EXIT; trap on_signal TERM INT HUP
  for g in 0 1 2 3 4 5 6 7; do
    ( set -o noclobber; echo "$TAG $$ $(date +%s)" > $MK/gpu_single_$g ) 2>/dev/null && continue
    if takeover_ok $g; then echo "$TAG $$ $(date +%s)" > $MK/.gpu_single_$g.$$ && mv -f $MK/.gpu_single_$g.$$ $MK/gpu_single_$g && continue; fi
    fail "claim lost: $MK/gpu_single_$g was taken meanwhile ('$(head -c 40 $MK/gpu_single_$g 2>/dev/null)')"
  done; }

# ---- is this host free for an A7-5 run? Prints the first reason why not, or nothing. Removes stale markers.
host_check(){ local g idx uuid u n x p
  [ -f $RZ/R-Zero/.a75_patched ] || { echo "environment not built (run rz_setup.sh)"; return; }
  [ "$(cat $RZ/R-Zero/.a75_patched)" = "$(sha256sum < $D/rzero_A7_5.patch | cut -c1-64)" ] || { echo "working copy built from another patch (rerun rz_setup.sh)"; return; }
  [ "$(cat $RZ/R-Zero/.a75_tree_sha256 2>/dev/null)" = "$A75_TREE" ] || { echo "working-tree hash is not the registered $A75_TREE (fix 15)"; return; }
  [ "$(gpu_mem | wc -l)" -eq 8 ] || { echo "not 8 GPUs"; return; }
  x=$(launchers_alive); [ -z "$x" ] || { echo "our own job scripts are alive on this host: $(echo $x | cut -c1-200)"; return; }
  n=$(gpu_apps | wc -l); [ "$n" -eq 0 ] || { echo "$n GPU compute processes present"; return; }
  while IFS=', ' read -r idx uuid u; do [ "$u" -lt 1500 ] || { echo "GPU $idx has $u MiB in use"; return; }; done < <(gpu_mem)
  for g in 0 1 2 3 4 5 6 7; do marker_ok $g || { echo "$MK/gpu_single_$g is held ('$(head -c 40 $MK/gpu_single_$g 2>/dev/null)')"; return; }; done
  for p in 5000 5001 5002 5003; do port_open $p && { echo "port $p is in use"; return; }; done
  [ "$(df --output=avail -BG /run/ti | tail -1 | tr -dc 0-9)" -ge 1500 ] || { echo "/run/ti has < 1.5 TB free"; return; }
  [ "$(df --output=avail -BG $MK | tail -1 | tr -dc 0-9)" -ge 250 ] || { echo "/dev/shm has < 250 GB free"; return; }
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 300 ] || { echo "MemAvailable < 300 GB"; return; }
  x=$(headroom_gib); [ "$x" -ge 250 ] 2>/dev/null || { echo "memory-cgroup headroom $x GiB < 250 GiB (D-memory)"; return; }; }
# the host must be completely free and ours alone for the whole run (vLLM memory fractions are fractions of total memory):
# host_check twice, 60 s apart
preflight(){ local r t; for t in 1 2; do r=$(host_check); [ -z "$r" ] || fail "$r"; [ $t = 2 ] || sleep 60; done; }

# ---- 60 s monitor (fixes 5a and 9, D-memory; review items 1 and 4): stop this run on foreign GPU use (exit 6),
# memory-cgroup headroom < 100 GiB (exit 7; Ray's own memory monitor, on by default in ray 2.49.2, kills workers at 95 % of
# the cgroup limit counted without page cache, i.e. at about 70-75 GiB of this headroom) or the compute deadline T0+71h
# (exit 8); a caller may define extra_reason for further rules. The monitor writes stop_code, kills our tagged processes
# and signals the driver, whose on_signal exits with the code. The monitor runs without RZ_A75_TAG, so a kill_own of the
# driver never cuts its waits short; while the driver tears down a failed attempt ($STORAGE_PATH/teardown exists before or
# after the 30 s confirmation) it skips the foreign-use rule, and the driver's own check after the teardown decides.
stop_reason(){ local f h
  [ -f $STORAGE_PATH/teardown ] || { f=$(foreign_confirmed); [ -z "$f" ] || [ -f $STORAGE_PATH/teardown ] || { echo "6 foreign GPU use: $f"; return; }; }
  h=$(headroom_gib); case "$h" in ''|*[!0-9-]*) ;; *) [ "$h" -ge 100 ] || { echo "7 memory-cgroup headroom $h GiB < 100 GiB"; return; };; esac
  [ "$(date +%s)" -lt $A75_STOP ] || { echo "8 compute deadline T0+71h (X5)"; return; }
  [ "$(type -t extra_reason)" != function ] || extra_reason; }
start_monitor(){ DRIVER=$$; ( unset RZ_A75_TAG; while :; do sleep 60; r=$(stop_reason); [ -n "$r" ] || continue
    echo "$r" > $STORAGE_PATH/stop_code; L "monitor: ${r#* }: stopping this run (exit ${r%% *})"; kill_own; kill -TERM $DRIVER 2>/dev/null; exit 0
  done ) & MON=$!; }

# ---- seeds and hosts (fix 4). Ids 0 and 1 are the registered seeds; 2 and 3 replace abandoned ids and need >= 1 and >= 2
# abandoned ids, so at most two seed runs can ever count. A seed runs only on the host that first claimed it.
seed_ok(){ local n; [ ! -f $X/s$1.abandoned ] || { echo "refusing: seed $1 was abandoned ($(cat $X/s$1.abandoned))"; return 1; }
  n=$(ls $X/s[0-9].abandoned 2>/dev/null | wc -l)
  [ "$1" -lt 2 ] || [ "$n" -ge $(( $1 - 1 )) ] || { echo "refusing: seed id $1 replaces an abandoned seed and needs $(( $1 - 1 )) abandoned seed ids ($n found)"; return 1; }
  [ ! -f $X/s$1.host ] || [ "$(cut -d' ' -f1 $X/s$1.host)" = "$(hostname)" ] || {
    echo "refusing: seed $1 belongs to host $(cut -d' ' -f1 $X/s$1.host) ($X/s$1.host); to restart it elsewhere, run 'rz_stop.sh $1 abandon <reason>' and start a new seed id"; return 1; }; }
claim_host(){ ( set -o noclobber; echo "$(hostname) $STORAGE_PATH $(date +%s)" > $X/s$1.host ) 2>/dev/null \
  || [ "$(cut -d' ' -f1 $X/s$1.host)" = "$(hostname)" ] || { L "refusing: seed $1 was claimed by $(cut -d' ' -f1 $X/s$1.host) meanwhile"; exit 2; }; }
# software and hardware stack: patch, working tree, venv package list, GPU model and driver, system python
stack_fp(){ { cat $RZ/R-Zero/.a75_patched $RZ/R-Zero/.a75_tree_sha256; sha256sum < $RZ/venv_freeze.txt; gpu_names; /usr/bin/python3 --version; } 2>&1 | sha256sum | cut -c1-16; }
# may seed <S> skip the smoke test? It has started before, or a smoke test PASSED on the same stack (for seed 0: on this host)
smoke_not_needed(){ local fp; [ -f $X/s$1.host ] && return 0; fp=$(stack_fp)
  if [ "$1" = 0 ]; then grep 'smoke PASSED' $X/ledger_smoke.log 2>/dev/null | grep -- " host=$(hostname) " | grep -q -- " stack=$fp "
  else grep 'smoke PASSED' $X/ledger_smoke.log 2>/dev/null | grep -q -- " stack=$fp "; fi; }

# ---- UMWP dump files (fix 10): cut a partial last line (a kill mid-write); count unique parseable ids
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

# ---- driver helpers (rz_run.sh)
cur_stage(){ local i t; for i in 1 2 3; do for t in questioner solver; do [ -f $M/${ABBR}_${t}_v$i/A75_DONE ] || { echo ${ABBR}_${t}_v$i; return; }; done; done; }
tracker(){ cat $M/$1/latest_global_step.txt 2>/dev/null || echo 0; }
# fix 12: an admitted set smaller than one rollout batch (512 rows, after the trainer's overlong-prompt filter) leaves the
# repository's drop_last=True dataloader empty, and its assertion fails on every attempt. A finding about the default
# pipeline: logged with n and written to ext/rzero_runs/s<S>/finding_admitted_lt512.json; no retry (exit 9).
small_admitted(){ local off=$1 i f n hit
  hit=$(tail -c +$(( off + 1 )) $STORAGE_PATH/main.log 2>/dev/null | grep -c 'assert len(train_dataloader) >= 1')
  for i in 1 2 3; do f=$GQ/${ABBR}_solver_v${i}_admitted.parquet
    [ -f $f ] && [ ! -f $M/${ABBR}_solver_v$i/A75_DONE ] || continue
    n=$(/usr/bin/python3 -c 'import sys, pyarrow.parquet as p; print(p.read_metadata(sys.argv[1]).num_rows)' $f)
    if [ "$n" -lt 512 ] || [ "$hit" -gt 0 ]; then
      echo "{\"seed\": $S, \"iteration\": $i, \"n_admitted\": $n, \"assertion_in_log\": $hit, \"time\": $(date +%s)}" > $X/s$S/finding_admitted_lt512.json
      L "finding (X8): the admitted set of iteration $i has $n rows; the solver cannot train (drop_last with 512-row batches; assertion lines in this attempt: $hit); no retry"
      return 0; fi; done; return 1; }
# fix 14: resume fallback. The attempt loaded checkpoint k of the stage it was running, saved no newer one, and failed with a
# traceback after 'Load from checkpoint', either through the checkpoint-loading code or for the second attempt in a row at
# the same k: the stage directory is moved aside (its checkpoints deleted, the rest kept) and the stage reruns from its start.
resume_fallback(){ local st=$1 k0=$2 off=$3 seg dst lf=$STORAGE_PATH/loadfail
  [ -n "$st" ] && [ -d $M/$st ] && [ "$k0" -gt 0 ] && [ "$(tracker $st)" = "$k0" ] || { rm -f $lf; return 0; }
  seg=$(tail -c +$(( off + 1 )) $STORAGE_PATH/main.log 2>/dev/null | sed -n '/Load from checkpoint: /,$p')
  echo "$seg" | grep -q '^Traceback\|Traceback (most recent call last)' || { rm -f $lf; return 0; }
  if echo "$seg" | grep -qE 'in _?load_checkpoint$|in load_state_dict$' || [ "$(cat $lf 2>/dev/null)" = "$st:$k0" ]; then
    dst=$M/$st.discarded_$(date +%s); mv $M/$st $dst; rm -rf $dst/global_step_*; rm -f $lf
    L "resume fallback: $st failed after loading global_step_$k0; stage directory moved to $dst (checkpoints deleted); the stage reruns from its start"
  else echo "$st:$k0" > $lf; L "attempt failed after loading $st/global_step_$k0 (a second such failure triggers the resume fallback)"; fi; }
