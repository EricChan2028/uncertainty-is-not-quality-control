#!/bin/bash
# A12 provenance: CPU-only test of a12_launch.sh on SYNTHETIC state under /tmp/a12lt. The launcher is copied with its roots
# rewritten (/home/tione/notebook/rsi_rl, /dev/shm, /root/models, /run/ti/rzero -> /tmp/a12lt/...), its waits shortened, and
# the plain runner / the wrapper / the judge replaced by fake jobs that only sleep and write complete outputs; nvidia-smi is a
# fake on PATH (memory and compute apps from files). Nothing touches the real GPUs, markers or project tree.
# Tests: T1 full run (all local cells complete, markers and claims released), T1b rerun skips everything, T2 safe stop mid-run
# (stop flag, jobs killed, wait, cleanup; no job process or marker left), T3 co-tenant yield (foreign compute app on our GPU ->
# job killed, GPU released and cooling down, the cell resumes on ANOTHER GPU and completes), T4 busy GPUs are never claimed,
# T5 STOP file starts nothing, T6 a foreign compute app on a low-memory (608 MiB) GPU is never claimed, T7 a reparented
# grandchild of a job (its environment dropped, as setproctitle does) is killed on a safe stop and on a yield, T8 after the
# launcher and its subshells are SIGKILLed, a new launch kills the orphan jobs, releases the dead launcher's markers and
# completes, T9 three descendants of a job that ignore TERM, have an empty environment and left its process group (a child
# that calls setpgid(0, 0), a child that calls setsid(), a grandchild reparented at once that calls setpgid(0, 0)) survive
# neither a safe stop nor a yield.
#   bash dev_a12_launch_test.sh <dir with a12_launch.sh>
set -u
SRC=$(cd "$1" && pwd); T=/tmp/a12lt; rm -rf $T; mkdir -p $T/bin $T/gpu $T/shm/models $T/C $T/R/docs $T/R/code $T/R/dirB/code $T/R/dirB/data $T/R/dirB/results/evals $T/R/dirA5/results
FP=""
fail(){ echo "FAIL: $*"; bash $T/C/a12_launch.sh stop >/dev/null 2>&1; [ -n "$FP" ] && kill $FP 2>/dev/null; tail -30 $T/R/ext/a12_runs/a12_launch.log 2>/dev/null; exit 1; }
# fake nvidia-smi
cat > $T/bin/nvidia-smi <<'EOF'
#!/bin/bash
G=/tmp/a12lt/gpu; i=""; q=""
for a in "$@"; do case $a in -i) nx=i;; --query-gpu=*) q=${a#*=};; --query-compute-apps=*) q=apps;; *) [ "${nx:-}" = i ] && { i=$a; nx=""; };; esac; done
case $q in
  memory.used) cat $G/mem_$i 2>/dev/null || echo 0;;
  uuid) echo GPU-$i;;
  apps) cat $G/apps 2>/dev/null;;
  *) for g in 0 1 2 3 4 5 6 7; do echo "$g, GPU-$g, $(cat $G/mem_$g 2>/dev/null || echo 0)"; done;;
esac
EOF
chmod +x $T/bin/nvidia-smi; export PATH=$T/bin:$PATH
# fake model dirs (tokenizer files linked from the real base model, read-only)
mk_model(){ mkdir -p $1; for f in tokenizer.json tokenizer_config.json vocab.json merges.txt config.json; do ln -sf /root/models/Qwen3-4B-Base/$f $1/$f; done
  echo '{"weight_map": {"a": "model-00001-of-00001.safetensors"}}' > $1/model.safetensors.index.json; echo x > $1/model-00001-of-00001.safetensors; }
mk_model $T/models/Qwen3-4B-Base
for i in 1 2 3; do F=$T/RZ/s1/models/rzero_qwen3_4b_Base_s1_solver_v$i; mk_model $F/global_step_15/actor/huggingface; touch $F/A75_DONE; done
mk_model $T/shm/models/Qwen2.5-32B-Instruct; touch $T/shm/models/Qwen2.5-32B-Instruct.ok
for n in math500_train math_heldout1000 gsm8k_test500; do echo '{"id": 1}' > $T/R/dirB/data/$n.jsonl; done
echo "# fake" > $T/R/dirB/code/b_eval.py; echo "# fake" > $T/R/code/loto_common.py; echo "# fake" > $T/R/dirB/code/pdc_util.py
# fake jobs (FAKE_REPARENT=1: the job leaves a grandchild that is reparented and has no environment, then sleeps;
# FAKE_LEAVE=1: the job leaves three TERM-ignoring descendants with no environment outside its process group, then sleeps)
cat > $T/fakejob.py <<'EOF'
import os, signal, sys, time
def reparent():
    if os.environ.get("FAKE_REPARENT") != "1":
        return
    if os.fork() == 0:
        if os.fork() == 0:
            open("/tmp/a12lt/orphan.pid", "w").write(str(os.getpid()))
            os.execve("/bin/sleep", ["sleep", "600"], {})
        os._exit(0)
    os.wait()
def leave():
    # a child that calls setpgid(0, 0), a child that calls setsid(), and a grandchild that is reparented at once (its parent
    # exits) and calls setpgid(0, 0); all three ignore TERM and exec sleep with an empty environment
    if os.environ.get("FAKE_LEAVE") != "1":
        return
    def spawn(name, how):
        pid = os.fork()
        if pid == 0:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_IGN); how()
                if name:
                    open(f"/tmp/a12lt/leave_{name}.pid", "w").write(str(os.getpid()))
                    os.execve("/bin/sleep", ["sleep", "600"], {})
            finally:
                os._exit(0)
        return pid
    spawn("pgid", lambda: os.setpgid(0, 0))
    spawn("sid", os.setsid)
    os.waitpid(spawn("", lambda: spawn("orphan", lambda: os.setpgid(0, 0))), 0)
EOF
cat > $T/C/a12_plain_eval.py <<'EOF'
import argparse, json, os, sys, time
sys.path.insert(0, "/tmp/a12lt"); from fakejob import leave, reparent
ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tag"); ap.add_argument("--data"); ap.add_argument("--k"); a = ap.parse_args()
assert os.environ.get("A12_TAG") and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "") and a.k == "1"
assert os.getpgid(0) == os.getpid() or os.getpgid(0) == os.getppid(), "the job is not its own process group"
open(f"/tmp/a12lt/jobs.log", "a").write(f"plain {a.tag} gpu={os.environ['CUDA_VISIBLE_DEVICES']} pid={os.getpid()}\n")
reparent(); leave(); time.sleep(float(os.environ.get("FAKE_SLEEP", "2")))
s = {n: {"n": 1, "greedy_acc": 0.5, "greedy_acc_lenient": 0.5} for n in a.data.split(",")}
json.dump(s, open(f"/tmp/a12lt/R/dirB/results/evals/{a.tag}.summary.json", "w"))
EOF
cat > $T/C/a12_rzchat_eval.py <<'EOF'
import argparse, json, os, sys, time
sys.path.insert(0, "/tmp/a12lt"); from fakejob import leave, reparent
ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--tag"); a = ap.parse_args()
assert os.environ.get("A12_TAG") and os.environ.get("CUDA_VISIBLE_DEVICES")
open(f"/tmp/a12lt/jobs.log", "a").write(f"rzchat {a.tag} gpu={os.environ['CUDA_VISIBLE_DEVICES']} pid={os.getpid()}\n")
reparent(); leave(); time.sleep(float(os.environ.get("FAKE_SLEEP", "2")))
json.dump({n: {"n": 1, "complete": True} for n in ("math500_train", "math_heldout1000", "gsm8k_test500")},
          open(f"/tmp/a12lt/R/dirB/results/evals/{a.tag}.summary.json", "w"))
EOF
cat > $T/C/a12_open_judge.py <<'EOF'
import os, time
g = os.environ["CUDA_VISIBLE_DEVICES"]; assert len(g.split(",")) == 2 and os.environ.get("A12_TAG")
open(f"/tmp/a12lt/jobs.log", "a").write(f"open gpu={g} pid={os.getpid()}\n")
time.sleep(float(os.environ.get("FAKE_SLEEP", "2")))
os.makedirs("/tmp/a12lt/R/dirA5/results/a12_open", exist_ok=True)
open("/tmp/a12lt/R/dirA5/results/a12_open/verdicts.jsonl", "w").write("{}\n" * 1158)
EOF
S_(){ sha256sum $1 | cut -c1-64; }
sed -e "s|/home/tione/notebook/rsi_rl|$T/R|g" -e "s|/dev/shm|$T/shm|g" -e "s|/root/models/Qwen3-4B-Base|$T/models/Qwen3-4B-Base|g" \
    -e "s|/run/ti/rzero|$T/RZ|g" -e "s|^IDLE_GAP=60$|IDLE_GAP=1|" -e "s|^COOLDOWN=600$|COOLDOWN=30|" -e "s|^YIELD_WAIT=600$|YIELD_WAIT=3|" \
    -e "s|^MON=60$|MON=2|" -e "s|^CONFIRM=30$|CONFIRM=2|" -e "s|^KILL_GRACE=20$|KILL_GRACE=2|" -e "s|^STOP_GRACE=120$|STOP_GRACE=20|" \
    -e "s|^SCAN=30$|SCAN=1|" \
    -e "s|BEVAL_SHA=[0-9a-f]*|BEVAL_SHA=$(S_ $T/R/dirB/code/b_eval.py)|" \
    -e "s|code/loto_common.py:[0-9a-f]*|code/loto_common.py:$(S_ $T/R/code/loto_common.py)|" \
    -e "s|dirB/code/pdc_util.py:[0-9a-f]*|dirB/code/pdc_util.py:$(S_ $T/R/dirB/code/pdc_util.py)|" \
    -e "s|math500_train:[0-9a-f]*|math500_train:$(S_ $T/R/dirB/data/math500_train.jsonl)|" \
    -e "s|math_heldout1000:[0-9a-f]*|math_heldout1000:$(S_ $T/R/dirB/data/math_heldout1000.jsonl)|" \
    -e "s|gsm8k_test500:[0-9a-f]*|gsm8k_test500:$(S_ $T/R/dirB/data/gsm8k_test500.jsonl)|" \
    $SRC/a12_launch.sh > $T/C/a12_launch.sh
for v in IDLE_GAP=1 COOLDOWN=30 YIELD_WAIT=3 MON=2 CONFIRM=2 KILL_GRACE=2 STOP_GRACE=20 SCAN=1; do grep -qx "$v" $T/C/a12_launch.sh || fail "sed of $v"; done
FILES=$(grep '^FILES=' $T/C/a12_launch.sh | cut -d'"' -f2)
for f in $FILES; do [ -e $T/C/$f ] || { mkdir -p $(dirname $T/C/$f); echo "test $f" > $T/C/$f; }; done
find $T/C -type f | while read -r f; do echo "$(S_ $f)  $f" >> $T/R/docs/registration_hashes.txt; done
L=$T/R/ext/a12_runs/a12_launch.log; LA="bash $T/C/a12_launch.sh"
markers(){ ls $T/shm/gpu_single_* 2>/dev/null | wc -l; }
jobs_alive(){ pgrep -f "^/usr/bin/python3 $T/C/(a12_plain_eval|a12_rzchat_eval|a12_open_judge)\.py" | wc -l; }   # the python job, not its timeout
alive(){ [ -n "$1" ] && [ -d /proc/$1 ] && ! grep -q '^State:[[:space:]]*Z' /proc/$1/status 2>/dev/null; }
since(){ tail -n +$(( $1 + 1 )) $L; }
gpus_used(){ grep -o 'gpu=[0-9,]*' $T/jobs.log | sort -u | tr '\n' ' '; }
reset_out(){ rm -rf $T/R/dirB/results/evals/* $T/R/dirA5/results/a12_open $T/R/ext/a12_runs/claims $T/R/ext/a12_runs/cooldown $T/jobs.log $T/orphan.pid $T/leave_*.pid; }

echo "== locate / check"; $LA locate; $LA check | tail -3
$LA check | grep -q "NOT REGISTERED" && fail "check: a file is not registered"
echo "== T1 full run"; A12_MAXPAR=3 FAKE_SLEEP=2 timeout 600 $LA run all; rc=$?
[ $rc = 0 ] || fail "T1 rc $rc"
for m in qwen3_4b_base rzero_s1_it1 rzero_s1_it2 rzero_s1_it3; do
  [ -f $T/R/dirB/results/evals/a12_plain_$m.summary.json ] && [ -f $T/R/dirB/results/evals/a12rz_$m.summary.json ] || fail "T1 $m missing"; done
[ -f $T/R/dirA5/results/a12_open/verdicts.jsonl ] || fail "T1 open missing"
grep -q "rzero_s0_it3: model not on this host" $L || fail "T1 s0 cells should be missing here"
[ "$(markers)" = 0 ] && [ -z "$(ls $T/R/ext/a12_runs/claims 2>/dev/null)" ] && [ ! -e $T/shm/a12_launch.pid ] || fail "T1 leftovers"
[ "$(grep -c '^open gpu=[0-7],[0-7] ' $T/jobs.log)" = 1 ] || fail "T1 open judge not on a GPU pair"
[ "$(gpus_used | wc -w)" -ge 2 ] || fail "T1 all jobs on one GPU (random order expected)"
echo "T1 ok: $(wc -l < $T/jobs.log) jobs, $(grep -c COMPLETE $L) complete, GPUs used: $(gpus_used)"
echo "== T1b rerun: everything skipped"; A12_MAXPAR=3 timeout 120 $LA run all; [ "$(grep -c skipped_complete $T/R/ext/a12_runs/status_*.tsv)" -ge 9 ] || fail "T1b"
echo "== T2 safe stop"; reset_out
A12_MAXPAR=3 FAKE_SLEEP=300 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 90); do [ "$(markers)" = 3 ] && [ "$(jobs_alive)" -ge 3 ] && break; sleep 1; done; sleep 2
[ "$(jobs_alive)" -ge 3 ] && [ "$(markers)" = 3 ] || fail "T2 jobs did not start ($(jobs_alive) jobs, $(markers) markers)"
$LA stop; for i in $(seq 180); do kill -0 $LP 2>/dev/null || break; sleep 1; done
kill -0 $LP 2>/dev/null && fail "T2 launcher still alive"
wait $LP; rc=$?; [ $rc = 143 ] || fail "T2 rc $rc"
sleep 2; [ "$(jobs_alive)" = 0 ] && [ "$(markers)" = 0 ] && [ -z "$(ls $T/R/ext/a12_runs/claims 2>/dev/null)" ] || fail "T2 leftovers: jobs $(jobs_alive) markers $(markers)"
grep -q "signal: stop flag set" $L && grep -q "stopped" $L || fail "T2 log"
echo "T2 ok: launcher exit 143, no job, marker or claim left"
echo "== T3 co-tenant yield"; reset_out; : > $T/gpu/apps; n0=$(wc -l < $L); ns=$(cat $T/R/ext/a12_runs/status_*.tsv | wc -l)
A12_MAXPAR=1 FAKE_SLEEP=20 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 60); do [ "$(markers)" = 1 ] && [ "$(jobs_alive)" = 1 ] && break; sleep 1; done
g=$(ls $T/shm/gpu_single_* | head -1); g=${g##*_}; echo "99999, GPU-$g" > $T/gpu/apps
for i in $(seq 60); do since $n0 | grep -q "YIELD on gpus $g" && break; sleep 1; done
since $n0 | grep -q "YIELD on gpus $g" || fail "T3 no yield"; : > $T/gpu/apps
for i in $(seq 60); do [ "$(since $n0 | grep -c 'start plain:qwen3_4b_base')" -ge 2 ] && break; sleep 1; done
touch $T/R/ext/a12_runs/STOP     # the yielded cell's retry has started: let it finish, start nothing new
[ "$(since $n0 | grep -c 'start plain:qwen3_4b_base')" = 2 ] || fail "T3 no retry after the yield"
g2=$(since $n0 | grep 'start plain:qwen3_4b_base' | tail -1 | sed 's/.* on gpus \([0-9,]*\) .*/\1/')
[ -n "$g2" ] && [ "$g2" != "$g" ] || fail "T3 the retry re-claimed the yielded GPU $g within its cooldown ($g2)"
for i in $(seq 240); do kill -0 $LP 2>/dev/null || break; sleep 1; done; kill -0 $LP 2>/dev/null && { $LA stop; fail "T3 launcher did not end"; }
cat $T/R/ext/a12_runs/status_*.tsv | tail -n +$((ns + 1)) | grep -q "yielded" && since $n0 | grep -q "COMPLETE plain:qwen3_4b_base" || fail "T3 yielded cell did not complete"
rm -f $T/R/ext/a12_runs/STOP; [ "$(markers)" = 0 ] || fail "T3 markers"
echo "T3 ok: yield on gpu $g, GPU released and cooling down, cell resumed on gpu $g2 and completed"
echo "== T4 busy GPUs"; reset_out; for g in 0 1 2 3 4 5 6; do echo 5000 > $T/gpu/mem_$g; done
A12_MAXPAR=3 FAKE_SLEEP=1 timeout 600 $LA run p2 || fail "T4 rc"
[ -z "$(grep -v 'gpu=7 ' $T/jobs.log)" ] || fail "T4 a busy GPU was used: $(grep -v 'gpu=7 ' $T/jobs.log | head -2)"
rm -f $T/gpu/mem_*; echo "T4 ok: $(wc -l < $T/jobs.log) jobs, all on the only free GPU 7"
echo "== T5 STOP file"; reset_out; touch $T/R/ext/a12_runs/STOP; timeout 60 $LA run all; [ ! -e $T/jobs.log ] || fail "T5 started a job"
rm -f $T/R/ext/a12_runs/STOP; echo "T5 ok"
echo "== T6 foreign app on a low-memory GPU"; reset_out; for g in 0 1 2 3 4 5 6 7; do echo 608 > $T/gpu/mem_$g; done; n0=$(wc -l < $L)
sleep 900 & FP=$!; echo "$FP, GPU-0" > $T/gpu/apps
A12_MAXPAR=3 FAKE_SLEEP=1 timeout 600 $LA run p2 || fail "T6 rc"
[ -z "$(grep 'gpu=0 ' $T/jobs.log)" ] || fail "T6 the co-tenant's GPU 0 was claimed: $(grep 'gpu=0 ' $T/jobs.log | head -2)"
since $n0 | grep -q "YIELD" && fail "T6 a job yielded: $(since $n0 | grep YIELD | head -1)"
[ "$(grep -c '^\(plain\|rzchat\) ' $T/jobs.log)" = 8 ] || fail "T6 not all 8 cells ran"
kill $FP; FP=""; : > $T/gpu/apps; rm -f $T/gpu/mem_*
echo "T6 ok: $(wc -l < $T/jobs.log) jobs on $(gpus_used); GPU 0 (608 MiB, foreign app) never claimed"
echo "== T7 reparented grandchild killed (safe stop, then yield)"; reset_out
A12_MAXPAR=1 FAKE_SLEEP=300 FAKE_REPARENT=1 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 60); do [ -s $T/orphan.pid ] && [ "$(jobs_alive)" = 1 ] && break; sleep 1; done
O=$(cat $T/orphan.pid 2>/dev/null); alive "$O" || fail "T7 no orphan"
[ "$(ps -o ppid= -p $O | tr -d ' ')" != "$(pgrep -f "^/usr/bin/python3 $T/C/a12_plain_eval.py" | head -1)" ] || fail "T7 orphan not reparented"
grep -q A12_TAG /proc/$O/environ 2>/dev/null && fail "T7 orphan still has the tag"
$LA stop; for i in $(seq 120); do kill -0 $LP 2>/dev/null || break; sleep 1; done; wait $LP
sleep 1; alive "$O" && fail "T7 safe stop left the reparented grandchild $O alive"
[ "$(jobs_alive)" = 0 ] && [ "$(markers)" = 0 ] || fail "T7 stop leftovers"
echo "  stop: reparented grandchild $O killed"
reset_out; : > $T/gpu/apps; n0=$(wc -l < $L)
A12_MAXPAR=1 FAKE_SLEEP=300 FAKE_REPARENT=1 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 60); do [ -s $T/orphan.pid ] && [ "$(markers)" = 1 ] && break; sleep 1; done
O=$(cat $T/orphan.pid 2>/dev/null); alive "$O" || fail "T7b no orphan"
g=$(ls $T/shm/gpu_single_* | head -1); g=${g##*_}; rm -f $T/orphan.pid; echo "99999, GPU-$g" > $T/gpu/apps
for i in $(seq 60); do since $n0 | grep -q "YIELD on gpus $g" && break; sleep 1; done; : > $T/gpu/apps
since $n0 | grep -q "YIELD on gpus $g" || fail "T7b no yield"
sleep 4; alive "$O" && { $LA stop; fail "T7b yield left the reparented grandchild $O alive"; }
touch $T/R/ext/a12_runs/STOP; $LA stop; for i in $(seq 120); do kill -0 $LP 2>/dev/null || break; sleep 1; done; wait $LP
O2=$(cat $T/orphan.pid 2>/dev/null); [ -n "$O2" ] && alive "$O2" && fail "T7b the retry's grandchild $O2 survived the stop"
rm -f $T/R/ext/a12_runs/STOP
echo "T7 ok: reparented grandchild (no environment) killed on safe stop and on yield"
echo "== T8 launcher SIGKILLed: orphan jobs and markers of the dead launcher"; reset_out; n0=$(wc -l < $L)
A12_MAXPAR=2 FAKE_SLEEP=300 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 60); do [ "$(markers)" = 2 ] && [ "$(jobs_alive)" = 2 ] && break; sleep 1; done
[ "$(jobs_alive)" = 2 ] || fail "T8 jobs did not start"
pkill -KILL -f "$T/C/a12_launch.sh run"; sleep 2; wait $LP 2>/dev/null
[ "$(jobs_alive)" = 2 ] && [ "$(markers)" = 2 ] || fail "T8 setup: expected 2 orphan jobs and 2 markers ($(jobs_alive), $(markers))"
OJ=$(pgrep -f "$T/C/(a12_plain_eval|a12_rzchat_eval)\.py" | tr '\n' ' ')
for i in $(seq 30); do flock -n $T/shm/a12_launch.lock true && break; sleep 1; done   # the dead shells' short sleeps may hold the lock a few s
flock -n $T/shm/a12_launch.lock true || fail "T8 the orphan jobs hold the launcher lock"
A12_MAXPAR=3 FAKE_SLEEP=1 timeout 600 $LA run p2 || fail "T8 relaunch rc"
for q in $OJ; do alive $q && fail "T8 orphan job $q still alive"; done
since $n0 | grep -q "killing orphan job" && since $n0 | grep -q "released marker" || fail "T8 no reap in the log"
[ "$(markers)" = 0 ] && [ "$(grep -c '^\(plain\|rzchat\) ' $T/jobs.log)" -ge 8 ] || fail "T8 relaunch did not complete"
since $n0 | grep -q "took over a stale claim" || fail "T8 the dead launcher's claims were not taken over"
[ -z "$(ls -d $T/shm/a12_pids_* 2>/dev/null)" ] || fail "T8 pid directory of the dead launcher left"
echo "T8 ok: orphan jobs $OJ killed, dead launcher's markers released, relaunch completed"
echo "== T9 TERM-ignoring descendants without environment that left the job's process group (safe stop, then yield)"; reset_out
lv(){ local f; for f in $T/leave_pgid.pid $T/leave_sid.pid $T/leave_orphan.pid; do [ -s $f ] && echo -n "$(cat $f) "; done; }
lv_ready(){ [ -s $T/leave_pgid.pid ] && [ -s $T/leave_sid.pid ] && [ -s $T/leave_orphan.pid ]; }
lv_check(){ local j js q A B C pp
  j=$(pgrep -f "^/usr/bin/python3 $T/C/a12_plain_eval.py" | head -1); [ -n "$j" ] || fail "T9$1 no job"
  js=$(ps -o sid= -p $j | tr -d ' '); A=$(cat $T/leave_pgid.pid); B=$(cat $T/leave_sid.pid); C=$(cat $T/leave_orphan.pid)
  for q in $A $B $C; do alive $q || fail "T9$1 descendant $q not alive"
    [ "$(ps -o pgid= -p $q | tr -d ' ')" = $q ] || fail "T9$1 $q did not leave the job's process group"
    grep -qa A12_TAG /proc/$q/environ && fail "T9$1 $q still has the tag"
    [ $(( 0x$(awk '/^SigIgn/{print $2}' /proc/$q/status) & 0x4000 )) != 0 ] || fail "T9$1 $q does not ignore TERM"; done
  [ "$(ps -o ppid= -p $A | tr -d ' ')" = $j ] && [ "$(ps -o sid= -p $A | tr -d ' ')" = $js ] || fail "T9$1 setpgid child"
  [ "$(ps -o ppid= -p $B | tr -d ' ')" = $j ] && [ "$(ps -o sid= -p $B | tr -d ' ')" = $B ] || fail "T9$1 setsid child"
  pp=$(ps -o ppid= -p $C | tr -d ' ')
  [ "$(ps -o sid= -p $C | tr -d ' ')" = $js ] && [ "$(ps -o sid= -p $pp | tr -d ' ')" != $js ] || fail "T9$1 grandchild not reparented"; }
A12_MAXPAR=1 FAKE_SLEEP=300 FAKE_LEAVE=1 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 60); do lv_ready && [ "$(jobs_alive)" = 1 ] && break; sleep 1; done; sleep 1
lv_check a; LV=$(lv); [ "$(echo $LV | wc -w)" = 3 ] || fail "T9 expected 3 descendant pids ($LV)"
$LA stop; for i in $(seq 120); do kill -0 $LP 2>/dev/null || break; sleep 1; done; wait $LP
sleep 1; for q in $LV; do alive $q && fail "T9 safe stop left the descendant $q alive"; done
[ "$(jobs_alive)" = 0 ] && [ "$(markers)" = 0 ] || fail "T9 stop leftovers"
echo "  stop: descendants $LV (setpgid child, setsid child, reparented setpgid grandchild) killed"
reset_out; : > $T/gpu/apps; n0=$(wc -l < $L)
A12_MAXPAR=1 FAKE_SLEEP=300 FAKE_LEAVE=1 $LA run p2 > /dev/null 2>&1 & LP=$!
for i in $(seq 60); do lv_ready && [ "$(markers)" = 1 ] && [ "$(jobs_alive)" = 1 ] && break; sleep 1; done; sleep 1
lv_check b; LV=$(lv); [ "$(echo $LV | wc -w)" = 3 ] || fail "T9b expected 3 descendant pids ($LV)"
g=$(ls $T/shm/gpu_single_* | head -1); g=${g##*_}; rm -f $T/leave_*.pid; echo "99999, GPU-$g" > $T/gpu/apps
for i in $(seq 60); do since $n0 | grep -q "YIELD on gpus $g" && break; sleep 1; done; : > $T/gpu/apps
since $n0 | grep -q "YIELD on gpus $g" || fail "T9b no yield"
sleep 4; for q in $LV; do alive $q && { $LA stop; fail "T9b yield left the descendant $q alive"; }; done
touch $T/R/ext/a12_runs/STOP; $LA stop; for i in $(seq 120); do kill -0 $LP 2>/dev/null || break; sleep 1; done; wait $LP
sleep 1; for q in $(lv); do alive $q && fail "T9b the retry's descendant $q survived the stop"; done
rm -f $T/R/ext/a12_runs/STOP
echo "T9 ok: descendants outside the job's process group (no environment, TERM ignored) killed on safe stop and on yield"
echo "ALL A12 LAUNCHER TESTS PASSED"
