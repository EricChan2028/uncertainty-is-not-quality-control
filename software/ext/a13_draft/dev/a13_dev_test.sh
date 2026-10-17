#!/bin/bash
# A13 dev tests (synthetic data only, no GPU, no write outside WORKDIR):  bash dev/a13_dev_test.sh [WORKDIR [PAPER]]
# WORKDIR (default /tmp/a13_dev, must be under /tmp) is deleted and rebuilt. PAPER (optional): a copy of the paper's
# sections/*.tex, only read (the edits are applied to copies inside WORKDIR).
#  B  a13_build.py on a synthetic tree: invariants, determinism, refusal to overwrite, C2 and C1 failures
#  L  two simulated hosts run a13_launch.sh at once: host-local checkpoints, preferred seeds, a co-tenant process on a
#     training GPU (yield, resume with optimizer state), a crashed training attempt, a crashed probe with a partial last line,
#     busy GPUs, cleanup of markers, claims and pid directories, tidy of step20/step40 and archived finals; L10 starvation:
#     every GPU of one host held by co-tenant processes, the other host takes its seeds; train cells before oabs cells
#  S  STOP file (nothing starts), 'stop' during training (TERM, stopped sidecar, cleanup) and restart (resume, completion)
#  D  a13_read.py on the launcher's output (ADDS, checked against an independent computation), refusal to overwrite, and
#     rewritten versions of the tree (in place, restored afterwards): NOT DETECTED with a positive and with a non-positive
#     estimate, REDUCES, NOT RUN (3 seeds unfinished; 3 seeds after rejected configs), failed R-S check, changed input of
#     the R-S check, rejected config (4 seeds, same-seed R-S in the text), edited dump, a CI bound that rounds to 0.0
#  P  (with PAPER) the paper edits of every reading apply to the current sources exactly once each
#  X  real data, read-only: R-S reproduction (--check_only), dry build with the registered hashes, launcher check / locate
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); W=${1:-/tmp/a13_dev}; PAPER=${2:-}; PY=/usr/bin/python3
case $W in /tmp/?*) ;; *) echo "WORKDIR must be under /tmp"; exit 2;; esac
rm -rf $W; mkdir -p $W
P=0; F=0
ok(){ echo "PASS $*"; P=$((P + 1)); }
ko(){ echo "FAIL $*"; F=$((F + 1)); }
t(){ local name=$1; shift; if "$@" > $W/last.out 2>&1; then ok "$name"; else ko "$name"; sed 's/^/    /' $W/last.out | tail -25; fi; }
r0(){ echo qwen3_4b_Base_r0_a13_s$1_r3; }
waitpid(){ local p=$1 lim=$2 i; for i in $(seq $lim); do kill -0 $p 2>/dev/null || { wait $p; return $?; }; sleep 1; done; kill -TERM $p; return 124; }

# ======================================================================================= B: build
D1=$W/t1; $PY $SRC/dev/a13_dev_tree.py $D1 $SRC > $W/tree1.json 2> $W/tree1.err || { cat $W/tree1.err; echo "tree build failed"; exit 1; }
R1=$D1/rsi_rl; A=$R1/ext/a13_draft; X1=$R1/ext/a13_runs
t B1_build_idempotent env A13_PRIORITY=0 A13_DEV_ROOT=$D1 A13_DEV_HOST=hostA bash $A/a13_launch.sh build
t B2_invariants_and_determinism $PY - $R1 $A $W <<'EOF'
import json, os, subprocess, sys, hashlib
R, A, W = sys.argv[1:4]
man = json.load(open(f"{R}/ext/a13_runs/runs/a13_build_manifest.json"))
for s in range(5):
    Rl = open(f"{R}/dirB/runs/qwen3_4b_Base_rnd1_a6_s{s}_r3/admitted.jsonl", "rb").read().split(b"\n")[:-1]
    R0l = open(f"{R}/ext/a13_runs/runs/qwen3_4b_Base_r0_a13_s{s}_r3/admitted.jsonl", "rb").read().split(b"\n")[:-1]
    band = set(open(f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s{s}_r3/admitted.jsonl", "rb").read().split(b"\n")[:-1])
    Rr, R0r = [json.loads(x) for x in Rl], [json.loads(x) for x in R0l]
    dpos = [i for i, r in enumerate(Rr) if r["kind"] == "tracer_deleted"]
    assert len(R0l) == len(Rl) and dpos, s
    assert all(R0l[i] == Rl[i] for i in range(len(Rl)) if i not in dpos), s
    assert all(R0r[i]["kind"] == "proposed" and R0r[i]["id"] not in {r["id"] for r in Rr} for i in dpos), s
    assert not any(r["kind"] == "tracer_deleted" for r in R0r) and set(R0l) <= band, s
    assert len({r["id"] for r in R0r}) == len(R0r), s
    m = man["seeds"][str(s)]
    assert m["R0_sha256"] == hashlib.sha256(b"\n".join(R0l) + b"\n").hexdigest() and m["deleted_positions"] == dpos, s
    assert m["composition"]["R0"]["n"] == m["composition"]["R_rnd1"]["n"] == m["composition"]["S_cln1"]["n"], s
b = subprocess.run([sys.executable, f"{A}/a13_build.py", "--out_root", f"{W}/b2", "--root", R, "--dev"], capture_output=True, text=True)
assert b.returncode == 0, b.stdout + b.stderr
for s in range(5):
    n = f"qwen3_4b_Base_r0_a13_s{s}_r3/admitted.jsonl"
    assert open(f"{W}/b2/{n}", "rb").read() == open(f"{R}/ext/a13_runs/runs/{n}", "rb").read(), "not deterministic"
print("ok")
EOF
cp -r $X1/runs $W/b3; echo '{"id": "extra"}' >> $W/b3/$(r0 0)/admitted.jsonl
$PY $A/a13_build.py --out_root $W/b3 --root $R1 --dev > $W/b3.out 2>&1; rc=$?
[ $rc = 2 ] && grep -q "not overwritten" $W/b3.out && ok "B3_refuses_to_overwrite_a_different_set" || { ko "B3 rc=$rc"; cat $W/b3.out; }
cp -r $R1 $W/bad; f=$W/bad/dirB/runs/qwen3_4b_Base_rnd1_a6_s1_r3/admitted.jsonl
$PY -c "import sys; l=open(sys.argv[1]).readlines(); l[0],l[1]=l[1],l[0]; open(sys.argv[1],'w').writelines(l)" $f
$PY $A/a13_build.py --out_root $W/b4 --root $W/bad --dev > $W/b4.out 2>&1; rc=$?
[ $rc = 1 ] && grep -q "C2 seed 1" $W/b4.out && [ ! -e $W/b4/a13_build_manifest.json ] && ok "B4_C2_failure_writes_nothing" || { ko "B4 rc=$rc"; cat $W/b4.out; }
$PY $A/a13_build.py --out_root $W/b5 --root $R1 > $W/b5.out 2>&1; rc=$?
[ $rc = 1 ] && grep -q "C1 seed 0" $W/b5.out && ok "B5_registered_input_hashes_enforced" || { ko "B5 rc=$rc"; cat $W/b5.out; }

# ======================================================================================= L: two hosts
for h in hostA hostB; do for g in $(seq 0 7); do echo 0 > $D1/fakegpu/$h/$g.mem; done; done
echo 30000 > $D1/fakegpu/hostB/0.mem; echo 30000 > $D1/fakegpu/hostB/1.mem        # co-tenant memory on two GPUs of host B
touch $D1/inject/slow_$(r0 0) $D1/inject/crash_$(r0 4) $D1/inject/crash_plain_$(r0 1)
env A13_PRIORITY=0 A13_DEV_ROOT=$D1 A13_DEV_HOST=hostA A13_DEV_PREF="3 0 4" bash $A/a13_launch.sh run > $W/la.out 2>&1 & PA=$!
sleep 2
env A13_PRIORITY=0 A13_DEV_ROOT=$D1 A13_DEV_HOST=hostB A13_DEV_PREF="2 1" bash $A/a13_launch.sh run > $W/lb.out 2>&1 & PB=$!
# co-tenant: once seed 0 trains, put a foreign compute process on its GPU until the launcher yields
yh=""; for i in $(seq 240); do
  sc=$X1/runs/$(r0 0)/a13_train.json
  if [ -f $sc ] && [ "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $sc)" = running ] \
     && [ "$(cat $X1/runs/$(r0 0)/log.jsonl 2>/dev/null | grep -c '"loss"')" -ge 3 ]; then
    read yh yg <<< "$($PY -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["host"], m["history"][-1]["gpu"])' $sc)"
    echo "999999, GPU-dev-$yh-$yg" > $D1/fakegpu/$yh/$yg.apps; break; fi
  sleep 0.5; done
for i in $(seq 120); do grep -q "YIELD on gpu" $X1/a13_launch.log 2>/dev/null && break; sleep 1; done
rm -f $D1/fakegpu/*/*.apps
waitpid $PA 600; ra=$?; waitpid $PB 600; rb=$?
[ $ra = 0 ] && [ $rb = 0 ] && ok "L1_both_launchers_exit_0" || { ko "L1 ra=$ra rb=$rb"; tail -5 $W/la.out $W/lb.out; }
t L2_every_cell_complete_host_local $PY - $R1 <<'EOF'
import json, sys
R = sys.argv[1]; X = f"{R}/ext/a13_runs"; UM = f"{R}/dirA4/results/umwp"
st = lambda p: json.load(open(p))
n = lambda s: f"qwen3_4b_Base_r0_a13_s{s}_r3"
bad = []
for s in range(5):
    for p in (f"{X}/runs/{n(s)}/a13_train.json", f"{UM}/plain_{n(s)}.a13.json", f"{UM}/absfix_{n(s)}.a13.json", f"{X}/evals/{n(s)}_step60.a13.json"):
        if st(p)["state"] != "complete": bad.append(p)
    for a in ("cln1", "rnd1"):
        p = f"{UM}/absfix_qwen3_4b_Base_{a}_a6_s{s}_r3.a13.json"
        if st(p)["state"] != "complete": bad.append(p)
    tr = st(f"{X}/runs/{n(s)}/a13_train.json")
    evh = {st(f"{UM}/plain_{n(s)}.a13.json")["host"], st(f"{UM}/absfix_{n(s)}.a13.json")["host"], st(f"{X}/evals/{n(s)}_step60.a13.json")["host"]}
    if evh != {tr["host"]}: bad.append(f"evals of seed {s} not on its training host {tr['host']}: {evh}")
    if tr["start_ckpt_sha_ok"] is not True: bad.append(f"seed {s} start checkpoint not verified")
assert st(f"{X}/runs/{n(3)}/a13_train.json")["host"] == "hostA" and st(f"{X}/runs/{n(2)}/a13_train.json")["host"] == "hostB"
assert st(f"{X}/runs/{n(1)}/a13_train.json")["host"] == "hostB", "preferred seed 1 not on host B"
for a, s, h in (("cln1", 0, "hostA"), ("rnd1", 3, "hostA"), ("cln1", 2, "hostB"), ("rnd1", 4, "hostB")):
    assert st(f"{UM}/absfix_qwen3_4b_Base_{a}_a6_s{s}_r3.a13.json")["host"] == h
assert not bad, bad
print("ok")
EOF
t L3_yield_and_resume $PY - $R1 <<'EOF'
import json, sys
R = sys.argv[1]; X = f"{R}/ext/a13_runs"
tr = json.load(open(f"{X}/runs/qwen3_4b_Base_r0_a13_s0_r3/a13_train.json"))
log = [json.loads(l) for l in open(f"{X}/runs/qwen3_4b_Base_r0_a13_s0_r3/log.jsonl")]
assert tr["n_yields"] >= 1 and any("resumed_from_step" in x and x["resumed_from_step"] > 0 for x in log), (tr, log[-3:])
assert any(r["from_step"] > 0 and r["optimizer_restored"] is True for r in tr["resumes"]), tr["resumes"]
assert all(e.get("script_sha256") == tr["script_sha256"] for e in tr["history"]), "script sha256 not recorded per attempt"
assert "YIELD on gpu" in open(f"{X}/a13_launch.log").read()
print("ok")
EOF
t L4_crashed_training_retried $PY - $R1 <<'EOF'
import json, sys
X = f"{sys.argv[1]}/ext/a13_runs"
tr = json.load(open(f"{X}/runs/qwen3_4b_Base_r0_a13_s4_r3/a13_train.json"))
log = [json.loads(l) for l in open(f"{X}/runs/qwen3_4b_Base_r0_a13_s4_r3/log.jsonl")]
assert tr["n_attempts"] == 1 and any(x.get("resumed_from_step") == 0 for x in log), tr
assert tr["resumes"] == [{"from_step": 0, "optimizer_restored": False}], tr["resumes"]
print("ok")
EOF
t L5_partial_line_cut_and_resumed bash -c "grep -q 'cut a partial last line' $X1/a13_launch.log && [ \$($PY -c 'import json,sys; print(len({json.loads(l)[\"id\"] for l in open(sys.argv[1])}))' $R1/dirA4/results/umwp/plain_$(r0 1).jsonl) = 1200 ]"
t L6_cleanup bash -c "! ls $D1/shm_host*/gpu_single_* 2>/dev/null | grep -q . && [ -z \"\$(ls $X1/claims)\" ] && ! ls -d $D1/shm_host*/a13_pids_* 2>/dev/null | grep -q . && [ ! -e $D1/shm_hostA/a13_launch.pid ] && [ ! -e $D1/shm_hostB/a13_launch.pid ]"
t L7_tidy_moved_snapshots_and_archived_final bash -c "for s in 0 1 2 3 4; do n=qwen3_4b_Base_r0_a13_s\${s}_r3; ls -d $D1/arch_host*/\$n/step20 > /dev/null 2>&1 || exit 1; ls -d $D1/shm_host*/ckpt_b/\$n/step20 2>/dev/null && exit 1; ls -d $D1/shm_host*/ckpt_b/\$n/final > /dev/null || exit 1; ls $D1/arch_host*/\$n/final/tokenizer.json > /dev/null 2>&1 || exit 1; ls -d $D1/arch_host*/\$n/final.tmp 2>/dev/null && exit 1; done; exit 0"
t L8_busy_gpus_untouched $PY - $R1 <<'EOF'
import glob, json, sys
for p in glob.glob(f"{sys.argv[1]}/**/*.a13.json", recursive=True) + glob.glob(f"{sys.argv[1]}/ext/a13_runs/runs/*/a13_train.json"):
    for e in json.load(open(p)).get("history", []):
        assert not (e["host"] == "hostB" and str(e["gpu"]) in ("0", "1")), (p, e)
print("ok")
EOF
t L9_nothing_in_real_shm bash -c "! ls /dev/shm/a13_* 2>/dev/null | grep -q . && ! grep -ls '^a13 ' /dev/shm/gpu_single_* 2>/dev/null | grep -q ."

# L10 starvation: every GPU of host A is held by co-tenant processes (640 MiB, visible compute processes), host B has only
# GPUs 6 and 7 free. The preferences are the adverse ones of the first draft (A: 3 0 4, B: 2 1). Host A must not hold seeds 0
# and 4: host B trains them after PREF_DELAY; seed 3 (start checkpoint only on A) ends not_run_deadline without a sidecar;
# on host B no oabs cell starts before the last training started; B's co-tenant GPUs are never used.
D3=$W/t3; $PY $SRC/dev/a13_dev_tree.py $D3 $SRC > /dev/null || exit 1
R3=$D3/rsi_rl; A3=$R3/ext/a13_draft; X3=$R3/ext/a13_runs
for g in $(seq 0 7); do echo 640 > $D3/fakegpu/hostA/$g.mem; echo "99990$g, GPU-dev-hostA-$g" > $D3/fakegpu/hostA/$g.apps; done
for g in $(seq 0 7); do echo 0 > $D3/fakegpu/hostB/$g.mem; done
for g in 0 1 2 3 4 5; do echo 640 > $D3/fakegpu/hostB/$g.mem; echo "99991$g, GPU-dev-hostB-$g" > $D3/fakegpu/hostB/$g.apps; done
touch $D3/inject/slow_$(r0 2) $D3/inject/slow_$(r0 1)
env A13_PRIORITY=0 A13_DEV_ROOT=$D3 A13_DEV_HOST=hostA A13_DEV_PREF="3 0 4" A13_DEV_T_TRAIN_LAST=120 A13_DEV_T_EVAL_LAST=200 bash $A3/a13_launch.sh run > $W/ta.out 2>&1 & PA=$!
sleep 2
env A13_PRIORITY=0 A13_DEV_ROOT=$D3 A13_DEV_HOST=hostB A13_DEV_PREF="2 1" A13_DEV_T_TRAIN_LAST=120 A13_DEV_T_EVAL_LAST=200 bash $A3/a13_launch.sh run > $W/tb.out 2>&1 & PB=$!
waitpid $PA 420; ra=$?; waitpid $PB 420; rb=$?
t L10_starved_host_leaves_its_seeds_train_before_oabs $PY - $R3 $ra $rb $D3 <<'EOF'
import glob, json, os, re, sys
R, ra, rb, D = sys.argv[1:5]; X = f"{R}/ext/a13_runs"; UM = f"{R}/dirA4/results/umwp"
assert ra == "0" and rb == "0", (ra, rb)
n = lambda s: f"qwen3_4b_Base_r0_a13_s{s}_r3"
st = lambda p: json.load(open(p))
for s in (0, 1, 2, 4):
    for p in (f"{X}/runs/{n(s)}/a13_train.json", f"{UM}/plain_{n(s)}.a13.json", f"{UM}/absfix_{n(s)}.a13.json", f"{X}/evals/{n(s)}_step60.a13.json"):
        m = st(p); assert m["state"] == "complete" and m["host"] == "hostB", (p, m["state"], m["host"])
assert not os.path.exists(f"{X}/runs/{n(3)}/a13_train.json"), "seed 3 was never claimed, so no sidecar"
sa = open(f"{X}/status_hostA.tsv").read()
assert re.search(r"\ttrain:3\tnot_run_deadline\t", sa), sa[-600:]
for p in glob.glob(f"{R}/**/*.a13.json", recursive=True) + glob.glob(f"{X}/runs/*/a13_train.json"):
    for e in st(p).get("history", []):
        assert e["host"] == "hostB" and str(e["gpu"]) in ("6", "7", "-"), (p, e)
log = [l for l in open(f"{X}/a13_launch.log") if " hostB a13[" in l]
tr = [j for j, l in enumerate(log) if ": start train:" in l]
oa = [j for j, l in enumerate(log) if ": start oabs:" in l]
pl = [j for j, l in enumerate(log) if ": start plain:" in l]
assert len(tr) == 4 and oa and min(oa) > max(tr) and min(oa) > min(pl), (tr, oa, pl)
assert not os.listdir(f"{X}/claims") and not glob.glob(f"{D}/shm_host*/gpu_single_*") and not glob.glob(f"{D}/shm_host*/a13_pids_*")
print("ok: seeds 0 1 2 4 on hostB, seed 3 not_run_deadline on hostA, first oabs start after the last train start")
EOF

# ======================================================================================= S: STOP file, stop, restart
D2=$W/t2; $PY $SRC/dev/a13_dev_tree.py $D2 $SRC > /dev/null || exit 1
R2=$D2/rsi_rl; A2=$R2/ext/a13_draft; X2=$R2/ext/a13_runs
for h in hostA hostB; do for g in $(seq 0 7); do echo 0 > $D2/fakegpu/$h/$g.mem; done; done
touch $X2/STOP
timeout 60 env A13_PRIORITY=0 A13_DEV_ROOT=$D2 A13_DEV_HOST=hostB A13_DEV_PREF="2" bash $A2/a13_launch.sh run > $W/s0.out 2>&1; rc=$?
[ $rc = 0 ] && [ ! -e $X2/runs/$(r0 2)/a13_train.json ] && grep -q "no new cells" $X2/a13_launch.log && ok "S1_STOP_file_starts_nothing" || { ko "S1 rc=$rc"; tail $X2/a13_launch.log; }
rm -f $X2/STOP; touch $D2/inject/slow_$(r0 2)
env A13_PRIORITY=0 A13_DEV_ROOT=$D2 A13_DEV_HOST=hostB A13_DEV_PREF="2" A13_MAXPAR=1 bash $A2/a13_launch.sh run > $W/s1.out 2>&1 & PS=$!
for i in $(seq 240); do [ "$(cat $X2/runs/$(r0 2)/log.jsonl 2>/dev/null | grep -c '"loss"')" -ge 5 ] && break; sleep 0.5; done
env A13_PRIORITY=0 A13_DEV_ROOT=$D2 A13_DEV_HOST=hostB bash $A2/a13_launch.sh stop > $W/s2.out 2>&1
waitpid $PS 120; rc=$?
st=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $X2/runs/$(r0 2)/a13_train.json)
[ $rc = 143 ] && [ "$st" = stopped ] && [ -f $D2/shm_hostB/ckpt_b/$(r0 2)/resume/state.json ] && ! pgrep -f "$R2/dirB/code/b_train.py" > /dev/null \
  && ! ls $D2/shm_hostB/gpu_single_* 2>/dev/null | grep -q . && [ -z "$(ls $X2/claims)" ] && ok "S2_stop_during_training" || { ko "S2 rc=$rc state=$st"; tail -5 $X2/a13_launch.log; }
rm -f $D2/inject/slow_$(r0 2)
timeout 300 env A13_PRIORITY=0 A13_DEV_ROOT=$D2 A13_DEV_HOST=hostB A13_DEV_PREF="2" bash $A2/a13_launch.sh run > $W/s3.out 2>&1; rc=$?
t S3_restart_resumes_and_completes $PY - $R2 $rc <<'EOF'
import json, sys
R, rc = sys.argv[1], sys.argv[2]; X = f"{R}/ext/a13_runs"
assert rc == "0", rc
tr = json.load(open(f"{X}/runs/qwen3_4b_Base_r0_a13_s2_r3/a13_train.json"))
log = [json.loads(l) for l in open(f"{X}/runs/qwen3_4b_Base_r0_a13_s2_r3/log.jsonl")]
assert tr["state"] == "complete" and any(x.get("resumed_from_step", 0) > 0 for x in log), tr["state"]
assert json.load(open(f"{R}/dirA4/results/umwp/plain_qwen3_4b_Base_r0_a13_s2_r3.a13.json"))["state"] == "complete"
import os
assert not os.path.exists(f"{X}/runs/qwen3_4b_Base_r0_a13_s3_r3/a13_train.json"), "seed 3 has no start checkpoint on host B"
print("ok")
EOF

# ======================================================================================= D: read
RC=$D1/read_constants.json
t D1_read_ADDS_matches_independent_computation $PY - $A $R1 $RC $D1/targets.json <<'EOF'
import json, math, statistics, subprocess, sys
from scipy import stats
A, R, RC, TG = sys.argv[1:5]
p = subprocess.run([sys.executable, f"{A}/a13_read.py", "--root", R, "--constants", RC], capture_output=True, text=True)
assert p.returncode == 0, p.stdout + p.stderr
o = json.load(open(f"{R}/ext/a13_runs/a13_read.json"))
T = json.load(open(TG))
ca = lambda n: round(T[n]["CA"] * 600) / 600
S = [ca(f"qwen3_4b_Base_cln1_a6_s{s}_r3") for s in range(5)]
Rr = [ca(f"qwen3_4b_Base_rnd1_a6_s{s}_r3") for s in range(5)]
R0 = [ca(f"qwen3_4b_Base_r0_a13_s{s}_r3") for s in range(5)]
d = [a - b for a, b in zip(R0, S)]; m = statistics.mean(d); se = statistics.stdev(d) / math.sqrt(5); t = stats.t.ppf(0.975, 4)
assert o["check_ok"] and o["reading"] == "ADDS" and o["accepted_seeds"] == [0, 1, 2, 3, 4], (o["reading"], o["per_seed"])
assert abs(o["E_prop"]["mean"] - m) < 1e-12 and abs(o["E_prop"]["ci95"][0] - (m - t * se)) < 1e-12
dp = [a - b for a, b in zip(Rr, R0)]
assert abs(o["E_plant"]["mean"] - statistics.mean(dp)) < 1e-12 and o["identity_max_abs_error"] < 1e-12
e = o["paper"]["edits"]
assert len(e) == 8 and e[0]["replace"].startswith("The drifted curriculum thus adds about six points") and "in all five seeds" in e[0]["replace"]
assert "against 5.6 points with them;" in e[0]["replace"] and "putting the planted problems back changes it by" in e[0]["replace"]
assert any(x["file"] == "01_introduction.tex" and "and still" in x["replace"] for x in e)
assert "Same size, no planted (5)" in e[1]["replace"] and o["descriptive"]["bench_complete"] is True
assert all("resumes" in o["per_seed"][str(s)]["train"] for s in range(5))
assert o["descriptive"]["absfix_seeds_all_arms"] == [0, 1, 2, 3, 4] and "appendix_absfix_sentence" in o["paper"]
print(e[0]["replace"]); print(e[1]["replace"].split("\n")[1])
EOF
sha_before=$(sha256sum $X1/a13_read.json | cut -c1-64)
$PY $A/a13_read.py --root $R1 --constants $RC > $W/d2.out 2>&1; rc=$?
[ $rc = 3 ] && [ "$(sha256sum $X1/a13_read.json | cut -c1-64)" = $sha_before ] && ok "D2_second_read_refused" || ko "D2 rc=$rc"
# rewritten versions of the tree for the other readings. The tree is rewritten IN PLACE (same absolute paths, as the
# config.json files and sidecars record them) from a pristine copy, and restored afterwards; the rewrite also updates the
# sidecar's output hash, as a real complete dump would have.
cp -a $R1 $W/r1_pristine
variant(){ local name=$1 spec=$2 rc=${3:-$RC}; rm -rf $R1 $W/v_$name; cp -a $W/r1_pristine $R1; mkdir -p $W/v_$name; rm -f $X1/a13_read.json
  $PY - $R1 "$spec" <<'EOF'
import hashlib, json, sys
R, spec = sys.argv[1], json.loads(sys.argv[2]); UM = f"{R}/dirA4/results/umwp"
S_CA = [0.600, 0.450, 0.620, 0.700, 0.660]
for s, dlt in enumerate(spec.get("r0_delta", [])):
    n = f"qwen3_4b_Base_r0_a13_s{s}_r3"; f = f"{UM}/plain_{n}.jsonl"
    rows = [json.loads(l) for l in open(f)]
    k = round((S_CA[s] + dlt) * 600); j = 0
    for r in rows:
        if r["answerable"] is False:
            r["mode"], r["p_mode"] = (7, 0.875) if j < k else (None, 0.625); j += 1
    open(f, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    sc = f"{UM}/plain_{n}.a13.json"; m = json.load(open(sc)); m["output_sha256"] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    json.dump(m, open(sc, "w"))
for s in spec.get("drop_plain_sidecar", []):
    import os; os.remove(f"{UM}/plain_qwen3_4b_Base_r0_a13_s{s}_r3.a13.json")
if "attr_shift" in spec:
    p = f"{R}/paper_arr/analyses/attribution.json"; a = json.load(open(p)); a["paired"]["cln1-rnd1:CA"]["mean"] += spec["attr_shift"]
    json.dump(a, open(p, "w"))
for s in spec.get("bad_config", []):
    p = f"{R}/ext/a13_runs/runs/qwen3_4b_Base_r0_a13_s{s}_r3/config.json"; c = json.load(open(p)); c["lr"] = 2e-6; json.dump(c, open(p, "w"))
for s in spec.get("edit_dump", []):
    f = f"{UM}/plain_qwen3_4b_Base_r0_a13_s{s}_r3.jsonl"; open(f, "a").write(open(f).readline())
EOF
  $PY $A/a13_read.py --root $R1 --constants $rc --out $W/v_$name/o.json > $W/v_$name/out.txt 2>&1; }
field(){ $PY -c 'import json,sys; o=json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$@"; }
cp $X1/a13_read.json $W/d1_adds.json
variant nd '{"r0_delta": [0.05, -0.04, 0.03, -0.02, 0.01]}'
[ "$(field $W/v_nd/o.json 'o["reading"]')" = "NOT DETECTED" ] && [ "$(field $W/v_nd/o.json 'o["not_detected_subcase"]')" = "estimate > 0" ] && grep -q "an interval that includes zero; higher in three of five seeds" $W/v_nd/o.json && ! grep -q "reaches" $W/v_nd/o.json && [ "$(field $W/v_nd/o.json 'len(o["paper"]["edits"])')" = 7 ] && ok "D3_NOT_DETECTED_positive_estimate" || { ko D3; cat $W/v_nd/out.txt; }
variant nd0 '{"r0_delta": [-0.05, 0.04, -0.03, 0.02, -0.01]}'
[ "$(field $W/v_nd0/o.json 'o["reading"]')" = "NOT DETECTED" ] && [ "$(field $W/v_nd0/o.json 'o["not_detected_subcase"]')" = "estimate <= 0" ] && [ "$(field $W/v_nd0/o.json 'len(o["paper"]["edits"])')" = 9 ] && grep -q "not detected without its planted problems" $W/v_nd0/o.json && grep -q "partly through the ill-posed curriculum" $W/v_nd0/o.json && grep -q "by \$-\$0.6 points on average" $W/v_nd0/o.json && ok "D3b_NOT_DETECTED_nonpositive_estimate_qualifies_claims_and_conclusion" || { ko D3b; cat $W/v_nd0/out.txt; }
variant red '{"r0_delta": [-0.04, -0.05, -0.03, -0.035, -0.045]}'
[ "$(field $W/v_red/o.json 'o["reading"]')" = "REDUCES" ] && [ "$(field $W/v_red/o.json 'len(o["paper"]["edits"])')" = 12 ] && grep -q "in all five seeds, by 4.0 points" $W/v_red/o.json && grep -q "mostly by itself rather than through the drift" $W/v_red/o.json && grep -q "premise-deleted problems we plant among its candidates" $W/v_red/o.json && ok "D4_REDUCES_with_heading_abstract_intro_claims_conclusion_edits" || { ko D4; cat $W/v_red/out.txt; }
variant nr '{"drop_plain_sidecar": [0, 1]}'
[ "$(field $W/v_nr/o.json 'o["reading"]')" = "NOT RUN" ] && grep -q "3 of 5 seeds accepted" $W/v_nr/o.json && grep -q "three of five runs finished in time" $W/v_nr/o.json && [ "$(field $W/v_nr/o.json 'len(o["paper"]["edits"])')" = 2 ] && ok "D5_NOT_RUN_three_seeds" || { ko D5; cat $W/v_nr/out.txt; }
variant chk '{"attr_shift": 0.001}'
[ "$(field $W/v_chk/o.json 'o["reading"]')" = "NOT RUN" ] && [ "$(field $W/v_chk/o.json 'o["check_ok"]')" = False ] && grep -q "could not be reproduced from its dumps" $W/v_chk/o.json && ok "D6_failed_RS_check_blocks_reading" || { ko D6; cat $W/v_chk/out.txt; }
$PY -c 'import json,sys; c=json.load(open(sys.argv[1])); c["old_sha256"]={"paper_arr/analyses/attribution.json": "0"*64}; json.dump(c, open(sys.argv[2], "w"))' $RC $W/rc_changed.json
variant inp '{}' $W/rc_changed.json
[ "$(field $W/v_inp/o.json 'o["reading"]')" = "NOT RUN" ] && [ "$(field $W/v_inp/o.json 'o["check_R_minus_S"]["equal_to_stored"]')" = True ] && grep -q "an input of the stored same-size contrast changed after registration" $W/v_inp/o.json && ! grep -q "could not be reproduced" $W/v_inp/o.json && ok "D6b_changed_input_of_RS_check_named_as_such" || { ko D6b; cat $W/v_inp/out.txt; }
variant cfg2 '{"bad_config": [1, 2]}'
[ "$(field $W/v_cfg2/o.json 'o["reading"]')" = "NOT RUN" ] && grep -q "three of five runs finished in time and passed the registered checks, four were required" $W/v_cfg2/o.json && ok "D5b_NOT_RUN_after_rejected_configs" || { ko D5b; cat $W/v_cfg2/out.txt; }
variant cfg '{"bad_config": [2]}'
[ "$(field $W/v_cfg/o.json 'o["accepted_seeds"]')" = "[0, 1, 3, 4]" ] && grep -q "config differs from rnd1" $W/v_cfg/o.json && grep -q "in all four seeds" $W/v_cfg/o.json && grep -q "points with them on the same four seeds" $W/v_cfg/o.json && grep -q "same-size contrast on these four seeds (" $W/v_cfg/o.json && ! grep -q "against 5.6 points" $W/v_cfg/o.json && ok "D7_rejected_config_four_seeds_same_seed_contrast" || { ko D7; cat $W/v_cfg/out.txt; }
variant dump '{"edit_dump": [3]}'
[ "$(field $W/v_dump/o.json 'o["accepted_seeds"]')" = "[0, 1, 2, 4]" ] && grep -q "output changed after completion" $W/v_dump/o.json && ok "D8_edited_dump_rejected" || { ko D8; cat $W/v_dump/out.txt; }
variant lo '{"r0_delta": [0.0, 0.005, 0.0166667, 0.0166667, 0.025]}'
[ "$(field $W/v_lo/o.json 'o["reading"]')" = "ADDS" ] && grep -q 'CI \$\[0.02, ' $W/v_lo/o.json && grep -q 'scriptsize\$\[-[0-9.]*, -0.02\]' $W/v_lo/o.json && ok "D10_CI_bound_rounding_to_zero_printed_with_two_decimals" || { ko D10; cat $W/v_lo/out.txt; grep -o 'CI [^;]*' $W/v_lo/o.json | head -3; }
rm -rf $R1; cp -a $W/r1_pristine $R1                                          # restore the tree as the launcher left it
t D9_tree_restored bash -c "cmp -s $X1/a13_read.json $W/d1_adds.json"

# ======================================================================================= P: paper edits (optional)
# PAPER = a copy of paper_arr_oct2026/sections (only .tex files; default: none, then P is skipped). For each reading, every
# find string must occur exactly once in its file at the time it is applied, the replacement must keep the braces and $
# balanced as the find string does, and after all edits every replacement text must be in the file.
if [ -n "$PAPER" ] && [ -d "$PAPER" ]; then
  for v in d1_adds.json v_nd/o.json v_nd0/o.json v_red/o.json v_nr/o.json v_cfg/o.json v_lo/o.json v_inp/o.json; do
    t "P_$(echo $v | tr '/.' '__')_edits_apply" $PY - "$PAPER" "$W/$v" "$W/paper_$(echo $v | tr '/.' '__')" <<'EOF'
import json, os, shutil, sys
src, oj, dst = sys.argv[1:4]
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
o = json.load(open(oj)); E = o["paper"]["edits"]
assert E, "no edits"
bal = lambda s: (s.count("{") - s.count("}"), s.count("$") % 2)
for e in E:
    p = os.path.join(dst, e["file"]); t = open(p, encoding="utf-8").read()
    assert t.count(e["find"]) == 1, (e["file"], t.count(e["find"]), e["find"][:80])
    assert bal(e["find"]) == bal(e["replace"]), ("unbalanced replacement", e["replace"][:120])
    assert "?" not in e["replace"], ("unrendered placeholder", e["replace"][:160])
    open(p, "w", encoding="utf-8").write(t.replace(e["find"], e["replace"]))
for e in E:
    assert e["replace"] in open(os.path.join(dst, e["file"]), encoding="utf-8").read(), e["replace"][:80]
print(o["reading"], len(E), "edits applied")
EOF
  done
else
  echo "SKIP P (no PAPER directory given)"
fi

# ======================================================================================= X: real data, read-only
$PY $SRC/a13_read.py --check_only > $W/x1.out 2>&1; rc=$?
[ $rc = 0 ] && grep -q '"check_ok": true' $W/x1.out && ok "X1_real_R_minus_S_reproduced_-5.6_[-9.0,-2.1]" || { ko "X1 rc=$rc"; cat $W/x1.out; }
$PY $SRC/a13_build.py --out_root $W/real_dry > $W/x2.out 2>&1; rc=$?
[ $rc = 0 ] && grep -q '"ok": true' $W/x2.out && ok "X2_real_dry_build_registered_hashes" || { ko "X2 rc=$rc"; cat $W/x2.out; }
env -u A13_DEV_ROOT bash $SRC/a13_launch.sh check > $W/x3.out 2>&1; rc=$?
[ $rc = 0 ] && grep -q "packages as registered" $W/x3.out && ok "X3_real_launcher_check_readonly" || { ko "X3 rc=$rc"; cat $W/x3.out; }
env -u A13_DEV_ROOT bash $SRC/a13_launch.sh locate > $W/x4.out 2>&1; rc=$?
[ $rc = 0 ] && [ "$(wc -l < $W/x4.out)" = 30 ] && ok "X4_real_launcher_locate_readonly" || { ko "X4 rc=$rc"; cat $W/x4.out; }
[ ! -e /home/tione/notebook/rsi_rl/ext/a13_runs ] && ok "X5_no_real_run_directory_created" || ko "X5 ext/a13_runs exists"

# ======================================================================================= L11: priority mode (lead author,
# 2026-09-27): a foreign process on the training GPU is logged and killed; the job does not yield
D4=$W/t4; $PY $SRC/dev/a13_dev_tree.py $D4 $SRC > /dev/null || exit 1
R4=$D4/rsi_rl; A4=$R4/ext/a13_draft; X4=$R4/ext/a13_runs
for h in hostA hostB; do for g in $(seq 0 7); do echo 0 > $D4/fakegpu/$h/$g.mem; done; done
touch $D4/inject/slow_$(r0 0)
sleep 600 & SP=$!
env A13_PRIORITY=1 A13_DEV_ROOT=$D4 A13_DEV_HOST=hostA A13_DEV_PREF="0" A13_DEV_T_TRAIN_LAST=120 A13_DEV_T_EVAL_LAST=200 bash $A4/a13_launch.sh run > $W/pa.out 2>&1 & PA=$!
for i in $(seq 240); do
  sc=$X4/runs/$(r0 0)/a13_train.json
  if [ -f $sc ] && [ "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' $sc)" = running ]; then
    read yh yg <<< "$($PY -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["host"], m["history"][-1]["gpu"])' $sc)"
    echo "$SP, GPU-dev-$yh-$yg" > $D4/fakegpu/$yh/$yg.apps; break; fi
  sleep 0.5; done
for i in $(seq 180); do grep -q "PRIORITY: killing foreign pid $SP" $X4/a13_launch.log 2>/dev/null && break; sleep 1; done
sleep 12; rm -f $D4/fakegpu/*/*.apps
waitpid $PA 600; ra=$?
if kill -0 $SP 2>/dev/null; then ko "L11 foreign process still alive"; kill $SP; fi
t L11_priority_kills_foreign_no_yield $PY - $R4 $SP $ra <<'EOF'
import json, sys
R, sp, ra = sys.argv[1:4]; X = f"{R}/ext/a13_runs"
assert ra == "0", ra
tr = json.load(open(f"{X}/runs/qwen3_4b_Base_r0_a13_s0_r3/a13_train.json"))
assert tr["state"] == "complete" and tr.get("n_yields", 0) == 0, tr
log = open(f"{X}/a13_launch.log").read()
assert f"PRIORITY: killing foreign pid {sp}" in log and "YIELD on gpu" not in log, log[-2000:]
k = open(f"{X}/killed_foreign.log").read()
assert f"pid {sp}" in k and "sleep 600" in k, k
print("ok")
EOF

echo "== $P passed, $F failed"
[ $F = 0 ]
