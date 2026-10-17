#!/bin/bash
# CPU-only tests of the A7-5 design scripts in a scratch dir outside rsi_rl. Starts no GPU process (nvidia-smi queries only)
# and kills only its own test processes (sleeps tagged rzA75_dtest*); GPU markers are redirected to scratch directories.
# DT_D may point at a staged copy of the design files (default: the design directory itself). T13 and T14 run staged
# copies of the waiter, driver, smoke and stop scripts with the host checks stubbed, markers in scratch and a guard that ends
# the driver before rz_env.sh, so no stage can start.
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
W=/run/ti/rzero_design_scratch; D=${DT_D:-/home/tione/notebook/rsi_rl/ext/rzero_design}; T=$W/dt; rm -rf $T; mkdir -p $T
T0=1790233800
echo "== T0 syntax: bash -n on every shell script, py_compile on every python file, no CR anywhere"
for f in $D/*.sh $D/dev/*.sh; do bash -n $f && echo "bash -n ok $(basename $f)"; done
for f in $D/*.py $D/dev/*.py; do /usr/bin/python3 -c "import py_compile, sys; py_compile.compile(sys.argv[1], cfile='$T/pyc', doraise=True)" $f && echo "py_compile ok $(basename $f)"; done
echo "files with CR: $(grep -lI $'\r' $D/* $D/dev/* 2>/dev/null | wc -l)"

echo "== T1 kill by environment tag (grep -z over /proc/*/environ; waits until gone)"
RZ_A75_TAG=rzA75_dtest bash -c 'sleep 300 & bash -c "sleep 300" & wait' &
RZ_A75_TAG=rzA75_dtestX sleep 300 & V=$!
sleep 45 & keep=$!
sleep 1
( TAG=x; STORAGE_PATH=$T; LEDGER=$T/ledger; source $D/rz_common.sh; echo "tagged before: $(tagged_pids rzA75_dtest | wc -l)"; t=$(date +%s)
  kill_tag rzA75_dtest; echo "tagged after: $(tagged_pids rzA75_dtest | wc -l) ($(( $(date +%s) - t )) s)" )
kill -0 $keep 2>/dev/null && echo "untagged sleep alive" || echo "untagged sleep DEAD?!"
kill -0 $V 2>/dev/null && echo "sleep with another tag alive" || echo "sleep with another tag DEAD?!"; kill $V $keep; wait 2>/dev/null

echo "== T2 host_check and preflight on this busy host (must refuse; build and markers redirected to scratch)"
FRZ=$T/rz; mkdir -p $FRZ/R-Zero $T/mk2; sha256sum < $D/rzero_A7_5.patch | cut -c1-64 > $FRZ/R-Zero/.a75_patched
( RZ=$FRZ; X=$T/x; TAG=rzA75_dtest; STORAGE_PATH=$T; LEDGER=$T/ledger; source $D/rz_common.sh; MK=$T/mk2
  echo bad > $FRZ/R-Zero/.a75_tree_sha256; echo "wrong tree: $(host_check)"
  echo $A75_TREE > $FRZ/R-Zero/.a75_tree_sha256; echo "headroom $(headroom_gib) GiB; our job scripts alive: $(launchers_alive | wc -l)"
  echo "host_check: $(host_check | cut -c1-150)"; preflight; echo "preflight passed?!" ); echo "preflight exit $?"; tail -1 $T/ledger | cut -c1-160
( RZ=$FRZ; D2=$T/d2; mkdir -p $D2; cp $D/rzero_A7_5.patch $D2/; echo "# changed" >> $D2/rzero_A7_5.patch; source $D/rz_common.sh; D=$D2; echo "other patch: $(host_check)" )

echo "== T3 markers: stale rule, takeover, smoke hand-over, atomic claim (stubbed GPU queries, markers in scratch)"
( RZ=$FRZ; X=$T/x3; mkdir -p $X; TAG=rzA75_s0; STORAGE_PATH=$T/sp3; LEDGER=$T/ledger3; mkdir -p $STORAGE_PATH; source $D/rz_common.sh; MK=$T/mk3; mkdir -p $MK
  gpu_apps(){ :; }; gpu_mem(){ for g in 0 1 2 3 4 5 6 7; do echo "$g, GPU-$g, 5"; done; }; launchers_alive(){ :; }; port_open(){ return 1; }
  echo "a) free host: '$(host_check)'"
  touch $MK/gpu_single_3; echo "b) fresh empty marker: '$(host_check)'"
  touch -d '-11 min' $MK/gpu_single_3; echo "c) 11-min-old empty marker: '$(host_check)'; still there: $([ -e $MK/gpu_single_3 ] && echo yes || echo no)"
  echo "rzA75_s0 999999 1" > $MK/gpu_single_4; echo "d) own leftover, driver dead: '$(host_check)'"
  echo "rzA75_smoke 999998 1" > $MK/gpu_single_5; echo "e) smoke marker, no PASS: '$(host_check)'"
  echo "x smoke PASSED epoch=$(( $(date +%s) - 4000 )) host=$(hostname) stack=abc y" > $X/ledger_smoke.log; echo "f) smoke marker, PASS 67 min ago: '$(host_check)'"
  echo "x smoke PASSED epoch=$(date +%s) host=otherhost stack=abc y" > $X/ledger_smoke.log; echo "g) smoke marker, PASS now on another host: '$(host_check)'"
  echo "x smoke PASSED epoch=$(date +%s) host=$(hostname) stack=abc y" > $X/ledger_smoke.log; echo "h) smoke marker, PASS now on this host: '$(host_check)'"
  echo "rzA75_s1 999997 1" > $MK/gpu_single_6; echo "i) another seed's fresh marker: '$(host_check)'"; rm -f $MK/gpu_single_6
  ( claim; echo "j) claim: $(cat $MK/gpu_single_* | cut -d' ' -f1 | sort | uniq -c | tr -s ' \n' ' ')"; exit 0 ); echo "   after exit: $(ls $MK | wc -l) markers left"
  echo "rzA75_smoke 999998 1" > $MK/gpu_single_5; : > $MK/gpu_single_2
  ( claim; echo "NOT REACHED" ); echo "k) claim with a foreign marker on GPU 2: exit $?, markers left: $(ls $MK | tr '\n' ' ')"; tail -1 $LEDGER | cut -c1-150
  rm -f $MK/*; ( TAG=rzA75_smoke; claim; KEEP_MARKERS=1; exit 0 ); echo "l) smoke claim with KEEP_MARKERS=1: $(ls $MK | wc -l) kept ('$(cut -d' ' -f1 $MK/gpu_single_0)')"
  echo "m) stale-marker ledger lines: $(grep -c 'stale marker' $LEDGER)" )

echo "== T4 foreign GPU use (stubbed nvidia-smi rows, real /proc environ)"
RZ_A75_TAG=rzA75_dtest4 sleep 120 & OURS=$!
sleep 120 & THEIRS=$!
sleep 1
( TAG=rzA75_dtest4; source $D/rz_common.sh
  gpu_apps(){ echo "$OURS, GPU-a"; }; gpu_mem(){ echo "0, GPU-a, 30000"; echo "1, GPU-b, 900"; }; echo "a) ours only: '$(foreign_gpu)'"
  gpu_apps(){ echo "$OURS, GPU-a"; echo "$THEIRS, GPU-b"; }; gpu_mem(){ echo "0, GPU-a, 30000"; echo "1, GPU-b, 30000"; }; echo "b) untagged pid: '$(foreign_gpu | tr '\n' ';')'"
  gpu_apps(){ echo "$OURS, GPU-a"; echo "4194000, GPU-b"; }; echo "c) pid not visible here: '$(foreign_gpu | tr '\n' ';')'"
  gpu_apps(){ echo "$OURS, GPU-a"; }; gpu_mem(){ echo "0, GPU-a, 30000"; echo "1, GPU-b, 4000"; }; echo "d) memory, none of ours on the GPU: '$(foreign_gpu | tr '\n' ';')'"
  ( sleep 0.3 & exec sleep 30 ) & ZP=$!; sleep 1; Z=$(ps -o pid= --ppid $ZP | tr -d ' ' | head -1)
  gpu_apps(){ echo "$OURS, GPU-a"; echo "$Z, GPU-b"; }; gpu_mem(){ echo "0, GPU-a, 30000"; echo "1, GPU-b, 30000"; }
  echo "f) exiting process of this container (review item 1; a zombie, state $(awk '{print $3}' /proc/$Z/stat 2>/dev/null)): '$(foreign_gpu | tr '\n' ';')'"
  env -i /bin/sleep 30 & EP=$!; sleep 0.5; gpu_apps(){ echo "$OURS, GPU-a"; echo "$EP, GPU-b"; }
  echo "g) live process with an empty environment: '$(foreign_gpu | tr '\n' ';')'"; kill $ZP $EP 2>/dev/null
  unset -f gpu_apps gpu_mem; source $D/rz_common.sh; echo "e) real nvidia-smi on this busy host (nothing carries this test tag): $(foreign_gpu | wc -l) findings" )
kill $OURS $THEIRS; wait 2>/dev/null

echo "== T5 monitor: foreign use -> exit 6, memory headroom -> exit 7; the exit handler extracts (review item 5; two fake drivers; about 2 min)"
for c in 6 7; do
cat > $T/drv$c.sh <<EOF
TAG=rzA75_dtest5_$c; STORAGE_PATH=$T/drv$c; LEDGER=$T/ledger5; D=$D; RZ=$T/rz; X=$T/x; mkdir -p \$STORAGE_PATH
source $D/rz_common.sh; MK=$T/mk5_$c; mkdir -p \$MK
gpu_apps(){ :; }; gpu_mem(){ :; }
if [ $c = 6 ]; then foreign_gpu(){ [ -f $T/flag5 ] && echo "pid 1 (fake) on GPU-x"; }; fi
if [ $c = 7 ]; then headroom_gib(){ echo 42; }; fi
claim; export RZ_A75_TAG=\$TAG
extract(){ echo ran >> \$STORAGE_PATH/extract.log; }
start_monitor; trap 'cleanup; extract' EXIT
sleep 1000; stop_check; echo "NOT REACHED"
EOF
( t=$(date +%s); bash $T/drv$c.sh; echo "case $c: driver exit $? after $(( $(date +%s) - t )) s, markers left $(ls $T/mk5_$c | wc -l), stop_code '$(cat $T/drv$c/stop_code)', tagged left $(grep -lzx "RZ_A75_TAG=rzA75_dtest5_$c" /proc/[0-9]*/environ 2>/dev/null | wc -l), exit-handler extraction: $(cat $T/drv$c/extract.log 2>/dev/null)" ) &
done
touch $T/flag5; wait; grep -h monitor $T/ledger5 | cut -c1-160

echo "== T6 seed ids and host affinity"
( X=$T/x6; mkdir -p $X; TAG=t6; STORAGE_PATH=$T; LEDGER=$T/ledger6; source $D/rz_common.sh
  seed_ok 0 && echo "a) seed 0, never started: ok"
  echo "otherhost /run/ti/rzero/s0 1" > $X/s0.host; echo "b) $(seed_ok 0)"
  echo "$(hostname) /run/ti/rzero/s0 1" > $X/s0.host; seed_ok 0 && echo "c) seed 0 on its own host: ok"
  echo "d) $(seed_ok 2)"; echo "1 h reason" > $X/s0.abandoned; echo "e) $(seed_ok 0)"; seed_ok 2 && echo "f) seed 2 after one abandoned id: ok"; echo "g) $(seed_ok 3)"
  STORAGE_PATH=$T/s1; mkdir -p $STORAGE_PATH; claim_host 1; echo "h) claim_host 1 -> '$(cut -d' ' -f1-2 $X/s1.host)'"; ( claim_host 1 ) && echo "i) re-claim on the same host: ok"
  echo "otherhost x 1" > $X/s3.host; ( claim_host 3 ); echo "j) claim of a seed owned elsewhere: exit $?" )

echo "== T7 resume fallback and the admitted-set finding (synthetic logs)"
( S=0; ABBR=rzero_qwen3_4b_Base_s0; X=$T/x7; STORAGE_PATH=$T/sp7; M=$STORAGE_PATH/models; GQ=$STORAGE_PATH/generated_question; TAG=t7; LEDGER=$T/ledger7
  mkdir -p $M $GQ $X/s0; : > $STORAGE_PATH/main.log; source $D/rz_common.sh
  mkstage(){ rm -rf $M/*; mkdir -p $M/${ABBR}_questioner_v1/global_step_3/actor; echo 3 > $M/${ABBR}_questioner_v1/latest_global_step.txt; }
  mkstage; echo "cur_stage $(cur_stage), tracker $(tracker $(cur_stage))"
  off=$(stat -c %s $STORAGE_PATH/main.log)
  printf 'Load from checkpoint: /x/global_step_3.\nTraceback (most recent call last):\n  File "/run/ti/rzero/R-Zero/verl/trainer/ray_trainer.py", line 431, in _load_checkpoint\nRuntimeError: bad\n' >> $STORAGE_PATH/main.log
  resume_fallback ${ABBR}_questioner_v1 3 $off; echo "a) traceback through _load_checkpoint: $(ls $M | tr '\n' ' '); left inside: $(ls $M/*.discarded_* | tr '\n' ' ')"
  mkstage; for n in 1 2; do off=$(stat -c %s $STORAGE_PATH/main.log)
    printf 'Load from checkpoint: /x/global_step_3.\nstep 4\nTraceback (most recent call last):\n  File "x.py", line 1, in fit\nValueError: boom\n' >> $STORAGE_PATH/main.log
    resume_fallback ${ABBR}_questioner_v1 3 $off; echo "b$n) other traceback after the load, attempt $n: $(ls $M | tr '\n' ' ')"; done
  mkstage; off=$(stat -c %s $STORAGE_PATH/main.log); printf 'Load from checkpoint: /x/global_step_3.\nTraceback (most recent call last):\n  File "x.py", line 1, in load_checkpoint\n' >> $STORAGE_PATH/main.log
  echo 4 > $M/${ABBR}_questioner_v1/latest_global_step.txt; resume_fallback ${ABBR}_questioner_v1 3 $off; echo "c) a newer checkpoint was saved: $(ls $M | tr '\n' ' ')"
  resume_fallback ${ABBR}_solver_v1 0 0; echo "d) fresh stage (k = 0): $(ls $M | wc -l) dir"
  mk(){ /usr/bin/python3 -c "import pyarrow as pa, pyarrow.parquet as pq; pq.write_table(pa.table({'problem': ['q'] * $1, 'answer': ['1'] * $1, 'score': [0.5] * $1}), '$GQ/${ABBR}_solver_v1_admitted.parquet')"; }
  mk 300; small_admitted 0 && echo "e) 300 rows: finding $(cat $X/s0/finding_admitted_lt512.json)"; rm -f $X/s0/finding*
  mk 900; off=$(stat -c %s $STORAGE_PATH/main.log); small_admitted $off || echo "f) 900 rows, no assertion: no finding"
  printf '    assert len(train_dataloader) >= 1\nAssertionError\n' >> $STORAGE_PATH/main.log
  small_admitted $off && echo "g) 900 rows but the assertion fired (overlong filter): finding $(grep -o '"n_admitted": [0-9]*' $X/s0/finding_admitted_lt512.json)"
  grep -h "resume fallback\|finding" $LEDGER | cut -c1-170 )

echo "== T8 UMWP dump helpers (partial last line, unique parseable ids)"
( source $D/rz_common.sh; f=$T/u.jsonl; for i in $(seq 1 1199); do echo "{\"id\": $i}"; done > $f; printf '{"id": 1200, "answ' >> $f
  cp $f $T/u_old.jsonl; echo '{"id": 1200}' >> $T/u_old.jsonl; echo "old behaviour (append onto the partial line): ids $(n_ids $T/u_old.jsonl)"
  trim_partial $f; echo "after trim: lines $(wc -l < $f), ids $(n_ids $f)"; echo '{"id": 1200}' >> $f; echo "resumed append after trim: ids $(n_ids $f)"
  trim_partial $f; printf '{"id": 5}\n' >> $f; echo "no-op trim, a duplicate id added: ids $(n_ids $f); missing file: $(n_ids $T/none.jsonl)" )

echo "== T9 setup, paths redirected to scratch (CPU and network only)"
sed -e "s#/run/ti/rzero\b#$T/rz#g" -e "s#^RZ=.*#RZ=$T/rz; D=$D; E=$T/e; X=$T/x#" -e "s#^SRC=.*#SRC=/home/tione/notebook/rsi_rl/ext/R-Zero_5699329; DATA=$T/x/rzero_data; export HF_HOME=\$RZ/hf#" \
    -e "s#/home/tione/notebook/rsi_rl/ext/rzero_data#$T/x/rzero_data#g" $D/rz_setup.sh > $T/setup_t.sh
grep -n "^RZ=\|^SRC=\|to_parquet\|startswith\|data_files" $T/setup_t.sh | cut -c1-150
rm -rf $FRZ; timeout 1500 bash $T/setup_t.sh > $T/setup.log 2>&1; echo "setup exit $?"; grep -v -i "warn\|INFO" $T/setup.log | tail -5 | cut -c1-250
echo "patched $(cat $T/rz/R-Zero/.a75_patched) tree $(cat $T/rz/R-Zero/.a75_tree_sha256)"
t=$(date +%s); bash $T/setup_t.sh > $T/setup2.log 2>&1; echo "second run (no rebuild): exit $? in $(( $(date +%s) - t )) s; rebuilt: $(grep -c 'patching file' $T/setup2.log)"
rm -rf "${T:?}"/rz/R-Zero; env LC_ALL=C LANG=C bash $T/setup_t.sh > $T/setup4.log 2>&1
echo "rebuild with the caller in the C locale (review item 6): exit $?; rebuilt: $(grep -c 'patching file' $T/setup4.log); tree $(cat $T/rz/R-Zero/.a75_tree_sha256 2>/dev/null)"
D3=$T/d3; mkdir -p $D3; cp $D/*.sh $D/*.py $D/rzero_A7_5.patch $D3/; sed -i 's/# A7-5 (P9): no wandb/# A7-5 (P9): no wandb (changed)/' $D3/rzero_A7_5.patch
sed "s#D=$D;#D=$D3;#" $T/setup_t.sh > $T/setup_t3.sh; bash $T/setup_t3.sh > $T/setup3.log 2>&1; echo "changed patch: exit $?; $(grep FATAL $T/setup3.log | cut -c1-160); working copy kept: $(cat $T/rz/R-Zero/.a75_tree_sha256 | cut -c1-16), patched $(cat $T/rz/R-Zero/.a75_patched | cut -c1-16)"
grep -n "blinker\|Flask\|nltk\|scikit\|stopit\|jsonlines\|^torch=\|^vllm=\|^numpy=\|^ray=" $T/rz/venv_freeze.txt | head -12 | tr '\n' ' '; echo

echo "== T10 extract + read on synthetic data"
FR=$T/fakeR; FZ=$T/fakeRZ; XR=$FR/ext/rzero_runs; mkdir -p $FR/dirB/runs $FR/dirA4/results/umwp $XR $FZ/s0/generated_question $FZ/s1/generated_question
sed -e "s#^R = \"/home/tione/notebook/rsi_rl\"; RZ = \"/run/ti/rzero\"#R = \"$FR\"; RZ = \"$FZ\"#" $D/rz_extract.py > $T/ex.py
sed -e "s#^R = \"/home/tione/notebook/rsi_rl\"#R = \"$FR\"#" $D/rz_read.py > $T/rd.py
echo "redirected: $(grep -c "$FR" $T/ex.py) $(grep -c "$FR" $T/rd.py)"
cp /home/tione/notebook/rsi_rl/dirA4/results/umwp/plain_qwen3_4b_base.jsonl $FR/dirA4/results/umwp/
cp /home/tione/notebook/rsi_rl/dirA4/results/umwp/plain_qwen3_4b_base.jsonl $FR/dirA4/results/umwp/plain_rzero_s0_it3.jsonl
/usr/bin/python3 - $FZ <<'EOF'
import json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
for S in (0, 1):
    gq = f"{sys.argv[1]}/s{S}/generated_question"
    for it in (1, 3):
        n = 600 + 100 * it; name = f"rzero_qwen3_4b_Base_s{S}_solver_v{it}"
        rows = [{"problem": f"q{it}-{k} " + ("word " * (3000 if k < 3 else 5)), "answer": random.choice(["3", "\\frac{1}{2}", "x+1", "1,000"]), "score": 0.5} for k in range(n)]
        with open(f"{gq}/{name}_admitted.jsonl", "w") as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
        pq.write_table(pa.Table.from_pylist(rows), f"{gq}/{name}_admitted.parquet")
        json.dump(rows + [{"problem": "p", "answer": "1", "score": 1.0}] * 50, open(f"{gq}/{name}_all_scored.json", "w"))
        for k in range(8): json.dump([{"question": "x", "answer": "1", "score": 0}] * 100 + [{"question": "y", "answer": "", "score": -1}] * 25, open(f"{gq}/{name}_{k}.json.raw", "w"))
EOF
/usr/bin/python3 $T/ex.py --seed 0; echo "a) no host file: exit $?"
echo "$(hostname) $FZ/s0 1" > $XR/s0.host; /usr/bin/python3 $T/ex.py --seed 0 2>&1 | tail -4; /usr/bin/python3 $T/ex.py --seed 0 --iters 1 | head -1
echo "otherhost $FZ/s1 1" > $XR/s1.host; /usr/bin/python3 $T/ex.py --seed 1; echo "b) seed owned by another host: exit $?"
head -c 200 $FR/dirB/runs/rzero_qwen3_4b_Base_s0_it1/admitted.jsonl; echo
/usr/bin/python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print({k: d[k] for k in ('n_admitted','n_admitted_over_2048_tokens','n_solver_batches_of_512','audit_sample_n','extract_host')})" $XR/s0/rzero_qwen3_4b_Base_s0_solver_v1_summary.json
/usr/bin/python3 - $FR <<'EOF'
import json, sys
R = sys.argv[1]
for it, p in ((1, 0.20), (3, 0.31)):
    d = f"{R}/dirB/runs/rzero_qwen3_4b_Base_s0_it{it}"
    ids = [json.loads(l)["id"] for l in open(f"{d}/admitted.jsonl")]
    v = [{"k": k, "id": i, "verdict": "UNSOLVABLE" if k < int(p * 200) else "SOLVABLE", "confidence": 0.9, "note": ""} for k, i in enumerate(ids)]
    json.dump({"verdicts": v}, open(f"{d}/audit_admitted.json", "w"))
EOF
RD(){ /usr/bin/python3 $T/rd.py | /usr/bin/python3 -c "import json,sys; o=json.load(sys.stdin); s=o['per_seed'].get('0', {}); print('P1', o['RZ-P1'], '| P2', o['RZ-P2'], '| s0 P1', s.get('RZ-P1', {}).get('diff_points'), '| s0 P2', s.get('RZ-P2'), '| counted', o['counted_seed_ids'], '| abandoned', list(o['abandoned']), '| not started', o['not_started'], '| error', o.get('ERROR'))"; }
echo "c) full mode:"; RD
echo "p1only x" > $XR/s0/mode; echo "d) seed 0 in RZ-P1-only mode:"; RD; rm -f $XR/s0/mode
cp $XR/s0/rzero_qwen3_4b_Base_s0_solver_v3_summary.json $T/keep.json
/usr/bin/python3 -c "import json,sys; f=sys.argv[1]; d=json.load(open(f)); d['admitted_parquet_mtime']=$T0+72*3600; json.dump(d,open(f,'w'))" $XR/s0/rzero_qwen3_4b_Base_s0_solver_v3_summary.json
echo "e) it3 admitted set created after T0+71h:"; RD; cp $T/keep.json $XR/s0/rzero_qwen3_4b_Base_s0_solver_v3_summary.json
echo "1 otherhost lost" > $XR/s1.abandoned; echo "f) seed 1 abandoned:"; RD
echo "$(hostname) x 1" > $XR/s2.host; echo "$(hostname) x 1" > $XR/s3.host; echo "g) three non-abandoned seed ids (rule broken):"; RD; rm -f $XR/s2.host $XR/s3.host
head -c -30 $FR/dirA4/results/umwp/plain_rzero_s0_it3.jsonl > $T/p.jsonl; mv $T/p.jsonl $FR/dirA4/results/umwp/plain_rzero_s0_it3.jsonl; echo "h) final dump with a partial last line (1,199 ids):"; RD

( flock $FZ/s0/extract.lock sleep 8 ) & LP=$!; sleep 1; t=$(date +%s); /usr/bin/python3 $T/ex.py --seed 0 --iters 1 > $T/ex_lock.out 2>&1
echo "i) extraction while another holds the lock (review item 9): exit $? after $(( $(date +%s) - t )) s ('$(tail -1 $T/ex_lock.out | cut -c1-60)'); .tmp files left: $(find $FR/dirB/runs $XR -name '*.tmp*' | wc -l)"; wait $LP

echo "== T11 projected-finish rule (rz_clock.py; synthetic markers and times)"
sed -e "s#^R = \"/home/tione/notebook/rsi_rl\"#R = \"$T/ck\"#" $D/rz_clock.py > $T/clock.py; C="/usr/bin/python3 $T/clock.py"
for s in 0 1 2; do mkdir -p $T/ck/s$s/models $T/ck/s$s/generated_question; done
m(){ $C mode --seed $1 --storage $T/ck/s$1 --now $(( T0 + $2 )) ${3:-} | cut -c1-230; }
echo "a) seed 0, T0+8h, smoke:  $(m 0 28800 '--smoke 1')"
echo "b) seed 0, T0+41h, smoke: $(m 0 147600 '--smoke 1')"
echo "c) seed 0, T0+44h:        $(m 0 158400)"
echo "d) seed 0, T0+46h:        $(m 0 165600)"
echo "e) seed 0, T0+50.5h:      $(m 0 181800)"
SP=$T/ck/s1; A=rzero_qwen3_4b_Base_s1; t0=$(( T0 + 36000 ))
$C attempt --seed 1 --storage $SP --now $t0
mkdir -p $SP/models/${A}_questioner_v1; touch -d @$(( t0 + 7200 )) $SP/models/${A}_questioner_v1/A75_DONE
touch -d @$(( t0 + 9000 )) $SP/generated_question/${A}_solver_v1_admitted.parquet; $C observe --seed 1 --storage $SP
$C attempt --seed 1 --storage $SP --now $(( t0 + 20000 )); mkdir -p $SP/models/${A}_solver_v1; touch -d @$(( t0 + 30000 )) $SP/models/${A}_solver_v1/A75_DONE
echo "f) seed 1 with q1 2 h and a1 0.5 h measured, s1 interrupted (not measured): $(m 1 40000)"
echo "   events: $(tr '\t\n' ': ' < $T/ck/ext/rzero_runs/s1/events.tsv)"
echo "g) seed 2 uses seed 1's measurements, T0+46h: $(m 2 165600)"
echo "h) seed 2, commit at T0+52h, then ask at T0+10h: $(m 2 187200 --commit | cut -c1-6) -> $(m 2 36000 | cut -c1-90)"
echo "== T12 monitor and the driver's teardown (review item 1): stop_reason rules; the monitor carries no tag"
( TAG=rzA75_dtest12; STORAGE_PATH=$T/sp12; LEDGER=$T/ledger12; mkdir -p $STORAGE_PATH; source $D/rz_common.sh
  headroom_gib(){ echo 300; }; foreign_confirmed(){ echo "pid 7 (x) on GPU-x;"; }
  echo "a) foreign finding, no teardown: '$(stop_reason)'"
  touch $STORAGE_PATH/teardown; echo "b) foreign finding while the driver tears down: '$(stop_reason)'"; rm -f "${STORAGE_PATH:?}"/teardown
  foreign_confirmed(){ touch $STORAGE_PATH/teardown; echo "pid 7 (x) on GPU-x;"; }
  echo "c) teardown began during the 30 s confirmation: '$(stop_reason)'"; rm -f "${STORAGE_PATH:?}"/teardown
  foreign_confirmed(){ :; }; headroom_gib(){ echo 99; }; echo "d) headroom 99 GiB: '$(stop_reason)'"
  headroom_gib(){ echo 100; }; echo "e) headroom 100 GiB: '$(stop_reason)'" )
cat > $T/drv12.sh <<EOF
TAG=rzA75_dtest12m; STORAGE_PATH=$T/drv12; LEDGER=$T/ledger12; D=$D; RZ=$T/rz; X=$T/x; mkdir -p \$STORAGE_PATH
source $D/rz_common.sh; MK=$T/mk12; mkdir -p \$MK
gpu_apps(){ :; }; gpu_mem(){ :; }; headroom_gib(){ echo 300; }
claim; export RZ_A75_TAG=\$TAG
start_monitor; trap cleanup EXIT
sleep 300 & SL=\$!; sleep 2; m=\$(ps -o pid= --ppid \$MON | tr -d ' ' | head -1)
tg(){ grep -qzx "RZ_A75_TAG=\$TAG" /proc/\$1/environ 2>/dev/null && echo tagged || echo untagged; }
echo "f) the monitor's wait '\$(ps -o args= -p \$m)' is \$(tg \$m); the driver's own sleep is \$(tg \$SL)"
kill_own
echo "g) after the driver's kill_own: its sleep \$(kill -0 \$SL 2>/dev/null && echo alive || echo killed); the monitor's wait \$(kill -0 \$m 2>/dev/null && echo 'still running' || echo 'cut short'); the monitor \$(kill -0 \$MON 2>/dev/null && echo alive || echo dead)"
EOF
bash $T/drv12.sh; echo "   fake driver exit $?, markers left $(ls $T/mk12 | wc -l)"

echo "== T13 clock errors (review item 2) and smoke-kept markers at waiter exit (review item 3); staged copies, host checks stubbed"
D4=$T/d13; R13=$T/rz13; X13=$T/x13; mkdir -p $D4 $R13 $X13 $T/mk13; cp $D/*.sh $D/*.py $D/rzero_A7_5.patch $D4/
for f in rz_wait_and_run.sh rz_run.sh rz_smoke.sh rz_stop.sh; do
  sed -e "s#^RZ=/run/ti/rzero; D=.*#RZ=$R13; D=$D4; X=$X13#" -e "s#^source \$D/rz_env.sh#echo 'T13 guard: past the projection'; exit 99#" $D/$f > $D4/$f; done
cat >> $D4/rz_common.sh <<EOF
# T13 stubs: markers in scratch, the host always free, no smoke test needed, nothing killed, no 60 s preflight wait
MK=$T/mk13; host_check(){ :; }; smoke_not_needed(){ return 0; }; kill_tag(){ :; }; preflight(){ :; }
EOF
cat > $D4/rz_clock.py <<EOF
import sys  # T13 stub: each call prints the next line of the sequence file; CRASH = a failing rz_clock.py
q = "$T/clockseq"; L = open(q).read().split("\n"); open(q, "w").write("\n".join(L[1:]))
if L[0] == "CRASH": sys.exit("simulated rz_clock.py failure")
print(L[0])
EOF
echo "staged: $(grep -c "^RZ=$R13; D=$D4; X=$X13" $D4/rz_wait_and_run.sh $D4/rz_run.sh $D4/rz_smoke.sh $D4/rz_stop.sh | cut -d: -f2 | tr '\n' ' ')guards $(grep -c 'T13 guard' $D4/rz_run.sh $D4/rz_smoke.sh | cut -d: -f2 | tr '\n' ' ')"
echo "rzA75_smoke 999998 1" > $T/mk13/gpu_single_0; echo "otherjob 1 1" > $T/mk13/gpu_single_1
printf 'CRASH\nfull x\nCRASH\nnone x\n' > $T/clockseq
t=$(date +%s); timeout 400 bash $D4/rz_wait_and_run.sh 0; echo "a) waiter exit $? after $(( $(date +%s) - t )) s; markers left: $(ls $T/mk13 | tr '\n' ' '); failures in driver.log: $(grep -c 'simulated rz_clock.py failure' $R13/s0/driver.log)"
grep -h "waiter\|clock error\|removed" $X13/ledger_s0.log | cut -d' ' -f5- | cut -c1-190
rm -f "${T:?}"/mk13/*; printf 'full x\nCRASH\n' > $T/clockseq; bash $D4/rz_run.sh 1
echo "b) driver, clock error after the claim: exit $?; markers left $(ls $T/mk13 | wc -l); s1.host $([ -f $X13/s1.host ] && echo written || echo missing)"
printf 'full x\nnone x\n' > $T/clockseq; bash $D4/rz_run.sh 1; echo "c) driver, projection none after the claim: exit $?; markers left $(ls $T/mk13 | wc -l)"
printf 'none x\n' > $T/clockseq; bash $D4/rz_run.sh 1; echo "d) driver, projection none before the preflight: exit $?"
grep -h "clock error\|not started" $X13/ledger_s1.log | cut -d' ' -f5- | cut -c1-190

echo "== T14 rz_stop.sh <S> removes the markers a passed smoke test kept once no live run holds them (review item 3)"
rm -f "${T:?}"/mk13/*; bash -c 'sleep 60; :' fake_rz_smoke.sh & LIVE=$!; sleep 0.5
echo "rzA75_smoke 999998 1" > $T/mk13/gpu_single_0; echo "rzA75_smoke $LIVE 1" > $T/mk13/gpu_single_1
echo "rzA75_s0 999997 1" > $T/mk13/gpu_single_2; echo "otherjob 1 1" > $T/mk13/gpu_single_3
bash $D4/rz_stop.sh 0 > /dev/null; echo "markers left: $(cd $T/mk13 && for f in *; do printf '%s=%s ' $f "$(cut -d' ' -f1 $f)"; done)"; kill $LIVE; wait $LIVE 2>/dev/null
grep -h "removed\|stopped by rz_stop" $X13/ledger_s0.log | tail -2 | cut -d' ' -f5- | cut -c1-170

echo "== T15 hang rule (review item 8; hung and extra_reason taken from rz_run.sh; synthetic mtimes)"
( S=0; ABBR=rzero_qwen3_4b_Base_s0; STORAGE_PATH=$T/sp15; M=$STORAGE_PATH/models; GQ=$STORAGE_PATH/generated_question; TAG=t15; LEDGER=$T/ledger15
  mkdir -p $M/${ABBR}_solver_v1 $GQ $STORAGE_PATH/temp_results; source $D/rz_common.sh
  eval "$(sed -n '/^hung(){/p; /^extra_reason(){/,/; }$/p' $D/rz_run.sh)"
  clock(){ :; }; kill_own(){ echo "kill_own" >> $T/k15; }; extract(){ :; }; MODE=full; rm -f "${T:?}"/k15
  old(){ touch -d '-121 min' $STORAGE_PATH/in_attempt $STORAGE_PATH/main.log $M $M/${ABBR}_solver_v1 $GQ $STORAGE_PATH/temp_results; }
  yn(){ hung && echo yes || echo no; }
  : > $STORAGE_PATH/main.log; touch $STORAGE_PATH/in_attempt; echo "a) attempt just started: hung=$(yn)"
  old; echo "b) nothing changed for 121 min: hung=$(yn)"
  echo x >> $STORAGE_PATH/main.log; echo "c) main.log just written: hung=$(yn)"; old
  touch $M/${ABBR}_solver_v1/latest_global_step.txt; echo "d) a tracker just written: hung=$(yn)"; touch -d '-121 min' $M/${ABBR}_solver_v1/latest_global_step.txt; old
  touch $GQ/${ABBR}_solver_v1_3.json; echo "e) a generated-question shard just written: hung=$(yn)"; touch -d '-121 min' $GQ/${ABBR}_solver_v1_3.json; old
  echo "f) all old again: hung=$(yn)"
  out=$(extra_reason); echo "g) extra_reason when hung: stdout '$out', $(cat $T/k15 2>/dev/null | wc -l) kill_own call(s), in_attempt left: $([ -e $STORAGE_PATH/in_attempt ] && echo yes || echo no)"
  out=$(extra_reason); echo "h) one minute later: $(cat $T/k15 | wc -l) kill_own call(s) in total, hung=$(yn)"
  grep -h "hang rule" $LEDGER | cut -d' ' -f5- | cut -c1-170 )
echo "== done"
