#!/bin/bash
# A7-5 (c, d, e, f): one R-Zero seed on this host = seed and host checks, projected-finish mode, preflight, atomic claim of
# all 8 GPUs, the patched official scripts/main.sh (3 iterations; every stage resumes) under a 60 s monitor, then the
# admitted-set extraction (d) and the UMWP dumps (e). Normally started by rz_wait_and_run.sh <S>.
# By hand: nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_run.sh <S> > /dev/null 2>&1 &   (resume: the same)
# Stop: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh <S>
# Exit: 0 done (or the RZ-P1-only end), 2 refused, 3 preflight or clock error, 4 import check, 6 foreign GPU use (yield),
# 7 memory headroom, 8 deadline or no time left, 9 admitted set below one batch (finding X8), 10 three failed attempts (X4).
S=${1:-}; case "$S" in 0|1|2|3) ;; *) echo "usage: rz_run.sh 0|1|2|3"; exit 2;; esac
[ -z "${RZ_A75_TAG:-}" ] || { echo "refusing: already inside an A7-5 run environment"; exit 2; }
for v in RZ_Q_STEPS RZ_Q_MERGE RZ_S_STEPS RZ_S_MERGE RZ_GEN_N RZ_SMOKE_ARGS; do [ -z "${!v:-}" ] || { echo "refusing: $v is set (smoke-test hook)"; exit 2; }; done
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_s$S; ABBR=rzero_qwen3_4b_Base_s$S
export STORAGE_PATH=$RZ/s$S; LEDGER=$X/ledger_s$S.log
M=$STORAGE_PATH/models; GQ=$STORAGE_PATH/generated_question
mkdir -p $STORAGE_PATH/evaluation $M $GQ $STORAGE_PATH/temp_results $X/s$S
source $D/rz_common.sh
r=$(seed_ok $S) || { L "$r"; echo "$r"; exit 2; }
p=$(cat $STORAGE_PATH/driver.pid 2>/dev/null)
[ -n "$p" ] && grep -q rz_run.sh /proc/$p/cmdline 2>/dev/null && { echo "a driver for seed $S is already running (pid $p)"; exit 2; }
extract(){ /usr/bin/python3 $D/rz_extract.py --seed $S >> $STORAGE_PATH/driver.log 2>&1; }
clock(){ /usr/bin/python3 $D/rz_clock.py "$@" --seed $S --storage $STORAGE_PATH; }
# projected-finish rule (section 8): full, p1only (stop once the iteration-3 admitted set exists; no RZ-P2 for this seed) or
# none (exit 8). Anything else means that rz_clock.py failed (e.g. a transient NFS error): a clock error, exit 3, which the
# waiter retries (review item 2)
mode_ok(){ case "$MODE" in full|p1only) ;; none) L "not started: the projected-finish rule leaves no time ($*)"; exit 8;;
  *) L "clock error: rz_clock.py printed '$(echo "$*" | cut -c1-120)' (its error is in driver.log); exit 3, retried by the waiter"; exit 3;; esac; }
set -- $(clock mode 2>> $STORAGE_PATH/driver.log); MODE=${1:-}; mode_ok "$@"
kill_tag $TAG                     # leftovers of an interrupted attempt of this seed (our tagged processes only)
rm -f "${STORAGE_PATH:?}"/stop_code "${STORAGE_PATH:?}"/teardown "${STORAGE_PATH:?}"/in_attempt
preflight; claim; claim_host $S; echo $$ > $STORAGE_PATH/driver.pid
set -- $(clock mode --commit 2>> $STORAGE_PATH/driver.log); MODE=${1:-}; mode_ok "$@"
L "driver start: mode $MODE, projection $2"
if [ "$MODE" = p1only ] && [ -f $GQ/${ABBR}_solver_v3_admitted.parquet ]; then extract; L "RZ-P1-only mode and the iteration-3 admitted set exists: nothing to run"; exit 0; fi
export RZ_A75_TAG=$TAG RZ_SEED=$S
source $D/rz_env.sh
cd $RZ/R-Zero
python -c "import verl; assert verl.__version__ == '0.3.1.dev0' and verl.__file__.startswith('/run/ti/rzero/R-Zero/')" || { L "verl import check failed"; exit 4; }
done_list(){ (cd $M && ls -d */A75_DONE 2>/dev/null | sed 's#/A75_DONE##' | tr '\n' ' '); }
# (d) each iteration's audit sample is drawn as soon as its admitted set exists, so the Opus audit can start during the run
start_watcher(){ ( while sleep 600; do /usr/bin/python3 $D/rz_extract.py --seed $S > $STORAGE_PATH/extract.log 2>&1; done ) & XW=$!; }
# monitor rules beyond foreign GPU use, memory and the deadline: record stage completions (rz_clock.py); the hang rule
# (review item 8): while an attempt runs (file in_attempt), nothing changed for 120 min in main.log, the stage directories
# (trackers, checkpoints, markers), the generated-question files or temp_results stops the attempt (our tagged processes
# killed; it counts as a failed attempt, X4); in RZ-P1-only mode, end the run once the iteration-3 admitted set exists
hung(){ [ -f $STORAGE_PATH/in_attempt ] && [ -z "$(find $STORAGE_PATH/in_attempt $STORAGE_PATH/main.log $M $GQ $STORAGE_PATH/temp_results -maxdepth 2 -mmin -120 -print -quit 2>/dev/null)" ]; }
extra_reason(){ clock observe > /dev/null 2>&1
  if hung; then rm -f "${STORAGE_PATH:?}"/in_attempt; L "hang rule: nothing changed for 120 min (main.log, stage directories, generated questions): this attempt is stopped and counts as failed"; kill_own; fi
  [ "$MODE" = p1only ] && [ -f $GQ/${ABBR}_solver_v3_admitted.parquet ] && { extract; echo "0 RZ-P1-only mode (section 8): the iteration-3 admitted set exists"; }; }
# exit handler: watcher and monitor stopped, our processes gone, markers released, then the extraction once more (CPU only;
# review item 5), so that an admitted set created shortly before any stop gets its audit sample and summary at once
start_watcher; start_monitor; trap 'cleanup; extract' EXIT
for attempt in 1 2 3; do            # up to 3 attempts per driver start; every attempt resumes where the last one stopped
  stop_check; clock attempt > /dev/null 2>&1
  st=$(cur_stage); k0=$(tracker "$st"); off=$(stat -c %s $STORAGE_PATH/main.log 2>/dev/null || echo 0)
  L "main.sh $ABBR attempt $attempt (stages done: $(done_list); next: $st at checkpoint $k0)"
  touch $STORAGE_PATH/in_attempt
  bash scripts/main.sh /root/models/Qwen3-4B-Base $ABBR >> $STORAGE_PATH/main.log 2>&1; rc=$?; rm -f "${STORAGE_PATH:?}"/in_attempt
  stop_check; clock observe > /dev/null 2>&1
  L "main.sh exit $rc; stages done: $(done_list)"
  [ -f $M/${ABBR}_solver_v3/A75_DONE ] && break
  if small_admitted $off; then extract; exit 9; fi
  # teardown of the failed attempt: our leftovers killed (kill_own waits until they are gone) while the monitor skips its
  # foreign-use rule; this driver's own check after the teardown decides (review item 1)
  touch $STORAGE_PATH/teardown; kill_own
  stop_check
  f=$(foreign_confirmed); rm -f "${STORAGE_PATH:?}"/teardown; [ -z "$f" ] || { L "foreign GPU use after the failed attempt ($f): yielding the host"; exit 6; }
  resume_fallback "$st" "$k0" $off
  start_watcher; [ $attempt = 3 ] || sleep 120
done
kill $XW 2>/dev/null; extract
[ -f $M/${ABBR}_solver_v3/A75_DONE ] || { L "X4: 3 attempts ended without the final solver (stages done: $(done_list))"; exit 10; }
bash $D/rz_umwp.sh $S 0 >> $STORAGE_PATH/driver.log 2>&1; stop_check; clock observe > /dev/null 2>&1
L "UMWP dumps: $(tail -4 $STORAGE_PATH/driver.log | tr '\n' ' ' | cut -c1-300)"
L "driver end"
