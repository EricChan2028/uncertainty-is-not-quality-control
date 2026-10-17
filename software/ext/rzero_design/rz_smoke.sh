#!/bin/bash
# A7-5 smoke test: 1.5-2.5 h on a fully free host, immediately before the first confirmatory run on that host; started by
# rz_wait_and_run.sh. It runs the patched stage scripts with tiny sizes (smoke hooks RZ_Q_STEPS ... RZ_SMOKE_ARGS, never set
# in a confirmatory run). Nothing here is data. GO = S1, S3 and S4 ok ("smoke PASSED" in ledger_smoke.log); a failed S2
# alone is logged, and the automatic resume fallback of addendum section 7 then covers a failing resume. On PASS the 8 GPU
# markers are kept for the run that follows (hand-over, fix 8). Exit: 0 PASS, 2 refused, 3 preflight, 5 FAILED, 6/7/8
# stopped by the monitor (foreign GPU use / memory headroom / deadline).
[ -z "${RZ_A75_TAG:-}" ] || exit 2
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_smoke; B=/root/models/Qwen3-4B-Base
export STORAGE_PATH=$RZ/smoke; LEDGER=$X/ledger_smoke.log
p=$(cat $STORAGE_PATH/driver.pid 2>/dev/null); [ -n "$p" ] && grep -q rz_smoke.sh /proc/$p/cmdline 2>/dev/null && { echo "a smoke test is already running (pid $p)"; exit 2; }
rm -rf $STORAGE_PATH; mkdir -p $STORAGE_PATH/evaluation $STORAGE_PATH/models $STORAGE_PATH/generated_question $STORAGE_PATH/temp_results $X
source $D/rz_common.sh
kill_tag $TAG; preflight; claim; echo $$ > $STORAGE_PATH/driver.pid
export RZ_A75_TAG=$TAG RZ_SEED=0
source $D/rz_env.sh
cd $RZ/R-Zero
start_monitor; trap cleanup EXIT
LOG=$STORAGE_PATH/smoke.log; M=$STORAGE_PATH/models; Q=rzsmoke_qwen3_4b_Base_questioner_v1; SV=rzsmoke_qwen3_4b_Base_solver_v1
python -c "import pyarrow.parquet as pq; t = pq.read_table('$RZ/data/math12k_test.parquet'); pq.write_table(t.slice(0, 32), '$RZ/data/smoke_val32.parquet')"
V="data.val_files=$RZ/data/smoke_val32.parquet trainer.val_before_train=false"
ok(){ stop_check; if [ -f $M/$1/A75_DONE ]; then L "smoke $2: ok ($(( $(date +%s) - T )) s)"; else L "smoke $2: FAILED (see $LOG)"; exit 5; fi; }
L "smoke start (stack $(stack_fp))"
# S1 questioner, 1 GRPO step: 4 reward servers, caller_penalty reward with BLEU clustering, FSDP + vLLM 0.11 rollout and
#    weight sync on sm_120, checkpoint, merge, targeted server shutdown
T=$(date +%s); RZ_Q_STEPS=1 RZ_Q_MERGE=1 RZ_SMOKE_ARGS="data.rollout_batch_size=32 $V" bash scripts/questioner_train_penalty.sh $B $B $Q >> $LOG 2>&1
ok $Q "S1 questioner step"
# S2 the same stage continued to step 2 from the step-1 checkpoint (repository trainer.load_checkpoint_path), merged at step 2
rm -f $M/$Q/A75_DONE; T=$(date +%s)
RZ_Q_STEPS=2 RZ_Q_MERGE=2 RZ_SMOKE_ARGS="data.rollout_batch_size=32 $V" bash scripts/questioner_train_penalty.sh $B $B $Q >> $LOG 2>&1
stop_check
if grep -q "Load from checkpoint: .*global_step_1" $LOG && [ -f $M/$Q/A75_DONE ]; then L "smoke S2 questioner resume: ok ($(( $(date +%s) - T )) s)"; QS=2
else L "smoke S2 questioner resume: FAILED (see $LOG); the automatic resume fallback of addendum section 7 covers resumes"; QS=1; S2F=1; fi
# S3 solver stage: 8 x 100 generations, 9-sample pseudo-labels, 0.3-0.8 filter, local parquet, 1 solver GRPO step on 8 GPUs,
#    checkpoint at step 1 (save_freq 1 here only; the confirmatory solver keeps the repository's save_freq 5), merge
T=$(date +%s)
RZ_GEN_N=100 RZ_S_STEPS=1 RZ_S_MERGE=1 RZ_SMOKE_ARGS="data.rollout_batch_size=16 worker.actor.global_batch_size=16 trainer.val_freq=-1 trainer.save_freq=1 $V" \
  bash scripts/solver_train.sh $B $M/$Q/global_step_$QS/actor/huggingface $SV >> $LOG 2>&1
ok $SV "S3 solver stage"
L "smoke S3 counts: raw $(cat $STORAGE_PATH/generated_question/${SV}_[0-7].json.raw 2>/dev/null | grep -c '"score"') scored $(python -c "import json; print(len(json.load(open('$STORAGE_PATH/generated_question/${SV}_all_scored.json'))))") admitted $(wc -l < $STORAGE_PATH/generated_question/${SV}_admitted.jsonl)"
# S4 the merged smoke solver loads in vLLM through the frozen UMWP probe (5 unanswerable + their originals); the dump has a
#    per-host name, so smoke tests on two hosts never share a file (review item 10), and is deleted before and after
T=$(date +%s); U=/home/tione/notebook/rsi_rl/dirA4/results/umwp; ST=smoke_rzA75_$(hostname | cut -c1-12); rm -f "${U:?}/${ST:?}.jsonl"
env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/tione/notebook/rsi_rl/hf_home TOKENIZERS_PARALLELISM=false \
  /usr/bin/python3 /home/tione/notebook/rsi_rl/dirA3/code/umwp_probe.py --model $M/$SV/global_step_1/actor/huggingface --tag $ST --n_uns 5 >> $LOG 2>&1
stop_check
n=$(cat "$U/$ST.jsonl" 2>/dev/null | wc -l); rm -f "${U:?}/${ST:?}.jsonl"
[ "$n" -ge 6 ] || { L "smoke S4 probe: FAILED ($n rows)"; exit 5; }
L "smoke S4 probe on the merged solver: ok ($(( $(date +%s) - T )) s)"
# PASS: our processes gone, the 8 markers kept for rz_run.sh (which takes them over within the hour)
rm -rf $M; kill $MON 2>/dev/null; kill_own; KEEP_MARKERS=1
L "smoke PASSED${S2F:+ except S2 (the resume fallback covers resumes)} epoch=$(date +%s) host=$(hostname) stack=$(stack_fp) (smoke checkpoints removed; log kept at $LOG; 8 GPU markers kept for the run that follows)"
