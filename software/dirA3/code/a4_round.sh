#!/bin/bash
# A4 pair variant (own | own_ag) of the A3 self-generated-curriculum loop round for one lineage, on a GPU pair (GA = vLLM, GB = trainer):
#   propose (frozen base proposer) -> guard (current solver, R-Zero band, tracers planted) -> train the solver
#   on the admitted set (arm own = published loop; arm pdc = loop + premise-deletion contrast) -> evaluate.
# Usage: a3_round.sh <lineage> <base_model> <arm> <seed> <round> <GA> <GB>
# Lineage run dirs: dirB/runs/<lineage>_r<round>; solver weights: /dev/shm/ckpt_b/<lineage>_r<round>/final.
set -u
L=$1; BASE=$2; ARM=$3; SEED=$4; RD=$5; GA=$6; GB=$7
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=${L}_r${RD}; OUT=$B/runs/$NAME; mkdir -p $OUT
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
log(){ echo "$(date '+%F %T') a4 $NAME: $*" >> $LOG; }
touch /dev/shm/gpu_reserved_$GA /dev/shm/gpu_reserved_$GB                 # single-GPU jobs (re-eval, MiP, 8B probes) skip reserved cards
trap 'rm -f /dev/shm/gpu_reserved_$GA /dev/shm/gpu_reserved_$GB' EXIT
if [ "$RD" -eq 1 ]; then PREV=$BASE; else PREV=$CK/${L}_r$((RD-1))/final; fi
[ -e $PREV/config.json ] || { log "previous solver missing: $PREV"; exit 3; }
ABS=0; case "$ARM" in pdc|own_ag) ABS=1;; esac
TARM=own; [ "$ARM" = pdc ] && TARM=pdc
GUARD=rzero; [ "$ARM" = own_ag ] && GUARD=abstain      # A4 addition 1
# 1. propose (fresh problems each round; seed encodes lineage seed and round)
if [ ! -s $OUT/proposed.jsonl ]; then
  CUDA_VISIBLE_DEVICES=$GA python3 $A3/code/a3_propose.py --model $BASE --out $OUT/proposed.jsonl --n 1200 --seed $((SEED*100+RD)) >> $OUT/propose.log 2>&1 || { log "propose failed"; exit 1; }
fi
# 2. guard with the CURRENT solver
if [ ! -s $OUT/admitted.jsonl ]; then
  CUDA_VISIBLE_DEVICES=$GA python3 $A3/code/a3_guard.py --model $PREV --candidates $OUT/proposed.jsonl --out_dir $OUT --seed $((SEED*100+RD)) --abstain_prompt $ABS --guard $GUARD --abs_max 0.25 >> $OUT/guard.log 2>&1 || { log "guard failed"; exit 1; }
fi
log "guard: $(cat $OUT/guard_stats.json | tr -d '\n ' | cut -c1-300)"
# 3. train the solver on the admitted set (resumable; rollout server on GA)
if [ ! -f $OUT/.trained ]; then
  RDIR=/dev/shm/rollout_$NAME; rm -rf $RDIR; mkdir -p $RDIR
  EXTRA="--answer_fallback 1"; [ "$ARM" = pdc ] && EXTRA="$EXTRA --n_del 16"      # A3b: the loop parses unboxed numeric answers like the guard does
  case "$BASE" in *8B*) EXTRA="$EXTRA --optimizer_device cpu";; esac          # 8B full-parameter: master + AdamW on host RAM (amendment A3a)
  CUDA_VISIBLE_DEVICES=$GB setsid python3 $B/code/b_train.py --model $PREV --arm $TARM --seed $SEED --out $OUT --rollout_dir $RDIR --data $OUT/admitted.jsonl --steps 60 $EXTRA >> $OUT/train.log 2>&1 < /dev/null &
  TPID=$!
  for i in $(seq 1 90); do [ -f $RDIR/live/config.json ] && break; kill -0 $TPID 2>/dev/null || break; sleep 10; done
  SPID=""
  if [ -f $RDIR/live/config.json ]; then CUDA_VISIBLE_DEVICES=$GA setsid python3 $B/code/b_rollout_server.py --dir $RDIR --gpu_util 0.85 >> $OUT/server.log 2>&1 < /dev/null & SPID=$!; fi
  wait $TPID; rc=$?
  touch $RDIR/stop 2>/dev/null; sleep 3; [ -n "$SPID" ] && kill -- -$SPID 2>/dev/null; sleep 3; [ -n "$SPID" ] && kill -9 -- -$SPID 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GA 2>/dev/null | tr -d ' '); do kill -9 $p 2>/dev/null; done
  rm -rf $RDIR
  [ -f $OUT/.trained ] || { log "train stopped without finishing (rc=$rc)"; exit 2; }
fi
# 4. evaluate the round's solver
if [ ! -f $OUT/.evaluated ]; then
  F=$CK/$NAME/final
  # both GPUs of the pair evaluate at once: accuracy evals on GA, the GSM-HP probes on GB
  ( CUDA_VISIBLE_DEVICES=$GA python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null \
    && CUDA_VISIBLE_DEVICES=$GA python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60_abs --abstain_prompt 1 --data math500_train,math_heldout1000,gsm8k_test500,hp_noop500 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null ) &
  EA=$!
  ( CUDA_VISIBLE_DEVICES=$GB python3 $R/dirA/code/c0_probe.py --model $F --tag hp_${NAME} --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $OUT/eval_hp.log 2>&1 < /dev/null \
    && CUDA_VISIBLE_DEVICES=$GB python3 $R/dirA/code/c0_probe.py --model $F --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $OUT/eval_hp.log 2>&1 < /dev/null ) &
  EB=$!
  wait $EA; ra=$?; wait $EB; rb=$?
  [ $ra -eq 0 ] && [ $rb -eq 0 ] && touch $OUT/.evaluated || { log "eval failed (acc rc=$ra hp rc=$rb)"; exit 4; }
fi
rm -rf $CK/$NAME/step20 $CK/$NAME/step40 $CK/$NAME/resume
touch $OUT/.done; log "round done"
