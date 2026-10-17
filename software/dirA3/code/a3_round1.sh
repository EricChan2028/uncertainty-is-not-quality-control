#!/bin/bash
# Single-GPU variant of a3_round.sh: propose -> guard -> (subset) -> train (in-process vLLM, sleep mode) -> evaluate, all on GPU $6.
# Arms (amendment A3c): own | pdc | own_sub (random 50% of the admitted set) | own_clean (audit-SOLVABLE proposals + intact tracers)
#                       | own_dirty (audit-UNSOLVABLE/AMBIGUOUS proposals + deleted tracers). own_* arms train with the own reward.
# Round 1 of own_sub/own_clean/own_dirty reuses the proposals and guard output of the own lineage with the same seed
# (frozen proposer, base-model guard: identical by construction). Usage: a3_round1.sh <lineage> <base_model> <arm> <seed> <round> <GPU>
set -u
L=$1; BASE=$2; ARM=$3; SEED=$4; RD=$5; G=$6
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=${L}_r${RD}; OUT=$B/runs/$NAME; mkdir -p $OUT
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
ulimit -c 0                                            # the in-process engine allocator can abort at interpreter exit; never dump a 60 GB core
log(){ echo "$(date '+%F %T') a3 $NAME: $*" >> $LOG; }
touch /dev/shm/gpu_reserved_$G; trap 'rm -f /dev/shm/gpu_reserved_$G' EXIT
if [ "$RD" -eq 1 ]; then PREV=$BASE; else PREV=$CK/${L}_r$((RD-1))/final; fi
[ -e $PREV/config.json ] || { log "previous solver missing: $PREV"; exit 3; }
ABS=0; [ "$ARM" = "pdc" ] && ABS=1
TARM=$ARM; case "$ARM" in own_sub|own_clean|own_dirty|own_tg|own_azr) TARM=own;; esac
GUARD=rzero; case "$ARM" in own_tg) GUARD=tracer;; own_azr) GUARD=azr;; esac      # amendment A3e
SRC=$B/runs/${L%%_own_*}_own_a3_s${SEED}_r1                     # own lineage of the same seed
if [ "$RD" -eq 1 ] && [ "$GUARD" = rzero ] && [ "$TARM" != "$ARM" ] && [ ! -s $OUT/admitted.jsonl ]; then
  w=0; while [ ! -s $SRC/admitted.jsonl ] || [ ! -s $SRC/guard_stats.json ]; do [ $w -eq 0 ] && log "waiting for round-1 guard output of $(basename $SRC)"; w=$((w+1)); [ $w -gt 180 ] && { log "wait for $(basename $SRC) timed out"; exit 5; }; sleep 60; done
  cp $SRC/proposed.jsonl $SRC/admitted.jsonl $SRC/guard_stats.json $OUT/ && log "round-1 proposals/guard copied from $(basename $SRC)"
fi
[ -s $OUT/proposed.jsonl ] || python3 $A3/code/a3_propose.py --model $BASE --out $OUT/proposed.jsonl --n 1200 --seed $((SEED*100+RD)) >> $OUT/propose.log 2>&1 || { log "propose failed"; exit 1; }
[ -s $OUT/admitted.jsonl ] || python3 $A3/code/a3_guard.py --model $PREV --candidates $OUT/proposed.jsonl --out_dir $OUT --seed $((SEED*100+RD)) --abstain_prompt $ABS --guard $GUARD >> $OUT/guard.log 2>&1 || { log "guard failed"; exit 1; }
log "guard: $(cat $OUT/guard_stats.json | tr -d '\n ' | cut -c1-300)"
DATA=$OUT/admitted.jsonl
case "$ARM" in
  own_sub) DATA=$OUT/admitted_sub.jsonl
    [ -s $DATA ] || python3 $A3/code/a3_subset.py --admitted $OUT/admitted.jsonl --mode sub --frac 0.5 --seed $((SEED*100+RD)) --out $DATA >> $OUT/subset.log 2>&1 || { log "subset failed"; exit 1; };;
  own_clean|own_dirty) DATA=$OUT/admitted_${ARM#own_}.jsonl
    AUD=$OUT/audit_full.json; [ "$RD" -eq 1 ] && AUD=$SRC/audit_full.json
    if [ ! -s $DATA ]; then
      w=0; while [ ! -s $AUD ]; do [ $w -eq 0 ] && log "waiting for $AUD"; w=$((w+1)); [ $w -gt 360 ] && { log "audit wait timed out"; exit 5; }; sleep 60; done
      python3 $A3/code/a3_subset.py --admitted $OUT/admitted.jsonl --mode ${ARM#own_} --audit $AUD --out $DATA >> $OUT/subset.log 2>&1 || { log "subset failed"; exit 1; }
    fi;;
esac
log "training data $(basename $DATA): $(wc -l < $DATA) items"
if [ ! -f $OUT/.trained ]; then
  EXTRA="--answer_fallback 1 --vllm_inproc 1"; [ "$ARM" = pdc ] && EXTRA="$EXTRA --n_del 16"
  case "$BASE" in *8B*) EXTRA="$EXTRA --optimizer_device cpu --vllm_util 0.22";; esac
  python3 $B/code/b_train.py --model $PREV --arm $TARM --seed $SEED --out $OUT --data $DATA --steps 60 $EXTRA >> $OUT/train.log 2>&1 < /dev/null; rc=$?
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  [ -f $OUT/.trained ] || { log "train stopped without finishing (rc=$rc)"; exit 2; }
  hostname > $OUT/host
fi
if [ ! -f $OUT/.evaluated ]; then
  F=$CK/$NAME/final
  python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null \
   && python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60_abs --abstain_prompt 1 --data math500_train,math_heldout1000,gsm8k_test500,hp_noop500 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null \
   && python3 $R/dirA/code/c0_probe.py --model $F --tag hp_${NAME} --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $OUT/eval_hp.log 2>&1 < /dev/null \
   && python3 $R/dirA/code/c0_probe.py --model $F --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $OUT/eval_hp.log 2>&1 < /dev/null \
   && touch $OUT/.evaluated || { log "eval failed"; exit 4; }
fi
rm -rf $CK/$NAME/step20 $CK/$NAME/step40 $CK/$NAME/resume $CK/$NAME/rollout
touch $OUT/.done; log "round done (single GPU $G)"
