#!/bin/bash
# A4e loop arm own_evo (frontier refill): propose fresh -> guard fresh (current solver) -> evolve the fresh candidates the current
# solver answers unanimously into harder variants (frozen proposer) -> guard the evolved pool (no tracers) -> train on the union -> evaluate.
# Arms: own (A3 loop) | own_ag (abstention guard, A4 addition 1) | coev (co-evolving proposer via filtered SFT, R-Zero band)
#       | coev_ag (co-evolving proposer + abstention guard). All arms train the solver with the own reward.
# Round 1 of coev reuses the own round-1 proposals / admitted set of the same seed (frozen proposer, base solver, same band:
# identical by construction). Usage: a4_round1.sh <lineage> <base_model> <arm> <seed> <round> <GPU>
set -u
L=$1; BASE=$2; ARM=$3; SEED=$4; RD=$5; G=$6
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=${L}_r${RD}; OUT=$B/runs/$NAME; mkdir -p $OUT
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
ulimit -c 0
log(){ echo "$(date '+%F %T') a4 $NAME: $*" >> $LOG; }
touch /dev/shm/gpu_reserved_$G; trap 'rm -f /dev/shm/gpu_reserved_$G' EXIT
if [ "$RD" -eq 1 ]; then PREV=$BASE; else PREV=$CK/${L}_r$((RD-1))/final; fi
[ -e $PREV/config.json ] || { log "previous solver missing: $PREV"; exit 3; }
ABS=0; case "$ARM" in pdc|own_ag|coev_ag) ABS=1;; esac
TARM=own; [ "$ARM" = pdc ] && TARM=pdc
GUARD=rzero; case "$ARM" in own_ag|coev_ag) GUARD=abstain;; esac
PROP=$BASE
case "$ARM" in coev|coev_ag)
  if [ "$RD" -gt 1 ]; then PROP=$CK/${L}_r$((RD-1))/proposer_Base; [ -e $PROP/config.json ] || { log "previous proposer missing: $PROP"; exit 3; }; fi;;
esac
if [ "$ARM" = coev ] && [ "$RD" -eq 1 ] && [ ! -s $OUT/admitted.jsonl ]; then
  SRC=$B/runs/${L%%_coev_*}_own_a3_s${SEED}_r1
  if [ -s $SRC/admitted.jsonl ] && [ -s $SRC/guard_stats.json ] && [ -s $SRC/proposed.jsonl ]; then
    cp $SRC/proposed.jsonl $SRC/admitted.jsonl $SRC/guard_stats.json $OUT/; cp $SRC/guard_records.jsonl $SRC/audit_admitted.json $OUT/ 2>/dev/null
    log "round-1 proposals/guard/audit copied from $(basename $SRC)"
  fi
fi
[ -s $OUT/proposed.jsonl ] || python3 $A3/code/a3_propose.py --model $PROP --out $OUT/proposed.jsonl --n 1200 --seed $((SEED*100+RD)) >> $OUT/propose.log 2>&1 || { log "propose failed"; exit 1; }
if [ ! -s $OUT/admitted.jsonl ]; then
  mkdir -p $OUT/fresh $OUT/evo
  [ -s $OUT/fresh/admitted.jsonl ] || python3 $A3/code/a3_guard.py --model $PREV --candidates $OUT/proposed.jsonl --out_dir $OUT/fresh --seed $((SEED*100+RD)) --abstain_prompt 0 --guard rzero >> $OUT/guard.log 2>&1 || { log "guard failed"; exit 1; }
  [ -s $OUT/evo/proposed.jsonl ] || python3 $A3/code/a4_evolve.py --model $BASE --pool $OUT/proposed.jsonl --records $OUT/fresh/guard_records.jsonl --out $OUT/evo/proposed.jsonl --k 2 --tag evo$((SEED*100+RD)) --seed $((SEED*100+RD)) >> $OUT/evolve.log 2>&1 || { log "evolve failed"; exit 1; }
  [ -s $OUT/evo/admitted.jsonl ] || python3 $A3/code/a3_guard.py --model $PREV --candidates $OUT/evo/proposed.jsonl --out_dir $OUT/evo --seed $((SEED*100+RD)) --abstain_prompt 0 --guard rzero --n_tracer 0 >> $OUT/guard.log 2>&1 || { log "evo guard failed"; exit 1; }
  cat $OUT/fresh/admitted.jsonl $OUT/evo/admitted.jsonl > $OUT/admitted.jsonl.tmp && mv $OUT/admitted.jsonl.tmp $OUT/admitted.jsonl
  cp $OUT/fresh/guard_stats.json $OUT/guard_stats.json; cat $OUT/fresh/guard_records.jsonl $OUT/evo/guard_records.jsonl > $OUT/guard_records.jsonl
  cat $OUT/evo/proposed.jsonl >> $OUT/proposed_all.jsonl 2>/dev/null; log "evolved $(wc -l < $OUT/evo/proposed.jsonl) from unanimous fresh items; evo admitted $(wc -l < $OUT/evo/admitted.jsonl)"
fi
log "guard(evo): $(cat $OUT/guard_stats.json | tr -d '\n ' | cut -c1-300)"
DATA=$OUT/admitted.jsonl
log "training data $(basename $DATA): $(wc -l < $DATA) items (proposer $(basename $PROP))"
if [ ! -f $OUT/.trained ]; then
  EXTRA="--answer_fallback 1 --vllm_inproc 1"; [ "$ARM" = pdc ] && EXTRA="$EXTRA --n_del 16"
  case "$BASE" in *8B*|*7B*) EXTRA="$EXTRA --optimizer_device cpu --vllm_util 0.22";; esac
  python3 $B/code/b_train.py --model $PREV --arm $TARM --seed $SEED --out $OUT --data $DATA --steps 60 $EXTRA >> $OUT/train.log 2>&1 < /dev/null; rc=$?
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  [ -f $OUT/.trained ] || { log "train stopped without finishing (rc=$rc)"; exit 2; }
  hostname > $OUT/host
fi
case "$ARM" in coev|coev_ag)
  if [ ! -f $OUT/.proposer ]; then
    python3 $A3/code/a4_proposer_sft.py --model $PROP --data $OUT/admitted.jsonl --proposed $OUT/proposed.jsonl --out $CK/$NAME/proposer_Base --seed $((SEED*100+RD)) >> $OUT/proposer_sft.log 2>&1 \
      && touch $OUT/.proposer || { log "proposer SFT failed"; exit 6; }
    log "proposer SFT done: $(cat $CK/$NAME/proposer_Base/sft_meta.json | tr -d '\n ')"
  fi;;
esac
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
