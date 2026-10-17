#!/bin/bash
# A5 HZN (registered 08:04): continue the abstention-guard lineage (own_ag) of seed S through rounds 4 and 5, paired with the
# own lineage's raw round-4/5 proposal files; guard = band + reject if >= 2/8 abstain (abstention prompt), guard seed
# S*100+RD as in own; training = arm own, plain prompt, 60 steps (same flags as a4_round1.sh). Evals per round launched in
# the background on claimed GPUs (twins hp/hpabs 2 shards each, b_eval; UMWP at round 5) while the next round proceeds.
# Usage: a5_hzn.sh <seed>   (on the host holding /dev/shm/ckpt_b/qwen3_4b_Base_own_ag_a4_s<S>_r3/final)
set -u
S=$1
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
L=qwen3_4b_Base_own_ag_a4_s${S}
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
ulimit -c 0
claim(){
  while true; do
    for g in $(seq 0 7); do
      [ -e /dev/shm/gpu_single_$g ] && continue
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      [ "$m" -lt 1500 ] || continue
      sleep 15
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      if [ "$m" -lt 1500 ] && ( set -o noclobber; : > /dev/shm/gpu_single_$g ) 2>/dev/null; then echo $g; return; fi
    done
    sleep 20
  done
}
for RD in 4 5; do
  NAME=${L}_r${RD}; OUT=$B/runs/$NAME; mkdir -p $OUT
  log(){ echo "$(date '+%F %T') a5hzn $NAME: $*" >> $LOG; }
  PREV=$CK/${L}_r$((RD-1))/final
  [ -e $PREV/config.json ] || { log "previous solver missing: $PREV"; exit 3; }
  [ -s $OUT/proposed.jsonl ] || cp $B/runs/qwen3_4b_Base_own_a3_s${S}_r${RD}/proposed.jsonl $OUT/
  if [ ! -s $OUT/admitted.jsonl ]; then
    G=$(claim); log "guard (abstain) on gpu $G"
    CUDA_VISIBLE_DEVICES=$G python3 $A3/code/a3_guard.py --model $PREV --candidates $OUT/proposed.jsonl --out_dir $OUT --seed $((S*100+RD)) --abstain_prompt 1 --guard abstain --abs_max 0.25 >> $OUT/guard.log 2>&1 < /dev/null
    rm -f /dev/shm/gpu_single_$G
    [ -s $OUT/admitted.jsonl ] || { log "guard failed"; exit 1; }
    log "guard done: $(head -c 300 $OUT/guard_stats.json)"
  fi
  if [ ! -f $OUT/.trained ]; then
    G=$(claim); log "train own on gpu $G ($(wc -l < $OUT/admitted.jsonl) items)"
    CUDA_VISIBLE_DEVICES=$G python3 $B/code/b_train.py --model $PREV --arm own --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
    for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
    rm -f /dev/shm/gpu_single_$G
    [ -f $OUT/.trained ] || { log "train stopped"; exit 2; }
  fi
  F=$CK/$NAME/final
  P="python3 $R/dirA/code/c0_probe.py --model $F --gpu_util 0.85 --out $B/results/hp"
  (
    ev(){ [ -f $OUT/.done ] && return 0; local tag=$1; shift; local g; g=$(claim); log "eval $tag on gpu $g"
      ( CUDA_VISIBLE_DEVICES=$g bash -c "$*" && log "eval $tag done" || log "eval $tag FAILED"; rm -f /dev/shm/gpu_single_$g ) & sleep 5; }
    ev hp0 "$P --tag hp_${NAME} --shard 0 --nshards 2 >> $OUT/eval_hp0.log 2>&1 < /dev/null"
    ev hp1 "$P --tag hp_${NAME} --shard 1 --nshards 2 >> $OUT/eval_hp1.log 2>&1 < /dev/null"
    ev hpabs0 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 2 >> $OUT/eval_hpabs0.log 2>&1 < /dev/null"
    ev hpabs1 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 1 --nshards 2 >> $OUT/eval_hpabs1.log 2>&1 < /dev/null"
    ev beval "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
    if [ "$RD" = 5 ]; then
      ev umwp "python3 $R/dirA3/code/umwp_probe.py --model $F --tag plain_${NAME} --abstain_prompt 0 >> $OUT/eval_umwp.log 2>&1 < /dev/null && python3 $R/dirA3/code/umwp_probe.py --model $F --tag abs_${NAME} --abstain_prompt 1 >> $OUT/eval_umwp.log 2>&1 < /dev/null"
    fi
    wait; touch $OUT/.evaluated $OUT/.done; log "all evals finished"
  ) &
done
wait
