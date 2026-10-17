#!/bin/bash
# A5 TG (registered 08:04): reward isolation at round 4. Start = rescue_uv r3 solver of seed S (as A4i loop_uv r4).
#   abs4: arm own_abs on the loop_uv r4 admitted set (prompt-, start- and item-matched control; unknown never rewarded)
#   t4:   arm own_uv on the tri-gate subset of the same records (a5_trigate.py)
# 60 steps, same trainer flags as A4i; evals: twins hp/hpabs (2 shards each), b_eval, UMWP plain+abs.
# Usage: a5_tg.sh <seed> <abs4|t4>   (on the host holding /dev/shm/ckpt_b/qwen3_4b_Base_rescue_uv_a4_s<S>_r3/final)
set -u
S=$1; ARM=$2
R=/home/tione/notebook/rsi_rl; B=$R/dirB; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
SRC=$B/runs/qwen3_4b_Base_loop_uv_a4_s${S}_r4
NAME=qwen3_4b_Base_tg_${ARM}_a5_s${S}_r4; OUT=$B/runs/$NAME; mkdir -p $OUT
PREV=$CK/qwen3_4b_Base_rescue_uv_a4_s${S}_r3/final
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
ulimit -c 0
log(){ echo "$(date '+%F %T') a5tg $NAME: $*" >> $LOG; }
claim(){
  while true; do
    for g in $(seq 0 7); do
      [ -e /dev/shm/gpu_single_$g ] && continue
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      [ "$m" -lt 1500 ] || continue
      sleep 15
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      if [ "$m" -lt 1500 ] && [ ! -e /dev/shm/gpu_single_$g ]; then touch /dev/shm/gpu_single_$g; echo $g; return; fi
    done
    sleep 20
  done
}
job(){ local tag=$1; shift; local g; g=$(claim); log "eval $tag on gpu $g"
  ( CUDA_VISIBLE_DEVICES=$g bash -c "$*" && log "eval $tag done" || log "eval $tag FAILED"; rm -f /dev/shm/gpu_single_$g ) &
  sleep 5; }
[ -e $PREV/config.json ] || { log "missing $PREV"; exit 3; }
if [ ! -s $OUT/admitted.jsonl ]; then
  if [ "$ARM" = t4 ]; then python3 $R/dirA5/code/a5_trigate.py $SRC $OUT >> $OUT/trigate.log 2>&1; TARM=own_uv
  else cp $SRC/admitted.jsonl $OUT/; TARM=own_abs; fi
fi
[ "$ARM" = t4 ] && TARM=own_uv || TARM=own_abs
if [ ! -f $OUT/.trained ]; then
  G=$(claim); log "train $TARM on gpu $G ($(wc -l < $OUT/admitted.jsonl) items)"
  CUDA_VISIBLE_DEVICES=$G python3 $B/code/b_train.py --model $PREV --arm $TARM --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  rm -f /dev/shm/gpu_single_$G
  [ -f $OUT/.trained ] || { log "train stopped"; exit 2; }
fi
F=$CK/$NAME/final
P="python3 $R/dirA/code/c0_probe.py --model $F --gpu_util 0.85 --out $B/results/hp"
U="python3 $R/dirA3/code/umwp_probe.py --model $F"
job hp0 "$P --tag hp_${NAME} --shard 0 --nshards 2 >> $OUT/eval_hp0.log 2>&1 < /dev/null"
job hp1 "$P --tag hp_${NAME} --shard 1 --nshards 2 >> $OUT/eval_hp1.log 2>&1 < /dev/null"
job hpabs0 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 2 >> $OUT/eval_hpabs0.log 2>&1 < /dev/null"
job hpabs1 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 1 --nshards 2 >> $OUT/eval_hpabs1.log 2>&1 < /dev/null"
job beval "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
job umwp "$U --tag plain_${NAME} --abstain_prompt 0 >> $OUT/eval_umwp.log 2>&1 < /dev/null && $U --tag abs_${NAME} --abstain_prompt 1 >> $OUT/eval_umwp.log 2>&1 < /dev/null"
wait
touch $OUT/.evaluated $OUT/.done; log "all evals finished"
