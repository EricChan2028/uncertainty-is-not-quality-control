#!/bin/bash
# A6 X (registered 09:32): one round from own r2 of Qwen3-4B seed S on CLN1 or RND1 (a6_cln_build.py), arm own, plain prompt,
# 60 steps (same flags as a4_round1.sh); evals: twins hp/hpabs (2 shards each), b_eval, UMWP plain + abs.
# Usage: a6_cln.sh <seed> <cln1|rnd1>   (on a host holding /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s<S>_r2/final)
set -u
S=$1; ARM=$2
R=/home/tione/notebook/rsi_rl; B=$R/dirB; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=qwen3_4b_Base_${ARM}_a6_s${S}_r3; OUT=$B/runs/$NAME; PREV=$CK/qwen3_4b_Base_own_a3_s${S}_r2/final
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
ulimit -c 0
log(){ echo "$(date '+%F %T') a6x $NAME: $*" >> $LOG; }
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
[ -e $PREV/config.json ] || { log "missing $PREV"; exit 3; }
[ -s $OUT/admitted.jsonl ] || { log "missing admitted set (run a6_cln_build.py $S)"; exit 3; }
if [ ! -f $OUT/.trained ]; then
  G=$(claim); log "train own on gpu $G ($(wc -l < $OUT/admitted.jsonl) items)"
  CUDA_VISIBLE_DEVICES=$G python3 $B/code/b_train.py --model $PREV --arm own --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  rm -f /dev/shm/gpu_single_$G
  [ -f $OUT/.trained ] || { log "train stopped"; exit 2; }
fi
F=$CK/$NAME/final
P="python3 $R/dirA/code/c0_probe.py --model $F --gpu_util 0.85 --out $B/results/hp"
U="python3 $R/dirA3/code/umwp_probe.py --model $F"
ev(){ [ -f $OUT/.done ] && return 0; local tag=$1; shift; local g; g=$(claim); log "eval $tag on gpu $g"
  ( CUDA_VISIBLE_DEVICES=$g bash -c "$*" && log "eval $tag done" || log "eval $tag FAILED"; rm -f /dev/shm/gpu_single_$g ) & sleep 5; }
ev hp0 "$P --tag hp_${NAME} --shard 0 --nshards 2 >> $OUT/eval_hp0.log 2>&1 < /dev/null"
ev hp1 "$P --tag hp_${NAME} --shard 1 --nshards 2 >> $OUT/eval_hp1.log 2>&1 < /dev/null"
ev hpabs0 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 2 >> $OUT/eval_hpabs0.log 2>&1 < /dev/null"
ev hpabs1 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 1 --nshards 2 >> $OUT/eval_hpabs1.log 2>&1 < /dev/null"
ev beval "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
ev umwp "$U --tag plain_${NAME} --abstain_prompt 0 >> $OUT/eval_umwp.log 2>&1 < /dev/null && $U --tag abs_${NAME} --abstain_prompt 1 >> $OUT/eval_umwp.log 2>&1 < /dev/null"
wait
touch $OUT/.evaluated $OUT/.done; log "all evals finished"
