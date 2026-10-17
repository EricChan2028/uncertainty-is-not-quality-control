#!/bin/bash
# A4i: one real loop round with the abstention-attractor reward. Round-4 proposals of own seed S (same frozen proposer,
# same file), guard = R-Zero band on the majority share counting "unknown" (abstention prompt, --count_unknown 1), solver =
# rescue_uv round-3 solver of seed S, reward own_uv, 60 steps; then sharded probes and b_eval on claimed free GPUs.
# Usage: a4_loop_uv_r.sh <seed> <round>   (A4j: round >= 5 continues the loop_uv lineage; waits for the previous round)
set -u
S=$1; RD=$2
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=qwen3_4b_Base_loop_uv_a4_s${S}_r${RD}; OUT=$B/runs/$NAME; mkdir -p $OUT
PN=qwen3_4b_Base_loop_uv_a4_s${S}_r$((RD-1)); PREV=$CK/$PN/final
until [ -f $B/runs/$PN/.trained ] && [ -e $PREV/config.json ]; do sleep 60; done
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
ulimit -c 0
log(){ echo "$(date '+%F %T') a4j $NAME: $*" >> $LOG; }
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
[ -s $OUT/proposed.jsonl ] || cp $B/runs/qwen3_4b_Base_own_a3_s${S}_r${RD}/proposed.jsonl $OUT/
if [ ! -s $OUT/admitted.jsonl ]; then
  G=$(claim); log "guard (count_unknown) on gpu $G"
  CUDA_VISIBLE_DEVICES=$G python3 $A3/code/a3_guard.py --model $PREV --candidates $OUT/proposed.jsonl --out_dir $OUT --seed $((S*100+RD)) --abstain_prompt 1 --guard rzero --count_unknown 1 >> $OUT/guard.log 2>&1 < /dev/null
  rm -f /dev/shm/gpu_single_$G
  [ -s $OUT/admitted.jsonl ] || { log "guard failed"; exit 1; }
  log "guard done: $(head -c 300 $OUT/guard_stats.json)"
fi
if [ ! -f $OUT/.trained ]; then
  G=$(claim); log "train own_uv on gpu $G ($(wc -l < $OUT/admitted.jsonl) items)"
  CUDA_VISIBLE_DEVICES=$G python3 $B/code/b_train.py --model $PREV --arm own_uv --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  rm -f /dev/shm/gpu_single_$G
  [ -f $OUT/.trained ] || { log "train stopped"; exit 2; }
fi
F=$CK/$NAME/final
P="python3 $R/dirA/code/c0_probe.py --model $F --gpu_util 0.85 --out $B/results/hp"
job hp0 "$P --tag hp_${NAME} --shard 0 --nshards 2 >> $OUT/eval_hp0.log 2>&1 < /dev/null"
job hp1 "$P --tag hp_${NAME} --shard 1 --nshards 2 >> $OUT/eval_hp1.log 2>&1 < /dev/null"
job hpabs0 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 2 >> $OUT/eval_hpabs0.log 2>&1 < /dev/null"
job hpabs1 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 1 --nshards 2 >> $OUT/eval_hpabs1.log 2>&1 < /dev/null"
job beval "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
wait
touch $OUT/.evaluated $OUT/.done; log "all evals finished"
