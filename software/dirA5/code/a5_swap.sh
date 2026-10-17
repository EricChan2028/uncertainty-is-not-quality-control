#!/bin/bash
# A5 SWAP (registered 08:04), seed S: (1) flag pass: own r2 solver on the own r3 admitted set, abstention prompt, G = 8,
# no extra tracers; (2) wait for the full audit of the own r3 admitted proposals; (3) build G1 / C1 (a5_swap_build.py);
# (4) train each from own r2 (arm own, plain prompt, 60 steps, same flags as a4_round1.sh); (5) twins hp/hpabs + b_eval.
# Usage: a5_swap.sh <seed>   (on a host holding /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s<S>_r2/final)
set -u
S=$1
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
PREV=$CK/qwen3_4b_Base_own_a3_s${S}_r2/final; FL=$B/runs/swap_flag_a5_s${S}; mkdir -p $FL
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
ulimit -c 0
log(){ echo "$(date '+%F %T') a5swap s$S: $*" >> $LOG; }
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
if [ ! -s $FL/guard_records.jsonl ]; then
  G=$(claim); log "flag pass on gpu $G"
  CUDA_VISIBLE_DEVICES=$G python3 $A3/code/a3_guard.py --model $PREV --candidates $B/runs/qwen3_4b_Base_own_a3_s${S}_r3/admitted.jsonl --out_dir $FL --seed $((S*100+3)) --abstain_prompt 1 --guard rzero --n_tracer 0 >> $FL/guard.log 2>&1 < /dev/null
  rm -f /dev/shm/gpu_single_$G
  [ -s $FL/guard_records.jsonl ] || { log "flag pass failed"; exit 1; }
fi
NEED=$(wc -l < $B/runs/swap_audit_a5_s${S}/admitted.jsonl)
until [ "$(cat $B/runs/swap_audit_a5_s${S}/audit_parts/*.json 2>/dev/null | python3 -c 'import json,sys; t=sys.stdin.read(); import re; print(len(re.findall(r"\"verdict\"", t)))')" -ge "$NEED" ]; do sleep 60; done
[ -s $B/runs/qwen3_4b_Base_swap_c1_a5_s${S}_r3/admitted.jsonl ] || python3 $R/dirA5/code/a5_swap_build.py $S >> $FL/build.log 2>&1 || { log "build failed"; exit 1; }
log "built: $(tail -c 400 $FL/build.log)"
python3 -c "import json,sys; d=json.load(open('$FL/swap_build_stats.json')); sys.exit(1 if d['underpowered'] else 0)" || { log "underpowered (flagged < 10%), excluded"; exit 0; }
for ARM in g1 c1; do
  (
    NAME=qwen3_4b_Base_swap_${ARM}_a5_s${S}_r3; OUT=$B/runs/$NAME
    if [ ! -f $OUT/.trained ]; then
      G=$(claim); log "train $ARM on gpu $G ($(wc -l < $OUT/admitted.jsonl) items)"
      CUDA_VISIBLE_DEVICES=$G python3 $B/code/b_train.py --model $PREV --arm own --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
      for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
      rm -f /dev/shm/gpu_single_$G
      [ -f $OUT/.trained ] || { log "$ARM train stopped"; exit 2; }
    fi
    F=$CK/$NAME/final
    P="python3 $R/dirA/code/c0_probe.py --model $F --gpu_util 0.85 --out $B/results/hp"
    ev(){ [ -f $OUT/.done ] && return 0; local tag=$1; shift; local g; g=$(claim); log "$ARM eval $tag on gpu $g"
      ( CUDA_VISIBLE_DEVICES=$g bash -c "$*" && log "$ARM eval $tag done" || log "$ARM eval $tag FAILED"; rm -f /dev/shm/gpu_single_$g ) & sleep 5; }
    ev hp0 "$P --tag hp_${NAME} --shard 0 --nshards 2 >> $OUT/eval_hp0.log 2>&1 < /dev/null"
    ev hp1 "$P --tag hp_${NAME} --shard 1 --nshards 2 >> $OUT/eval_hp1.log 2>&1 < /dev/null"
    ev hpabs0 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 2 >> $OUT/eval_hpabs0.log 2>&1 < /dev/null"
    ev hpabs1 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 1 --nshards 2 >> $OUT/eval_hpabs1.log 2>&1 < /dev/null"
    ev beval "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
    wait; touch $OUT/.evaluated $OUT/.done; log "$ARM all evals finished"
  ) &
done
wait
