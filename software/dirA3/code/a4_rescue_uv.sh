#!/bin/bash
# A4g rescue with an abstention attractor: own r2 solver + own r3 admitted set, arm own_uv ("unknown" votes and is rewarded
# when it is the majority). Evaluations run in parallel on free GPUs of this host (claimed with gpu_single markers).
# Usage: a4_rescue_uv.sh <seed> <train_gpu>
set -u
S=$1; G=$2
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=qwen3_4b_Base_rescue_uv_a4_s${S}_r3; OUT=$B/runs/$NAME; mkdir -p $OUT
SRC=$B/runs/qwen3_4b_Base_own_a3_s${S}_r3; PREV=$CK/qwen3_4b_Base_own_a3_s${S}_r2/final
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
ulimit -c 0
log(){ echo "$(date '+%F %T') a4g $NAME: $*" >> $LOG; }
claim(){ while true; do for g in $(seq 0 7); do [ -e /dev/shm/gpu_single_$g ] && continue
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      [ "$m" -lt 1500 ] || continue; sleep 15
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      [ "$m" -lt 1500 ] && [ ! -e /dev/shm/gpu_single_$g ] && { touch /dev/shm/gpu_single_$g; echo $g; return; }
    done; sleep 30; done; }
job(){ g=$(claim); log "eval $1 on gpu $g"; ( export CUDA_VISIBLE_DEVICES=$g; shift_args="${@:2}"; eval "$shift_args" && log "eval $1 done" || log "eval $1 FAILED"; rm -f /dev/shm/gpu_single_$g ) & sleep 5; }
touch /dev/shm/gpu_single_$G
[ -e $PREV/config.json ] || { log "missing $PREV"; rm -f /dev/shm/gpu_single_$G; exit 3; }
[ -s $OUT/admitted.jsonl ] || cp $SRC/admitted.jsonl $SRC/guard_stats.json $OUT/
cp $SRC/audit_admitted.json $OUT/ 2>/dev/null
log "start: own r2 solver + own r3 admitted set ($(wc -l < $OUT/admitted.jsonl) items), reward own_uv, gpu $G"
if [ ! -f $OUT/.trained ]; then
  CUDA_VISIBLE_DEVICES=$G python3 $B/code/b_train.py --model $PREV --arm own_uv --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  rm -f /dev/shm/gpu_single_$G
  [ -f $OUT/.trained ] || { log "train stopped"; exit 2; }
fi
rm -f /dev/shm/gpu_single_$G
F=$CK/$NAME/final
P="python3 $R/dirA/code/c0_probe.py --model $F --gpu_util 0.85 --out $B/results/hp"
job hp0 "$P --tag hp_${NAME} --shard 0 --nshards 2 >> $OUT/eval_hp0.log 2>&1 < /dev/null"
job hp1 "$P --tag hp_${NAME} --shard 1 --nshards 2 >> $OUT/eval_hp1.log 2>&1 < /dev/null"
job hpabs0 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 2 >> $OUT/eval_hpabs0.log 2>&1 < /dev/null"
job hpabs1 "$P --tag hpabs_${NAME} --abstain_prompt 1 --shard 1 --nshards 2 >> $OUT/eval_hpabs1.log 2>&1 < /dev/null"
job beval "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
case $S in 0) PS=0; GS=1;; 1) PS=1; GS=2;; *) PS=;; esac
if [ -n "$PS" ]; then
  O=$A3/results/guard_diag/uv/pool_s$PS/rescue_uv_s$S; mkdir -p $O
  job pool "python3 $A3/code/a3_guard.py --model $F --candidates $B/runs/qwen3_4b_Base_own_a3_s${PS}_r1/proposed.jsonl --out_dir $O --seed $GS --abstain_prompt 1 --guard rzero >> $O/guard.log 2>&1 < /dev/null"
fi
job bevalabs "python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60_abs --abstain_prompt 1 --data math500_train,math_heldout1000,gsm8k_test500,hp_noop500 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null"
wait
touch $OUT/.evaluated $OUT/.done; log "all evals finished"
