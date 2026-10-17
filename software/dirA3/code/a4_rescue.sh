#!/bin/bash
# A4f rescue at round 3: own round-2 solver + own round-3 admitted set, trained with the abstention-aware reward (arm own_abs).
# Usage: a4_rescue.sh <seed> <gpu>
set -u
S=$1; G=$2
R=/home/tione/notebook/rsi_rl; B=$R/dirB; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
NAME=qwen3_4b_Base_rescue_abs_a4_s${S}_r3; OUT=$B/runs/$NAME; mkdir -p $OUT
SRC=$B/runs/qwen3_4b_Base_own_a3_s${S}_r3; PREV=$CK/qwen3_4b_Base_own_a3_s${S}_r2/final
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$G
ulimit -c 0
log(){ echo "$(date '+%F %T') a4f $NAME: $*" >> $LOG; }
touch /dev/shm/gpu_single_$G; trap 'rm -f /dev/shm/gpu_single_$G' EXIT
[ -e $PREV/config.json ] || { log "missing $PREV"; exit 3; }
[ -s $OUT/admitted.jsonl ] || cp $SRC/admitted.jsonl $SRC/guard_stats.json $OUT/
cp $SRC/audit_admitted.json $OUT/ 2>/dev/null
log "start: own r2 solver + own r3 admitted set ($(wc -l < $OUT/admitted.jsonl) items), reward own_abs, gpu $G"
if [ ! -f $OUT/.trained ]; then
  python3 $B/code/b_train.py --model $PREV --arm own_abs --seed $S --out $OUT --data $OUT/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1 >> $OUT/train.log 2>&1 < /dev/null
  for p in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && kill -KILL $p; done
  [ -f $OUT/.trained ] || { log "train stopped"; exit 2; }
fi
F=$CK/$NAME/final
python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null \
 && python3 $R/dirA/code/c0_probe.py --model $F --tag hp_${NAME} --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $OUT/eval_hp.log 2>&1 < /dev/null \
 && python3 $R/dirA/code/c0_probe.py --model $F --tag hpabs_${NAME} --abstain_prompt 1 --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $OUT/eval_hp.log 2>&1 < /dev/null \
 && python3 $B/code/b_eval.py --model $F --tag ${NAME}_step60_abs --abstain_prompt 1 --data math500_train,math_heldout1000,gsm8k_test500,hp_noop500 --out $B/results/evals >> $OUT/eval.log 2>&1 < /dev/null \
 && touch $OUT/.evaluated $OUT/.done && log "done" || log "eval failed"
