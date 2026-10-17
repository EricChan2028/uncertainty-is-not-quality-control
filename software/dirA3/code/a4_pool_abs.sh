#!/bin/bash
# A4g exploratory comparator: rescue_abs r3 solver (seed S) on its fully audited pool, abstention prompt, records only.
S=$1; R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; export HF_HOME=$R/hf_home; LOG=$B/results/runner.log
case $S in 0) PS=0; GS=1;; 1) PS=1; GS=2;; esac
F=/dev/shm/ckpt_b/qwen3_4b_Base_rescue_abs_a4_s${S}_r3/final; O=$A3/results/guard_diag/uv/pool_s$PS/rescue_abs_s$S; mkdir -p $O
[ -s $O/guard_records.jsonl ] && exit 0
while true; do for g in $(seq 0 7); do [ -e /dev/shm/gpu_single_$g ] && continue
  m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [ "$m" -lt 1500 ] || continue; sleep 15
  m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  [ "$m" -lt 1500 ] && [ ! -e /dev/shm/gpu_single_$g ] && { G=$g; break 2; }; done; sleep 30; done
touch /dev/shm/gpu_single_$G; echo "$(date '+%F %T') a4g pool_abs s$S on gpu $G" >> $LOG
CUDA_VISIBLE_DEVICES=$G python3 $A3/code/a3_guard.py --model $F --candidates $B/runs/qwen3_4b_Base_own_a3_s${PS}_r1/proposed.jsonl --out_dir $O --seed $GS --abstain_prompt 1 --guard rzero >> $O/guard.log 2>&1 < /dev/null
echo "$(date '+%F %T') a4g pool_abs s$S finished rc=$?" >> $LOG; rm -f /dev/shm/gpu_single_$G
