#!/bin/bash
# A9 (registered 00:00 25 Sep): fixed-pool guard passes for the per-class law in Qwen3-8B-Base s0 and Phi-4-mini s0, exactly
# as /root/guard_diag_qwen_s0.sh (same candidates, seed 1, plain prompt, R-Zero band). Atomic GPU claims; resumable.
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; export HF_HOME=$R/hf_home
CAND=$B/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl; LOG=$B/results/runner.log
log(){ echo "$(date '+%F %T') a9gdiag: $*" >> $LOG; }
claim(){ while true; do for g in $(seq 0 7); do
  m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [ "$m" -lt 1500 ] || continue
  sleep 15; m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  if [ "$m" -lt 1500 ] && ( set -o noclobber; : > /dev/shm/gpu_single_$g ) 2>/dev/null; then echo $g; return; fi
  done; sleep 20; done; }
run_ck(){ local fam=$1 n=$2 m=$3 o g; o=$A3/results/guard_diag/$fam/$n; mkdir -p $o
  [ -s $o/guard_stats.json ] && { log "$fam/$n already done"; return 0; }
  g=$(claim); log "$fam/$n on gpu $g"
  ( CUDA_VISIBLE_DEVICES=$g python3 $A3/code/a3_guard.py --model $m --candidates $CAND --out_dir $o --seed 1 --abstain_prompt 0 --guard rzero >> $o/guard.log 2>&1 \
      && log "$fam/$n done: $(head -c 160 $o/guard_stats.json)" || log "$fam/$n FAILED"; rm -f /dev/shm/gpu_single_$g ) & sleep 5; }
run_ck fam_qwen3_8b_s0 base /dev/shm/models/Qwen3-8B-Base
for r in 1 2 3; do run_ck fam_qwen3_8b_s0 r$r /dev/shm/ckpt_b/qwen3_8b_Base_own_a3_s0_r$r/final; done
run_ck fam_phi4mini_s0 base /dev/shm/models/Phi-4-mini-instruct
for r in 1 2 3; do run_ck fam_phi4mini_s0 r$r /run/ti/a9_ckpt/phi4mini_own_a4_s0_r$r; done
wait; log "all A9 guard passes finished"
