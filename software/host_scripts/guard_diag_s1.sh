#!/bin/bash
# A4a.1: held-out validation pool (own s1 round-1 proposals + seed-2 tracers) scored with the base and own-s1 r1..r5 solvers,
# plain R-Zero guard (qwen_s1) and abstention guard (qwen_s1_ag). GPUs 1, 3, 6 of host <HOST> (gpu_single markers).
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; export HF_HOME=$R/hf_home
CAND=$B/runs/qwen3_4b_Base_own_a3_s1_r1/proposed.jsonl
log(){ echo "$(date '+%F %T') gdiag_s1@$(hostname | cut -c1-12): $*" >> $B/results/runner.log; }
ck(){ [ "$1" = base ] && echo /root/models/Qwen3-4B-Base || echo /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s1_$1/final; }
run(){ g=$1; mode=$2; c=$3
  if [ "$mode" = plain ]; then o=$A3/results/guard_diag/qwen_s1/$c; F="--abstain_prompt 0 --guard rzero"; else o=$A3/results/guard_diag/qwen_s1_ag/$c; F="--abstain_prompt 1 --guard abstain --abs_max 0.25"; fi
  mkdir -p $o; [ -s $o/guard_stats.json ] && return 0
  log "$mode $c on gpu $g"
  CUDA_VISIBLE_DEVICES=$g python3 $A3/code/a3_guard.py --model $(ck $c) --candidates $CAND --out_dir $o --seed 2 $F >> $o/guard.log 2>&1 && log "$mode $c done" || log "$mode $c FAILED"
}
touch /dev/shm/gpu_single_1 /dev/shm/gpu_single_3 /dev/shm/gpu_single_6
( for c in base r1 r2 r3; do run 1 plain $c; done; rm -f /dev/shm/gpu_single_1 ) &
( for c in r4 r5 base r1; do [ $c = base ] || [ $c = r1 ] && m=abs || m=plain; run 3 $m $c; done; rm -f /dev/shm/gpu_single_3 ) &
( for c in r2 r3 r4 r5; do run 6 abs $c; done; rm -f /dev/shm/gpu_single_6 ) &
wait
log "validation-pool guard passes finished"
