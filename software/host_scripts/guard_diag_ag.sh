#!/bin/bash
# A4 pilot (PF1): the fixed pool of A3 section 2d (own s0 round-1 proposals, 1,158 audited candidates + the seed-1 tracers)
# scored by the ABSTENTION guard (abstention prompt, reject when >= 2/8 samples abstain, then the R-Zero band) with the
# base solver and the own s0 r1..r5 solvers. Output dirA3/results/guard_diag/qwen_s0_ag/<ck>/. Two idle GPUs of 11067.
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; D=$A3/results/guard_diag/qwen_s0_ag; export HF_HOME=$R/hf_home
CAND=$B/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl
log(){ echo "$(date '+%F %T') gdiag_ag@$(hostname | cut -c1-12): $*" >> $B/results/runner.log; }
run_ck(){ g=$1; n=$2; m=$3; o=$D/$n; mkdir -p $o
  [ -s $o/guard_stats.json ] && { log "$n already done"; return 0; }
  log "abstention-guard diag $n on gpu $g"
  CUDA_VISIBLE_DEVICES=$g python3 $A3/code/a3_guard.py --model $m --candidates $CAND --out_dir $o --seed 1 --abstain_prompt 1 --guard abstain --abs_max 0.25 >> $o/guard.log 2>&1 \
    && log "abstention-guard diag $n done: $(head -c 200 $o/guard_stats.json)" || log "abstention-guard diag $n FAILED"
}
touch /dev/shm/gpu_single_1 /dev/shm/gpu_single_3
( run_ck 1 base /root/models/Qwen3-4B-Base; run_ck 1 r1 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_r1/final; run_ck 1 r2 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_r2/final; rm -f /dev/shm/gpu_single_1 ) &
( run_ck 3 r3 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_r3/final; run_ck 3 r4 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_r4/final; run_ck 3 r5 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_r5/final; rm -f /dev/shm/gpu_single_3 ) &
wait
log "abstention-guard diag finished"
