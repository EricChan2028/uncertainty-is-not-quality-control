#!/bin/bash
# A4e offline frontier-refill test on GPU 7 of this host: evolve the round-3 solver's unanimous candidates (fixed s0 pool),
# then score the evolved pool with the r3, r4, r5 solvers (R-Zero band, seed-1 tracers).
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; export HF_HOME=$R/hf_home CUDA_VISIBLE_DEVICES=7
log(){ echo "$(date '+%F %T') evo@$(hostname | cut -c1-12): $*" >> $B/results/runner.log; }
touch /dev/shm/gpu_single_7; trap 'rm -f /dev/shm/gpu_single_7' EXIT
P=$A3/results/frontier/evo_r3; mkdir -p $P
[ -s $P/proposed.jsonl ] || python3 $A3/code/a4_evolve.py --model /root/models/Qwen3-4B-Base --pool $B/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl \
   --records $A3/results/guard_diag/qwen_s0/r3/guard_records.jsonl --out $P/proposed.jsonl --k 2 --tag evo3 >> $P/evolve.log 2>&1
log "evolved pool: $(wc -l < $P/proposed.jsonl)"
for c in r3 r4 r5; do o=$A3/results/guard_diag/evo_r3/$c; mkdir -p $o; [ -s $o/guard_stats.json ] && continue
  python3 $A3/code/a3_guard.py --model /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_$c/final --candidates $P/proposed.jsonl --out_dir $o --seed 1 --abstain_prompt 0 --guard rzero >> $o/guard.log 2>&1
  log "evo guard $c done"; done
log "evo test finished"
