#!/bin/bash
# A4b frontier-supply test. Usage: frontier.sh <tag> <proposer_path> <propose_seed> <no_think> <gpu> <cks...>
# If proposed.jsonl is missing, this GPU generates it (only the first invocation passes "propose"); then guard passes
# (R-Zero band, seed-1 tracers) with the own-s0 solvers listed.
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; export HF_HOME=$R/hf_home
TAG=$1; PROP=$2; PSEED=$3; NT=$4; g=$5; shift 5
P=$A3/results/frontier/$TAG; mkdir -p $P; D=$A3/results/guard_diag/front_$TAG
log(){ echo "$(date '+%F %T') frontier_$TAG@$(hostname | cut -c1-12): $*" >> $B/results/runner.log; }
touch /dev/shm/gpu_single_$g; trap 'rm -f /dev/shm/gpu_single_$g' EXIT
export CUDA_VISIBLE_DEVICES=$g
for c in "$@"; do
  if [ "$c" = propose ]; then
    [ -s $P/proposed.jsonl ] || { log "propose with $(basename $PROP) on gpu $g"; python3 $A3/code/a3_propose.py --model $PROP --out $P/proposed.jsonl --n 1200 --seed $PSEED --no_think $NT >> $P/propose.log 2>&1 && log "proposed $(wc -l < $P/proposed.jsonl)" || { log "propose FAILED"; exit 1; }; }
    continue
  fi
  w=0; while [ ! -s $P/proposed.jsonl ]; do sleep 30; w=$((w+1)); [ $w -gt 60 ] && { log "no pool after 30 min"; exit 1; }; done
  m=/root/models/Qwen3-4B-Base; [ "$c" = base ] || m=/dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_$c/final
  o=$D/$c; mkdir -p $o; [ -s $o/guard_stats.json ] && continue
  log "guard $c on gpu $g"
  python3 $A3/code/a3_guard.py --model $m --candidates $P/proposed.jsonl --out_dir $o --seed 1 --abstain_prompt 0 --guard rzero >> $o/guard.log 2>&1 && log "guard $c done" || log "guard $c FAILED"
done
