#!/bin/bash
# Direction B per-GPU queue runner. Claims the next undone run from dirB/queue.txt (atomic mkdir on the
# shared volume), trains it (b_train.py resumes from /dev/shm/ckpt_b/<run>/resume if present), evaluates
# the final checkpoint (b_eval.py on the three datasets, c0_probe.py on the GSM-HP probe items), marks
# .done, releases the claim, and loops until the queue has nothing left. A run that exits without the
# .trained marker (preempted or crashed) has its claim released so any runner can resume it later.
G=$1
R=/home/tione/notebook/rsi_rl; B=$R/dirB; Q=$B/queue.txt; RUNS=$B/runs; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
export CUDA_VISIBLE_DEVICES=$G HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
mkdir -p $RUNS $B/results/evals $B/results/hp $B/results/logs
log(){ echo "$(date '+%F %T') gpu$G: $*" >> $LOG; }
claim(){ # name -> 0 if we hold the claim
  local c=$RUNS/$1.claim
  if mkdir $c 2>/dev/null; then echo "$$ gpu$G $(date +%s)" > $c/owner; return 0; fi
  local opid; opid=$(cut -d' ' -f1 $c/owner 2>/dev/null)
  if [ -n "$opid" ] && ! kill -0 $opid 2>/dev/null; then       # stale claim from a dead runner (same host)
    rm -rf $c; mkdir $c 2>/dev/null && { echo "$$ gpu$G $(date +%s)" > $c/owner; log "took over stale claim on $1"; return 0; }
  fi
  return 1
}
gpu_busy(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" -gt 2000 ]; }
idle_rounds=0
while :; do
  picked=""
  while IFS="|" read -r name model arm seed extra; do
    [ -z "$name" ] && continue; case $name in \#*) continue;; esac
    [ -f $RUNS/$name/.done ] && continue
    claim $name || continue
    picked=$name; break
  done < $Q
  if [ -z "$picked" ]; then
    idle_rounds=$((idle_rounds+1)); [ $idle_rounds -ge 3 ] && { log "queue exhausted, exiting"; exit 0; }
    sleep 120; continue
  fi
  idle_rounds=0
  if gpu_busy; then log "GPU busy before $picked, waiting"; rm -rf $RUNS/$picked.claim; sleep 60; continue; fi
  mkdir -p $RUNS/$picked
  if [ ! -f $RUNS/$picked/.trained ]; then
    log "train $picked ($model $arm seed $seed)"
    python3 $B/code/b_train.py --model $model --arm $arm --seed $seed --out $RUNS/$picked $extra >> $RUNS/$picked/train.log 2>&1 < /dev/null
    rc=$?
    if [ ! -f $RUNS/$picked/.trained ]; then log "$picked stopped without finishing (rc=$rc); claim released"; rm -rf $RUNS/$picked.claim; sleep 30; continue; fi
    log "$picked trained"
  fi
  if [ ! -f $RUNS/$picked/.evaluated ]; then
    log "eval $picked"
    python3 $B/code/b_eval.py --model $CK/$picked/final --tag ${picked}_step60 --out $B/results/evals >> $RUNS/$picked/eval.log 2>&1 < /dev/null \
      && python3 $R/dirA/code/c0_probe.py --model $CK/$picked/final --tag hp_${picked} --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $RUNS/$picked/eval.log 2>&1 < /dev/null \
      && touch $RUNS/$picked/.evaluated || { log "$picked eval failed; claim released"; rm -rf $RUNS/$picked.claim; sleep 30; continue; }
    log "$picked evaluated"
  fi
  touch $RUNS/$picked/.done; rm -rf $RUNS/$picked.claim
done
