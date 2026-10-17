#!/bin/bash
# A3 single-GPU runner: claims round-tasks from $QUEUE (lineage|base_model|arm|seed|round) and runs
# a3_round1.sh on GPU $1 (propose, guard, in-process-vLLM training, evals all on one card).
# Same NFS claim / dependency protocol as a3_runner.sh. Usage: a3_runner1.sh <GPU>
G=$1
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; Q=${QUEUE:-$A3/queue_a3.txt}; RUNS=$B/runs; LOG=$B/results/runner.log
HOST=$(hostname)
log(){ echo "$(date '+%F %T') a3one$G@${HOST:0:12}: $*" >> $LOG; }
claim(){ local c=$RUNS/$1.claim
  if mkdir $c 2>/dev/null; then echo "$$ $HOST a3one$G $(date +%s)" > $c/owner; return 0; fi
  local opid ohost; opid=$(cut -d' ' -f1 $c/owner 2>/dev/null); ohost=$(cut -d' ' -f2 $c/owner 2>/dev/null)
  if [ -n "$opid" ] && [ "$ohost" = "$HOST" ] && ! kill -0 $opid 2>/dev/null; then rm -rf $c; mkdir $c 2>/dev/null && { echo "$$ $HOST a3one$G $(date +%s)" > $c/owner; log "took over stale claim on $1"; return 0; }; fi
  return 1; }
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '; }
idle=0
while :; do
  picked=""
  while IFS='|' read -r L base arm seed rd; do
    [ -z "$L" ] && continue; case $L in \#*) continue;; esac
    name=${L}_r${rd}
    [ -f $RUNS/$name/.done ] && continue
    if [ "$rd" -gt 1 ] && [ ! -f $RUNS/${L}_r$((rd-1))/.done ]; then continue; fi
    if [ "$rd" -gt 1 ] && [ ! -e /dev/shm/ckpt_b/${L}_r$((rd-1))/final/config.json ]; then continue; fi   # previous final must be on this host
    claim $name || continue
    picked=$name; break
  done < $Q
  if [ -z "$picked" ]; then idle=$((idle+1)); [ $idle -ge 6 ] && { log "nothing claimable, exiting"; exit 0; }; sleep 120; continue; fi
  idle=0
  if [ "$(used $G)" -gt 2000 ]; then log "GPU busy before $picked, waiting"; rm -rf $RUNS/$picked.claim; sleep 60; continue; fi
  log "start $picked ($base $arm seed $seed round $rd) on gpu $G"
  bash $A3/code/a3_round1.sh $L $base $arm $seed $rd $G; rc=$?
  if [ ! -f $RUNS/$picked/.done ]; then log "$picked not finished (rc=$rc); claim released"; rm -rf $RUNS/$picked.claim; sleep 60; continue; fi
  rm -rf $RUNS/$picked.claim
done
