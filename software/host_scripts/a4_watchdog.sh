#!/bin/bash
# Single-GPU watchdog: keeps one a4_runner1.sh on every GPU that is idle >= 60 s and not marked
# /dev/shm/gpu_single_<g> (kept for single-GPU eval jobs) or /dev/shm/gpu_reserved_<g> (a pair or single round in flight).
# A GPU owned by a pair runner is skipped unless /dev/shm/gpu_free_<g> exists (operator override: the pair's
# sequential-eval phase leaves that card idle for hours). Exits when every round-task in $QUEUE is .done. Sweeps orphan EngineCores.
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; LOG=$B/results/runner.log; export QUEUE=${QUEUE:-$A3/queue_a4_single.txt}
log(){ echo "$(date '+%F %T') a4wd@$(hostname | cut -c1-12): $*" >> $LOG; }
undone(){ local n=0; while IFS='|' read -r L b a s rd; do [ -z "$L" ] && continue; case $L in \#*) continue;; esac; [ -f $B/runs/${L}_r${rd}/.done ] || n=$((n+1)); done < $QUEUE; echo $n; }
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '; }
log "start v2 ($QUEUE)"
while :; do
  u=$(undone); [ "$u" -eq 0 ] && { log "all round-tasks done, exiting"; exit 0; }
  for g in 0 1 2 3 4 5 6 7; do
    [ -e /dev/shm/gpu_single_$g ] && continue
    [ -e /dev/shm/gpu_reserved_$g ] && [ ! -e /dev/shm/gpu_free_$g ] && continue
    pgrep -f "^bash $A3/code/a4_runner1.sh $g\$" >/dev/null && continue
    if [ ! -e /dev/shm/gpu_free_$g ]; then
      pgrep -f "b_runner2.sh [0-9] $g\$|b_runner2.sh $g [0-9]\$|a3_runner.sh [0-9] $g\$|a3_runner.sh $g [0-9]\$" >/dev/null && continue   # a pair owns it
    fi
    [ "$(used $g)" -gt 2000 ] && continue
    sleep 60; [ "$(used $g)" -gt 2000 ] && continue
    export HF_HOME=$R/hf_home
    rm -f /dev/shm/gpu_free_$g
    setsid nohup bash $A3/code/a4_runner1.sh $g > /dev/null 2>&1 < /dev/null &
    log "started single-GPU runner on gpu $g ($u undone)"; sleep 5
  done
  for pid in $(pgrep -f "^VLLM::EngineCore"); do [ "$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')" = "1" ] && { kill -KILL $pid 2>/dev/null; log "killed orphan EngineCore $pid"; }; done
  sleep 180
done
