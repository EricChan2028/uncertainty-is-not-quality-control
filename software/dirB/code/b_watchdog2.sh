#!/bin/bash
# Local-disk watchdog for paired-GPU runs (cycle 1+): keeps one b_runner2.sh per GPU pair alive while the
# queue named in $QUEUE (default dirB/queue.txt) has undone runs; sweeps orphan EngineCores. Pairs are
# (server, trainer) = (0,1) (2,3) (4,5) (6,7).
R=/home/tione/notebook/rsi_rl; B=$R/dirB; LOG=$B/results/runner.log; export QUEUE=${QUEUE:-$B/queue.txt}
PAIRS=${PAIRS:-"0 1|2 3|4 5|6 7"}                      # server/trainer GPU pairs; override per host, e.g. PAIRS="0 1|3 4|5 6"
log(){ echo "$(date '+%F %T') watchdog2@$(hostname | cut -c1-12): $*" >> $LOG; }
undone(){ local n=0; while IFS='|' read -r name rest; do [ -z "$name" ] && continue; case $name in \#*) continue;; esac; [ -f $B/runs/$name/.done ] || n=$((n+1)); done < $QUEUE; echo $n; }
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '; }
log "start ($QUEUE)"
while :; do
  u=$(undone); [ "$u" -eq 0 ] && { log "all runs done, exiting"; exit 0; }
  pairs="$PAIRS"
  if [ "$PAIRS" = "dynamic" ]; then                       # pair up whatever GPUs are free right now (co-tenant leaving piecemeal)
    free=""; for g in 0 1 2 3 4 5 6 7; do pgrep -f "b_runner2.sh [0-9] $g\$|b_runner2.sh $g [0-9]\$" >/dev/null && continue; [ "$(used $g)" -gt 2000 ] && continue; free="$free $g"; done
    set -- $free; pairs=""; while [ $# -ge 2 ]; do pairs="$pairs$1 $2|"; shift 2; done
  fi
  echo "$pairs" | tr "|" "\n" | while read -r a b; do
    [ -z "$b" ] && continue
    pgrep -f "^bash $B/code/b_runner2.sh $a $b\$" >/dev/null && continue
    { [ "$(used $a)" -gt 2000 ] || [ "$(used $b)" -gt 2000 ]; } && continue
    sleep 60                                                # the pair must be idle for a full minute (co-tenant rule)
    { [ "$(used $a)" -gt 2000 ] || [ "$(used $b)" -gt 2000 ]; } && continue
    setsid nohup bash $B/code/b_runner2.sh $a $b > /dev/null 2>&1 < /dev/null &
    log "started pair runner $a/$b ($u undone)"; sleep 5
  done
  for pid in $(pgrep -f "^VLLM::EngineCore"); do
    [ "$(ps -o ppid= -p $pid 2>/dev/null | tr -d ' ')" = "1" ] && { kill -KILL $pid 2>/dev/null; log "killed orphan EngineCore $pid"; }
  done
  sleep 180
done
