#!/bin/bash
# A7-5 waiter (addendum section 8, drain rule; decision D-host): polls this host every 60 s and, once it is free of our GPU
# markers, processes and job scripts (host_check in rz_common.sh), runs the smoke test (seed 0; a later seed skips it when
# a smoke test PASSED on the same software/hardware stack) and then rz_run.sh <S>. After a co-tenant yield (exit 6), a
# memory stop (7) or a lost preflight or clock error (3) it waits again and resumes. The projected-finish rule decides
# whether a seed still starts; a failed rz_clock.py call is a clock error, retried every 60 s (review item 2). When the
# waiter ends, the GPU markers that a passed smoke test kept and no run took over are removed (review item 3).
# Log: ext/rzero_runs/ledger_s<S>.log (a "host not free" or "clock error" line when the reason changes, and hourly).
# Start (seed 0 on host B -p <PORT>, seed 1 on host A -p <PORT>):
#   nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_wait_and_run.sh <S> > /dev/null 2>&1 &
# Stop together with its run: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh <S>
S=${1:-}; case "$S" in 0|1|2|3) ;; *) echo "usage: rz_wait_and_run.sh 0|1|2|3"; exit 2;; esac
[ -z "${RZ_A75_TAG:-}" ] || { echo "refusing: inside an A7-5 run environment"; exit 2; }
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_s$S; export STORAGE_PATH=$RZ/s$S; LEDGER=$X/ledger_s$S.log; W=$RZ/wait_s$S.pid
mkdir -p $STORAGE_PATH $X/s$S
source $D/rz_common.sh
r=$(seed_ok $S) || { L "waiter: $r"; echo "$r"; exit 2; }
p=$(cat $W 2>/dev/null); [ -n "$p" ] && grep -q rz_wait_and_run.sh /proc/$p/cmdline 2>/dev/null && { echo "a waiter for seed $S is already running (pid $p)"; exit 2; }
echo $$ > $W; trap 'drop_smoke_markers; rm -f $W' EXIT; trap 'L "waiter stopped by signal"; exit 143' TERM INT HUP
L "waiter start (pid $$; polls every 60 s)"
last=""; beat=0
while :; do
  [ "$(date +%s)" -lt $A75_STOP ] || { L "waiter: T0+71h reached; nothing more is started (X5)"; exit 0; }
  r=$(host_check); key=$(echo "$r" | cut -c1-40 | tr -d '0-9')
  if [ -n "$r" ]; then   # logged when the kind of reason changes, and hourly
    { [ "$key" = "$last" ] && [ $(( $(date +%s) - beat )) -lt 3600 ]; } || { L "waiter: host not free: $(echo $r | cut -c1-300)"; beat=$(date +%s); }
    last=$key; sleep 60; continue; fi
  need=1; smoke_not_needed $S && need=0
  # only full, p1only or none is a projection; anything else means that rz_clock.py failed (e.g. a transient NFS error)
  set -- $(/usr/bin/python3 $D/rz_clock.py mode --seed $S --storage $STORAGE_PATH --smoke $need 2>> $STORAGE_PATH/driver.log)
  case "${1:-}" in full|p1only) ;;
    none) L "waiter: NOT STARTED - the projected-finish rule leaves no time (X1; $*)"; exit 0;;
    *) { [ "$last" = clock ] && [ $(( $(date +%s) - beat )) -lt 3600 ]; } || { L "waiter: clock error: rz_clock.py printed '$(echo "$*" | cut -c1-120)' (its error is in $STORAGE_PATH/driver.log); retrying every 60 s"; beat=$(date +%s); }
       last=clock; sleep 60; continue;;
  esac
  last=""
  if [ $need = 1 ]; then
    L "waiter: host free; smoke test (projected mode $1, $2)"; bash $D/rz_smoke.sh; rc=$?
    case $rc in 0) ;; 3|6|7) L "waiter: smoke test exit $rc; waiting again"; sleep 60; continue;;
      8) L "waiter: smoke test stopped at T0+71h (X5); waiter ends"; exit 8;;
      *) L "waiter: smoke test exit $rc; waiter ends (X2: debugging by hand within the allowed changes)"; exit $rc;; esac
  fi
  L "waiter: starting rz_run.sh $S"; bash $D/rz_run.sh $S; rc=$?
  case $rc in 0) L "waiter: seed $S driver ended (exit 0); waiter ends"; exit 0;;
    3|6|7) L "waiter: driver exit $rc; waiting to resume"; sleep 60;;
    *) L "waiter: driver exit $rc; waiter ends"; exit $rc;; esac
done
