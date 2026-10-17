#!/bin/bash
# A7-5 (f): stop the A7-5 run of seed <S> on this host at once (co-tenant priority, or by hand): its waiter
# (rz_wait_and_run.sh), its driver and every process carrying RZ_A75_TAG=rzA75_s<S>; removes its GPU markers and those a
# passed smoke test kept that no live run holds (review item 3); keeps every file.
# Resume later: nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_wait_and_run.sh <S> > /dev/null 2>&1 &
# 'abandon <reason>' also marks seed <S> abandoned (addendum section 7; may be run on any host): it is reported, never
# resumed and never counted, and a restart elsewhere uses the next free replacement seed id (2, then 3).
# (The monitor's own automatic yield, exit 6, keeps the waiter, which resumes when the host is free again.)
# Usage: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh <0|1|2|3|smoke> [abandon "<reason>"]
S=${1:-}; case "$S" in 0|1|2|3|smoke) ;; *) echo "usage: rz_stop.sh 0|1|2|3|smoke [abandon <reason>]"; exit 2;; esac
[ -z "${RZ_A75_TAG:-}" ] || { echo "run this from a normal shell"; exit 2; }
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_s$S; STORAGE_PATH=$RZ/s$S; LEDGER=$X/ledger_s$S.log
[ "$S" = smoke ] && { TAG=rzA75_smoke; STORAGE_PATH=$RZ/smoke; LEDGER=$X/ledger_smoke.log; }
mkdir -p $STORAGE_PATH; source $D/rz_common.sh
if [ "${2:-}" = abandon ]; then
  [ "$S" != smoke ] && [ -n "${3:-}" ] || { echo "usage: rz_stop.sh <0|1|2|3> abandon \"<reason>\""; exit 2; }
  ( set -o noclobber; echo "$(date +%s) $(hostname) $3" > $X/s$S.abandoned ) 2>/dev/null || { echo "seed $S is already abandoned: $(cat $X/s$S.abandoned)"; }
  L "seed $S ABANDONED by rz_stop.sh on $(hostname): $3"
fi
# a smoke test started by this seed's waiter is stopped with it
wp=$(cat $RZ/wait_s$S.pid 2>/dev/null); sp=$(cat $RZ/smoke/driver.pid 2>/dev/null)
[ "$S" != smoke ] && [ -n "$wp" ] && [ -n "$sp" ] && [ "$(ps -o ppid= -p $sp 2>/dev/null | tr -d ' ')" = "$wp" ] && bash $D/rz_stop.sh smoke
for f in $RZ/wait_s$S.pid $STORAGE_PATH/driver.pid; do
  p=$(cat $f 2>/dev/null); [ -n "$p" ] && grep -q 'rz_wait_and_run\.sh\|rz_run\.sh\|rz_smoke\.sh' /proc/$p/cmdline 2>/dev/null && kill -TERM $p
done
kill_tag $TAG
for g in 0 1 2 3 4 5 6 7; do [ "$(m_field $g 1)" = "$TAG" ] && rm -f $MK/gpu_single_$g; done
drop_smoke_markers
L "stopped by rz_stop.sh"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -8
