#!/bin/bash
# A7-5: re-run the registered rz_read.py at the registered read times (T0+72h RZ-P2, T0+84h RZ-P1) and compare with the
# early read of 26 Sep 06:47 (run after all A7-5 compute had ended). Output: ext/rzero_runs/rz_reread.log
R=/home/tione/notebook/rsi_rl; X=$R/ext/rzero_runs; E=$X/rz_read_final_2026-09-26.json
for spec in "1790493060 T72h" "1790536260 T84h"; do
  set -- $spec; t=$1; tag=$2
  while [ "$(date +%s)" -lt "$t" ]; do sleep 60; done
  python3 $R/ext/rzero_design/rz_read.py > $X/rz_read_$tag.json
  if cmp -s $E $X/rz_read_$tag.json; then r=IDENTICAL; else r=DIFFERENT; fi
  echo "$(date '+%F %T %Z') $tag $(sha256sum $X/rz_read_$tag.json | cut -c1-64) vs early $(sha256sum $E | cut -c1-64): $r" >> $X/rz_reread.log
done
