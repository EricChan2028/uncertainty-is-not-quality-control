#!/bin/bash
# rev6 (operations only): host B's GPUs are taken by the co-tenant, so the round-3 checkpoints that live only in host B's
# /dev/shm are copied to host A over the internal network (3 streams), each into a temporary directory and then renamed, so a
# checkpoint path appears on host A only once its copy is complete. Copies are deleted after the rev6 runs.
# (tar h: some host-B checkpoints are symlinks into /run/ti/rsi_ckpt_archive.)
# Usage (on host A): bash rev6_xfer.sh jobs_hostB.txt
D=/home/tione/notebook/rsi_rl/paper_arr/rev6; B=root@<PRIVATE_IP>; TMP=/dev/shm/ckpt_b_xfer_tmp
mkdir -p $TMP
one(){ name=$1
  [ -f /dev/shm/ckpt_b/$name/final/config.json ] && { echo "$(date '+%F %T') have $name" >> $D/logs/xfer.log; return; }
  rm -rf $TMP/$name
  if ssh -o BatchMode=yes $B "tar chf - -C /dev/shm/ckpt_b $name/final" | tar xf - -C $TMP && [ -f $TMP/$name/final/config.json ]; then
    mkdir -p /dev/shm/ckpt_b/$name && mv $TMP/$name/final /dev/shm/ckpt_b/$name/final && rm -rf $TMP/$name
    echo "$(date '+%F %T') copied $name $(du -sb /dev/shm/ckpt_b/$name/final | cut -f1) bytes" >> $D/logs/xfer.log
  else echo "$(date '+%F %T') FAILED $name" >> $D/logs/xfer.log; fi; }
export -f one; export D B TMP
awk '{print $3}' $1 | sed -n 's#^/dev/shm/ckpt_b/\([^/]*\)/final$#\1#p' | awk '!s[$0]++' | xargs -P 3 -I{} bash -c 'one {}'
echo "$(date '+%F %T') all copies of $1 finished" >> $D/logs/xfer.log
