#!/bin/bash
# A4h launcher (one per host): runs the UMWP probe (plain + abstention prompt) for every listed checkpoint present on this
# host, claiming free GPUs (gpu_single markers); rescue_uv checkpoints wait for their .trained marker.
# Usage: umwp_launch.sh name1 name2 ...   (names = checkpoint dirs under /dev/shm/ckpt_b, or qwen3_4b_base)
R=/home/tione/notebook/rsi_rl; B=$R/dirB; LOG=$B/results/runner.log
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
mkdir -p $R/dirA4/results/umwp
claim(){
  while true; do
    for g in $(seq 0 7); do
      [ -e /dev/shm/gpu_single_$g ] && continue
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      [ "$m" -lt 1500 ] || continue
      sleep 15
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      if [ "$m" -lt 1500 ] && [ ! -e /dev/shm/gpu_single_$g ]; then touch /dev/shm/gpu_single_$g; echo $g; return; fi
    done
    sleep 30
  done
}
for n in "$@"; do
  if [ "$n" = qwen3_4b_base ]; then
    M=/root/models/Qwen3-4B-Base
  else
    M=/dev/shm/ckpt_b/$n/final
    case $n in *rescue_uv*) until [ -f $B/runs/$n/.trained ]; do sleep 60; done;; esac
  fi
  if [ ! -e $M/config.json ]; then echo "$(date '+%F %T') a4h umwp: $n missing on this host" >> $LOG; continue; fi
  for P in 0 1; do
    if [ $P = 1 ]; then t=abs_$n; else t=plain_$n; fi
    f=$R/dirA4/results/umwp/$t.jsonl
    if [ -f $f ] && [ $(wc -l < $f) -ge 1200 ]; then continue; fi
    g=$(claim)
    echo "$(date '+%F %T') a4h umwp $t on gpu $g" >> $LOG
    ( CUDA_VISIBLE_DEVICES=$g python3 $R/dirA3/code/umwp_probe.py --model $M --tag $t --abstain_prompt $P >> $R/dirA4/results/umwp/$t.log 2>&1 < /dev/null
      echo "$(date '+%F %T') a4h umwp $t finished rc=$?" >> $LOG
      rm -f /dev/shm/gpu_single_$g ) &
    sleep 5
  done
done
wait
