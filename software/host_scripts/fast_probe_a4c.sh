#!/bin/bash
# A4c fast path: as soon as each confirmatory round-3 solver is trained and a GPU on this host has been free for 90 s
# (no single/reserved marker), run the plain-prompt GSM-HP probe under tag hpq_<run> (same code/data as hp_<run>).
R=/home/tione/notebook/rsi_rl; B=$R/dirB; export HF_HOME=$R/hf_home
log(){ echo "$(date "+%F %T") fastprobe@$(hostname | cut -c1-12): $*" >> $B/results/runner.log; }
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d " "; }
for n in qwen3_4b_Base_own_clean_a3_s2_r3 qwen3_4b_Base_own_clean_a3_s3_r3 qwen3_4b_Base_own_sub_a3_s3_r3; do
  until [ -f $B/runs/$n/.trained ]; do sleep 60; done
  ls $B/results/hp/ | grep -q "^hp_${n}_shard" && { log "$n already probed"; continue; }
  g=""; while [ -z "$g" ]; do
    for c in 0 1 2 3 4 5 6 7; do [ -e /dev/shm/gpu_single_$c ] && continue; [ -e /dev/shm/gpu_reserved_$c ] && continue; [ "$(used $c)" -lt 2000 ] || continue; sleep 90; [ "$(used $c)" -lt 2000 ] && [ ! -e /dev/shm/gpu_reserved_$c ] && { g=$c; break; }; done
    [ -z "$g" ] && sleep 60
  done
  touch /dev/shm/gpu_single_$g; log "hpq probe $n on gpu $g"
  CUDA_VISIBLE_DEVICES=$g python3 $R/dirA/code/c0_probe.py --model /dev/shm/ckpt_b/$n/final --tag hpq_$n --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $B/runs/$n/eval_hpq.log 2>&1 < /dev/null && log "hpq $n done" || log "hpq $n FAILED"
  rm -f /dev/shm/gpu_single_$g
done
