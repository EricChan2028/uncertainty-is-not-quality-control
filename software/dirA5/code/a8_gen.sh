#!/bin/bash
# A8 (registered 23:16 24 Sep): generate the 8 plain-prompt solutions (texts saved) for the fixed item set with base and
# own_a3 r3/r5 of Qwen3-4B seeds 0-2. Atomic GPU claims (noclobber), one GPU per model, resumable (the probe skips done ids).
R=/home/tione/notebook/rsi_rl; LOG=$R/dirB/results/runner.log; OUT=$R/dirA5/results/a8/texts; IDS=$R/dirA5/results/a8/items.json
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
log(){ echo "$(date '+%F %T') a8gen: $*" >> $LOG; }
claim(){ while true; do for g in $(seq 0 7); do
  m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "); [ "$m" -lt 1500 ] || continue
  sleep 15; m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
  if [ "$m" -lt 1500 ] && ( set -o noclobber; : > /dev/shm/gpu_single_$g ) 2>/dev/null; then echo $g; return; fi
  done; sleep 20; done; }
run(){ local tag=$1 model=$2 g; g=$(claim); log "$tag on gpu $g"
  ( CUDA_VISIBLE_DEVICES=$g python3 $R/dirA3/code/umwp_probe_text.py --model $model --tag $tag --abstain_prompt 0 --ids $IDS --outdir $OUT >> $OUT/$tag.log 2>&1 < /dev/null \
      && log "$tag done" || log "$tag FAILED"; rm -f /dev/shm/gpu_single_$g ) & sleep 5; }
mkdir -p $OUT
run a8_base /root/models/Qwen3-4B-Base
for s in 0 1 2; do for r in 5 3; do run a8_own_s${s}_r${r} /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s${s}_r${r}/final; done; done
wait; log "all generation finished"
