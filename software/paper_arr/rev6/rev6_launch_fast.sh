#!/bin/bash
# rev6: re-run the UMWP probe with texts saved (plan: paper_arr/rev6/rev6_plan_2026-10-06.md). One GPU per job, atomic GPU
# claims (noclobber lock + two memory checks 15 s apart, only GPUs with < 1500 MiB in use), resumable (the probe skips done ids).
# fast variant (host A only, 6 Oct 23:45): one memory check per claim, 5 s between starts, skips jobs already running. Variant for the checkpoints copied from host B (rev6_xfer.sh): waits until a model path exists before claiming a GPU.
# Usage: bash rev6_launch_xfer.sh JOBFILE MAXGPU   (JOBFILE lines: plain|absfix  TAG  MODEL_PATH)
R=/home/tione/notebook/rsi_rl; D=$R/paper_arr/rev6; OUT=$D/texts; LOGD=$D/logs; IDS=$D/ids_all.json
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
mkdir -p $OUT $LOGD
JOBS=$1; MAXG=${2:-5}; HOST=$(hostname)
log(){ echo "$(date '+%F %T') $HOST rev6: $*" >> $LOGD/launch.log; }
claim(){ while true; do
  n=$(ls /dev/shm/gpu_single_* 2>/dev/null | xargs -r -n1 cat 2>/dev/null | grep -c "^rev6$")
  if [ "$n" -lt "$MAXG" ]; then
    for g in $(seq 0 7); do
      m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
      if [ "$m" -lt 1500 ] && ( set -o noclobber; echo rev6 > /dev/shm/gpu_single_$g ) 2>/dev/null; then echo $g; return; fi
    done
  fi
  sleep 10; done; }
while read -r kind tag model; do
  [ -z "$kind" ] && continue
  case $kind in
    plain)  script=$R/dirA3/code/umwp_probe_text.py; flag=0;;
    absfix) script=$D/umwp_probe_text_absfix.py; flag=1;;
    *) continue;;
  esac
  out=$OUT/${kind}_${tag}.jsonl
  if [ -f "$out" ] && [ "$(wc -l < $out)" -ge 1200 ]; then log "skip $kind $tag (complete)"; continue; fi
  if pgrep -f -- "--tag ${kind}_${tag} " > /dev/null; then log "skip $kind $tag (running)"; continue; fi
  [ -e "$model/config.json" ] || log "wait for $model"; while [ ! -e "$model/config.json" ]; do sleep 10; done
  g=$(claim); log "start $kind $tag on gpu $g"
  ( CUDA_VISIBLE_DEVICES=$g python3 $script --model $model --tag ${kind}_${tag} --abstain_prompt $flag --ids $IDS --outdir $OUT \
      > $LOGD/${kind}_${tag}.log 2>&1 < /dev/null && log "done $kind $tag" || log "FAILED $kind $tag"; rm -f /dev/shm/gpu_single_$g ) &
  sleep 5
done < $JOBS
wait; log "all jobs of $JOBS finished"
