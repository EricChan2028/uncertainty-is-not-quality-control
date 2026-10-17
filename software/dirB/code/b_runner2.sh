#!/bin/bash
# Direction B paired-GPU runner (cycle 1+): trainer on GPU $2, vLLM rollout server on GPU $1, weights
# handed over through /dev/shm/rollout_<run>. Same queue/claim/marker protocol as b_runner.sh; the
# queue line may carry extra trainer args in a 5th field.
GA=$1; GB=$2
R=/home/tione/notebook/rsi_rl; B=$R/dirB; Q=${QUEUE:-$B/queue.txt}; RUNS=$B/runs; CK=/dev/shm/ckpt_b; LOG=$B/results/runner.log
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
mkdir -p $RUNS $B/results/evals $B/results/hp
log(){ echo "$(date '+%F %T') pair$GA/$GB: $*" >> $LOG; }
HOST=$(hostname)
claim(){ local c=$RUNS/$1.claim                       # claims are shared over NFS by every host; only a same-host dead owner is stale
  if mkdir $c 2>/dev/null; then echo "$$ $HOST pair$GA/$GB $(date +%s)" > $c/owner; return 0; fi
  local opid ohost; opid=$(cut -d' ' -f1 $c/owner 2>/dev/null); ohost=$(cut -d' ' -f2 $c/owner 2>/dev/null)
  if [ -n "$opid" ] && [ "$ohost" = "$HOST" ] && ! kill -0 $opid 2>/dev/null; then rm -rf $c; mkdir $c 2>/dev/null && { echo "$$ $HOST pair$GA/$GB $(date +%s)" > $c/owner; log "took over stale claim on $1"; return 0; }; fi
  return 1; }
used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '; }
stop_server(){ touch $RD/stop 2>/dev/null; sleep 3; [ -n "$SPID" ] && kill -- -$SPID 2>/dev/null; sleep 3; [ -n "$SPID" ] && kill -9 -- -$SPID 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GA 2>/dev/null | tr -d ' '); do kill -9 $p 2>/dev/null; done; }
idle=0
while :; do
  picked=""
  while IFS='|' read -r name model arm seed extra; do
    [ -z "$name" ] && continue; case $name in \#*) continue;; esac
    [ -f $RUNS/$name/.done ] && continue
    [ -f $RUNS/$name/.trained ] && [ ! -e $CK/$name/final/config.json ] && continue   # final lives on another host
    claim $name || continue
    picked=$name; break
  done < $Q
  if [ -z "$picked" ]; then idle=$((idle+1)); [ $idle -ge 3 ] && { log "queue exhausted, exiting"; exit 0; }; sleep 120; continue; fi
  idle=0
  if [ "$(used $GA)" -gt 2000 ] || [ "$(used $GB)" -gt 2000 ]; then log "GPUs busy before $picked, waiting"; rm -rf $RUNS/$picked.claim; sleep 60; continue; fi
  mkdir -p $RUNS/$picked
  touch /dev/shm/gpu_reserved_$GA /dev/shm/gpu_reserved_$GB       # single-GPU jobs skip reserved cards while this pair works
  trap 'rm -f /dev/shm/gpu_reserved_$GA /dev/shm/gpu_reserved_$GB' EXIT
  RD=/dev/shm/rollout_$picked; SPID=""
  if [ ! -f $RUNS/$picked/.trained ]; then
    rm -rf $RD; mkdir -p $RD
    log "train $picked ($model $arm seed $seed $extra) trainer gpu$GB server gpu$GA"
    CUDA_VISIBLE_DEVICES=$GB setsid python3 $B/code/b_train.py --model $model --arm $arm --seed $seed --out $RUNS/$picked --rollout_dir $RD $extra >> $RUNS/$picked/train.log 2>&1 < /dev/null &
    TPID=$!
    for i in $(seq 1 90); do [ -f $RD/live/config.json ] && break; kill -0 $TPID 2>/dev/null || break; sleep 10; done
    if [ -f $RD/live/config.json ]; then
      CUDA_VISIBLE_DEVICES=$GA setsid python3 $B/code/b_rollout_server.py --dir $RD --gpu_util 0.85 >> $RUNS/$picked/server.log 2>&1 < /dev/null &
      SPID=$!
    fi
    wait $TPID; rc=$?
    stop_server
    if [ ! -f $RUNS/$picked/.trained ]; then log "$picked stopped without finishing (rc=$rc); claim released"; rm -rf $RUNS/$picked.claim; sleep 30; continue; fi
    log "$picked trained"
  fi
  if [ ! -f $RUNS/$picked/.evaluated ]; then
    log "eval $picked"
    # both GPUs evaluate at once: accuracy evals on GA, GSM-HP probes on GB
    ( CUDA_VISIBLE_DEVICES=$GA python3 $B/code/b_eval.py --model $CK/$picked/final --tag ${picked}_step60 --out $B/results/evals >> $RUNS/$picked/eval.log 2>&1 < /dev/null \
      && CUDA_VISIBLE_DEVICES=$GA python3 $B/code/b_eval.py --model $CK/$picked/final --tag ${picked}_step60_abs --abstain_prompt 1 --data math500_train,math_heldout1000,gsm8k_test500,hp_noop500 --out $B/results/evals >> $RUNS/$picked/eval.log 2>&1 < /dev/null ) &
    EA=$!
    ( CUDA_VISIBLE_DEVICES=$GB python3 $R/dirA/code/c0_probe.py --model $CK/$picked/final --tag hp_${picked} --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $RUNS/$picked/eval_hp.log 2>&1 < /dev/null \
      && CUDA_VISIBLE_DEVICES=$GB python3 $R/dirA/code/c0_probe.py --model $CK/$picked/final --tag hpabs_${picked} --abstain_prompt 1 --shard 0 --nshards 1 --gpu_util 0.85 --out $B/results/hp >> $RUNS/$picked/eval_hp.log 2>&1 < /dev/null && touch $RUNS/$picked/.hpabs_v2 ) &
    EB=$!
    wait $EA; ra=$?; wait $EB; rb=$?
    [ $ra -eq 0 ] && [ $rb -eq 0 ] && touch $RUNS/$picked/.evaluated || { log "$picked eval failed (acc rc=$ra hp rc=$rb); claim released"; rm -rf $RUNS/$picked.claim; sleep 30; continue; }
    log "$picked evaluated"
  fi
  rm -rf $RD $CK/$picked/step20 $CK/$picked/step40 $CK/$picked/resume     # keep only final/ in RAM
  touch $RUNS/$picked/.done; rm -rf $RUNS/$picked.claim
done
