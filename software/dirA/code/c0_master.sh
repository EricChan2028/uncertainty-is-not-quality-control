#!/bin/bash
# Direction A master scheduler for the unattended window (v2, rewritten after the 13:25 read-only audit).
# Single instance (flock on local disk), re-entrant: every stage checks its own completion marker and
# probe shards resume from their own JSONL. Global deadline DEADLINE_H hours from the FIRST start
# (persisted in .t0), so restarts do not extend it. Shards run in their own sessions and are reaped by
# process group AND by GPU (nvidia-smi compute apps), so a killed probe can never leave an EngineCore
# holding a card. Rounds are bounded by the deadline and by a liveness detector; a tag that makes no
# progress in 3 rounds is marked stuck (PARTIAL readout) and skipped instead of blocking the others.
R=/home/tione/notebook/rsi_rl; M=/root/models; C=$R/dirA/code; RES=$R/dirA/results; LOG=$RES/master.log
LOCK=/root/.c0_master.lock; WD=/root/c0_watchdog.sh
export HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false
mkdir -p $RES; cd $C
exec 9>$LOCK; flock -n 9 || exit 0                       # another master holds the lock
[ -f $RES/.t0 ] || date +%s > $RES/.t0
T0=$(cat $RES/.t0); DEADLINE=$((T0 + ${DEADLINE_H:-5}*3600))
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
left(){ echo $(( (DEADLINE - $(date +%s)) / 60 )); }
log "master v2 start pid $$ (deadline in $(left) min)"
ALL="0 1 2 3 4 5 6 7"; MYPG=$(ps -o pgid= -p $$ | tr -d ' ')

complete(){ # model dir usable: config + weights + marker, and no ModelScope download in flight
  local d=$1
  [ -e $d/config.json ] || return 1
  ls $d/*.safetensors >/dev/null 2>&1 || return 1
  [ -n "$(ls -A $d/._____temp 2>/dev/null)" ] && return 1   # non-empty temp dir = still downloading (modelscope leaves an empty one on success)
  case $d in */Llama-3.2-3B-Instruct) ;; *) [ -e $d/.complete ] || return 1;; esac
  if [ -e $d/model.safetensors.index.json ]; then
    for f in $(python3 - "$d" <<'PY'
import json,sys
print(" ".join(sorted(set(json.load(open(sys.argv[1]+"/model.safetensors.index.json"))["weight_map"].values()))))
PY
); do [ -e $d/$f ] || return 1; done
  fi
  return 0
}

gpu_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null | tr -d ' '; }
kill_group(){ # pid signal: signal the whole process group (never our own group as a whole)
  local pg; pg=$(ps -o pgid= -p $1 2>/dev/null | tr -d ' '); [ -n "$pg" ] || return 0
  if [ "$pg" = "$MYPG" ]; then kill -$2 $1 2>/dev/null; else kill -$2 -- -$pg 2>/dev/null; fi
}
gpu_pids(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2>/dev/null | tr -d ' '; }
reap(){ # gpus...: stop every probe of ours and every compute app on these GPUs; wait until the cards are empty
  local gpus=("$@") pid g i busy
  for pid in $(pgrep -f "^python3 $C/c0_probe.py "); do kill_group $pid TERM; done
  for g in "${gpus[@]}"; do for pid in $(gpu_pids $g); do kill_group $pid TERM; done; done
  sleep 8
  for pid in $(pgrep -f "^python3 $C/c0_probe.py ") $(pgrep -f "^VLLM::EngineCore"); do kill -KILL $pid 2>/dev/null; done
  for g in "${gpus[@]}"; do for pid in $(gpu_pids $g); do kill -KILL $pid 2>/dev/null; done; done
  for i in $(seq 1 30); do
    busy=0; for g in "${gpus[@]}"; do [ "$(gpu_used $g)" -lt 1500 ] 2>/dev/null || busy=1; done
    [ $busy -eq 0 ] && return 0; sleep 3
  done
  log "WARN: GPUs still busy after reap: $(for g in "${gpus[@]}"; do echo -n "$g=$(gpu_used $g)MiB "; done)"
  return 1
}
ensure_watchdog(){ pgrep -f "^bash $WD" >/dev/null || { setsid nohup bash $WD >/dev/null 2>&1 </dev/null & log "watchdog was gone, restarted"; }; }
lines(){ cat $RES/${1}_shard*of8.jsonl 2>/dev/null | wc -l; }
newest(){ stat -c %Y $RES/${1}_shard*.log $RES/${1}_shard*of8.jsonl 2>/dev/null | sort -n | tail -1; }
readout(){ # tag status: per-tag section appended to the registered readout file
  local tmp=$RES/.readout_$1.tmp
  { echo; echo "## $1  ($(date '+%F %T'))  [$2]"; echo '```'; python3 $C/c0_analyze.py $1 2>&1; echo '```'; } > $tmp
  cat $tmp >> $RES/C0_readout.md; rm -f $tmp
}

run_shards(){ # tag path extra_args : one round of 8 shards, bounded; returns 0 only when every shard reported SHARD_DONE
  local tag=$1 path=$2 extra=$3; local gpus=($ALL); local n=8 s pid alive now before after ok a
  [ -f $RES/.done_$tag ] && return 0
  [ -f $RES/.stuck_$tag ] && return 1
  [ $(date +%s) -lt $DEADLINE ] || return 1
  ensure_watchdog
  reap "${gpus[@]}"
  before=$(lines $tag)
  log "$tag: launching $n shards on GPUs ${gpus[*]} ($before records on disk)"
  local pids=()
  for s in $(seq 0 $((n-1))); do
    CUDA_VISIBLE_DEVICES=${gpus[$s]} setsid nohup python3 $C/c0_probe.py --model $path --tag $tag --shard $s --nshards $n --gpu_util 0.85 $extra > $RES/${tag}_shard${s}.log 2>&1 < /dev/null &
    pids+=($!); sleep 10
  done
  while :; do                                                # bounded wait: deadline + liveness (logs/JSONL mtime)
    alive=0; for pid in "${pids[@]}"; do kill -0 $pid 2>/dev/null && alive=1; done
    [ $alive -eq 0 ] && break
    now=$(date +%s)
    if [ $now -ge $DEADLINE ]; then log "$tag: deadline reached mid-round, stopping shards"; reap "${gpus[@]}"; return 1; fi
    if [ $((now - $(newest $tag))) -gt 2700 ]; then log "$tag: no output for 45 min, killing round"; reap "${gpus[@]}"; break; fi
    sleep 20
  done
  ok=$(grep -l SHARD_DONE $RES/${tag}_shard[0-7].log 2>/dev/null | wc -l)
  after=$(lines $tag)
  log "$tag: round finished ($ok/$n clean, $after records)"
  if [ "$ok" -ne "$n" ]; then
    a=0; [ -f $RES/.attempts_$tag ] && a=$(cat $RES/.attempts_$tag)
    if [ "$after" -gt "$before" ]; then a=0; else a=$((a+1)); fi
    echo $a > $RES/.attempts_$tag
    if [ $a -ge 3 ]; then log "$tag: no progress in 3 rounds, marking stuck; PARTIAL readout"; readout $tag PARTIAL; touch $RES/.stuck_$tag; fi
    return 1
  fi
  readout $tag COMPLETE; touch $RES/.done_$tag; rm -f $RES/.attempts_$tag
  return 0
}

proposer_stage(){
  [ -f $RES/.done_proposed_4b ] && return 0
  [ -f $RES/.stuck_proposed_4b ] && return 1
  if [ ! -s $RES/proposed_4b.jsonl ]; then
    reap $ALL
    log "proposer: generating items on GPU 1"
    CUDA_VISIBLE_DEVICES=1 setsid timeout 1500 python3 $C/c0_proposer.py --model $M/Qwen3-4B-Base --n 600 --out $RES/proposed_4b.jsonl > $RES/proposer_4b.log 2>&1 < /dev/null
    log "proposer: exit $? ($(wc -l < $RES/proposed_4b.jsonl 2>/dev/null) items)"
  fi
  [ -s $RES/proposed_4b.jsonl ] || { touch $RES/.stuck_proposed_4b; log "proposer: no items, marking stuck"; return 1; }
  run_shards proposed_4b $M/Qwen3-4B-Base "--items_file $RES/proposed_4b.jsonl"
}

# STAGE 1: registered models, whichever is downloaded, no tag may block the others; poll until all done or deadline.
ORDER="llama3p2_3b:$M/Llama-3.2-3B-Instruct qwen3_4b_base:$M/Qwen3-4B-Base qwen3_1p7b_base:$M/Qwen3-1.7B-Base phi4mini:$M/Phi-4-mini-instruct"
while [ $(date +%s) -lt $DEADLINE ]; do
  ensure_watchdog
  pending=0; ran=0
  for e in $ORDER; do
    tag=${e%%:*}; path=${e#*:}
    [ -f $RES/.done_$tag ] && continue
    [ -f $RES/.stuck_$tag ] && continue
    pending=$((pending+1))
    if complete $path; then
      ran=1
      run_shards $tag $path "" || { log "$tag: incomplete, will retry after the other tags"; sleep 30; }
    else
      [ -f $RES/.waiting_$tag ] || { log "$tag: model not complete yet at $path, waiting"; touch $RES/.waiting_$tag; }
    fi
    [ $(date +%s) -lt $DEADLINE ] || break
  done
  [ $pending -eq 0 ] && break
  # STAGE 2 fills download waits: once the 4B is done and nothing else is ready, measure the proposer's own items.
  if [ $ran -eq 0 ]; then
    if [ -f $RES/.done_qwen3_4b_base ] && [ ! -f $RES/.done_proposed_4b ] && [ ! -f $RES/.stuck_proposed_4b ]; then proposer_stage; else sleep 120; fi
  fi
done
# STAGE 3: with time left, the proposer measurement (if not yet) and the full 3,336-task extension on the 4B.
if [ $(date +%s) -lt $DEADLINE ] && [ -f $RES/.done_qwen3_4b_base ]; then
  proposer_stage
  [ $(date +%s) -lt $DEADLINE ] && run_shards qwen3_4b_base_all $M/Qwen3-4B-Base "--all_tasks 1"
fi
reap $ALL
cd $R && sha256sum dirA/results/C0_readout.md >> docs/registration_hashes.txt 2>/dev/null
log "master done (deadline in $(left) min)"
touch $RES/.master_done
