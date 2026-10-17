#!/bin/bash
# A4a.2 scale test: the fixed pool (own s0 round-1 proposals + seed-1 tracers) scored by the R-Zero guard with a base model
# (-> guard_diag/scale_<tag>/base) and its post-trained sibling (-> guard_diag/scale_<tag>/r1).
# Usage: scale_diag.sh <tag> <base_path> <post_path> <gpus e.g. 7 or 6,7> <tp> <post_no_think>
R=/home/tione/notebook/rsi_rl; B=$R/dirB; A3=$R/dirA3; export HF_HOME=$R/hf_home
TAG=$1; MB=$2; MP=$3; GP=$4; TP=$5; NT=$6
CAND=$B/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl; D=$A3/results/guard_diag/scale_$TAG
log(){ echo "$(date '+%F %T') scale_$TAG@$(hostname | cut -c1-12): $*" >> $B/results/runner.log; }
complete(){ python3 - "$1" <<'EOF'
import json, os, sys
d = sys.argv[1]
try:
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    shards = set(idx["weight_map"].values())
except Exception:
    shards = {"model.safetensors"}
ok = os.path.exists(os.path.join(d, "config.json")) and all(os.path.exists(os.path.join(d, s)) and os.path.getsize(os.path.join(d, s)) > 1e6 for s in shards)
sys.exit(0 if ok else 1)
EOF
}
for g in ${GP//,/ }; do touch /dev/shm/gpu_single_$g; done
trap 'for g in ${GP//,/ }; do rm -f /dev/shm/gpu_single_$g; done' EXIT
run(){ ck=$1; m=$2; nt=$3; o=$D/$ck; mkdir -p $o; [ -s $o/guard_stats.json ] && return 0
  w=0; until complete $m; do [ $w -eq 0 ] && log "waiting for $m"; w=$((w+1)); [ $w -gt 240 ] && { log "model $m not complete after 4 h"; return 1; }; sleep 60; done
  log "guard $ck with $(basename $m) on gpus $GP (tp $TP)"
  CUDA_VISIBLE_DEVICES=$GP python3 $A3/code/a3_guard.py --model $m --candidates $CAND --out_dir $o --seed 1 --abstain_prompt 0 --guard rzero --tp $TP --no_think $nt >> $o/guard.log 2>&1 && log "guard $ck done: $(head -c 160 $o/guard_stats.json)" || log "guard $ck FAILED"
}
run base $MB 0
run r1 $MP $NT
log "scale diag $TAG finished"
