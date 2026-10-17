#!/bin/bash
# A7-5 (e): UMWP dumps of the final solver (iteration 3, merged global_step_15) with the frozen probe dirA3/code/umwp_probe.py
# (G = 8, T = 1.0, max_new 1024, seed 17; same interpreter, HF_HOME and flags as every HX dump). rz_run.sh calls it while it
# still holds all 8 GPUs; run by hand only on GPUs G..G+3 claimed the usual way, and never after T0+71h. The probe resumes
# an interrupted dump; before each call this script cuts a partial last line (a kill mid-write) and it calls the probe up to
# 3 times until the dump has 1,200 unique parseable ids (fix 10). Counts reported are unique parseable ids.
# Usage: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_umwp.sh <seed> [first GPU, default 0]
S=${1:?seed}; G=${2:-0}; case "$S" in 0|1|2|3) ;; *) echo "seed must be 0-3"; exit 1;; esac
R=/home/tione/notebook/rsi_rl; RZ=/run/ti/rzero; OUT=$R/dirA4/results/umwp; D=$R/ext/rzero_design; X=$R/ext/rzero_runs
source $D/rz_common.sh
[ "$(date +%s)" -lt $A75_STOP ] || { echo "T0+71h has passed: no A7-5 compute (X5)"; exit 1; }
F=$RZ/s$S/models/rzero_qwen3_4b_Base_s${S}_solver_v3; M=$F/global_step_15/actor/huggingface
[ -f $F/A75_DONE ] && [ -f $M/config.json ] || { echo "final solver of seed $S is not complete"; exit 1; }
# a model path containing 'Base' makes the probe use the base-model prompt, as in every HX dump (plain_qwen3_4b_base included)
case "$M" in *Base*) ;; *) echo "model path must contain 'Base'"; exit 1;; esac
[ "$(n_ids $OUT/plain_qwen3_4b_base.jsonl)" -ge 1200 ] || { echo "base dump incomplete"; exit 1; }
probe(){ env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=$1 HF_HOME=$R/hf_home TOKENIZERS_PARALLELISM=false \
  /usr/bin/python3 $R/dirA3/code/umwp_probe.py --model $2 --tag $3 --abstain_prompt $4 >> $OUT/$3.log 2>&1; }
dump(){ local t; for t in 1 2 3; do trim_partial $OUT/$3.jsonl; [ "$(n_ids $OUT/$3.jsonl)" -ge 1200 ] && break; probe "$@"; done; trim_partial $OUT/$3.jsonl; }
# confirmatory (RZ-P2): plain prompt
dump $G $M plain_rzero_s${S}_it3 0 &
# descriptive: abstention prompt (ABS_U / ABS_A)
dump $((G+1)) $M abs_rzero_s${S}_it3 1 &
# exploratory: the same plain instruction wrapped in the chat template the R-Zero solver was trained with; a model path without
# 'Base' switches the probe to the chat template, so both models are reached through symlinks
mkdir -p $RZ/links; ln -sfn $M $RZ/links/rz_s${S}_it3_chat; ln -sfn /root/models/Qwen3-4B-Base $RZ/links/qwen3_4b_chat
dump $((G+2)) $RZ/links/rz_s${S}_it3_chat plainchat_rzero_s${S}_it3 0 &
# the base chat-template dump goes to one shared file, so only one seed writes it (fix 16): seed 0, or seed 2 if 0 was abandoned
BC=0; [ "$S" = 0 ] && BC=1; [ "$S" = 2 ] && [ -f $X/s0.abandoned ] && BC=1
[ $BC = 1 ] && dump $((G+3)) $RZ/links/qwen3_4b_chat plainchat_qwen3_4b_base 0 &
wait
for t in plain_rzero_s${S}_it3 abs_rzero_s${S}_it3 plainchat_rzero_s${S}_it3 plainchat_qwen3_4b_base; do echo "$t unique parseable ids $(n_ids $OUT/$t.jsonl)"; done
[ "$(n_ids $OUT/plain_rzero_s${S}_it3.jsonl)" -ge 1200 ] && touch $RZ/s$S/A75_UMWP_DONE
