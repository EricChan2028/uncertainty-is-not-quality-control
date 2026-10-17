#!/bin/bash
# A7-5 (a): build the R-Zero environment on this host. CPU and network only: starts no GPU process, touches no running
# process, writes only under /run/ti/rzero (and, once, the math12k copy in rsi_rl/ext/rzero_data). Idempotent. The working
# copy is rebuilt when the patch changed (never while an A7-5 run is active); a working-tree hash other than the registered
# A75_TREE (rz_common.sh) is fatal (fix 15).
# Usage: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_setup.sh
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; E=/home/tione/notebook/rsi_rl/ext; X=$E/rzero_runs
SRC=$E/R-Zero_5699329; DATA=$E/rzero_data; export HF_HOME=$RZ/hf
source $D/rz_common.sh
mkdir -p $RZ/data $RZ/hf $RZ/tmp $RZ/links $X
P=$(sha256sum < $D/rzero_A7_5.patch | cut -c1-64)
# 1. private working copy of the official code (commit 5699329): line endings restored, A7-5 patch applied
if [ ! -f $RZ/R-Zero/.a75_patched ] || [ "$(cat $RZ/R-Zero/.a75_patched)" != "$P" ]; then
  if pgrep -f 'rz_run\.sh|rz_smoke\.sh' > /dev/null; then echo "an A7-5 run is active on this host: not rebuilding the working copy"; exit 1; fi
  rm -rf $RZ/R-Zero.tmp; cp -a $SRC $RZ/R-Zero.tmp; cd $RZ/R-Zero.tmp
  grep -qx 5699329d018d79535b7910abdedf5a6eebf355fd COMMIT
  find . -name __pycache__ -type d -prune -exec rm -rf {} +
  # the shared copy passed through Windows, which turned 92 text files into CRLF; bash cannot run CRLF scripts
  grep -rlI $'\r' . | xargs -r sed -i 's/\r$//'
  [ -z "$(grep -rlI $'\r' . || true)" ]
  patch -p1 --forward --no-backup-if-mismatch < $D/rzero_A7_5.patch
  # tree hash: the file list is sorted in the C locale, so the hash does not depend on the caller's locale (review item 6)
  find . -type f ! -name '.a75_*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64 > .a75_tree_sha256
  echo $P > .a75_patched
  cd $RZ
  # a new tree whose hash is not the registered one never replaces the working copy
  [ "$(cat $RZ/R-Zero.tmp/.a75_tree_sha256)" = "$A75_TREE" ] || { echo "FATAL: patched tree hash $(cat $RZ/R-Zero.tmp/.a75_tree_sha256) is not the registered $A75_TREE"; rm -rf $RZ/R-Zero.tmp; exit 1; }
  rm -rf $RZ/R-Zero; mv $RZ/R-Zero.tmp $RZ/R-Zero
fi
[ "$(cat $RZ/R-Zero/.a75_tree_sha256)" = "$A75_TREE" ] || { echo "FATAL: working-tree hash $(cat $RZ/R-Zero/.a75_tree_sha256) is not the registered $A75_TREE"; exit 1; }
# 2. venv over the global stack (torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.1, ray 2.49.2, flash-attn 2.8.1) plus the
#    small packages the pipeline needs; the constraint file keeps every global package at its installed version
[ -x $RZ/venv/bin/python ] || /usr/bin/python3 -m venv --system-site-packages $RZ/venv
/usr/bin/python3 -m pip freeze 2>/dev/null | grep -vE '^-e|@ file|^blinker==' > $RZ/cons.txt
$RZ/venv/bin/pip install -q -c $RZ/cons.txt stopit==1.1.2 nltk==3.9.1 scikit-learn==1.7.0 Flask==3.1.1 jsonlines==4.0.0
$RZ/venv/bin/pip freeze > $RZ/venv_freeze.txt
# 3. hiyouga/math12k (train + test) through hf-mirror, stored once on the shared volume (a few MB) and copied here
mkdir -p $DATA
if [ ! -f $DATA/math12k_test.parquet ]; then
  HF_ENDPOINT=https://hf-mirror.com $RZ/venv/bin/python - <<'EOF'
from datasets import load_dataset
d = load_dataset("hiyouga/math12k")
for s in d:
    d[s].to_parquet(f"/home/tione/notebook/rsi_rl/ext/rzero_data/math12k_{s}.parquet.tmp")
    print(s, len(d[s]), d[s].column_names)
EOF
  for s in train test; do mv $DATA/math12k_$s.parquet.tmp $DATA/math12k_$s.parquet; done
fi
cp $DATA/math12k_train.parquet $DATA/math12k_test.parquet $RZ/data/
# 4. CPU-only checks
cd $RZ/R-Zero
PYTHONPATH=$RZ/R-Zero $RZ/venv/bin/python - <<'EOF'
import datasets, flask, jsonlines, nltk, sklearn, stopit, verl
assert verl.__version__ == "0.3.1.dev0" and verl.__file__.startswith("/run/ti/rzero/R-Zero/"), verl.__file__
import verl.trainer.main, verl.workers.fsdp_workers  # noqa: F401
for s in ("train", "test"):
    d = datasets.load_dataset("parquet", data_files=f"/run/ti/rzero/data/math12k_{s}.parquet", split="train")
    print("math12k", s, len(d), d.column_names)
print("imports ok, verl", verl.__version__, verl.__file__)
EOF
for f in scripts/main.sh scripts/questioner_train_penalty.sh scripts/solver_train.sh vllm_service_init/start.sh question_evaluate/evaluate.sh question_generate/question_generate.bash; do bash -n $f; done
echo "$(date '+%F %T %Z') $(hostname | cut -c1-22) setup ok: patch $(cat .a75_patched) tree $(cat .a75_tree_sha256) data $(sha256sum $RZ/data/math12k_train.parquet $RZ/data/math12k_test.parquet | cut -c1-16 | tr '\n' ' ') stack=$(stack_fp) " | tee -a $X/ledger_setup.log
