#!/bin/bash
# CPU-only check of rzero_A7_5.patch on a fresh copy of the shared repo (scratch dir outside rsi_rl).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
W=/run/ti/rzero_design_scratch; T=$W/test
rm -rf $T; cp -a /home/tione/notebook/rsi_rl/ext/R-Zero_5699329 $T
cd $T
find . -name __pycache__ -type d -prune -exec rm -rf {} +
n=$(grep -rlI $'\r' . | wc -l); grep -rlI $'\r' . | xargs sed -i 's/\r$//'; echo "CR-stripped files: $n; left: $(grep -rlI $'\r' . | wc -l)"
patch -p1 --forward --no-backup-if-mismatch < $W/w/rzero_A7_5.patch
# working-tree hash exactly as rz_setup.sh computes it (the registered A75_TREE in rz_common.sh; file list in C-locale order)
echo "tree sha256 $(find . -type f ! -name '.a75_*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-64)"
for f in scripts/main.sh scripts/questioner_train_penalty.sh scripts/solver_train.sh vllm_service_init/start.sh question_evaluate/evaluate.sh question_generate/question_generate.bash; do bash -n $f && echo "bash -n ok $f"; done
for f in question_evaluate/upload.py question_evaluate/evaluate.py question_generate/question_generate.py vllm_service_init/start_vllm_server.py verl/trainer/main.py verl/protocol.py verl/workers/actor/dp_actor.py; do python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" $f && echo "ast ok $f"; done
PYTHONPATH=$T python3 - <<'EOF'
from omegaconf import OmegaConf
from verl.trainer.config import PPOConfig
import verl
print("verl", verl.__version__, verl.__file__)
base = OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load("examples/config.yaml"))
q = ("data.max_response_length=4096 worker.actor.model.model_path=/root/models/Qwen3-4B-Base trainer.experiment_name=x "
     "trainer.save_checkpoint_path=/tmp/x trainer.total_epochs=1000 worker.reward.reward_function=./examples/reward_function/caller_penalty.py:compute_score "
     "trainer.val_freq=-1 trainer.n_gpus_per_node=4 data.format_prompt=./examples/format_prompt/questioner.jinja worker.rollout.n=4 "
     "worker.actor.global_batch_size=16 trainer.max_steps=6 trainer.save_freq=1 data.seed=2 worker.rollout.seed=2 "
     "trainer.load_checkpoint_path=/tmp/x/global_step_3").split()
s = ("data.max_response_length=4096 worker.actor.model.model_path=/root/models/Qwen3-4B-Base trainer.experiment_name=y "
     "trainer.save_checkpoint_path=/tmp/y/ data.train_files=/tmp/y_admitted.parquet trainer.total_epochs=100 trainer.max_steps=20 "
     "data.format_prompt=./examples/format_prompt/solver.jinja trainer.val_freq=4 worker.actor.micro_batch_size_per_device_for_update=1 "
     "worker.actor.micro_batch_size_per_device_for_experience=1 data.seed=1 worker.rollout.seed=1 "
     "data.rollout_batch_size=16 worker.actor.global_batch_size=16 trainer.val_before_train=false trainer.val_freq=-1 trainer.save_freq=1").split()
for name, ov in (("questioner", q), ("solver+smoke", s)):
    c = OmegaConf.to_object(OmegaConf.merge(base, OmegaConf.from_dotlist(ov)))
    c.deep_post_init()
    print(name, "train", c.data.train_files, "val", c.data.val_files, "seed", c.data.seed, c.worker.rollout.seed, "rbs", c.data.rollout_batch_size,
          "n", c.worker.rollout.n, "gbs", c.worker.actor.global_batch_size, "steps", c.trainer.max_steps, "val_freq", c.trainer.val_freq,
          "vbt", c.trainer.val_before_train, "load", c.trainer.load_checkpoint_path, "logger", c.trainer.logger, "tp", c.worker.rollout.tensor_parallel_size,
          "save_freq", c.trainer.save_freq)
import inspect, ray
print("ray", ray.__version__, "ray.init accepts object_store_memory:", "object_store_memory" in inspect.signature(ray.init).parameters)
EOF
# stage-guard logic dry run with stub scripts (no python, no GPU)
G=$W/guard; rm -rf $G; mkdir -p $G/scripts $G/sp/models
sed -n '1,/^stage(){/p' scripts/main.sh | tail -1 > $G/stagefn.sh
cat > $G/scripts/ok.sh <<'EOF'
mkdir -p $STORAGE_PATH/models/$1 && touch $STORAGE_PATH/models/$1/A75_DONE && echo "ran $1"
EOF
cat > $G/scripts/bad.sh <<'EOF'
echo "ran $1 (fails)"
EOF
cd $G && STORAGE_PATH=$G/sp bash -c 'source stagefn.sh; stage a scripts/ok.sh a; stage a scripts/ok.sh a; echo "second call skipped"; stage b scripts/bad.sh b; echo "NOT REACHED"' || echo "stopped as expected (exit $?)"
# P17 wait loop with fake servers (sleep processes; no python, no GPU): the port-5000 server is alive, the others are missing,
# so the loop must stop this RUN_ID's processes (a75_stop) and fail after the 30 s grace period
a75f(){ sed -n '/^a75_desc(){/,/^  done; }$/p' scripts/questioner_train_penalty.sh; }
cd $T; { a75f; sed -n '/^a75_alive(){/,/^done$/p' scripts/questioner_train_penalty.sh; } > $G/p17.sh; RID=p17test$$
echo "P17 extract: $(grep -o '^a75_[a-z]*(){' $G/p17.sh | tr -d '(){' | tr '\n' ' ')($(wc -l < $G/p17.sh) lines)"
RZ_VLLM_RUN=$RID bash -c 'sleep 300; :' --port 5000 --model_path x > /dev/null 2>&1 & FK=$!
RZ_VLLM_RUN=other$$ bash -c 'sleep 300; :' --port 5001 --model_path x > /dev/null 2>&1 & OK=$!
sleep 1; RUN_ID=$RID bash -c "source $G/p17.sh; echo NOT REACHED" && echo "P17: loop passed?!" || echo "P17: failed fast as expected (exit $?)"
kill -0 $FK 2>/dev/null && echo "P17: tagged fake server still alive?!" || echo "P17: tagged fake server stopped"
kill -0 $OK 2>/dev/null && echo "P17: other RUN_ID untouched" || echo "P17: other RUN_ID killed?!"
kill $(grep -lzx "RZ_VLLM_RUN=other$$" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3) 2>/dev/null; wait 2>/dev/null || true
RZ_VLLM_RUN=$RID bash -c 'sleep 60; :' --port 5002 --model_path x > /dev/null 2>&1 & FK=$!; sleep 1
RUN_ID=$RID bash -c "$(grep '^a75_alive(){' scripts/questioner_train_penalty.sh); a75_alive 5002 && echo 'alive(5002)=yes'; a75_alive 5003 || echo 'alive(5003)=no'"
kill $(grep -lzx "RZ_VLLM_RUN=$RID" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3) 2>/dev/null; wait 2>/dev/null || true
echo "P17 test processes left: $(grep -lz "RZ_VLLM_RUN=$RID\|RZ_VLLM_RUN=other$$" /proc/[0-9]*/environ 2>/dev/null | wc -l)"
# P1 end-of-stage path (compat fix F3): the stage script's own P1 line (a75_stop) against a fake tagged server whose child
# runs without RZ_VLLM_RUN in its environment (as the renamed vLLM 0.11 engine core does) and has a child of its own: all
# three must stop, while another RUN_ID's server with such a child and an unrelated untagged process stay untouched
{ a75f; grep '^a75_stop  # A7-5 compat' scripts/questioner_train_penalty.sh; } > $G/p1.sh
echo "P1 extract: $(grep -o '^a75_[a-z]*(){' $G/p1.sh | tr -d '(){' | tr '\n' ' ')+ '$(tail -1 $G/p1.sh | cut -c1-60)'"
alive(){ [ -e /proc/$1 ] && [ "$(sed 's/.*) //' /proc/$1/stat 2>/dev/null | cut -d' ' -f1)" != Z ]; }
kids(){ local c; for c in $(pgrep -P $1 || true); do echo $c; kids $c; done; }
RZ_VLLM_RUN=$RID bash -c 'env -u RZ_VLLM_RUN bash -c "sleep 300 & wait" & wait' --port 5000 --model_path x > /dev/null 2>&1 & FK=$!
RZ_VLLM_RUN=other$$ bash -c 'env -u RZ_VLLM_RUN bash -c "sleep 300 & wait" & wait' --port 5001 --model_path x > /dev/null 2>&1 & OK=$!
sleep 300 & UN=$!
sleep 1; FKD=$(kids $FK | tr '\n' ' '); OKD=$(kids $OK | tr '\n' ' ')
echo "P1: tagged fake server $FK, descendants [ $FKD], of which carry RZ_VLLM_RUN: $(for p in $FKD; do grep -qzx "RZ_VLLM_RUN=$RID" /proc/$p/environ 2>/dev/null && echo $p; done | wc -l)"
RUN_ID=$RID bash $G/p1.sh
n=0; for p in $FK $FKD; do alive $p && n=$((n + 1)); done; echo "P1: tagged server and its untagged descendants alive after a75_stop: $n of $(echo $FK $FKD | wc -w)"
n=0; for p in $OK $OKD $UN; do alive $p && n=$((n + 1)); done; echo "P1: other RUN_ID's server with its descendants, and the unrelated process, alive: $n of $(echo $OK $OKD $UN | wc -w)"
kill $OK $OKD $UN 2>/dev/null || true; wait 2>/dev/null || true
n=0; for p in $FK $FKD $OK $OKD $UN; do alive $p && n=$((n + 1)); done; echo "P1 test processes left: $n"
echo "patch sha256 $(sha256sum $W/w/rzero_A7_5.patch | cut -c1-64)"
