"""Build rzero_A7_5.patch: pristine (CR-stripped) copies of the target files in a/, edited copies in b/, unified diff a -> b.
Every replacement must match exactly once. Run in a scratch dir outside rsi_rl. CPU only."""
import os, shutil, subprocess, sys
SRC = "/home/tione/notebook/rsi_rl/ext/R-Zero_5699329"
W = sys.argv[1]
R = {}  # file -> list of (old, new)

R["scripts/main.sh"] = [
("""echo "Model_abbr: $Model_abbr"
""",
"""echo "Model_abbr: $Model_abbr"
# A7-5 (P16): stage guards. A stage whose A75_DONE marker exists is skipped; the loop stops if a stage ends without it.
stage(){ local out=$1; shift; [ -f ${STORAGE_PATH}/models/$out/A75_DONE ] || bash "$@"; [ -f ${STORAGE_PATH}/models/$out/A75_DONE ] || { echo "A7-5: stage $out did not complete"; exit 1; }; }
"""),
("bash scripts/questioner_train_penalty.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1",
 "stage ${Model_abbr}_questioner_v1 scripts/questioner_train_penalty.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1"),
("bash scripts/solver_train.sh $Base_model ${STORAGE_PATH}",
 "stage ${Model_abbr}_solver_v1 scripts/solver_train.sh $Base_model ${STORAGE_PATH}"),
("for i in {2..5}; do", "for i in {2..3}; do  # A7-5 (P12): 3 iterations (repository: 5)"),
("    bash scripts/questioner_train_penalty.sh \\", "    stage ${Model_abbr}_questioner_v${i} scripts/questioner_train_penalty.sh \\"),
("    bash scripts/solver_train.sh \\", "    stage ${Model_abbr}_solver_v${i} scripts/solver_train.sh \\"),
("bash evaluation/evaluate.bash $Base_model",
 "# A7-5 (P10): the repository's benchmark evaluation is not run (Hub datasets, OpenAI re-check, all GPUs); RZ-P2 uses umwp_probe\n# bash evaluation/evaluate.bash $Base_model"),
]

R["scripts/questioner_train_penalty.sh"] = [
("""echo "RUN_ID=$RUN_ID"

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"
""",
"""echo "RUN_ID=$RUN_ID"

# A7-5 (P15): resume. k = last complete checkpoint of this stage (the repository's tracker file). Training is skipped when it
# is complete; otherwise it resumes from global_step_k through the repository's trainer.load_checkpoint_path.
CKPT=${STORAGE_PATH}/models/$save_path; Q_STEPS=${RZ_Q_STEPS:-6}; Q_MERGE=${RZ_Q_MERGE:-5}; LOAD_ARG=""; k=0
[ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
[ "$k" -gt 0 ] && [ "$k" -lt $Q_STEPS ] && [ -f $CKPT/global_step_$k/dataloader.pt ] && LOAD_ARG="trainer.load_checkpoint_path=$CKPT/global_step_$k"
echo "A7-5: $save_path last complete step $k $LOAD_ARG"
# A7-5 compat (vLLM 0.11): each reward server's engine-core subprocess renames itself (setproctitle "VLLM::EngineCore",
# vllm/v1/engine/core.py:697), which blanks its /proc environ view: the RZ_VLLM_RUN test alone misses it, and it keeps its
# GPU after its server is gone (vLLM 0.9.1 does not rename it, so there the same test selects it). a75_stop signals the
# tagged processes and all their descendants, collected with their start times before the first signal (a pid is
# signalled only while its start time is unchanged).
a75_desc(){ local c; for c in $(pgrep -P $1 2>/dev/null); do echo $c; a75_desc $c; done; }
a75_start(){ sed 's/.*) //' /proc/$1/stat 2>/dev/null | cut -d' ' -f20; }
a75_stop(){ local d p s set=""
  for d in /proc/[0-9]*; do tr '\\0' '\\n' 2>/dev/null < $d/environ | grep -qx "RZ_VLLM_RUN=$RUN_ID" || continue
    for p in ${d#/proc/} $(a75_desc ${d#/proc/}); do set="$set $p:$(a75_start $p)"; done; done
  for s in TERM KILL; do
    for d in /proc/[0-9]*; do tr '\\0' '\\n' 2>/dev/null < $d/environ | grep -qx "RZ_VLLM_RUN=$RUN_ID" && kill -$s ${d#/proc/} 2>/dev/null; done
    for p in $set; do [ -n "${p#*:}" ] && [ "$(a75_start ${p%%:*})" = "${p#*:}" ] && kill -$s ${p%%:*} 2>/dev/null; done
    sleep 15
  done; }
if [ "$k" -lt $Q_STEPS ]; then

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"
# A7-5 (P17): wait until the 4 reward servers accept connections (at most 30 min each). If a server has died (no live process
# tagged RZ_VLLM_RUN=$RUN_ID with --port <p>, checked from 30 s on) or 30 min pass, stop this stage's servers and fail at once.
a75_alive(){ local f; for f in $(grep -lzx "RZ_VLLM_RUN=$RUN_ID" /proc/[0-9]*/environ 2>/dev/null); do tr '\\0' ' ' < ${f%/environ}/cmdline 2>/dev/null | grep -q -- "--port $1 " && return 0; done; return 1; }
for p in 5000 5001 5002 5003; do t=0
  until (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; do
    for q in 5000 5001 5002 5003; do [ $t -lt 30 ] || a75_alive $q || { echo "A7-5: reward server on port $q is not running"; t=9999; break; }; done
    if [ $t -ge 1800 ]; then
      echo "A7-5: reward servers not ready; stopping them and failing this attempt"
      a75_stop  # A7-5 compat (vLLM 0.11): was the RZ_VLLM_RUN loop alone (see a75_stop)
      exit 1
    fi
    sleep 10; t=$((t+10))
  done
done
"""),
("""    trainer.max_steps=6 \\
    trainer.save_freq=1

sleep 5
""",
"""    trainer.max_steps=$Q_STEPS \\
    trainer.save_freq=1 \\
    data.seed=$((1+${RZ_SEED:-0})) worker.rollout.seed=$((1+${RZ_SEED:-0})) $LOAD_ARG ${RZ_SMOKE_ARGS}

fi
sleep 5
"""),
("""python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python
""",
"""[ -f $CKPT/A75_MERGED_step$Q_MERGE ] || { python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_$Q_MERGE/actor && touch $CKPT/A75_MERGED_step$Q_MERGE; }

sleep 10

# A7-5 (P1): was `pkill python`, which kills every python process on the host. Stop only this stage's 4 reward servers:
# the processes whose environment carries RZ_VLLM_RUN=$RUN_ID (set in vllm_service_init/start.sh, inherited by their children).
a75_stop  # A7-5 compat (vLLM 0.11): was the same RZ_VLLM_RUN loop without the servers' descendants (see a75_stop)
# A7-5 (P16): completion marker, then removal of the checkpoints the pipeline never reads (storage only)
k=0; [ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
if [ "$k" -ge $Q_STEPS ] && [ -f $CKPT/A75_MERGED_step$Q_MERGE ]; then
  touch $CKPT/A75_DONE
  for g in $CKPT/global_step_*; do [ "$g" = "$CKPT/global_step_$Q_MERGE" ] || rm -rf "$g"; done
fi
"""),
]

R["vllm_service_init/start.sh"] = [
("export VLLM_DISABLE_COMPILE_CACHE=1\n",
 "export VLLM_DISABLE_COMPILE_CACHE=1\n# A7-5 (P2): RZ_VLLM_RUN tags the 4 servers so that the questioner stage stops exactly these processes\n"),
] + [(f"CUDA_VISIBLE_DEVICES={g} python vllm_service_init/start_vllm_server.py --port 500{g-4}",
      f"RZ_VLLM_RUN=$run_id CUDA_VISIBLE_DEVICES={g} python vllm_service_init/start_vllm_server.py --port 500{g-4}") for g in (4, 5, 6, 7)]

R["scripts/solver_train.sh"] = [
("""export VLLM_DISABLE_COMPILE_CACHE=1
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path 1000 $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'start upload'
python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name}
echo 'start train'
""",
"""export VLLM_DISABLE_COMPILE_CACHE=1
# A7-5 (P16): generate -> pseudo-label -> filter runs once per iteration. Its admitted set is a local parquet that upload.py
# writes last and atomically (P5); the trainer reads it (P7). An interrupted attempt is discarded and rerun from the start.
GQ=${STORAGE_PATH}/generated_question; ADMITTED=$GQ/${experiment_name}_admitted.parquet
if [ ! -f $ADMITTED ]; then
rm -f $GQ/${experiment_name}_[0-7].json $GQ/${experiment_name}_[0-7].json.raw $GQ/${experiment_name}_[0-7]_results.json
echo 'start generate question'
bash question_generate/question_generate.bash $questioner_model_path ${RZ_GEN_N:-1000} $experiment_name
echo 'start evaluate generated question'
bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
echo 'start upload'
python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name}
fi
[ -f $ADMITTED ] || { echo "A7-5: no admitted set for $experiment_name"; exit 1; }
echo 'start train'
# A7-5 (P15): resume from the last complete checkpoint (saved every 5 steps), as in the questioner stage
CKPT=${STORAGE_PATH}/models/${experiment_name}; S_STEPS=${RZ_S_STEPS:-20}; S_MERGE=${RZ_S_MERGE:-15}; LOAD_ARG=""; k=0
[ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
[ "$k" -gt 0 ] && [ "$k" -lt $S_STEPS ] && [ -f $CKPT/global_step_$k/dataloader.pt ] && LOAD_ARG="trainer.load_checkpoint_path=$CKPT/global_step_$k"
echo "A7-5: $experiment_name last complete step $k $LOAD_ARG"
if [ "$k" -lt $S_STEPS ]; then
"""),
("    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \\", "    data.train_files=${ADMITTED} \\"),
("    trainer.max_steps=20 \\", "    trainer.max_steps=$S_STEPS \\"),
("""    worker.actor.micro_batch_size_per_device_for_experience=1 \\

echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor
""",
"""    worker.actor.micro_batch_size_per_device_for_experience=1 \\
    data.seed=$((1+${RZ_SEED:-0})) worker.rollout.seed=$((1+${RZ_SEED:-0})) $LOAD_ARG ${RZ_SMOKE_ARGS}
fi

echo "merging model"
[ -f $CKPT/A75_MERGED_step$S_MERGE ] || { python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_$S_MERGE/actor && touch $CKPT/A75_MERGED_step$S_MERGE; }
"""),
("""echo "solver training finished"

bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface""",
"""echo "solver training finished"
# A7-5 (P16): completion marker, then removal of the checkpoints the pipeline never reads (storage only)
k=0; [ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
if [ "$k" -ge $S_STEPS ] && [ -f $CKPT/A75_MERGED_step$S_MERGE ]; then
  touch $CKPT/A75_DONE
  for g in $CKPT/global_step_*; do [ "$g" = "$CKPT/global_step_$S_MERGE" ] || rm -rf "$g"; done
fi

# A7-5 (P10): the repository's benchmark evaluation is not run (Hub datasets, OpenAI re-check, all GPUs); RZ-P2 uses umwp_probe
# bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface"""),
]

R["question_evaluate/evaluate.sh"] = [
(""") &

for i in {1..7}; do
  wait ${pids[$i]} 2>/dev/null
done""",
""") &
timer_pid=$!

for i in {1..7}; do
  wait ${pids[$i]} 2>/dev/null
done
# A7-5 (P3): cancel the 3,600 s timer once every shard has ended, so that it cannot kill -9 recycled PIDs later
kill $timer_pid 2>/dev/null"""),
]

R["question_evaluate/upload.py"] = [
("login(token=token)\n", "# login(token=token)  # A7-5 (P4): no Hugging Face login; nothing is pushed\n"),
("""

for i in range(8):
    try:
        os.remove(""",
"""
# A7-5 (P6): keep the whole scored pool (before the filter) for the audit denominators
with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_all_scored.json', 'w') as f:
    json.dump(datas, f)

for i in range(8):
    try:
        os.remove("""),
("""    dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)""",
"""    # A7-5 (P5): local files instead of the Hub push below; the parquet is written last and atomically (it marks this step done)
    # dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)
    out = f"{STORAGE_PATH}/generated_question/{args.repo_name}_admitted"
    train_dataset.to_json(out + ".jsonl")
    train_dataset.to_parquet(out + ".parquet.tmp")
    os.replace(out + ".parquet.tmp", out + ".parquet")"""),
]

R["question_evaluate/evaluate.py"] = [
("    os.remove(INPUT_FILE)", "    os.rename(INPUT_FILE, INPUT_FILE + \".raw\")  # A7-5 (P14): keep the raw generations (repository: os.remove)"),
("    seed=int(args.suffix),", "    seed=int(args.suffix) + 100 * int(os.getenv(\"RZ_SEED\", \"0\")),  # A7-5 (P18): seed knob; RZ_SEED=0 is the repository default"),
]

R["question_generate/question_generate.py"] = [
("        seed=int(args.suffix),", "        seed=int(args.suffix) + 100 * int(os.getenv(\"RZ_SEED\", \"0\")),  # A7-5 (P18): seed knob; RZ_SEED=0 is the repository default"),
("    questions, answers = dataset_handler.load_data()", "    questions, answers = [\"\"], [\"\"]  # A7-5 (P13): the repository fetches the MATH-500 csv here but never uses it"),
]

R["vllm_service_init/start_vllm_server.py"] = [
("""    gpu_memory_utilization=args.gpu_mem_util,
)""",
"""    gpu_memory_utilization=args.gpu_mem_util,
    **({"seed": int(os.environ["RZ_SEED"])} if int(os.getenv("RZ_SEED", "0")) > 0 else {}),  # A7-5 (P18): seed knob; unset (repository default) for seed 0
)"""),
]

R["examples/config.yaml"] = [
("""  train_files: hiyouga/math12k@train
  val_files: hiyouga/math12k@test""",
"""  train_files: /run/ti/rzero/data/math12k_train.parquet  # A7-5 (P8): local copy of hiyouga/math12k (fetched once via hf-mirror)
  val_files: /run/ti/rzero/data/math12k_test.parquet"""),
("""  logger: ["console", "wandb"]""", """  logger: ["console"]  # A7-5 (P9): no wandb"""),
]

R["verl/trainer/main.py"] = [
("        ray.init(runtime_env=runtime_env,num_cpus=16)",
 "        # A7-5 (P11): always a private Ray instance; its object store (in /dev/shm, which counts against the host's memory cgroup)\n"
 "        # is capped at 32 GiB instead of Ray's default 30 % of RAM\n"
 "        ray.init(address=\"local\", runtime_env=runtime_env, num_cpus=16, include_dashboard=False, object_store_memory=32 * 1024**3)"),
]

R["verl/protocol.py"] = [
("""        buffer = io.BytesIO()
        if self.batch is not None:
            self.batch: TensorDict = self.batch.contiguous()
            self.batch: TensorDict = self.batch.consolidate()

        torch.save(self.batch, buffer)
""",
"""        buffer = io.BytesIO()
        # A7-5 compat (tensordict 0.10): consolidate() returns a locked TensorDict (0.8.3: unlocked), and a consolidated
        # TensorDict can only be saved while locked. The consolidated copy is therefore serialized without replacing
        # self.batch (0.8.3 replaced it), so the sender can still add keys to its batch; the bytes sent are the same.
        batch = self.batch
        if batch is not None:
            batch: TensorDict = batch.contiguous()
            batch: TensorDict = batch.consolidate()

        torch.save(batch, buffer)
"""),
]

R["verl/workers/actor/dp_actor.py"] = [
("from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input\n",
 """# A7-5 compat (transformers 4.57): was `from transformers.modeling_flash_attention_utils import ...`; that module no longer
# exports these names. Up to 4.52.4 (the pin) it imported them from flash_attn.bert_padding (line 36), so these are the same
# functions (verl/workers/critic/dp_critic.py imports them from there too).
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
"""),
]

shutil.rmtree(W, ignore_errors=True)
for f, reps in R.items():
    txt = open(f"{SRC}/{f}", "rb").read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    for side in ("a", "b"):
        os.makedirs(os.path.dirname(f"{W}/{side}/{f}"), exist_ok=True)
    open(f"{W}/a/{f}", "w", newline="\n").write(txt)
    new = txt
    for old, rep in reps:
        c = new.count(old)
        assert c == 1, (f, c, old[:80])
        new = new.replace(old, rep)
    open(f"{W}/b/{f}", "w", newline="\n").write(new)
    print("ok", f, len(reps))
p = subprocess.run(["diff", "-ruN", "a", "b"], cwd=W, capture_output=True, text=True)
import re
out = re.sub(r"^(---|\+\+\+) (\S+)\t.*$", r"\1 \2", p.stdout, flags=re.M)
open(f"{W}/rzero_A7_5.patch", "w", newline="\n").write(out)
print("patch lines", out.count("\n"))
