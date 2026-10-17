# A7-5 addendum: R-Zero replication in the official code (exact commands)

Status: DRAFT v2, to be hashed before any R-Zero data exists. Written 2026-09-24 (CST); revised the same day after an
adversarial critique (17 fixes) and four author decisions (host plan, clock, memory, watchdogs), then after a verification
review (10 fixes to our run-control scripts; none changes the algorithm, a default or a verdict rule); section 14 maps
each fix to the text and files.
Registration: `dirA5/docs/A7_registration_2026-09-24.md`, item 5. This addendum adds the exact commands the item promised.
It does not change RZ-P1 or RZ-P2.
Files: `/home/tione/notebook/rsi_rl/ext/rzero_design/` (this file plus the companion files in section 13, which are part of
the addendum). Changing any of them after the hash requires a dated amendment appended here before the run it affects.

Preparation used no GPU. The investigations and every check behind this addendum were read-only or CPU-only
(`CUDA_VISIBLE_DEVICES=""`; `nvidia-smi` queries only). No running process was touched. Scratch work went to
`/run/ti/rzero_design_scratch` on host A, outside `rsi_rl`.

## 0. Item A7-5 (verbatim) and summary

> "Replication in the official R-Zero code (confirmatory; design now, exact commands in an addendum before data). Run the
> public R-Zero implementation with its default challenger / solver and uncertainty filter on Qwen3-4B-Base (or the closest
> supported base model), 1-2 seeds, 3 iterations, with the repository's defaults except compute settings. Audit, with the
> frozen Opus prompt, 200 filtered (admitted) questions per iteration.
> - RZ-P1: the audited ill-posed share of admitted questions at iteration 3 >= iteration 1 + 10 points in >= 1 of the seeds run.
> - RZ-P2: the final solver's UMWP CA (plain prompt, as HX) exceeds the base model's in every seed run.
> If the official code cannot be run on this cluster within 3 days, this item is reported as not run with the reason."

Summary of the plan:
- **Code.** The official R-Zero code at commit `5699329d018d79535b7910abdedf5a6eebf355fd` (shared copy
  `ext/R-Zero_5699329`), using its bundled training framework (EasyR1 / verl 0.3.1.dev0), on Qwen3-4B-Base
  (`/root/models/Qwen3-4B-Base`), the model the repository targets.
- **Changes.** The repository cannot run here as shipped. It pushes the admitted set to the Hugging Face Hub, which is
  unreachable, and reads it back from there. It runs `pkill python`, which would kill every python process on the host. Its
  pinned binaries have no kernels for these Blackwell GPUs. The patch (section 3) changes only I/O, process handling,
  seeds, resume and Ray's memory footprint. The questioner, solver, reward, uncertainty filter, sampling, batch sizes, step
  counts and GPU layout all stay at the repository defaults (section 11).
- **Hardware.** One fully free 8-GPU host per seed, for its whole duration. The repository hard-codes 4 training GPUs plus
  4 reward servers, 8 generation shards and 8 solver GPUs, all as fractions of total GPU memory. Seed 0 runs on host B
  (`-p <PORT>`), which drains first; seed 1 on host A (`-p <PORT>`). Each starts automatically when its host has drained
  (section 8).
- **Measurements.** 200 admitted questions are sampled per iteration with `random.Random(20260924 + iteration)` and audited
  by the frozen A3 Opus workflow, prompt unchanged (section 5). The final solver is dumped with `umwp_probe.py`
  (plain prompt, tag `plain_rzero_s<S>_it3`) and compared with the existing `plain_qwen3_4b_base` dump (section 6).
- **Budget.** T0 = 2026-09-24 15:10 CST (the A7 registration). All compute stops at T0 + 71 h; RZ-P2 is read at T0 + 72 h;
  audits of admitted sets that exist by T0 + 71 h may finish by T0 + 84 h. A projected-finish rule decides whether a seed
  starts, in full or in RZ-P1-only mode. "Not run" is reported with a reason (section 8).

## 1. What runs: the official per-iteration pipeline

Naming: Q0 = S0 = Qwen3-4B-Base; abbreviation `rzero_qwen3_4b_Base_s<S>`. Q_i is `<abbr>_questioner_v<i>/global_step_5`
(merged) and S_i is `<abbr>_solver_v<i>/global_step_15` (merged). Iteration i trains Q_i from Q_(i-1) against S_(i-1),
then S_i from S_(i-1) on questions of Q_i filtered by S_(i-1). The loop runs iterations 1, 2 and 3.
The final solver is S_3 = `$STORAGE_PATH/models/<abbr>_solver_v3/global_step_15/actor/huggingface`.

**Questioner stage** (`scripts/questioner_train_penalty.sh`)
- 4 vLLM reward servers run S_(i-1) on GPUs 4-7, ports 5000-5003. Each returns 10 samples per question (T 1.0, top_p 1.0,
  top_k 40, max 4,096 tokens).
- GRPO runs on Q_(i-1) on GPUs 0-3: 512 fixed-prompt rows x n = 4 per step, global batch 16 (x n), lr 1e-6, low_var_kl KL
  loss with coefficient 1e-2, rollout T 1.0 and top_p 0.99, TP 2, max response 4,096 tokens.
- It runs 6 steps with a checkpoint every step. The merge takes step 5. With `total_epochs=1000` the repository's trainer
  then passes through the remaining epochs without training (about 0.42 s each, about 7 min per questioner stage).
- Reward (`caller_penalty.py`): min(score, 1 - score) for a parsed question (-1 otherwise), minus the BLEU-cluster share
  over the batch (agglomerative clustering, distance 1 - BLEU, threshold 0.5).

**Solver stage** (`scripts/solver_train.sh`)
- Generation: Q_i writes 8 shards x 1,000 questions (T 1.0, top_p 0.95, max 4,096 tokens, shard seed k = 0..7).
- Pseudo-labelling with S_(i-1): 9 samples (T 1.0, top_k 40). The label is the majority answer under `grade_answer`, and
  score = majority count / number of non-empty boxed answers. Items containing "证明" or "box", or whose majority answer
  contains "text", are dropped. Shards still running 3,600 s after shard 0 finishes are killed.
- Uncertainty filter (`upload.py --min_score 0.3 --max_score 0.8`): keep 0.3 <= score <= 0.8 with a non-empty, non-"None"
  answer. **This is the admitted set of iteration i.**
- GRPO on S_(i-1), 8 GPUs: 512 prompts x n = 5 per step, global batch 128 (x n), micro-batch 1, 20 steps, checkpoint every
  5 steps, validation every 4 steps. The repository's trainer drops prompts over 2,048 tokens (`filter_overlong_prompts`)
  and the last incomplete batch of 512 (`drop_last`). Reward (`math.py`): 0.9 x agreement with the pseudo-label + 0.1 x
  format. The merge takes step 15.

## 2. (a) Environment

The pinned environment (`requirements.txt`: torch 2.7.0 cu126, vLLM 0.9.1, transformers 4.52.4, ...) cannot run on these
GPUs (RTX PRO 5000 Blackwell, sm_120; driver 580.126.20):
- The torch 2.7.0 wheel has kernels only up to sm_90 and no PTX.
- vLLM 0.9.1's flash-attention library has no sm_120 kernels and no PTX.
- Building vLLM 0.9.1 needs CUTLASS and vllm-flash-attn from GitHub, which is unreachable.
- These findings come from cuobjdump on the downloaded wheels.

The run therefore uses the repository's own bundled training code (verl 0.3.1.dev0, unmodified except P11) on the global
stack: torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.1, ray 2.49.2, flash-attn 2.8.1, tensordict 0.10.0,
datasets 4.3.0, numpy 1.26.4. It runs in a venv on `/run/ti` with `--system-site-packages`, adding only stopit 1.1.2,
nltk 3.9.1, scikit-learn 1.7.0, Flask 3.1.1 and jsonlines 4.0.0 (plus blinker 1.9.0, itsdangerous, joblib and
threadpoolctl as dependencies). Host B has the same stack.

A constraint file pins every global package at its installed version. Imports, config parsing and the vLLM 0.11 API used
by the rollout and weight sync were checked on the CPU. The GPU path is checked by the smoke test (section 4.2).

**Line endings (D0).** The shared copy reached the cluster through Windows, and 92 of its text files, including every
`.sh`, are CRLF. bash cannot run them. The setup strips the trailing CR. This is byte-level restoration and does not
change Python or YAML semantics.

**Memory (D7).** Each host has a 1,400 GiB memory cgroup that counts `/dev/shm` (tmpfs), which already holds about 900 GB
of our checkpoints (a janitor keeps >= 500 GB of it free). Every A7-5 temp and cache directory is therefore on the local
disk `/run/ti/rzero/tmp` (`rz_env.sh`: `TMPDIR`, `RAY_TMPDIR`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`,
`TORCHINDUCTOR_CACHE_DIR`, `HF_DATASETS_CACHE`, `CUDA_CACHE_PATH`, `TORCH_EXTENSIONS_DIR`), and Ray's object store, which
lives in `/dev/shm` and defaults to 30 % of RAM, is capped at 32 GiB (`object_store_memory` in P11, and
`RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES` for any other Ray start). The preflight requires >= 250 GiB of cgroup headroom
(limit minus rss + shmem from `memory.stat`), and the monitor stops a run at < 100 GiB (exit 7, resumable). Ray's own
memory monitor, on by default in ray 2.49.2, kills workers once the node's usage without page cache reaches 95 % of the
cgroup limit, which is at about 70-75 GiB of this headroom; the 100 GiB threshold makes our clean, resumable stop come
first (review item 4). On 24 Sep, with our other jobs still running, the headroom was about 480 GiB on host A and 520 GiB
on host B. Checkpoints are written to `/run/ti/rzero/s<S>` (disk), never to `/dev/shm`.

Commands, once per host. They use CPU and network only, so they can run while the host is busy. The script is idempotent.
```bash
bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_setup.sh
```
`rz_setup.sh` (Appendix B) does the following:
1. Copies `ext/R-Zero_5699329` to `/run/ti/rzero/R-Zero.tmp` and checks `COMMIT`. It rebuilds whenever the recorded patch
   hash differs from the current patch, and never while an A7-5 run is active.
2. Drops `__pycache__`, strips CRs and applies `rzero_A7_5.patch` with `patch -p1`.
3. Records the sha256 of the patch and of the whole tree (file list sorted with `LC_ALL=C`, so the hash does not depend
   on the caller's locale; review item 6). The tree hash must be the registered
   `69ae729158dce32ea0e0d59216b2542001affbb11ab61739093020351a1f7ea2` (`A75_TREE` in `rz_common.sh`, obtained in a scratch rehearsal on host A). **A different hash is fatal**
   (fix 15): the new tree is discarded and never replaces the working copy, and every preflight checks the patch and tree
   hashes again.
4. Builds `/run/ti/rzero/venv` as above.
5. Fetches `hiyouga/math12k` once through `HF_ENDPOINT=https://hf-mirror.com` into `ext/rzero_data/`. Expected: train
   12,000 rows and test 500 rows, columns problem and answer. Rehearsal sha256: train
   `9c7a7a987088f050aa9d8593e3cf22e1ce3e9aba62be78de62a8321e4523e0e8`, test
   `542c4db27b0ca73cae6c56f5be1a9faa20cfd9aa5f04433338c57f1bd8129bb2`. A different hash from a changed mirror is logged;
   the row counts and columns must match. The files are then copied to `/run/ti/rzero/data/`.
6. CPU-only checks: `verl.__version__ == "0.3.1.dev0"` imported from the working copy, trainer and worker modules import,
   the parquet files load, and `bash -n` passes on every stage script.
7. Writes one line to `ext/rzero_runs/ledger_setup.log`, including the stack fingerprint (section 4.2).

The full setup was rehearsed on host A with paths redirected to scratch (`dev/designtest.sh` T9): exit 0, 12,000 / 500
rows, verl imported from the working copy, tree `69ae729158dce32ea0e0d59216b2542001affbb11ab61739093020351a1f7ea2`; a second run did not rebuild; a rebuild with the caller in
the C locale gave the same tree hash; a changed patch was refused with the old working copy kept.

## 3. (b) The patch: `rzero_A7_5.patch`

The patch is a unified diff against the CR-stripped shared copy, 322 lines over 11 files (Appendix A). It was
generated by `dev/mkpatch.py`, where every replacement must match exactly once, and checked by `dev/patchtest.sh`:
- it applies cleanly, and the patched tree has the registered hash;
- `bash -n` passes on all stage scripts and `ast` parses all edited Python;
- the repository's `PPOConfig` parses with the questioner and solver overrides, including the smoke test's
  (`deep_post_init` passes), and `ray.init` of ray 2.49.2 accepts `object_store_memory`;
- the stage guard skips a completed stage and stops the loop on a failed one;
- the P17 wait loop, run against fake tagged processes, fails fast when a server is missing and stops only that run's
  processes.

No patch changes the algorithm.

| # | File | Change | Why |
|---|---|---|---|
| P1 | `scripts/questioner_train_penalty.sh` | `pkill python` becomes a loop that TERMs, then KILLs, only processes whose environment carries `RZ_VLLM_RUN=$RUN_ID` | `pkill python` kills every python process on the host (our jobs, the co-tenant's) |
| P2 | `vllm_service_init/start.sh` | prefix `RZ_VLLM_RUN=$run_id` on the 4 server lines | lets P1 and P17 find exactly these servers and their children |
| P3 | `question_evaluate/evaluate.sh` | `timer_pid=$!` after the 3,600 s timer; `kill $timer_pid` once all shards end | the leftover timer would later `kill -9` recycled PIDs |
| P4 | `question_evaluate/upload.py` | `login(token=token)` commented out | huggingface.co unreachable; nothing is pushed |
| P5 | `question_evaluate/upload.py` | `push_to_hub(...)` replaced by `<name>_admitted.jsonl` and then `<name>_admitted.parquet`, written via `.tmp` + `os.replace` | the admitted set stays on local disk; the parquet doubles as the step's completion marker |
| P6 | `question_evaluate/upload.py` | dump of the full scored pool (`<name>_all_scored.json`) before the per-shard files are deleted | audit denominators and pool statistics |
| P7 | `scripts/solver_train.sh` | `data.train_files=${ADMITTED}` (the local parquet) instead of `${HUGGINGFACENAME}/<name>@train` | reads the admitted set back through the repository's own parquet branch (`dataset.py`) |
| P8 | `examples/config.yaml` | `train_files` / `val_files` become the local `math12k_{train,test}.parquet` | Hub ids unreachable; same data, fetched once through hf-mirror |
| P9 | `examples/config.yaml` | `logger: ["console"]` | no wandb here (placeholder key) |
| P10 | `scripts/solver_train.sh`, `scripts/main.sh` | `bash evaluation/evaluate.bash ...` commented out | outside the loop; needs Hub datasets, an OpenAI re-check and every GPU. RZ-P2 uses the registered UMWP probe |
| P11 | `verl/trainer/main.py` | `ray.init(address="local", ..., num_cpus=16, include_dashboard=False, object_store_memory=32 * 1024**3)` | never attach to another Ray cluster on the host; cap the object store, which lives in `/dev/shm` inside the memory cgroup (D-memory) |
| P12 | `scripts/main.sh` | `for i in {2..3}` (repository: `{2..5}`) | 3 iterations, as registered |
| P13 | `question_generate/question_generate.py` | `questions, answers = [""], [""]` in place of `dataset_handler.load_data()` | the fetched MATH-500 csv is never used (the prompt is fixed); removes a network dependency |
| P14 | `question_evaluate/evaluate.py` | `os.rename(INPUT_FILE, INPUT_FILE + ".raw")` in place of `os.remove` | keeps the 8,000 raw generations |
| P15 | both training scripts | resume: read the repository's `latest_global_step.txt` (k); skip training if k >= max_steps, otherwise pass `trainer.load_checkpoint_path=.../global_step_k` if its `dataloader.pt` exists | runs can be killed at any time (section 7) |
| P16 | `main.sh`, both training scripts | `A75_DONE` marker per stage (training complete and merge exited 0); `stage()` guard in main.sh; solver stage runs generate, pseudo-label and filter only while no admitted parquet exists; unused checkpoints removed after `A75_DONE` | resume without ever regenerating an iteration's admitted set; storage |
| P17 | `scripts/questioner_train_penalty.sh` | wait (up to 30 min per port) until ports 5000-5003 accept connections before training starts; from 30 s on, if no live process tagged `RZ_VLLM_RUN=$RUN_ID` with `--port <p>` exists for one of the 4 ports, or after 30 min, stop this stage's servers and exit 1 at once (fix 11) | the trainer calls the servers during its validation before training; a dead server no longer costs 30 min per port |
| P18 | `question_generate.py`, `evaluate.py`, `start_vllm_server.py`, both training scripts | seed knob `RZ_SEED` (section 4.1) | a second seed; `RZ_SEED=0` reproduces the repository seeds exactly |
| P19 | both training scripts, `solver_train.sh` | smoke hooks `RZ_Q_STEPS` (default 6), `RZ_Q_MERGE` (5), `RZ_S_STEPS` (20), `RZ_S_MERGE` (15), `RZ_GEN_N` (1000), `RZ_SMOKE_ARGS` (empty) | used only by `rz_smoke.sh`; `rz_run.sh` refuses to start if any is set |

## 4. (c) Running a seed

### 4.1 Seeds

| | seed 0 (`RZ_SEED=0`) | seed S > 0 (`RZ_SEED=S`) |
|---|---|---|
| `data.seed`, `worker.rollout.seed` | 1, 1 (repository defaults) | 1 + S, 1 + S |
| question-generation vLLM seed, shard k | k (repository) | k + 100 S |
| pseudo-labelling vLLM seed, shard k | k (repository) | k + 100 S |
| reward-server vLLM seed | unset (vLLM default, repository) | S |

Seed 0 is therefore the repository configuration with no seed change. Training is not bit-deterministic in either case.

**Seed ids and host affinity (fix 4).** Seed ids 0 and 1 are the registered seeds. `STORAGE_PATH=/run/ti/rzero/s<S>` is
local to a host, so a seed belongs to the host that first claims it: at its first claim `rz_run.sh` writes
`ext/rzero_runs/s<S>.host` (hostname, storage path, time; created atomically), and `rz_run.sh`, `rz_wait_and_run.sh` and
`rz_extract.py` refuse that seed on any other host. A seed whose host is lost cannot resume elsewhere. It is abandoned with
`rz_stop.sh <S> abandon "<reason>"` (any host; writes `ext/rzero_runs/s<S>.abandoned`), reported with the stage it reached,
never counted, and a restart elsewhere uses a new seed id: 2 for the first abandoned id, 3 for the second (`RZ_SEED` 2 or 3
by the table above). `rz_run.sh` accepts id 2 only after >= 1 abandoned id and id 3 only after >= 2, so at most two seed
runs can ever count, as registered.

### 4.2 Commands

Everything runs from a normal root shell on the chosen host (A: `-p <PORT>`, B: `-p <PORT>`), after `rz_setup.sh`.

```bash
# at the hash (drain rule, section 8): the waiters start the smoke test and the seeds by themselves
#   host B (-p <PORT>):
nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_wait_and_run.sh 0 > /dev/null 2>&1 &
#   host A (-p <PORT>):
nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_wait_and_run.sh 1 > /dev/null 2>&1 &
# progress
tail -5 /home/tione/notebook/rsi_rl/ext/rzero_runs/ledger_s0.log; ls /run/ti/rzero/s0/models/*/A75_DONE; tail -c 3000 /run/ti/rzero/s0/main.log
# stop at once (waiter, driver and every tagged process; files kept); resume later with the same rz_wait_and_run.sh command
bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh 0
```

**Waiter** (`rz_wait_and_run.sh <S>`, decision D-host). It polls every 60 s. When the host is free by the preflight
criteria below (evaluated once per poll, stale markers removed), it applies the projected-finish rule (section 8), runs the
smoke test (seed 0 always; another seed skips it when a smoke test PASSED on the same stack, i.e. the same patch, working
tree, venv package list, GPU model and driver, and system python, as fingerprinted in the PASS line) and then
`rz_run.sh <S>`. After a co-tenant yield (exit 6), a memory stop (exit 7), a lost preflight or a clock error (exit 3) it
waits again and resumes the seed. Any other exit ends the waiter with a ledger line. "Host not free" is logged when the
kind of reason changes and hourly.
- Only `full`, `p1only` or `none` from `rz_clock.py` is a projection (review item 2). Any other output (`rz_clock.py`
  failed, for example after a transient NFS error on `ext/rzero_runs`) is a clock error: the waiter logs it (when it
  starts and hourly) and retries every 60 s, and the driver exits 3, which the waiter also retries. Only a printed `none`
  ends a seed as not started (X1). A smoke test stopped at T0 + 71 h is logged as X5.
- When the waiter ends for any reason (a refusal, a "no time left" exit, T0 + 71 h, `rz_stop.sh`), it removes the GPU
  markers that a passed smoke test kept and no run took over (first field `rzA75_smoke`, process no longer a live A7-5
  driver); `rz_stop.sh <S>` does the same (review item 3). Our other launchers only test whether a marker exists, so such
  leftovers would otherwise block all 8 GPUs for them.

**Preflight** (`host_check` in `rz_common.sh`, run twice 60 s apart; `rz_run.sh` and `rz_smoke.sh` refuse with exit 3 and
one ledger line) requires:
- the working copy built from the current patch, with the registered tree hash (fix 15);
- 8 GPUs, no compute process on any GPU and < 1,500 MiB used on each;
- none of our job scripts alive on the host: queue runners, launchers, chain scripts, and any `bash /root/<name>.sh`
  except the passive daemons `hang_watchdog.sh` and `ckpt_janitor[2].sh` (`a4_pair_watchdog.sh` starts jobs and counts);
- no `/dev/shm/gpu_single_<g>` marker held by another job. A marker is ours to take over if it carries our tag and its
  driver is dead, or if it was kept by a smoke test that PASSED on this host in the last hour (hand-over, fix 8). A marker is
  **stale** (fix 6) if it is older than 10 min and names no live A7-5 driver while the checks before it found no GPU
  compute process, < 1,500 MiB on every GPU and none of our launchers alive; a stale marker is logged and removed. Any
  other marker fails the preflight;
- ports 5000-5003 free;
- `/run/ti` >= 1.5 TB free, `/dev/shm` >= 250 GB free, MemAvailable >= 300 GB and memory-cgroup headroom >= 250 GiB.

**Claim.** All 8 markers are created atomically (`set -o noclobber`; content `rzA75_s<S> <pid> <time>`), taken-over
markers by an atomic rename, and all are removed by the exit handler, which no second signal can interrupt. If a marker is
taken between the preflight and the claim, the markers already created are released and the driver exits 3. While the
markers exist, the host is reserved: no other job of ours starts on it.

**Monitor** (every 60 s while a smoke test or a run holds the host; fixes 5, 9 and D-memory). It stops the run, kills our
tagged processes and ends the driver with:
- exit 6 on foreign GPU use: a compute process whose `/proc/<pid>/environ` lacks `RZ_A75_TAG=<tag>` (a process of another
  container has no `/proc` entry here and counts as foreign), or > 1,500 MiB used on a GPU none of our processes is on;
  a finding must still be there 30 s later. A process of this container that is exiting (no address space left, so no
  environment to read, while `nvidia-smi` still lists it) is not counted (review item 1);
- exit 7 when memory-cgroup headroom is < 100 GiB, before Ray's own memory monitor acts (section 2; review item 4);
- exit 8 at T0 + 71 h (X5);
- exit 0 in RZ-P1-only mode once the iteration-3 admitted set exists and its audit sample is drawn (section 8).

It also applies the **hang rule** (review item 8): while an attempt of `main.sh` runs, if nothing has changed for 120 min
in `main.log`, the stage directories (trackers, checkpoints, markers), the generated-question files and `temp_results`,
the attempt is stopped (our tagged processes killed) and counts as a failed attempt (X4). Normal progress changes them far
more often: the stage scripts' output goes to `main.log`, the questioner saves a checkpoint every step and the solver every
5 steps. A hung stage (for example a reward-server request without a timeout) therefore costs at most 2 h per attempt
instead of the host until T0 + 71 h.

Exits 6 and 7 are resumable; the waiter resumes when the host is free again. Between failed attempts the driver kills its
leftovers and waits up to 120 s until no tagged process is left before it classifies GPU use as foreign (fix 5b). The
monitor itself runs without the tag, so this cleanup never cuts its 60 s and 30 s waits short, and while the driver tears
down a failed attempt (file `teardown` in the storage path) the monitor skips its foreign-use rule; the driver's own check
after the teardown decides (review item 1). A dying process of ours can therefore no longer produce a false co-tenant
yield, which would restart the waiter with 3 fresh attempts and bypass X4.

**Run.** `rz_run.sh` exports `RZ_A75_TAG=rzA75_s<S>` and `RZ_SEED=<S>` and sources `rz_env.sh`, which sets:
- the venv first on `PATH` and `PYTHONPATH` = the working copy;
- `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME` on `/run/ti`, `HUGGINGFACENAME=offline`,
  `WANDB_MODE=disabled`;
- telemetry off: `RAY_USAGE_STATS_ENABLED=0`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `HF_HUB_DISABLE_TELEMETRY=1`
  (fix 17);
- every temp and cache directory under `/run/ti/rzero/tmp` and the Ray object-store cap (section 2).

It asserts that verl 0.3.1.dev0 is imported from the working copy. It then runs
`bash scripts/main.sh /root/models/Qwen3-4B-Base rzero_qwen3_4b_Base_s<S>` from `/run/ti/rzero/R-Zero` with
`STORAGE_PATH=/run/ti/rzero/s<S>`:
- Up to 3 attempts per driver start, each resuming (section 7), with the automatic resume fallback (section 7) and the
  admitted-set finding (X8, section 8) between attempts.
- A background watcher draws each iteration's audit sample as soon as its admitted set exists (section 5).
- `rz_clock.py` records stage completion times for the projected-finish rule (section 8).
- After S_3 is done, it writes the UMWP dumps (section 6) while it still holds the GPUs.
- Its exit handler, after any exit including a monitor stop, runs the extraction once more (section 5; review item 5).

Exit codes of `rz_run.sh`: 0 done (or the RZ-P1-only end), 2 refused, 3 preflight or clock error, 4 import check,
6 foreign GPU use, 7 memory headroom, 8 deadline or no time left (a printed `none`), 9 admitted set below one batch (X8),
10 three failed attempts (X4).

Large outputs stay on the local `/run/ti`. The shared volume receives only ledgers, stage times, admitted sets (a few MB),
audit samples and UMWP dumps.

**Smoke test** (`rz_smoke.sh`, Appendix B; 1.5-2.5 h, including the questioner's idle epochs). It is not data. Its
checkpoints are deleted afterwards and its log is kept. It runs under the same preflight, claim and monitor.
- S1: questioner stage, 1 GRPO step at rollout batch 32. Covers the 4 reward servers, the reward with BLEU clustering,
  FSDP with vLLM 0.11 rollout and weight sync on sm_120, checkpoint, merge and the targeted server shutdown.
- S2: the same stage continued to step 2 from the step-1 checkpoint (resume path), merged at step 2.
- S3: solver stage with 8 x 100 generations, pseudo-labels, filter, local parquet, 1 GRPO step at rollout batch 16 on
  8 GPUs, merge. S3 passes `trainer.save_freq=1` (fix 1): with the repository's `save_freq` 5 a 1-step run saves no step-1
  checkpoint (the trainer's final save would land at the step count reached after the idle epochs). The confirmatory
  solver keeps `save_freq` 5, whose checkpoints 5, 10, 15 and 20 are saved inside the loop.
- S4: the merged smoke solver loads in vLLM through `umwp_probe.py` (5 + 5 items). The dump is named
  `smoke_rzA75_<first 12 characters of the hostname>`, so smoke tests on two hosts never share a file (review item 10),
  and it is deleted before and after.

GO requires S1, S3 and S4. If S2 alone fails, the run still goes ahead; a failing resume is then handled by the automatic
resume fallback (section 7). On PASS the smoke test kills its processes, waits until they are gone, writes
`smoke PASSED epoch=<t> host=<hostname> stack=<fingerprint>` to `ext/rzero_runs/ledger_smoke.log` and keeps the 8 markers,
which `rz_run.sh` takes over within the hour (fix 8), so no launcher can take a GPU during the run's preflight.

**Watchdogs and janitors on the hosts (D-watchdogs).** `/root/hang_watchdog.sh` kills python processes started as
`python3 /home/tione/notebook/rsi_rl/(dirA/code/c0_probe|dirA3/code/umwp_probe|dirB/code/b_eval|dirB/code/b_train|dirA3/code/a3_guard).py`
whose GPU shows < 1,500 MiB for 15 min. The UMWP steps (smoke S4 and `rz_umwp.sh`) start the probe as
`/usr/bin/python3 /home/...umwp_probe.py`, which the watchdog's `^python3 ` pattern does not select; even if it did, a
loaded model holds far more than 1,500 MiB on the probe's GPU. The R-Zero stage processes run from `/run/ti/rzero/R-Zero`
and are outside its pattern (the monitor's hang rule covers them). `/root/ckpt_janitor2.sh` only touches
`/dev/shm/ckpt_b`, where A7-5 writes nothing.
`/root/a4_pair_watchdog.sh` (host B) starts queue runners on idle GPU pairs without looking at markers; the preflight
counts it as a job script, so no seed starts while it is alive (it exits by itself once its queue is done). On host A,
`/root/run_8b_probe_claimed.sh` (idle since 22 Sep, waiting on a claim owned by host B) would take any GPU that is idle for
90 s regardless of markers; the preflight counts it too, so it must be ended as part of host A's drain.

**Allowed changes (compute settings), each logged in the ledger and in a dated amendment before the run it affects:**
- GPU placement;
- vLLM memory and scheduling knobs (`gpu_memory_utilization`, `enforce_eager`, `max_num_seqs`, chunked prefill);
- training rollout `tensor_parallel_size`;
- per-device micro-batch sizes (the global batch is unchanged), CPU-offload flags and gradient checkpointing;
- Ray CPU and object-store settings, NCCL / CUDA environment variables and our own wait-loop timeouts;
- versions of the five helper packages;
- API-compatibility edits in the bundled verl or the scripts that vLLM 0.11 / torch 2.8 make necessary and that keep the
  computation identical (for example a renamed attribute on the weight-sync path).

**How an allowed change is applied (review item 7).** A change to the stage scripts or the bundled verl goes into the
patch: the patch is regenerated with `dev/mkpatch.py`, rehearsed with `dev/patchtest.sh` and `dev/designtest.sh` (CPU
only), and `A75_TREE` in `rz_common.sh` is set to the new tree hash, which `rz_setup.sh` otherwise refuses as fatal. The
amendment records the new patch and tree hashes and the new sha256 of every changed companion file. `rz_setup.sh` then
rebuilds the working copy (never while an A7-5 run is active; checkpoints under `/run/ti/rzero/s<S>` are kept, so a seed
resumes). The stack fingerprint changes with the patch, so a seed that has not started yet cannot reuse a smoke PASS from
the earlier stack and runs its own smoke test (seed 1 then no longer reuses seed 0's PASS). A change to environment
variables in `rz_env.sh` is recorded the same way and leaves the tree hash unchanged.

**Never changed:** everything listed in section 11.

## 5. (d) Admitted sets and the Opus audit

**Admitted set of iteration i.** The rows `upload.py` keeps for `<abbr>_solver_v<i>`: questions of Q_i that parsed and were
pseudo-labelled by S_(i-1), with 0.3 <= score <= 0.8 and a non-empty label. `upload.py`'s output is the dataset passed to
the solver trainer of iteration i, which additionally drops prompts over 2,048 tokens (`filter_overlong_prompts`) and the
last incomplete batch of 512 (`drop_last`) (fix 13). `rz_extract.py` reports how many admitted rows exceed 2,048 tokens,
computed with the trainer's tokenizer, messages and chat template. Files:
- `/run/ti/rzero/s<S>/generated_question/<abbr>_solver_v<i>_admitted.{parquet,jsonl}`, with columns problem, answer
  (majority pseudo-label) and score;
- plus the pre-filter pool `..._all_scored.json` and the raw generations `..._<k>.json.raw`.

**Extraction** (`rz_extract.py`, Appendix B; idempotent, CPU only, only on the seed's host). It is run by the watcher every
10 min, by the driver after the last attempt, and by the driver's exit handler after any exit, including a monitor stop
(review item 5), so an admitted set created shortly before a stop still gets its sample and summary at once. Concurrent
calls are serialized by a lock on the seed's local storage, `/run/ti/rzero/s<S>/extract.lock` (review item 9). It may also
be run by hand at any time, including after T0 + 71 h for the admitted sets that existed by then:
```bash
/usr/bin/python3 /home/tione/notebook/rsi_rl/ext/rzero_design/rz_extract.py --seed <S>          # iterations 1 2 3
```
For each iteration whose admitted parquet exists:
1. Checks that the jsonl and parquet row counts agree.
2. Draws `idx = random.Random(20260924 + i).sample(range(n), min(200, n))` over the n admitted rows. A repeated question
   counts by multiplicity, as in training. If n < 200 it takes all n (the solver of that iteration then cannot train,
   finding X8).
3. Writes the pseudo-run `/home/tione/notebook/rsi_rl/dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/admitted.jsonl` with fields
   id `rz<S>i<i>r<row>`, problem, prompt, `kind: "proposed"`, rz_row, rz_rank, rz_label and rz_score. This is the format
   the frozen workflow reads.
4. An existing sample is never redrawn.
5. Copies the full admitted set to `ext/rzero_runs/s<S>/` with a pool summary: generated, parsed, scored, admitted, unique
   admitted, rows over 2,048 tokens, solver batches of 512, score histogram, numeric-label share, the admitted parquet's
   mtime (for the T0 + 71 h rule) and sha256.

**Audit.** The frozen A3 200-item workflow runs as `rz_audit200_wf.js`: 8 Opus judges x 25 questions per run. Compared with
`a3-audit200-qwen-s1-r4-wf_93af8c2d-710.js` (sha256 `0c046d00...9a52`), the edits are the meta name and description, the
SSH port and the RUNS list (both from the workflow args), the job list (a sample under 200 items ends at n) and the judge
label (Appendix C). The prompt function is byte-identical, and with the default port the prompt text of each job equals
the frozen workflow's for the same run name (checked with stub agents in node):
- the judge fetches problems 0..199 of the Random(0)-shuffled pseudo-run and sees only id and problem text, never the
  pseudo-label or score;
- judges are told the questions were "written by a Qwen3-4B-Base language model and ADMITTED by a self-play filter";
- verdicts are SOLVABLE / UNSOLVABLE / AMBIGUOUS under the strict rule "a default assumption you would have to invent
  makes it UNSOLVABLE".

**Run-time input (fixes 2 and 3).** The workflow args are its only run-time input, and no hashed file is edited to run it:
`{"runs": [[S, i], ...], "port": P}`, validated to S in {0, 1, 2, 3}, i in {1, 2, 3}, no duplicates, 1-12 entries. An
optional third element n (1-200, default 200) is the sample size of an admitted set with fewer than 200 rows (the
`audit_sample_n` of its summary). Only pairs whose `admitted.jsonl` exists are listed, so the audit of iteration 1 can run
early, and seed 1 needs no edit. The SSH endpoint is pre-declared replaceable: `port` may be the port of any live host of
ours at <SERVER> that mounts the shared volume (default <HOST>; also <HOST>). The data are on NFS, so the fetched text
is byte-identical, and this is not a prompt change. Before each workflow run the operator creates the local part
directory the frozen prompt writes to:
```bash
mkdir -p <LOCAL_HOME>/AppData/Local/Temp/claude/<PROJECT>/43f8b914-5d13-4ce7-83fd-de9b7ae44f11/scratchpad/audit_parts
```
Example: `{"runs": [[0, 1]], "port": <HOST>}` as soon as seed 0's iteration-1 sample exists; later `{"runs": [[0, 2], [0, 3]]}`.

The audit of iteration i may start as soon as its sample exists (API only, no GPU). Iterations 1, 2 and 3 are audited for
every seed run; iteration 2 is descriptive. Audits of admitted sets that exist by T0 + 71 h may finish by T0 + 84 h
(section 8).

Then, for each run:
```bash
python3 /home/tione/notebook/rsi_rl/dirA3/code/a3_audit200_assemble.py --run /home/tione/notebook/rsi_rl/dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>
```
If assembly reports missing parts, the workflow is rerun for those runs only, at most twice. An audit counts as complete
only with a verdict for every sampled item. (The assembler covers the first min(200, n) shuffled items, so a smaller
sample assembles unchanged.)

**Ill-posed share** = (UNSOLVABLE + AMBIGUOUS) / verdicts, the A4/A7 definition. The `illposed_share` field written by the
assembly script counts UNSOLVABLE only and is not used for RZ-P1.

## 6. (e) UMWP dump of the final solver

`rz_run.sh` calls `rz_umwp.sh <S> 0` once `<abbr>_solver_v3/A75_DONE` exists. To run it by hand, claim 4 free GPUs G..G+3
the usual way and run `bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_umwp.sh <S> <G>`; it refuses after T0 + 71 h.
The confirmatory call is:
```bash
M=/run/ti/rzero/s<S>/models/rzero_qwen3_4b_Base_s<S>_solver_v3/global_step_15/actor/huggingface
env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=<G> \
  HF_HOME=/home/tione/notebook/rsi_rl/hf_home TOKENIZERS_PARALLELISM=false \
  /usr/bin/python3 /home/tione/notebook/rsi_rl/dirA3/code/umwp_probe.py --model $M --tag plain_rzero_s<S>_it3 --abstain_prompt 0
```
- **Probe and format.** The probe is frozen (sha256 `32de59f9...e56d`): 600 unanswerable UMWP questions plus their 600
  answerable originals, G = 8, T 1.0, max_new 1,024, seed 17. It uses the same interpreter and `HF_HOME` as every HX dump.
  The probe picks the base-model prompt ("Question: ... Answer: Let's think step by step.") whenever the model path contains
  "Base". That is why the abbreviation contains `Base`, and `rz_umwp.sh` refuses a path without it. The final solver is
  therefore probed exactly as the base model and every HX checkpoint.
- **Output and resume (fix 10).** `dirA4/results/umwp/plain_rzero_s<S>_it3.jsonl`. The probe appends only the ids it has
  not written. A kill mid-write can leave a partial last line, onto which the resumed first record would be appended
  (1,199 parseable ids). Without touching the frozen probe, `rz_umwp.sh` therefore cuts a trailing partial line before
  each call and calls the probe up to 3 times until the dump has 1,200 unique parseable ids; it reports unique parseable
  ids, not line counts. A dump counts at 1,200 unique parseable ids and must be complete by T0 + 72 h.
  `/run/ti/rzero/s<S>/A75_UMWP_DONE` marks a complete confirmatory dump.
- **Base anchor (not recomputed).** `dirA4/results/umwp/plain_qwen3_4b_base.jsonl`, 1,200 rows, sha256
  `225ad7eb493a2a3ae314ecdd7b0b90e152036089b0fbf3fa5ec6e4ab07b0b054`, produced by the same probe. CA = 172 / 600 = 0.2867,
  ACC_A 91.3.
- **CA definition** (as in `umwp_read.py` and HX): the share of the 600 unanswerable questions whose modal answer is a
  number with p_mode >= 0.5.
- **Descriptive dump.** `abs_rzero_s<S>_it3` (abstention prompt: ABS_U and ABS_A, against the existing `abs_qwen3_4b_base`).
- **Exploratory dump.** `plainchat_rzero_s<S>_it3` and `plainchat_qwen3_4b_base`: the same plain instruction inside the
  chat template, which the R-Zero solver was trained with (system "Please reason step by step, and put your final answer
  within \boxed{}"). They are reached through symlinks without "Base" in the path. The base chat-template dump is one
  shared NFS file, so only seed 0 writes it (seed 2 if seed 0 was abandoned; fix 16); if that seed has no final solver,
  the exploratory base read is not run. They are reported only as exploratory and cannot change RZ-P2.
- **Watchdog.** See section 4.2: `hang_watchdog.sh` does not select these probe processes, and each holds its GPU's memory.

## 7. (f) Checkpoints and resume

A co-tenant may stop our jobs at any time, and we yield at once: the monitor does it by itself (exit 6), and
`rz_stop.sh <S>` does it by hand (waiter, driver and every tagged process; files kept). Resuming is always
`rz_wait_and_run.sh <S>` (or `rz_run.sh <S>`); after an automatic yield the running waiter resumes by itself.
- The driver's first step kills leftovers of this seed only, meaning processes whose environment carries
  `RZ_A75_TAG=rzA75_s<S>`, and waits up to 120 s until they are gone. `rz_stop.sh`, the monitor, the P1 and P17 loops (by
  `RZ_VLLM_RUN`) and the driver's cleanup all select processes this way and nothing else. This was checked with test
  processes: tagged ones were killed, untagged ones and ones with another tag survived (`dev/designtest.sh` T1).
- It then repeats the host checks (seed and host, mode, preflight) and continues:

| interrupted during | on restart |
|---|---|
| a finished stage (`A75_DONE` present) | skipped |
| questioner training at step k | servers restarted; training resumes from the last complete checkpoint j (tracker file, written after the actor and `dataloader.pt` are saved; saved every step) via `trainer.load_checkpoint_path`; steps j+1.. are redone |
| questioner merge or server shutdown | training skipped (tracker >= 6); merge rerun unless `A75_MERGED_step5` exists |
| generation, pseudo-labelling or filter | all partial files of that iteration deleted; the three steps rerun from the start with the same seeds. The admitted set is fixed once its parquet exists and is never regenerated |
| solver training | resumes from the last complete checkpoint (every 5 steps: 5, 10, 15) |
| solver merge | rerun unless `A75_MERGED_step15` exists |
| UMWP dump | partial last line cut; the probe appends only the missing ids |

- Pruning never removes a checkpoint a later stage reads. The repository's `save_limit: 3` rule always keeps the step-5 and
  step-15 checkpoints. Our pruning runs only after `A75_DONE`.
- A resumed run is not bit-identical to an uninterrupted one; neither is an uninterrupted run. Every interruption and
  resume is written to the ledger and reported.
- **Resume fallback (automatic; fix 14).** Implemented in `rz_run.sh` (`resume_fallback` in `rz_common.sh`). After a failed
  attempt of a stage that resumed from checkpoint k > 0 and saved no newer checkpoint: if the attempt's part of `main.log`
  shows `Load from checkpoint:` followed by a traceback that passes through `load_checkpoint`, `_load_checkpoint` or
  `load_state_dict`, or if the previous attempt failed the same way after loading the same k, the stage directory is moved
  aside (`models/<stage>.discarded_<time>`, its `global_step_*` deleted, the rest kept), a ledger line is written, and the
  stage reruns from its start with the same seeds (the admitted set, kept outside the stage directory, is not
  regenerated). Tested on synthetic logs (`dev/designtest.sh` T7). S2 of the smoke test exercises the resume path itself.
- **Host lost.** A seed never resumes on another host (section 4.1): it is abandoned and reported, and a new seed id may
  start elsewhere.

## 8. (g) The 3-day clock, the drain rule and abort rules

**Clock (D-clock).** T0 = 2026-09-24 15:10 CST (A7 registration; `A75_T0` in `rz_common.sh`, `T0` in `rz_clock.py` and
`rz_read.py`). All A7-5 compute stops at T0 + 71 h = 2026-09-27 14:10 (the monitor enforces it, exit 8, and nothing starts
later). RZ-P2 is read at T0 + 72 h = 2026-09-27 15:10. Opus audits (API only) of admitted sets that exist by T0 + 71 h may
finish by T0 + 84 h = 2026-09-28 03:10, and RZ-P1 is read then. Interruptions by the co-tenant do not extend the window.
The audit sample of such a set is drawn during the run or, at the latest, by the driver's exit handler at the stop;
`rz_extract.py` (CPU only) may also be run by hand after T0 + 71 h for the sets that existed by then (section 5).

**Drain rule (fix 7, D-host).** From the moment this addendum's sha256 is appended to `docs/registration_hashes.txt`, no
new job of ours starts on host B except finishing the chains already running there (A5 CTRL-L s2 rnd/lowp to r3, SWAP
s1/s2/s4, A6 CLN1/RND1 s1/s2/s4 + Phi, Qwen3-8B own_ag s1 r3); expected free about 22:00-24:00 CST 24 Sep
(T0 + 6.8-8.8 h). The same rule holds on host A, whose chains (CTRL-L s0/s1, HZN s0/s2 r5, SWAP s0/s3, A6 s0/s3) are
expected to finish about 06:00-10:00 CST 25 Sep (T0 + 14.8-18.8 h). The waiters are started at the hash
(`rz_wait_and_run.sh 0` on host B, `rz_wait_and_run.sh 1` on host A) and start a seed as soon as all 8 GPUs of their host
are free of our markers, processes and job scripts (section 4.2). If a seed cannot start because our own jobs outlived the
drain, that is reported as such ("host still occupied by our own jobs"), not as a co-tenant outcome.

**Size estimate** from the environment investigation, not measured here, as upper values used by the projection:
- questioner stage 1.5-3.5 h (including about 7 min of idle epochs);
- generation plus pseudo-labelling plus filter 0.4-1.1 h;
- solver training plus merge 1.7-3.7 h;
- UMWP dumps up to 1.0 h; smoke test 1.5-2.5 h;
- total 12-26 h per seed on one 8-GPU host (upper: 3 x (3.5 + 1.1 + 3.7) + 1.0 = 25.9 h).

**Projected-finish rule (fix 9; `rz_clock.py`).** Stages in order are q1 a1 s1 q2 a2 s2 q3 a3 s3 u (questioner stage,
admitted set, solver stage, UMWP dumps). A stage's completion time is its marker's mtime; a stage is measured when it
started and ended inside one driver attempt, and once a stage type has been measured in any seed its estimate becomes
1.1 x its longest measurement (otherwise the upper values above). At each start (the waiter before the smoke test, the
driver before its preflight and again after its claim):
- **full mode** if now (+ 2.5 h for a smoke test) + the estimates of all remaining stages end by T0 + 71 h;
- otherwise **RZ-P1-only mode** if the remaining stages up to the iteration-3 admitted set end by T0 + 71 h: the run
  stops once that set exists and its audit sample is drawn, and RZ-P2 is declared not run for this seed;
- otherwise the seed is **not started**.
A committed mode is never upgraded (`ext/rzero_runs/s<S>/mode`); a later start can only downgrade it. Under the upper
values alone, full mode needs a start by T0 + 45.1 h (T0 + 42.6 h with a smoke test; the critique's T0 + 41 h leaves
1.6 h of margin), and RZ-P1-only mode by T0 + 49.8 h (T0 + 47.3 h with a smoke test).

| time (CST) | milestone |
|---|---|
| T0 = 24 Sep 15:10 | clock starts |
| at the hash | drain rule in force; `rz_setup.sh` on both hosts (CPU only), then the waiters |
| about 24 Sep 22:00-24:00 | host B free: smoke test, then seed 0 (upper estimate: done by about 26 Sep 04:40) |
| about 25 Sep 06:00-10:00 | host A free: seed 1, reusing the seed-0 smoke PASS on the same stack (upper estimate: done by about 26 Sep 12:00) |
| 26 Sep 09:46 (T0 + 42.6 h) | latest smoke-test start for full mode under the upper values alone; a later start needs measured stage times for full mode, else RZ-P1-only mode (until T0 + 47.3 h with a smoke test, T0 + 49.8 h without) |
| 27 Sep 14:10 (T0 + 71 h) | all A7-5 compute stops (monitor, exit 8); admitted sets existing now are the last audited |
| 27 Sep 15:10 (T0 + 72 h) | RZ-P2 read (section 9) on whatever is complete |
| 28 Sep 03:10 (T0 + 84 h) | last moment for the audits; RZ-P1 read |

**Abort rules.** Each outcome is reported with its code.

| code | trigger | outcome |
|---|---|---|
| X1 | no fully free 8-GPU host early enough for the projected-finish rule to allow even RZ-P1-only mode | NOT RUN: "no free 8-GPU host within the 3-day window" (stating whether our own jobs or the co-tenant held the hosts) |
| X2 | the smoke test fails and is not fixed with the allowed changes (section 4.2) within 8 h of accumulated debugging, or before the projected-finish rule stops allowing RZ-P1-only mode | NOT RUN: the failing step (S1-S4), component and error line |
| X3 | making the code run would need a change outside the allowed list | NOT RUN: the change that would be needed |
| X4 | a confirmatory stage fails 3 times with the same error (resume and fallback included; an attempt stopped by the hang rule counts as failed; driver exit 10) and allowed changes do not fix it within 4 h | that seed stops; if no seed can still finish, NOT RUN with the X2 wording |
| X5 | T0 + 71 h reached | compute stops; each registered prediction is read only if its inputs are complete; otherwise NOT RUN ("incomplete at the 3-day limit"), with the stage reached |
| X6 | foreign GPU use (co-tenant) | yield at once (monitor exit 6, or `rz_stop.sh`); resume when the host is free again; no clock extension |
| X7 | memory-cgroup headroom < 100 GiB | the run stops (exit 7, resumable) and resumes when the preflight's 250 GiB holds again |
| X8 | an admitted set with fewer rows than one rollout batch of 512 (after the trainer's overlong-prompt filter): `drop_last=True` leaves the dataloader empty and the repository's `assert len(train_dataloader) >= 1` fails on every attempt | a **finding about the default pipeline**, reported with n (`ext/rzero_runs/s<S>/finding_admitted_lt512.json`); no retry and no X4 debugging (driver exit 9). If it happens at iteration 3, RZ-P1 is read for that seed (its it1 and it3 admitted sets exist; the it3 audit covers min(200, n) items) and RZ-P2 is not run for it; at iteration 1 or 2 the seed counts for neither |

**Seed 1** starts when host A has drained and the projected-finish rule allows it; otherwise the item is run with 1 seed,
which the registration allows. A seed stopped by X4, X5 or X8, or run in RZ-P1-only mode, before its final solver exists
does not count as a seed run for RZ-P2.

**Fixed report text when not run:**
> "A7-5 (R-Zero replication, official code 5699329): NOT RUN - <code>: <reason>. Attempted: environment <built / failed>,
> smoke test <passed / failed at S<k>: <error>>, confirmatory seeds <none / seed <S> reached stage <stage>>. No RZ-P1 or
> RZ-P2 verdict is given, and no partial number is presented as a test of them."

## 9. Reads and verdict rules (`rz_read.py`, Appendix B)

```bash
python3 /home/tione/notebook/rsi_rl/ext/rzero_design/rz_read.py
```
- **Seeds.** Every seed id with an `s<S>.host` file is listed; an abandoned seed is reported under "abandoned" with its
  stages and never counts. More than two non-abandoned seed ids would break section 4.1; the read then gives no verdict.
- **RZ-P1, per seed.** ill(it3) - ill(it1) >= 10 points, where ill is the (UNSOLVABLE + AMBIGUOUS) share of the complete
  audit of that iteration's sample (200 items, or all n < 200). An iteration's audit counts only if its admitted set
  existed by T0 + 71 h (the parquet mtime in the summary) and the audit was assembled by T0 + 84 h. A seed counts as run
  for RZ-P1 once both its it1 and it3 audits count. RZ-P1 HOLDS if it holds in >= 1 seed run, FAILS if none, and is
  NOT RUN if no seed qualifies.
- **RZ-P2, per seed.** CA(`plain_rzero_s<S>_it3`) > CA(`plain_qwen3_4b_base`) = 0.2867, strictly. A seed counts as run once
  its final-solver dump is complete (1,200 unique parseable ids) by T0 + 72 h; a seed run in RZ-P1-only mode has no
  RZ-P2. RZ-P2 HOLDS if it holds in every seed run, FAILS if not, and is NOT RUN if no seed qualifies.
- **Evaluated separately.** RZ-P1 needs only the admitted sets of iterations 1 and 3. Iteration 3's admitted set exists
  before S_3 is trained, so RZ-P1 can be evaluated even if S_3 misses the deadline, runs in RZ-P1-only mode or cannot
  train (X8). In those cases RZ-P2 is NOT RUN for that seed.

**Secondary reads** are declared now and printed next to the verdicts, but they cannot change them:
- **Numeric-label subset.** RZ-P1's difference restricted to audited items whose pseudo-label is a plain number or
  fraction. The frozen prompt defines SOLVABLE as "determines a unique numeric answer", while R-Zero's questioner may also
  write problems with symbolic answers.
- **UNSOLVABLE-only difference.**

**Descriptive reads:**
- the iteration-2 audit;
- pool sizes (generated, parsed, scored, admitted, unique admitted, over 2,048 tokens) and score histograms per iteration;
- ACC_A of the final solver against the base's 91.3;
- ABS_U and ABS_A under the abstention prompt;
- any X8 finding.

**Exploratory read:** the chat-template CA pair (final solver against base).

**Pre-stated readings and flagged downside:**
- Both hold: the admitted-set drift and the UMWP harm replicate in the official R-Zero pipeline under its own default
  filter.
- RZ-P1 fails: the paper states that R-Zero's default pipeline did not raise the audited ill-posed share of admitted
  questions by 10 points over 3 iterations, and restricts the drift claim to the loops where it was measured.
- RZ-P2 fails: the paper states that R-Zero's final solver answers UMWP's unanswerable questions no more confidently than
  the base model under the registered prompt.
- X8: the paper states that R-Zero's default pipeline admitted fewer than 512 questions at iteration i (n given), so its
  solver could not train; this is a property of the default pipeline, reported as such.
- Not run: stated with the code and reason.
- No outcome is re-analysed to look better. The exploratory chat-template read is labelled as such whatever it shows.
- RZ-P1 compares iterations 1 and 3 by design. Iteration 1's questioner is already trained 5 steps against the base
  solver, so a drift that happens entirely within iteration 1 is not detected by RZ-P1.

## 10. Every deviation from the repository

| # | Deviation | Why | Effect on the algorithm |
|---|---|---|---|
| D0 | CRLF stripped from 92 text files of the shared copy | the transfer through Windows added them; bash cannot run CRLF scripts | none (restores bytes) |
| D1 | framework versions: torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.1, ray 2.49.2, flash-attn 2.8.1, tensordict 0.10.0, datasets 4.3.0, numpy 1.26.4 (pins: 2.7.0, 0.9.1, 4.52.4, 2.46.0, 2.7.4.post1, 0.8.3, 3.6.0, 2.2.6) | the pinned binaries have no sm_120 kernels; building vLLM 0.9.1 needs GitHub | none intended; numerical kernels differ, as on any new hardware |
| D2 | venv with 5 helper packages (versions as pinned or current) | missing on the cluster | none |
| D3 | math12k read from a local parquet copy fetched through hf-mirror (P8) | Hub unreachable | none (same dataset) |
| D4 | admitted set kept as a local parquet and jsonl instead of a private Hub dataset (P4, P5, P7) | Hub unreachable | none (same rows) |
| D5 | no wandb logger (P9); `WANDB_MODE=disabled` | no wandb | none |
| D6 | targeted server shutdown instead of `pkill python` (P1, P2); timer cancel (P3) | the repository would kill every python process on the host | none |
| D7 | private Ray instance with its object store capped at 32 GiB (P11; Ray's default is 30 % of RAM in `/dev/shm`); every temp and cache directory on `/run/ti/rzero/tmp`; usage statistics and telemetry off | shared hosts; a memory cgroup that counts `/dev/shm`; no outbound reporting | none |
| D8 | repository benchmark evaluation skipped (P10) | Hub datasets, OpenAI re-check, all GPUs; RZ-P2 is measured with the registered probe | none on training |
| D9 | 3 iterations instead of 5 (P12) | registered | the loop stops after iteration 3 |
| D10 | resume (P15), stage markers and guards, removal of unused checkpoints after a stage (P16), server readiness wait that fails fast on a dead server (P17), driver retries, automatic resume fallback (section 7) | co-tenant preemption; storage | none; a resumed stage redoes at most the steps since its last checkpoint, a fallback reruns the stage from its start with the same seeds |
| D11 | raw generations and the scored pool kept (P6, P14); unused MATH-500 fetch removed (P13) | audit bookkeeping; one network dependency fewer | none (outputs only; the fetched data were unused) |
| D12 | seed knob (P18): seed 0 = repository seeds; seeds 1-3 shifted (section 4.1) | 1-2 seeds registered; replacement ids after an abandoned run | none for seed 0 |
| D13 | smoke hooks (P19), unset in every confirmatory run; the smoke test's 1-step solver saves every step | smoke test | none |
| D14 | exclusive use of a fully free host; atomic GPU markers; 60 s monitor (foreign GPU use, memory, deadline, hang rule); drain rule and waiters | vLLM memory fractions are of total memory; co-tenancy | none |
| D15 | any allowed compute change made during the run (section 4.2) | listed in amendments as it happens | none by construction |
| D16 | projected-finish rule: a seed may run in RZ-P1-only mode and stop once its iteration-3 admitted set exists (section 8) | 3-day window | the loop ends after the iteration-3 filter for that seed; everything before it is unchanged |
| D17 | a seed never resumes on another host; a replacement seed id (2 or 3) starts instead (section 4.1) | host-local storage | none |

## 11. What stays exactly at the repository default

Everything not in section 10:
- base model and prompts (questioner system and user prompts, solver system prompt, pseudo-labelling prompt);
- questioner GRPO: 512 x n 4, global batch 16, 6 steps, lr 1e-6, KL low_var_kl 1e-2, rollout T 1.0 and top_p 0.99, max
  response 4,096, save every step, merge step 5, `total_epochs` 1000;
- reward `caller_penalty.py`, including its quirks: `answers[-1]` takes the last character of the boxed string, and "None"
  counts in the denominator;
- reward servers: 4 x n 10, T 1.0, top_k 40, `gpu_mem_util` 0.8, and the idle matmul worker;
- BLEU clustering: threshold 0.5, over the whole batch;
- generation: 8 x 1,000, T 1.0, top_p 0.95, max 4,096;
- pseudo-labelling: 9 samples, T 1.0, top_k 40, `grade_answer` majority, the "证明" / "box" / "text" drops, the 3,600 s
  shard timeout;
- filter: 0.3 <= score <= 0.8;
- solver GRPO: 512 x n 5, global batch 128, micro-batch 1, 20 steps, val every 4, save every 5, merge step 15, reward
  `math.py` 0.9 / 0.1, `total_epochs` 100, `max_prompt_length` 2,048 with `filter_overlong_prompts`, `drop_last`;
- validation on math12k test before training;
- `save_limit` 3, TP 2, `gpu_memory_utilization` 0.7 / 0.85 / vLLM default;
- GPU layout: 0-3 train plus 4-7 servers; 8 shards; 8 solver GPUs;
- Ray `num_cpus` 16.

## 12. Known limitations, declared before data

1. The frozen judge prompt was written for our own loop's numeric word problems. R-Zero questions with non-numeric answers
   may be judged UNSOLVABLE by its "unique numeric answer" clause. See the numeric-label secondary read.
2. The plain UMWP prompt is the base-model format, not the chat format the R-Zero solver was trained with. See the
   exploratory chat read.
3. The base anchor is an existing dump (same probe and seed), not a re-run.
4. With 1-2 seeds, "in every seed run" and "in >= 1 of the seeds run" are weak, as registered.
5. The wall-clock figures are estimates; the smoke test and stage times are logged and reported.
6. The projected-finish rule, X8 or a lost host can leave RZ-P2 unmeasured for a seed while RZ-P1 is still read.
7. An admitted set under 200 rows gives a coarser ill-posed share (all n items audited).

## 13. Files and hashes (all in `/home/tione/notebook/rsi_rl/ext/rzero_design/`)

| file | role | sha256 |
|---|---|---|
| `rzero_A7_5.patch` | the patch (section 3, Appendix A) | `7b8c11c0a83cf16b0cad013df567d62c41ff7c2736aa958f91e6ab07c541e188` |
| `rz_setup.sh` | (a) environment build | `e0924be2a0449550bb032b1772292b9c6ea47e48df804bfe30e5514f0e8fce33` |
| `rz_env.sh` | run environment (sourced) | `9c3312bfa4f15767dbc2e1d795db4919951650e41ea2899d6f0cb26b8e14b092` |
| `rz_common.sh` | constants, preflight, markers, monitor, tag-based cleanup, fallback (sourced) | `a6a34b9d2654758e31ae1c0e3b441dd3662cd89c9d741276e74daad02a6d02eb` |
| `rz_wait_and_run.sh` | waiter: drain, smoke test, seed run, resume (section 4.2) | `653b4e9da13ede2579c9c33b0ca27086fd57f6d660e6f03e50f783c6f8d54d28` |
| `rz_smoke.sh` | smoke test (section 4.2) | `fe92ea4127f89bc4f9fe2a7d68351d231f29883ce831229301fec10af60bb90e` |
| `rz_run.sh` | (c) driver for one seed | `561d1b8ebf79f6a13f86a65b41b8dbfda643174c2aace9e8d5a66c898adc466b` |
| `rz_stop.sh` | (f) stop / yield / abandon | `0410d01600c28b719284e1c0cdf61df23c9cdf611ab866ab2b6e8de9399997b6` |
| `rz_clock.py` | clock and projected-finish rule (section 8) | `a29a51f1d6863196c6c56c3f81bc00ae440a4be021872d93e0d61226493a841c` |
| `rz_extract.py` | (d) admitted sets and audit samples | `1e64b05425b2e2eac52498af293d94926aa5224086df9d1f9eca40a426cd9d4b` |
| `rz_audit200_wf.js` | (d) frozen Opus audit workflow; runs and port from args (Appendix C) | `b4d89f1d2f977e628ff2ae71788bf9a7420d93c05ab7b5f3001c80a6d28baafc` |
| `rz_umwp.sh` | (e) UMWP dumps | `22126f180c6a4e1b73d858a104f022e0151639f8386a0032dba092980fb4fc8b` |
| `rz_read.py` | reads and verdicts (section 9) | `1fbb53505a2af6c6fc68b4226dc9c68ccc8c0d8568bc1d7e9d7728a3c47065a1` |
| `dev/mkpatch.py` | provenance: generates the patch | `a227888a39ef295f40774031867dc5fe06c933315b6bcdd76a8eefda8e5e9af9` |
| `dev/patchtest.sh` | provenance: CPU-only patch checks | `8fafcbb090b36e3476d5a4bf93cead63ec6b06cde44da283e677cf80e7c37429` |
| `dev/designtest.sh` | provenance: CPU-only script checks | `57b92a3e92c3c0553d33da81e9d57e5bc0f1c9132cb062cd1570a325257f4a95` |

Inputs used unchanged (sha256):
- `dirA3/code/umwp_probe.py` `32de59f95a296fa0c026b44cf8a6110172898b87b258179ab3a54da00ee1e56d`
- `dirA3/code/a3_audit200_assemble.py` `212cc987bfb887e856cc1a6f1a4b6509e0c7bb372686bd3b1b31ae522bc2bc83`
- `dirA3/code/a3_part_check.py` `6468c1df385a0f88316ca9a952c45752559d111a09c08e6aee60ac0b3af317c5`
- `dirA4/results/umwp/plain_qwen3_4b_base.jsonl` `225ad7eb493a2a3ae314ecdd7b0b90e152036089b0fbf3fa5ec6e4ab07b0b054`
- `dirA4/results/umwp/abs_qwen3_4b_base.jsonl` `03e0340b00c2ca4dc0e39bd2f3de0e3a77ee1a94c6ae1908506401e0ae19b232`
- frozen audit workflow `a3-audit200-qwen-s1-r4-wf_93af8c2d-710.js`
  `0c046d00a14687349995e57bea90564831adfed974a39caefcad83dc88899a52`
- official code: commit `5699329d018d79535b7910abdedf5a6eebf355fd`; patched working-tree hash registered as
  `69ae729158dce32ea0e0d59216b2542001affbb11ab61739093020351a1f7ea2` (`A75_TREE`; file list sorted with `LC_ALL=C`). Before review item 6 the list was sorted in the caller's
  locale: under en_US.UTF-8 the same tree gave `d3a72c922a3ddbdb895f4e7b3a664a60e1e67e7c46c77641c489a864ac44b0b6`, and the
  critiqued draft's patch gave `407cb284f0703832ed413cba7d6a754881ef1c2ff4cc04151e439f1fc7546d58`.

Written at run time (not part of the hash): `ext/rzero_runs/ledger_{setup,smoke,s0,s1,s2,s3}.log`, `ext/rzero_runs/s<S>/`
(admitted sets, summaries, `events.tsv`, `mode`, findings), `ext/rzero_runs/s<S>.host`, `ext/rzero_runs/s<S>.abandoned`,
`ext/rzero_data/`, `dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/`, `dirA4/results/umwp/*rzero*`, and `/run/ti/rzero/`.

## 14. Revisions after the adversarial critique and the verification review

| fix | change | where |
|---|---|---|
| 1 | smoke S3 passes `trainer.save_freq=1` (smoke only) | `rz_smoke.sh`; section 4.2 |
| 2 | audit RUNS come from the workflow args, validated; the args are the only run-time input | `rz_audit200_wf.js`; section 5, Appendix C |
| 3 | SSH endpoint pre-declared replaceable (`port` arg); `mkdir -p` of the local part directory before each run | `rz_audit200_wf.js`; section 5 |
| 4 | host affinity file, refusal elsewhere, abandon + replacement seed ids | `rz_common.sh` (`seed_ok`, `claim_host`), `rz_run.sh`, `rz_wait_and_run.sh`, `rz_extract.py`, `rz_stop.sh`, `rz_read.py`; section 4.1 |
| 5 | 60 s monitor for foreign GPU use (exit 6); wait up to 120 s for our tagged processes before classifying | `rz_common.sh` (`foreign_gpu`, `start_monitor`, `kill_tag`), `rz_run.sh`, `rz_smoke.sh`; section 4.2 |
| 6 | stale-marker rule | `rz_common.sh` (`marker_ok`); section 4.2 |
| 7 | drain rule with expected free times | section 8; `rz_wait_and_run.sh` |
| 8 | smoke keeps its markers on PASS; the run takes them over within the hour | `rz_common.sh` (`takeover_ok`), `rz_smoke.sh`; section 4.2 |
| 9 | smoke 1.5-2.5 h; projected-finish rule for starts and RZ-P2; audits by T0 + 84 h | `rz_clock.py`, `rz_run.sh`, `rz_wait_and_run.sh`, `rz_read.py`; sections 5, 8, 9 |
| 10 | partial last line cut, up to 3 probe calls, unique parseable ids | `rz_common.sh` (`trim_partial`, `n_ids`), `rz_umwp.sh`; section 6 |
| 11 | P17 fails fast when a server process is gone | `rzero_A7_5.patch` via `dev/mkpatch.py`; section 3 |
| 12 | admitted set < 512 rows is finding X8, no retry | `rz_common.sh` (`small_admitted`), `rz_run.sh`, `rz_read.py`; sections 8, 9 |
| 13 | admitted-set wording; overlong-prompt count | `rz_extract.py`; sections 1, 5 |
| 14 | resume fallback implemented | `rz_common.sh` (`resume_fallback`), `rz_run.sh`; section 7 |
| 15 | tree-hash mismatch fatal (setup and preflight) | `rz_setup.sh`, `rz_common.sh`; section 2 |
| 16 | base chat-template dump only from seed 0 (2 if 0 abandoned) | `rz_umwp.sh`; section 6 |
| 17 | telemetry off | `rz_env.sh`; section 4.2 |
| D-host, D-clock, D-memory, D-watchdogs | waiters per host, absolute clock, memory cap and headroom checks, watchdog notes and atomic markers | `rz_wait_and_run.sh`, `rz_clock.py`, `rz_env.sh`, P11, `rz_common.sh`; sections 2, 4.2, 8 |

**Verification review.** It confirmed the 17 fixes and the four decisions, every hash, the appendices, the tree
rebuilt independently and every repository default. The ten items below change only our run-control scripts, never the
algorithm, a default or a verdict rule. Tests: `dev/designtest.sh` T4 f-g, T5, T9, T10 i and T12-T15 (CPU only).

| item | change | where |
|---|---|---|
| 1 | the monitor runs without `RZ_A75_TAG`, so the driver's cleanup no longer cuts its waits short; it skips the foreign-use rule while the driver tears down a failed attempt (`teardown` file), whose own check decides; an exiting process of this container (no address space) is not counted as foreign | `rz_common.sh` (`start_monitor`, `stop_reason`, `foreign_gpu`), `rz_run.sh`; section 4.2 |
| 2 | only `full`, `p1only` or `none` is a projection; any other `rz_clock.py` output is a clock error (waiter: retry every 60 s; driver: exit 3); a smoke test stopped at T0 + 71 h is logged as X5 | `rz_wait_and_run.sh`, `rz_run.sh`; section 4.2 |
| 3 | markers a passed smoke test kept and no run took over are removed when the waiter ends and by `rz_stop.sh <S>` | `rz_common.sh` (`drop_smoke_markers`), `rz_wait_and_run.sh`, `rz_stop.sh`; section 4.2 |
| 4 | exit-7 threshold 100 GiB (Ray's memory monitor acts at about 70-75 GiB of headroom) | `rz_common.sh` (`stop_reason`); sections 2, 4.2, 8 (X7) |
| 5 | the driver's exit handler also runs the extraction; `rz_extract.py` may be run by hand after T0 + 71 h for sets that existed by then | `rz_run.sh`, `rz_extract.py`; sections 4.2, 5, 8 |
| 6 | tree hash over a file list sorted with `LC_ALL=C`; `A75_TREE` updated | `rz_setup.sh`, `rz_common.sh`, `dev/patchtest.sh`; sections 2, 13 |
| 7 | how an allowed change is applied (patch regenerated and rehearsed, `A75_TREE` updated, hashes in the amendment, no reuse of an earlier smoke PASS) | section 4.2 |
| 8 | hang rule: no progress for 120 min stops the attempt, which counts as failed (X4) | `rz_run.sh` (`hung`, `extra_reason`); sections 4.2, 8 (X4), 10 (D14) |
| 9 | concurrent extractions serialized by a lock | `rz_extract.py`; section 5 |
| 10 | per-host name for the smoke S4 dump, deleted before and after | `rz_smoke.sh`; section 4.2 |

## Appendix A: `rzero_A7_5.patch` (apply with `patch -p1` to the CR-stripped copy)

```diff
diff -ruN a/examples/config.yaml b/examples/config.yaml
--- a/examples/config.yaml
+++ b/examples/config.yaml
@@ -1,6 +1,6 @@
 data:
-  train_files: hiyouga/math12k@train
-  val_files: hiyouga/math12k@test
+  train_files: /run/ti/rzero/data/math12k_train.parquet  # A7-5 (P8): local copy of hiyouga/math12k (fetched once via hf-mirror)
+  val_files: /run/ti/rzero/data/math12k_test.parquet
   prompt_key: problem
   answer_key: answer
   image_key: images
@@ -80,7 +80,7 @@
   max_steps: null
   project_name: easy_r1
   experiment_name: qwen2_5_7b_math_grpo
-  logger: ["console", "wandb"]
+  logger: ["console"]  # A7-5 (P9): no wandb
   nnodes: 1
   n_gpus_per_node: 8
   val_freq: 3  # -1 to disable
diff -ruN a/question_evaluate/evaluate.py b/question_evaluate/evaluate.py
--- a/question_evaluate/evaluate.py
+++ b/question_evaluate/evaluate.py
@@ -61,7 +61,7 @@
     with open(INPUT_FILE, "r") as f:
         data = json.load(f)
     # Clean up the input file immediately after loading to save space
-    os.remove(INPUT_FILE)
+    os.rename(INPUT_FILE, INPUT_FILE + ".raw")  # A7-5 (P14): keep the raw generations (repository: os.remove)
 except FileNotFoundError:
     print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
     exit()
@@ -86,7 +86,7 @@
     model=args.model,
     tokenizer=args.model,
     gpu_memory_utilization=0.85,
-    seed=int(args.suffix),
+    seed=int(args.suffix) + 100 * int(os.getenv("RZ_SEED", "0")),  # A7-5 (P18): seed knob; RZ_SEED=0 is the repository default
 )
 sample_params = vllm.SamplingParams(
     max_tokens=4096,
diff -ruN a/question_evaluate/evaluate.sh b/question_evaluate/evaluate.sh
--- a/question_evaluate/evaluate.sh
+++ b/question_evaluate/evaluate.sh
@@ -25,7 +25,10 @@
     fi
   done
 ) &
+timer_pid=$!
 
 for i in {1..7}; do
   wait ${pids[$i]} 2>/dev/null
 done
+# A7-5 (P3): cancel the 3,600 s timer once every shard has ended, so that it cannot kill -9 recycled PIDs later
+kill $timer_pid 2>/dev/null
diff -ruN a/question_evaluate/upload.py b/question_evaluate/upload.py
--- a/question_evaluate/upload.py
+++ b/question_evaluate/upload.py
@@ -10,7 +10,7 @@
 print(STORAGE_PATH)
 with open('tokens.json', 'r') as f:
     token = json.load(f)['huggingface']
-login(token=token)
+# login(token=token)  # A7-5 (P4): no Hugging Face login; nothing is pushed
 parser = argparse.ArgumentParser()
 parser.add_argument("--repo_name", type=str, default="")
 parser.add_argument("--max_score", type=float, default=0.7)
@@ -28,6 +28,9 @@
         print(f"File {args.experiment_name}_{i}_results.json not found")
         continue
 
+# A7-5 (P6): keep the whole scored pool (before the filter) for the audit denominators
+with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_all_scored.json', 'w') as f:
+    json.dump(datas, f)
 
 for i in range(8):
     try:
@@ -50,7 +53,12 @@
     dataset_dict = {"train": train_dataset}
     config_name = f"{args.experiment_name}"
     dataset = DatasetDict(dataset_dict)
-    dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)
+    # A7-5 (P5): local files instead of the Hub push below; the parquet is written last and atomically (it marks this step done)
+    # dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)
+    out = f"{STORAGE_PATH}/generated_question/{args.repo_name}_admitted"
+    train_dataset.to_json(out + ".jsonl")
+    train_dataset.to_parquet(out + ".parquet.tmp")
+    os.replace(out + ".parquet.tmp", out + ".parquet")
 
 
 
diff -ruN a/question_generate/question_generate.py b/question_generate/question_generate.py
--- a/question_generate/question_generate.py
+++ b/question_generate/question_generate.py
@@ -54,10 +54,10 @@
         model=args.model,
         tokenizer=args.model,
         # gpu_memory_utilization=0.8,
-        seed=int(args.suffix),
+        seed=int(args.suffix) + 100 * int(os.getenv("RZ_SEED", "0")),  # A7-5 (P18): seed knob; RZ_SEED=0 is the repository default
     )
     dataset_handler = get_dataset_handler("math")
-    questions, answers = dataset_handler.load_data()
+    questions, answers = [""], [""]  # A7-5 (P13): the repository fetches the MATH-500 csv here but never uses it
     question = questions[0]
     answer = answers[0]
     chat = [
diff -ruN a/scripts/main.sh b/scripts/main.sh
--- a/scripts/main.sh
+++ b/scripts/main.sh
@@ -1,24 +1,27 @@
 Base_model=$1
 Model_abbr=$2
 echo "Model_abbr: $Model_abbr"
+# A7-5 (P16): stage guards. A stage whose A75_DONE marker exists is skipped; the loop stops if a stage ends without it.
+stage(){ local out=$1; shift; [ -f ${STORAGE_PATH}/models/$out/A75_DONE ] || bash "$@"; [ -f ${STORAGE_PATH}/models/$out/A75_DONE ] || { echo "A7-5: stage $out did not complete"; exit 1; }; }
 # Initialize first iteration with base model
-bash scripts/questioner_train_penalty.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1
-bash scripts/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1
+stage ${Model_abbr}_questioner_v1 scripts/questioner_train_penalty.sh  $Base_model $Base_model ${Model_abbr}_questioner_v1
+stage ${Model_abbr}_solver_v1 scripts/solver_train.sh $Base_model ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_5/actor/huggingface ${Model_abbr}_solver_v1
 
 
-for i in {2..5}; do
+for i in {2..3}; do  # A7-5 (P12): 3 iterations (repository: 5)
     prev=$((i-1))
     
-    bash scripts/questioner_train_penalty.sh \
+    stage ${Model_abbr}_questioner_v${i} scripts/questioner_train_penalty.sh \
         ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
         ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_5/actor/huggingface \
         ${Model_abbr}_questioner_v${i}
 
     # Train solver
-    bash scripts/solver_train.sh \
+    stage ${Model_abbr}_solver_v${i} scripts/solver_train.sh \
         ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_15/actor/huggingface \
         ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_5/actor/huggingface \
         ${Model_abbr}_solver_v${i}
 done
 
-bash evaluation/evaluate.bash $Base_model
+# A7-5 (P10): the repository's benchmark evaluation is not run (Hub datasets, OpenAI re-check, all GPUs); RZ-P2 uses umwp_probe
+# bash evaluation/evaluate.bash $Base_model
diff -ruN a/scripts/questioner_train_penalty.sh b/scripts/questioner_train_penalty.sh
--- a/scripts/questioner_train_penalty.sh
+++ b/scripts/questioner_train_penalty.sh
@@ -10,9 +10,31 @@
 
 echo "RUN_ID=$RUN_ID"
 
+# A7-5 (P15): resume. k = last complete checkpoint of this stage (the repository's tracker file). Training is skipped when it
+# is complete; otherwise it resumes from global_step_k through the repository's trainer.load_checkpoint_path.
+CKPT=${STORAGE_PATH}/models/$save_path; Q_STEPS=${RZ_Q_STEPS:-6}; Q_MERGE=${RZ_Q_MERGE:-5}; LOAD_ARG=""; k=0
+[ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
+[ "$k" -gt 0 ] && [ "$k" -lt $Q_STEPS ] && [ -f $CKPT/global_step_$k/dataloader.pt ] && LOAD_ARG="trainer.load_checkpoint_path=$CKPT/global_step_$k"
+echo "A7-5: $save_path last complete step $k $LOAD_ARG"
+if [ "$k" -lt $Q_STEPS ]; then
+
 # 启动 vllm 服务（记录 PID）
 bash vllm_service_init/start.sh $solver_model_path $RUN_ID
 echo "vLLM services started with RUN_ID=$RUN_ID"
+# A7-5 (P17): wait until the 4 reward servers accept connections (at most 30 min each). If a server has died (no live process
+# tagged RZ_VLLM_RUN=$RUN_ID with --port <p>, checked from 30 s on) or 30 min pass, stop this stage's servers and fail at once.
+a75_alive(){ local f; for f in $(grep -lzx "RZ_VLLM_RUN=$RUN_ID" /proc/[0-9]*/environ 2>/dev/null); do tr '\0' ' ' < ${f%/environ}/cmdline 2>/dev/null | grep -q -- "--port $1 " && return 0; done; return 1; }
+for p in 5000 5001 5002 5003; do t=0
+  until (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; do
+    for q in 5000 5001 5002 5003; do [ $t -lt 30 ] || a75_alive $q || { echo "A7-5: reward server on port $q is not running"; t=9999; break; }; done
+    if [ $t -ge 1800 ]; then
+      echo "A7-5: reward servers not ready; stopping them and failing this attempt"
+      for sig in TERM KILL; do for d in /proc/[0-9]*; do tr '\0' '\n' 2>/dev/null < $d/environ | grep -qx "RZ_VLLM_RUN=$RUN_ID" && kill -$sig ${d#/proc/} 2>/dev/null; done; sleep 15; done
+      exit 1
+    fi
+    sleep 10; t=$((t+10))
+  done
+done
 
 # 开始训练 Questioner
 echo "Start training questioner: $questioner_model_path -> $save_path"
@@ -30,17 +52,30 @@
     data.format_prompt=./examples/format_prompt/questioner.jinja \
     worker.rollout.n=4 \
     worker.actor.global_batch_size=16 \
-    trainer.max_steps=6 \
-    trainer.save_freq=1
+    trainer.max_steps=$Q_STEPS \
+    trainer.save_freq=1 \
+    data.seed=$((1+${RZ_SEED:-0})) worker.rollout.seed=$((1+${RZ_SEED:-0})) $LOAD_ARG ${RZ_SMOKE_ARGS}
 
+fi
 sleep 5
 
 # 合并模型
 echo "merging model"
-python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor
+[ -f $CKPT/A75_MERGED_step$Q_MERGE ] || { python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_$Q_MERGE/actor && touch $CKPT/A75_MERGED_step$Q_MERGE; }
 
 sleep 10
 
-pkill python
+# A7-5 (P1): was `pkill python`, which kills every python process on the host. Stop only this stage's 4 reward servers:
+# the processes whose environment carries RZ_VLLM_RUN=$RUN_ID (set in vllm_service_init/start.sh, inherited by their children).
+for sig in TERM KILL; do
+  for d in /proc/[0-9]*; do tr '\0' '\n' 2>/dev/null < $d/environ | grep -qx "RZ_VLLM_RUN=$RUN_ID" && kill -$sig ${d#/proc/} 2>/dev/null; done
+  sleep 15
+done
+# A7-5 (P16): completion marker, then removal of the checkpoints the pipeline never reads (storage only)
+k=0; [ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
+if [ "$k" -ge $Q_STEPS ] && [ -f $CKPT/A75_MERGED_step$Q_MERGE ]; then
+  touch $CKPT/A75_DONE
+  for g in $CKPT/global_step_*; do [ "$g" = "$CKPT/global_step_$Q_MERGE" ] || rm -rf "$g"; done
+fi
 
 echo "questioner training finished"
diff -ruN a/scripts/solver_train.sh b/scripts/solver_train.sh
--- a/scripts/solver_train.sh
+++ b/scripts/solver_train.sh
@@ -7,13 +7,26 @@
 echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 
 
 export VLLM_DISABLE_COMPILE_CACHE=1
+# A7-5 (P16): generate -> pseudo-label -> filter runs once per iteration. Its admitted set is a local parquet that upload.py
+# writes last and atomically (P5); the trainer reads it (P7). An interrupted attempt is discarded and rerun from the start.
+GQ=${STORAGE_PATH}/generated_question; ADMITTED=$GQ/${experiment_name}_admitted.parquet
+if [ ! -f $ADMITTED ]; then
+rm -f $GQ/${experiment_name}_[0-7].json $GQ/${experiment_name}_[0-7].json.raw $GQ/${experiment_name}_[0-7]_results.json
 echo 'start generate question'
-bash question_generate/question_generate.bash $questioner_model_path 1000 $experiment_name
+bash question_generate/question_generate.bash $questioner_model_path ${RZ_GEN_N:-1000} $experiment_name
 echo 'start evaluate generated question'
 bash question_evaluate/evaluate.sh $solver_model_path $experiment_name
 echo 'start upload'
 python question_evaluate/upload.py --repo_name ${experiment_name} --max_score 0.8 --min_score 0.3 --experiment_name ${experiment_name}
+fi
+[ -f $ADMITTED ] || { echo "A7-5: no admitted set for $experiment_name"; exit 1; }
 echo 'start train'
+# A7-5 (P15): resume from the last complete checkpoint (saved every 5 steps), as in the questioner stage
+CKPT=${STORAGE_PATH}/models/${experiment_name}; S_STEPS=${RZ_S_STEPS:-20}; S_MERGE=${RZ_S_MERGE:-15}; LOAD_ARG=""; k=0
+[ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
+[ "$k" -gt 0 ] && [ "$k" -lt $S_STEPS ] && [ -f $CKPT/global_step_$k/dataloader.pt ] && LOAD_ARG="trainer.load_checkpoint_path=$CKPT/global_step_$k"
+echo "A7-5: $experiment_name last complete step $k $LOAD_ARG"
+if [ "$k" -lt $S_STEPS ]; then
 
 python3 -m verl.trainer.main \
     config=examples/config.yaml \
@@ -21,19 +34,28 @@
     worker.actor.model.model_path=$solver_model_path \
     trainer.experiment_name=${experiment_name} \
     trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
-    data.train_files=${HUGGINGFACENAME}/${experiment_name}@train \
+    data.train_files=${ADMITTED} \
     trainer.total_epochs=100 \
-    trainer.max_steps=20 \
+    trainer.max_steps=$S_STEPS \
     data.format_prompt=./examples/format_prompt/solver.jinja \
     trainer.val_freq=4 \
     worker.actor.micro_batch_size_per_device_for_update=1 \
     worker.actor.micro_batch_size_per_device_for_experience=1 \
+    data.seed=$((1+${RZ_SEED:-0})) worker.rollout.seed=$((1+${RZ_SEED:-0})) $LOAD_ARG ${RZ_SMOKE_ARGS}
+fi
 
 echo "merging model"
-python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor
+[ -f $CKPT/A75_MERGED_step$S_MERGE ] || { python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_$S_MERGE/actor && touch $CKPT/A75_MERGED_step$S_MERGE; }
 
 sleep 10
 
 echo "solver training finished"
+# A7-5 (P16): completion marker, then removal of the checkpoints the pipeline never reads (storage only)
+k=0; [ -f $CKPT/latest_global_step.txt ] && k=$(cat $CKPT/latest_global_step.txt)
+if [ "$k" -ge $S_STEPS ] && [ -f $CKPT/A75_MERGED_step$S_MERGE ]; then
+  touch $CKPT/A75_DONE
+  for g in $CKPT/global_step_*; do [ "$g" = "$CKPT/global_step_$S_MERGE" ] || rm -rf "$g"; done
+fi
 
-bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
+# A7-5 (P10): the repository's benchmark evaluation is not run (Hub datasets, OpenAI re-check, all GPUs); RZ-P2 uses umwp_probe
+# bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
diff -ruN a/verl/trainer/main.py b/verl/trainer/main.py
--- a/verl/trainer/main.py
+++ b/verl/trainer/main.py
@@ -123,7 +123,9 @@
                 "PYTHONUNBUFFERED": "1",
             }
         }
-        ray.init(runtime_env=runtime_env,num_cpus=16)
+        # A7-5 (P11): always a private Ray instance; its object store (in /dev/shm, which counts against the host's memory cgroup)
+        # is capped at 32 GiB instead of Ray's default 30 % of RAM
+        ray.init(address="local", runtime_env=runtime_env, num_cpus=16, include_dashboard=False, object_store_memory=32 * 1024**3)
 
     runner = Runner.remote()
     ray.get(runner.run.remote(ppo_config))
diff -ruN a/vllm_service_init/start.sh b/vllm_service_init/start.sh
--- a/vllm_service_init/start.sh
+++ b/vllm_service_init/start.sh
@@ -1,7 +1,8 @@
 model_path=$1
 run_id=$2
 export VLLM_DISABLE_COMPILE_CACHE=1
-CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &
-CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path &
-CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 5002 --model_path $model_path &
-CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 5003 --model_path $model_path &
\ No newline at end of file
+# A7-5 (P2): RZ_VLLM_RUN tags the 4 servers so that the questioner stage stops exactly these processes
+RZ_VLLM_RUN=$run_id CUDA_VISIBLE_DEVICES=4 python vllm_service_init/start_vllm_server.py --port 5000 --model_path $model_path &
+RZ_VLLM_RUN=$run_id CUDA_VISIBLE_DEVICES=5 python vllm_service_init/start_vllm_server.py --port 5001 --model_path $model_path &
+RZ_VLLM_RUN=$run_id CUDA_VISIBLE_DEVICES=6 python vllm_service_init/start_vllm_server.py --port 5002 --model_path $model_path &
+RZ_VLLM_RUN=$run_id CUDA_VISIBLE_DEVICES=7 python vllm_service_init/start_vllm_server.py --port 5003 --model_path $model_path &
\ No newline at end of file
diff -ruN a/vllm_service_init/start_vllm_server.py b/vllm_service_init/start_vllm_server.py
--- a/vllm_service_init/start_vllm_server.py
+++ b/vllm_service_init/start_vllm_server.py
@@ -44,6 +44,7 @@
     model=args.model_path,
     tokenizer=args.model_path,
     gpu_memory_utilization=args.gpu_mem_util,
+    **({"seed": int(os.environ["RZ_SEED"])} if int(os.getenv("RZ_SEED", "0")) > 0 else {}),  # A7-5 (P18): seed knob; unset (repository default) for seed 0
 )
 
 sample_params = vllm.SamplingParams(
```

## Appendix B: scripts

### `rz_setup.sh`

```bash
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
```

### `rz_env.sh`

```bash
# A7-5 run environment, sourced by rz_run.sh and rz_smoke.sh after RZ_A75_TAG is exported. No algorithmic setting here.
RZ=/run/ti/rzero; TMP=/run/ti/rzero/tmp
export PATH=$RZ/venv/bin:$PATH PYTHONPATH=$RZ/R-Zero
export HF_HOME=$RZ/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HUGGINGFACENAME=offline WANDB_MODE=disabled
# no usage statistics or telemetry from Ray, vLLM or the Hugging Face libraries (fix 17)
export RAY_USAGE_STATS_ENABLED=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
# every temp and cache directory on the local disk /run/ti, never in RAM-backed /dev/shm, which counts against the host's
# memory cgroup (D-memory). Ray's session files go to $RAY_TMPDIR/ray; its object store stays in /dev/shm, capped at
# 32 GiB here and in ray.init (P11).
export TMPDIR=$TMP RAY_TMPDIR=$TMP XDG_CACHE_HOME=$TMP/cache TRITON_CACHE_DIR=$TMP/triton VLLM_CACHE_ROOT=$TMP/vllm \
  TORCHINDUCTOR_CACHE_DIR=$TMP/inductor HF_DATASETS_CACHE=$TMP/hf_datasets CUDA_CACHE_PATH=$TMP/nv TORCH_EXTENSIONS_DIR=$TMP/torch_ext
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=34359738368 VLLM_DISABLE_COMPILE_CACHE=1
unset RAY_ADDRESS CUDA_VISIBLE_DEVICES
mkdir -p $TMP/cache $TMP/triton $TMP/vllm $TMP/inductor $TMP/hf_datasets $TMP/nv $TMP/torch_ext $HF_HOME
```

### `rz_common.sh`

```bash
# A7-5 shared constants and helpers, sourced by rz_setup.sh, rz_wait_and_run.sh, rz_smoke.sh, rz_run.sh, rz_stop.sh and
# rz_umwp.sh. Defines constants and functions only. Callers that log set TAG, STORAGE_PATH and LEDGER first.
: ${RZ:=/run/ti/rzero} ${D:=/home/tione/notebook/rsi_rl/ext/rzero_design} ${X:=/home/tione/notebook/rsi_rl/ext/rzero_runs}
MK=/dev/shm                          # GPU markers $MK/gpu_single_<g>: the claim protocol all our launchers respect
A75_T0=1790233800                    # T0 = 2026-09-24 15:10 CST (A7 registration)
A75_STOP=$((A75_T0 + 71 * 3600))     # 2026-09-27 14:10 CST: all A7-5 compute stops (enforced by the monitor)
A75_TREE=69ae729158dce32ea0e0d59216b2542001affbb11ab61739093020351a1f7ea2   # patched working tree (C-locale file order); any other hash is fatal
L(){ echo "$(date '+%F %T %Z') $(hostname | cut -c1-22) $TAG: $*" | tee -a $STORAGE_PATH/driver.log >> $LEDGER; }
fail(){ L "preflight failed: $*"; exit 3; }
port_open(){ (exec 3<>/dev/tcp/127.0.0.1/$1) 2>/dev/null; }

# ---- our own processes: exactly those whose exec-time environment contains RZ_A75_TAG=<tag> (a driver exports the tag
# after its own exec, so its /proc environ does not carry it)
tagged_pids(){ grep -lzx "RZ_A75_TAG=$1" /proc/[0-9]*/environ 2>/dev/null | cut -d/ -f3; }
# TERM, then KILL, then wait up to 120 s until none is left (a dying CUDA process can stay listed for a while; fix 5b)
kill_tag(){ local sig p w=0
  for sig in TERM KILL; do p=$(tagged_pids $1); [ -z "$p" ] || kill -$sig $p 2>/dev/null; sleep 10; done
  while p=$(tagged_pids $1); [ -n "$p" ] && [ $w -lt 120 ]; do kill -KILL $p 2>/dev/null; sleep 5; w=$((w + 5)); done
  [ -z "$p" ] || L "warning: tagged processes still present 120 s after KILL: $(echo $p | cut -c1-100)"; }
# the same from inside a tagged run: an untagged child does the killing, so it cannot hit itself
kill_own(){ env -u RZ_A75_TAG bash -c "TAG=$TAG STORAGE_PATH=$STORAGE_PATH LEDGER=$LEDGER; source $D/rz_common.sh; kill_tag $TAG"; }

# ---- GPUs (nvidia-smi queries only)
gpu_apps(){ timeout 60 nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+, '; }
gpu_mem(){ timeout 60 nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null | grep -E '^[0-9]+, '; }
gpu_names(){ timeout 60 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | sort -u; }
# GPU use that is not ours (fix 5a): a compute process whose /proc/<pid>/environ lacks RZ_A75_TAG=$TAG (a process of another
# container has no /proc entry here and counts as foreign), or > 1500 MiB in use on a GPU none of our processes is on.
# A process of this container that is exiting (no address space left, so no environment to read, while nvidia-smi still
# lists it) is not counted as foreign, and its GPU is not flagged by the memory rule (review item 1).
# One line per finding; nothing when the GPUs are ours alone.
foreign_gpu(){ local pid uuid idx m ours=" "
  while IFS=', ' read -r pid uuid; do
    if grep -qzx "RZ_A75_TAG=$TAG" /proc/$pid/environ 2>/dev/null; then ours="$ours$uuid "
    elif [ -d /proc/$pid ] && ! grep -q '^VmRSS:' /proc/$pid/status 2>/dev/null; then ours="$ours$uuid "
    else echo "pid $pid ($(tr '\0' ' ' 2>/dev/null < /proc/$pid/cmdline | cut -c1-60)) on $uuid"; fi
  done < <(gpu_apps)
  while IFS=', ' read -r idx uuid m; do
    [ "$m" -gt 1500 ] 2>/dev/null && case "$ours" in *" $uuid "*) ;; *) echo "gpu $idx: $m MiB in use, none of it ours";; esac
  done < <(gpu_mem); }
# a finding counts only if it is still there 30 s later (a process of ours that just ended can stay listed for seconds)
foreign_confirmed(){ local f; f=$(foreign_gpu); [ -n "$f" ] || return 0; sleep 30; f=$(foreign_gpu); [ -z "$f" ] || echo "$f" | head -3 | tr '\n' ';'; }

# ---- memory: the host's 1400 GiB memory cgroup counts /dev/shm (tmpfs); headroom = limit - (rss + shmem), in GiB (D-memory)
headroom_gib(){ local c=/sys/fs/cgroup lim used
  if [ -r $c/memory/memory.limit_in_bytes ]; then lim=$(cat $c/memory/memory.limit_in_bytes)
    used=$(awk '$1=="total_rss"||$1=="total_shmem"{s+=$2} END{printf "%.0f", s}' $c/memory/memory.stat)
  else lim=$(cat $c/memory.max); used=$(awk '$1=="anon"||$1=="shmem"{s+=$2} END{printf "%.0f", s}' $c/memory.stat); fi
  [ "$lim" = max ] && lim=$(awk '/^MemTotal/{printf "%.0f", $2 * 1024}' /proc/meminfo)
  echo $(( (lim - used) / 1073741824 )); }

# ---- our other jobs on this host: queue runners, launchers, chain scripts and /root job scripts (the passive daemons
# hang_watchdog.sh and ckpt_janitor[2].sh excepted; a4_pair_watchdog.sh starts jobs and counts); rz_* scripts excluded
launchers_alive(){ { pgrep -af '(runner[0-9]*|umwp_launch[a-z_]*|a[3-7][a-z]?_[a-z0-9_]+)\.sh'; pgrep -af '^(/usr)?(/bin/)?bash /root/[A-Za-z0-9_.-]+\.sh'; } \
  | grep -v 'rz_' | grep -vE 'bash /root/(hang_watchdog|ckpt_janitor2?)\.sh( |$)' | sort -u; }

# ---- GPU markers (fixes 6 and 8, D-watchdogs)
m_field(){ cut -d' ' -f$2 $MK/gpu_single_$1 2>/dev/null; }
pid_is_rz(){ [ -n "$1" ] && grep -q 'rz_run\.sh\|rz_smoke\.sh' /proc/$1/cmdline 2>/dev/null; }
# a PASS of the smoke test on this host within the last hour (the smoke keeps its markers for the run that follows)
smoke_pass_recent(){ local e; e=$(grep 'smoke PASSED' $X/ledger_smoke.log 2>/dev/null | grep -- " host=$(hostname) " | tail -1 | grep -o ' epoch=[0-9]*' | cut -d= -f2)
  [ -n "$e" ] && [ $(( $(date +%s) - e )) -le 3600 ]; }
# may this run take over marker g? own leftovers (our tag, that driver dead) and the markers of a smoke test that PASSED here
# in the last hour
takeover_ok(){ local t p; t=$(m_field $1 1); p=$(m_field $1 2)
  [ "$t" = "$TAG" ] && ! pid_is_rz "$p" && return 0
  [ "$t" = rzA75_smoke ] && [ "$TAG" != rzA75_smoke ] && ! pid_is_rz "$p" && smoke_pass_recent && return 0
  return 1; }
# marker g is free for us: absent, ours to take over, or stale. Stale (fix 6) = older than 10 min and not naming a live A7-5
# driver, while the host-wide checks run before it found no GPU compute process, < 1500 MiB on every GPU and none of our
# launchers alive. A stale marker is logged and removed.
marker_ok(){ local f=$MK/gpu_single_$1
  [ -e $f ] || return 0
  takeover_ok $1 && return 0
  if [ -z "$(find $f -mmin -10 2>/dev/null)" ] && [ -e $f ] && ! pid_is_rz "$(m_field $1 2)"; then
    L "stale marker $f removed (content '$(head -c 60 $f 2>/dev/null)', age $(( $(date +%s) - $(stat -c %Y $f) )) s)"; rm -f $f; return 0; fi
  return 1; }
release(){ local g; [ "${KEEP_MARKERS:-0}" = 1 ] && return 0
  for g in 0 1 2 3 4 5 6 7; do [ "$(m_field $g 1)" = "$TAG" ] && rm -f $MK/gpu_single_$g; done; rm -f $STORAGE_PATH/driver.pid; }
# markers a PASSED smoke test kept (first field rzA75_smoke) that no live A7-5 driver holds any more (review item 3):
# removed when a waiter ends and by rz_stop.sh <S>, so that no leftover blocks our other launchers
drop_smoke_markers(){ local g n=0
  for g in 0 1 2 3 4 5 6 7; do [ "$(m_field $g 1)" = rzA75_smoke ] && ! pid_is_rz "$(m_field $g 2)" && rm -f "${MK:?}"/gpu_single_$g && n=$((n + 1)); done
  [ $n = 0 ] || L "removed $n GPU markers that a passed smoke test kept and no run took over"; }
# a stop requested by the monitor (stop_code = "<exit code> <reason>") ends the run with that code; any other signal with 143
on_signal(){ local c; if [ -f $STORAGE_PATH/stop_code ]; then c=$(cut -d' ' -f1 $STORAGE_PATH/stop_code)
  L "stopped by the monitor: $(cut -d' ' -f2- $STORAGE_PATH/stop_code)"; exit $c; fi; L "stopped by signal"; exit 143; }
stop_check(){ [ ! -f $STORAGE_PATH/stop_code ] || on_signal; }
# exit handler of rz_run.sh and rz_smoke.sh: no second signal can cut it short; watcher and monitor stopped, our processes
# gone, markers released (unless a PASSED smoke test keeps them)
cleanup(){ trap '' TERM INT HUP; kill $XW $MON 2>/dev/null; kill_own; release; }
# claim all 8 GPUs: each marker is created atomically (noclobber); own leftovers and a PASSED smoke test's kept markers are
# taken over by an atomic rename; released on exit
claim(){ local g; trap release EXIT; trap on_signal TERM INT HUP
  for g in 0 1 2 3 4 5 6 7; do
    ( set -o noclobber; echo "$TAG $$ $(date +%s)" > $MK/gpu_single_$g ) 2>/dev/null && continue
    if takeover_ok $g; then echo "$TAG $$ $(date +%s)" > $MK/.gpu_single_$g.$$ && mv -f $MK/.gpu_single_$g.$$ $MK/gpu_single_$g && continue; fi
    fail "claim lost: $MK/gpu_single_$g was taken meanwhile ('$(head -c 40 $MK/gpu_single_$g 2>/dev/null)')"
  done; }

# ---- is this host free for an A7-5 run? Prints the first reason why not, or nothing. Removes stale markers.
host_check(){ local g idx uuid u n x p
  [ -f $RZ/R-Zero/.a75_patched ] || { echo "environment not built (run rz_setup.sh)"; return; }
  [ "$(cat $RZ/R-Zero/.a75_patched)" = "$(sha256sum < $D/rzero_A7_5.patch | cut -c1-64)" ] || { echo "working copy built from another patch (rerun rz_setup.sh)"; return; }
  [ "$(cat $RZ/R-Zero/.a75_tree_sha256 2>/dev/null)" = "$A75_TREE" ] || { echo "working-tree hash is not the registered $A75_TREE (fix 15)"; return; }
  [ "$(gpu_mem | wc -l)" -eq 8 ] || { echo "not 8 GPUs"; return; }
  x=$(launchers_alive); [ -z "$x" ] || { echo "our own job scripts are alive on this host: $(echo $x | cut -c1-200)"; return; }
  n=$(gpu_apps | wc -l); [ "$n" -eq 0 ] || { echo "$n GPU compute processes present"; return; }
  while IFS=', ' read -r idx uuid u; do [ "$u" -lt 1500 ] || { echo "GPU $idx has $u MiB in use"; return; }; done < <(gpu_mem)
  for g in 0 1 2 3 4 5 6 7; do marker_ok $g || { echo "$MK/gpu_single_$g is held ('$(head -c 40 $MK/gpu_single_$g 2>/dev/null)')"; return; }; done
  for p in 5000 5001 5002 5003; do port_open $p && { echo "port $p is in use"; return; }; done
  [ "$(df --output=avail -BG /run/ti | tail -1 | tr -dc 0-9)" -ge 1500 ] || { echo "/run/ti has < 1.5 TB free"; return; }
  [ "$(df --output=avail -BG $MK | tail -1 | tr -dc 0-9)" -ge 250 ] || { echo "/dev/shm has < 250 GB free"; return; }
  [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 300 ] || { echo "MemAvailable < 300 GB"; return; }
  x=$(headroom_gib); [ "$x" -ge 250 ] 2>/dev/null || { echo "memory-cgroup headroom $x GiB < 250 GiB (D-memory)"; return; }; }
# the host must be completely free and ours alone for the whole run (vLLM memory fractions are fractions of total memory):
# host_check twice, 60 s apart
preflight(){ local r t; for t in 1 2; do r=$(host_check); [ -z "$r" ] || fail "$r"; [ $t = 2 ] || sleep 60; done; }

# ---- 60 s monitor (fixes 5a and 9, D-memory; review items 1 and 4): stop this run on foreign GPU use (exit 6),
# memory-cgroup headroom < 100 GiB (exit 7; Ray's own memory monitor, on by default in ray 2.49.2, kills workers at 95 % of
# the cgroup limit counted without page cache, i.e. at about 70-75 GiB of this headroom) or the compute deadline T0+71h
# (exit 8); a caller may define extra_reason for further rules. The monitor writes stop_code, kills our tagged processes
# and signals the driver, whose on_signal exits with the code. The monitor runs without RZ_A75_TAG, so a kill_own of the
# driver never cuts its waits short; while the driver tears down a failed attempt ($STORAGE_PATH/teardown exists before or
# after the 30 s confirmation) it skips the foreign-use rule, and the driver's own check after the teardown decides.
stop_reason(){ local f h
  [ -f $STORAGE_PATH/teardown ] || { f=$(foreign_confirmed); [ -z "$f" ] || [ -f $STORAGE_PATH/teardown ] || { echo "6 foreign GPU use: $f"; return; }; }
  h=$(headroom_gib); case "$h" in ''|*[!0-9-]*) ;; *) [ "$h" -ge 100 ] || { echo "7 memory-cgroup headroom $h GiB < 100 GiB"; return; };; esac
  [ "$(date +%s)" -lt $A75_STOP ] || { echo "8 compute deadline T0+71h (X5)"; return; }
  [ "$(type -t extra_reason)" != function ] || extra_reason; }
start_monitor(){ DRIVER=$$; ( unset RZ_A75_TAG; while :; do sleep 60; r=$(stop_reason); [ -n "$r" ] || continue
    echo "$r" > $STORAGE_PATH/stop_code; L "monitor: ${r#* }: stopping this run (exit ${r%% *})"; kill_own; kill -TERM $DRIVER 2>/dev/null; exit 0
  done ) & MON=$!; }

# ---- seeds and hosts (fix 4). Ids 0 and 1 are the registered seeds; 2 and 3 replace abandoned ids and need >= 1 and >= 2
# abandoned ids, so at most two seed runs can ever count. A seed runs only on the host that first claimed it.
seed_ok(){ local n; [ ! -f $X/s$1.abandoned ] || { echo "refusing: seed $1 was abandoned ($(cat $X/s$1.abandoned))"; return 1; }
  n=$(ls $X/s[0-9].abandoned 2>/dev/null | wc -l)
  [ "$1" -lt 2 ] || [ "$n" -ge $(( $1 - 1 )) ] || { echo "refusing: seed id $1 replaces an abandoned seed and needs $(( $1 - 1 )) abandoned seed ids ($n found)"; return 1; }
  [ ! -f $X/s$1.host ] || [ "$(cut -d' ' -f1 $X/s$1.host)" = "$(hostname)" ] || {
    echo "refusing: seed $1 belongs to host $(cut -d' ' -f1 $X/s$1.host) ($X/s$1.host); to restart it elsewhere, run 'rz_stop.sh $1 abandon <reason>' and start a new seed id"; return 1; }; }
claim_host(){ ( set -o noclobber; echo "$(hostname) $STORAGE_PATH $(date +%s)" > $X/s$1.host ) 2>/dev/null \
  || [ "$(cut -d' ' -f1 $X/s$1.host)" = "$(hostname)" ] || { L "refusing: seed $1 was claimed by $(cut -d' ' -f1 $X/s$1.host) meanwhile"; exit 2; }; }
# software and hardware stack: patch, working tree, venv package list, GPU model and driver, system python
stack_fp(){ { cat $RZ/R-Zero/.a75_patched $RZ/R-Zero/.a75_tree_sha256; sha256sum < $RZ/venv_freeze.txt; gpu_names; /usr/bin/python3 --version; } 2>&1 | sha256sum | cut -c1-16; }
# may seed <S> skip the smoke test? It has started before, or a smoke test PASSED on the same stack (for seed 0: on this host)
smoke_not_needed(){ local fp; [ -f $X/s$1.host ] && return 0; fp=$(stack_fp)
  if [ "$1" = 0 ]; then grep 'smoke PASSED' $X/ledger_smoke.log 2>/dev/null | grep -- " host=$(hostname) " | grep -q -- " stack=$fp "
  else grep 'smoke PASSED' $X/ledger_smoke.log 2>/dev/null | grep -q -- " stack=$fp "; fi; }

# ---- UMWP dump files (fix 10): cut a partial last line (a kill mid-write); count unique parseable ids
trim_partial(){ /usr/bin/python3 -c '
import os, sys
f = sys.argv[1]
if os.path.exists(f) and os.path.getsize(f):
    b = open(f, "rb").read()
    if not b.endswith(b"\n"):
        k = b.rfind(b"\n") + 1
        open(f, "r+b").truncate(k)
        print("cut a partial last line of %d bytes from %s" % (len(b) - k, f))' "$1"; }
n_ids(){ /usr/bin/python3 -c '
import json, os, sys
ids = set()
if os.path.exists(sys.argv[1]):
    for l in open(sys.argv[1], errors="replace"):
        try:
            ids.add(json.loads(l)["id"])
        except (ValueError, KeyError, TypeError):
            pass
print(len(ids))' "$1"; }

# ---- driver helpers (rz_run.sh)
cur_stage(){ local i t; for i in 1 2 3; do for t in questioner solver; do [ -f $M/${ABBR}_${t}_v$i/A75_DONE ] || { echo ${ABBR}_${t}_v$i; return; }; done; done; }
tracker(){ cat $M/$1/latest_global_step.txt 2>/dev/null || echo 0; }
# fix 12: an admitted set smaller than one rollout batch (512 rows, after the trainer's overlong-prompt filter) leaves the
# repository's drop_last=True dataloader empty, and its assertion fails on every attempt. A finding about the default
# pipeline: logged with n and written to ext/rzero_runs/s<S>/finding_admitted_lt512.json; no retry (exit 9).
small_admitted(){ local off=$1 i f n hit
  hit=$(tail -c +$(( off + 1 )) $STORAGE_PATH/main.log 2>/dev/null | grep -c 'assert len(train_dataloader) >= 1')
  for i in 1 2 3; do f=$GQ/${ABBR}_solver_v${i}_admitted.parquet
    [ -f $f ] && [ ! -f $M/${ABBR}_solver_v$i/A75_DONE ] || continue
    n=$(/usr/bin/python3 -c 'import sys, pyarrow.parquet as p; print(p.read_metadata(sys.argv[1]).num_rows)' $f)
    if [ "$n" -lt 512 ] || [ "$hit" -gt 0 ]; then
      echo "{\"seed\": $S, \"iteration\": $i, \"n_admitted\": $n, \"assertion_in_log\": $hit, \"time\": $(date +%s)}" > $X/s$S/finding_admitted_lt512.json
      L "finding (X8): the admitted set of iteration $i has $n rows; the solver cannot train (drop_last with 512-row batches; assertion lines in this attempt: $hit); no retry"
      return 0; fi; done; return 1; }
# fix 14: resume fallback. The attempt loaded checkpoint k of the stage it was running, saved no newer one, and failed with a
# traceback after 'Load from checkpoint', either through the checkpoint-loading code or for the second attempt in a row at
# the same k: the stage directory is moved aside (its checkpoints deleted, the rest kept) and the stage reruns from its start.
resume_fallback(){ local st=$1 k0=$2 off=$3 seg dst lf=$STORAGE_PATH/loadfail
  [ -n "$st" ] && [ -d $M/$st ] && [ "$k0" -gt 0 ] && [ "$(tracker $st)" = "$k0" ] || { rm -f $lf; return 0; }
  seg=$(tail -c +$(( off + 1 )) $STORAGE_PATH/main.log 2>/dev/null | sed -n '/Load from checkpoint: /,$p')
  echo "$seg" | grep -q '^Traceback\|Traceback (most recent call last)' || { rm -f $lf; return 0; }
  if echo "$seg" | grep -qE 'in _?load_checkpoint$|in load_state_dict$' || [ "$(cat $lf 2>/dev/null)" = "$st:$k0" ]; then
    dst=$M/$st.discarded_$(date +%s); mv $M/$st $dst; rm -rf $dst/global_step_*; rm -f $lf
    L "resume fallback: $st failed after loading global_step_$k0; stage directory moved to $dst (checkpoints deleted); the stage reruns from its start"
  else echo "$st:$k0" > $lf; L "attempt failed after loading $st/global_step_$k0 (a second such failure triggers the resume fallback)"; fi; }
```

### `rz_wait_and_run.sh`

```bash
#!/bin/bash
# A7-5 waiter (addendum section 8, drain rule; decision D-host): polls this host every 60 s and, once it is free of our GPU
# markers, processes and job scripts (host_check in rz_common.sh), runs the smoke test (seed 0; a later seed skips it when
# a smoke test PASSED on the same software/hardware stack) and then rz_run.sh <S>. After a co-tenant yield (exit 6), a
# memory stop (7) or a lost preflight or clock error (3) it waits again and resumes. The projected-finish rule decides
# whether a seed still starts; a failed rz_clock.py call is a clock error, retried every 60 s (review item 2). When the
# waiter ends, the GPU markers that a passed smoke test kept and no run took over are removed (review item 3).
# Log: ext/rzero_runs/ledger_s<S>.log (a "host not free" or "clock error" line when the reason changes, and hourly).
# Start (seed 0 on host B -p <PORT>, seed 1 on host A -p <PORT>):
#   nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_wait_and_run.sh <S> > /dev/null 2>&1 &
# Stop together with its run: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh <S>
S=${1:-}; case "$S" in 0|1|2|3) ;; *) echo "usage: rz_wait_and_run.sh 0|1|2|3"; exit 2;; esac
[ -z "${RZ_A75_TAG:-}" ] || { echo "refusing: inside an A7-5 run environment"; exit 2; }
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_s$S; export STORAGE_PATH=$RZ/s$S; LEDGER=$X/ledger_s$S.log; W=$RZ/wait_s$S.pid
mkdir -p $STORAGE_PATH $X/s$S
source $D/rz_common.sh
r=$(seed_ok $S) || { L "waiter: $r"; echo "$r"; exit 2; }
p=$(cat $W 2>/dev/null); [ -n "$p" ] && grep -q rz_wait_and_run.sh /proc/$p/cmdline 2>/dev/null && { echo "a waiter for seed $S is already running (pid $p)"; exit 2; }
echo $$ > $W; trap 'drop_smoke_markers; rm -f $W' EXIT; trap 'L "waiter stopped by signal"; exit 143' TERM INT HUP
L "waiter start (pid $$; polls every 60 s)"
last=""; beat=0
while :; do
  [ "$(date +%s)" -lt $A75_STOP ] || { L "waiter: T0+71h reached; nothing more is started (X5)"; exit 0; }
  r=$(host_check); key=$(echo "$r" | cut -c1-40 | tr -d '0-9')
  if [ -n "$r" ]; then   # logged when the kind of reason changes, and hourly
    { [ "$key" = "$last" ] && [ $(( $(date +%s) - beat )) -lt 3600 ]; } || { L "waiter: host not free: $(echo $r | cut -c1-300)"; beat=$(date +%s); }
    last=$key; sleep 60; continue; fi
  need=1; smoke_not_needed $S && need=0
  # only full, p1only or none is a projection; anything else means that rz_clock.py failed (e.g. a transient NFS error)
  set -- $(/usr/bin/python3 $D/rz_clock.py mode --seed $S --storage $STORAGE_PATH --smoke $need 2>> $STORAGE_PATH/driver.log)
  case "${1:-}" in full|p1only) ;;
    none) L "waiter: NOT STARTED - the projected-finish rule leaves no time (X1; $*)"; exit 0;;
    *) { [ "$last" = clock ] && [ $(( $(date +%s) - beat )) -lt 3600 ]; } || { L "waiter: clock error: rz_clock.py printed '$(echo "$*" | cut -c1-120)' (its error is in $STORAGE_PATH/driver.log); retrying every 60 s"; beat=$(date +%s); }
       last=clock; sleep 60; continue;;
  esac
  last=""
  if [ $need = 1 ]; then
    L "waiter: host free; smoke test (projected mode $1, $2)"; bash $D/rz_smoke.sh; rc=$?
    case $rc in 0) ;; 3|6|7) L "waiter: smoke test exit $rc; waiting again"; sleep 60; continue;;
      8) L "waiter: smoke test stopped at T0+71h (X5); waiter ends"; exit 8;;
      *) L "waiter: smoke test exit $rc; waiter ends (X2: debugging by hand within the allowed changes)"; exit $rc;; esac
  fi
  L "waiter: starting rz_run.sh $S"; bash $D/rz_run.sh $S; rc=$?
  case $rc in 0) L "waiter: seed $S driver ended (exit 0); waiter ends"; exit 0;;
    3|6|7) L "waiter: driver exit $rc; waiting to resume"; sleep 60;;
    *) L "waiter: driver exit $rc; waiter ends"; exit $rc;; esac
done
```

### `rz_smoke.sh`

```bash
#!/bin/bash
# A7-5 smoke test: 1.5-2.5 h on a fully free host, immediately before the first confirmatory run on that host; started by
# rz_wait_and_run.sh. It runs the patched stage scripts with tiny sizes (smoke hooks RZ_Q_STEPS ... RZ_SMOKE_ARGS, never set
# in a confirmatory run). Nothing here is data. GO = S1, S3 and S4 ok ("smoke PASSED" in ledger_smoke.log); a failed S2
# alone is logged, and the automatic resume fallback of addendum section 7 then covers a failing resume. On PASS the 8 GPU
# markers are kept for the run that follows (hand-over, fix 8). Exit: 0 PASS, 2 refused, 3 preflight, 5 FAILED, 6/7/8
# stopped by the monitor (foreign GPU use / memory headroom / deadline).
[ -z "${RZ_A75_TAG:-}" ] || exit 2
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_smoke; B=/root/models/Qwen3-4B-Base
export STORAGE_PATH=$RZ/smoke; LEDGER=$X/ledger_smoke.log
p=$(cat $STORAGE_PATH/driver.pid 2>/dev/null); [ -n "$p" ] && grep -q rz_smoke.sh /proc/$p/cmdline 2>/dev/null && { echo "a smoke test is already running (pid $p)"; exit 2; }
rm -rf $STORAGE_PATH; mkdir -p $STORAGE_PATH/evaluation $STORAGE_PATH/models $STORAGE_PATH/generated_question $STORAGE_PATH/temp_results $X
source $D/rz_common.sh
kill_tag $TAG; preflight; claim; echo $$ > $STORAGE_PATH/driver.pid
export RZ_A75_TAG=$TAG RZ_SEED=0
source $D/rz_env.sh
cd $RZ/R-Zero
start_monitor; trap cleanup EXIT
LOG=$STORAGE_PATH/smoke.log; M=$STORAGE_PATH/models; Q=rzsmoke_qwen3_4b_Base_questioner_v1; SV=rzsmoke_qwen3_4b_Base_solver_v1
python -c "import pyarrow.parquet as pq; t = pq.read_table('$RZ/data/math12k_test.parquet'); pq.write_table(t.slice(0, 32), '$RZ/data/smoke_val32.parquet')"
V="data.val_files=$RZ/data/smoke_val32.parquet trainer.val_before_train=false"
ok(){ stop_check; if [ -f $M/$1/A75_DONE ]; then L "smoke $2: ok ($(( $(date +%s) - T )) s)"; else L "smoke $2: FAILED (see $LOG)"; exit 5; fi; }
L "smoke start (stack $(stack_fp))"
# S1 questioner, 1 GRPO step: 4 reward servers, caller_penalty reward with BLEU clustering, FSDP + vLLM 0.11 rollout and
#    weight sync on sm_120, checkpoint, merge, targeted server shutdown
T=$(date +%s); RZ_Q_STEPS=1 RZ_Q_MERGE=1 RZ_SMOKE_ARGS="data.rollout_batch_size=32 $V" bash scripts/questioner_train_penalty.sh $B $B $Q >> $LOG 2>&1
ok $Q "S1 questioner step"
# S2 the same stage continued to step 2 from the step-1 checkpoint (repository trainer.load_checkpoint_path), merged at step 2
rm -f $M/$Q/A75_DONE; T=$(date +%s)
RZ_Q_STEPS=2 RZ_Q_MERGE=2 RZ_SMOKE_ARGS="data.rollout_batch_size=32 $V" bash scripts/questioner_train_penalty.sh $B $B $Q >> $LOG 2>&1
stop_check
if grep -q "Load from checkpoint: .*global_step_1" $LOG && [ -f $M/$Q/A75_DONE ]; then L "smoke S2 questioner resume: ok ($(( $(date +%s) - T )) s)"; QS=2
else L "smoke S2 questioner resume: FAILED (see $LOG); the automatic resume fallback of addendum section 7 covers resumes"; QS=1; S2F=1; fi
# S3 solver stage: 8 x 100 generations, 9-sample pseudo-labels, 0.3-0.8 filter, local parquet, 1 solver GRPO step on 8 GPUs,
#    checkpoint at step 1 (save_freq 1 here only; the confirmatory solver keeps the repository's save_freq 5), merge
T=$(date +%s)
RZ_GEN_N=100 RZ_S_STEPS=1 RZ_S_MERGE=1 RZ_SMOKE_ARGS="data.rollout_batch_size=16 worker.actor.global_batch_size=16 trainer.val_freq=-1 trainer.save_freq=1 $V" \
  bash scripts/solver_train.sh $B $M/$Q/global_step_$QS/actor/huggingface $SV >> $LOG 2>&1
ok $SV "S3 solver stage"
L "smoke S3 counts: raw $(cat $STORAGE_PATH/generated_question/${SV}_[0-7].json.raw 2>/dev/null | grep -c '"score"') scored $(python -c "import json; print(len(json.load(open('$STORAGE_PATH/generated_question/${SV}_all_scored.json'))))") admitted $(wc -l < $STORAGE_PATH/generated_question/${SV}_admitted.jsonl)"
# S4 the merged smoke solver loads in vLLM through the frozen UMWP probe (5 unanswerable + their originals); the dump has a
#    per-host name, so smoke tests on two hosts never share a file (review item 10), and is deleted before and after
T=$(date +%s); U=/home/tione/notebook/rsi_rl/dirA4/results/umwp; ST=smoke_rzA75_$(hostname | cut -c1-12); rm -f "${U:?}/${ST:?}.jsonl"
env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/tione/notebook/rsi_rl/hf_home TOKENIZERS_PARALLELISM=false \
  /usr/bin/python3 /home/tione/notebook/rsi_rl/dirA3/code/umwp_probe.py --model $M/$SV/global_step_1/actor/huggingface --tag $ST --n_uns 5 >> $LOG 2>&1
stop_check
n=$(cat "$U/$ST.jsonl" 2>/dev/null | wc -l); rm -f "${U:?}/${ST:?}.jsonl"
[ "$n" -ge 6 ] || { L "smoke S4 probe: FAILED ($n rows)"; exit 5; }
L "smoke S4 probe on the merged solver: ok ($(( $(date +%s) - T )) s)"
# PASS: our processes gone, the 8 markers kept for rz_run.sh (which takes them over within the hour)
rm -rf $M; kill $MON 2>/dev/null; kill_own; KEEP_MARKERS=1
L "smoke PASSED${S2F:+ except S2 (the resume fallback covers resumes)} epoch=$(date +%s) host=$(hostname) stack=$(stack_fp) (smoke checkpoints removed; log kept at $LOG; 8 GPU markers kept for the run that follows)"
```

### `rz_run.sh`

```bash
#!/bin/bash
# A7-5 (c, d, e, f): one R-Zero seed on this host = seed and host checks, projected-finish mode, preflight, atomic claim of
# all 8 GPUs, the patched official scripts/main.sh (3 iterations; every stage resumes) under a 60 s monitor, then the
# admitted-set extraction (d) and the UMWP dumps (e). Normally started by rz_wait_and_run.sh <S>.
# By hand: nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_run.sh <S> > /dev/null 2>&1 &   (resume: the same)
# Stop: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh <S>
# Exit: 0 done (or the RZ-P1-only end), 2 refused, 3 preflight or clock error, 4 import check, 6 foreign GPU use (yield),
# 7 memory headroom, 8 deadline or no time left, 9 admitted set below one batch (finding X8), 10 three failed attempts (X4).
S=${1:-}; case "$S" in 0|1|2|3) ;; *) echo "usage: rz_run.sh 0|1|2|3"; exit 2;; esac
[ -z "${RZ_A75_TAG:-}" ] || { echo "refusing: already inside an A7-5 run environment"; exit 2; }
for v in RZ_Q_STEPS RZ_Q_MERGE RZ_S_STEPS RZ_S_MERGE RZ_GEN_N RZ_SMOKE_ARGS; do [ -z "${!v:-}" ] || { echo "refusing: $v is set (smoke-test hook)"; exit 2; }; done
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_s$S; ABBR=rzero_qwen3_4b_Base_s$S
export STORAGE_PATH=$RZ/s$S; LEDGER=$X/ledger_s$S.log
M=$STORAGE_PATH/models; GQ=$STORAGE_PATH/generated_question
mkdir -p $STORAGE_PATH/evaluation $M $GQ $STORAGE_PATH/temp_results $X/s$S
source $D/rz_common.sh
r=$(seed_ok $S) || { L "$r"; echo "$r"; exit 2; }
p=$(cat $STORAGE_PATH/driver.pid 2>/dev/null)
[ -n "$p" ] && grep -q rz_run.sh /proc/$p/cmdline 2>/dev/null && { echo "a driver for seed $S is already running (pid $p)"; exit 2; }
extract(){ /usr/bin/python3 $D/rz_extract.py --seed $S >> $STORAGE_PATH/driver.log 2>&1; }
clock(){ /usr/bin/python3 $D/rz_clock.py "$@" --seed $S --storage $STORAGE_PATH; }
# projected-finish rule (section 8): full, p1only (stop once the iteration-3 admitted set exists; no RZ-P2 for this seed) or
# none (exit 8). Anything else means that rz_clock.py failed (e.g. a transient NFS error): a clock error, exit 3, which the
# waiter retries (review item 2)
mode_ok(){ case "$MODE" in full|p1only) ;; none) L "not started: the projected-finish rule leaves no time ($*)"; exit 8;;
  *) L "clock error: rz_clock.py printed '$(echo "$*" | cut -c1-120)' (its error is in driver.log); exit 3, retried by the waiter"; exit 3;; esac; }
set -- $(clock mode 2>> $STORAGE_PATH/driver.log); MODE=${1:-}; mode_ok "$@"
kill_tag $TAG                     # leftovers of an interrupted attempt of this seed (our tagged processes only)
rm -f "${STORAGE_PATH:?}"/stop_code "${STORAGE_PATH:?}"/teardown "${STORAGE_PATH:?}"/in_attempt
preflight; claim; claim_host $S; echo $$ > $STORAGE_PATH/driver.pid
set -- $(clock mode --commit 2>> $STORAGE_PATH/driver.log); MODE=${1:-}; mode_ok "$@"
L "driver start: mode $MODE, projection $2"
if [ "$MODE" = p1only ] && [ -f $GQ/${ABBR}_solver_v3_admitted.parquet ]; then extract; L "RZ-P1-only mode and the iteration-3 admitted set exists: nothing to run"; exit 0; fi
export RZ_A75_TAG=$TAG RZ_SEED=$S
source $D/rz_env.sh
cd $RZ/R-Zero
python -c "import verl; assert verl.__version__ == '0.3.1.dev0' and verl.__file__.startswith('/run/ti/rzero/R-Zero/')" || { L "verl import check failed"; exit 4; }
done_list(){ (cd $M && ls -d */A75_DONE 2>/dev/null | sed 's#/A75_DONE##' | tr '\n' ' '); }
# (d) each iteration's audit sample is drawn as soon as its admitted set exists, so the Opus audit can start during the run
start_watcher(){ ( while sleep 600; do /usr/bin/python3 $D/rz_extract.py --seed $S > $STORAGE_PATH/extract.log 2>&1; done ) & XW=$!; }
# monitor rules beyond foreign GPU use, memory and the deadline: record stage completions (rz_clock.py); the hang rule
# (review item 8): while an attempt runs (file in_attempt), nothing changed for 120 min in main.log, the stage directories
# (trackers, checkpoints, markers), the generated-question files or temp_results stops the attempt (our tagged processes
# killed; it counts as a failed attempt, X4); in RZ-P1-only mode, end the run once the iteration-3 admitted set exists
hung(){ [ -f $STORAGE_PATH/in_attempt ] && [ -z "$(find $STORAGE_PATH/in_attempt $STORAGE_PATH/main.log $M $GQ $STORAGE_PATH/temp_results -maxdepth 2 -mmin -120 -print -quit 2>/dev/null)" ]; }
extra_reason(){ clock observe > /dev/null 2>&1
  if hung; then rm -f "${STORAGE_PATH:?}"/in_attempt; L "hang rule: nothing changed for 120 min (main.log, stage directories, generated questions): this attempt is stopped and counts as failed"; kill_own; fi
  [ "$MODE" = p1only ] && [ -f $GQ/${ABBR}_solver_v3_admitted.parquet ] && { extract; echo "0 RZ-P1-only mode (section 8): the iteration-3 admitted set exists"; }; }
# exit handler: watcher and monitor stopped, our processes gone, markers released, then the extraction once more (CPU only;
# review item 5), so that an admitted set created shortly before any stop gets its audit sample and summary at once
start_watcher; start_monitor; trap 'cleanup; extract' EXIT
for attempt in 1 2 3; do            # up to 3 attempts per driver start; every attempt resumes where the last one stopped
  stop_check; clock attempt > /dev/null 2>&1
  st=$(cur_stage); k0=$(tracker "$st"); off=$(stat -c %s $STORAGE_PATH/main.log 2>/dev/null || echo 0)
  L "main.sh $ABBR attempt $attempt (stages done: $(done_list); next: $st at checkpoint $k0)"
  touch $STORAGE_PATH/in_attempt
  bash scripts/main.sh /root/models/Qwen3-4B-Base $ABBR >> $STORAGE_PATH/main.log 2>&1; rc=$?; rm -f "${STORAGE_PATH:?}"/in_attempt
  stop_check; clock observe > /dev/null 2>&1
  L "main.sh exit $rc; stages done: $(done_list)"
  [ -f $M/${ABBR}_solver_v3/A75_DONE ] && break
  if small_admitted $off; then extract; exit 9; fi
  # teardown of the failed attempt: our leftovers killed (kill_own waits until they are gone) while the monitor skips its
  # foreign-use rule; this driver's own check after the teardown decides (review item 1)
  touch $STORAGE_PATH/teardown; kill_own
  stop_check
  f=$(foreign_confirmed); rm -f "${STORAGE_PATH:?}"/teardown; [ -z "$f" ] || { L "foreign GPU use after the failed attempt ($f): yielding the host"; exit 6; }
  resume_fallback "$st" "$k0" $off
  start_watcher; [ $attempt = 3 ] || sleep 120
done
kill $XW 2>/dev/null; extract
[ -f $M/${ABBR}_solver_v3/A75_DONE ] || { L "X4: 3 attempts ended without the final solver (stages done: $(done_list))"; exit 10; }
bash $D/rz_umwp.sh $S 0 >> $STORAGE_PATH/driver.log 2>&1; stop_check; clock observe > /dev/null 2>&1
L "UMWP dumps: $(tail -4 $STORAGE_PATH/driver.log | tr '\n' ' ' | cut -c1-300)"
L "driver end"
```

### `rz_stop.sh`

```bash
#!/bin/bash
# A7-5 (f): stop the A7-5 run of seed <S> on this host at once (co-tenant priority, or by hand): its waiter
# (rz_wait_and_run.sh), its driver and every process carrying RZ_A75_TAG=rzA75_s<S>; removes its GPU markers and those a
# passed smoke test kept that no live run holds (review item 3); keeps every file.
# Resume later: nohup bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_wait_and_run.sh <S> > /dev/null 2>&1 &
# 'abandon <reason>' also marks seed <S> abandoned (addendum section 7; may be run on any host): it is reported, never
# resumed and never counted, and a restart elsewhere uses the next free replacement seed id (2, then 3).
# (The monitor's own automatic yield, exit 6, keeps the waiter, which resumes when the host is free again.)
# Usage: bash /home/tione/notebook/rsi_rl/ext/rzero_design/rz_stop.sh <0|1|2|3|smoke> [abandon "<reason>"]
S=${1:-}; case "$S" in 0|1|2|3|smoke) ;; *) echo "usage: rz_stop.sh 0|1|2|3|smoke [abandon <reason>]"; exit 2;; esac
[ -z "${RZ_A75_TAG:-}" ] || { echo "run this from a normal shell"; exit 2; }
RZ=/run/ti/rzero; D=/home/tione/notebook/rsi_rl/ext/rzero_design; X=/home/tione/notebook/rsi_rl/ext/rzero_runs
TAG=rzA75_s$S; STORAGE_PATH=$RZ/s$S; LEDGER=$X/ledger_s$S.log
[ "$S" = smoke ] && { TAG=rzA75_smoke; STORAGE_PATH=$RZ/smoke; LEDGER=$X/ledger_smoke.log; }
mkdir -p $STORAGE_PATH; source $D/rz_common.sh
if [ "${2:-}" = abandon ]; then
  [ "$S" != smoke ] && [ -n "${3:-}" ] || { echo "usage: rz_stop.sh <0|1|2|3> abandon \"<reason>\""; exit 2; }
  ( set -o noclobber; echo "$(date +%s) $(hostname) $3" > $X/s$S.abandoned ) 2>/dev/null || { echo "seed $S is already abandoned: $(cat $X/s$S.abandoned)"; }
  L "seed $S ABANDONED by rz_stop.sh on $(hostname): $3"
fi
# a smoke test started by this seed's waiter is stopped with it
wp=$(cat $RZ/wait_s$S.pid 2>/dev/null); sp=$(cat $RZ/smoke/driver.pid 2>/dev/null)
[ "$S" != smoke ] && [ -n "$wp" ] && [ -n "$sp" ] && [ "$(ps -o ppid= -p $sp 2>/dev/null | tr -d ' ')" = "$wp" ] && bash $D/rz_stop.sh smoke
for f in $RZ/wait_s$S.pid $STORAGE_PATH/driver.pid; do
  p=$(cat $f 2>/dev/null); [ -n "$p" ] && grep -q 'rz_wait_and_run\.sh\|rz_run\.sh\|rz_smoke\.sh' /proc/$p/cmdline 2>/dev/null && kill -TERM $p
done
kill_tag $TAG
for g in 0 1 2 3 4 5 6 7; do [ "$(m_field $g 1)" = "$TAG" ] && rm -f $MK/gpu_single_$g; done
drop_smoke_markers
L "stopped by rz_stop.sh"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -8
```

### `rz_umwp.sh`

```bash
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
```

### `rz_clock.py`

```python
"""A7-5 clock (addendum section 8): stage times and the projected-finish rule. CPU only; writes only
ext/rzero_runs/s<S>/events.tsv and ext/rzero_runs/s<S>/mode.
T0 = 2026-09-24 15:10 CST (A7 registration). All A7-5 compute stops at T0 + 71 h; RZ-P2 is read at T0 + 72 h; audits of
admitted sets that exist by T0 + 71 h may finish by T0 + 84 h.
Stages in order: q1 a1 s1 q2 a2 s2 q3 a3 s3 u (q = questioner stage, a = generation + pseudo-labels + filter, ending with the
admitted set, s = solver training and merge, u = the final solver's UMWP dumps). A stage's completion time is the mtime of
its marker (A75_DONE, the admitted parquet, A75_UMWP_DONE). Upper estimates before any measurement (h): q 3.5, a 1.1, s 3.7,
u 1.0; smoke test 2.5. A stage is measured when it started and ended inside one driver attempt (the event before it is its
predecessor stage; for q1, the seed's first attempt); once a stage type is measured in any seed, its estimate becomes
1.1 x its longest measurement.
Rule: mode "full" if now (+ the smoke test) + the estimates of all remaining stages end by T0 + 71 h; else "p1only" (the run
stops once the iteration-3 admitted set exists, and RZ-P2 is not run for this seed) if the remaining stages up to a3 end by
T0 + 71 h; else "none" (the seed is not started). A committed mode is never upgraded.
  rz_clock.py attempt --seed S --storage DIR     record a driver attempt start and the stages completed so far
  rz_clock.py observe --seed S --storage DIR     record newly completed stages
  rz_clock.py mode --seed S --storage DIR [--smoke 0|1] [--commit]   print "<mode> <details without spaces>" """
import argparse, glob, os, time

R = "/home/tione/notebook/rsi_rl"
T0 = 1790233800; STOP = T0 + 71 * 3600
EST = {"q": 3.5, "a": 1.1, "s": 3.7, "u": 1.0}; SMOKE_H = 2.5
PIPE = ["q1", "a1", "s1", "q2", "a2", "s2", "q3", "a3", "s3", "u"]
RANK = {"none": 0, "p1only": 1, "full": 2}

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["attempt", "observe", "mode"])
ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2, 3]); ap.add_argument("--storage", required=True)
ap.add_argument("--smoke", type=int, default=0, choices=[0, 1]); ap.add_argument("--commit", action="store_true")
ap.add_argument("--now", type=float, default=None)  # for the design tests only
a = ap.parse_args()
now = time.time() if a.now is None else a.now
X = f"{R}/ext/rzero_runs"; E = f"{X}/s{a.seed}"; os.makedirs(E, exist_ok=True)
abbr = f"rzero_qwen3_4b_Base_s{a.seed}"


def marker(st):
    if st == "u":
        return f"{a.storage}/A75_UMWP_DONE"
    i = st[1]
    return {"q": f"{a.storage}/models/{abbr}_questioner_v{i}/A75_DONE",
            "a": f"{a.storage}/generated_question/{abbr}_solver_v{i}_admitted.parquet",
            "s": f"{a.storage}/models/{abbr}_solver_v{i}/A75_DONE"}[st[0]]


def read_events(path):
    ev = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 2 and (p[1] in PIPE or p[1] == "attempt"):
                ev.append((float(p[0]), p[1]))
    return ev


def observe(attempt):
    path = f"{E}/events.tsv"
    seen = {n for _, n in read_events(path)}
    new = [(os.path.getmtime(marker(st)), st) for st in PIPE if st not in seen and os.path.exists(marker(st))]
    if attempt:
        new.append((now, "attempt"))
    if new:
        with open(path, "a") as fh:
            for t, n in new:
                fh.write(f"{t:.0f}\t{n}\n")


def measured():
    m = {k: [] for k in EST}
    for f in sorted(glob.glob(f"{X}/s[0-9]/events.tsv")):
        ev = sorted(read_events(f))
        for j, (t, n) in enumerate(ev):
            if n == "attempt" or j == 0:
                continue
            k = PIPE.index(n); pt, pn = ev[j - 1]
            if k == 0:
                ok = pn == "attempt" and not any(x == "attempt" for _, x in ev[:j - 1])
            else:
                ok = pn == PIPE[k - 1]
            if ok and t >= pt:
                m[n[0]].append(t - pt)
    return m


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M", time.localtime(t))


if a.cmd in ("attempt", "observe"):
    observe(a.cmd == "attempt")
else:
    observe(False)
    m = measured()
    est = {k: (1.1 * max(v) / 3600 if v else EST[k]) for k, v in m.items()}
    rem = [st for st in PIPE if not os.path.exists(marker(st))]
    base = now + (SMOKE_H * 3600 if a.smoke else 0)
    fin = base + sum(est[st[0]] * 3600 for st in rem)
    a3 = base + sum(est[st[0]] * 3600 for st in rem if PIPE.index(st) <= PIPE.index("a3"))
    md = "full" if fin <= STOP else ("p1only" if a3 <= STOP else "none")
    mf = f"{E}/mode"
    old = open(mf).read().split()[0] if os.path.exists(mf) else None
    if old in RANK and RANK[old] < RANK[md]:
        md = old
    detail = (f"now={iso(now)},final={iso(fin)},it3_admitted={iso(a3)},stop={iso(STOP)},smoke={a.smoke},"
              f"remaining={'+'.join(rem) or '-'},est_h=" + "/".join(f"{k}:{est[k]:.2f}" for k in EST) +
              ",measured_n=" + "/".join(f"{k}:{len(m[k])}" for k in EST) + (f",committed={old}" if old else ""))
    if a.commit:
        with open(mf + ".tmp", "w") as fh:
            fh.write(f"{md} {detail}\n")
        os.replace(mf + ".tmp", mf)
    print(md, detail)
```

### `rz_extract.py`

```python
"""A7-5 (d): per iteration, the admitted (filtered) questions -> a shared copy, a pool summary, and the 200-item audit sample
for the frozen Opus audit workflow.
The admitted set of iteration i is the file upload.py wrote for <abbr>_solver_v<i>: questions of questioner Q_i that parsed,
were pseudo-labelled by solver S_(i-1) (9 samples, majority) and passed 0.3 <= score <= 0.8. It is the dataset passed to the
solver trainer of iteration i, which additionally drops prompts over 2,048 tokens (filter_overlong_prompts) and the last
incomplete batch of 512 (drop_last); the summary counts the rows over 2,048 tokens with the trainer's tokenizer, messages and
chat template. Audit sample: random.Random(20260924 + i).sample(range(n), min(200, n)) over its n rows (a repeated question
counts by multiplicity, as in training). It is written as the pseudo-run dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/admitted.jsonl
(kind "proposed", the format the frozen workflow reads); judges see only id and problem text. Idempotent: an existing sample
is never redrawn. Runs only on the host that owns the seed (ext/rzero_runs/s<S>.host, fix 4). CPU only. Concurrent calls (the
watcher, the driver and its exit handler, the monitor, by hand) are serialized by a lock on the seed's local storage (review
item 9). May be run by hand at any time, also after T0 + 71 h for the admitted sets that existed by then (review item 5).
    python3 rz_extract.py --seed S [--iters 1 2 3]"""
import argparse, collections, fcntl, hashlib, json, os, random, re, shutil, socket, sys, time
import pyarrow.parquet as ppq

R = "/home/tione/notebook/rsi_rl"; RZ = "/run/ti/rzero"
TOK = "/root/models/Qwen3-4B-Base"  # the tokenizer of every solver checkpoint (merged checkpoints keep it)
SYS = r"Please reason step by step, and put your final answer within \boxed{}."  # solver system message, verl/utils/dataset.py
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2, 3])
ap.add_argument("--iters", type=int, nargs="+", default=[1, 2, 3])
a = ap.parse_args()
abbr = f"rzero_qwen3_4b_Base_s{a.seed}"; gq = f"{RZ}/s{a.seed}/generated_question"
hf = f"{R}/ext/rzero_runs/s{a.seed}.host"
if not os.path.exists(hf):
    sys.exit(f"s{a.seed}: this seed has not started on any host ({hf} missing)")
owner = open(hf).read().split()[0]
if owner != socket.gethostname():
    sys.exit(f"refusing: seed {a.seed} belongs to host {owner}; this is {socket.gethostname()}")
X = f"{R}/ext/rzero_runs/s{a.seed}"; os.makedirs(X, exist_ok=True)
# review item 9: one extraction at a time per seed (a lock on the seed's local storage); a call that cannot get it within
# 15 min gives up, and the next call extracts
os.makedirs(f"{RZ}/s{a.seed}", exist_ok=True)
_lock = open(f"{RZ}/s{a.seed}/extract.lock", "a"); _t = time.time()
while True:
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except OSError:
        if time.time() - _t > 900:
            sys.exit(f"s{a.seed}: another extraction has held {RZ}/s{a.seed}/extract.lock for 15 min; not run")
        time.sleep(2)


def is_numeric(label):
    t = re.sub(r"\s|\$|\\[,!;]", "", str(label)).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", t)
    return bool(re.fullmatch(r"-?\d+(\.\d+)?|-?\d+/\d+|-?\\frac\{\d+\}\{\d+\}", t))


_tok = []


def n_overlong(rows):
    """rows the trainer's filter_overlong_prompts drops (same messages and length test as verl/utils/dataset.py)"""
    try:
        if not _tok:
            from transformers import AutoTokenizer
            _tok.append(AutoTokenizer.from_pretrained(TOK))
        t = _tok[0]
        if t.chat_template:
            return sum(len(t.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": r["problem"]}],
                                                 add_generation_prompt=True)) > 2048 for r in rows)
        return sum(len("system: " + SYS + "\n" + "user: " + r["problem"]) > 2048 for r in rows)
    except Exception as e:  # descriptive only
        print("overlong count unavailable:", repr(e)[:200])
        return None


for it in a.iters:
    name = f"{abbr}_solver_v{it}"
    pq, js, pool_f = f"{gq}/{name}_admitted.parquet", f"{gq}/{name}_admitted.jsonl", f"{gq}/{name}_all_scored.json"
    if not os.path.exists(pq):
        print(f"s{a.seed} it{it}: no admitted set yet"); continue
    rows = [json.loads(l) for l in open(js)]
    assert ppq.read_metadata(pq).num_rows == len(rows), "admitted jsonl and parquet row counts differ"
    run = f"rzero_qwen3_4b_Base_s{a.seed}_it{it}"; d = f"{R}/dirB/runs/{run}"
    if os.path.exists(f"{d}/admitted.jsonl"):
        print(f"s{a.seed} it{it}: audit sample exists ({d}), not redrawn")
    else:
        idx = random.Random(20260924 + it).sample(range(len(rows)), min(200, len(rows)))
        os.makedirs(f"{d}/audit_parts", exist_ok=True)
        with open(f"{d}/admitted.jsonl.tmp", "w") as fh:
            for rank, j in enumerate(idx):
                r = rows[j]
                fh.write(json.dumps({"id": f"rz{a.seed}i{it}r{j}", "problem": r["problem"], "prompt": r["problem"], "kind": "proposed",
                                     "rz_row": j, "rz_rank": rank, "rz_label": r["answer"], "rz_score": r["score"]}, ensure_ascii=False) + "\n")
        os.replace(f"{d}/admitted.jsonl.tmp", f"{d}/admitted.jsonl")
        print(f"s{a.seed} it{it}: audit sample of {len(idx)} / {len(rows)} admitted written to {d}")
    shutil.copy(js, f"{X}/{name}_admitted.jsonl")
    raw, shards = [], 0
    for k in range(8):
        f = f"{gq}/{name}_{k}.json.raw"
        if os.path.exists(f):
            raw += json.load(open(f)); shards += 1
    pool = json.load(open(pool_f)) if os.path.exists(pool_f) else []
    sha = hashlib.sha256(open(js, "rb").read()).hexdigest()
    sf = f"{X}/{name}_summary.json"
    old = json.load(open(sf)) if os.path.exists(sf) else {}
    over = old.get("n_admitted_over_2048_tokens") if old.get("sha256_admitted_jsonl") == sha else None
    if over is None:
        over = n_overlong(rows)
    summ = {"seed": a.seed, "iteration": it, "sample_seed": 20260924 + it, "audit_run": run, "audit_sample_n": min(200, len(rows)),
            "n_raw_shards": shards, "n_generated": len(raw), "n_parsed": sum(r.get("score") == 0 for r in raw),
            "n_scored": len(pool), "n_admitted": len(rows), "n_admitted_unique": len({r["problem"] for r in rows}),
            "n_admitted_over_2048_tokens": over,
            "n_solver_batches_of_512": (len(rows) - over) // 512 if over is not None else None,
            "admitted_share_of_scored": len(rows) / len(pool) if pool else None,
            "share_numeric_label_admitted": sum(is_numeric(r["answer"]) for r in rows) / len(rows) if rows else None,
            "pool_score_hist": dict(sorted(collections.Counter(round(r["score"], 3) for r in pool).items())),
            "admitted_parquet_mtime": os.path.getmtime(pq), "extract_host": socket.gethostname(),
            "sha256_admitted_jsonl": sha}
    json.dump(summ, open(sf + ".tmp", "w"), indent=1); os.replace(sf + ".tmp", sf)
    print(json.dumps({k: summ[k] for k in ("iteration", "n_generated", "n_parsed", "n_scored", "n_admitted", "n_admitted_unique",
                                           "n_admitted_over_2048_tokens")}))
```

### `rz_read.py`

```python
"""A7-5 reads. Prints one JSON object; safe at any time (an incomplete input is reported as missing and never read).
Seeds: ids 0-3 (2 and 3 only replace abandoned ids, section 4.1); an abandoned seed is reported and never counts; at most two
seeds can count. Clock (section 8): T0 = 2026-09-24 15:10 CST.
RZ-P1, per seed: ill(it3) >= ill(it1) + 10 points, ill = (UNSOLVABLE + AMBIGUOUS) / verdicts of the audit of that
       iteration's sample (dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/audit_admitted.json with a verdict for every sampled
       item). An iteration's audit counts only if its admitted set existed by T0 + 71 h and the audit was assembled by
       T0 + 84 h. The item HOLDS if it holds in >= 1 seed run (a seed counts once both its it1 and it3 audits count).
RZ-P2, per seed: CA(plain_rzero_s<S>_it3) > CA(plain_qwen3_4b_base); CA and the 1,200-id completeness rule as in
       dirA3/code/umwp_read.py; the final dump must be complete by T0 + 72 h, and a seed run in RZ-P1-only mode
       (projected-finish rule) has no RZ-P2. HOLDS if it holds in every seed run (a seed counts once its dump counts).
Secondary and descriptive reads are printed next to them and never change the two verdicts.   python3 rz_read.py"""
import json, os, re
R = "/home/tione/notebook/rsi_rl"; U = f"{R}/dirA4/results/umwp"; X = f"{R}/ext/rzero_runs"; ILL = ("UNSOLVABLE", "AMBIGUOUS")
T0 = 1790233800; ADMITTED_BY = T0 + 71 * 3600; P2_BY = T0 + 72 * 3600; AUDIT_BY = T0 + 84 * 3600


def is_numeric(label):  # same rule as rz_extract.py
    t = re.sub(r"\s|\$|\\[,!;]", "", str(label)).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", t)
    return bool(re.fullmatch(r"-?\d+(\.\d+)?|-?\d+/\d+|-?\\frac\{\d+\}\{\d+\}", t))


def jload(f):
    return json.load(open(f)) if os.path.exists(f) else None


def audit(S, it):
    d = f"{R}/dirB/runs/rzero_qwen3_4b_Base_s{S}_it{it}"
    if not os.path.exists(f"{d}/audit_admitted.json"):
        return None
    summ = jload(f"{X}/s{S}/rzero_qwen3_4b_Base_s{S}_solver_v{it}_summary.json") or {}
    t_adm = summ.get("admitted_parquet_mtime")
    if t_adm is None or t_adm > ADMITTED_BY:
        return {"complete": False, "reason": "admitted set missing from the summary or created after T0+71h", "admitted_mtime": t_adm}
    if os.path.getmtime(f"{d}/audit_admitted.json") > AUDIT_BY:
        return {"complete": False, "reason": "audit assembled after T0+84h"}
    rows = {}
    for l in open(f"{d}/admitted.jsonl"):
        r = json.loads(l); rows[r["id"]] = r
    v = [x for x in json.load(open(f"{d}/audit_admitted.json"))["verdicts"] if x["id"] in rows]
    if len(v) < len(rows):
        return {"complete": False, "n": len(v), "expected": len(rows)}
    n = len(v); c = {k: sum(x["verdict"] == k for x in v) for k in ("SOLVABLE",) + ILL}
    num = [x for x in v if is_numeric(rows[x["id"]].get("rz_label", ""))]
    return {"complete": True, "n": n, "counts": c, "ill": (c["UNSOLVABLE"] + c["AMBIGUOUS"]) / n, "uns_only": c["UNSOLVABLE"] / n,
            "n_numeric_label": len(num), "ill_numeric_label": sum(x["verdict"] in ILL for x in num) / len(num) if num else None}


def umwp(tag, by=None):
    f = f"{U}/{tag}.jsonl"
    if not os.path.exists(f):
        return None
    rows = {}
    for l in open(f, errors="replace"):
        try:
            r = json.loads(l); rows[r["id"]] = r
        except (ValueError, KeyError, TypeError):
            pass
    if len(rows) < 1200:
        return {"complete": False, "unique_parseable_ids": len(rows)}
    if by is not None and os.path.getmtime(f) > by:
        return {"complete": False, "reason": "dump completed after T0+72h"}
    p = list(rows.values()); u = [r for r in p if not r["answerable"]]; s = [r for r in p if r["answerable"]]
    return {"complete": True, "CA": sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u) / len(u),
            "ACC_A": 100 * sum(r["correct"] for r in s) / len(s),
            "ABS_U": 100 * sum(r["abstain"] for r in u) / len(u), "ABS_A": 100 * sum(r["abstain"] for r in s) / len(s)}


ok = lambda m: bool(m) and m.get("complete")
out = {"per_seed": {}, "abandoned": {}, "not_started": []}
base = umwp("plain_qwen3_4b_base")
p1, p2 = {}, {}
for S in (0, 1, 2, 3):
    host = open(f"{X}/s{S}.host").read().split() if os.path.exists(f"{X}/s{S}.host") else None
    if host is None:
        out["not_started"].append(S); continue
    A = {it: audit(S, it) for it in (1, 2, 3)}
    mode = open(f"{X}/s{S}/mode").read().split()[0] if os.path.exists(f"{X}/s{S}/mode") else None
    e = {"host": host[0], "mode": mode, "audit": A, "pool_summaries": {},
         "finding_admitted_lt512": jload(f"{X}/s{S}/finding_admitted_lt512.json")}
    for it in (1, 2, 3):
        f = f"{X}/s{S}/rzero_qwen3_4b_Base_s{S}_solver_v{it}_summary.json"
        if os.path.exists(f):
            e["pool_summaries"][it] = json.load(open(f))
    if os.path.exists(f"{X}/s{S}.abandoned"):
        e["abandoned"] = open(f"{X}/s{S}.abandoned").read().strip()
        out["abandoned"][S] = e; continue
    if ok(A[1]) and ok(A[3]):
        dp = 100 * (A[3]["ill"] - A[1]["ill"])
        e["RZ-P1"] = {"ill_it1": A[1]["ill"], "ill_it3": A[3]["ill"], "diff_points": round(dp, 2), "holds": dp >= 10 - 1e-9}
        p1[S] = e["RZ-P1"]["holds"]
        e["secondary"] = {"unsolvable_only_diff_points": round(100 * (A[3]["uns_only"] - A[1]["uns_only"]), 2),
                          "numeric_label_subset_diff_points": round(100 * (A[3]["ill_numeric_label"] - A[1]["ill_numeric_label"]), 2)
                          if A[1]["ill_numeric_label"] is not None and A[3]["ill_numeric_label"] is not None else None}
    m = umwp(f"plain_rzero_s{S}_it3", by=P2_BY)
    if mode == "p1only":
        e["RZ-P2"] = {"not_run": "RZ-P1-only mode under the projected-finish rule (section 8)"}
    elif ok(m) and ok(base):
        e["RZ-P2"] = {"CA_base": base["CA"], "CA_final": m["CA"], "holds": m["CA"] > base["CA"],
                      "ACC_A_base": base["ACC_A"], "ACC_A_final": m["ACC_A"]}
        p2[S] = e["RZ-P2"]["holds"]
    e["descriptive_abstention_prompt"] = {"final": umwp(f"abs_rzero_s{S}_it3"), "base": umwp("abs_qwen3_4b_base")}
    e["exploratory_chat_template"] = {"final": umwp(f"plainchat_rzero_s{S}_it3"), "base": umwp("plainchat_qwen3_4b_base")}
    out["per_seed"][S] = e
out["counted_seed_ids"] = sorted(out["per_seed"])
if len(out["per_seed"]) > 2:
    out["ERROR"] = "more than two non-abandoned seed ids: the seed rule of section 4.1 was broken; no verdict"
    p1, p2 = {}, {}
out["RZ-P1"] = {"verdict": ("HOLDS" if any(p1.values()) else "FAILS") if p1 else "NOT RUN", "seeds_run": sorted(p1)}
out["RZ-P2"] = {"verdict": ("HOLDS" if all(p2.values()) else "FAILS") if p2 else "NOT RUN", "seeds_run": sorted(p2)}
print(json.dumps(out, indent=1))
```

## Appendix C: audit workflow

`diff a3-audit200-qwen-s1-r4-wf_93af8c2d-710.js rz_audit200_wf.js` (the only differences):

```diff
2,3c2,3
<   name: 'a3-audit200-qwen-s1-r4',
<   description: 'Opus audit of the 200-item sample of admitted proposer questions for A3 round qwen3_4b own s1 r4, parts pushed to the remote run dir',
---
>   name: 'rz-audit200',
>   description: 'A7-5: frozen Opus audit (A3 200-item workflow, prompt unchanged) of the 200-item samples of R-Zero admitted questions, parts pushed to the remote run dir',
6,7c6,26
< const SSH = "ssh -o ConnectTimeout=30 -o BatchMode=yes -p <PORT> root@<SERVER>"
< const RUNS = [["qwen3_4b_Base_own_a3_s1_r4", "Qwen3-4B-Base"]]
---
> // A7-5: the only edits to the frozen A3 workflow are meta, the SSH port and the RUNS list (both from args), the job list and
> // the judge label; the prompt function is byte-identical. The workflow args are the ONLY run-time input (fixes 2 and 3):
> //   {runs: [[S, i], ...], port: P}   S = seed id 0-3, i = iteration 1-3, one entry per audit sample that exists
> //   (dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/admitted.jsonl, written by rz_extract.py); an optional third element n
> //   (1-200, default 200) is the sample size when an admitted set had fewer than 200 rows; port = the SSH port of any live
> //   host of ours at <SERVER> that mounts the shared volume (default <HOST>; the data are on NFS, so the fetched text is
> //   byte-identical).
> const A = (args && typeof args === 'object') ? args : {}
> const PORT = A.port === undefined ? <HOST> : A.port
> if (!Number.isInteger(PORT) || PORT < 1 || PORT > 65535) throw new Error('args.port must be an integer port, got ' + JSON.stringify(A.port))
> const SSH = "ssh -o ConnectTimeout=30 -o BatchMode=yes -p " + PORT + " root@<SERVER>"
> if (!Array.isArray(A.runs) || A.runs.length < 1 || A.runs.length > 12) throw new Error('args.runs must list 1-12 [seed, iteration(, n)] entries')
> const SEEN = {}
> const RUNS = A.runs.map(r => {
>   const n = Array.isArray(r) && r.length === 3 ? r[2] : 200
>   if (!Array.isArray(r) || r.length < 2 || r.length > 3 || ![0, 1, 2, 3].includes(r[0]) || ![1, 2, 3].includes(r[1]) || !Number.isInteger(n) || n < 1 || n > 200) throw new Error('bad args.runs entry ' + JSON.stringify(r))
>   const run = "rzero_qwen3_4b_Base_s" + r[0] + "_it" + r[1]
>   if (SEEN[run]) throw new Error('duplicate args.runs entry ' + run)
>   SEEN[run] = 1
>   return [run, "Qwen3-4B-Base", n]
> })
9c28
< for (const [run, model] of RUNS) for (let a = 0; a < 200; a += 25) jobs.push({ run, a, b: a + 25, model })
---
> for (const [run, model, n] of RUNS) for (let a = 0; a < n; a += 25) jobs.push({ run, a, b: Math.min(a + 25, n), model })
46c65
< const results = await pipeline(jobs, j => agent(prompt(j), { label: "judge:" + j.run.replace("_a3_", ":").replace("_Base", "") + ":" + j.a, phase: 'Judge', schema: SCHEMA, model: 'opus' }))
---
> const results = await pipeline(jobs, j => agent(prompt(j), { label: "judge:" + j.run.replace("rzero_qwen3_4b_Base_", "rz:") + ":" + j.a, phase: 'Judge', schema: SCHEMA, model: 'opus' }))
```

The frozen judge prompt (identical in both files):

```javascript
function prompt(j) {
  const remote = "/home/tione/notebook/rsi_rl/dirB/runs/" + j.run
  const part = remote + "/audit_parts/" + j.a + "-" + j.b + ".json"
  const fname = j.run + "_" + j.a + "-" + j.b + ".json"
  const local = "<LOCAL_HOME>\\AppData\\Local\\Temp\\claude\\<PROJECT>\\43f8b914-5d13-4ce7-83fd-de9b7ae44f11\\scratchpad\\audit_parts\\" + fname
  const localBash = "<LOCAL_HOME>/AppData/Local/Temp/claude/<PROJECT>/43f8b914-5d13-4ce7-83fd-de9b7ae44f11/scratchpad/audit_parts/" + fname
  const m = j.b - j.a
  return "You are a careful mathematician auditing self-generated training questions.\n" +
"Step 1: fetch questions " + j.a + ".." + (j.b - 1) + " with this exact command (Bash tool, one call):\n" +
SSH + " 'cd " + remote + "; python3 -c \"\n" +
"import json, random\n" +
"A=[json.loads(l) for l in open(\\\"admitted.jsonl\\\")]\n" +
"P=[a for a in A if a[\\\"kind\\\"]==\\\"proposed\\\"]\n" +
"random.Random(0).shuffle(P)\n" +
"for k in range(" + j.a + "," + j.b + "):\n" +
"    print(\\\"=== k\\\", k, \\\"id\\\", P[k][\\\"id\\\"]); print(P[k][\\\"problem\\\"])\n" +
"\"'\n" +
"These were written by a " + j.model + " language model and ADMITTED by a self-play filter. For EACH question decide:\n" +
"  SOLVABLE    - the text determines a unique numeric answer (state it briefly in the note);\n" +
"  UNSOLVABLE  - a needed quantity/relation is missing, or the text is self-contradictory, so no unique answer exists;\n" +
"  AMBIGUOUS   - two reasonable readings give different answers.\n" +
"Be strict: a default assumption you would have to invent makes it UNSOLVABLE. Confidence 0-1, note <= 20 words.\n" +
"Step 2: write ALL " + m + " verdicts as one JSON object {\"verdicts\": [{\"k\": 0, \"id\": \"...\", \"verdict\": \"SOLVABLE\", \"confidence\": 0.9, \"note\": \"...\"}, ...]} to the local file " + local + " with the Write tool (plain JSON, nothing else in the file).\n" +
"Step 3: push and verify with this exact command (Bash tool, one call):\n" +
SSH + " \"cat > " + part + "\" < " + localBash + " && " + SSH + " \"python3 /home/tione/notebook/rsi_rl/dirA3/code/a3_part_check.py " + part + "\"\n" +
"The printed number must equal " + m + ". If it does, set pushed=true and verified_n to that number; otherwise retry the push once, and if it still fails set pushed=false.\n" +
"Step 4: delete the local file: rm " + localBash + "\n" +
"Return run=\"" + j.run + "\", a=" + j.a + ", b=" + j.b + ", n=" + m + ", counts (SOLVABLE/UNSOLVABLE/AMBIGUOUS), pushed, verified_n. Include the verdicts array in your return ONLY if pushed is false."
}
```
