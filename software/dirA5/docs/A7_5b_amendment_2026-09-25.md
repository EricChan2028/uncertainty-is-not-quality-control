# A7-5b amendment (25 Sep 2026, CST; final after the verification review): three API-compatibility edits for the declared stack, before any R-Zero data

Status: DRAFT, to be hashed (appended to `docs/registration_hashes.txt` by the author) before the working copy is rebuilt
and before the smoke test is rerun. It amends `ext/rzero_design/A7_5_addendum_draft.md` (hashed 2026-09-24 19:23:40) and
follows A7-5a (`dirA5/docs/A7_5a_amendment_2026-09-24.md`, 23:08). It changes the patch, the registered tree hash and two
provenance scripts. It does not change the algorithm, any repository default (addendum section 11), the seeds, the T0
clock, the projected-finish rule, the reads or any verdict rule.

**No R-Zero data exist.** No confirmatory seed has started: there is no `ext/rzero_runs/s<S>.host`, `ext/rzero_runs/s0/`
and `s1/` are empty, `/run/ti/rzero/s0/` holds only its `driver.log`, and there is no `rzero` run in `dirB/runs/` and no
`*rzero*` dump in `dirA4/results/umwp/`. Neither smoke test passed. Everything below comes from failed smoke attempts,
a development session in a scratch copy and CPU-only checks; none of it is data.

## 1. What failed

| time (CST) | where | what |
|---|---|---|
| 24 Sep 23:03:56-23:05:26 | smoke test, host B | stopped by its own monitor (exit 6). Its 4 `VLLM::EngineCore` processes (about 59 GB each on GPUs 4-7) had blanked their `/proc` environ view, so they were read as foreign and then orphaned (ppid 1). Run-control fix: A7-5a (`rz_common.sh` only) |
| 23:10:07-23:12:08 | smoke test, host B, S1 | FAILED. `verl/workers/actor/dp_actor.py:27`: `ImportError: cannot import name 'index_first_axis' from 'transformers.modeling_flash_attention_utils'` under transformers 4.57.1. `fsdp_workers.py:390` imports the module lazily inside `init_model`, so the actor workers failed at start-up (`/run/ti/rzero/smoke/smoke.log`, line 565) |
| 23:12:28 / 23:14:41 | waiters | seed-0 waiter (host B) and seed-1 waiter (host A) stopped by signal; not restarted |
| 23:25:43-23:38:40 | dev session, stage s1 with fix F1 | FAILED (777 s). The first GRPO step reached `ray_trainer.py:547`, `batch = batch.union(old_log_probs)`, and stopped in `union_tensor_dict` with `RuntimeError: Cannot modify locked TensorDict` (23:29:34). Under tensordict 0.10.0, `DataProto.__getstate__` had replaced the sender's `self.batch` with the output of `consolidate()`, which 0.10 returns locked (`/run/ti/rzero_dev/logs/s1.232543.log`, line 963) |
| same mechanism as 23:05 | the patch's own P1 and P17 | vLLM 0.11's engine-core subprocess renames itself (`set_process_title("EngineCore")`, `vllm/v1/engine/core.py:697`; vLLM 0.9.1's `core.py` has no such call). That blanks its environ view, so P1 (end of each questioner stage) and the P17 failure path, which signal only processes whose environ carries `RZ_VLLM_RUN=$RUN_ID`, would stop the 4 servers and leave their engine cores running, orphaned, with about 59 GB of GPU memory each, as observed at 23:05. In the failed dev s1 attempt the stage script's last output line came at 23:30:25, but `stage.sh`, which waits for every writer of its log pipe, returned only at 23:38:40 |

## 2. The three fixes (F1-F3)

The reviewed diff is `/run/ti/rzero_dev/compat_fixes.diff` on host B (sha256
`52a330bf10e72d9a1d97a7e8242e8125dd23b5710d401b06e803cfa8f0dd0696`; review verdict: safe, no algorithmic change). The
regenerated patch reproduces it byte for byte (section 4).

**F1 `verl/workers/actor/dp_actor.py:27` (transformers 4.57.1).**
- Old: `from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input`.
- New: `from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input`, with a 3-line comment.
- Computation unchanged. In transformers 4.52.4, the pin, `modeling_flash_attention_utils.py` line 36 is
  `from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input`, checked in the downloaded 4.52.4 wheel.
  Under the pin, dp_actor therefore already called flash-attn's own functions, and it now imports the same functions
  directly. transformers 4.57.1 no longer re-exports them. The bundled `verl/workers/critic/dp_critic.py:37` already
  imports them from `flash_attn.bert_padding`. The flash-attn version itself (2.8.1) is part of D1.

**F2 `verl/protocol.py`, `DataProto.__getstate__` (tensordict 0.10.0).**
- Old: `self.batch = self.batch.contiguous()`, then `self.batch = self.batch.consolidate()`, then
  `torch.save(self.batch, buffer)`.
- New: the same two calls on a local `batch = self.batch`, then `torch.save(batch, buffer)`. `self.batch` is no longer
  replaced.
- Why the change is needed. In tensordict 0.10, `consolidate()` returns a locked TensorDict (0.8.3: unlocked), and a
  consolidated TensorDict can only be saved while locked. Replacing the sender's batch with it made every later key
  insertion fail, as in the 23:29 failure.
- Computation unchanged. The serialized object is the same consolidated TensorDict, so the bytes sent are the same, and
  the sender keeps its batch with the same values (`contiguous()` changes only the layout).
- CPU check for this amendment: `/run/ti/rzero_dev/ref/dp_ops_test.py` (sha256
  `dad1f42e5f23af806a9a9b6d94b679e11af295e10f71dc1e1d5760af9043db65`) runs every DataProto operation the trainer uses,
  including pickling round-trips, union after pickling, chunk/concat, select, pad/unpad, split and the mini-batch
  iterator, and prints a digest per operation. The digests are identical for:
  - the registered verl with tensordict 0.8.3;
  - the fixed verl with 0.8.3;
  - the fixed verl with 0.10.0.

  The registered verl with 0.10.0 fails with the same locked-TensorDict error as dev s1.

**F3 `scripts/questioner_train_penalty.sh`: `a75_desc` / `a75_start` / `a75_stop` (vLLM 0.11).**
- Old (P1 at the end of each questioner stage, and the P17 failure path): TERM, then KILL, the processes whose environ
  carries `RZ_VLLM_RUN=$RUN_ID`.
- New: `a75_stop`, defined once after the P15 resume lines and used at both places.
  - It first collects the tagged processes and all their descendants (`a75_desc`, recursive `pgrep -P`), each with its
    start time (`a75_start`, field 22 of `/proc/<pid>/stat`).
  - Then, for TERM and then KILL (15 s apart), it signals the currently tagged processes and every collected pid whose
    start time is unchanged. The start-time test is a pid-reuse guard.
- Computation unchanged. F3 decides only which processes are stopped, and it runs only after training has ended (P1, the
  end-of-stage shutdown of the reward servers) or before training starts (P17, a failed server start); it never runs during
  training.
- It closes the orphaning path of section 1 inside the patch, as A7-5a did for `kill_tag`.

**Allowed-change category.** Addendum section 4.2 allows "API-compatibility edits in the bundled verl or the scripts that
vLLM 0.11 / torch 2.8 make necessary and that keep the computation identical". F3 is caused by vLLM 0.11. F1 and F2 are
caused by transformers 4.57.1 and tensordict 0.10.0. We read the item as covering the whole declared stack: D1 (section
10) lists torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.1, ray 2.49.2, flash-attn 2.8.1, tensordict 0.10.0,
datasets 4.3.0 and numpy 1.26.4 as the framework versions in use. This reading is declared here. X3 (a change outside the
allowed list) is therefore not triggered.

## 3. Development evidence (host B, 24-25 Sep)

A development session (`rzA75_dev`) held all 8 GPU markers of host B from 23:23:18 to 00:52:34 (`hold_markers.sh`;
`/run/ti/rzero_dev/markers.log`).
- Code: a scratch copy, `/run/ti/rzero_dev/R-Zero` = the registered tree plus F1-F3.
- Environment: the registered `rz_env.sh`, with only the code, storage and temp paths redirected (`dev_env.sh`, sha256
  `a7feb670e050c68f5cd5ae87c777fa32a10f3cae1a112625dcd8ad6f52747da5`).
- Stage runner: `stage.sh`, sha256 `cb49f90718ed175c70b53ac68eb42d525d1436b05a55f95b413ff1a9001e904b`.
- Sizes: the smoke test's hook values, except 16 generations per shard (smoke: 100) and 10 + 10 UMWP items (smoke:
  5 + 5).
- Nothing was written to `/run/ti/rzero`. The s4 dump was deleted from `dirA4/results/umwp/` after it was copied to the
  dev logs.

| stage | what | start-end | wall | result |
|---|---|---|---|---|
| s1 (= smoke S1) | questioner stage, 1 GRPO step at rollout batch 32: 4 reward servers, `caller_penalty` with BLEU clustering, FSDP + vLLM 0.11 rollout and weight sync, checkpoint, merge, P1 shutdown (`a75_stop`) | 23:43:04-23:52:54 | 590 s | ok (`A75_DONE`) |
| s3 (= smoke S3) | solver stage: 8 x 16 generations (128 raw, 76 scored, 32 admitted), pseudo-labels, 0.3-0.8 filter, local parquet, 1 GRPO step on 8 GPUs, merge | 23:53:52-00:17:50 | 571 s (see note) | ok |
| s4 (= smoke S4) | the merged solver through `umwp_probe.py` (10 + 10 items) | 00:21:31-00:22:09 | 38 s | ok (20 rows) |
| s3b (resume) | the solver stage continued from `global_step_1` to step 2 ("Load from checkpoint: .../global_step_1"), merged at step 2 | 00:35:33-00:39:46 | 253 s | ok |
| s2 (= smoke S2, resume) | the questioner stage continued from `global_step_1` to step 2 ("Load from checkpoint: .../global_step_1"), merged at step 2 | 00:40:13-00:50:35 | 622 s | ok |

Note on s3: the stage log ends at 00:03:23 ("solver training finished"); `stage.sh` returned only at 00:17:50 (+867 s),
most likely because P3's `kill $timer_pid` in question_evaluate/evaluate.sh kills the timer subshell but leaves its
`sleep 3600` running as an orphan that held the dev log pipe (reproduced in scratch; why it ended at 00:17:50 is not
explained). The confirmatory drivers redirect to files, not pipes, so this does not affect them; 1,438 s is not the stage
duration.

All five stage runs passed, including the two resume runs:
- Both resume stages end with `A75_DONE`, `A75_MERGED_step1`, `A75_MERGED_step2` and only `global_step_2` kept (P16).
- After s1, s3, s3b and s2, `stage.sh` found no process of the session left, counting processes by working directory
  under `/run/ti/rzero_dev` or by the session tag, which also catches the renamed engine cores. So `a75_stop` stopped the
  servers' engine cores in both questioner stages.
- The one leftover note, after s4, names two processes of a separate CPU/GPU timing diagnostic that started at 00:22:07,
  not processes of s4.

Per-step trainer timings (`timing_s`, from the stage logs):

| stage | step | gen | old | ref | adv | update_actor |
|---|---|---|---|---|---|---|
| s1 questioner | 171.2 | 36.5 | 3.6 | 3.3 | 82.9 | 28.3 |
| s2 questioner, resumed step 2 | 153.5 | 35.8 | 4.1 | 3.2 | 73.3 | 24.2 |
| s3 solver | 251.9 | 34.1 | 4.6 | 4.4 | 178.2 | 20.4 |
| s3b solver, resumed step 2 | 69.9 | 33.1 | 4.6 | 4.4 | 0.002 | 17.9 |

Provenance caveat. `dp_actor.py` in the dev copy was last written at 00:41:18, and its compiled cache was regenerated at
00:41:19, during s2 (the s2 workers began loading the model at 00:41:15, and the lazy import comes after model build and
FSDP wrap). So s2 probably ran on the reviewed F1 text, with a margin of about one second; no dev stage is certain to have
run the final F1 text. s1, s3 and s3b ran on an earlier text of that file. That text
cannot be reconstructed, but its import line must have resolved: a failing import stops a stage at worker start-up, as at
23:10. `protocol.py` (23:40:00) and `questioner_train_penalty.sh` (23:42:53) were final before s1 started. The rebuilt tree
has not run on a GPU. This is one more reason for the fresh smoke test (section 7).

## 4. How the change was applied (addendum section 4.2, review item 7)

All work was CPU-only, in `/run/ti/rzero_design_scratch` on host B. No GPU process was started, no running process was
touched, and `/run/ti/rzero/R-Zero` was not modified.
1. `dev/mkpatch.py` gained two replacement entries, `verl/protocol.py` (F2) and `verl/workers/actor/dp_actor.py` (F1).
   Its questioner entries now produce the F3 text: the `a75_*` block after the P15 echo line, `a75_stop` in the P17
   failure path and `a75_stop` as the P1 line. The texts were taken verbatim from the reviewed dev tree. Every
   replacement still matches exactly once.
2. Reproducibility check: the registered `mkpatch.py`, rerun, reproduces the registered patch
   (`7b8c11c0...e188`) byte for byte.
3. The new `rzero_A7_5.patch` has 372 lines over 13 files (registered: 322 lines over 11 files). It was applied, as
   `rz_setup.sh` does, to a fresh copy of `ext/R-Zero_5699329`: `COMMIT` checked, `__pycache__` dropped, 92 CR-stripped
   files, `patch -p1 --forward`. The result is byte-identical to `/run/ti/rzero_dev/R-Zero` in all 98 source files. The
   dev tree has only `.a75_patched`, `.a75_tree_sha256`, `test.json`, `scores_distribution.png` and `__pycache__`
   beyond them (`diff -r` exit 0 with those excluded).
4. Tree hash, computed as `rz_setup.sh` does (`find . -type f ! -name '.a75_*' -print0 | LC_ALL=C sort -z | xargs -0
   sha256sum | sha256sum`), is `eb84092160b556ad9c4382f5795dd4b519414c3dbff66c949a18796fe37b0a1b`. The same command on
   the registered patch gives the registered `69ae7291...7ea2`. `A75_TREE` in `rz_common.sh` is set to the new value.
   That line is the only change to the file, and the A7-5a helpers (`rz_cwd`, `anc_tagged`, `rz_engine_pids`) are intact.
5. `dev/patchtest.sh` (CPU only): the P17 test now extracts `a75_desc`, `a75_start` and `a75_stop` together with
   `a75_alive` and the wait loop. `verl/protocol.py` and `verl/workers/actor/dp_actor.py` joined the `ast` list. A P1-path
   test was added: a fake tagged server whose child runs without `RZ_VLLM_RUN` (`env -u`) and has a child of its own,
   plus another RUN_ID's server with such a child and an unrelated untagged process. The script's own P1 line
   (`a75_stop`) must stop the first three and leave the other four.
   Result (98 s, exit 0):
   - tree `eb840921...0a1b`;
   - `bash -n` ok on the 6 stage scripts; `ast` ok on 7 Python files;
   - `PPOConfig` parses with the questioner and solver+smoke overrides, and `ray.init` accepts `object_store_memory`;
   - the stage guard skips a completed stage and stops on a failed one;
   - P17: the extract has 4 functions, the loop fails fast and stops the tagged server, and the other RUN_ID is untouched;
   - P1: the 2 descendants carry no tag; alive after `a75_stop`: 0 of 3 for the tagged server tree and 4 of 4 for the
     rest; test processes left: 0;
   - patch `de85bc30...1b06`.

   Control run in scratch: the registered P1 loop, against the same fake tree, left 2 of 3 alive (the untagged child and
   grandchild).
6. Import sweep, which supersedes the CPU import claim of section 2 (section 5 below). With the run venv
   (`/run/ti/rzero/venv/bin/python`, `CUDA_VISIBLE_DEVICES=""`), every module of the bundled verl was imported from the
   new tree: all 61 submodules plus the package root (62 `.py` files), 0 failures. On the registered tree the same sweep
   fails in exactly one module, `verl.workers.actor.dp_actor` (the 23:10 ImportError).
7. `dev/designtest.sh` (CPU only; unchanged file), run against the design directory with the new patch and `A75_TREE`:
   exit 0 after 571 s.
   - T0 (syntax): `bash -n` ok on 10 shell scripts, `py_compile` ok on 4 Python files, no CR.
   - T1, T3-T8 and T10-T15 give the expected results, as in the registered rehearsal.
   - T2: the wrong-tree refusal names the new `A75_TREE`, and the changed-patch refusal holds. The live `host_check` /
     `preflight` half printed "preflight passed?!" because host B was completely free at 01:30 (0 MiB on every GPU, no
     job script, no marker). That subtest presumes a busy host and says nothing about the changed files.
   - T9 (`rz_setup.sh` with its paths redirected to scratch):
     - the build applied the new patch, with tree `eb840921...0a1b` = `A75_TREE`;
     - the first run then exited 1 at the one-time math12k fetch, after the build and the venv step, because
       hf-mirror.com timed out from host B (it answered from host A at the same time);
     - the second run fetched the data (12,000 / 500 rows, the registered sha256), passed the CPU import checks and
       wrote `setup ok ... tree eb840921...0a1b ... stack=236a8e4298d75ff1` to the scratch ledger, without rebuilding;
     - a rebuild with the caller in the C locale gave the same tree;
     - a changed patch was refused as fatal (`18ccabe2...`), with the working copy kept.

     The real setup reads math12k from `ext/rzero_data/`, which already holds both files.

## 5. Corrections to the addendum's wording (read into `A7_5_addendum_draft.md`, which stays as hashed)

- **Section 2, stack paragraph.** "verl 0.3.1.dev0, unmodified except P11" now reads "unmodified except P11 and the two
  API-compatibility edits of A7-5b: F1 (`dp_actor.py` import) and F2 (`DataProto.__getstate__`)".
- **Section 2, CPU claim.** "Imports, config parsing and the vLLM 0.11 API used by the rollout and weight sync were
  checked on the CPU" is superseded. That check (and step 6 of `rz_setup.sh`) imported `verl.trainer.main` and
  `verl.workers.fsdp_workers`, which load `dp_actor` only inside `init_model`, so it missed F1. It is replaced by the full
  import sweep of the 61 verl submodules (section 4, item 6). `rz_setup.sh` itself is unchanged.
- **Section 2, item 3, and section 13.** Registered tree hash `eb84092160b556ad9c4382f5795dd4b519414c3dbff66c949a18796fe37b0a1b`
  (was `69ae7291...7ea2`).
- **Section 3.** The patch has 372 lines over 13 files and now carries F1-F3 besides P1-P19. P1 and the P17 failure path
  stop the tagged servers and all their descendants (`a75_stop`, pid-reuse guard); the P2 row's "and their children"
  now holds through `a75_stop`. `patchtest.sh` also runs the P1-path test, and `ast`-parses the two edited verl files.
- **Section 10.** D1 now includes the edits F1 and F2 (and F3 for vLLM 0.11's process titles); effect on the algorithm:
  none. D15 lists this amendment.
- **Appendix A** is superseded by the new `rzero_A7_5.patch` (hash below).

## 6. Clock-risk note (watched, not changed)

- **What the adv timer measures.** In the trainer, the reward is dispatched asynchronously (`timing_s/reward`, about
  0.02 s) and collected inside the `adv` timer (`ray.get(reward_ref)`, `ray_trainer.py:563`). `timing_s/adv` is
  therefore the reward time not hidden behind the old/ref log-prob passes.
- **Questioner.** adv was 82.9 s and 73.3 s of 171.2 s and 153.5 s steps at rollout batch 32. The confirmatory step has
  512 prompts, 16 times more. Scaling adv in proportion, which ignores batching gains in the 4 reward servers and is not a
  measurement, gives about 20-22 min per step for adv alone. update_actor, old and ref also grow with the 16x batch (32
  updates per step instead of 2); scaling adv + update_actor + old + ref in proportion gives about 28-31.5 min per step,
  i.e. 2.8-3.2 h for the 6 steps before generation, checkpoints, server start, merge and ~7 min of idle epochs - at or
  above the 3.5 h upper value used for the whole questioner stage by the projected-finish rule. Flagged downside: once q1
  is measured the rule uses 1.1 x the measurement, which may downgrade a committed full mode to RZ-P1-only (RZ-P2 not run
  for that seed).
- **Solver.** adv was 178.2 s in s3 and 0.002 s in the resumed s3b at the same sizes. Graded one response at a time on 160
  base-solver rollouts of the dev admitted set, `math.py` took 0.03 s in total (`/run/ti/rzero_dev/logs/reward_timing.log`).
  The 178 s is therefore not explained by normal grading, and its cause was not identified.
- **No timeout in the grader.** `examples/reward_function/math.py` grades through `mathruler.grader.grade_answer`
  (sympy), which has no timeout; its source notes that sympy may hang. One pathological response can therefore stall a
  solver step. Only the monitor's 120-min hang rule ends such a stall, and the attempt then counts as failed (X4).
- **Not changed.** A timeout would change the reward function, which stays at the repository default (section 11).
- **What to watch.** Read `timing_s/adv` and `timing_s/step` in `main.log` during q1 and s1 of the first seed. After the
  first measured stage, the projected-finish rule uses 1.1 x the measured duration in place of the upper values, and it
  can only downgrade a committed mode.
- **X2 budget.** The smoke test failed at 23:12:08 on 24 Sep. Counted continuously from then, which is the conservative
  reading, the 8 h of accumulated debugging allowed by X2 end at 07:12 on 25 Sep.

Procedure note (added before hashing): dev/designtest.sh must not be run on a host with an active A7-5 smoke or seed run:
its T1 calls kill_tag without setting RZ, so since A7-5a it would also stop every VLLM::* or GPU process whose working
directory is under /run/ti/rzero, i.e. a live seed's engines.

## 7. Consequences and next steps (for the author)

- The patch and tree hashes are inputs of the stack fingerprint (`stack_fp`), so the fingerprint changes. The formula,
  reproduced on host B, gives the current `b484b203c7f43e80` for the registered values and `236a8e4298d75ff1` for the new
  values if `rz_setup.sh` refreezes the same venv package list. The redirected setup of designtest T9 printed the same
  value. **A fresh smoke PASS on the new stack is required**
  before seed 0. Seed 1 can skip its smoke test only on a PASS with the new fingerprint (addendum section 4.2).
- `rz_setup.sh` must be rerun on each host after the hash, so that the working copy is rebuilt from the new patch. Until
  then `host_check` refuses ("working copy built from another patch"). The setup refuses to rebuild while an A7-5 run is
  active; none is active now.
- Both waiters, stopped at 23:12:28 and 23:14:41, are restarted after the rebuild with the commands of section 4.2.
- The T0 clock, the X5 stop at T0 + 71 h (27 Sep 14:10) and the projected-finish rule are unchanged.
- `/run/ti/rzero_dev` (18 GB, host B: the dev copy, its checkpoints and logs) is not data and is not read by any A7-5
  script. It is kept as the evidence for section 3.

## 8. Hashes

- New patch `rzero_A7_5.patch`: `de85bc30b75cf75dc081741488a383f5ef108cad7ee6216dcd30e50d72d61b06`
  (was `7b8c11c0a83cf16b0cad013df567d62c41ff7c2736aa958f91e6ab07c541e188`).
- New `A75_TREE`: `eb84092160b556ad9c4382f5795dd4b519414c3dbff66c949a18796fe37b0a1b`
  (was `69ae729158dce32ea0e0d59216b2542001affbb11ab61739093020351a1f7ea2`).

Companion files changed by this amendment (all in `ext/rzero_design/`):

| file | new sha256 | previous sha256 |
|---|---|---|
| `rzero_A7_5.patch` | `de85bc30b75cf75dc081741488a383f5ef108cad7ee6216dcd30e50d72d61b06` | `7b8c11c0a83cf16b0cad013df567d62c41ff7c2736aa958f91e6ab07c541e188` |
| `rz_common.sh` (`A75_TREE` line only) | `9c6abf834a91ed5d271fbc5d3c44a7c35fb404f8d1aa68b4c09ea2599380538c` | `364d129086eb81cf033c1593e28f85486ecbfe2f7369da886193e1e63d85af47` (A7-5a) |
| `dev/mkpatch.py` | `12531cae9fcf209294b19f1e7b79248f33184f51f3a086d69e5829ebb7b01ffc` | `a227888a39ef295f40774031867dc5fe06c933315b6bcdd76a8eefda8e5e9af9` |
| `dev/patchtest.sh` | `efe26e3b6756e831b8f3534b30ad0ad496259a1f77f7963ac9409ea4409d84ee` | `8fafcbb090b36e3476d5a4bf93cead63ec6b06cde44da283e677cf80e7c37429` |

Unchanged (sha256 as registered on 24 Sep):
- `A7_5_addendum_draft.md` `966da88d0564dacc6506adce226c1eb2a5e6cc2b830e75ab04cf8155d52056b1`
- `rz_setup.sh` `e0924be2a0449550bb032b1772292b9c6ea47e48df804bfe30e5514f0e8fce33`
- `rz_env.sh` `9c3312bfa4f15767dbc2e1d795db4919951650e41ea2899d6f0cb26b8e14b092`
- `rz_wait_and_run.sh` `653b4e9da13ede2579c9c33b0ca27086fd57f6d660e6f03e50f783c6f8d54d28`
- `rz_smoke.sh` `fe92ea4127f89bc4f9fe2a7d68351d231f29883ce831229301fec10af60bb90e`
- `rz_run.sh` `561d1b8ebf79f6a13f86a65b41b8dbfda643174c2aace9e8d5a66c898adc466b`
- `rz_stop.sh` `0410d01600c28b719284e1c0cdf61df23c9cdf611ab866ab2b6e8de9399997b6`
- `rz_clock.py` `a29a51f1d6863196c6c56c3f81bc00ae440a4be021872d93e0d61226493a841c`
- `rz_extract.py` `1e64b05425b2e2eac52498af293d94926aa5224086df9d1f9eca40a426cd9d4b`
- `rz_audit200_wf.js` `b4d89f1d2f977e628ff2ae71788bf9a7420d93c05ab7b5f3001c80a6d28baafc`
- `rz_umwp.sh` `22126f180c6a4e1b73d858a104f022e0151639f8386a0032dba092980fb4fc8b`
- `rz_read.py` `1fbb53505a2af6c6fc68b4226dc9c68ccc8c0d8568bc1d7e9d7728a3c47065a1`
- `dev/designtest.sh` `57b92a3e92c3c0553d33da81e9d57e5bc0f1c9132cb062cd1570a325257f4a95`

Evidence referenced (host B, not companion files):
- `/run/ti/rzero_dev/compat_fixes.diff` `52a330bf10e72d9a1d97a7e8242e8125dd23b5710d401b06e803cfa8f0dd0696`
- `/run/ti/rzero_dev/ref/dp_ops_test.py` `dad1f42e5f23af806a9a9b6d94b679e11af295e10f71dc1e1d5760af9043db65`
- `/run/ti/rzero_dev/stage.sh` `cb49f90718ed175c70b53ac68eb42d525d1436b05a55f95b413ff1a9001e904b`
- `/run/ti/rzero_dev/dev_env.sh` `a7feb670e050c68f5cd5ae87c777fa32a10f3cae1a112625dcd8ad6f52747da5`
- `/run/ti/rzero_dev/hold_markers.sh` `453f05052f4e9dbae26d0d1d916a426ab1cb0d2cedf59f228f5361478ef22a30`
