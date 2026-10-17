# A11 registration: corrected abstention-prompt re-run on UMWP and the A6 external sets (DRAFT, 26 Sep 2026, before any corrected data)

Status: draft in ext/a11_draft/. It becomes a registration only when this file and the four scripts next to it are hashed
into docs/registration_hashes.txt (command in section 4) [internal note removed]. The launcher
refuses to run until all five files are in the registry. Revision 2 (26 Sep, after two reviews): signal handling of the
launcher rewritten, Tier 2 made required and extended by the four TG cells, P2 switched to the registered E2 aggregation, a
12 h read cut-off, a repo-wide defect sweep, and a fact-to-cell map.

## 1. Why: the defect

- dirA3/code/umwp_probe.py L42 (frozen, sha256 32de59f95a296fa0c026b44cf8a6110172898b87b258179ab3a54da00ee1e56d) and
  dirA5/code/a6_probe.py L37 (sha256 af94764de247aa2c762b8b7bad67fe835a89fab4e814e3547f4e9b60741ec9f9) both read
  `SOLVE_ABS = "Solve the problem step by step and output the final answer within \boxed{}."`. The string is not raw, so
  Python turns `\b` into U+0008 (backspace). The value the interpreter holds (checked with `ast` on both files, 26 Sep) is
  `'Solve the problem step by step and output the final answer within \x08oxed{}.'`. Every abstention-prompt call of these
  two probes therefore sent "... within <BS>oxed{}." to the model.
- These parts are correct: the appended option (dirB/code/pdc_util.py L13, ABSTAIN_SUFFIX
  `" If the problem does not provide enough information to determine the answer, output \\boxed{unknown}."`), the twin
  probe's instruction (dirA/code/c0_probe.py L100, `"...within \\boxed{}."`), the guard and training prompts (a3_guard.py
  L44, a3_propose.py L32, a4_evolve.py L20, build_mixed.py L7), the plain-prompt instruction (no backslash), and parsing
  (loto_common.last_boxed / pdc_util.is_abstain on the last \boxed{}).
- Affected dumps: every dirA4/results/umwp/abs_*.jsonl (65 dumps) and dirA5/results/a6_sweep/abs_*.jsonl (8 dumps).
  Affected reads: E2 (A6, a6_e_read.py); the ABS_U / ABS_A clauses of HX3 and HX6 (A4h, umwp_read.py, umwp_read_c.py); the
  UMWP cells of F-P1, F-P2 and F-P3 and the UMWP frontier descriptives (A5, a5_frontier.py -> frontier_read.json); the UMWP
  cells of FB1 and FB2 and the CA_U at FA_A = 0.05 descriptives (a5_frontier_b.py); the descriptive ABS_U / ABS_A
  (umwp_read.py, umwp_read_b.py, umwp_read_c.py, ext/rzero_design/rz_read.py); and every abstention-prompt CA derived from
  these dumps. Not affected: plain-prompt numbers, all twin hp/hpabs numbers (c0_probe.py), and the twin cells of F-P1/FB3.
- The size of the effect is unknown because the dumps store parsed answers, not texts. The factsheets already disclose the
  defect (H-NEW-01). This re-run replaces the affected numbers with measured ones and leaves the defective-prompt numbers
  reported (section 10).

Repo-wide sweep (26 Sep, before hashing). An `ast` scan of every Python file in the repository (319 files: 242 project files and 77 in the vendored
ext/R-Zero_5699329; hf_home, the model cache, excluded) for str or bytes constants that contain U+0008 finds 11 constants in 7 files:
- dirA3/code/umwp_probe.py L42 and dirA5/code/a6_probe.py L37: the defect above, fixed by this re-run.
- dirA3/code/umwp_probe_text.py L46 (sha256 09625cb5…44ec): the same literal in the A8 text probe. Its only caller,
  dirA5/code/a8_gen.sh L13, runs it with `--abstain_prompt 0`, so the literal is never sent and no data is affected. Latent.
- release_v1/probe/code/umwp_probe.py L42 (sha256 e5f77b7d…dfef03a): the public release copy of the probe. It must be fixed
  before release (reporting item 7). release_v1/build_release.py L1024 globs dirA4/results/umwp/*.jsonl, so a rebuild would
  also pick up the absfix_* dumps; item 7 covers the labels.
- code/prep_math.py L11, L28 (two constants), L29, L33: backspace in the boxed-answer parser and in the instruction. Its
  output data/math/math500.parquet has U+0008 in all 500 prompts. Its only consumers are code/loto_train.py L616 and
  code/posthoc_retention.py L24, from the dead LOTO phase; no number in the current results comes from them.
  data/math/train.parquet and test.parquet (rebuilt by prep_math2.py) have 0 of 7,498 and 0 of 5,000 rows with U+0008.
- dirB/code/build_mixed.py L8 and dirC/code/build_c_data.py L12: `assert "\x08" not in SUF`, deliberate guards against this
  defect (not a defect).
- Two legacy files do not parse (code/_recover.py, code/_promote_cheap.py); neither contains a backslash-b or a 0x08 byte.
- Data: the 16 .jsonl files under dirA*/data, dirB/data, dirC/data and data_ext (training sets, UMWP and A6 items) have 0
  rows with U+0008. None of the 88 .sh files builds a boxed instruction (the only "boxed" matches are comments and this
  launcher's preflight).
So the only affected numbers in the results are those computed from the abs_ dumps of umwp_probe.py and a6_probe.py.

## 2. What is re-run

Only the abstention prompt (`--abstain_prompt 1`). Everything else is as in the old dumps: same items (UMWP: the Random(0)
600 unanswerable + their 600 answerable originals; A6: data_ext/a6/sweep.jsonl, 1,552 rows = SUM 284 U, MiP-GSM8K 582 U +
582 A, MiP-MATH 52 U + 52 A), G = 8, T = 1.0, top_p = 1.0, max_new 1024, vLLM seed 17, bf16, gpu_memory_utilization 0.85,
the base-model prompt (the model path contains "Base"), /usr/bin/python3 with vLLM 0.11.0, RTX PRO 5000 72GB GPUs, and the
same checkpoints. The plain-prompt dumps are not re-run; P3 and HX6 reuse them unchanged.

Host A = port <PORT> (<NODE>), host B = port <PORT> (<NODE>). The paths come from
`a11_launch.sh locate` on both hosts (26 Sep 16:16 CST). Every listed checkpoint had config.json, all shards and a
tokenizer. Paths marked (L) are symlinks from /dev/shm/ckpt_b/<run>/final to /run/ti/rsi_ckpt_archive/<run>/final
(host-local disk). Every one of the 38 cells has its checkpoint on at least one host.

Both tiers are required: the registered run command is `run` (= `run all`), and the launcher has no partial-tier mode.
Tier 1 runs first (the confirmatory cells and the named descriptives; 23 cells). UMWP = dirA3-style probe, A6 = the three
sets. The two R-Zero cells come first because each checkpoint exists on one host only and the A7-5 teardown may remove
/run/ti/rzero (section 8).

| cell | bench | used by | host A | host B |
|---|---|---|---|---|
| rzero_s0_it3 | UMWP | descriptive (FR-7, RZ-27, RZ-34, H-RZ-02) | MISSING | /run/ti/rzero/s0/models/rzero_qwen3_4b_Base_s0_solver_v3/global_step_15/actor/huggingface (A75_DONE present) |
| rzero_s1_it3 | UMWP | descriptive (FR-7, RZ-27, RZ-34, H-RZ-02) | /run/ti/rzero/s1/models/rzero_qwen3_4b_Base_s1_solver_v3/global_step_15/actor/huggingface (A75_DONE present) | MISSING |
| qwen3_4b_Base_own_a3_s0_r3 | UMWP, A6 | P1, P2, P3 | /dev/shm/ckpt_b/…/final (L) | /dev/shm/ckpt_b/…/final (L) |
| qwen3_4b_Base_own_a3_s1_r3 | UMWP, A6 | P1, P2, P3 | /dev/shm/ckpt_b/…/final (L) | MISSING |
| qwen3_4b_Base_own_a3_s2_r3 | UMWP, A6 | P1, P2, P3 | /dev/shm/ckpt_b/…/final (L) | /dev/shm/ckpt_b/…/final (L) |
| qwen3_4b_Base_own_a3_s3_r3 | UMWP, A6 | P1, P2 | /dev/shm/ckpt_b/…/final | MISSING |
| qwen3_4b_Base_own_clean_a3_s0_r3 | UMWP, A6 | P1, P2 | /dev/shm/ckpt_b/…/final | MISSING |
| qwen3_4b_Base_own_clean_a3_s1_r3 | UMWP, A6 | P1, P2 | MISSING | /dev/shm/ckpt_b/…/final |
| qwen3_4b_Base_own_clean_a3_s2_r3 | UMWP, A6 | P1, P2 | /dev/shm/ckpt_b/…/final | MISSING |
| qwen3_4b_Base_own_clean_a3_s3_r3 | UMWP, A6 | P1, P2 | /dev/shm/ckpt_b/…/final | MISSING |
| qwen3_4b_Base_own_ag_a4_s0_r3 (guard) | UMWP | P3 (guard FA_A), HX6 | /dev/shm/ckpt_b/…/final | MISSING |
| qwen3_4b_Base_own_ag_a4_s1_r3 (guard) | UMWP | P3, HX6 | MISSING | /dev/shm/ckpt_b/…/final |
| qwen3_4b_Base_own_ag_a4_s2_r3 (guard) | UMWP | P3, HX6 | /dev/shm/ckpt_b/…/final | MISSING |
| qwen3_4b_base | UMWP | descriptive (base) | /root/models/Qwen3-4B-Base | /root/models/Qwen3-4B-Base |
| qwen3_4b_Base_own_a3_s4_r3 | UMWP | descriptive, FB1 comparator | /dev/shm/ckpt_b/…/final | MISSING |

The A6 bench uses own_a3 and own_clean_a3 s0-s3 only (8 cells); the UMWP bench uses all 15 rows (15 cells). Where to find
the guard: F-P1 in a5_frontier.py (L175-195) reads the guard's plain dump for CA_U and its abstention-prompt dump for FA_A,
and own r3's abstention-prompt dump for the frontier (plus own's plain dump for the sensitivity frontier). So the guard's
abstention-prompt dump is re-run, and its plain dump is reused.

Tier 2 (required; descriptive cells; 15 UMWP cells, started after every Tier 1 cell has been claimed), all prefixed
qwen3_4b_Base_ and all under /dev/shm/ckpt_b/<run>/final: rescue_uv_a4 r3 s0 (A), s1 (B), s2 (A, B), s4 (B) [HX3, FB1,
UV-3, PR-12]; rescue_abs_a4 r3 s0 (A), s1 (A), s2 (B) [AB-2, FR-6, H-ABS-03]; loop_uv_a4 r4 s0 (A), s2 (A) [FB2, FR-5];
coev_ag_a4 r3 s0 (B), s1 (A) [A4h addendum c descriptive]; tg_abs4_a5 r4 s0 (A), s1 (B), s2 (A) and tg_t4_a5 r4 s1 (B)
[FR-6, the a5_frontier_b descriptive TG rows].

Not re-run (35 of the 65 old UMWP abs_ dumps). Their abstention-prompt numbers stay defective-prompt only and go to the
appendix with the flag: Qwen3-8B base and own r3 s0-2; Phi-4-mini base, own r3 s0-1, own_ag r3 s0-1 and cln1/rnd1_a6 s0-1
(chat template; the defect applies there too); Qwen3-4B own r5 s0-2, own_ag r5 s0-2, cln1_a6 / rnd1_a6 s0-4 and
ctrl_lowp / ctrl_rnd a5 s0-2. Consequently F-P2 (it needs the Phi own/own_ag lineages), the F-P1 Phi exploratory rows,
and every Qwen3-8B and Phi-4-mini ABS_U / ABS_A number (umwp_read_b.py) remain defective-prompt only: appendix, flagged,
never primary. loop_uv s1 r4 never had a UMWP dump and stays NOT RUN, as before; tg_t4 s0/s2 r4 were cancelled by the A5
kill gate and have no dump.

## 3. Code (new files; nothing existing is modified)

| file | sha256 |
|---|---|
| ext/a11_draft/umwp_probe_absfix.py | 11f796c2af5c56afe72c802cf34e444e391c1b53f10561e5c115bb0b7f6cfa68 |
| ext/a11_draft/a6_probe_absfix.py | 9576635bf12570f48918785572bbea01b8a7933a6e1a428b31fa90598f6f532e |
| ext/a11_draft/a11_launch.sh | f30026dbfdf047577a28e635852fa25776e7cb3b80d68cbdecbe981b081f7799 |
| ext/a11_draft/a11_read.py | 3d208d62b2cba96a3394267884aba37e5f2de11c0c1983cc9cc8b90c893147d5 |
| this file | recorded in docs/registration_hashes.txt at registration |

Each probe is a byte copy of its registered original with two edits. (1) The instruction is made a raw string. (2) One line
is added after argument parsing: the probe refuses any tag without the `absfix_` prefix and any run without
`--abstain_prompt 1`. Without it, a wrong tag would silently "resume" a complete old abs_ dump by skipping every id and
exiting as done. The unified diffs:

```diff
--- dirA3/code/umwp_probe.py	2026-09-24 05:33:58.844737085 +0800
+++ ext/a11_draft/umwp_probe_absfix.py	2026-09-26 14:53:41.491551046 +0800
@@ -17,6 +17,7 @@
 ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--gpu_util", type=float, default=0.85)
 ap.add_argument("--n_uns", type=int, default=600)
 args = ap.parse_args()
+assert args.tag.startswith("absfix_") and args.abstain_prompt == 1, "A11: this probe writes only corrected abstention-prompt dumps, tagged absfix_<tag>"
 R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA4/results/umwp"; os.makedirs(OUT, exist_ok=True)
 D = [json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl")]
 byid = {x["id"]: x for x in D}
@@ -39,7 +40,7 @@
 llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048 + args.max_new, dtype="bfloat16", enable_prefix_caching=True)
 tok = llm.get_tokenizer(); is_chat = tok.chat_template is not None and "Base" not in args.model
 SOLVE = "Solve the problem. Show your reasoning briefly, then give the final numeric answer on the last line as 'Answer: <number>'."
-SOLVE_ABS = "Solve the problem step by step and output the final answer within \boxed{}."
+SOLVE_ABS = r"Solve the problem step by step and output the final answer within \boxed{}."
 if args.abstain_prompt: SOLVE = SOLVE_ABS
 def wrap(problem, instruction):
     if args.abstain_prompt: problem = problem + ABSTAIN_SUFFIX
--- dirA5/code/a6_probe.py	2026-09-24 09:39:17.211839786 +0800
+++ ext/a11_draft/a6_probe_absfix.py	2026-09-26 14:53:41.501551055 +0800
@@ -17,6 +17,7 @@
 ap.add_argument("--max_new", type=int, default=1024); ap.add_argument("--gpu_util", type=float, default=0.85)
 ap.add_argument("--n_uns", type=int, default=600)
 args = ap.parse_args()
+assert args.tag.startswith("absfix_") and args.abstain_prompt == 1, "A11: this probe writes only corrected abstention-prompt dumps, tagged absfix_<tag>"
 R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA5/results/a6_sweep"; os.makedirs(OUT, exist_ok=True)
 items = [json.loads(l) for l in open(f"{R}/data_ext/a6/sweep.jsonl")]
 outp = f"{OUT}/{args.tag}.jsonl"
@@ -34,7 +35,7 @@
 llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_util, max_model_len=2048 + args.max_new, dtype="bfloat16", enable_prefix_caching=True)
 tok = llm.get_tokenizer(); is_chat = tok.chat_template is not None and "Base" not in args.model
 SOLVE = "Solve the problem. Show your reasoning briefly, then give the final numeric answer on the last line as 'Answer: <number>'."
-SOLVE_ABS = "Solve the problem step by step and output the final answer within \boxed{}."
+SOLVE_ABS = r"Solve the problem step by step and output the final answer within \boxed{}."
 if args.abstain_prompt: SOLVE = SOLVE_ABS
 def wrap(problem, instruction):
     if args.abstain_prompt: problem = problem + ABSTAIN_SUFFIX
```

The fixed instruction is at L43 (UMWP) and L38 (A6) of the copies. The launcher's preflight rebuilds each copy from its
registered original plus exactly these two edits and compares bytes. It pins the sha256 of both originals, pdc_util.py,
code/loto_common.py (901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11; last_boxed and norm_num, imported
by both probes; not in docs/registration_hashes.txt before, so this pin registers it through the launcher's hash),
StandardDataset.jsonl and sweep.jsonl. It checks that both helpers resolve exactly as in the probe (script directory, then
code/ for loto_common, then dirB/code for pdc_util; no bytecode is written), checks that the interpreted instruction has no
U+0008 and ends in `within \boxed{}.`, and prints the rendered base-model prompt:
`Question: <problem> If the problem does not provide enough information to determine the answer, output \boxed{unknown}.\n\nSolve the problem step by step and output the final answer within \boxed{}.\nAnswer: Let's think step by step.`

a11_launch.sh (one instance per host):
- It finds each checkpoint on its own host: /dev/shm/ckpt_b/<run>/final, then /run/ti/rsi_ckpt_archive/<run>/final, then a
  unique /run/ti/rsi_ckpt_archive/**/<run>/final with the exact run name (symlinks followed). R-Zero solvers are found under
  /run/ti/rzero/s<S>/ when A75_DONE is present, and the base model under /root/models.
- It de-duplicates cells across hosts with an atomic mkdir claim on NFS. The claim names the host, the launcher PID and the
  PID of the cell's subshell; a claim of the same host is taken over only when both processes are gone. Right after a claim
  succeeds, the cell is re-checked and skipped if another launcher completed it in between.
- It claims GPUs with `( set -o noclobber; : > /dev/shm/gpu_single_<g> )`, only when nvidia-smi shows < 1500 MiB twice, 15 s
  apart. It runs one probe per GPU and at most 4 per host (A11_MAXPAR), and releases the marker after every attempt.
- Each attempt is capped at 30 min (`timeout -k 60 1800`; about 4x the longest old dump). The hang watchdog's pattern does
  not match these probes. The probe runs with the host-lock descriptor closed (`9>&-`), so an orphaned engine process cannot
  keep the lock.
- It skips complete outputs (1,200 unique ids for UMWP, 1,552 for A6). It resumes a partial output after cutting a partial
  last line, with up to 3 attempts per launch, each on a newly claimed GPU.
- It writes a sidecar absfix_<tag>.a11.json with state, host, model path, a model fingerprint (size and sha256 of the first
  64 MiB of every file), the probe's sha256, and one event per attempt with the GPU index.
- It logs to the new file ext/a11_runs/a11_launch.log and to status_<host>.tsv.
- Stop conditions, checked before every GPU claim and every attempt: a STOP or STOP_<host> file under ext/a11_runs (STOP_ followed by the
  first 22 characters of the hostname, as in status_<host>.tsv; yield to the co-tenant; running probes finish, nothing new starts, and a probe killed by the co-tenant is not retried), the
  launcher's own stop flag (set by a signal), and the clock (no claim or attempt after T0 + 11 h; section 8).
- On TERM (`kill <launcher pid>`) the launcher first sets its stop flag, then sends TERM to every running probe (timeout
  forwards it to the probe's process group and sends KILL 60 s later) and to every cell subshell, and then waits, with no
  fixed sleep, until every child has exited. Each cell subshell waits for its own probe, releases its GPU marker, writes
  state "stopped" and releases its claim. Only after that does the launcher clean up (its remaining markers and claims) and
  exit 143. No attempt runs after the flag is set (a probe started in the instant the signal arrives is sent TERM at once), and no marker or claim is released while its probe runs (sandbox test,
  section 12).

a11_read.py re-implements no metric. It executes the registered read scripts unchanged (sha256 checked). Each one is split at
a fixed marker, the way a5_frontier_b.py already splits a5_frontier.py: definitions, then the loader is wrapped so that
abs_<tag> opens absfix_<tag>, then the body. The final json.dump line (for a5_frontier.py, also its os.makedirs) is not
executed, so no registered output is touched.
- umwp_read.py (sha 36a9f123…b51a): load, metrics, body (M, HX1-HX4). Gives ABS_U, ABS_A, ACC_A_abs, P1 and HX3.
- umwp_read_c.py (4240ae0f…b70c): gives HX6.
- a6_e_read.py (c1ab4803…c4ee): load, ns, body. Gives per set and seed ABS_U, and E2, which is P2 (section 6).
- a5_frontier.py (afdcafac…762a): definitions and the F-P1..F-P4 and descriptive body. Gives P3 from res["F_P1"] UMWP rows.
- a5_frontier_b.py (fba8ff2b…48da): ca_at and FB1-FB3, and the descriptive rows (TG arms included).

Abstention-prompt CA is the registered metrics() and a6 body with the plain-prompt slot pointed at the abstention-prompt
dump. That is the umwp_read L29 / a6_e_read L25 CA expression, applied as rz_read.py L64 does. The one new formula is ABS_A
on answerable MiP items: a6_e_read L26 with `not r["answerable"]` changed to `r["answerable"]`, marked "new formula". A
corrected dump is read only if the registered loader accepts it (1,200 unique ids; for a5_frontier exactly 600 U + 600 A
with 8 answers; 1,552 for A6) and its sidecar says "complete", its first sidecar event shows an empty dump, and every event
names the registered probe sha256. Rows of unknown origin are therefore never read. For every cell the read reports the
sidecar verdict and, separately, whether each registered loader accepts the file (state RUN, PARTIAL with the rejecting
loader named, or NOT RUN with the reason). It writes one JSON file itself, with O_EXCL, and refuses to start if that file
exists.

## 4. Commands

Register first, before any corrected data (from the repo root on either host):
```
cd /home/tione/notebook/rsi_rl && { echo "$(date '+%F %T') A11 corrected abstention-prompt re-run REGISTERED (registration + 2 probes + launcher + read), before any absfix data"; sha256sum ext/a11_draft/A11_registration_2026-09-26.md ext/a11_draft/umwp_probe_absfix.py ext/a11_draft/a6_probe_absfix.py ext/a11_draft/a11_launch.sh ext/a11_draft/a11_read.py; } >> docs/registration_hashes.txt
```
Read-only checks on both hosts: `bash /home/tione/notebook/rsi_rl/ext/a11_draft/a11_launch.sh check` and `... locate`.
Launch on both hosts at once, with the registered command `run` (= `run all`; both tiers). The cell claim prevents double
work for checkpoints present on both hosts:
```
for P in <HOST> <HOST>; do ssh -o BatchMode=yes -o ConnectTimeout=20 -p $P root@<SERVER> 'mkdir -p /home/tione/notebook/rsi_rl/ext/a11_runs && setsid nohup bash /home/tione/notebook/rsi_rl/ext/a11_draft/a11_launch.sh run all >> /home/tione/notebook/rsi_rl/ext/a11_runs/nohup_$(hostname | cut -c1-22).out 2>&1 < /dev/null &'; done
```
T0 is the epoch in ext/a11_runs/launch_epoch, written once (noclobber) by the first launcher that passes its preflight; its
value also appears in that launcher's "launch" log line. The launcher runs each probe as:
```
env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=<g> HF_HOME=/home/tione/notebook/rsi_rl/hf_home TOKENIZERS_PARALLELISM=false timeout -k 60 1800 /usr/bin/python3 /home/tione/notebook/rsi_rl/ext/a11_draft/umwp_probe_absfix.py --model <checkpoint> --tag absfix_<tag> --abstain_prompt 1 >> <dump dir>/absfix_<tag>.log 2>&1 < /dev/null 9>&-
```
(a6_probe_absfix.py for A6). Yielding: `touch ext/a11_runs/STOP` (all hosts) or `STOP_<host>`; stopping at once: `kill
<launcher pid>` (TERM). Never `kill -9` the launcher: its probes would survive in their own process groups. If it happens,
kill every *_absfix.py process on that host and remove its /dev/shm/gpu_single_<g> markers before relaunching; the
launcher also refuses to take over a claim or start a cell while a probe for the same tag is alive on the host. A cell left incomplete or stopped may be relaunched once with the same command, before T0 + 11 h
(after removing a STOP file, if one caused the stop).

Confirmatory read: run ONCE, as soon as both launchers have logged "launcher done", and at the latest at T0 + 12 h. Cells
that are not complete by then are NOT RUN. A read that happens after T0 + 12 h is still the one confirmatory read and is
reported as a deviation (it cannot change the data: nothing starts after T0 + 11 h and an attempt lasts at most 31 min). The read writes ext/a11_runs/a11_read.json itself and refuses to overwrite it:
```
/usr/bin/python3 /home/tione/notebook/rsi_rl/ext/a11_draft/a11_read.py
```
Immediately afterwards the author hashes its output into the registry (this is the author's step, not the script's):
```
cd /home/tione/notebook/rsi_rl && { echo "$(date '+%F %T') A11 confirmatory read output (a11_read.py, T0 + <h> h)"; sha256sum ext/a11_runs/a11_read.json; } >> docs/registration_hashes.txt
```
No read of the corrected dumps is run before it. Any other read of absfix dumps (another --out path, a second read) is a
deviation and is reported as such; the confirmatory result is the hashed a11_read.json.

Stale claims across hosts: a claim held by a launcher on the other host cannot be taken over automatically. If a host
becomes unavailable, its claims may be removed by hand only after `ps` on that host (or its loss) shows no a11_launch.sh and
no *_absfix.py process, with one log line per claim: `echo "$(date '+%F %T %Z') manual: removed claim <bench>__<tag> of
<host> (<reason>)" >> ext/a11_runs/a11_launch.log; rm -rf ext/a11_runs/claims/<bench>__<tag>`.

## 5. Outputs (new names only; the old abs_*.jsonl files are never opened for writing)

- dirA4/results/umwp/absfix_<tag>.jsonl, .log, .a11.json
- dirA5/results/a6_sweep/absfix_<tag>.jsonl, .log, .a11.json
- ext/a11_runs/: a11_launch.log, status_<host>.tsv, claims/, launch_epoch, nohup_<host>.out, a11_read.json

Row format is identical to the old dumps. About 0.25 MB per UMWP dump and 0.36 MB per A6 dump, under 20 MB in total.

## 6. Registered predictions (confirmatory, corrected prompt)

P1-P3 are corrected-prompt replications of directions observed under the defective prompt: their old-prompt cells were
seen before this registration (listed at the end of this section), and each prediction registers the direction the old
cells showed.

- P1 (UMWP, own vs clean with the abstention option): for each paired seed s0-s3,
  |ABS_U(own_a3 r3) - ABS_U(own_clean_a3 r3)| <= 0.05. ABS_U is umwp_read.metrics()["ABS_U"] / 100: the mean, over the 600
  unanswerable items, of the share of the 8 samples that abstain, in the corrected abstention-prompt dump. P1 holds iff all
  4 seeds pass.
- P2 (external sets): the already registered E2 rule of a6_e_read.py, applied verbatim to the corrected dumps. Cell test
  (L29): |ABS(own_a3 r3) - ABS(own_clean_a3 r3)| <= 0.05 for a set and seed, where ABS is L26 on the unanswerable items of
  that set. A set passes iff >= 3 of its 4 seeds pass (L31-32); P2 HOLDS iff >= 2 of the 3 sets (SUM, MiP-GSM8K,
  MiP-MATH) pass (L33-36). The verdict is taken from a6_e_read's own output on the corrected dumps: True = HOLDS, False =
  FAILS, None = NOT READ (the registered rule returns None when any of the 8 corrected A6 dumps is not read). The number of
  the 12 set x seed cells within 0.05 is reported descriptively; the paper's "in every seed on all three sets" wording
  (FR-4, H-E-02) is used only if that count is 12/12.
- P3 (F-P1 frontier dominance, UMWP, seeds 0-2): use the definitions of a5_frontier.py unchanged.
  - Guard default point: FA_A = mean abstention share over the 600 answerable items in the guard's (own_ag_a4 r3) corrected
    abstention-prompt dump; CA_U = share of the 600 unanswerable items whose plain-prompt numeric mode holds >= 0.5 of 8
    samples (guard's plain dump, unaffected).
  - Own primary frontier: the strict Pareto set of the registered label-free rule set on own_a3 r3's corrected
    abstention-prompt dump. The rules are the single sample, modal share tau in {2/8..8/8}, and abstention count k = 1..8.
    The frontier is linearly interpolated, flat below its smallest FA_A and undefined above its largest.
  - A seed holds iff own frontier(guard FA_A) <= guard CA_U - 0.05. P3 holds iff 3 of 3 seeds hold.
  - Mixed-prompt caveat (inherited from F-P1, RM-S4): the guard is scored with the plain prompt (CA_U) and its abstention
    prompt only for FA_A, while own is scored on its abstention-prompt frontier. P3 compares these two deployments; it is
    not a same-prompt comparison.
  - The registered F-P1 predicted the opposite direction (guard <= own - 0.05) and FAILED. P3 registers the direction the
    paper states ("the guard is dominated") with the same 0.05 margin. The F-P1 cell itself is recomputed descriptively.
    P3 tests the 0.05 margin only; ratio wording such as FR-1's "more than twice" is re-derived descriptively from the
    corrected numbers and is used only if the corrected numbers support it.
- All thresholds are inclusive with a 1e-9 tolerance. Three states for P1 and P3:
  - HOLDS: every required cell was read and passes.
  - FAILS: at least one read cell fails, whatever the other cells show.
  - NOT READ: no read cell fails, but at least one required cell is NOT RUN, incomplete, or has an undefined frontier.
  P2 has the three states of the E2 rule given above. NOT READ is never counted as HOLDS.
- Seen before this registration (defective prompt), computed by a11_read.py's comparison block:
  - P1 cells: -0.014 / -0.026 / +0.011 / +0.002.
  - P2: E2 HOLDS (every set 4/4); largest |gap| 0.047 (SUM s2); all other cells <= 0.036.
  - P3 margins (guard CA_U - own frontier): 0.387 / 0.270 / 0.243.
  All three would hold on the old dumps. No corrected dump exists.

## 7. Descriptive reads (no verdicts)

For every re-run tag, new, old and new - old:
- UMWP: ABS_U, ABS_A, ACC_A_abs, abstention-prompt CA (overall and by category), default point (FA_A, CA_U), CA_U at
  FA_A = 0.05 (a5_frontier_b ca_at) and the abstention-prompt Pareto set.
- A6: ABS_U, abstention-prompt CA, and ABS_A on MiP answerable items (new formula), per set.

Registered rules re-applied to the corrected dumps, reported beside their registered (old) labels: E2 per set (P2's inputs),
HX3, HX6, the F-P1 cells in the registered direction (twin cells are unchanged), F-P3, FB1, FB2, and the a5_frontier /
a5_frontier_b descriptive blocks (including the TG rows). Structurally incomplete by design, because their inputs are not
re-run: F-P2 and the F-P1 Phi exploratory rows (new value empty), the own r5 / own_ag r5 ABS values in umwp_read's metrics,
and a5_frontier_b's TG t4 s0/s2 rows (no dumps, as before). F-P4 is plain-prompt and is recomputed unchanged.

The read also recomputes the old reads and compares them with their stored outputs, leaf by leaf and by key set in both
directions. The pre-registration dry run (section 12) found 0 differing leaves. One key-set difference is known and
disclosed: the stored dirA4/results/umwp_read.json (written 24 Sep 07:29:04) predates abs_qwen3_4b_Base_rescue_abs_a4_s2_r3
(07:37:28), so it lacks that cell's ABS_U, ABS_A and ACC_A_abs: the stored file gives rescue_abs ABS values for s0-s1
only, and AB-2's s2 value does not come from it.

## 8. Clock, missing checkpoint, incomplete or stopped cell

- Clock. T0 = ext/a11_runs/launch_epoch (section 4). No launcher claims a GPU or starts an attempt after T0 + 11 h; with
  30 min per attempt, no probe runs after about T0 + 11.5 h. The confirmatory read runs once, at the latest at T0 + 12 h,
  and records T0, its own start time and whether it is within the cut-off. A cell whose sidecar is not "complete" at the
  read is NOT RUN, whatever happens later.
- A cell whose checkpoint is on neither host (status missing_here from both launchers) is NOT RUN. The read lists it with
  the launchers' reasons, and it goes in the paper appendix.
- A cell still incomplete or stopped after the one relaunch (3 attempts per launch) is NOT RUN in the same way.
- A prediction with a NOT RUN cell is NOT READ, unless another cell already FAILS it (P1, P3). The old defective number for
  that cell stays appendix-only and is never used as primary. No substitute checkpoint (another round or seed) is used.
- A dump resumed on the other host's copy of a checkpoint is flagged when the fingerprints differ.
- The R-Zero cells are A11 descriptive cells. They write only dirA4/results/umwp/absfix_rzero_s{0,1}_it3.{jsonl,log,a11.json}
  and the A11 files under ext/a11_runs; they never touch A7-5 files (A75_*_DONE markers, ledgers, rz_* outputs). They are
  therefore outside the A7-5 clock (A7-5 compute stops after 2026-09-27 14:10 CST) and change no A7-5 verdict (RZ-P2 is
  plain-prompt). rz_reread.sh's T72h / T84h re-reads load exact tags (plain_ / abs_) and are unaffected.

## 9. Downside, flagged before data

- P2 is now the registered E2 rule, so the false-FAIL risk is small: with per-cell pass probabilities of 0.90-0.95 (the old
  spread: SUM -0.029 to +0.047, MiP-MATH 52 items with paired SE 0.024-0.033), the chance that E2 fails with no true
  difference is about 1-3%. The risk moves to the wording: the chance that all 12 cells pass is only about 50-65%. If the
  descriptive count is below 12/12, the paper can no longer say "in every seed on all three sets" (FR-4, H-E-02) and
  reports the count.
- P1 is low risk (old largest gap 0.026 with SE about 0.007), unless the corrected instruction affects own and clean
  differently.
- P3 is low-to-moderate risk (old margins >= 0.24). An instruction that now asks for \boxed{} may raise boxed numeric answers
  for own r3 and lift its frontier. If P3 fails, the claim "the guard is dominated by a test-time prompt plus a label-free
  rule" (FR-1, H-ABS-03) is lost, and the remedies section says the guard is competitive at its operating point. FR-1's
  "more than twice" may not survive even if P3 holds.
- Every abstention-prompt number may move by several points, in an unknown direction. The numbers in RZ-27, RZ-34, FR-4 to
  FR-7, AB-2, UV-3, PR-12, H-ABS-01 and H-ABS-03 are replaced, and any qualitative reversal is reported.
- Not recoverable by this re-run: FR-2's "F-P2 HOLDS 5/5", the F-P1 Phi exploratory rows, and every Qwen3-8B and Phi-4-mini
  abstention-prompt number stay defective-prompt only and leave the main text (appendix, flagged).
[internal note removed]

## 10. Reporting rule

1. For every abstention-prompt statement on UMWP, SUM, MiP-GSM8K or MiP-MATH, the paper reports the corrected-prompt
   (absfix) number as primary. This covers ABS_U, ABS_A, abstention-prompt CA, frontier values, the E2 / HX3 / HX6 /
   F-P1 / F-P3 / FB1 / FB2 cells, and the R-Zero abstention-prompt numbers.
2. An appendix table gives, per cell, the defective-prompt number, the corrected number and the difference (new - old),
   from a11_read.py's descriptive block. It states the defect: file, line, and the backspace in place of "\b".
3. The registered verdicts E2, HX3, HX6, F-P1, F-P2, F-P3, FB1 and FB2 keep their labels, since they were read under the
   defective prompt, and are reported as a protocol deviation. P1-P3 are the confirmatory statements for the corrected
   prompt, described as corrected-prompt replications of directions observed under the defective prompt. The corrected
   re-applications of the old rules (other than E2, which is P2) are descriptive.
4. A statement whose dump is not re-run (section 2 list) or whose cell is NOT RUN is dropped from the main text or reported
   in the appendix only, marked "defective prompt, not re-run". This applies to F-P2, the F-P1 Phi exploratory rows and all
   Qwen3-8B and Phi-4-mini abstention-prompt numbers without exception.
5. Ratio and "every seed" wordings are re-derived from the corrected numbers: "more than twice" (FR-1) only if every
   corrected UMWP seed supports it, and "in every seed on all three sets" (FR-4, H-E-02) only if the descriptive P2 count
   is 12/12.
6. Every outcome is reported, including FAILS and NOT READ. No complete cell is re-run, and no corrected dump is discarded.
   Plain-prompt and twin numbers are unchanged and carry no flag.
7. Release: release_v1 must ship the absfix probes in place of the defective probe copy (release_v1/probe/code/umwp_probe.py
   L42 must be replaced before release), flag every abs_* dump as defective-prompt, and document the absfix_ label
   (corrected instruction; same items and settings). A rebuild with release_v1/build_release.py picks up absfix_* dumps
   through its glob, so the labels must be set before any rebuild.

Fact-to-cell map (factbase_2026-09-26 IDs; T1 / T2 = tier; "own", "clean", "guard" = qwen3_4b_Base own_a3 / own_clean_a3 /
own_ag_a4 r3). Rule 4 applies mechanically to every row marked "appendix".

| fact | abstention-prompt content | A11 cells | disposition |
|---|---|---|---|
| FR-1, SUM-1, PR-16 | F-P1 UMWP cells: guard FA_A, own frontier; "more than twice" | T1 own s0-2, guard s0-2 (UMWP) | corrected primary (P3; F-P1 re-applied); ratio re-derived (rule 5); twin cells unchanged; Phi exploratory clause appendix |
| FR-2 | F-P2 (5 lineages incl. Phi); F-P3 | F-P3: T1 own s0-2, guard s0-2 | F-P3 corrected (re-applied); Qwen UMWP FA_A values corrected; F-P2 verdict and Phi values appendix |
| FR-4, H-E-02 | UMWP ABS_U own / clean; E2 on SUM, MiP-GSM8K, MiP-MATH | T1 own s0-4, clean s0-3 (UMWP); T1 own, clean s0-3 (A6) | P1, P2; "every seed" wording by rule 5 |
| FR-5 | FB1, FB2 (UMWP) | T2 rescue_uv s0, s1, s2, s4, loop_uv r4 s0, s2; T1 own s0-2, s4 | corrected (re-applied); FB3 twins unchanged |
| FR-6 | UMWP CA_U@0.05: guard, clean, rescue_abs, TG ABS4, TG T4 | T1 guard s0-2, clean s0-3, own s0-3; T2 rescue_abs s0-2, tg_abs4 r4 s0-2, tg_t4 r4 s1 | corrected; twin values unchanged |
| FR-7, RZ-27, RZ-34 | R-Zero, base and own ABS_U, ABS_A, abstention-prompt CA | T1 rzero_s0_it3, rzero_s1_it3, base, own s0-4 | corrected |
| H-RZ-02 | R-Zero abstention clause; chat-template clause | T1 rzero s0-1, base | abstention clause corrected; chat-template clause is plain-prompt, unchanged |
| H-ABS-01 | ABS_U / ABS_A own, clean, base | T1 own s0-3, clean s0-3, base | corrected |
| H-ABS-03 | own frontier at guard FA_A; CA_U@0.05 own, clean, guard, rescue_abs, base | T1 own, clean, guard, base; T2 rescue_abs s0-2 | corrected |
| AB-2 | rescue_abs ABS_U and CA_U@0.05 vs own | T2 rescue_abs s0-2; T1 own s0-2 | corrected (stored umwp_read.json lacks s2, section 7) |
| UV-3, PR-12 | HX3: rescue_uv ABS_U vs own-based bars | T2 rescue_uv s0, s1, s2, s4; T1 own s0-2, s4 | HX3 re-applied (descriptive); HX4 plain, unchanged |
| PF-8 | HX6 ABS clauses (guard vs own) | T1 guard s0-2, own s0-2 | HX6 re-applied (descriptive) |
| H-GUARD, PR-09, H-A4C-RULE | guard vs own frontier (F-P1, F-P3; HX6 on s2) | T1 own s0-2, guard s0-2 | caveats re-derived from the corrected F-P1, F-P3 and HX6 (rule 1) |
| H-SETUP-02, H-SETUP-03, H-NEW-01 | probe description, defect disclosure | umwp_probe_absfix.py, a6_probe_absfix.py | text updated to name the absfix probes and the sweep (section 1) |
| H-X-UMWP | CLN1 vs RND1 (stated number is plain-prompt CA) | none | plain clause unchanged; any cln1/rnd1 ABS number appendix |
| H-UMWP-05 | Qwen3-8B (stated number is plain-prompt) | none | plain clause unchanged; any 8B ABS number appendix |
| H-UMWP-06 | Phi-4-mini (stated number is plain-prompt) | none | plain clause unchanged; Phi ABS_U / ABS_A appendix |
| twin and pool facts (H-ABS-02, AB-1, UV-2, UV-4 to UV-6, UW-1, PF-1 to PF-7, NG-*, DR-*, MECH-*) | abstention prompt from c0_probe.py, a3_guard.py or the pool scorers | none | unaffected (correct instruction; section 1 sweep) |

## 11. Compute

- Old durations from dirB/results/runner.log (24 Sep, "on gpu" to "finished"): UMWP abstention dumps took 3.95-5.42 min
  (mean 4.58 min, n = 27 Qwen3-4B dumps) and A6 dumps 6.82-7.42 min (mean 7.05, n = 8). The R-Zero dumps are assumed to
  take the UMWP mean, since rz_umwp.sh does not log to runner.log.
- Tier 1: 15 x 4.58 + 8 x 7.05 = 125 GPU-min, about 2.1 GPU-h. Tier 2: 15 x 4.58 = 69 GPU-min, about 1.1 GPU-h. Total
  about 3.2 GPU-h, and at most about 4.9 GPU-h with a 1.5x allowance for longer outputs and co-tenant restarts.
- Wall-clock is about 40-60 min at 4 GPUs per host; host A holds most cells (22 of the 38 cells exist only there).
- The read is CPU only, about 8-9 min.

## 12. What was done before hashing

- The launcher's `check` and `locate` modes (read-only) ran on both hosts (26 Sep 16:16 CST, and 16:40 CST for the final
  launcher): preflight ok, all five files NOT REGISTERED, all 38 cells located on at least one host, and `run tier1`
  refused. No ext/a11_runs directory and no GPU marker was created.
- Sandbox tests of the launcher on host A, in a copy under /tmp/a11_fix whose only changes are paths (log, claims, status,
  launch_epoch, dump dirs, GPU markers, lock and pid dir under /tmp; fake checkpoints), fake probes and a fake nvidia-smi,
  shortened sleeps, and one sandbox-only hook that completes a cell between the pre-claim check and the claim. Results:
  - Normal run of all 38 cells: 35 complete, 2 missing_here (two fake checkpoints were left out on purpose), 1 skipped by
    the post-claim re-check; the busy GPU 7 was never used; a crash that left a partial last line was cut and resumed on
    the next attempt (sidecar events running 0 ids / running 512 / complete 1,200, each with its GPU); no probe had fd 9
    open; a second launch started nothing.
  - TERM to the launcher with 4 probes running, one of them ignoring TERM (like a hung engine): three probes exited after
    their 3 s teardown and the fourth was killed by timeout's KILL 60 s later; the launcher exited only after that
    (t + 60.4 s). At no sample did live probes outnumber GPU markers, and no attempt started after the signal. Afterwards:
    0 probes, 0 engine children, 0 markers, 0 claims, pid directory removed, host lock free, the four sidecars "stopped".
    A relaunch resumed and completed them.
  - STOP file with 4 probes running, then a co-tenant kill of one probe's process group: no retry, no new cell, the other
    three finished, the launcher ended, nothing left behind.
  - Clock: with T0 + 11 h already past, nothing started; with every GPU busy, the four waiting cells stopped at the clock
    (state "stopped", 0 ids) and the launcher ended.
- a11_read.py ran on the real repo with no absfix dump present (output to /tmp, then deleted) and on a /tmp mirror where
  every absfix dump is a byte copy of its old dump with a valid sidecar. Results:
  - Real repo (473 s): all 38 cells NOT RUN and P1-P3 NOT READ on the corrected prompt. On the old dumps P1, P2 (E2, every
    set 4/4, 12/12 cells) and P3 hold with the cells listed in section 6. The five stored outputs were reproduced with 0
    differing leaves (umwp_read 231, umwp_read_c 12, a6_e_read 62, frontier_read 677, frontier_b_read 187 compared); the
    only key-set difference is the three rescue_abs s2 ABS keys (section 7). A second call with the same output path
    refused to start.
  - Mirror (466 s): 35 cells RUN; three planted defects came out as PARTIAL (a5_frontier's loader rejects a row with 7
    answers), NOT RUN (sidecar "stopped") and NOT RUN (checkpoint missing on both hosts). P1-P3, E2, HX3, HX6, F-P1, F-P3
    and FB1 equal their old values, and every re-run per-tag and per-set value has new - old = 0. F-P2 is NOT READ (the Phi
    lineages are not re-run), as registered.
- No GPU job was run, and no corrected dump, result file or registry line exists. The sandbox was deleted afterwards.
