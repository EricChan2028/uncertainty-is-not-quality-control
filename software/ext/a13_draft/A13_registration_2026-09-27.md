# A13 registration (draft of 2026-09-27): the same-size contrast without planted premise-deleted problems

Status: DRAFT. It becomes the registration when the lead author appends the sha256 of this file and of every file in
section 11 to `docs/registration_hashes.txt` (the drafting agent never writes that file). Nothing below may change after
the first launcher has written `ext/a13_runs/launch_epoch` (T0); a change before T0 is a new draft with new hashes.
No A13 data exist when this is registered: no R0 set has been trained on and no R0 model has been probed.

## 1. Purpose

Table 3, row "Same size" (paper sources `paper_arr_oct2026/sections/05_accommodation.tex`): Qwen3-4B, one round of 60
steps from each of the five round-2 solvers, on S (`cln1_a6`: the admitted proposals the full audit labels SOLVABLE plus
the admitted intact planted problems) or on R (`rnd1_a6`: a random subset of the admitted round-3 band of the same size).
Delta CA = CA(S) - CA(R) = -5.6 points, 95% t-interval over the five seeds [-9.0, -2.1] (UMWP, plain prompt,
confident-answer rate on the 600 unanswerable questions). R and S differ by 79-95 ill-posed proposer-written problems AND
by 20-39 admitted planted premise-deleted GSM-Symbolic problems, so the paper cannot say how much of the 5.6 points is due
to proposer-written ill-posed problems (Sec. 5, last sentence of "Most of the rise is sharpening ..."; Limitations,
"Attribution").

A13 adds one arm, R0: R with every planted premise-deleted problem replaced in place by a random admitted proposal, trained
exactly as R and S were. It splits the stored contrast into two seed-level parts that add up to it exactly:
E_prop = CA(R0) - CA(S) (no planted deletions in either arm) and E_plant = CA(R) - CA(R0) (the planted deletions).

## 2. Existing artefacts and exact reproduction of R (verified 2026-09-27, read-only)

- Band of seed s: `dirB/runs/qwen3_4b_Base_own_a3_s<s>_r3/admitted.jsonl` (593, 593, 593, 587, 586 lines).
- Labels: `.../own_a3_s<s>_r3/audit_admitted.json` overlaid by `dirB/runs/swap_audit_a5_s<s>/audit_parts/*.json`.
- Builder of R and S: `dirA5/code/a6_cln_build.py` (sha256 706b914e748a44ab...): S = SOLVABLE proposals + `tracer_intact`
  in band order; R = `random.Random(20260924 + s).sample(band, |S|)`. Launcher: `dirA5/code/a6_cln.sh` (sha256 b93fbb90...;
  the one edit visible on disk, made on 2026-09-24 with the previous version kept as `.bak_a6_cln.sh_1530`, concerns only
  GPU claiming and skipping finished evaluations; the training and probe commands are the same in both versions).
- R and S were rebuilt from the band with that rule for all five seeds and compared with the stored
  `dirB/runs/qwen3_4b_Base_{rnd1,cln1}_a6_s<s>_r3/admitted.jsonl`: byte-identical, same order, same id sets and same
  multisets of rows (every seed, both arms; checked twice on 2026-09-27 by independent scripts). The label map does not
  depend on the order of the part files. `a13_build.py` repeats this comparison (check C2) before it writes anything.
- Planted deletions = rows with `"kind": "tracer_deleted"`; R holds 32, 20, 39, 33, 26 of them (seeds 0-4), S none.
- Starting checkpoints: `/dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s<s>_r2/final` (the `model` string in all ten R/S
  `config.json`), a symlink into the host-local archive `/run/ti/rsi_ckpt_archive/`. Seeds 0, 1, 4 are on both hosts,
  seed 3 only on the host of port <PORT>, seed 2 only on the host of port <PORT>. Full sha256 of both weight shards and of
  config.json agree across hosts (recomputed independently on both hosts on 2026-09-27); the shards carry write times of
  2026-09-22 15:40-23:40 (written by `own_a3` round 2, before R and S were trained on 2026-09-24 09:51-20:23). The
  /dev/shm symlinks were created on 2026-09-24 15:22-16:23, when the finals were moved to the archive with their write
  times preserved; R and S therefore started from these files, but no hash was recorded at that time, so identity of
  content is inferred from the preserved write times and sizes, not checked.

| seed | shard 1 sha256 | shard 2 sha256 |
|---|---|---|
| 0 | 4e9746b85a6afa3036deaf288b851b5a86d8089d154a36e174b471d29cb86add | abd1f8b3d4f30dc5a11d949256bc5e44285baca559cbf00ebe619bd14e44fda3 |
| 1 | 79f96c0e57045619369b25ee079c8ee2e00e0f04507e1e50e42fc7792c0299bb | 2fdf751eb566ca8541f102d5ef764f32765dc4ec078a94506349e836ec6cf7c1 |
| 2 | b99d62e64ab2201c597202c21e1162626aa591899cfb94ba59033bd8944013be | fee99b8a605fa819e87f23a22e99857a48c3721d3377f273546203ba6a225e1d |
| 3 | 35599ddb0fc6cb786536ee3f64c02628cfd4447dd1af84de5fca76f1a7191fa9 | 637590e8dd94e35437ab02b6f4e667c86c2763e34cbca7df3f06238995fc2a1a |
| 4 | 32f011dc44ae99a8363dbde27c212beebee5a2c31ba7cea3b8bff8d683a5f394 | 25f4016c1d84390dc3e7ecbddc3a43765ece6a20e7d73bb5742a5af3f2f8ab4d |

  config.json of every start checkpoint: 0a7cb83af085ee2147ad20ad7e302df9f843263b32c13d47462033f5b6ddc79e.
- Training code: `dirB/code/b_train.py` sha256 fb9156b0ef6a685029c14c2ca52c4f17b6f0302a97c657c43cffeec1945d97d7
  (last modified 2026-09-24 06:36, before the R/S runs). Flags as in `a6_cln.sh`: `--arm own --seed <s> --steps 60
  --answer_fallback 1 --vllm_inproc 1`, every other flag at its default (qps 32, G 8, lr 1e-6, max_new 1024, temp 1.0,
  AdamW-8bit, ckpt_every 10, save_at 20,40, vllm_util 0.25); the R/S `config.json` files record exactly these values.
- Probes: plain UMWP probe `dirA3/code/umwp_probe.py` (sha256 32de59f9...), output `dirA4/results/umwp/plain_<run>.jsonl`;
  corrected abstention probe `ext/a11_draft/umwp_probe_absfix.py` (A11, sha256 11f796c2...), output `absfix_<run>.jsonl`;
  benchmark `dirB/code/b_eval.py` (sha256 5bf7c94a...). The R/S plain dumps were written by `a6_cln.sh` with
  `umwp_probe.py --model <final> --tag plain_<run> --abstain_prompt 0` (defaults: G 8, T 1.0, max_new 1024, vLLM seed 17,
  gpu_util 0.85); umwp_probe.py, b_eval.py, pdc_util.py and loto_common.py are older than the R/S runs (last modified
  2026-09-24 05:33 or earlier). The R/S runs have no corrected-abstention dump.
- Table 3 script: `paper_arr/analyses/attribution.py` (sha256 63bc30e8...), output `attribution.json` (sha256 6146256c...),
  pair "cln1-rnd1": CA diffs -0.0650, -0.0850, -0.0717, -0.0417, -0.0150; mean -0.0557; sd 0.0276; CI [-0.0900, -0.0214];
  bench +0.81. `a13_read.py --check_only` recomputes this from the ten plain dumps and the ten summaries and matches it to
  1e-12 (done 2026-09-27). The sha256 of those twenty input files and of attribution.json are pinned in `a13_read.py`.
- Software on both hosts: torch 2.8.0+cu128, transformers 4.57.1, vllm 0.11.0, bitsandbytes 0.48.1, flash_attn 2.8.1,
  installed 2025 in the image (unchanged since before the R/S runs); python3 = /usr/bin/python3.

## 3. Construction of R0 (a13_build.py)

For seed s: take R's lines in R's order. Let D be the line indices whose kind is `tracer_deleted` (k = |D|). The pool is
every band line of kind `proposed` whose id and whitespace-normalised problem text are not in R, in band-file order.
Replacements = `random.Random(20260927 + s).sample(pool, k)`; the i-th replacement goes to the i-th index of D in
increasing order. All other lines are copied byte for byte. Consequences: |R0| = |R| = |S|; R0 has no planted deletion;
since `b_train.py` visits items through `random.Random(seed * 1000 + epoch).shuffle(range(n))`, every non-replaced item of
R is visited at exactly the same steps and batch positions in R0, and each replacement takes the slot of the item it
replaces; passes per item are those of R and S (10.8-13.0).

Checks before anything is written (exit 1 otherwise): C1 registered sha256 of band, R, S, audit file and audit parts;
C2 R and S reproduced from the band byte for byte; C3 size, no deletion left, unique ids and texts, every line a band line,
non-deleted lines unchanged, k distinct replacements not in R; C4 the five R0 sha256 below. An existing different output
file is never overwritten (exit 2); an identical one is accepted.

| seed | n | k replaced | replacements (S/U/A) | ill-posed proposals R0 / R / S | solvable proposals R0 / S | intact planted R0 / S | R0 sha256 |
|---|---|---|---|---|---|---|---|
| 0 | 162 | 32 | 5 / 22 / 5 | 109 / 82 / 0 | 36 / 95 | 17 / 67 | a4cd93eb55911a77b8adfd6fb5a0ac41d6b28718cc2e348807863fb285e0e64c |
| 1 | 148 | 20 | 6 / 9 / 5 | 103 / 89 / 0 | 31 / 88 | 14 / 60 | 8b9a2f5723628edc226a2756b9369ce1ae0e831e284595c4c546b7c5bbcaa9b3 |
| 2 | 177 | 39 | 10 / 20 / 9 | 117 / 88 / 0 | 42 / 106 | 18 / 71 | f3c672c53c1b33a2cb37a18e78b8f555113aff28995ea77eb6302135a8c9c27b |
| 3 | 151 | 33 | 9 / 19 / 5 | 103 / 79 / 0 | 27 / 83 | 21 / 68 | ab04539c7d4f1d3b3423c2196d79362cb3acb85a4468974ea9d3e299d726338a |
| 4 | 166 | 26 | 6 / 12 / 8 | 115 / 95 / 0 | 25 / 84 | 26 / 82 | 52bb2cf1326833742a4f6a17c65c048cc705ae98e64bd0793d3404b0944e34f4 |

(S/U/A = audit labels SOLVABLE / UNSOLVABLE / AMBIGUOUS; ill-posed = U + A.) The replacements follow the band's label
mix, so R0 holds more ill-posed proposals than R. What E_prop compares is therefore: 103-117 ill-posed proposals in R0
against none in S, and, because S also holds every intact planted problem of the band, 14-26 against 60-82 intact
planted problems and 25-42 against 83-106 solvable proposals. The second difference is inherited from the R-S design; it
is stated in the pre-written text (section 7) and not removed by A13.

## 4. Training and probes (identical to R and S)

Run name `qwen3_4b_Base_r0_a13_s<s>_r3`, run directory `ext/a13_runs/runs/<run>/` (the set is written there by
`bash a13_launch.sh build`), checkpoints in `/dev/shm/ckpt_b/<run>/` on the training host.

    env -u PYTHONPATH -u HF_HUB_OFFLINE -u HF_DATASETS_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME=$R/hf_home \
      TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=<g> python3 $R/dirB/code/b_train.py \
      --model /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s<s>_r2/final --arm own --seed <s> \
      --out $R/ext/a13_runs/runs/<run> --data $R/ext/a13_runs/runs/<run>/admitted.jsonl --steps 60 --answer_fallback 1 --vllm_inproc 1
    python3 $R/dirA3/code/umwp_probe.py --model /dev/shm/ckpt_b/<run>/final --tag plain_<run> --abstain_prompt 0      (primary)
    /usr/bin/python3 $R/ext/a11_draft/umwp_probe_absfix.py --model /dev/shm/ckpt_b/<run>/final --tag absfix_<run> --abstain_prompt 1
    python3 $R/dirB/code/b_eval.py --model /dev/shm/ckpt_b/<run>/final --tag <run>_step60 --out $R/ext/a13_runs/evals

(the variables unset are unset in the hosts' environment anyway, so this is the environment of `a6_cln.sh`; core dumps off
as there). Tier 2 (descriptive): the corrected abstention probe on the ten existing R/S checkpoints
(`/dev/shm/ckpt_b/qwen3_4b_Base_{cln1,rnd1}_a6_s<s>_r3/final` or the archive; tags `absfix_qwen3_4b_Base_{cln1,rnd1}_a6_s<s>_r3`).
The launcher verifies before each training that the set and both start-checkpoint shards have the registered sha256, and
that b_train, both probes, b_eval, pdc_util, loto_common, the UMWP data and the three benchmark files have their pinned
sha256 and the packages their registered versions; in addition the script of every attempt (and, for a training, the set)
is hashed again when the attempt starts, an attempt whose script differs from its pin is not started, and the hash is
recorded in the attempt's sidecar entry. After a training its final is copied to the host-local disk archive (file sizes
verified, then renamed into place); the evaluations use the RAM-disk final and fall back to that copy.

Not identical, by nature: the 20-39 replaced items; GPU sampling and bf16 kernels are not bitwise reproducible, and the
rollouts of all items diverge once one batch differs; R and S were trained on 2026-09-24, R0 is trained on 2026-09-27 (same
image, same GPU model, same code). A preempted run resumes from its resume checkpoint as in b_train (four of the ten R/S runs
resumed at step 0); resumes are recorded and do not disqualify a run. For each resume the train sidecar and the read record
the step and whether the optimizer state was restored (b_train's preemption save carries it; its periodic saves every 10
steps do not, so a run killed without a TERM resumes with fresh AdamW moments). A seed whose training moves to the other
host (section 5) restarts there from step 0, since resume checkpoints are host-local.

## 5. GPU plan, time budget, co-tenant rules (a13_launch.sh)

One launcher per host (`bash a13_launch.sh run` on the hosts of ports <PORT> and <HOST>), cross-host claims by mkdir on the
shared volume. Train cells prefer a host: <HOST> takes seed 3 (its start checkpoint is only there), <HOST> takes seeds 2, 1,
0, 4 (seed 2's start checkpoint is only there; those of 0, 1, 4 are on both). The split follows the load seen on
2026-09-27 02:20-02:50: the co-tenant held 2-6 of the 8 GPUs of <HOST> with short per-video jobs and none of 10905. A
non-preferred seed is taken only if still unclaimed PREF_DELAY = 5 min after T0. A train cell takes its cross-host claim
only after it holds a GPU (if the claim fails, it releases the GPU at once), and gives the claim up when it has waited 5 min
for a GPU between two attempts (after a yield or a failed attempt; its resume checkpoint stays on the first host, and the
other host would start from step 0). A host without a free GPU therefore never holds a seed that the other host could
train: if <HOST> is loaded, seeds 0, 1, 4 move to <HOST> after 5 min, and vice versa. A seed's three evaluation cells run on
the host that trained it (host-local checkpoint). Tier 2 cells fill GPUs that no other cell waits for.

- GPU use: one job per GPU, at most 6 GPUs per host (12 of the 16). A GPU counts as free only with no marker, < 1500 MiB
  used, no foreign compute process and not cooling down, at two checks 60 s apart (idle > 1 min). A GPU cools down for
  10 min after a yield and after every check that finds a foreign compute process or >= 1500 MiB on it without a marker; the
  co-tenant's GPUs were idle in about 5% of 2-s samples on 2026-09-27 (gaps between its per-video jobs), and such a gap is
  thus not taken. Priority on a host: training and plain probe first, then corrected-abstention probe and b_eval, then Tier
  2. A cell takes a GPU only if more GPUs are free than cells of higher priority wait for one; a lower-priority cell is
  started only if a GPU is free for it, and a Tier-2 cell only when no training or plain probe of that host waits for a GPU.
  Budget: 5 x ~1 GPU-h training (R/S took 54-63 min from their last start), 15 probes x ~5-10 min, 5 b_eval x 15-30 min:
  about 8-10 GPU-h.
- The other 16 GPUs (hosts of ports <PORT> and <HOST>) are not used: they hold none of the start checkpoints, and the five
  single-GPU trainings run in parallel, so the critical path (one training, ~1 h, then its evaluations) does not shorten
  with more GPUs.
- Wall clock: ~65 min training in parallel + ~30-45 min evaluations (<HOST> evaluates four seeds on at most 6 GPUs), i.e.
  about 1.8-2 h from T0 if <HOST> has 6 free GPUs and <HOST> at least 1. Hard limits: no training attempt starts after T0 +
  2 h, no evaluation attempt after T0 + 3.5 h; the read runs when both launchers have exited, at the latest at T0 + 4.5 h.
  Lead-author time: review and hashing 0.5 h, build and launch 0.2 h, compute <= 4.5 h (expected ~2 h), read 5 min, paper
  edits 0.5-1 h.
- **Priority override (lead author, 2026-09-27, before T0; supersedes the co-tenant rule below for this run).** The lead
  author gave A13 top priority on all GPUs: A13 must not wait for or yield to other tasks. The launcher therefore runs with
  A13_PRIORITY=1 (the default): GPUs are still claimed free-first (the claim rules above are unchanged), but a foreign
  compute process found on a GPU that an A13 job is using (confirmed 30 s later) is logged (time, host, GPU, pid, command
  line) to ext/a13_runs/killed_foreign.log and killed (TERM, then KILL after 10 s), and the job continues without a
  yield. A foreign process that is not visible from the container cannot be killed; the job then keeps running beside
  it. The yield path below is kept for A13_PRIORITY=0 and is what the dev tests L1-L10 exercise; dev test L11 exercises
  priority mode. This changes only process handling, not training, probes, the read or any reading.
- Co-tenant priority (the other project may interrupt A13, never the reverse): a job whose GPU shows a foreign compute
  process (confirmed 30 s later) yields at once: TERM (b_train writes its resume checkpoint with optimizer state), KILL after
  20 s (120 s for training; each job's own timeout sends its KILL only 210 s after a TERM, so it never cuts this grace
  short), GPU cooled down 10 min for A13, the cell retries after 10 min; at most 12 yields and 3 failed attempts per cell.
  `touch ext/a13_runs/STOP` (or `STOP_<host>`) stops new work; `bash a13_launch.sh stop` stops a host's launcher cleanly.
  Per-attempt caps: training 100 min, probes 30 min, b_eval 60 min.
- Memory: each run keeps ~30 GB on /dev/shm until it finishes; then step20/step40/rollout move to the host-local archive
  (verified copy, as dirB's janitor does) and the final (7.6 GB) is copied there too; the RAM-disk final stays for the
  evaluations. Archive: > 5 TB free on both hosts. Shared volume: < 1 GB.

## 6. Read (a13_read.py; run once; writes ext/a13_runs/a13_read.json with mode "x")

6.1 Check (gates everything): CA of S and R recomputed from their ten plain dumps with the Table 3 formulas
(attribution.py: rows de-duplicated by id, exactly 600 + 600 rows, CA = share of unanswerable rows with mode not None and
p_mode >= 0.5), paired over seeds 0-4 as in attribution.py (mean +- t_{n-1,0.975} sd/sqrt(n), sd = statistics.stdev); it
must equal attribution.json paired "cln1-rnd1:CA" (mean, sd, CI, diffs; 1e-12), round to -5.6 [-9.0, -2.1], and the
twenty inputs and attribution.json must have their pinned sha256. Otherwise: NOT RUN. attribution.json and the twenty
inputs must therefore not be regenerated or touched before the read (a byte-identical regeneration is harmless).

6.2 A seed counts only if: `.trained` = 60; log.jsonl has steps 1-60 and the done line; config.json equals the seed's R
config.json in every field except `out` and `data`, with `data` = the run directory's admitted.jsonl of the registered
sha256; the train sidecar says complete, registered b_train, start checkpoint verified, and config.json unchanged since;
the plain dump is complete (600 + 600) with an A13 sidecar (state complete, registered probe, this run's final, output sha256
unchanged since completion). Missing any of these by the read time: that seed is NOT RUN.

6.3 Registered quantities on the accepted seeds (same formula as 6.1): E_prop = CA(R0) - CA(S); E_plant = CA(R) - CA(R0);
the identity E_prop + E_plant = CA(R) - CA(S) holds per seed (checked). Also reported in the Table 3 convention:
CA(S) - CA(R0) = -E_prop.

6.4 Readings of E_prop (95% CI in points), with at least 4 accepted seeds: ADDS if the lower bound > 0; REDUCES if the
upper bound < 0; NOT DETECTED otherwise (a bound exactly 0 counts as containing 0). NOT DETECTED has the registered sub-case
"estimate <= 0" (E_prop mean <= 0, unrounded), which also qualifies the claims row and the conclusion (7.2). Fewer than 4
accepted seeds: NOT RUN. With 4 seeds the reading is applied, the text says "four seeds", and the stored contrast it is set
against is CA(R) - CA(S) on the same four seeds.

6.5 E_plant is descriptive: its mean and CI are reported in Sec. 5 next to E_prop, and with its seed count in the appendix
(section 7), whatever the reading; it has no reading of its own and no significance wording, and its numbers never appear
in the abstract or the introduction (the REDUCES abstract and introduction state in words where the six points come from). Also descriptive: benchmark accuracy
(S - R0, R - R0; b_eval, seeds with an accepted summary; the Table 3 cell is "--" unless every accepted seed has one),
ACC_A on UMWP, and the corrected abstention prompt (ABS_U, ABS_A, CA_abs for R0, R, S on the seeds where all three have
an accepted dump; seed means and paired differences).

6.6 NOT RUN (whole A13): failed check 6.1, build manifest without the registered sets, or fewer than 4 accepted seeds.
A failed 6.1 is reported as "an input changed after registration" when the recomputation still equals attribution.json and
the paper's rounding but one of the twenty inputs or attribution.json has a different sha256, and as "not reproduced"
otherwise. A rejected seed counts as unfinished when all its reasons are of the kind "not finished" (no .trained = 60,
incomplete log, train sidecar not complete, plain dump without a complete sidecar or incomplete), and as failing a check
otherwise.
Then Sec. 5, Table 3 and the limitations stay as they are, and the appendix registers the analysis as not completed with
its reason (section 7.4). A crash of the read script itself may be fixed only by an amendment registered before the fixed
script runs, with the formulas unchanged.

## 7. Pre-written paper text

The read renders these with its numbers (points, one decimal, or two or three decimals for a value that would otherwise
print as 0.0; no "-0.0") as exact find/replace pairs on the sources in `paper_arr_oct2026/sections` (every find string
occurs exactly once there, checked 2026-09-27 on the sources last modified 01:06-01:19; the dev test P applies the rendered
edits of every reading and sub-case to a copy of those sources); the lead author applies them without rewording. If a find
string no longer matches because the surrounding text was edited, the replacement goes to the same sentence unchanged.
{e}, {lo}, {hi}: E_prop mean and 95% CI; {p}, {plo}, {phi}: E_plant mean (signed) and 95% CI; {rs}: CA(R) - CA(S) on the
accepted seeds (5.6 with all five; with four, "points with them" is followed by "on the same four seeds"); {k}/{n}: seeds
with a positive (ADDS, NOT DETECTED) or negative (REDUCES) difference / accepted seeds, in words ("in all five seeds", "in
four of five seeds", "in none of the five seeds"). {intact}: range over the accepted seeds of the intact planted problems in
R0 and S (14--26 against 60--82 for all five).

Replaced sentence (Sec. 5, end of the paragraph "Most of the rise is sharpening; the ill-posed curriculum adds to it."):
"The drifted curriculum thus adds about six points in one round to the sharpening any label-free curriculum shares, an
estimate that does not separate proposer-written from planted ill-posed ones."

Common parts: L = "With the random subset's planted premise-deleted problems replaced by other random admitted proposals, "
(S holds none by construction); B = "putting the planted problems back changes it by {p} points (95% CI [{plo}, {phi}])".

7.1 ADDS: "The drifted curriculum thus adds about six points in one round to the sharpening any label-free curriculum
shares. L one round on the subset still raises the rate over the solvable problems {in k of n seeds}, by {e} points on
average (95% CI [{lo}, {hi}]), against {rs} points with them; B. Proposer-written ill-posed problems thus raise the rate
by {e} points without any planted premise-deleted problem."
Introduction: "... the drifted curriculum adds about six points in one round, and still {e} points with the premise-deleted
problems we plant among its candidates replaced by model-written ones."
Limitations (replaces the paragraph "Attribution."): "The drift's own contribution to the solver's behaviour is estimated
from one-round, same-size contrasts over {n} seeds; without planted premise-deleted problems the random subset also holds
fewer intact planted problems than the solvable set ({intact}; Appendix A)."
Abstract, heading, claims table and conclusion: unchanged (the abstract has 201 words).

7.2 NOT DETECTED: "The drifted curriculum adds about six points in one round to the sharpening any label-free curriculum
shares. L one round on the subset changes the rate relative to the solvable problems by {+e} points on average (95% CI
[{lo}, {hi}], an interval that includes zero; higher {in k of n seeds}), against {rs} points with them; B."
Limitations: "The drift's own contribution to the solver's behaviour is estimated from one-round, same-size contrasts over
{n} seeds, which do not resolve how much of it proposer-written rather than planted ill-posed problems contribute (Sec. 5);
without planted premise-deleted problems the random subset also holds fewer intact planted problems than the solvable set
({intact}; Appendix A)."
Sub-case "estimate <= 0" in addition: claims table "Most of the rise is sharpening; at equal size the ill-posed curriculum
adds about six points, not detected without its planted problems"; conclusion "Sharpening turns these readings into one
stated assumption, mostly by itself and partly through the ill-posed curriculum, and benchmark accuracy reveals neither."
Otherwise the abstract, introduction, heading, claims table and conclusion are unchanged (the six-point same-size claim
itself is not affected).

7.3 REDUCES: "L one round on the subset lowers the rate below that of the solvable problems {in k of n seeds}, by {-e}
points on average (95% CI of the difference [{lo}, {hi}]), against a {rs}-point rise with them; B. The six points by which
the drifted curriculum exceeds the solvable problems in one round thus come from its planted premise-deleted problems, not
from its proposer-written ones."
Also: heading "Most of the rise is sharpening; planted premise-deleted problems add to it."; abstract "Most of this rise is
sharpening itself; the six points the ill-posed curriculum adds at equal training-set size come from premise-deleted
problems we plant among its candidates, not from model-written ones."; introduction "Most of this rise is majority-vote
sharpening, which any label-free curriculum shares; the drifted curriculum adds about six points in one round, which come
from the premise-deleted problems we plant among its candidates rather than from model-written ones."; claims table "Most
of the rise is sharpening; at equal size the ill-posed curriculum adds about six points, from planted premise-deleted
problems, not proposer-written ones"; conclusion "Sharpening turns these readings into one stated assumption, mostly by
itself rather than through the drift, and benchmark accuracy reveals neither."; limitations as in 7.1.

7.4 Every reading except NOT RUN also:
- Table 3, a row after "Same size": "Same size, no planted ({n}) & 148--177 (0) & $0$ & {S - R0 = -e} [{-hi}, {-lo}] &
  {Delta Acc S - R0, or -- unless every accepted seed has a b_eval summary}" (no dagger: registered); caption: "Same size,
  no planted: as Same size, with the random subset's planted premise-deleted problems replaced by other random admitted
  proposals." inserted before "Last row: ...".
- Appendix A, after the same-size description: "The same-size contrast without planted problems, registered before any of
  its new runs existed, replaces in place each premise-deleted problem of that random subset (20--39 per seed) with a random
  admitted proposal not already in it, leaving the size, the order of the other problems and every setting unchanged; its
  runs were trained three days after the others, and GPU sampling is not bitwise reproducible. Of its problems, 103--117 are
  ill-posed proposals (79--95 in the random subset) and 14--26 intact planted problems, against none and 60--82 in the
  solvable set. Putting the planted premise-deleted problems back in place of these proposals changes the rate by {p}
  points on average (95% CI [{plo}, {phi}]; higher in .. of .. seeds); this and the difference without them add up to the
  same-size contrast by construction." (ranges over the accepted seeds; with four seeds "... the same-size contrast on
  these four seeds ({rs} points) by construction.")
- "Further pre-registered analyses": "Five analyses ..., three before any of their data existed, the same-size contrast
  without planted problems before any of its new runs existed and the human annotation ..." and the list ends "..., a
  verifier on top of the band (...) and the same-size contrast without planted problems (ADDS: "without planted problems
  the rate still rises by {e} points" / NOT DETECTED: "not resolved at {n} seeds" / REDUCES: "the same-size difference
  comes from the planted problems"; Sec. 5)."
- One descriptive appendix sentence on the corrected abstention prompt, rendered by the read when all three arms have
  accepted dumps on at least two common seeds: "With the abstention option, one round on the solvable problems, on the
  random subset and on the subset without planted problems abstains on ..%, ..% and ..% of the unanswerable UMWP questions
  and on .., .. and ..% of the answerable ones (seed means, {m} seeds)."

NOT RUN: only the two edits of the "Further pre-registered analyses" paragraph, with the clause "not completed: <reason>",
reason one of "the stored same-size contrast could not be reproduced from its dumps", "an input of the stored same-size
contrast changed after registration", "the registered training sets were not built", "{m} of five runs finished in time,
four were required" (every rejected seed unfinished) or "{m} of five runs finished in time and passed the registered
checks, four were required" (a rejected seed failed a registered check). Sec. 5, Table 3, the abstract, the introduction,
the conclusion and the limitations stay as they are. The count of 136 predictions is unchanged in every case (A13 registers
readings, not predictions).

Page budget (not registered text): the main text currently ends exactly at the bottom of page 8. In every reading the Sec. 5
replacement is about four to five lines longer than the sentence it replaces; 7.1 adds about one line to the introduction,
7.3 about one to the introduction; Table 3 gains one row and its caption about one line. The lead author makes room by
trimming other main-text sentences, never by shortening the registered replacements. The abstract has 201 words; the
REDUCES abstract sentence is 10 words longer than the one it replaces, and the lead author then shortens other abstract
sentences if the venue's abstract limit requires it.

## 8. Power

The five R - S seed differences have sd 2.76 points. Assuming E_prop's seed differences have the same sd, the 95%
half-width with five seeds is 3.4 points (4.4 with four). Probability of ADDS (noncentral t, df 4) if the true E_prop is
0 / 1.5 / 2.8 (half of 5.6) / 4.0 / 5.6 points: 0.025 / 0.15 / 0.41 / 0.68 / 0.91 (four seeds: 0.025 / 0.12 / 0.30 / 0.51 /
0.77); the effect detected with 80% probability is 4.6 points (5.9 with four seeds, or with five seeds at sd 3.5). A false
REDUCES has probability 0.025 if the true E_prop is 0 and below 0.002 if it is 1.5 or more. A13 can therefore show that
proposer-written problems carry most of the 5.6 points, but a contribution of about half of it is more likely than not to
read NOT DETECTED. REDUCES becomes likely once the true E_prop is negative: its probability is 0.09 at a true E_prop of -1
point and 0.24 at -2 (four seeds: 0.08 / 0.18). The NOT DETECTED sub-case "estimate <= 0" has probability 0.48 at a true
E_prop of 0, 0.11 at 1.5, 0.01 at 2.8 and 0.70 at -1. With the true E_prop equally likely to be 0, 1.4, 2.8, 4.2 or 5.6
points: ADDS 0.44, NOT DETECTED 0.55 (sub-case 0.12), REDUCES 0.005, and an observed estimate below 1.5 points 0.32.

## 9. Downside (stated before any data)

A13 cannot guarantee that the paper's standing does not drop: its result can weaken Sec. 5. What it guarantees is that
every outcome, including the unfavourable ones, has its text fixed now. The probabilities below are those of section 8
(sd 2.76 assumed, five seeds unless stated); the true E_prop is unknown.
- Unchanged in every reading: the existing Table 3 rows and every existing number (A13 reuses the stored R/S dumps and
  checks them against their hashes), the six-point same-size claim itself (R - S) in the abstract, the introduction and the
  claims table, and all drift results (Secs. 3-4).
- ADDS: strengthens Sec. 5, adds one clause to the introduction and removes one limitation. Probability 0.41 if
  proposer-written problems carry half of the 5.6 points (true E_prop 2.8), 0.68 at 4.0, 0.92 if they carry all of it,
  0.03 if they carry none (four seeds: 0.30 / 0.51 / 0.77 / 0.03).
- NOT DETECTED with a positive estimate: Sec. 5 reports an interval for E_prop that includes zero, next to E_plant; the
  six-point claim stays in the abstract, the introduction and the claims table, but its attribution to proposer-written
  problems is now explicitly open, and a reader may take it as "the proposer's part could be nil". This is not a mild
  outcome for a claim that appears in the abstract. Probability 0.58 at a true E_prop of 2.8, 0.73 at 1.5, 0.48 at 0.
- NOT DETECTED with an estimate <= 0 (registered sub-case): in addition the claims row says that the six points are not
  detected without planted problems, and the conclusion replaces "partly through the drift" by "partly through the
  ill-posed curriculum". With E_plant then typically clearly positive, reviewers will read this as "the six points come from
  the planted problems" although REDUCES is not reached. Probability 0.48 at a true E_prop of 0, 0.11 at 1.5, 0.70 at -1.
- REDUCES: moves the six points from the proposer's drift to the planted premise-deleted problems, which are the study's own
  monitoring instrument: the abstract, the introduction, the Sec. 5 heading, the claims table and the conclusion change,
  and the message "the drifted curriculum adds" is weakened. This is the outcome most likely to lower the paper's standing.
  It needs the upper bound of E_prop below 0, i.e. R0 answering confidently less often than S by more than about 3.4 points
  at the assumed sd. Its probability depends on the unknown true E_prop: 0.025 at 0 and 0.002 at 1.5, but 0.09 at -1 and
  0.24 at -2. A negative true E_prop is plausible: if the UMWP-like planted deletions carry the whole effect and the 60-82
  intact planted problems of S (14-26 in R0) themselves raise S's rate.
- Day effect: R0 is trained on 2026-09-27, R and S were trained on 2026-09-24 (same image, GPU model and code). A
  systematic day effect would enter E_prop and E_plant unseen and could produce a false REDUCES or the sub-case above. The
  only check is a same-day re-run of R; A13 does not include it, because it re-measures R and so puts the published -5.6
  at risk. The appendix sentence states the three days.
- NOT RUN (co-tenant, crash, deadline): only one appendix clause changes.
- Our subjective guess, not a measurement: ADDS 0.35-0.5, NOT DETECTED 0.45-0.6 (of it the sub-case about 0.1), REDUCES
  0.02-0.07.
- Not running A13 at all leaves the paper as it is, with the confound stated in Sec. 5 and the limitations. Whether to run
  it is the lead author's decision, taken before T0 with this list in view.

## 10. Order of operations for the lead author

All commands from `/home/tione/notebook/rsi_rl` on either host unless stated (the hosts share this volume).
1. Read this draft; hash this file and every file in section 11 into `docs/registration_hashes.txt`:
   `sha256sum ext/a13_draft/A13_registration_2026-09-27.md ext/a13_draft/a13_build.py ext/a13_draft/a13_launch.sh
   ext/a13_draft/a13_read.py ext/a13_draft/dev/a13_dev_test.sh ext/a13_draft/dev/a13_dev_tree.py >> docs/registration_hashes.txt`
   Never start `a13_launch.sh run` for practice: the first launcher that passes the preflight writes T0
   (`ext/a13_runs/launch_epoch`) once, for good, and every deadline is counted from it.
2. `bash ext/a13_draft/a13_launch.sh build` (once, either host; writes the five sets and the manifest).
3. `bash ext/a13_draft/a13_launch.sh check full` on each host (read-only; verifies the start checkpoints there).
4. On both hosts, <HOST> first and <HOST> within 5 min (each ssh returns at once; the launcher survives it):
   `timeout 30 ssh -n -p <port> root@<SERVER> 'setsid nohup bash
   /home/tione/notebook/rsi_rl/ext/a13_draft/a13_launch.sh run >
   /home/tione/notebook/rsi_rl/ext/a13_runs/nohup_$(hostname | cut -c1-22).out 2>&1 < /dev/null &'`
   (ports <PORT> and <HOST>; no `cd ... &&` before it, which would keep ssh open until the launcher exits). A second launch
   on the same host is refused by its lock. Progress: `bash ext/a13_draft/a13_launch.sh status`.
5. When both launchers have logged "launcher done" (or at T0 + 4.5 h at the latest, after `touch ext/a13_runs/STOP` and
   `bash ext/a13_draft/a13_launch.sh stop` on both hosts): `/usr/bin/python3 ext/a13_draft/a13_read.py` (once); hash
   `ext/a13_runs/a13_read.json` into the registry (`sha256sum ext/a13_runs/a13_read.json >> docs/registration_hashes.txt`);
   apply `paper.edits` of that file.

## 11. Files

All in `ext/a13_draft/`. This file's own sha256 goes only into `docs/registration_hashes.txt`.

| file | sha256 |
|---|---|
| a13_build.py | ab6ef460d74b81df036d83b3d9fba9c857f597077eac17348d8b7726de33c4ae |
| a13_launch.sh | 76307f01f48aae070a499f6e7d1e002dadec8c45b46fb9bf76dbb64e6714bdb8 |
| a13_read.py | 59304a6389fe8040d679d7d81dfaa380096a87a43d259a0dc44bcf2d452f5f80 |
| dev/a13_dev_test.sh | 8c46999a5ee80c6a47bc7be1f73ac372075852f61ced10913c5faad2974741e7 |
| dev/a13_dev_tree.py | ff2ced4803c3d86a3d307219bc964fa841198892469f6883787a8e0e80e84db8 |

Dev tests (synthetic trees under /tmp, no GPU, nothing written outside WORKDIR):
`bash dev/a13_dev_test.sh /tmp/<WORKDIR> [<copy of paper_arr_oct2026/sections>]`.

## 12. Dev-test record (revision of 2026-09-27, host of port <PORT>)

Run from `ext/a13_draft/` with exactly the files of section 11 and this file:
`bash ext/a13_draft/dev/a13_dev_test.sh /tmp/a13_fix/dev_final /tmp/a13_fix/paper_sections` (the second argument: a copy
of `00_abstract.tex`, `01_introduction.tex`, `05_accommodation.tex`, `08_conclusion.tex`, `09_limitations.tex`,
`11_appendix.tex` as last modified 01:06-01:19 on 2026-09-27). Result: 44 passed, 0 failed.
- B1-B5 build: idempotent; invariants and determinism; refuses to overwrite a different set (exit 2); a C2 failure writes
  nothing (exit 1); the registered input hashes are enforced (C1).
- L1-L9 two simulated hosts: both launchers exit 0; every cell complete, evaluations on the training host; a co-tenant
  process on a training GPU causes a yield, and the run resumes from its checkpoint with the optimizer state restored; the
  script sha256 is recorded at every attempt; a crashed training is retried (resume from step 0, optimizer not restored,
  recorded); a crashed probe's partial last line is cut and the probe resumes; markers, claims and pid directories cleaned
  up; step20/step40 moved to the archive and the final copied there; GPUs with co-tenant memory never used; nothing written
  to the real /dev/shm.
- L10 starvation: all 8 GPUs of host A held by visible co-tenant processes (640 MiB), host B with GPUs 6 and 7 free, and the
  adverse preferences of the first draft (A: 3 0 4; B: 2 1). Host B trains seeds 0, 1, 2 and 4 and evaluates them, host A
  never holds a seed claim, seed 3 ends not_run_deadline on A without a sidecar, B's co-tenant GPUs are never used, and on B
  no oabs cell starts before the last training started or before the first plain probe started.
- S1-S3 STOP file starts nothing; `stop` during training leaves a resumable checkpoint and a clean host; a restart resumes
  and completes.
- D1-D10 read: ADDS equal (1e-12) to an independent computation; a second read is refused and changes nothing; NOT DETECTED
  with a positive estimate (7 edits) and with an estimate <= 0 (9 edits: claims row and conclusion qualified); REDUCES (12
  edits); NOT RUN with three unfinished seeds and with three seeds after rejected configs (the two reasons); a failed R-S
  check ("could not be reproduced") and a changed input of it ("changed after registration") each blocking every reading;
  a changed config.json (four seeds: the text sets E_prop against R - S on the same four seeds) and an edited dump each
  rejecting their seed; a CI bound of +0.02 printed with two decimals; the tree restored.
- P (paper): the rendered edits of ADDS, both NOT DETECTED sub-cases, REDUCES, ADDS with four seeds, the rounding case and
  two NOT RUN cases apply to the sources, each find string exactly once, braces and $ balanced, no unrendered placeholder.
- X1-X5 real data, read-only: `a13_read.py --check_only` reproduces the stored R - S contrast from the ten plain dumps
  (diffs -0.0650, -0.0850, -0.0717, -0.0417, -0.0150; CI [-0.08997, -0.02136]; rounded -5.6 [-9.0, -2.1]; equal to
  attribution.json; the twenty inputs and attribution.json unchanged; bench +0.8); a dry build into /tmp gives the five
  registered R0 sha256; `check` and `locate` of the launcher are read-only; `ext/a13_runs` does not exist.
- Also checked on 2026-09-27: the launch form of section 10.4 returns at once (tested with `sleep 20`: 0 s; the earlier
  `cd ... && setsid nohup ... &` form held ssh for 21 s).


Priority revision (2026-09-27, before T0): a13_launch.sh gained A13_PRIORITY (default 1: log and kill foreign processes on
A13 GPUs, never yield); dev tests L1-L10 run with A13_PRIORITY=0; new L11 (priority mode: a foreign process on the
training GPU is logged in killed_foreign.log and killed, the job completes with 0 yields). Final run on the host of port
<PORT> with the current paper sections: 45 passed, 0 failed.
