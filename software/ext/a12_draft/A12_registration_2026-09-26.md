# A12 registration: where the R-Zero drift acts, whether the R-Zero solvers improve, and a text-based verifier on top of the band

Status: DRAFT, written 26 Sep 2026 (CST) in `ext/a12_draft/`, revised the same day after two reviews (section 9). Not
registered yet. It counts as registered only once this file and every file in section 6 have their sha256 in
`docs/registration_hashes.txt`, before any A12 data exist. After that, a change to any hashed file needs a dated amendment
appended here before the data it affects. Hashed code is never edited once its data exist.
No A12 data existed when this draft was written or revised. To design it, only these were used: the registered pool counts in
`ext/rzero_runs/s<S>/*_summary.json`; the length distribution of the raw questions (character counts only, to size the judge
batches); the number of raw questions containing U+2028, U+2029, U+0085 or CR (1 of 44,629); and record field names. No
raw question text was read. For the revision, read-only checks of tokenizer files (the chat template of the base model and of
all six solvers is the same, sha256 87a2728c...) and of data provenance (section 2.1) were added; they involve no A12 data.

A12 has three parts. **All three are descriptive.** Each has readings stated in advance, and no pass/fail threshold except
the natural ones given below. None changes a registered verdict (RZ-P1, RZ-P2, RB-P1..P3, J2-P1..P2, A4 or A9).

Part 1 addresses one question. The admitted-set drift of R-Zero (registered RZ-P1: +23.0 / +27.0 points; blind A7-5d: +22 / +26)
could come from two places: downstream selection (pseudo-labelling and the 0.3-0.8 uncertainty filter), or the challenger's own
output. Which is it?
Part 2 addresses a second. The drift is measured in our replication of R-Zero. Does that replication's solver show the benchmark
gains the R-Zero paper reports?
Part 3 addresses a third. Suppose a text-based well-posedness verifier is added on top of the band on the fixed pool. Does it
lower the admitted ill-posed share, and is its effect on the drift what Corollary 1 of the mechanism note predicts?

## 1. Part 1: blind audit of RAW R-Zero challenger questions

### 1.1 Population and sample (`a12_p1_sample.py`)
RAW, for seed S (0, 1) and iteration i (1, 2, 3), means every parsed question the questioner Q_i wrote in the 8 generation
shards of iteration i. These are the records with `score == 0` in
`/run/ti/rzero/s<S>/generated_question/rzero_qwen3_4b_Base_s<S>_solver_v<i>_<k>.json.raw`, k = 0..7, taken in shard order. The
A7-5 patch P14 kept these files, which R-Zero deletes. A record with score -1 is not a question: it has no `<question>` block or
no boxed answer. Records with score -1 are excluded. From the registered summaries (8,000 generations per cell) they make up
1-15% of generations: 6.3 / 11.5 / 14.5% (s0, it1 / it2 / it3) and 6.4 / 2.5 / 1.0% (s1). The exclusion therefore moves in
opposite directions with iteration in the two seeds, and the per-cell parse-failure rate is a registered covariate (1.6).
Nothing downstream of the questioner has acted on this population. That includes pseudo-labelling by S_(i-1), its drops, and the
0.3-0.8 filter. The registered pool summaries give the RAW sizes:

| | it1 | it2 | it3 |
|---|---|---|---|
| s0 parsed / scored / admitted | 7,499 / 6,859 / 3,678 | 7,083 / 6,290 / 3,817 | 6,840 / 6,379 / 4,271 |
| s1 parsed / scored / admitted | 7,489 / 6,593 / 3,423 | 7,801 / 6,033 / 3,697 | 7,917 / 7,391 / 4,941 |

The share of parsed questions dropped while pseudo-labelling (parsed - scored) is 8.5 / 11.2 / 6.7% in s0 and 12.0 / 22.7 /
6.6% in s1; the 22.7% of s1 it2 looks like a killed shard and is reported as a covariate.

The files are host-local, so `stage` runs on the host that owns each seed (`ext/rzero_runs/s<S>.host`): seed 0 on `-p <PORT>`
and seed 1 on `-p <PORT>`. Everything it writes is CPU-only output to NFS. Per iteration, `stage` does four things:
(a) It requires the counts to equal the registered summaries: 8 shards, n_parsed, n_scored, n_admitted, and the admitted file's
sha256.
(b) It gives every raw question a downstream status, used only for secondary reads. It aligns the scored pool
(`..._all_scored.json`) onto the parsed questions in order. This works because `upload.py` concatenates shards 0..7, and
`evaluate.py` keeps each shard's parsed order. The status is one of:
- `not_scored`: dropped while pseudo-labelling. The cause is either no boxed answer in 9 samples of S_(i-1), the "证明" / "box" /
  "text" rules, or a killed shard.
- `scored_rejected`.
- `admitted`: the filter `0.3 <= score <= 0.8`, with label not '' / 'None', re-applied to the aligned pool. The derived
  admitted rows must equal `..._admitted.jsonl` row by row.

If the alignment or that equality fails, status comes from text membership instead and is flagged (`status_method: "text"`).
The primary outcome does not use status.
(c) It draws 200 positions with `random.Random(2026092612 + 10 * S + i).sample(range(n_parsed), 200)`. A repeated question
counts by multiplicity, as in A7-5.
(d) It writes `dirA5/results/a12_key/stage_s<S>.jsonl` and `stage_s<S>_meta.json`, and never redraws.

`build` pools the 1,200 items and shuffles them with `random.Random(20260926 + 12)`. It assigns neutral keys y0000..y1199, where
key yNNNN is line NNNN + 1. The letter y avoids the A7-5d z-keys (and the pilot uses q, 1.5). `build` writes two files:
- `dirA5/results/a12_blind/items.jsonl`, the only file judges see. It holds key and problem only, in UTF-8 with
  ensure_ascii=False, as A7-5d did. U+2028, U+2029 and U+0085 are written as JSON escapes, so each item is exactly one line
  for `sed`. The decoded text is unchanged, and the build checks this item by item.
- `dirA5/results/a12_key/key_full.jsonl`, the key file. It sits in a separate directory that judges are never pointed to.

The build aborts, and nothing is judged, if any 25-line batch exceeds 25,000 characters. The judges' fetch must fit in one
tool result, whose limit is about 30,000. In 2,000 simulated random batches the largest was 14,225 characters. Right after the
build, and before any judging, the sha256 of `items.jsonl`, `key_full.jsonl` and both stage files are appended to the hash log.

### 1.2 Judging (`a12_p1_blind_wf.js`) - the A7-5d workflow with four changes
The workflow is `a75d_blind_wf.js` (registered, sha256 711d4f0c...) unchanged except for:
1. the items file;
2. the key letter y;
3. the framing sentence (1.3);
4. a pilot mode on dummy items with key letter q (1.5).

Everything else is identical: 48 Opus judges x 25 consecutive lines, model opus with effort xhigh, and a structured schema whose
keys are restricted to the judge's own 25. The frozen definitions are verbatim: SOLVABLE / UNSOLVABLE / AMBIGUOUS, "Be strict: a
default assumption you would have to invent makes it UNSOLVABLE", confidence, and note <= 20 words. The operational lines are
also verbatim: fetch by line range only, no other file, no server changes, local computation in a temp directory.
The first line, "auditing self-generated training questions", is kept: every raw question was generated as a training
candidate. Stub-agent tests in node (`dev/wf_stub_test.mjs`) check the prompt: the rendered pass-1 prompt differs from A7-5d's
in the framing sentence only, and the file path differs as a value. The tests also check the args validation (pass 1: 48
batches, keys y0000..y1199; passes 2-3: whole 25-line batches only; pilot: 2 batches, keys q0000..q0049 only).

### 1.3 The framing sentence (a registered condition difference)
The frozen sentence reads "These were written by a Qwen3-4B-Base language model and ADMITTED by a self-play filter." It is false
for every raw item the filter did not admit. From the counts above, that is 51 / 46 / 38% of the parsed questions in seed 0 and
54 / 53 / 38% in seed 1.
- Keeping it would make the prompt identical to the admitted audits. It would also tell the judge something untrue, and could
  push its prior toward SOLVABLE on exactly the items that matter.
- A sentence announcing "unfiltered" generations would push the prior the other way.

A12 therefore uses **"These were written by a Qwen3-4B-Base language model in a self-play loop."** It is true of every raw item
and says nothing about filtering. The checker aborts on any judge task text that lacks it (1.4). It is the only prompt
difference from the A7-5d blind audit. The original admitted audits differed from A7-5d, and so from A12, in the further ways
listed in the A7-5d addendum (`dirA5/docs/A7_5d_addendum_2026-09-26.md`, the paragraph on what RB-P2 tests): knowledge of run
names and ids; batches of a single run instead of mixed batches; raw text instead of JSON lines; a file push instead of a
structured return; different operational prompt lines; it1 and it3 judged about a day apart, with a different memory index and
harness version; and, for the original it1 judges only, a relayed user request. **A7-5d blind is therefore the primary reference
for the condition check; the original audits are the secondary one.** Every comparison states the framing difference.
What it affects:
- It does not affect the iteration contrast D_raw. All 1,200 items share the prompt and are shuffled across iterations.
- It does affect the comparison of raw levels with the registered admitted audits. Two controls are built in:
  - The within-sample comparison is framing-free. Raw items that were admitted and raw items that were rejected are judged
    together, under the same sentence.
  - A condition check. The in-sample admitted items are a uniform sample of the same admitted sets the registered audits
    sampled. For each iteration, their ill-posed share is computed per seed and averaged over the two seeds (the references
    are 50/50 by seed, the in-sample admitted items are not), and the A7-5d blind share (averaged over seeds) is subtracted.
    The mean of the three per-iteration differences gets a bootstrap CI from the part-1 replicates (in-sample items resampled
    within seed x iteration; the A7-5d cell shares, available only as shares of 200, redrawn as binomial). **The condition flag
    is raised when this CI excludes 0.** The same difference against the original audits (item-level, resampled) and the
    per-iteration differences are reported descriptively. With the flag raised, raw-vs-registered level comparisons are
    reported with it.

### 1.4 Blindness check (`a12_blind_check.py`) - the A7-5d checker, derived mechanically
`dev/dev_mk_check.py` produces the A12 checker from the registered `a75d_blind_check.py` (sha256 e42e8905...). Each replacement
must match exactly once. The changes are:
- the items file, plus a pilot file with 2 batches;
- the key letter y (q in the pilot);
- four names added to the ban list (`[/\\_]a12`, `generated_question`, `all_scored`, `.json.raw`);
- raw-sample fields added to the leak list (`a12s<S>i<i>k<k>` ids, `"status": "admitted"`, `"raw_id"`, `"pool_score"`,
  `"admitted":`, `scored_rejected`, `not_scored`);
- an abort when a judge's task text lacks the registered framing sentence (1.3);
- for every structured return, a record of whether the harness accepted it (no error result), so that the read can tell a
  null return from a batch missing from the saved verdicts (1.5);
- the fields `checker`, `items_file` and `pilot` in the output.

The rules are A7-5d's, verbatim. Every workflow subagent must be a judge with exactly one registered fetch, on port <PORT> or
10905. A batch passes only if all four hold:
- a full fetch of its own 25 keys;
- only allowed calls: local computation in a temp directory, and no project, key, audit, transcript or network access;
- no key or audit information in any non-fetch result;
- every turn ran as claude-opus-5-5 / xhigh.

Neutral and absent transcripts are handled as in A7-5d. A self-test on synthetic transcripts, whose task text is rendered from
the workflow's own template, covers: clean, banned call, leak, allowed local computation, a null structured return, a task text
without the framing sentence, the pilot's q keys (a pilot fetch of y keys is dropped), and wrong model.
Usage: `python3 a12_blind_check.py <out.json> all|pilot|<ranges json> <transcript dir> [...]`, run locally right after each pass.

### 1.5 Pilot, pre-launch scan, passes
**Pilot (after this revision, before hashing; run by the lead author).** `a12_p1_pilot_build.py`, run on the server, writes 50
dummy templated arithmetic items with no R-Zero or project content and keys q0000..q0049 to
`dirA5/results/a12_pilot/items.jsonl`. The letter q is not the real items' y, so no pilot file ever holds A12 keys and the
y-key scan below is not tripped. The same workflow runs with `{pass: 1, pilot: true}` (2 judges) and the same checker runs with
`pilot` on the local transcript directory. Both batches must pass on real transcripts, or the defect is fixed before hashing.
Pilot verdicts are never read, and the read refuses any pilot check file. `dirA5/results/a12_pilot/` is pre-hash, non-study
output: after the pilot, its items file, the pilot check file and the workflow output are deleted from the server and from
the launching machine, and the deletion is recorded in the hash-log entry of the package, together with the pilot check's
sha256 and its passed / dropped lists.
**Pre-launch scan.** This is A7-5d's scan, with patterns adapted. Immediately before pass 1, every local copy of R-Zero
item-level data is removed from the launching machine. The Temp tree and the working directory are then scanned with
`grep -rlaE 'rz[01]i[1-3]r[0-9]|a12s[01]i[1-3]k[0-7]j'`, which must find nothing. The scan also looks for files holding 25 or
more distinct y-keys (`\by[01][0-9]{3}\b`); there must be none apart from this package. For both greps, exit 1 (no match) is
a pass. Exit 2 (a path could not be read) with no match counts as a pass only when the unreadable paths are listed in the scan
output (grep's error lines are kept, not discarded). The path of the judges' scratchpad
(the session's temp directory) is tested against the checker's BAD_ANY pattern and must not match it, or every local
computation would count as a banned call. The `.claude` tree is skipped by the scan and banned for judges; it holds the
pre-hash review transcripts, one of which printed an R-Zero admitted item and two original audit ids. All three outputs go to
the hash log, together with the sha256 of MEMORY.md. **Until the part-1 read is done, no A12 result of any part (and no
raw-pool observation) is added to MEMORY.md or to any other file the judges inherit.**
**Passes and merge rule.** These are A7-5d's rules verbatim (the read's `load_pass` is a75d_read.py's with y-keys):
- A verdict is valid when all of these hold: its label is SOLVABLE / UNSOLVABLE / AMBIGUOUS after strip().upper(); its key is
  one of the 1,200; its line lies in its tagged range; and that batch passed the check.
- A key takes the label of the first pass in which all its valid verdicts agree.
- Passes 2 and 3 are mandatory re-judges of exactly the 25-line batches the read prints, each a fresh launch, saved unedited.
- A pass is resumed only through resumeFromRunId, and checked over its resume chain.
- For each pass, the run id(s) and the sha256 of its verdict and check files go into one hash-log entry before the read runs.

The read aborts on any of these, and each abort means NOT RUN:
- `items.jsonl` not decoding to key_full's (key, problem) pairs in order;
- a verdict file without its check;
- a check that is not an a12 check of the registered file;
- batches differing from the registered ones;
- saved verdicts differing from the judges' structured returns, including a batch that passed the check but has no verdict
  in the saved file although its judge's structured return was accepted. A passed batch with no saved verdict is legitimate
  only for a null return (every structured return of the batch rejected by the harness): it is listed
  (`passed_null_return`) and its keys are re-judged in the next pass;
- a shared transcript directory, or a port outside {<HOST>, <HOST>};
- a pass present while an earlier one is missing.

If keys are still missing after pass 3, part 1 is NOT RUN.

### 1.6 Read (`a12_p1_read.py`) and pre-stated reading
Ill-posed = UNSOLVABLE or AMBIGUOUS.
**Primary quantity.** D_raw = raw ill-posed share(it3) - share(it1), pooled over seeds (400 vs 400 items). Its 95% CI comes
from an item bootstrap: 4,000 replicates, `random.Random(20260926 + 12)`, items resampled within seed x iteration. The label
has four states:
- **RISES**: CI lower bound > 0;
- **FLAT**: CI inside [-10, +10] points and containing 0 (10 points is RZ-P1's drift threshold);
- **FALLS**: CI upper bound < 0;
- **INCONCLUSIVE**: CI contains 0 but reaches beyond +-10;
- **NOT RUN**: section 1.5.

**Downstream-selection share (second pre-stated state).** Under the exact identity logit(ill-posed share among the in-sample
admitted) = logit(raw share) + log E, where E = a_ill / a_solv (a_ill = P(admitted | ill-posed), a_solv = P(admitted |
solvable) among raw items) is the enrichment of all downstream stages together, i.e. of downstream selection (pseudo-labelling
and the 0.3-0.8 filter), the share of downstream selection in the admitted log-odds drift is
**Delta log E = Delta logit(in-sample admitted share) - Delta logit(raw share)**, it3 vs it1, pooled over seeds. Its 95% CI
comes from the same bootstrap replicates. States: **POSITIVE** (CI lower bound > 0), **NEGATIVE** (CI upper bound < 0), **NOT
ESTABLISHED** (CI contains 0, or undefined). The level of E per iteration is descriptive: a constant E > 1 raises the admitted
level but adds no drift.
**Band share (secondary; a descriptive label).** The same three states are computed from the CI of Delta log E(scored), the
same difference of log E among scored items only (status != not_scored), from the same replicates. It isolates the 0.3-0.8
filter from the pseudo-labelling drops. It is reported next to the primary state (`band_share_secondary` in the read's
`reading`), marked as secondary, and enters no registered wording.

**Pre-stated wording** (`reading_text` in the read; X = the CI upper bound of D_raw, Y = 100 X / 25.0, the registered
admitted drift pooled over seeds):
- **RISES**: "the challenger's raw output drifts upward; A12 does not separate its cause (the uncertainty reward, the repetition
  penalty or the format reward): consistent with, not proof of, the uncertainty reward", followed by "downstream selection
  (pseudo-labelling and the 0.3-0.8 filter) adds drift on top (Delta log E CI > 0): the drift acts at both stages" if
  POSITIVE, "the added share of downstream selection (pseudo-labelling and the 0.3-0.8 filter) is not established" if NOT
  ESTABLISHED, and "the enrichment by downstream selection (pseudo-labelling and the 0.3-0.8 filter) weakens over iterations"
  if NEGATIVE.
- **FLAT**: "no raw drift detected; raw drift within the CI (< X points, i.e. < Y% of the admitted drift)", followed by "the
  drift sits in downstream selection (pseudo-labelling and the 0.3-0.8 filter)" only if POSITIVE, and otherwise by "raw flat;
  share of downstream selection not established".
- **FALLS**: "the challenger writes fewer ill-posed questions over iterations", followed by "the drift sits in downstream
  selection (pseudo-labelling and the 0.3-0.8 filter)" only if POSITIVE, and otherwise by "raw falls; share of downstream
  selection not established".
- **INCONCLUSIVE**: "the direction of the raw challenger share is not determined at 400 items per iteration; numbers only".

The words "the drift sits in downstream selection" appear in no other case. No wording attributes the drift to the 0.3-0.8
filter alone; the secondary band state is reported beside the reading.

**Descriptive, reported whatever the state:**
- Per seed x iteration: raw ill-posed, UNSOLVABLE and AMBIGUOUS shares with Wilson 95% CIs.
- Per-seed D_raw with CIs, and the UNSOLVABLE-only D_raw.
- The admitted shares: the six registered original audits (`audit_admitted.json`, restricted to the 200 audited ids;
  hash-checked; their drift, +23.0 / +27.0 / +25.0, is reported as the check `matches_registered_RZ_P1`) and A7-5d blind (`a75d_read.json`, hash-checked). A hash
  mismatch of any of these seven files aborts the read.
- D_adm(orig) - D_raw with a bootstrap CI, and D_raw / D_adm(orig).
- The framing-free within-sample decomposition, per iteration pooled over seeds:
  - status shares;
  - a_ill, a_solv and E with CIs, and the same among scored items only (status != not_scored), which isolates the 0.3-0.8
    filter from the pseudo-labelling drops, with Delta log E(scored), its CI and its secondary state (above);
  - the exact identity logit(ill share among in-sample admitted) = logit(raw share) + log E;
  - D of the in-sample admitted share;
  - the challenger's fraction of the admitted log-odds drift, Delta logit(raw) / Delta logit(admitted in sample). This fraction
    is unstable when the admitted log-odds drift is near 0 and is reported with its CI only.
- The condition check (1.3).
- Agreement with the original verdict on raw items whose text equals an audited admitted item of the same run (about 30-60
  items are expected).
- Covariates per cell (registered): the parse-failure rate (n_generated - n_parsed) / n_generated, the not-scored rate among
  parsed questions, CJK share, multi-line share, median length, repeated text, and the ill-posed share by status.

### 1.7 Downside and power, flagged before data
- RISES weakens the statement that the R-Zero drift is a property of the filter. The paper would then have to say that R-Zero's
  challenger drifts too, with the cause not separated by A12; the "filter-located" wording no longer holds for R-Zero.
- If the Delta log E CI contains 0 or lies below 0, R-Zero's admitted drift is not shown to be produced by the enrichment of
  downstream selection (pseudo-labelling and the 0.3-0.8 filter). That is the real downside for the selection-enrichment
  mechanism as applied to R-Zero, and it is reported as such; the secondary band state shows whether the 0.3-0.8 filter alone
  carries an enrichment drift. The level of E (above or below 1) does not decide this.
- If the condition flag fires, raw-vs-registered level comparisons are uninterpretable without the within-sample control.
- **Power.** At 200 items per cell (400 per iteration pooled) and shares near 45%, SE(D_raw) is about 3.5 points and the CI
  half-width about 6.9. FLAT then needs |D_raw| <= about 3.1: **P(FLAT | true D_raw = 0) is about 62%** (about 69% at shares
  near 35%), P(INCONCLUSIVE | 0) about 33%, and P(RISES | true D_raw = +10) about 81%. The sample size is kept at 200 per cell
  (300 per cell would give P(FLAT | 0) of about 87% at 72 judges).
- NOT RUN or INCONCLUSIVE adds nothing and costs about 50 Opus sessions.

## 2. Part 2: R-Zero solver benchmark accuracy

### 2.1 Models, conditions and data
Models:
- Qwen3-4B-Base (`/root/models/Qwen3-4B-Base`, on both hosts).
- The R-Zero solvers S_i = `/run/ti/rzero/s<S>/models/rzero_qwen3_4b_Base_s<S>_solver_v<i>/global_step_15/actor/huggingface`
  (A75_DONE present), for i = 3 (final) and for i = 1, 2 ("if cheap"; they are, at about 0.3-0.6 GPU-h per model for both
  conditions). Seed 0's checkpoints are on host `-p <PORT>` and seed 1's on host `-p <PORT>`, both host-local. The base model and
  all six solvers have the same chat template (sha256 87a2728c...), eos 151643 and `<|im_end|>` 151645.

Conditions:
- **plain**: `dirB/code/b_eval.py` as registered (sha256 5bf7c94a...), byte-identical, run through `a12_plain_eval.py` with two
  registered settings. Greedy, T = 0, max 2,048 new tokens, the Base prompt format (every path contains "Base"). Tag
  `a12_plain_<m>`.
  - `--k 1`: b_eval's T = 1 samples (maj@8 / pass@8) are not read in A12. b_eval cannot skip its sampling call, so k = 1 keeps
    one sample per prompt. Greedy decoding is a separate generate call (T = 0, seed 0) made before the sampling call, so
    greedy accuracy does not depend on k.
  - A stop at `<|im_end|>` added to every SamplingParams, for every model of the condition (the base model included, so the
    paired comparison treats all models alike). The solvers were trained in the chat format. Without the stop, a solver that
    ends its turn in the Base-format prompt could continue into an invented next turn, and b_eval's last-box scorer would score
    that turn's box. Under greedy decoding the kept text is the prefix of the unstopped output; for a model that never emits
    `<|im_end|>` nothing changes. The runner records per set how many greedy outputs ended at `<|im_end|>`.
  - The runner checks the sha256 of b_eval.py and of the two project modules it imports (`code/loto_common.py` 90155811...,
    `dirB/code/pdc_util.py` f1ad5fe2...), requires `--k 1` and a Base-format model path, and refuses b_eval's `--chat`,
    `--abstain_prompt`, `--redo`, `--max_new` and `--gpu_util` (they stay at their registered defaults).
- **rzchat**: `a12_rzchat_eval.py`. b_eval has no system-message option, so a minimal wrapper implements R-Zero's own solver
  prompt as in `evaluation/generate.py` of the official code (5699329):
  - chat template, with the system message "Please reason step by step, and put your final answer within \boxed{}." and the
    bare question as the user message;
  - greedy decoding, max 4,096 new tokens;
  - the question is the row's `problem`; for GSM8K, b_eval's "####" instruction is removed, and it must end the problem exactly.

  The wrapper has one registered deviation. Generation also stops at `<|im_end|>`, the end of the assistant turn; R-Zero's call
  stops only at `<|endoftext|>`. Under greedy decoding, the kept text is the prefix of R-Zero's output.
  Scoring:
  - Primary: b_eval's own `extract()` / `correct()`. The wrapper takes their source from the registered b_eval.py at run time
    and checks the sha256 of b_eval.py, loto_common.py and pdc_util.py, so the scorer is identical.
  - Secondary: R-Zero's scorer without its GPT-4o re-check. It takes the first match of `(?i)\\boxed\s*{([^\n]+)}` and applies
    math_verify `verify(parse(gold), parse(answer))`.

  Tag `a12rz_<m>`. This condition also covers the base model, which the R-Zero paper evaluates the same way.

Data (sha256 registered, checked by the launcher's preflight):
- `math500_train` (0862774d...): MATH-500. Its source `data/math/math500.parquet` has data_source "HuggingFaceH4/MATH-500" and
  first item "Convert the point (0,3) in rectangular coordinates...": the Lightman et al. 500-problem split, the same split as
  R-Zero's `math_500_test.csv` (not compared item by item).
- `math_heldout1000` (d4bde312...): 1,000 MATH problems disjoint from MATH-500.
- `gsm8k_test500` (58a521c1...): 500 of the 1,319 GSM8K test problems of `data/gsm8k/test.parquet`, namely the first 500
  positions of `random.Random(1).shuffle(list(range(1319)))` (checked: this reproduces all 500 positions).

The existing summary `qwen3_4b_base_step0` (sha256 975e7e97...; b_eval with k = 8 and no `<|im_end|>` stop) is kept as a
reproduction check against the fresh plain base run (same code otherwise, same stack as the solvers). The fresh run is the
comparison.

### 2.2 R-Zero paper numbers (retrieved 26 Sep 2026)
R-Zero: Self-Evolving Reasoning LLM from Zero Data (Huang et al.), arXiv 2508.05004, accepted at ICLR 2026. Qwen3-4B-Base, math:

| | AVG (7) | AMC | Minerva | MATH | GSM8K | Olympiad | AIME25 | AIME24 |
|---|---|---|---|---|---|---|---|---|
| Base model (v1) | 42.58 | 45.70 | 38.24 | 68.20 | 87.79 | 41.04 | 6.15 | 10.94 |
| Base challenger (v1) | 44.36 | 45.00 | 45.22 | 72.80 | 87.87 | 41.19 | 7.29 | 11.15 |
| R-Zero Iter 1 (v1) | 48.06 | 51.56 | 51.47 | 78.60 | 91.28 | 43.85 | 9.17 | 10.52 |
| R-Zero Iter 2 (v1) | 48.44 | 52.50 | 51.47 | 79.80 | 91.66 | 44.30 | 4.27 | 15.10 |
| R-Zero Iter 3 (v1) | 49.07 | 57.27 | 52.94 | 79.60 | 92.12 | 44.59 | 4.27 | 12.71 |
| Base model (v4, Table 1) | 42.57 | 45.70 | 38.24 | 68.20 | 87.79 | 41.04 | 10.30 | 6.7 |
| R-Zero (v4, final) | 49.93 | 57.27 | 52.94 | 79.60 | 92.12 | 44.59 | 9.60 | 13.40 |

Their protocol is: greedy decoding, except mean@32 for AMC and AIME; the chat template with the boxed system prompt; max 4,096
tokens; and a GPT-4o answer re-check. The AIME columns and the averages differ between v1 and v4; MATH and GSM8K do not.
Comparable here: MATH-500 (+10.4 / +11.6 / +11.4 over the base) and GSM8K (+3.49 / +3.87 / +4.33). Our GSM8K is a 500-item
subset of the 1,319-item test set.

### 2.3 Runs (`a12_launch.sh`, section 4)
The launcher runs 14 cells: 7 models x 2 conditions, one GPU each, final solvers and the base first. A cell is complete when
its summary holds all three sets. b_eval and the wrapper both resume per set.

### 2.4 Read (`a12_p2_read.py`) and pre-stated reading
**Primary cell: rzchat, iteration 3, math500_train, b_eval scorer (greedy_ok).** **Secondary cell: plain, iteration 3,
math500_train, b_eval scorer (greedy_ok).** For each seed S and each of the two cells, the paired delta S_3 - base (same
condition, same 500 items) gets a paired item bootstrap CI: 4,000 replicates, its own stream
`random.Random(f"{20260926 + 14}:<comparison key>")` (so no CI depends on which other cells exist). Per-seed label, with margin 0
points at the 95% CI bound: **GAIN** (lower bound > 0), **LOSS** (upper bound < 0), **NO_GAIN** (otherwise).
**Reading** (`reading()` in the read; computed as `out["reading"]`), from the primary labels of the two seeds and the number of
secondary GAINs. Every combination of the 3 x 3 x 3 x 3 labels maps to exactly one name, and NOT RUN applies when any of the
four labels is missing:

| primary labels (seed 0, seed 1, any order) | secondary: 2 GAIN | secondary: 1 GAIN | secondary: 0 GAIN |
|---|---|---|---|
| GAIN, GAIN | REPRODUCES | REPRODUCES, FORMAT-DEPENDENT | REPRODUCES, FORMAT-DEPENDENT |
| GAIN, NO_GAIN or GAIN, LOSS | REPRODUCES IN ONE SEED | REPRODUCES IN ONE SEED | REPRODUCES IN ONE SEED |
| NO_GAIN, NO_GAIN | PLAIN-ONLY | PLAIN-ONLY | NOT REPRODUCED |
| NO_GAIN, LOSS or LOSS, LOSS | PLAIN-ONLY | PLAIN-ONLY | NOT REPRODUCED, LOSS |

Registered texts:
- **REPRODUCES**: under R-Zero's own prompt the final solver beats the base model on MATH-500 in both seeds, and so it does
  under the plain Base prompt: the replication reproduces the direction of R-Zero's reported gain (magnitude ratio reported).
- **REPRODUCES, FORMAT-DEPENDENT**: under R-Zero's own prompt the final solver beats the base model on MATH-500 in both seeds;
  the gain is not shown in both seeds under the plain Base prompt (format contrast and plain truncation reported). The words
  "the gain depends on the format" are added only if the paired format contrast delta(rzchat) - delta(plain) (iteration 3,
  MATH-500, b_eval scorer) has its 95% CI lower bound > 0 in both seeds; the truncation note of 2.5 is added when its flag is
  raised.
- **REPRODUCES IN ONE SEED**: gain under R-Zero's prompt in one seed only (the other seed's label reported); partial, and weak
  either way with two seeds.
- **PLAIN-ONLY**: no seed gains under R-Zero's prompt while at least one gains under the plain prompt: the gain is not
  reproduced under R-Zero's protocol; the plain-prompt gain is reported descriptively.
- **NOT REPRODUCED**: no seed gains under either prompt: the replication does not reproduce the reported gains at greedy
  decoding with our scorers (no GPT-4o re-check).
- **NOT REPRODUCED, LOSS**: as NOT REPRODUCED, and under R-Zero's prompt the final solver is worse than the base in at least one
  seed.
- **NOT RUN**: a primary or secondary cell of a seed is missing.

**Descriptive only** (no reading attached): every other (condition, set, iteration) with the same per-seed labels, i.e.
GSM8K, heldout and iterations 1-2; the R-Zero-style scorer (accuracy and paired deltas); b_eval's lenient scorer in the plain
condition; the magnitude ratio, seed-mean delta / paper delta; the format contrast delta(rzchat) - delta(plain), paired (its iteration-3
MATH-500 CIs also decide the FORMAT-DEPENDENT wording above); rzchat
truncation and `<|im_end|>` rates; plain diagnostics from the per-item files (greedy truncation rate at 2,048 tokens, share of
greedy outputs whose first and last boxed answers differ, share of outputs that ended at `<|im_end|>`); and the base
reproduction check (fresh plain base vs the stored summary, whose sha256 is pinned; flagged if any set differs by more than 2
points, with no other consequence).
**Read aborts** (part 2 NOT RUN until corrected): a summary whose `model` is not the cell's own checkpoint; an rzchat summary not
scored with the registered b_eval.py; a rendered first prompt of an rzchat set that differs between models.

### 2.5 Downside flagged before data
NO GAIN at it3 invites the objection that the R-Zero drift was measured in a failed replication. REPRODUCES IN ONE SEED,
FORMAT-DEPENDENT and PLAIN-ONLY weaken the replication's standing less, but must be stated; so must a gain that appears only on
GSM8K (descriptive). Every result is reported as it comes. The `<|im_end|>` stop removes the end-of-turn artifact from the plain
condition; the remaining difference between the conditions is the prompt format (system message, chat template) and the
2,048- vs 4,096-token cap. A FORMAT-DEPENDENT reading is therefore reported together with the plain truncation rate and the
first/last-box diagnostic. It is not called format adoption if the plain condition truncates much more often, defined as a
plain greedy truncation rate (2,048 tokens) that exceeds the rzchat truncation rate (4,096 tokens) by more than 5 points for the
base model or for S_3 of either seed on MATH-500 (`plain_truncates_more` in the read); the reading text then says so.

## 3. Part 3: open-weight well-posedness verifier on top of the band (fixed pool)

### 3.1 Items, judge, labels
**Items.** The 1,158 fixed-pool proposals (`dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl`, sha256 a97bf54e...; key =
proposal id, text = `problem`).
**Judge.** `a12_open_judge.py`, derived mechanically from the registered A10 judge (`a10_judge.py`, sha256 68b7b315...) by
`dev/dev_mk_judge.py`. The following are byte-identical to A10:
- Qwen2.5-32B-Instruct (`/dev/shm/models/Qwen2.5-32B-Instruct`, host `-p <PORT>`), vLLM TP 2, bf16, max_model_len 4,096;
- greedy decoding, max 400 new tokens, one item per request;
- the A10 prompt, with the frozen definitions and the "ADMITTED by a self-play filter" sentence;
- parsing of the last "VERDICT:" line, with one retry at T 0.7, seed 1, and still unparsable counted as missing.

Only the input, the output directory (`dirA5/results/a12_open/`), a refusal to overwrite and an atomic final write differ.
The sentence is kept because the verifier acts only on in-band items: every item whose verdict enters a per-checkpoint read has
been admitted by the band at that checkpoint. Items never in band enter only the pool-level rates.
**Labels.** The registered Opus labels (`dirA3/results/pool_audit/qwen_s0_r1/parts/*.json`, 47 files, read exactly as
`a3_guard_diag_score.py` reads them). The band per checkpoint comes from `dirA3/results/guard_diag/qwen_s0/<ck>/guard_records.jsonl`,
ck = base, r1..r5 (R-Zero band [0.25, 0.75], G = 8, kind "proposed"). The band-alone numbers must reproduce the registered
`result.json` (sha256 a250ce88...), or part 3 is NOT RUN.
**Pinned inputs.** The read checks the sha256 of `proposed.jsonl`, `result.json` and the six `guard_records.jsonl` (base
d7271ee7..., r1 b9aa0601..., r2 5da61cd0..., r3 9ccddc33..., r4 ca5dc3d6..., r5 6677ec70...), and one digest over the 47 label
parts (sha256 of the lines "<file name> <sha256>" in name order: d4718c01...). Any mismatch aborts the read: part 3 NOT RUN.

### 3.2 Rule
Admit an in-band proposal only if the open judge says SOLVABLE. A missing verdict counts as not SOLVABLE; a sensitivity analysis
counts it as SOLVABLE instead. More than 58 missing verdicts (5%, the A10 limit) makes every reading None.

### 3.3 Run
One cell on one claimed GPU pair, through `a12_launch.sh`. A10 took about 2 minutes for 900 items.

### 3.4 Read (`a12_p3_read.py`) and pre-stated reading
**Per checkpoint:**
- s_band, the Opus ill-posed share of the band-admitted proposals, and s_ver, the same share with the verifier on top, with
  admitted counts and Wilson CIs;
- the verifier's yield n_ver / n_band;
- the in-band miss rate eps_t = P(open SOLVABLE | Opus ill-posed, in band), the keep rate k_t = P(open SOLVABLE | Opus
  SOLVABLE, in band), and eps_t / k_t;
- the exact identity odds(s_ver) = odds(s_band) x eps_t / k_t, checked. It implies OR_ver / OR_band = (eps_r5 / k_r5) /
  (eps_base / k_base), where OR is the odds ratio r5 vs base.

**Pool level:**
- the miss rate on Opus-ill-posed items (overall, AMBIGUOUS, UNSOLVABLE);
- the false-reject rate on Opus-SOLVABLE items;
- the confusion table, and binary and three-class kappa.

**Bootstrap.** Paired item bootstrap over the 1,158 proposals, with the same draw at every checkpoint: 4,000 replicates,
`random.Random(20260926 + 13)`. It gives CIs for D_band = s_band(r5) - s_band(base), for D_ver, for the gap s_band - s_ver at
each checkpoint, and for OR_ver / OR_band. The CIs of D_band, D_ver and the gaps use the replicates in which the shares involved
are defined (at least one verifier-admitted item); the OR_ver / OR_band CI uses the replicates in which, in addition, all four
odds are defined (every share strictly between 0 and 100%). A CI needs at least 40 valid replicates and is undefined below
that: the Corollary reading is then NOT DETERMINED, an undefined D_ver CI gives NOT SHOWN, and an undefined gap CI counts as not
excluding 0. The numbers of valid replicates are reported.

**Pre-stated readings.**
Level, three states:
- **LOWERED**: the gap's CI excludes 0 upward at every checkpoint;
- **NOT LOWERED**: it never does;
- **MIXED**: otherwise.

The expectation is that a text-based verifier lowers the level.
Direction, a statement about direction and power only, three states:
- **PERSISTS**: D_ver's CI lower bound > 0: the admitted ill-posed share still rises from base to r5 with the verifier on top;
- **REVERSED**: D_ver's CI upper bound < 0: it falls;
- **NOT SHOWN**: otherwise: the direction is not established at this n (the verifier lowers the level and n_ver, so D_ver can
  lose power even when the Corollary holds exactly). NOT SHOWN alone is never read as evidence against Corollary 1.

**Corollary 1** (mechanism note, s.1) says that with the verifier's miss and keep rates fixed across checkpoints, the verifier
rescales the odds by the constant eps / k, so it predicts OR_ver = OR_band. It is read from the CI of OR_ver / OR_band:
- **CONSISTENT**: the CI contains 1: consistent with a constant eps/k (eps_t and k_t are reported across checkpoints);
- **WEAKENS**: the CI upper bound is < 1: the verifier weakens the drift more than a fixed miss rate allows (eps_t / k_t falls
  from base to r5; eps_t and k_t are reported across checkpoints);
- **AMPLIFIES**: the CI lower bound is > 1: the verifier amplifies the drift (eps_t / k_t rises);
- **NOT DETERMINED**: the CI is undefined: fewer than 40 of the 4,000 replicates have all four odds defined.

"eps near 0" (descriptive) means the Wilson upper bound of the in-band miss rate at r5 is < 5%: the Corollary's escape condition,
under which the verifier removes almost every in-band ill-posed item and the odds-ratio test has little power; it is noted with
the Corollary reading.

### 3.5 Downside flagged before data
A10 found this judge lenient: 57% ill-posed against Opus's 68%, and kappa 0.379 against blind Opus. A large eps is therefore
likely, and so are PERSISTS and CONSISTENT: the verifier would lower the level but leave the drift's odds ratio unchanged, as the
Corollary predicts, and a reviewer may call the verifier weak. That is the point of a text-based open verifier, but it must be
said. A stronger verifier (Opus) would be circular with the labels. NOT SHOWN with a clearly non-zero eps is not a
contradiction of Corollary 1: it says the direction is not established at the verifier's reduced n, and the Corollary is read
from the odds ratio alone. WEAKENS would mean that the verifier's miss rate falls as the solver sharpens, which the fixed-eps
Corollary does not cover; it would be reported with eps_t and k_t. AMPLIFIES would mean the verifier makes the drift worse.

## 4. GPU etiquette and run control (`a12_launch.sh`)
Commands, with one launcher per host:
- `bash ext/a12_draft/a12_launch.sh check` (read-only preflight);
- `nohup bash ext/a12_draft/a12_launch.sh run all > /dev/null 2>&1 &` on each host (`-p <PORT>` runs the seed-0 cells and, if
  claimed first, the base cells; `-p <PORT>` runs seed 1, the base and part 3);
- `bash ext/a12_draft/a12_launch.sh stop` (safe stop; it signals the PID in `/dev/shm/a12_launch.pid` only if that process is
  an `a12_launch.sh run`).

The preflight requires every file of section 6 (dev/ included) to be in the hash log. It also checks b_eval.py, the two project
modules b_eval imports (`code/loto_common.py`, `dirB/code/pdc_util.py`) and the three data files against their registered
sha256, and prints the rendered R-Zero prompt. The plain runner and the rzchat wrapper repeat the module checks before they run.

**GPUs.** A GPU is free only if all of these hold, at two checks 60 s apart (the standing rule: idle for more than a minute):
- it has no marker `/dev/shm/gpu_single_<g>`;
- nvidia-smi shows < 1,500 MiB used;
- it carries no foreign compute process (the yield rule's test, below);
- this host has not yielded it in the last 10 minutes.

Free GPUs are taken in random order, atomically with `( set -o noclobber; : > /dev/shm/gpu_single_<g> )`. Each job gets one
GPU, or a TP pair for part 3. There are at most A12_MAXPAR (4) jobs per host. Markers are released after every attempt and by
the exit handler. Cells are de-duplicated across hosts by an atomic mkdir claim under `ext/a12_runs/claims`.

**Jobs.** Every job starts with `setsid`, so it is its own session and process group, and carries `A12_TAG`; it does not
inherit the launcher's lock. To kill a job (stop or yield), the launcher first lists the job's descendants and the members of
its session, each with its start time, and only then signals: the job's process group, and every listed process whose start
time still matches. The KILL pass reuses the TERM pass's list together with a fresh one. vLLM workers that were reparented,
whose environment was overwritten by setproctitle, or that left the job's process group are therefore killed with it: a process
that called setpgid stays in the session, and one that called setsid is on the list if it was a descendant when the job was
first signalled, even after its parent died of that signal. After every attempt, whatever is left in the job's process group
is killed. A process counts as ours if it is in one of our jobs' process groups, has a tagged ancestor, or is an exiting
process of this container. The stop flag is checked before every attempt and right after every claim.

**Co-tenant priority.** A monitor checks the job's GPUs every 60 s. A compute process there that is not ours, confirmed 30 s
later, makes the job yield. This is A7-5's foreign-process rule, which worked on these hosts, extended by the process-group
test. On a yield:
- the job is killed as above (TERM, then KILL after 20 s);
- its GPUs are released and marked as cooling down for 10 min on this host (`ext/a12_runs/cooldown/`);
- the cell waits 10 min before claiming again, in steps that end early on a stop.

A yield is not a failed attempt, and a cell yields at most 12 times per launch. `touch ext/a12_runs/STOP` (all hosts) or
`STOP_<host>` stops new jobs.

**Safe stop.** On INT, TERM or HUP, the launcher does the following in order, and no second signal interrupts it:
1. sets a stop flag;
2. kills every job it started, and only those, as above (TERM, then KILL after 120 s; the KILL pass also covers a job that
   ended in the meantime, through its TERM-pass list), then kills any process still carrying its tag;
3. waits for its children;
4. releases its markers and claims;
5. exits 143.

**Restart after a crash.** At start, a launcher kills the jobs (with their process groups) of a dead A12 launcher of the same
host, recognised by a tag that names the dead launcher's PID, then releases that launcher's markers (`a12 <dead pid>`) and
removes its PID directory; its cell claims are taken over. Markers of other projects are never touched.

**Resumability.**
- Each attempt is capped by `timeout`: 2 h plain, 1.5 h rzchat, 1 h judge.
- There are up to 3 attempts per cell per launch.
- b_eval and the wrapper never recompute a finished set.
- The judge writes its verdicts once, atomically, and refuses to overwrite.

Relaunching the same command resumes.

**Watchdogs.** Jobs start as `/usr/bin/python3 ...`, which the `^python3 ` pattern of `/root/hang_watchdog.sh` does not
select (as for A7-5's probes), and each job holds its GPU's memory in any case. `/root/ckpt_janitor2.sh` touches only
`/dev/shm/ckpt_b`, where A12 writes nothing.
**Tests.** A CPU-only test on synthetic state under /tmp (`dev/dev_a12_launch_test.sh`; fake nvidia-smi and fake jobs) covers
the launcher: T1 a full run (GPUs taken in random order); T1b a rerun that skips everything; T2 a safe stop mid-run with no
job, marker or claim left; T3 a co-tenant yield, after which the cell resumes on another GPU (the yielded one is cooling down)
and completes; T4 busy GPUs never claimed; T5 the STOP file; **T6 a foreign compute process on a GPU with 608 MiB used: that GPU
is never claimed**; **T7 a reparented grandchild of a job, with its environment dropped, is killed on a safe stop and on a
yield**; T8 after the launcher and its shells are SIGKILLed, a new launch kills the orphan jobs, releases the dead launcher's
markers, takes over its claims and completes; **T9 three descendants of a job that ignore TERM, have an empty environment and
left its process group (a child that called setpgid(0, 0), a child that called setsid(), and a grandchild that was reparented
before the signal and called setpgid(0, 0)) survive neither a safe stop nor a yield**. Separate self-tests cover the sample,
pilot, checker, reads, wrapper, plain runner and judge (`dev/dev_a12_selftest.py`), all on synthetic data in /tmp.

## 5. Budget
- **GPU.**
  - Part 2 plain (k = 1): about 0.15-0.3 GPU-h per model. The measured b_eval wall time on 4B checkpoints here with k = 8 (nine
    sequences per prompt) was about 10 min MATH-500, 17 min heldout and 3 min GSM8K, plus about 2 min load; k = 1 keeps two
    sequences per prompt. That is about 1-2 GPU-h for 7 models.
  - Part 2 rzchat: greedy only, 2,000 prompts, about 0.15-0.3 GPU-h per model, so 1-2 GPU-h.
  - Part 3: at most 0.35 GPU-h (a pair for about 10 min, including load).
  - **Total: about 2.5-4 GPU-h expected, at most about 6 GPU-h** (with one full retry of the largest cells). The wall time is
    about 1-1.5 h with 3-4 GPUs per host; each claim waits at least 60 s.
- **API** (part 1 only): 2 pilot, 48 pass-1 and 1-4 expected re-judge Opus sessions at xhigh, so about 51-54 sessions. At
  roughly 10-20 model requests per session, that is about 500-1,100 API requests, similar to A7-5d (48+1 judges). The two
  pre-hash reviews used 2 sessions, and this revision 1.
- **CPU.** Stage and build take under a minute per host; the reads take under 2 minutes (the bootstraps).
- **Storage.** Under 150 MB on the shared volume: per-item eval files, the part-1 key and items, and verdicts.

## 6. Files (in `ext/a12_draft/`; registered in place, all hashed with this file)
The sha256 values below are those of this draft; the hash-log entry is the authority. This file's own sha256 is not in the
table.

| file | sha256 (draft) |
|---|---|
| `a12_p1_sample.py` | `c925ab8ec1ad5444f1e15ee1efbbaa1263ae9c2fc3d525b58e0a499124faf250` |
| `a12_p1_blind_wf.js` | `3716a92ae9b116536eaa13f41e0c7f5e50c624d48e156bcc201a731749d9f9dd` |
| `a12_blind_check.py` | `9f0b1eb06aa3e693f6736b91ac3b2b2c126895d183bf21d3e7cf39e72145be29` |
| `a12_p1_read.py` | `73c4ef19936d3c7d9d78f4e2b07de5f2d709a4578e390ab323ec34ff42d0ff87` |
| `a12_p1_pilot_build.py` | `567cc4562ec6c60086a01c6373a9028b9270e25b946c09b16d08c5d7e5198786` |
| `a12_rzchat_eval.py` | `50f5a8ba6897f1aaff6715344467e5eba9ed5904228e5b974d348f0ffd249fca` |
| `a12_plain_eval.py` | `2d12997ce39e7309e62cf0ece7181e9584e4d50b712dc5c17ecd0a924566ab6e` |
| `a12_open_judge.py` | `ba144a9341abf16f9549d3e24e0bdc58916efe411f537745f770cd5daf110f1c` |
| `a12_launch.sh` | `e69557e7a7393e417bb714224a808650442845db58d06a6e821cacf889c5449f` |
| `a12_p2_read.py` | `a696b813a82d584b3aa8f8a7aa818a96843fbd7f1878483471cde741829b05ea` |
| `a12_p3_read.py` | `c4c1a31f78126b341d10e65051b309c53f45a1d2dab2034d6a65c61b98866383` |
| `dev/dev_mk_check.py` | `2260fc7fb36978ff03120e5161791c311a4837f786e05a17e545c406f4f62920` |
| `dev/dev_mk_judge.py` | `89c52c9fd3098aa0d4a028f7d26f9a21586469225f4a47eeb155b53d36b51b33` |
| `dev/dev_a12_selftest.py` | `949aaa011a2621fa13dde7b8233fd8fcffd3a4685ee509d157593d791966297c` |
| `dev/dev_a12_launch_test.sh` | `9854d4e79042224357e362660377c70902d9811fcf572f28c555a584ba8f6787` |
| `dev/wf_stub_test.mjs` | `6755bdfe907fcc3b664169c157f0531453491b8679fd976308a36256c8bf6165` |

`a12_p1_sample.py`, `a12_p1_blind_wf.js`, `a12_blind_check.py`, `a12_p1_read.py`, `a12_p1_pilot_build.py`,
`a12_rzchat_eval.py`, `a12_plain_eval.py`, `a12_open_judge.py`, `a12_launch.sh`, `a12_p2_read.py`, `a12_p3_read.py`.
Provenance, also hashed: `dev/dev_mk_check.py`, `dev/dev_mk_judge.py`, `dev/dev_a12_selftest.py`, `dev/dev_a12_launch_test.sh`,
`dev/wf_stub_test.mjs`.
Inputs used unchanged, with their sha256 to be appended with this file (the scripts pin the ones marked *):
- `dirB/code/b_eval.py` 5bf7c94a...*, `code/loto_common.py` 90155811...* (not yet in the hash log; appended with the package),
  `dirB/code/pdc_util.py` f1ad5fe2...*, and the three dirB/data sets* (section 2.1);
- `dirA5/code/a75d_blind_check.py` e42e8905..., `a75d_blind_wf.js` 711d4f0c..., `a10_judge.py` 68b7b315...;
- `dirA3/code/a3_guard_diag_score.py` aa3c5398...;
- the six `audit_admitted.json`* (already hashed at A7-5d), `dirA5/results/a75d_blind/a75d_read.json` 7c9bd8bd...*, and the six
  R-Zero pool summaries;
- `dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl` a97bf54e...* (not yet in the hash log; appended with the package), the
  47 pool-audit parts* (digest d4718c01...), the six `guard_records.jsonl`*, `result.json` a250ce88...*, and
  `dirB/results/evals/qwen3_4b_base_step0.summary.json` 975e7e97...* (not yet in the hash log; appended with the package).

Written before hashing and deleted after the pilot (non-study output): `dirA5/results/a12_pilot/`.
Written at run time, not hashed with this file:
- `dirA5/results/a12_key/`, `dirA5/results/a12_blind/` and `dirA5/results/a12_open/`;
- `dirB/results/evals/a12_plain_*` (including `a12_plain_<m>.plain_stops.json`) and `a12rz_*`;
- `ext/a12_runs/` (log, status, claims, logs, cooldown, the part-2 read).

## 7. Order of operations
1. Two adversarial reviews (done) and this revision (section 9).
2. Pilot, by the lead author (1.5): `a12_p1_pilot_build.py` on the server, the workflow with `{pass: 1, pilot: true}`, then
   `a12_blind_check.py <out> pilot <transcript dir>` locally. Both batches must pass. Then delete the pilot outputs and record
   the deletion.
3. Hash this file, the section-6 files and the three inputs not yet in the hash log in one entry, whose header line also
   records the pilot deletion, the pilot check's sha256 and its passed / dropped lists (1.5):
   ```
   cd /home/tione/notebook/rsi_rl && { echo "$(date '+%F %T') A12 REGISTERED (registration + section-6 files + 3 inputs), before any A12 data; <pilot record>"
     sha256sum ext/a12_draft/A12_registration_2026-09-26.md ext/a12_draft/a12_p1_sample.py ext/a12_draft/a12_p1_blind_wf.js \
       ext/a12_draft/a12_blind_check.py ext/a12_draft/a12_p1_read.py ext/a12_draft/a12_p1_pilot_build.py \
       ext/a12_draft/a12_rzchat_eval.py ext/a12_draft/a12_plain_eval.py ext/a12_draft/a12_open_judge.py \
       ext/a12_draft/a12_launch.sh ext/a12_draft/a12_p2_read.py ext/a12_draft/a12_p3_read.py \
       ext/a12_draft/dev/dev_mk_check.py ext/a12_draft/dev/dev_mk_judge.py ext/a12_draft/dev/dev_a12_selftest.py \
       ext/a12_draft/dev/dev_a12_launch_test.sh ext/a12_draft/dev/wf_stub_test.mjs \
       code/loto_common.py dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl \
       dirB/results/evals/qwen3_4b_base_step0.summary.json; } >> docs/registration_hashes.txt
   ```
   Then `bash ext/a12_draft/a12_launch.sh check` must report every A12 file as registered.
4. Part 1: `stage --seed 0` on `-p <PORT>` and `stage --seed 1` on `-p <PORT>`, then `build`, then hash the stage, items and key
   files. Run the pre-launch scan, then pass 1, the check, a hash entry and the read. Continue with passes 2-3 as printed.
5. Parts 2-3, independently and in parallel with part 1: `a12_launch.sh run all` on both hosts, then `a12_p2_read.py` and
   `a12_p3_read.py`. Hash each read output. Nothing from parts 2-3 enters MEMORY.md before the part-1 read is done.
6. Report every part's state verbatim, NOT RUN included.

## 8. Known limitations
1. The framing sentence differs from the admitted audits (1.3); the original audits also differ from A7-5d in further ways.
2. RAW excludes unparsed generations, which are not questions; their rate differs by cell and moves in opposite directions in
   the two seeds (a registered covariate).
3. Status for rows that repeat a text is exact only up to swaps between identical texts, which judges cannot distinguish.
4. At 200 items per cell, P(FLAT | no raw drift) is only about 62% (1.7).
5. Part 2 has no GPT-4o re-check, uses a GSM8K subset, and runs AMC, Minerva, Olympiad and AIME not at all. MATH-500 is the
   same split as R-Zero's by provenance, not compared item by item.
6. The plain condition stops at `<|im_end|>` and uses k = 1, so its base run differs from the stored k = 8 summary in both;
   greedy accuracy is unaffected by k, and by the stop for a model that never emits `<|im_end|>`.
7. Part 3 uses a single fixed pool (seed 0), one open judge, and one threshold rule.
8. With two seeds, "both seeds" is weak.
9. The co-tenant ownership rule was tested with a fake nvidia-smi only; the process-group test covers the A7-5 smoke failure
   (vLLM engine processes whose environment setproctitle overwrote) as long as vLLM keeps its workers in the job's process group.
   Killing reaches further (group, session, listed descendants; T9): only a process that started its own session (setsid)
   and had already been reparented when the job was first signalled, with its environment overwritten, would escape it.

## 9. Revision after review (26 Sep 2026)
Two adversarial reviews of the first draft (sha256 2db1d274...) raised four blocking points each. Changes:
- Launcher: a GPU is free only with no marker, < 1,500 MiB and no foreign compute process, at two checks 60 s apart; random
  order; 10-min per-GPU cooldown after a yield; jobs in their own process groups, killed as groups; the stop flag checked
  before every attempt; orphan jobs and markers of a dead launcher removed at start; `stop` checks the PID's command line;
  loto_common.py and pdc_util.py pinned; tests T6-T8 added.
- Part 1: Delta log E with a CI and a pre-stated filter-share state; the FLAT / FALLS / RISES wording tied to it and to the CI
  bound; the downside restated in terms of Delta log E; the condition check weighted per seed, pooled over iterations, with a
  bootstrap CI and A7-5d blind as the primary reference; the framing-difference and parse-failure facts corrected; the
  parse-failure and not-scored rates registered as covariates; E among scored items added; the power stated; the read aborts
  on a passed batch missing from the saved verdicts (unless a null return) and on items/key mismatch; the checker aborts on a
  task text without the framing sentence; pilot keys q, pilot outputs deleted and the deletion recorded; judge-context embargo
  and scratchpad check added.
- Part 2: one primary and one secondary cell with per-seed GAIN / NO_GAIN / LOSS labels and an exhaustive mapping to one
  reading computed by the read; everything else descriptive; the plain condition run with k = 1 and a stop at `<|im_end|>`
  through `a12_plain_eval.py`; model-path, b_eval and prompt-identity checks; per-comparison bootstrap streams; plain
  diagnostics; base reproduction tolerance; data provenance stated.
- Part 3: the Corollary read from the CI of OR_ver / OR_band; PERSISTS / NOT SHOWN / REVERSED as statements about direction and
  power, NOT SHOWN never a contradiction; section 3.5 rewritten; inputs pinned by sha256.
- Text: the parse-failure range, the number of banned names, the preflight's file list (dev/ now included), the per-model cost
  and the pilot directory's status corrected.
- Verification pass (same day): a job's descendants and session members are listed before it is signalled and the KILL pass
  reuses that list (T9 added); the A7-5d blind read pin aborts the read; the part-1 wording names downstream selection
  (pseudo-labelling and the 0.3-0.8 filter), with a secondary band-only state; the FORMAT-DEPENDENT wording is gated by the
  format-contrast CI and the plain-truncation margin is defined; the CONSISTENT wording and the part-3 replicate rule are
  stated; the scan's grep exit-2 rule is stated; three pinned inputs are marked as not yet hashed, and the hash command is
  written out.
