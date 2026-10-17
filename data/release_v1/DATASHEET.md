# Datasheet: audited self-play curricula (release v1)

Release date 2026-09-24; the project tree was read at 2026-09-24T17:07:57+08:00 (snapshot time, also in INDEX.jsonl and
judge/validation/summary.json). Structure follows Gebru et al., "Datasheets for Datasets" (CACM 2021). Every number in
this file is computed by `build_release.py` from the files of this release; MANIFEST.tsv gives the SHA-256 of each file.

## 1. Motivation

**Purpose.** Label-free self-play for math reasoning (R-Zero-style loops) lets a proposer model write problems and admits
into the curriculum those on which the current solver's sampled answers disagree moderately (modal share in a band).
The loop itself never checks whether the admitted problems are well-posed. This release records, round by round and lineage by lineage,
which problems such loops admitted, how the admission-time solver answered them, and whether a strict LLM judge (Claude
Opus) finds them SOLVABLE, UNSOLVABLE or AMBIGUOUS. It also contains fixed candidate pools audited exhaustively (to
separate the guard's selection from the proposer's output), planted premise-deletion twins with their validation, the
exact judge prompts and a judge-free probe on an external human-labelled benchmark (UMWP).

**Creators and funding.** The authors of the accompanying submission (anonymous during review). Funding: to be stated
in the camera-ready version.

## 2. Composition

**Instances.** Grade-school math word problems in English, written by language models (the proposers), with one Opus
verdict per problem, the judge's confidence and a short note, and the admission-time solver statistics over G = 8 samples:
`solver_p_mode` (share of the modal parsed answer; unparsed or truncated samples count in the denominator, "unknown"
never votes), `solver_mode` (the modal answer) and `solver_abstain` (share of "unknown" answers; non-zero essentially
only under the abstention prompt). `solver_checkpoint` names the solver that scored the problem at admission: the base model in round
1, the previous round's trained checkpoint afterwards. `base_p_mode / base_mode / base_abstain` repeat these values when
that solver is the base model and are null otherwise (the base model did not score later-round proposals; base-model
agreement on fixed pools is in `fixed_pool_audits/`, key `<diag>/base`).

**Fields.** audited_pools rows: `lineage, run, model, condition, seed, round, id, problem, problem_sha256` (SHA-256 of the
released text), `redacted` (true if spans of the text were masked, see Confidentiality), `opus_verdict, opus_confidence,
opus_note, in_audit_sample200, audit_k` (the row's position in the run's Random(0) order of admitted proposer questions,
the order every audit of the run uses; set for every row, < 200 exactly for the 200-item sample; for the
38241 items whose k the run's own 200-item or full audit recorded, 0 differ), `audit_source`
(sample200, full_audit, or the extension pseudo-run), `solver_checkpoint, solver_p_mode, solver_mode, solver_abstain,
base_p_mode, base_mode, base_abstain`. INDEX.tsv: one row per audited run (admitted counts by kind, audited count,
coverage, verdict counts on the 200-item sample and on all audited rows, UNSOLVABLE-only and UNSOLVABLE + AMBIGUOUS
shares, guard rule, `admission_stat` and band, `identical_pool_as`). `admission_stat` is the statistic the guard compared
with the band: `p_mode` for all runs except the 6 loop_uv_a4 runs, which admitted on `max(p_mode,abstain)`
(`--count_unknown`); it is read off the guard records (the rule that reproduces every record's admit flag,
`admission_stat_basis` in INDEX.jsonl). INDEX.jsonl adds the verdict sources, the audit protocol string, the `audit_k`
check, `snapshot_time` and the raw guard_stats. fixed_pool_audits rows: `pool, proposer, id, k, problem,
problem_sha256, redacted, stem, opus_*`, `agreement` = map from `<diag>/<checkpoint>` to `p_mode, mode, abstain, admit`
(`evolved_from` for evo_r3). raw_proposal_audits rows mirror audited_pools rows without solver fields. Tracer, twin,
validation and probe files are described in their own sections; every file's row count is in MANIFEST.tsv (rows =
records for .jsonl, data rows for .tsv, lines otherwise).

**Counts.**

| directory | files | rows | MB |
|---|---|---|---|
| (top level) | 2 | 1745 | 0.11 |
| audited_pools | 161 | 44698 | 48.96 |
| fixed_pool_audits | 5 | 4642 | 7.47 |
| judge | 16 | 1372 | 0.26 |
| probe | 113 | 117337 | 21.40 |
| raw_proposal_audits | 12 | 2400 | 1.74 |
| twins | 5 | 80532 | 20.32 |

Total without DATASHEET.md and MANIFEST.tsv: 314 files, 100,267,514 bytes (100.3 MB; sum of file sizes).

*audited_pools/*: 159 audited (lineage, round) pools, 44380 rows; 144 distinct pools (36820 rows)
after removing pools that are copies by construction (`identical_pool_as` in INDEX.tsv: the rescue arms reuse the own
round-3 admitted set, co-evolving round 1 reuses own round 1). Over the distinct pools: 11662 SOLVABLE,
17396 UNSOLVABLE, 7762 AMBIGUOUS. The 36820 rows of the distinct pools hold
29312 unique problem texts: 5831 texts occur in more than one distinct pool
(see Relationships). 38 pools are audited in full (every admitted proposer question), the others on the
200-item random sample. Models: Llama-3.2-3B-Instruct (21 pools), Phi-4-mini-instruct (14 pools), Qwen3-14B-Base (2 pools), Qwen3-4B-Base (102 pools), Qwen3-8B-Base (20 pools).
13398 rows were scored at admission by the base model itself; 0 rows have no solver record.

| condition | group | pools | definition |
|---|---|---|---|
| coev_a4 | variant | 12 | co-evolving proposer: after each round the proposer is SFT-ed on the admitted proposals; R-Zero band; round 1 = own_a3 round 1 (A4) |
| coev_ag_a4 | variant | 6 | co-evolving proposer with the abstention guard (A4) |
| ctrl_lowp_a5 | control | 2 | plain band, then the lowest-modal-share in-band items dropped down to the guard lineage's count (A5 CTRL-L) |
| ctrl_rnd_a5 | control | 3 | plain band, then a random subsample down to the abstention-guard lineage's admitted count (A5 CTRL-L) |
| loop_uv_a4 | rescue | 6 | loop continued from the rescue_uv round-3 solver: abstention prompt, admission on max(p_mode, abstain) in the band, reward own_uv (A4i/A4j) |
| own_a3 | core_loop | 50 | R-Zero-style label-free loop: frozen proposer, R-Zero band guard p_mode in [0.25, 0.75] (plain prompt), GRPO on the solver's own majority answer (A3) |
| own_a4 | core_loop | 8 | same loop as own_a3 for Phi-4-mini-instruct and Qwen3-14B-Base (A4) |
| own_ag_a4 | guard_fix | 23 | abstention guard: abstention-option prompt, admit iff < 2/8 samples abstain and the band holds on the rest; training as own (A4) |
| own_azr_a3 | variant | 6 | Absolute-Zero learnability guard: admit p_mode in (0, 1) (A3e) |
| own_clean_a3 | control | 10 | trains only on the admitted proposals judged SOLVABLE by the full Opus audit (+ intact tracers); round 1 = own round 1 (A3) |
| own_sub_a3 | control | 12 | trains on a uniformly random 50% of the admitted set, size-matched control (A3) |
| own_tg_a3 | variant | 10 | tracer-calibrated band: each round the band [lo, hi] maximises intact-minus-deleted tracer admission (A3e) |
| rescue_abs_a4 | rescue | 3 | as rescue_uv but reward own_abs ('unknown' excluded from the vote, zero reward), prompt-matched control (A4f) |
| rescue_uv_a4 | rescue | 6 | own round-2 solver trained on own round-3 admitted set (copied) with reward own_uv ('unknown' can win the vote) (A4g) |
| rescue_uvw_a4 | rescue | 2 | as rescue_uv with weight w = 3 on 'unknown' votes (A4l) |

Ill-posed share (UNSOLVABLE + AMBIGUOUS) of the 200-item sample, per lineage and round:

| model | condition | seed | rounds audited | ill-posed share (U+A) of the 200-item sample by round |
|---|---|---|---|---|
| Llama-3.2-3B-Instruct | own_a3 | 0 | 1 | r1: 0.64 |
| Llama-3.2-3B-Instruct | own_a3 | 1 | 1 | r1: 0.69 |
| Llama-3.2-3B-Instruct | own_a3 | 2 | 3 | r1: 0.65, r2: 0.74, r3: 0.71 |
| Llama-3.2-3B-Instruct | own_a3 | 3 | 3 | r1: 0.67, r2: 0.73, r3: 0.69 |
| Llama-3.2-3B-Instruct | own_a3 | 4 | 3 | r1: 0.67, r2: 0.77, r3: 0.72 |
| Llama-3.2-3B-Instruct | own_clean_a3 | 0 | 2 | r2: 0.69, r3: 0.75 |
| Llama-3.2-3B-Instruct | own_sub_a3 | 0 | 2 | r2: 0.80, r3: 0.73 |
| Llama-3.2-3B-Instruct | own_sub_a3 | 2 | 2 | r2: 0.74, r3: 0.69 |
| Llama-3.2-3B-Instruct | own_tg_a3 | 0 | 1 | r1: 0.53 |
| Llama-3.2-3B-Instruct | own_tg_a3 | 1 | 3 | r1: 0.51, r2: 0.61, r3: 0.70 |
| Phi-4-mini-instruct | own_a4 | 0 | 3 | r1: 0.59, r2: 0.77, r3: 0.83 |
| Phi-4-mini-instruct | own_a4 | 1 | 3 | r1: 0.61, r2: 0.76, r3: 0.81 |
| Phi-4-mini-instruct | own_ag_a4 | 0 | 3 | r1: 0.52, r2: 0.69, r3: 0.76 |
| Phi-4-mini-instruct | own_ag_a4 | 1 | 3 | r1: 0.54, r2: 0.73, r3: 0.71 |
| Phi-4-mini-instruct | rescue_uv_a4 | 0 | 1 | r3: 0.83 |
| Phi-4-mini-instruct | rescue_uv_a4 | 1 | 1 | r3: 0.81 |
| Qwen3-14B-Base | own_a4 | 0 | 2 | r1: 0.57, r2: 0.72 |
| Qwen3-4B-Base | coev_a4 | 0 | 3 | r1: 0.55, r2: 0.71, r3: 0.74 |
| Qwen3-4B-Base | coev_a4 | 1 | 3 | r1: 0.56, r2: 0.70, r3: 0.76 |
| Qwen3-4B-Base | coev_a4 | 2 | 3 | r1: 0.59, r2: 0.70, r3: 0.77 |
| Qwen3-4B-Base | coev_a4 | 3 | 3 | r1: 0.56, r2: 0.67, r3: 0.70 |
| Qwen3-4B-Base | coev_ag_a4 | 0 | 3 | r1: 0.53, r2: 0.61, r3: 0.69 |
| Qwen3-4B-Base | coev_ag_a4 | 1 | 3 | r1: 0.41, r2: 0.63, r3: 0.71 |
| Qwen3-4B-Base | ctrl_lowp_a5 | 0 | 1 | r1: 0.52 |
| Qwen3-4B-Base | ctrl_lowp_a5 | 2 | 1 | r1: 0.49 |
| Qwen3-4B-Base | ctrl_rnd_a5 | 0 | 1 | r1: 0.56 |
| Qwen3-4B-Base | ctrl_rnd_a5 | 1 | 1 | r1: 0.57 |
| Qwen3-4B-Base | ctrl_rnd_a5 | 2 | 1 | r1: 0.53 |
| Qwen3-4B-Base | loop_uv_a4 | 0 | 2 | r4: 0.70, r5: 0.81 |
| Qwen3-4B-Base | loop_uv_a4 | 1 | 2 | r4: 0.81, r5: 0.81 |
| Qwen3-4B-Base | loop_uv_a4 | 2 | 2 | r4: 0.79, r5: 0.82 |
| Qwen3-4B-Base | own_a3 | 0 | 6 | r1: 0.55, r2: 0.72, r3: 0.76, r4: 0.81, r5: 0.81, r6: 0.83 |
| Qwen3-4B-Base | own_a3 | 1 | 6 | r1: 0.56, r2: 0.76, r3: 0.83, r4: 0.80, r5: 0.74, r6: 0.82 |
| Qwen3-4B-Base | own_a3 | 2 | 6 | r1: 0.59, r2: 0.68, r3: 0.78, r4: 0.79, r5: 0.81, r6: 0.77 |
| Qwen3-4B-Base | own_a3 | 3 | 3 | r1: 0.56, r2: 0.72, r3: 0.78 |
| Qwen3-4B-Base | own_a3 | 4 | 3 | r1: 0.57, r2: 0.76, r3: 0.84 |
| Qwen3-4B-Base | own_ag_a4 | 0 | 4 | r1: 0.48, r2: 0.67, r3: 0.63, r4: 0.67 |
| Qwen3-4B-Base | own_ag_a4 | 1 | 4 | r1: 0.39, r2: 0.67, r3: 0.69, r4: 0.79 |
| Qwen3-4B-Base | own_ag_a4 | 2 | 4 | r1: 0.38, r2: 0.65, r3: 0.65, r4: 0.69 |
| Qwen3-4B-Base | own_azr_a3 | 0 | 3 | r1: 0.50, r2: 0.68, r3: 0.73 |
| Qwen3-4B-Base | own_azr_a3 | 1 | 3 | r1: 0.51, r2: 0.62, r3: 0.67 |
| Qwen3-4B-Base | own_clean_a3 | 0 | 2 | r2: 0.79, r3: 0.81 |
| Qwen3-4B-Base | own_clean_a3 | 1 | 2 | r2: 0.73, r3: 0.75 |
| Qwen3-4B-Base | own_clean_a3 | 2 | 2 | r2: 0.80, r3: 0.78 |
| Qwen3-4B-Base | own_clean_a3 | 3 | 2 | r2: 0.79, r3: 0.80 |
| Qwen3-4B-Base | own_sub_a3 | 0 | 2 | r2: 0.69, r3: 0.81 |
| Qwen3-4B-Base | own_sub_a3 | 1 | 2 | r2: 0.72, r3: 0.74 |
| Qwen3-4B-Base | own_sub_a3 | 2 | 2 | r2: 0.77, r3: 0.77 |
| Qwen3-4B-Base | own_sub_a3 | 3 | 2 | r2: 0.74, r3: 0.79 |
| Qwen3-4B-Base | own_tg_a3 | 0 | 3 | r1: 0.43, r2: 0.58, r3: 0.69 |
| Qwen3-4B-Base | own_tg_a3 | 1 | 3 | r1: 0.49, r2: 0.52, r3: 0.57 |
| Qwen3-4B-Base | rescue_abs_a4 | 0 | 1 | r3: 0.76 |
| Qwen3-4B-Base | rescue_abs_a4 | 1 | 1 | r3: 0.83 |
| Qwen3-4B-Base | rescue_abs_a4 | 2 | 1 | r3: 0.78 |
| Qwen3-4B-Base | rescue_uv_a4 | 0 | 1 | r3: 0.76 |
| Qwen3-4B-Base | rescue_uv_a4 | 1 | 1 | r3: 0.83 |
| Qwen3-4B-Base | rescue_uv_a4 | 2 | 1 | r3: 0.78 |
| Qwen3-4B-Base | rescue_uv_a4 | 4 | 1 | r3: 0.84 |
| Qwen3-4B-Base | rescue_uvw_a4 | 0 | 1 | r3: 0.76 |
| Qwen3-4B-Base | rescue_uvw_a4 | 1 | 1 | r3: 0.83 |
| Qwen3-8B-Base | own_a3 | 0 | 3 | r1: 0.57, r2: 0.68, r3: 0.77 |
| Qwen3-8B-Base | own_a3 | 1 | 6 | r1: 0.58, r2: 0.69, r3: 0.69, r4: 0.79, r5: 0.77, r6: 0.78 |
| Qwen3-8B-Base | own_a3 | 2 | 6 | r1: 0.53, r2: 0.67, r3: 0.74, r4: 0.76, r5: 0.73, r6: 0.75 |
| Qwen3-8B-Base | own_ag_a4 | 1 | 2 | r1: 0.41, r2: 0.60 |
| Qwen3-8B-Base | own_ag_a4 | 2 | 3 | r1: 0.34, r2: 0.61, r3: 0.64 |

*fixed_pool_audits/* (every candidate audited; `agreement` holds, per scored checkpoint, p_mode / mode / abstain /
admit; CHECKPOINTS.tsv maps each key to its source directory, the solver checkpoint, prompt, guard, admission statistic
and band). Two groups of keys need care: `scale_qwen3_14b/Qwen3-14B` and `scale_qwen25_32b/Qwen2.5-32B-Instruct` are the vendors' post-trained models
(Qwen3-14B and Qwen2.5-32B-Instruct, source directories named `r1`), not round-1 checkpoints of our training; the
`uv/pool_s0/*` and `uv/pool_s1/*` passes used the abstention prompt with admission on p_mode (R-Zero band), not the
loop_uv rule; admission on max(p_mode, abstain) can be recomputed from the stored `p_mode` and `abstain`.

| file | audited | S / U / A | proposer | checkpoints | pool |
|---|---|---|---|---|---|
| fixed_pool_audits/qwen_s0_r1.jsonl | 1158 of 1158 | 639 / 339 / 180 | Qwen3-4B-Base | 19 | all round-1 raw proposals of the Qwen3-4B own seed-0 lineage (frozen proposer); mechanism train pool (A3 2d) |
| fixed_pool_audits/qwen_s1_r1.jsonl | 1136 of 1136 | 607 / 351 / 178 | Qwen3-4B-Base | 15 | all round-1 raw proposals of the Qwen3-4B own seed-1 lineage; held-out validation pool (A4a.1) |
| fixed_pool_audits/front_q25_7b.jsonl | 1199 of 1199 | 779 / 269 / 151 | Qwen2.5-7B-Instruct | 6 | 1,199-candidate pool (from 1,200 requests) of a stronger frozen proposer (same prompt, seed 7); proposer-side escape test (A4c) |
| fixed_pool_audits/evo_r3.jsonl | 1106 of 1106 | 643 / 270 / 193 | Qwen3-4B-Base (rewriting) | 3 | harder variants written by the frozen proposer from the round-3 solver's unanimous qwen_s0_r1 candidates (A4e) |

*raw_proposal_audits/* (200-item Random(0) sample of the raw, pre-guard proposals of the co-evolving proposer arms):

| file | audited | S / U / A |
|---|---|---|
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s0_r2.jsonl | 200 of 1172 | 116 / 50 / 34 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s0_r3.jsonl | 200 of 1183 | 118 / 51 / 31 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s1_r2.jsonl | 200 of 1180 | 133 / 44 / 23 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s1_r3.jsonl | 200 of 1197 | 107 / 53 / 40 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s2_r2.jsonl | 200 of 1182 | 119 / 52 / 29 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s2_r3.jsonl | 200 of 1193 | 104 / 56 / 40 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s3_r2.jsonl | 200 of 1181 | 124 / 54 / 22 |
| raw_proposal_audits/qwen3_4b_Base_coev_a4_s3_r3.jsonl | 200 of 1192 | 110 / 46 / 44 |
| raw_proposal_audits/qwen3_4b_Base_coev_ag_a4_s0_r2.jsonl | 200 of 1179 | 114 / 62 / 24 |
| raw_proposal_audits/qwen3_4b_Base_coev_ag_a4_s0_r3.jsonl | 200 of 1187 | 113 / 54 / 33 |
| raw_proposal_audits/qwen3_4b_Base_coev_ag_a4_s1_r2.jsonl | 200 of 1177 | 128 / 44 / 28 |
| raw_proposal_audits/qwen3_4b_Base_coev_ag_a4_s1_r3.jsonl | 200 of 1194 | 121 / 50 / 29 |

*twins/*: 3336 GSM-HP premise-deletion twins (an intact problem and the same problem with one premise sentence
removed; split kinds: 3192 value, 144 template), of which 543 form the registered probe set. Texts are NOT
included (see Distribution); `twin_patches.jsonl` rebuilds them exactly from GSM-Symbolic (2.2%
of the characters are literals: numbers, number words, names), and `rebuild_twins.py` writes a drop-in `--twins_file` for
`probe/code/c0_probe.py`. Validation: pass 1 (28 Opus judges, intact shown) on the probe set: 412 UNSOLVABLE, 131 SOLVABLE_DIFFERENT;
pass 2 (10 judges, solve-first, deleted text only) on 150 stratified probe items: 34 solvable
(pass 1 vs pass 2: n=150 agreement=0.993 kappa=0.981 pass1_unsolvable=0.767 pass2_unsolvable=0.773 tp=115 fn=0 fp=1 tn=34); a random 200 of all 3,336 twins (the first 200 of the hp_deleted500 random
sample): 177 UNSOLVABLE, 23 SOLVABLE_DIFFERENT. `tracer_guard_records.jsonl` (76400 rows) gives the solver
statistics of the 200 deleted + 200 intact tracers planted in every audited round's pool and in every fixed-pool guard
pass (ids only).

*judge/*: exact prompts, PROTOCOL.md, and validation sets (Section 4). *probe/*: judge-free probe code, the 1,200-item
UMWP subset (ids, labels, pairing), 84 UMWP dumps (84 complete) with a
per-dump summary, the twin dumps used by J-P3, the J-P1 / J-P2 / J-P3 read-outs and `reproduce_j1.py`, which recomputes
J-P2 and J-P3 from the released dumps alone (probe/README.md).

**Per-file counts of audited_pools/** (INDEX.tsv has the full metadata; MANIFEST.tsv has rows and bytes of every file):

| file | rows | S / U / A (all audited rows) | coverage | identical pool as |
|---|---|---|---|---|
| audited_pools/llama3p2_3b_own_a3_s0_r1.jsonl | 823 | 264 / 424 / 135 | full |  |
| audited_pools/llama3p2_3b_own_a3_s1_r1.jsonl | 811 | 269 / 404 / 138 | full |  |
| audited_pools/llama3p2_3b_own_a3_s2_r1.jsonl | 200 | 70 / 101 / 29 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s2_r2.jsonl | 200 | 51 / 125 / 24 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s2_r3.jsonl | 200 | 57 / 108 / 35 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s3_r1.jsonl | 200 | 66 / 97 / 37 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s3_r2.jsonl | 200 | 53 / 117 / 30 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s3_r3.jsonl | 200 | 62 / 115 / 23 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s4_r1.jsonl | 200 | 66 / 96 / 38 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s4_r2.jsonl | 200 | 47 / 120 / 33 | sample200 |  |
| audited_pools/llama3p2_3b_own_a3_s4_r3.jsonl | 200 | 56 / 95 / 49 | sample200 |  |
| audited_pools/llama3p2_3b_own_clean_a3_s0_r2.jsonl | 723 | 205 / 364 / 154 | full |  |
| audited_pools/llama3p2_3b_own_clean_a3_s0_r3.jsonl | 810 | 219 / 404 / 187 | full |  |
| audited_pools/llama3p2_3b_own_sub_a3_s0_r2.jsonl | 200 | 41 / 129 / 30 | sample200 |  |
| audited_pools/llama3p2_3b_own_sub_a3_s0_r3.jsonl | 200 | 53 / 104 / 43 | sample200 |  |
| audited_pools/llama3p2_3b_own_sub_a3_s2_r2.jsonl | 200 | 52 / 113 / 35 | sample200 |  |
| audited_pools/llama3p2_3b_own_sub_a3_s2_r3.jsonl | 200 | 61 / 97 / 42 | sample200 |  |
| audited_pools/llama3p2_3b_own_tg_a3_s0_r1.jsonl | 200 | 94 / 65 / 41 | sample200 |  |
| audited_pools/llama3p2_3b_own_tg_a3_s1_r1.jsonl | 200 | 99 / 62 / 39 | sample200 |  |
| audited_pools/llama3p2_3b_own_tg_a3_s1_r2.jsonl | 200 | 77 / 83 / 40 | sample200 |  |
| audited_pools/llama3p2_3b_own_tg_a3_s1_r3.jsonl | 200 | 59 / 110 / 31 | sample200 |  |
| audited_pools/phi4mini_own_a4_s0_r1.jsonl | 200 | 81 / 53 / 66 | sample200 |  |
| audited_pools/phi4mini_own_a4_s0_r2.jsonl | 200 | 47 / 103 / 50 | sample200 |  |
| audited_pools/phi4mini_own_a4_s0_r3.jsonl | 409 | 73 / 222 / 114 | full |  |
| audited_pools/phi4mini_own_a4_s1_r1.jsonl | 200 | 78 / 66 / 56 | sample200 |  |
| audited_pools/phi4mini_own_a4_s1_r2.jsonl | 200 | 48 / 103 / 49 | sample200 |  |
| audited_pools/phi4mini_own_a4_s1_r3.jsonl | 480 | 108 / 264 / 108 | full |  |
| audited_pools/phi4mini_own_ag_a4_s0_r1.jsonl | 200 | 97 / 51 / 52 | sample200 |  |
| audited_pools/phi4mini_own_ag_a4_s0_r2.jsonl | 200 | 61 / 77 / 62 | sample200 |  |
| audited_pools/phi4mini_own_ag_a4_s0_r3.jsonl | 200 | 48 / 89 / 63 | sample200 |  |
| audited_pools/phi4mini_own_ag_a4_s1_r1.jsonl | 200 | 93 / 55 / 52 | sample200 |  |
| audited_pools/phi4mini_own_ag_a4_s1_r2.jsonl | 200 | 54 / 96 / 50 | sample200 |  |
| audited_pools/phi4mini_own_ag_a4_s1_r3.jsonl | 200 | 57 / 90 / 53 | sample200 |  |
| audited_pools/phi4mini_rescue_uv_a4_s0_r3.jsonl | 409 | 73 / 222 / 114 | full | phi4mini_own_a4_s0_r3 |
| audited_pools/phi4mini_rescue_uv_a4_s1_r3.jsonl | 480 | 108 / 264 / 108 | full | phi4mini_own_a4_s1_r3 |
| audited_pools/qwen3_14b_Base_own_a4_s0_r1.jsonl | 200 | 86 / 75 / 39 | sample200 |  |
| audited_pools/qwen3_14b_Base_own_a4_s0_r2.jsonl | 200 | 56 / 87 / 57 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s0_r1.jsonl | 730 | 343 / 268 / 119 | full | qwen3_4b_Base_own_a3_s0_r1 |
| audited_pools/qwen3_4b_Base_coev_a4_s0_r2.jsonl | 200 | 57 / 95 / 48 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s0_r3.jsonl | 200 | 51 / 110 / 39 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s1_r1.jsonl | 736 | 337 / 305 / 94 | full | qwen3_4b_Base_own_a3_s1_r1 |
| audited_pools/qwen3_4b_Base_coev_a4_s1_r2.jsonl | 200 | 59 / 94 / 47 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s1_r3.jsonl | 200 | 49 / 92 / 59 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s2_r1.jsonl | 718 | 319 / 268 / 131 | full | qwen3_4b_Base_own_a3_s2_r1 |
| audited_pools/qwen3_4b_Base_coev_a4_s2_r2.jsonl | 200 | 59 / 97 / 44 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s2_r3.jsonl | 200 | 47 / 104 / 49 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s3_r1.jsonl | 698 | 300 / 263 / 135 | full | qwen3_4b_Base_own_a3_s3_r1 |
| audited_pools/qwen3_4b_Base_coev_a4_s3_r2.jsonl | 200 | 66 / 86 / 48 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_a4_s3_r3.jsonl | 200 | 59 / 81 / 60 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_ag_a4_s0_r1.jsonl | 200 | 94 / 53 / 53 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_ag_a4_s0_r2.jsonl | 200 | 78 / 69 / 53 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_ag_a4_s0_r3.jsonl | 200 | 63 / 73 / 64 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_ag_a4_s1_r1.jsonl | 200 | 118 / 43 / 39 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_ag_a4_s1_r2.jsonl | 200 | 74 / 62 / 64 | sample200 |  |
| audited_pools/qwen3_4b_Base_coev_ag_a4_s1_r3.jsonl | 200 | 57 / 75 / 68 | sample200 |  |
| audited_pools/qwen3_4b_Base_ctrl_lowp_a5_s0_r1.jsonl | 200 | 96 / 62 / 42 | sample200 |  |
| audited_pools/qwen3_4b_Base_ctrl_lowp_a5_s2_r1.jsonl | 200 | 102 / 63 / 35 | sample200 |  |
| audited_pools/qwen3_4b_Base_ctrl_rnd_a5_s0_r1.jsonl | 200 | 88 / 72 / 40 | sample200 |  |
| audited_pools/qwen3_4b_Base_ctrl_rnd_a5_s1_r1.jsonl | 200 | 86 / 65 / 49 | sample200 |  |
| audited_pools/qwen3_4b_Base_ctrl_rnd_a5_s2_r1.jsonl | 200 | 95 / 75 / 30 | sample200 |  |
| audited_pools/qwen3_4b_Base_loop_uv_a4_s0_r4.jsonl | 200 | 60 / 97 / 43 | sample200 |  |
| audited_pools/qwen3_4b_Base_loop_uv_a4_s0_r5.jsonl | 200 | 39 / 112 / 49 | sample200 |  |
| audited_pools/qwen3_4b_Base_loop_uv_a4_s1_r4.jsonl | 200 | 39 / 98 / 63 | sample200 |  |
| audited_pools/qwen3_4b_Base_loop_uv_a4_s1_r5.jsonl | 200 | 38 / 112 / 50 | sample200 |  |
| audited_pools/qwen3_4b_Base_loop_uv_a4_s2_r4.jsonl | 200 | 42 / 106 / 52 | sample200 |  |
| audited_pools/qwen3_4b_Base_loop_uv_a4_s2_r5.jsonl | 200 | 35 / 113 / 52 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s0_r1.jsonl | 730 | 343 / 268 / 119 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s0_r2.jsonl | 200 | 55 / 116 / 29 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s0_r3.jsonl | 419 | 95 / 262 / 62 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s0_r4.jsonl | 200 | 39 / 131 / 30 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s0_r5.jsonl | 200 | 38 / 130 / 32 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s0_r6.jsonl | 200 | 34 / 128 / 38 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s1_r1.jsonl | 736 | 337 / 305 / 94 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s1_r2.jsonl | 200 | 49 / 113 / 38 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s1_r3.jsonl | 435 | 88 / 251 / 96 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s1_r4.jsonl | 200 | 40 / 131 / 29 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s1_r5.jsonl | 200 | 52 / 118 / 30 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s1_r6.jsonl | 200 | 35 / 111 / 54 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s2_r1.jsonl | 718 | 319 / 268 / 131 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s2_r2.jsonl | 200 | 65 / 106 / 29 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s2_r3.jsonl | 415 | 106 / 236 / 73 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s2_r4.jsonl | 200 | 43 / 104 / 53 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s2_r5.jsonl | 200 | 39 / 117 / 44 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s2_r6.jsonl | 200 | 47 / 113 / 40 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s3_r1.jsonl | 698 | 300 / 263 / 135 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s3_r2.jsonl | 200 | 56 / 105 / 39 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s3_r3.jsonl | 424 | 83 / 251 / 90 | full |  |
| audited_pools/qwen3_4b_Base_own_a3_s4_r1.jsonl | 200 | 85 / 84 / 31 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s4_r2.jsonl | 200 | 48 / 118 / 34 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_a3_s4_r3.jsonl | 397 | 84 / 232 / 81 | full |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s0_r1.jsonl | 200 | 104 / 50 / 46 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s0_r2.jsonl | 200 | 67 / 76 / 57 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s0_r3.jsonl | 200 | 74 / 73 / 53 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s0_r4.jsonl | 200 | 66 / 84 / 50 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s1_r1.jsonl | 200 | 123 / 41 / 36 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s1_r2.jsonl | 200 | 66 / 70 / 64 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s1_r3.jsonl | 200 | 62 / 79 / 59 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s1_r4.jsonl | 200 | 43 / 107 / 50 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s2_r1.jsonl | 200 | 124 / 44 / 32 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s2_r2.jsonl | 200 | 71 / 70 / 59 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s2_r3.jsonl | 200 | 70 / 69 / 61 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_ag_a4_s2_r4.jsonl | 200 | 62 / 80 / 58 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_azr_a3_s0_r1.jsonl | 200 | 100 / 73 / 27 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_azr_a3_s0_r2.jsonl | 200 | 65 / 105 / 30 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_azr_a3_s0_r3.jsonl | 200 | 53 / 111 / 36 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_azr_a3_s1_r1.jsonl | 200 | 99 / 68 / 33 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_azr_a3_s1_r2.jsonl | 200 | 75 / 99 / 26 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_azr_a3_s1_r3.jsonl | 200 | 66 / 87 / 47 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s0_r2.jsonl | 481 | 111 / 305 / 65 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s0_r3.jsonl | 400 | 72 / 241 / 87 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s1_r2.jsonl | 452 | 128 / 241 / 83 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s1_r3.jsonl | 444 | 108 / 221 / 115 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s2_r2.jsonl | 500 | 109 / 275 / 116 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s2_r3.jsonl | 416 | 92 / 222 / 102 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s3_r2.jsonl | 480 | 116 / 233 / 131 | full |  |
| audited_pools/qwen3_4b_Base_own_clean_a3_s3_r3.jsonl | 419 | 76 / 238 / 105 | full |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s0_r2.jsonl | 200 | 62 / 105 / 33 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s0_r3.jsonl | 200 | 38 / 113 / 49 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s1_r2.jsonl | 200 | 55 / 108 / 37 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s1_r3.jsonl | 200 | 52 / 109 / 39 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s2_r2.jsonl | 200 | 47 / 113 / 40 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s2_r3.jsonl | 200 | 47 / 99 / 54 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s3_r2.jsonl | 200 | 52 / 97 / 51 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_sub_a3_s3_r3.jsonl | 200 | 43 / 104 / 53 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_tg_a3_s0_r1.jsonl | 200 | 113 / 59 / 28 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_tg_a3_s0_r2.jsonl | 200 | 84 / 79 / 37 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_tg_a3_s0_r3.jsonl | 200 | 62 / 101 / 37 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_tg_a3_s1_r1.jsonl | 200 | 102 / 74 / 24 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_tg_a3_s1_r2.jsonl | 200 | 97 / 67 / 36 | sample200 |  |
| audited_pools/qwen3_4b_Base_own_tg_a3_s1_r3.jsonl | 200 | 86 / 70 / 44 | sample200 |  |
| audited_pools/qwen3_4b_Base_rescue_abs_a4_s0_r3.jsonl | 419 | 95 / 262 / 62 | full | qwen3_4b_Base_own_a3_s0_r3 |
| audited_pools/qwen3_4b_Base_rescue_abs_a4_s1_r3.jsonl | 435 | 88 / 251 / 96 | full | qwen3_4b_Base_own_a3_s1_r3 |
| audited_pools/qwen3_4b_Base_rescue_abs_a4_s2_r3.jsonl | 415 | 106 / 236 / 73 | full | qwen3_4b_Base_own_a3_s2_r3 |
| audited_pools/qwen3_4b_Base_rescue_uv_a4_s0_r3.jsonl | 419 | 95 / 262 / 62 | full | qwen3_4b_Base_own_a3_s0_r3 |
| audited_pools/qwen3_4b_Base_rescue_uv_a4_s1_r3.jsonl | 435 | 88 / 251 / 96 | full | qwen3_4b_Base_own_a3_s1_r3 |
| audited_pools/qwen3_4b_Base_rescue_uv_a4_s2_r3.jsonl | 415 | 106 / 236 / 73 | full | qwen3_4b_Base_own_a3_s2_r3 |
| audited_pools/qwen3_4b_Base_rescue_uv_a4_s4_r3.jsonl | 397 | 84 / 232 / 81 | full | qwen3_4b_Base_own_a3_s4_r3 |
| audited_pools/qwen3_4b_Base_rescue_uvw_a4_s0_r3.jsonl | 419 | 95 / 262 / 62 | full | qwen3_4b_Base_own_a3_s0_r3 |
| audited_pools/qwen3_4b_Base_rescue_uvw_a4_s1_r3.jsonl | 435 | 88 / 251 / 96 | full | qwen3_4b_Base_own_a3_s1_r3 |
| audited_pools/qwen3_8b_Base_own_a3_s0_r1.jsonl | 200 | 85 / 72 / 43 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s0_r2.jsonl | 200 | 64 / 103 / 33 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s0_r3.jsonl | 200 | 47 / 119 / 34 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s1_r1.jsonl | 200 | 84 / 76 / 40 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s1_r2.jsonl | 200 | 61 / 106 / 33 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s1_r3.jsonl | 200 | 61 / 113 / 26 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s1_r4.jsonl | 200 | 43 / 116 / 41 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s1_r5.jsonl | 200 | 46 / 101 / 53 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s1_r6.jsonl | 200 | 45 / 103 / 52 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s2_r1.jsonl | 200 | 95 / 70 / 35 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s2_r2.jsonl | 200 | 67 / 102 / 31 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s2_r3.jsonl | 200 | 52 / 112 / 36 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s2_r4.jsonl | 200 | 49 / 99 / 52 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s2_r5.jsonl | 200 | 54 / 109 / 37 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_a3_s2_r6.jsonl | 200 | 50 / 114 / 36 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_ag_a4_s1_r1.jsonl | 200 | 118 / 39 / 43 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_ag_a4_s1_r2.jsonl | 200 | 80 / 71 / 49 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_ag_a4_s2_r1.jsonl | 200 | 131 / 33 / 36 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_ag_a4_s2_r2.jsonl | 200 | 77 / 71 / 52 | sample200 |  |
| audited_pools/qwen3_8b_Base_own_ag_a4_s2_r3.jsonl | 200 | 72 / 73 / 55 | sample200 |  |

**Missing information / gaps.** No gold answers exist (the loops are label-free); SOLVABLE notes often state the
judge's answer, which was not verified. Solver statistics are G = 8 samples (resolution 1/8). Later-round proposals were
not scored by the base model. Rounds were audited where compute allowed (see the lineage table); Qwen3-4B-Base own
lineages reach round 6 for seeds 0-2, Qwen3-8B-Base own seeds 1-2 reach round 6. Admitted pools of the conditions
above that had no Opus audit at the snapshot time: qwen3_4b_Base_ctrl_lowp_a5_s1_r1 (training not finished), qwen3_4b_Base_ctrl_lowp_a5_s2_r2, qwen3_4b_Base_ctrl_rnd_a5_s0_r2 (training not finished), qwen3_4b_Base_ctrl_rnd_a5_s1_r2, qwen3_4b_Base_ctrl_rnd_a5_s2_r2, qwen3_4b_Base_own_ag_a4_s1_r5, qwen3_4b_Base_own_ag_a4_s2_r5 (training not finished), qwen3_8b_Base_own_ag_a4_s1_r3 (training not finished); a later build will include them once
audited. Training sets built from audited pools and not audited themselves are not listed as pools; their items are rows
of the parent pool files where the parent is fully audited:

| derived training sets | parent pools | sets | proposer items | with a verdict | not fully covered |
|---|---|---|---|---|---|
| own_clean_a3 round 1 | own_a3 round 1 | 6 | 1,832 | 1,832 | none |
| own_dirty_a3 round 1 | own_a3 round 1 | 4 | 1,887 | 1,887 | none |
| own_sub_a3 round 1 | own_a3 round 1 | 7 | 2,675 | 2,360 | llama3p2_3b_own_sub_a3_s2_r1: 100 of 415 (parent llama3p2_3b_own_a3_s2_r1, sample200) |
| cln1_a6 round 3 | own_a3 / own_a4 round 3 | 7 | 637 | 637 | none |
| rnd1_a6 round 3 | own_a3 / own_a4 round 3 | 7 | 790 | 790 | none |
| swap_c1_a5 round 3 | own_a3 round 3 | 5 | 1,268 | 1,268 | none |
| swap_g1_a5 round 3 | own_a3 round 3 | 5 | 1,552 | 1,552 | none |
| tg_abs4_a5 round 4 | loop_uv_a4 round 4 | 3 | 1,316 | 600 | qwen3_4b_Base_tg_abs4_a5_s0_r4: 200 of 460 (parent qwen3_4b_Base_loop_uv_a4_s0_r4, sample200); qwen3_4b_Base_tg_abs4_a5_s1_r4: 200 of 432 (parent qwen3_4b_Base_loop_uv_a4_s1_r4, sample200); qwen3_4b_Base_tg_abs4_a5_s2_r4: 200 of 424 (parent qwen3_4b_Base_loop_uv_a4_s2_r4, sample200) |
| tg_t4_a5 round 4 | loop_uv_a4 round 4 | 3 | 871 | 395 | qwen3_4b_Base_tg_t4_a5_s0_r4: 131 of 300 (parent qwen3_4b_Base_loop_uv_a4_s0_r4, sample200); qwen3_4b_Base_tg_t4_a5_s1_r4: 132 of 292 (parent qwen3_4b_Base_loop_uv_a4_s1_r4, sample200); qwen3_4b_Base_tg_t4_a5_s2_r4: 132 of 279 (parent qwen3_4b_Base_loop_uv_a4_s2_r4, sample200) |

Not included at all: model checkpoints, full solver rollouts (only the per-item statistics), the twin and UMWP question
texts (licences, Section 6), the expert labels (Section 4) and the key of the sealed human-validation packet.

**Relationships.** Round t of a lineage is proposed by the lineage's proposer (frozen base model, except the coev arms)
and admitted by the solver trained in round t-1 (`solver_checkpoint`). Pools of different conditions with the same
model, seed and round draw on largely the same proposals: `a3_propose.py` samples with seed seed*100+round in every
condition, and its index counts only the outputs it keeps, so the same text often carries a different id. Ids
(`prop<seed*100+round>:<index>`) are therefore unique only within a run: the same (model, id) pair names different texts
in different runs in 4972 of 26738 cases (for example, the admitted
pools of qwen3_4b_Base_own_a3_s0_r1 and qwen3_4b_Base_own_azr_a3_s0_r1 share 106 texts, only
2 of them under the same id, while 124
ids name a different text in each). Join rows across runs on `problem_sha256`, not on `id`.
5831 texts occur in more than one distinct pool; each audit judged them separately
(Section 4). Tracer ids `trd:<fam>:<task>` / `tri:<fam>:<task>` point into `twins/twin_patches.jsonl`.

**Confidentiality, personal or sensitive content.** Names in the problems are mostly generic first names chosen by the
proposers, but base models sometimes continue a problem with copied web text, so not every name or string is
model-invented. A regex screen of the 34187 unique problem texts of this release (audited, fixed and
raw pools) found 38 texts with URLs, 1 with an e-mail address and
2 with a copied Stack Exchange comment thread that names its commenters (in audited_pools
alone: 38 with URLs, 1 with an e-mail address,
2 with a comment thread). These spans are masked in the released text: `[URL]`,
`[EMAIL]`, and `[USER]` for commenter names and @-handles inside the copied threads; 71 rows carry
`redacted` = true, and `problem_sha256` is the hash of the masked text. The Opus verdicts were given on the unmasked
text. Personal names were not screened otherwise, and the texts were not screened systematically for offensive content
(none was observed in the audit notes).

## 3. Collection process

- Proposals: each round the proposer (base model, zero-shot) receives "Write one new, challenging grade-school math word
  problem about <stem>. It must require several arithmetic steps and have a single numeric answer. Output only the
  problem text, no solution." (40 stems, 1,200 requests, T = 1.0, top-p 0.95, 300 tokens; `probe/code/a3_propose.py`).
- Admission (`probe/code/a3_guard.py`): the current solver samples G = 8 answers (T = 1.0, top-p 1.0, 1,024 tokens) for
  every proposal and for 400 planted tracers; the R-Zero band admits p_mode in [0.25, 0.75]; the other guards and the
  loop_uv admission on max(p_mode, abstain) are listed in the condition table and in `admission_stat`.
- Audits: Claude Opus judges, frozen prompt (judge/PROTOCOL.md), 20-25 items per independent judge instance, 22-24
  September 2026. The audit sample is the first 200 of a Random(0) shuffle of the admitted proposer questions; merging
  scripts in `judge/code/`.
- Twins: rule-based deletion of one premise sentence from GSM-Symbolic instances with resampled numbers (GSM-HP recipe,
  CPU only, no LLM).
- UMWP probe (`probe/code/umwp_probe.py`): 600 unanswerable UMWP questions (Random(0) sample of those whose answerable
  originals exist) and their 600 answerable originals, G = 8 samples, T = 1.0, seed 17, plain and abstention prompts.
- No human subjects were involved in the data. An expert annotation is user-reported but not documented or released
  (Sections 4 and 8).

## 4. Preprocessing, cleaning, labelling

Problem texts are the proposers' outputs exactly as admitted (the training suffix "Let's think step by step and output
the final answer within \boxed{}." is not included), except for the masked URL, e-mail and commenter spans
(`redacted`). No deduplication, spelling correction or filtering was applied; some texts contain artefacts of
base-model generation (a leaked "Result: 1058", a trailing note, a repeated question sentence, copied web text); the
judge labels such texts as written. Labels come only from the Opus audit; the raw part files are merged
first-verdict-wins. Released derived fields: shares in INDEX.tsv. UMWP rows keep only ids, labels and our model outputs;
the `ref` (gold answer) field of our dumps is dropped, `correct` is kept.

Judge validity (all in `judge/validation/`, summary.json). No audit was blind to run, round or condition (judge/PROTOCOL.md).
- UMWP-400 (200 unanswerable + 200 answerable, stratified): Opus detects 98.5% of the
  human-labelled unanswerable questions (cat 1: 0.984, cat 2: 1.000, cat 3: 0.955, cat 4: 1.000, cat 5: 0.909) and calls 4.5% of the answerable
  ones ill-posed; binary kappa against the human labels 0.94, agreement 0.970.
- Planted tracers (100 intact + 100 premise-deleted twins, same definitions, strictness sentence and framing; fetched
  from `sample.jsonl` in file order, and the judges could see the `trd:` / `tri:` id prefix that marks a deleted or
  intact premise): 88% of intact judged SOLVABLE,
  92% of deleted judged ill-posed (part of the deleted SOLVABLE verdicts
  are genuinely solvable deletions, see the twin validation).
- Opus vs Opus (8 new judges, reversed order, two 200-item samples): llama3p2_3b_own_a3_s3_r2: agreement 0.780, kappa 0.64 (three-way), 0.900 / 0.74 (binary); qwen3_4b_Base_own_a3_s0_r3: agreement 0.815, kappa 0.67 (three-way), 0.915 / 0.76 (binary).
- Fixed-pool audit vs run-level audit on the same admitted items: qwen_s0_r1 vs qwen3_4b_Base_own_a3_s0_r1: 598/730 identical, kappa 0.71 (three-way), 0.786 (binary); qwen_s1_r1 vs qwen3_4b_Base_own_a3_s1_r1: 622/736 identical, kappa 0.75 (three-way), 0.844 (binary).
- The same text in different audits: of the 5831 problem texts that occur in more than one
  distinct pool, 990 (17.0%) received different three-way verdicts and 518 (8.9%) different binary verdicts (SOLVABLE vs
  ill-posed).
- Expert annotation: user-reported kappa 0.77 (binary) and 0.69 (three-way); comparison target (expert vs Opus or
  expert vs expert), item count, annotators and blinding not yet documented; no label file exists, and nothing of it is
  in this release.
- Human validation packet: a sealed 150-item packet (50 per Opus class, drifted rounds) awaits non-author labels; its
  item key is withheld from this release until those labels are in.

## 5. Uses

Intended: measuring curriculum drift of self-play loops; benchmarking ill-posedness detectors, abstention and admission
guards on model-written problems; studying the mechanism with the fixed pools (how one pool's admitted mix changes as the
solver sharpens); reproducing the judge-free UMWP and twin reads. Not intended: training math solvers on the problems
(many are ill-posed and none has a verified answer); treating Opus verdicts as ground truth without the error rates of
Section 4; any use of the rebuilt twins in training corpora (GSM-Symbolic canary). Composition caveats that affect use:
the audit is a 200-item sample for most pools (binomial SE about 3.5 points on a share near 50%); round-1 pools are shared
across several conditions; one judge family.

## 6. Distribution

Planned distribution: with the paper (anonymous repository during review, public afterwards). Proposed licences, to be
confirmed by the authors before release: annotations, generated problem texts and code CC BY 4.0 / MIT, subject to:
- GSM-Symbolic (Apple) is CC BY-NC-ND 4.0 (the pinned snapshot's dataset card: `license: cc-by-nc-nd-4.0`), which does
  not allow sharing adapted material; the twin texts are therefore distributed only as patches plus
  `twins/rebuild_twins.py`, which rebuilds them from the user's own copy of GSM-Symbolic (revision 93b5b3758d9d9841ffe81d6cd2ae2b030685b078) and
  verifies each text's SHA-256. Whether patches with 2.2% literal characters are acceptable is
  for the authors to decide.
- UMWP (Sun et al., LREC-COLING 2024; data CC BY-SA 4.0, code Apache-2.0 (repository README)): https://github.com/Yuki-Asuuna/UMWP, file `data/StandardDataset.jsonl`, retrieved
  2026-09-24 05:30 CST from branch main (head commit 1b3ae0fae01a6c6f178361c716bbb4f7d7d54c3c of 2023-10-21; the
  file was last changed in commit f5eb429a4a0761fe151d55ec4ef7ff10f5434953; SHA-256 e8840e8383357238a08e9c5028e4758ceceb369e1db31c64678b0f851c9c9e73, 5200 rows). Only item
  ids, UMWP labels and our model outputs are included, no question text. The UMWP labels are CC BY-SA 4.0 data, so the
  files that carry them (`probe/umwp_subset1200.jsonl`, `probe/umwp_dumps/`,
  `judge/validation/umwp400_opus_vs_human_labels.jsonl`) should be shared under CC BY-SA 4.0 with attribution.
- Model-generated texts: Qwen3-4B/8B/14B-Base and Qwen2.5-7B-Instruct (Apache-2.0), Phi-4-mini-instruct (MIT),
  Llama-3.2-3B-Instruct (Llama 3.2 Community License and Acceptable Use Policy; we propose to mark the Llama-generated
  pools "Built with Llama"), Claude Opus notes (Anthropic usage policies). The authors should confirm these terms.
- The key of the sealed 150-item human-validation packet is not part of this release and must not be shared until the
  non-author labels are in.

## 7. Maintenance

Maintained by the authors through the paper's repository; errata and new versions get a new release directory and
MANIFEST. `build_release.py` regenerates every file from the project tree deterministically, apart from the snapshot
time.

## 8. Known limitations

- LLM-judge error: Opus misses 1.5% of human-labelled unanswerable UMWP questions
  and flags 4.5% of answerable ones (Opus vs UMWP human labels: binary kappa
  0.94); it flags 12% of intact GSM-HP twins as ill-posed.
  A user-reported expert annotation gives kappa 0.77 (binary) / 0.69 (three-way), but its comparison target (expert vs
  Opus or expert vs expert), item count, annotators and blinding are not yet documented. Two independent Opus panels on
  the same items agree at kappa 0.64-0.67 (three-way) and 0.74-0.76
  (binary) on two 200-item re-judged samples, and at 0.71-0.75 / 0.79-0.84
  on the 1466 round-1 admitted items judged both in the run audit and in the fixed-pool audit;
  of the problem texts judged in several distinct pools, 990 of 5831 (17.0%) got different three-way verdicts.
  The UNSOLVABLE / AMBIGUOUS boundary is the noisiest part: between the two re-judged panels the UNSOLVABLE-only share
  moves by 10.0, 9.5 points, the
  combined ill-posed share by 0.0, 2.5.
- The strictness rule ("an invented default assumption makes it UNSOLVABLE") counts under-specified but conventionally
  answerable problems as ill-posed; results are about this strict notion.
- Framing and blinding: judges were told the model family and, for admitted pools, that a self-play filter admitted the
  items (the UMWP and tracer validation audits used the same framing, with Qwen3-4B-Base as the stated author). Judges
  were also not blind to round or condition: every admitted-pool judge ran `cd <REPO>/dirB/runs/<RUN>` (extension
  audits: a pseudo-run directory) and, in the file-pushing template, returned `run="<RUN>"`; the run name gives model,
  condition, seed and round (for example `qwen3_4b_Base_own_a3_s0_r3`), and the printed proposer ids encode seed and
  round too. The UMWP-400 judges saw `dirB/runs/umwp400_j1` and ids `umwp:<id>`; the
  tracer-200 judges saw `dirA3/results/audit_validation` and ids `trd:` / `tri:`, whose prefix tells whether the premise
  was deleted or kept, so the tracer validation is not a blind test. The framing is constant across rounds and
  conditions, but it may shift the level of the ill-posed share, and knowledge of the round or condition could bias
  the round-by-round comparison; no round-blind audit exists.
- One judge family (Claude Opus); no second LLM judge. The expert annotation is user-reported; its item list and labels
  are not in this release (not present on the compute host).
- Coverage: most pools have a 200-item sample; Llama-3.2-3B lineages s0-s1 are audited at round 1 only; Qwen3-14B-Base
  has one seed (rounds 1-2); later rounds exist only for some lineages; 8 admitted pools were unaudited at the
  snapshot time.
- Solver statistics come from 8 samples and depend on the answer parser (A3b: boxed answer, else '####', else last
  number); instruct models that rarely box are parsed from the last number.
- English, grade-school arithmetic word problems only.
