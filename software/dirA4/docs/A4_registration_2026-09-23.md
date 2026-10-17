# A4 registration (2026-09-23 18:40 CST): fixing the weakest points of the A3 verdict

Motivation [internal note removed]: the A3 result (`dirA3/docs/A3_verdict_2026-09-23.md`) shows
guard-driven curriculum drift with a clean mechanism, but (1) no downstream cost attributable to the drift, (2) no working
fix (the tracer-calibrated band only delays it), (3) a frozen proposer, (4) Qwen only at <= 8B with Llama-3B unstable.
A4 attacks (1)-(4) with three additions to the A3 loop, run on the same infrastructure (`dirA3/code/a4_round1.sh`,
`a4_runner1.sh`, `queue_a4_single.txt`; pair variant for 7-8B models). All A3 conventions stay: frozen base proposer
unless stated, 1,200 candidates per round, 200 + 200 planted tracers, own-reward GRPO 60 steps, 200-item Opus audits of
every admitted set (`audit_trigger.sh`), lenient-extraction accuracy as primary (A3d), collapse rule of A3c.

## Addition 1: abstention guard (arm own_ag) - candidate fix

Mechanism read of A3 (verdict section 2d): ill-posed questions stay inside a solver-relative uncertainty band because the
solver invents an answer at moderate agreement. The A3 abstention-prompt reads showed a label-free signal the band ignores:
under the prompt "If the problem does not provide enough information to determine the answer, output \boxed{unknown}",
Qwen3-4B-Base abstains on 46.6% of unsolvable twins vs 11.7% of intact twins, and after the loop on 70-72% vs 5-9%
(the discrimination IMPROVES with sharpening). The abstention guard samples the G = 8 answers under that prompt and admits a
candidate iff fewer than abs_max = 2/8 of its samples abstain (fixed a priori, no calibration) AND the R-Zero band
p_mode in [0.25, 0.75] holds on the remaining samples (`a3_guard.py --guard abstain --abstain_prompt 1 --abs_max 0.25`).
Training is unchanged (arm own, plain prompt). Lineages: Qwen3-4B-Base seeds 0-2, 3 rounds (single GPU); Qwen3-8B-Base
seeds 1-2, 3 rounds (GPU pairs, host <HOST>); second-family lineages below.

Pilot (no training; run first): the fixed pool of A3 section 2d (1,158 audited candidates + the 400 seed-1 tracers) scored by
the abstention guard with the base and the own-s0 r1..r5 solvers (`/root/guard_diag_ag.sh`, output
`dirA3/results/guard_diag/qwen_s0_ag/`, scorer `a3_guard_diag_score.py qwen_s0_ag`).

Predictions
- PF1 (pilot): the abstention guard's admitted set has an ill-posed share (UNSOLVABLE + AMBIGUOUS) <= 35% at every checkpoint
  base..r5 and rises by < 10 points from base to r5 (plain band: 52.9 -> 81.2); it keeps >= 50% of the well-posed candidates
  the plain band admits at each checkpoint (i.e. the fix is not just "admit nothing").
- PF2 (loop): in own_ag the audited UNSOLVABLE share of admitted proposals is <= 30% at every round and rises < 10 points from
  round 1 to round 3; the deleted/intact tracer admission ratio is <= 0.6 at every round (own: 0.76-1.01 at r1, 1.3-2.2 later).
- PF3 (no cost): own_ag round-3 lenient mean accuracy >= own round-3 - 2 for the same seed (Qwen-4B own r3: 81.9 / 81.5 / 81.8;
  8B: 83.7 / 83.5).
- PF4 (secondary): own_ag plain-prompt Q1u at round 3 <= own's at round 3 (Qwen-4B own: 0.347 / 0.316 / 0.410); abstention
  under the abstention prompt on unsolvable twins >= own's.

## Addition 2: co-evolving proposer (arms coev, coev_ag) - harm and generality of the frozen-proposer result

The proposer is no longer frozen: after each round's solver training, the proposer is fine-tuned (supervised, 2 epochs,
lr 1e-5, `a4_proposer_sft.py`) on the proposals the guard ADMITTED that round (the R-Zero selection pressure: the proposer
learns to write questions the current solver finds "learnable"), and the next round proposes with it. Arm coev uses the
R-Zero band (as own); arm coev_ag uses the abstention guard. Round 1 of coev is identical to own round 1 by construction
(frozen proposer, base solver, same seeds) and reuses its proposals / admitted set. Lineages: Qwen3-4B-Base seeds 0-1,
3 rounds each. Every round's RAW proposal set (before the guard) gets a 200-item audit in addition to the admitted set.

Predictions
- PC1 (the proposer learns ill-posedness): in coev the audited UNSOLVABLE share of the RAW proposals rises by >= 15 points from
  round 1 (frozen proposer: 29% on the A3 pool) to round 3, and the admitted share at round 3 is >= own's round-3 share (64-66%).
- PC2 (harm becomes measurable): coev round-3 lenient mean accuracy <= own round-3 - 2 in both seeds (the curriculum's useful
  signal collapses). If it holds in one seed only: "partial"; in none: the harm remains latent even under co-evolution.
- PC3 (the fix transfers): coev_ag raw-proposal UNSOLVABLE share rises < 10 points r1 -> r3; coev_ag round-3 accuracy >= coev + 2
  in both seeds and >= own - 2.

## Addition 3: second model family (arms own, own_ag on Phi-4-mini-instruct; OLMo-2-7B-Instruct if it runs)

Phi-4-mini-instruct (3.8B, base Q1u 0.852 in cycle 0: the most hallucination-prone family measured) seeds 0-1, arms own and
own_ag, 3 rounds, single GPU. OLMo-2-1124-7B-Instruct seeds 0-1, same arms, on GPU pairs if the trainer runs it (smoke test
first; if it does not, this is reported as not run).

Predictions
- PG1 (drift generalises): own on Phi shows the A3 drift signature: deleted/intact tracer ratio rises r1 -> r2 and the audited
  UNSOLVABLE share of admitted proposals rises >= 10 points r1 -> r3, without collapse (collapse = accuracy < 0.5 x base or
  MATH truncation > 0.5; a collapsed lineage reads at the previous round, A3c).
- PG2 (fix generalises): PF2 and PF3 hold on Phi own_ag vs Phi own.

## Decision rules
- KF (fix): PF1 holds and PF2 + PF3 hold on >= 2 of 3 Qwen-4B seeds and on both 8B seeds -> "the abstention guard arrests the
  drift at no accuracy cost": the paper gains a method section. PF2 holds but PF3 fails -> the fix costs accuracy; reported as
  a trade-off. PF1 fails -> the abstention signal does not separate ill-posed candidates on the fixed pool; no method claim.
- KH (harm): PC1 and PC2 hold -> "under a co-evolving proposer the latent drift becomes an accuracy cost, which the abstention
  guard removes (if PC3 holds)". PC1 holds, PC2 fails -> the proposer learns ill-posedness but the solver does not pay for it at
  3 rounds; reported as measured. PC1 fails -> filtered SFT is too weak a selection pressure; reported as measured.
- KG (generality): PG1 on Phi -> the drift is a third-family result; PG2 -> the fix is.
- Anything else is reported as measured. No claim is made from a collapsed lineage.

## Execution
Queue `dirA3/queue_a4_single.txt` (Qwen own_ag s0-s2, coev s0-s1, coev_ag s0-s1, Phi own s0-s1, Phi own_ag s0-s1; rounds 1-3;
`/root/a4_watchdog.sh` on the three hosts, `a4_runner1.sh` per GPU); `queue_a4_8b.txt` for Qwen3-8B own_ag s1-s2 on host <HOST>
pairs (`a4_round.sh`, `a4_runner.sh`). Audits: `audit_trigger.sh` extended to `_a4_` runs and raw-proposal audits of coev rounds.
Readouts: `a4_analyze.py` (A3 tables for A4 lineages) and `a4_reads.py` (PF/PC/PG booleans). Hashes of this file, the code
and every audit in `docs/registration_hashes.txt`.

## Amendment A4a (2026-09-23 19:35 CST, before any of the data below exists)

Disclosed exploratory results (not registered reads):
- Pilot PF1 FAILS: on the fixed pool the abstention guard admits 49.0 / 64.3 / 69.6 / 73.0 / 73.6 / 74.6% ill-posed (base..r5;
  plain band 52.9..81.2). It filters the planted premise-deletion tracers (deleted/intact ratio 0.28-0.47 vs 0.84-1.48) but not
  the proposer's own ill-posed questions: tracer-based validation overstates a guard's effect.
- Signal search on the same pool (`a4_signal_search.py`, post hoc): every admission rule whose ceiling is below unanimity
  (so that every admitted item carries a GRPO gradient) drifts: floor [.5,.875] 32 -> 72%, floor [.5,.875] & no abstention
  22 -> 66%, band & abstention filters 46 -> 77%. Only [.5, 1] (which admits unanimous, zero-gradient items) is flat (28 -> 41%).
  Conjecture ("noisy TV of self-play"): as the solver learns, reducible uncertainty on well-posed questions disappears while
  irreducible uncertainty on ill-posed questions stays, so any curriculum that selects on the solver's residual uncertainty
  converges to ill-posed questions, whatever the label-free filter.

A4a.1 Held-out test of the conjecture (validation pool). Pool: the 1,206 round-1 proposals of Qwen3-4B own s1 (not used in the
search), every one audited by Opus (`dirA3/results/pool_audit/qwen_s1_r1/`), plus the seed-2 tracers. Guard passes with the
base and the own-s1 r1..r5 solvers, plain prompt (R-Zero guard) and abstention prompt (abstention guard), outputs
`dirA3/results/guard_diag/qwen_s1/` and `qwen_s1_ag/`. Rules (fixed now): R1 band [.25,.75]; R2 floor [.5,.875]; R3 floor
[.5,.875] & zero abstention; R4 band & abstention < 2/8; R5 band & zero abstention & same mode under both prompts; R6 [.5, 1].
- NG1: for R1-R5 the ill-posed share of the admitted set rises by >= 15 points from base to r5 on the held-out pool.
- NG2: R6 rises by < 10 points, and the number of admitted SOLVABLE candidates with p_mode < 1 (gradient-bearing) falls by
  >= 50% from base to r5 under R6.
- NG3: in the gradient-bearing region (p_mode <= 7/8), the share of UNSOLVABLE + AMBIGUOUS among candidates with p_mode >= 1/8
  rises monotonically base -> r5 (the residual-uncertainty set converges to ill-posed questions).
NG1-NG3 all hold -> the paper states the no-go result ("label-free guards on the solver's own agreement cannot stop the drift").

A4a.2 Scale test (no training). The fixed pool of A3 section 2d scored by the R-Zero guard with base / post-trained pairs of
larger models: Qwen3-14B-Base vs Qwen3-14B (thinking disabled), Qwen2.5-32B vs Qwen2.5-32B-Instruct (tensor-parallel on 2 GPUs).
Outputs `dirA3/results/guard_diag/scale_<pair>/{base,r1}` (r1 := the post-trained model). Post-training is a strong, external
sharpening; the mechanism predicts the same composition flip.
- NS1: in each pair the post-trained model's band admits a higher ill-posed share than the base model's (by >= 10 points),
  with a lower in-band share of SOLVABLE candidates and a higher mean p_mode on SOLVABLE candidates.
Models are downloaded to the RAM disk of host <HOST>; a pair whose download fails is reported as not run.

A4a.3 Co-evolving proposer, more seeds: coev s2 and s3 (Qwen3-4B, 3 rounds; own comparisons exist: r3 81.8 / 81.4). PC1-PC2 are
read over seeds 0-3 ("both seeds" in PC2 becomes ">= 3 of 4 seeds").

A4a.4 Guard code: `a3_guard.py` gains `--tp` (tensor parallel, default 1) and `--no_think` (chat template with thinking
disabled, default 0); defaults leave every running arm unchanged.

## Amendment A4b (2026-09-23 19:45 CST, before any of the data below exists): the escape condition (frontier supply)

If the drift is the self-play form of the noisy-TV problem (A4a), the only label-free escape is on the proposer side: the
pool must keep supplying WELL-POSED questions at the solver's current frontier, so that the band's well-posed mass does not
vanish as the solver sharpens. Test: replace the frozen Qwen3-4B-Base proposer by a stronger frozen proposer (same prompt,
"challenging grade-school math word problem ... several arithmetic steps ... single numeric answer"), and score its pool
with the SAME solvers as A3 section 2d (Qwen3-4B-Base and own-s0 r1..r5, R-Zero band, seed-1 tracers).
Proposers: Qwen2.5-7B-Instruct (1,200 candidates, seed 7) and, if its download completes, Qwen3-14B (thinking disabled,
seed 14). Every candidate audited by Opus (`dirA3/results/pool_audit/front_<p>/`); guard passes in
`dirA3/results/guard_diag/front_<p>/<ck>`.
- NE1: the stronger proposer's raw pool is less ill-posed than the 4B-Base pool (44.8% UNSOLVABLE + AMBIGUOUS).
- NE2 (escape): with the stronger proposer the band's ill-posed share rises by < 15 points from base to r5 (4B-Base pool:
  +28.3) and the in-band SOLVABLE count at r5 is >= 25% of its base value (4B-Base pool: 65 / 341 = 19%).
NE2 holds -> "proposer quality at the frontier is the lever"; NE2 fails -> the drift is not rescued by a stronger frozen
proposer at these sizes (the no-go extends to proposer strength), and only a proposer that tracks the solver's frontier
(co-evolution, A4 addition 2) remains, which PC1 tests.
`a3_propose.py` gains `--no_think` and `--tp` (defaults unchanged); `a3_guard_diag_score.py` takes the audit-parts dir as
its second argument.

## Amendment A4c (2026-09-23 19:50 CST): confirmatory test of a round-3 harm (seeds not yet run)

Disclosed observation (registered secondary read of A3c, Q1u on the 412 validated unsolvable twins, plain prompt): at round 3
the arm trained only on audit-SOLVABLE admitted items stops rising while the own arm keeps rising:
own_clean 0.240 / 0.218 vs own 0.347 / 0.316 (seeds 0 / 1; rounds 1-2 were indistinguishable: 0.204 / 0.231 vs 0.204 / 0.221,
0.226 / 0.269 vs 0.284 / 0.252), while the random-half arm own_sub shows no consistent shift (0.277 / 0.393). Accuracy is flat
across the three arms (KC2). By round 3 the admitted set is 60-66% unsolvable, so the ill-posed items would add confident
answering of unanswerable inputs that benchmarks do not see.

Confirmatory arms (same A3 protocol, `a3_round1.sh`, queue `dirA3/queue_a4c.txt`, host <HOST>): own_clean seeds 2 and 3 (full
Opus audits of every admitted set before training, as in A3c; the round-1 full audits of own s2 / s3 are being completed),
own_sub seed 3 (own s2 / s3 and own_sub s2 exist). Paired item-level reads by `a4_q1u_attrib.py` (McNemar on the 412 twins).
- HC1: at round 3, Q1u(own_clean) <= Q1u(own) - 0.05 in both new seeds (and in >= 3 of 4 seeds overall), with the paired
  McNemar p < 0.05 in >= 3 of 4 seeds.
- HC2: |Q1u(own_sub) - Q1u(own)| < |Q1u(own_clean) - Q1u(own)| in >= 3 of the 4 seeds with a sub arm read (seeds 0-3); the
  volume halving alone does not produce the effect.
- HC3: lenient mean accuracy of own_clean at round 3 within 2 points of own (KC2 extends; the harm is invisible to accuracy).
HC1-HC3 hold -> "drift-attributable harm at round 3: confident answering of unanswerable inputs, invisible to benchmarks".
HC1 fails in the new seeds -> the s0/s1 observation is not confirmed and is reported as such.

## Amendment A4d (2026-09-23 22:35 CST, before any of its data): an actual self-play loop at 14B

The scale evidence so far is external sharpening (base vs post-trained models, A4a.2). A4d runs the A3 own loop itself with
Qwen3-14B-Base (seed 0; frozen 14B base proposer; R-Zero band; tracers; own-reward GRPO 60 steps on a GPU pair of host <HOST>,
optimizer on CPU as for 8B), round 1 and the round-2 guard (the round-2 training continues only if time allows; solver
evaluations are skipped to fit the compute window). Script `dirA3/code/a4_round14.sh`, lineage `qwen3_14b_Base_own_a4_s0`.
- NL1: the round-2 guard (round-1 14B solver) admits the planted deleted tracers at a higher rate relative to intact ones
  than the round-1 guard (base 14B), and the audited UNSOLVABLE share of admitted proposals rises by >= 10 points r1 -> r2.
If round 1 does not finish before 07:30 CST, A4d is reported as not completed.

## Amendment A4e (2026-09-23 22:50 CST, before any of its data): frontier refill by evolving mastered questions

The law (drift = well-posed questions sharpen out of the band faster than ill-posed ones) implies the escape is to keep
well-posed mass at the frontier. Label-free source of well-posed seeds: questions the current solver answers unanimously
(fixed pool, round-3 solver: 80.1% SOLVABLE, 11.1% UNSOLVABLE; round 5: 77.5% / 12.9%). Frontier refill: the frozen proposer
(Qwen3-4B-Base, few-shot) rewrites each unanimous question into a harder variant ("add one or two steps, keep every number
needed, single numeric answer"), two variants per seed (`dirA3/code/a4_evolve.py`). Offline test on the fixed pool: seeds =
the round-3 solver's unanimous candidates; the evolved pool is scored by the round-3, round-4 and round-5 solvers (R-Zero band,
seed-1 tracers; `dirA3/results/guard_diag/evo_r3/<ck>`), and every evolved candidate is audited by Opus
(`dirA3/results/pool_audit/evo_r3/`).
- NF1: at the round-3 solver the band admits an evolved set that is <= 45% ill-posed (fresh pool: 77.9%) and contains >= 150
  SOLVABLE candidates per 1,000 evolved candidates (fresh pool: 85 per 1,158).
- NF2: at the round-5 solver the evolved set admitted by the band is <= 50% ill-posed (fresh pool: 81.2%).
NF1 and NF2 hold -> frontier refill is a label-free escape at the pool level, to be tested in the loop (arm own_evo) if time
allows; otherwise reported as a pool-level result only. NF1 fails -> the proposer cannot refill the frontier with well-posed
questions at this size, and the no-go extends to self-generated refills.

## Amendment A4f (2026-09-24 01:55 CST, before any of its data): abstention-aware reward on the drifted curriculum ("rescue at round 3")

The no-go (A4a) says the curriculum cannot be kept clean with label-free guards. A different lever is the reward: by round 3
the admitted curriculum is ~65% ill-posed, so a majority vote that may choose "unknown" (arm own_abs of A2: the abstention
option "If the problem does not provide enough information to determine the answer, output \boxed{unknown}" in the training
prompt, majority-vote reward) could turn the drifted curriculum into label-free abstention training instead of confident
hallucination. Clean causal test: start from the SAME own round-2 solver and train on the SAME own round-3 admitted set
(copied), changing only the reward (own -> own_abs); 60 steps; standard evaluations. Arm rescue_abs, Qwen3-4B seeds 1 and 2
(seed 0 if a GPU frees in time); script `dirA3/code/a4_rescue.sh`. Comparators: own round 3 (same start, same data) and
own_clean round 3 (seeds 0-3 where available).
- HA1 (harm removed): plain-prompt Q1u at round 3 of rescue_abs <= own round 3 - 0.05 in every seed run.
- HA2 (no accuracy cost): lenient mean accuracy (MATH-500 / held-out-1000 / GSM8K) >= own round 3 - 2 in every seed run.
- HA3 (abstention without over-abstention): under the abstention prompt, abstention on the 412 unsolvable twins >= own round 3
  + 10 points, and abstention on intact twins <= 15%.
HA1-HA3 hold -> "the drift cannot be prevented label-free, but an abstention-aware reward converts the drifted curriculum into
abstention training at no accuracy cost" (a method claim at the round-3 rescue scale; a full loop is not run in this window).

## Amendment A4g (2026-09-24 04:50 CST, before any of its data): an abstention attractor ("unknown votes"), derived from the sharpening law

Correction to A4f (found at 04:47 while designing this amendment): A4f describes own_abs as "a majority vote that may choose
unknown". In `dirB/code/b_train.py` (first pass of the reward: "abstentions never vote") abstentions are excluded from the
majority and always receive reward 0; own_abs only adds the abstention option to the prompt. HA1-HA3 are read as registered,
for the arm as run; the description is corrected in the verdict.

Rationale. The law says the band enriches ill-posed items because they have no attractor: well-posed items sharpen to unanimity
and leave the band; ill-posed items converge to an interior consensus on an invented answer. A reward in which abstentions vote
gives them one. Under the own r3 solver with the abstention prompt, "unknown" is already the plurality answer on 20.4% of the
audited unsolvable items of the fixed pool against 0.3% of the solvable ones (qwen_s0_ag/r3 records; computed at 04:48 for
this design, no new data). Arm own_uv (flag-gated in b_train.py; all other arms unchanged): abstention prompt; the majority is
taken over all parsed answers including "unknown"; reward 1 for matching it. Design identical to A4f: own round-2 solver, own
round-3 admitted set, 60 steps, Qwen3-4B seeds 0-2; script `dirA3/code/a4_rescue_uv.sh`; evaluations sharded across free GPUs.
- HU1 (harm removed): plain-prompt Q1u of rescue_uv r3 <= own r3 - 0.05 in every seed run.
- HU2 (no accuracy cost): lenient mean accuracy (MATH-500 / held-out-1000 / GSM8K) >= own r3 - 2 in every seed run.
- HU3 (abstention without over-abstention): under the abstention prompt, abstention on the 412 unsolvable twins >= own r3 + 10
  points and on the intact twins <= 15%.
- HU4 (attractor, law-derived): on the seed's fully audited pool (pool s0 for seed 0, pool s1 for seed 1), abstention prompt,
  band [0.25, 0.75] applied to the majority share counting "unknown" as an answer (the admission rule a full own_uv loop would
  use): the ill-posed share admitted with the rescue_uv r3 solver <= that with the own r3 solver (existing qwen_s0_ag/r3 and
  qwen_s1_ag/r3 records, same rule) - 10 points.
- HU5 (descriptive, no threshold): rescue_uv against rescue_abs on Q1u and on the HU4 composition (rescue_abs pool passes are
  exploratory comparators, script `dirA3/code/a4_pool_abs.sh`).
HU1-HU4 hold -> "an abstention attractor derived from the law turns the drifted curriculum into abstention training and stops
the band from enriching ill-posed items" (round-3 rescue scale plus one admission step; a full multi-round loop is not run).
Seed runs without complete evaluations at 07:45 CST are reported as not run.

A4g addendum (2026-09-24 05:03 CST, before any seed-4 data): seed 4 is added to A4g (own s4 round-2 solver, own s4 round-3
admitted set; comparators own s4 r3 Q1u / abstention / accuracy exist). HU1-HU3 apply to every seed run (0, 1, 2, 4); HU4 to
seeds 0 and 1 only. Seed 1 moved to host <HOST> (its GPU on <HOST> was taken by another job at launch; the failed start at 04:50
produced no data).

## Amendment A4h (2026-09-24 05:33 CST, before any of its data): external validity on UMWP

The harm and method reads use our own 412 validated premise-deletion twins. UMWP (Sun et al., LREC-COLING 2024;
CC-BY-SA-4.0; `data_ext/UMWP`) is an independent human-built benchmark of unanswerable math word problems in five categories
(missing key information, ambiguous key information, unrealistic conditions, unrelated object, question missing), each paired
with its answerable original. Sample: 600 unanswerable questions (Random(0) over all 2,600) and their 600 answerable
originals. Probe `dirA3/code/umwp_probe.py`: prompts, parsing and aggregation copied from c0_probe (G = 8, T = 1.0,
max_new 1024), plain and abstention prompt. Metrics: CA = share of unanswerable questions whose modal answer is a number with
p_mode >= 0.5 (plain prompt); ABS_U / ABS_A = mean abstention share under the abstention prompt on unanswerable / answerable
questions; ACC_A = modal-answer accuracy on answerable questions (plain prompt). Checkpoints (Qwen3-4B round 3 unless
stated): own s0-s4, own_clean s0-s3, rescue_abs s0-s2, rescue_uv s0-s2 and s4, the base model, own round 5 s0-s2. Read:
`dirA3/code/umwp_read.py`; launcher `dirA3/code/umwp_launch.sh`.
- HX1 (drift-attributable harm, external): CA(own_clean r3) < CA(own r3) in >= 3 of 4 seeds (0-3).
- HX2 (harm grows under self-play): CA(own r3) > CA(base) in every seed (0-4).
- HX3 (method, external): in every rescue_uv seed run, ABS_U >= ABS_U(own r3) + 10 points, ABS_A <= ABS_A(own r3) + 5
  points and ACC_A >= ACC_A(own r3) - 2.
- HX4: CA(rescue_uv) <= CA(own r3) - 0.05 in every seed run.
Descriptive only: per-category results, rescue_abs, own round 5. Runs incomplete at 07:50 CST are reported as not run.

A4h addendum b (2026-09-24 05:52 CST, before any of its data): the same UMWP sample and probe for two more families.
Qwen3-8B-Base (base model; own round 3, seeds 0-2) and Phi-4-mini-instruct (base model; own round 3, seeds 0-1; the probe
uses the chat template for this instruct model, as c0_probe does). Launcher `dirA3/code/umwp_launch_b.sh` (umwp_launch.sh
plus base-model names of the form base:<model dir>); read `dirA3/code/umwp_read_b.py`.
- HX2b (harm grows under self-play, other families): CA(own r3) > CA(base) in every seed of each family.
Descriptive only: ABS_U, ABS_A, ACC_A; Phi-4-mini own_ag round 3 (seeds 0-1). Runs incomplete at 07:50 CST are reported
as not run.

## Amendment A4i (2026-09-24 06:00 CST, before any of its data): one real loop round with the abstention attractor

A4g tests own_uv as a round-3 rescue. A4i continues the loop by one real round: the round-4 proposals of own seed S (the
same file the own lineage used), admitted by the R-Zero band applied to the majority share counting "unknown" as an answer
(abstention prompt; `a3_guard.py --count_unknown 1`, a flag-gated patch, default off), solver = the rescue_uv round-3
solver of seed S, reward own_uv, 60 steps; guard seed S*100+4 as in own round 4. Seeds 0 and 2. Script
`dirA3/code/a4_loop_uv.sh`. The admitted set gets the standard 200-item Opus audit. Comparator: own round 4 of the same
seed (audited ill-posed 80.5 / 78.5%, unsolvable 65.5 / 52.0%; Q1u 0.430 / 0.502; lenient mean accuracy 82.4 / 81.4).
- HL1 (the curriculum stops training invented answers): the share of audited ill-posed problems whose admitted majority
  answer is not "unknown" (they train an invented answer) among the loop_uv round-4 audit sample <= own round 4's audited
  ill-posed share - 15 points, in each seed.
- HL2 (harm removed across two rounds): plain-prompt Q1u of loop_uv round 4 <= own round 4 - 0.10, in each seed.
- HL3 (no accuracy cost): lenient mean accuracy >= own round 4 - 2, in each seed.
- HL4 (no over-abstention): under the abstention prompt, abstention on intact twins <= 15%, in each seed.
Descriptive: admitted n, tracer ratio, "unknown" plurality by audit class, the plain ill-posed share of admitted problems.
Seed runs incomplete at 07:50 CST are reported as not run (HL1 can be read from the guard and audit alone).

A4i addendum (2026-09-24 06:27 CST, before any seed-1 data): seed 1 is added to A4i (rescue_uv s1 round-3 solver, own s1
round-4 proposals, guard seed 104; comparator own s1 round 4: audited ill-posed 80.0%, unsolvable 65.5%, Q1u 0.323, lenient mean
accuracy 80.9). HL1-HL4 apply to seed 1 unchanged. Compute: all three hosts until 08:30 CST, then host <HOST> is released.

## Amendment A4j (2026-09-24 06:28 CST, before any of its data): a second loop round with the abstention attractor

Continues each A4i lineage (seeds 0, 1, 2) by one more real round: round-5 proposals of own seed S, admission by the band on
the count-unknown majority share (abstention prompt), solver = loop_uv round-4 solver of seed S, reward own_uv, 60 steps,
guard seed S*100+5 (as own round 5); script `dirA3/code/a4_loop_uv_r.sh <seed> 5` (waits for round 4); standard 200-item
Opus audit. Comparator: own round 5 of the same seed (audited ill-posed 81.0 / 74.0 / 80.5%; Q1u 0.478 / 0.345 / 0.571;
lenient mean accuracy 80.5 / 81.0 / 81.8 for seeds 0 / 1 / 2).
- HM1: invented-answer target share (audited ill-posed with a non-unknown majority) of the loop_uv round-5 audit sample <=
  own round 5's audited ill-posed share - 15 points, in each seed read.
- HM2 (no re-drift): the loop_uv invented-answer target share at round 5 <= its round-4 value + 5 points, in each seed read.
- HM3: plain-prompt Q1u of loop_uv round 5 <= own round 5 - 0.10; HM4: lenient mean accuracy >= own round 5 - 2;
  HM5: abstention on intact twins (abstention prompt) <= 15%; each in every seed read.
Host <HOST> is released at 08:30 CST; hosts <HOST> and <HOST> continue. Seed rounds that do not finish are reported as not
run, with the time they stopped; every step is resumable (guard, training checkpoints every 20 steps, per-shard probes).

## Amendment A4k (2026-09-24 06:30 CST, before any of its data): second family for the abstention attractor (Phi-4-mini)

Same design as A4g on Phi-4-mini-instruct: own round-2 solver + own round-3 admitted set, reward own_uv, 60 steps, the same
training flags as the Phi own rounds (--answer_fallback 1 --vllm_inproc 1), seeds 0 and 1, host <HOST> (released at 08:30
CST; seed runs without complete evaluations by 08:28 are reported as not run). Script `dirA3/code/a4_rescue_uv_phi.sh`
(a4_rescue_uv.sh with Phi run names; no pool pass, since no Phi pool is audited). Read `dirA3/code/a4_uv_read_phi.py`.
Comparator Phi own round 3 (same start, same data): Q1u 0.641 / 0.587, lenient mean accuracy 77.8 / 77.8, abstention under
the abstention prompt on unsolvable / intact twins 0.478 / 0.023 and 0.508 / 0.021 (seeds 0 / 1).
- HK1 = HU1, HK2 = HU2, HK3 = HU3 of A4g with the same thresholds, read on Phi seeds 0 and 1.

## Amendment A4l (2026-09-24 06:36 CST, before any of its data): a weighted abstention attractor (w = 3)

Motivation, disclosed: A4g's seed-0 reads (available at 06:35) show the attractor is too weak in one round (HU3 +0.081 vs
the +0.10 bar; HU4 -4.8 vs -10). Under the round-2 solver, "unknown" wins the plain vote on only 23% of audited unsolvable
pool items. Arm own_uvw (flag-gated in b_train.py, `--uv_weight`): an abstention counts w = 3 votes; "unknown" wins when
3 x #unknown > count of the modal numeric answer; reward 1 for samples matching the winner. w = 3 was chosen from existing
guard records only (`dirA3/code/a4l_w_choice.py`): on the two audited pools with the own r2 solvers, "unknown" would win on
35.7 / 34.5% of unsolvable items and 2.5 / 1.2% of solvable items (w = 1: 22.7 / 23.4% and 0.5 / 0.7%). Design identical to
A4g (own r2 solver + own r3 admitted set, 60 steps, abstention prompt), Qwen3-4B seeds 0 and 1, host <HOST>; script
`dirA3/code/a4_rescue_uvw.sh`; read `dirA3/code/a4_uvw_read.py`. Comparators: own round 3 (HW1-HW4 use the A4g thresholds)
and rescue_uv (descriptive).
- HW1 = HU1, HW2 = HU2, HW3 = HU3, HW4 = HU4 (pool s0 for seed 0, pool s1 for seed 1), same thresholds, both seeds.
This is the fourth reward variant tested for the rescue (own, own_abs, own_uv, own_uvw); all four are reported.

A4h addendum c (2026-09-24 06:39 CST, before any of its data): the same UMWP sample and probe for the abstention-guarded
loop at round 3 (own_ag, Qwen3-4B seeds 0-2; PF4 held on the twins for all three). Launcher `umwp_launch_b.sh`, read
`dirA3/code/umwp_read_c.py`.
- HX5: CA(own_ag r3) < CA(own r3) in every seed (0-2).
- HX6: ABS_A(own_ag r3) <= ABS_A(own r3) + 5 points and ACC_A(own_ag r3) >= ACC_A(own r3) - 2, in every seed.
Descriptive: coev_ag r3 seeds 0-1.
Disclosure: the Phi-4-mini own_ag UMWP files (addendum b, descriptive) were already read before addendum c: Phi own_ag r3
CA 0.787 / 0.763 vs Phi own s0 r3 0.787 (no reduction for Phi seed 0). Addendum c predicts only for Qwen3-4B.

A4h addendum d (2026-09-24 07:31 CST, before any of its data): UMWP (same sample and probe) on the own_uv loop solvers at
round 4 (loop_uv seeds 0 and 2; seed 1 if trained before 08:10), run on host <HOST> before its release. DESCRIPTIVE ONLY: no own
round-4 checkpoint survives, so the comparison is with own round 3 and own round 5 of the same seeds (bracketing).
