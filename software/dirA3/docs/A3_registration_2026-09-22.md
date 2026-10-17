# Direction A3, cycle 1 registration - FINAL (2026-09-22 09:25 CST, hashed before the first round; own arm; pdc arm conditional on A2)

## Why
The judge panel's unanimous gap: every harm so far is a latent property (guards admit ill-posed tasks;
sharpening raises confident ill-posedness). Nothing yet shows a self-improvement LOOP - proposer,
published guard, solver trained on what the guard admits - producing a cost the community measures.
A3 runs that loop with tracers planted and measures what drifts and what it costs.

## The loop (per lineage: base model, seed)
Round r = 1..3:
1. Propose: the FROZEN base model writes 1,200 grade-school word problems from 40 stems (fresh sampling
   seed per round; `a3_propose.py`). No labels anywhere in the loop.
2. Guard: the CURRENT solver samples G = 8 answers per candidate; the R-Zero band admits p_mode in
   [0.25, 0.75] (`a3_guard.py`). 200 premise-deleted GSM-HP twins and their 200 intact twins from
   OUTSIDE the registered 543-item probe set are planted in the candidate pool as tracers.
3. Train: the solver is trained on the admitted set (label-free majority-vote reward, 60 steps,
   32 x 8, the B protocol; `b_train.py --arm own`, or `--arm pdc` for the contrast arm), starting from
   the previous round's weights.
4. Evaluate the round's solver: MATH-500 / MATH-heldout-1000 / GSM8K (plain prompt, greedy), GSM-HP
   Q1 on the 412 judge-validated unsolvable twins, abstention controls, no-op control.
Audit (labels used only here): 200 admitted proposer questions per round are judged by the Opus panel
(UNSOLVABLE / SOLVABLE / AMBIGUOUS) -> the ill-posed share of the curriculum per round.

## Arms and grid
Arm `own` = the published loop (guard only). Arm `pdc` = the same loop with the premise-deletion
contrast in the solver update - run ONLY if A2's re-evaluated E2/E3 pass (P2/P3 of A2 with the
patched probe); otherwise A3 is the own-arm cost measurement alone.
Models: Qwen3-4B-Base, Llama-3.2-3B-Instruct. Seeds 0, 1. Rounds 3. = 12 round-trainings per arm
(~1 h each on a GPU pair). Lineage names: `qwen3_4b_Base_<arm>_a3_s<seed>`, `llama3p2_3b_<arm>_a3_s<seed>`.

## Predictions (falsifiable; base rates from cycle 0: Qwen R-Zero band admits unsolvable 0.27 vs intact
0.46, Llama 0.46 vs 0.51; after 60 steps of own on real questions Q1u Qwen 0.104 -> ~0.22, Llama 0.211 -> ~0.25)
P1  Tracer admission: the guard admits planted DELETED tracers at >= 0.8x the intact-tracer rate on
    Llama in every round, and the deleted/intact admission ratio RISES across rounds on both models
    (sharpening makes the solver confident on more ill-posed items -> they enter the band or leave it
    upward; we predict the confident (p_mode >= 0.5) share of deleted tracers rises monotonically).
P2  Curriculum drift: the judged ill-posed share of ADMITTED proposer questions is >= 15% in round 1
    and does not decrease across rounds.
P3  Cost: Q1u on the held-out unsolvable twins rises across rounds (round 3 >= round 1 + 0.05 on both
    models) while held-out MATH accuracy plateaus or falls from round 2 to round 3 (< +1 point); the
    no-op control accuracy falls by >= 3 points from base by round 3 (the loop learns to answer
    regardless of what the text supports).
P4  If the pdc arm runs: it keeps held-out MATH within 2 points of own at every round, keeps Q1u
    within +0.05 of base, and abstains on <= 10% of intact / no-op problems.

## Decision rules
KA3-1  P3 fails on both models (no cost in any measured number by round 3) -> the loop's ill-posed
       curriculum is harmless at this scale; the paper reports the audit + "no measurable downstream
       cost at <= 4B, 3 rounds" honestly; no further loop cycles.
KA3-2  P3 holds on >= 1 model -> cycle 2: a 7-8B lineage (Qwen3-8B-Base) and a trained proposer
       (real R-Zero co-evolution) to test whether the cost persists / grows.
KA3-3  P1/P2 hold but P3 fails -> the curriculum drifts without a downstream cost: report as the
       "latent-harm" paper shape with the drift quantified.

## Not claimed
Nothing about proposer training (frozen proposer here), nothing above 4B in this cycle.

Smoke test 08:50 CST (Qwen3-4B-Base, 40 candidates + 20 tracer pairs): proposer yields 36/40 valid problems; guard admits 50% proposed, 55% deleted tracers, 50% intact tracers; pipeline works end to end.

## Amendment A3a (2026-09-22, hashed before the 8B lineage starts): an 8B guard-only lineage
Motivation: the 8B probe (Qwen3-8B-Base: Q1 0.015 on unsolvable twins, 0.017 of the intact confident
rate; R-Zero band admits 0.23 unsolvable vs 0.39 intact) makes model scale the binding limitation of
every claim. The loop is therefore also run at 8B with the same protocol and endpoints:
lineage `qwen3_8b_Base_own_a3_s0` (frozen 8B proposer, 8B solver, R-Zero band, tracers, 3 rounds,
one seed). Trainer change: `b_train.py --optimizer_device cpu` keeps the fp32 master and AdamW on host
RAM (fp32 gradient buffer on the GPU), the only way an 8B full-parameter update fits one 72 GB card;
nothing else differs. Added prediction P5: the admitted-set ill-posed share and the tracer admission
ratio at 8B are within 0.10 of the 4B values in every round (the guard, not the solver's blind spot,
is what admits ill-posed tasks), while Q1 at 8B stays below 0.10 in all rounds. If P5 holds the
paper's claim is "published guards do not improve with solver scale"; if the 8B admission ratio falls
below 0.5x the 4B value, the claim is bounded to <= 4B and stated so.

## Amendment A3b (2026-09-22 10:35 CST, before any retained round): numeric-answer fallback in the loop
Round 1 showed Llama-3.2-3B-Instruct boxes only ~15% of its answers to GSM-style problems; the boxed-only
parser therefore reported no majority for most candidates and the R-Zero band admitted 9% of proposals
(4% of tracers) - a parser artifact, not a guard property. In the loop (guard AND solver reward, both
models) an unboxed rollout now counts its trailing "####"/last number as its answer (`a3_guard.py`,
`b_train.py --answer_fallback 1`); the B/A2 protocol (boxed only) is unchanged. The two round-1 lineages
started under the old parser (qwen3_4b s0, llama s0) were discarded and restarted; their curriculum
audits (28.5% / 26% unsolvable of admitted proposals) are superseded by the restarted rounds.


## Amendment A3c (2026-09-22 12:30 CST, registered before any result of the added arms): curriculum-attribution controls, extra seeds, 8B replication

Motivation. The round-1 audits show that 55-64% of the proposer questions admitted by the R-Zero band are ill-posed or ambiguous
(Qwen s0 37%+18%, s1 40.5%+15.5%; Llama s0 51%+12.5%, s1 49.5%+20%). Whatever the loop does to the solver must be attributed:
to the ill-posed admitted items, to self-training per se, or to data volume. The following arms are added; proposer, guard,
trainer, evaluations and the three-round schedule are exactly those of A3, only the training subset differs.

Added lineages (Qwen3-4B-Base and Llama-3.2-3B-Instruct unless stated):
- own_sub, seeds 0,1,2: trains on a uniformly random 50% subset of the admitted set (rng seed = seed*100+round). Size-matched control.
- own_clean, seeds 0,1: trains on the admitted proposer questions judged SOLVABLE by the full Opus audit (same judge prompt and
  protocol as the 200-item audits, extended to every admitted proposer question) plus the admitted intact tracers. Admitted deleted
  tracers and UNSOLVABLE/AMBIGUOUS proposals are excluded.
- own_dirty, seeds 0,1, round 1 only: the complement (UNSOLVABLE/AMBIGUOUS proposals plus admitted deleted tracers).
- own, seeds 3,4: extra replication of the main arm.
- Qwen3-8B-Base own, seeds 1,2 (paired GPUs): replication of amendment A3a.

Construction. Round 1 of own_sub/own_clean/own_dirty reuses the proposals and the guard output of the own lineage with the same seed
(the proposer is frozen and the round-1 guard solver is the base model, so these are identical by construction). Later rounds run
their own propose -> guard with the arm's own solver; the proposer seed is seed*100+round as in A3, so the proposal set is shared
across arms at every round and arms differ only through guard admission (solver-dependent) and subset selection.
Audits. Full audit of the admitted proposer questions of own s0/s1 round 1 (both models) now, stored as audit_full.json in the own
run directory and merged from the 200-item audit plus 20-25-item judge parts (a3_audit_merge.py). For own_clean rounds 2 and 3
the arm's own admitted set is audited before training (the run waits for audit_full.json). The 200-item audits continue for
every round of every arm.

Pre-specified reads (seed-level, paired by seed; greedy MATH-500 / held-out-1000 / GSM8K after round 3 unless a lineage collapsed;
mean of the three accuracies; Q1u and abstention are secondary):
- KC1  clean exceeds own by >= 2 points in both seeds of a model while |sub - own| < 2: the loop cost is attributed to the
       ill-posed admitted items (headline attribution).
- KC2  clean, own and sub within 2 points of each other: ill-posed items do not move accuracy at this scale; any cost claim
       then rests on Q1u, abstention and collapse only.
- KC3  sub differs from own by as much as clean does: data volume, not ill-posedness, drives the difference; no attribution claim.
- own_dirty round 1 is descriptive (one round on the ill-posed subset: accuracy and Q1u change vs. own round 1).
- Collapse (end-of-round eval accuracy below half the base accuracy, or truncation rate > 0.5) counts as a cost of that arm at
  that round; the lineage's reads stop at the previous round. Collapse in own but not in own_clean at the same seed is evidence
  for KC1 regardless of the accuracy margins.
Execution. Single-GPU rounds on any of the three hosts; a round after round 1 runs only on the host that holds the previous
round's final checkpoint (runners skip other hosts' lineages). 8B seeds 1,2 run on paired GPUs of host <HOST>, seed 0 on host 11067.


## Amendment A3d (2026-09-22 15:05 CST): answer extraction for the accuracy reads, decided before any control-arm result

Observation that triggered it: the Llama-3.2-3B own lineage, seed 2, round 1 trained without collapse (step 60: format rate
1.00, truncation 0.00, mean length 213 tokens) yet reads MATH-500 5.0% / held-out 4.0% under the registered boxed-only parser,
while GSM8K (which has a numeric fallback in the evaluator) reads 73.2% (base 72.0%). 78% of its greedy MATH-500 outputs contain
no \boxed{} (base model: 35%; the Direction B own arm trained on real MATH+GSM8K questions: 5%). The self-curriculum (GSM-style
word problems, reward with numeric fallback per A3b) trains the solver out of the \boxed{} convention on MATH-style problems, so
the strict parser conflates format drift with loss of ability.

Change (applies to every arm, round, seed and base model; nothing else in the evaluator changes; sampling statistics unchanged):
- b_eval.py additionally computes greedy_acc_lenient: \boxed{} if present, otherwise the '####' / 'answer is' / 'Answer:'
  patterns, otherwise the last number in the output; the candidate is graded by the same MATH grader (wrapped in \boxed{}) or
  the same numeric comparison for GSM8K. Abstentions stay wrong. It also records greedy_boxed_rate (share of greedy outputs with
  a \boxed{}) and stores the greedy output text per item.
- Primary accuracy for the KC1-KC3 reads and for the round tables is greedy_acc_lenient; the registered strict accuracy is
  reported next to it; greedy_boxed_rate is reported as a separate format-drift cost. If strict and lenient give different KC
  verdicts, both are reported and the lenient one is the headline for accuracy.
- Already-evaluated checkpoints (base models, the round-1 own rounds) are re-run through the new evaluator (greedy pass only;
  --redo 1) so every number in the tables comes from the same code. Collapsed solvers whose outputs are all truncated are
  re-run too when a card is free but are read as collapsed regardless.
Registered files: dirB/code/b_eval.py (v4), dirA3/code/a3_kc.py, dirA3/code/a3_analyze.py (lenient columns).


## Amendment A3e (2026-09-22 19:20 CST, registered before any run of these arms): a tracer-calibrated guard, an AZR-style guard, rounds 4-6

Motivation (from the 19:00 outcome panel): the drift finding is measured but (i) not yet shown to be guard-general, (ii) not
yet converted into a fix, and (iii) the harm horizon is three rounds. Three additions, same proposer, trainer and evaluations
as A3; proposer seed = seed*100+round as before, so proposals are shared with the own lineages at every round.

1. Arm own_tg (tracer-calibrated guard). The guard already plants 200 premise-deleted GSM-HP twins and their 200 intact twins
   (from outside the probe set) into every round's candidate pool. own_tg uses them to CALIBRATE the band each round: over the
   grid of p_mode thresholds k/8, choose [lo, hi] that maximises (intact-tracer admission - deleted-tracer admission) subject to
   intact-tracer admission >= 0.30 and hi <= 7/8 (unanimous items are excluded as in the learnability logic: a unanimous
   group gives the own reward zero variance and no gradient) (ties: larger intact admission), then admit proposals with p_mode
   in that band. No answer
   labels are used; the only external input is a pool of intact GSM-style questions with automatically deleted premises.
   Lineages: Qwen3-4B seeds 0,1; Llama-3.2-3B seeds 0,1; 3 rounds. (`a3_guard.py --guard tracer`)
2. Arm own_azr (learnability guard as in Absolute Zero: admit anything the solver neither always nor never solves, i.e.
   p_mode in (0, 1)). Lineages: Qwen3-4B seeds 0,1; 3 rounds. (`a3_guard.py --guard azr`)
3. Rounds 4-6 for the own lineages Qwen3-4B seeds 0,1,2 and Qwen3-8B seeds 1,2 (same protocol), to extend the drift and
   accuracy curves beyond three rounds.

Pre-specified reads:
- KT1 (drift arrested): in own_tg the deleted/intact tracer admission ratio is <= 0.8 at every round and the audited unsolvable
  share of admitted proposals at rounds 2-3 does not exceed the own arm's round-1 share for the same seed (own rises by 15-25
  points over rounds). KT1 fails if the ratio exceeds 1.0 at any round or the share rises by >= 10 points from round 1 to 3.
- KT2 (no cost): own_tg round-3 lenient mean accuracy >= own round-3 - 2 (Qwen); for Llama seeds 0/1 (own collapsed at round
  1) own_tg not collapsing at round 1 is itself the read. KT1 and KT2 together turn the negative result into a method.
- KA (guard generality): own_azr shows the same drift (tracer ratio > 1.0 at round 2 or an audited unsolvable-share rise
  >= 10 points from round 1 to 2); if it does not, drift is specific to the fixed band and is reported as such.
- Rounds 4-6: per-round audited unsolvable share, tracer ratio, lenient accuracy, Q1u and abstention; harm = a round where
  the lenient mean accuracy falls >= 2 points below the previous round or Q1u exceeds base + 0.10.
Execution: queue lines added at the top of queue_a3_single.txt (own_tg, own_azr round 1) and at the end (rounds 4-6); 8B
rounds 4-6 in queue_a3_8b.txt / queue_a3.txt; 200-item audits every round as before.
