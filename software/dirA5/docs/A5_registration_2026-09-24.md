# A5 registration: post-08:30 program (2026-09-24 08:04 CST, before any of its data or statistics)

Designed by an Opus design workflow (5 designers, 3 judges, 1 synthesizer; run wf_673448c7-e05, 07:30-08:00 CST) from the A4
state at 07:30. Seen at design time (disclosed): all A3/A4 reads listed in the A4 verdict draft, A4g seeds 0-1, A4i HL1 (3/3),
A4l collapse (seed 0), UMWP HX1-HX5 snapshots. Seen AFTER design and before this hash: A4j round-5 HM1/HM2 (fail), A4k HK1
(fails), A4i seed-2 HL2 (fails). The tri-gate variant T is post-hoc (designed after A4i/A4j/A4l data) and is labelled so in
every read; own_uv w=1 remains the registered primary reward variant. Compute: hosts <HOST> and <HOST> after 08:30 (duration
unknown); every line is resumable; lines not finished when compute ends are reported as not run with their stop time.
Order of execution (budget permitting): TG -> HZN -> J1F1 -> CTRL-L -> SWAP -> S8 -> T5/T-L (gated) -> MiP (conditional).

## 1. J1F1: Judge anchor on human labels and selective-answering frontier (eval-only, mostly offline) (2 GPUs, ~2 h)

This is mostly CPU and API work on dumps that already exist. Checked 07:57: dirA4/results/umwp/{plain_,abs_}*.jsonl hold 8 samples per item for base, own r3/r5, guard r3, clean r3, the rescue arms, 8B own r3 and Phi own/guard r3. dirB/results/hp/{hp_,hpabs_}* hold 32 samples per item for every lineage and round.
(a) Judge validity: API only. The frozen Opus audit prompt runs on 400 UMWP items: 200 unanswerable and 200 answerable, stratified by source x category, RNG 20260924.
(b) Judge-free drift on human labels: plain-prompt UMWP passes at r1 and r2, about 20 checkpoints at 5-8 min each, same decoding as the guard (G=8, T=1.0). Checkpoints: Qwen3-4B own seeds 0-4, Qwen3-8B own 0-2, Phi own 0-1. In-band [0.25,0.75] rates are computed for unanswerable (U) and answerable (A) items at base and r1-r3. On the 412 twin pairs, the same rates are computed from the 32-sample hp dumps by exact hypergeometric subsampling to G=8.
(c) Frontier: from the abstention-prompt dumps, CA_U vs FA_A curves for base, own r3, guard r3, clean r3 and loop_uv r4. Label-free rules: single sample; abstain if modal share < tau, tau in {0.25,...,1}; abstain if >= k/8 samples abstain, k=1..8. Plus an oracle tau tuned on a 50% UMWP dev split and scored on the other half. New GPU passes are needed only for checkpoints with no dump: loop_uv r4 s1/s2, loop_uv r5, 8B guard r3.
The already-observed default operating points are declared as inputs and are not confirmatory.

Predictions:
- J-P1 judge validity: Opus labels >= 85% of UMWP-unanswerable as ill-posed (U or A) and <= 12% of UMWP-answerable as ill-posed; binary Cohen kappa vs the UMWP human labels >= 0.70.
- J-P2 judge-free drift: under the loop's plain prompt and guard decoding, the UMWP in-band rate ratio U/A at r3 is >= 1.25x its base value in >= 80% of the powered own lineages (Qwen3-4B seeds 0-4, Qwen3-8B 0-2, Phi 0-1). A lineage is underpowered if < 30 answerable items are in band at base or at r3. r1 and r2 are descriptive.
- J-P3 (secondary; the twins are judge-validated): the same criterion on the 412 deleted/intact twin pairs holds in >= 8 of 10 own lineages.
- F-P1 dominance: at the guard's own FA_A (linear interpolation), guard r3 CA_U <= the best label-free rule on the paired own r3 - 0.05 on UMWP in >= 2 of 3 Qwen guard seeds (0, 1, 2), and on twins in >= 2 of 3. Phi guard seeds 0-1 are exploratory, since their UMWP default point already equals own's.
- F-P2 over-abstention: guard r3 FA_A <= paired own r3 FA_A + 0.05 on intact twins and on UMWP-answerable in >= 4 of 5 guard lineages; no lineage above 0.25.
- F-P3 label-tuned bound: guard r3 at its default point is within 0.05 CA_U of oracle-tuned own r3 (dev/test split) at matched FA_A in >= 2 of 3 Qwen seeds.
- F-P4 mechanism: the self-consistency AUROC (modal share, U vs A on UMWP) falls from base to own r3 by >= 0.05 in >= 8 of 10 own lineages.

## 2. CTRL-L: Guard vs size-matched and difficulty-matched loops (Qwen3-4B seeds 0, 1, 2; r1-r3) (6 GPUs, ~5 h)

Two new arms, each reusing the raw proposal file the guard lineage used for that seed and round, so all arms are paired.
- RND: the plain band exactly as in own, then a random subsample down to the guard lineage's admitted count for that seed-round. The count is read from dirB/runs/qwen3_4b_Base_own_ag_a4_s<seed>_r<t>/admitted.jsonl (for example 687 at s0 r1 and 367 at s0 r3, vs own 999/593). RNG 20260924+100*seed+round.
- LOWP: the plain band, then drop the lowest-modal-share in-band items until the count matches (ties random).
- Training is identical to own and guard: arm own, plain prompt, 60 GRPO steps, same trainer seed, single GPU.
- Evals every round: twin Q1u hp/hpabs, intact abstention, lenient accuracy, tracer ratio, 200-item Opus audit; UMWP at r3.
- Code: a CPU post-guard filter (a4_postfilter.py, rules rnd/lowp/trigate) hooked into a4_round.sh, unit-tested on the existing guard_records.jsonl (hand check on 20 items).

Predictions:
- C-P1 (primary): at r3, twin Q1u (plain) guard <= RND - 0.05 in >= 2 of 3 seeds; pooled item-paired bootstrap over the 3 seeds, one-sided p < 0.01.
- C-P2: at r3, UMWP CA guard <= RND - 0.05 in >= 2 of 3 seeds.
- C-P3: at r3, twin Q1u guard <= LOWP - 0.05 in >= 2 of 3 seeds. Validity: if at r1 (same base solver) the guard-rejected and LOWP-rejected sets overlap with Jaccard > 0.6, C-P3 is declared uninformative.
- C-P4 (size is not the cause): |Q1u(RND r3) - Q1u(own r3)| <= 0.05 in >= 2 of 3 seeds (own r3 0.347/0.316/0.410).
- C-P5 cost: RND and LOWP lenient accuracy within 2 points of own r3; the audited ill-posed share per arm and round is descriptive.

## 3. TG: Reward isolation and tri-gate at round 4 (Qwen3-4B seeds 0, 1, 2; same start, proposals and samples as A4i) (6 GPUs, ~1.3 h)

Every arm starts from the rescue_uv r3 solver of its seed, exactly like A4i loop_uv r4. Items and samples come from the A4i guard_records.jsonl: abstention prompt, count-unknown, per-item abstain share and numeric modal share.
- ABS4 (prompt-, start- and item-matched control): arm own_abs ('unknown' excluded from the vote, zero reward) on the loop_uv r4 admitted.jsonl, unchanged.
- T4: arm own_uv on the tri-gate subset of the same records:
  - abstain <= 1/8 and numeric mode share in [0.25, 0.75]: kept, numeric target;
  - abstain in {5/8, 6/8}: kept, 'unknown' majority;
  - 2-4 abstentions: dropped;
  - a 50% cap on unknown-majority items, registered as non-binding.
  T4 is a subset of the loop_uv admitted set, so no trainer code changes.
- Disclosed in the registration: T was designed after the A4i r4, A4j r5 and A4l data. own_uv w=1 stays the registered primary reward variant.
- Offline partition counts, and the unknown-majority rate on audited-solvable items, are reported as descriptive.
- Evals: twins hp/hpabs, intact abstention, lenient accuracy, UMWP; about 40 min training + 25 min evals per job.

Predictions:
- T-P1 (reward isolation): plain-prompt Q1u of loop_uv r4 <= ABS4 - 0.05 in >= 2 of 3 seeds. Registered alternative: |diff| <= 0.02 in >= 2 of 3 means 'the unknown reward adds nothing beyond the abstention prompt and start at round 4', reported as such. Any other outcome is inconclusive.
- T-P2: T4 Q1u <= ABS4 - 0.05 AND T4 Q1u <= loop_uv r4 Q1u in >= 2 of 3 seeds.
- T-P3: T4 lenient accuracy >= loop_uv r4 - 2 in 3 of 3 seeds; intact-twin abstention (abstention prompt) <= 0.15 in 3 of 3.
- T-P4: T4 UMWP CA <= ABS4 - 0.05 in >= 2 of 3 seeds.
- Kill and gate (registered): intact abstention > 0.30 or an accuracy drop > 5 in any seed stops T. T5 runs only if T-P2 and T-P3 hold.

## 4. HZN: Guard horizon to round 5 (Qwen3-4B seeds 0, 1, 2) (3 GPUs, ~3.5 h)

Continue each guard lineage from guard r3 (s0 on <HOST>, s1 on <HOST>, s2 copied from <HOST> before 08:28) through r4 and r5. It uses the own lineage's raw r4 and r5 proposal files (paired with own r4/r5, which exist), guard admission (band + reject if >= 2/8 abstain under the abstention prompt), and arm own training (plain prompt, 60 steps). Evals as in own r4/r5: twins, intact abstention, lenient accuracy, UMWP at r5, 200-item Opus audit, tracer ratio.

Predictions:
- H-P1: r5 twin Q1u guard <= own r5 - 0.10 in >= 2 of 3 seeds (own r5 0.478/0.345/0.571).
- H-P2 (no re-accumulation): guard r5 Q1u <= guard r3 + 0.05 in >= 2 of 3 seeds (guard r3 0.257/0.216/0.192).
- H-P3: lenient accuracy >= own r5 - 2.5 in 3 of 3 seeds (own r5 80.5/81.0/81.8).
- H-P4: r5 UMWP CA guard <= own r5 - 0.05 in >= 2 of 3 seeds.
- H-P5 (the registered expectation, consistent with the no-go): the audited unsolvable share of the guard's admitted set at r5 >= its r3 value - 5 points in >= 2 of 3 seeds (the drift continues and only the harm is removed).

## 5. SWAP: Self-suspicion vs composition, one-round mechanism test at r3 (Qwen3-4B seeds 0-4) (10 GPUs, ~2.3 h)

Start from own r2 of each seed; all own r2 checkpoints are on <HOST> or 10905. The data are the own r3 admitted set of that seed.
- Flagging pass: own r2, abstention prompt, G=8; flag if >= 2/8 abstain, the same rule as the guard. About 5 min per seed.
- Full Opus audit of each own r3 admitted set: about 390 unaudited items per seed beyond the existing 200.
- Arms:
  - G1: drop the flagged items.
  - C1: drop the same number of unflagged items with the same audited S/A/U mix as the flagged set (stratified random, RNG 20260924).
  - F: own r3, reused (same start, data and trainer config).
- Training: arm own, plain prompt, 60 steps. Tracers are handled exactly as in F.
- Underpowered rule: a seed with fewer than 10% of items flagged is excluded, and at least 3 seeds must remain.

Predictions:
- S-P1: G1 twin Q1u <= F - 0.04 in >= 3 of 5 seeds.
- S-P2 (H4, self-suspicion): G1 Q1u <= C1 - 0.03 in >= 3 of 5 seeds, pooled item-paired bootstrap one-sided p < 0.05. Registered alternative: |G1 - C1| <= 0.02 in >= 3 of 5 means the composition account holds. Any other outcome is inconclusive.
- S-P3 cost: G1 and C1 lenient accuracy within 2.5 points of F in >= 4 of 5 seeds.
- Constructibility: if a class has too few unflagged items, the closest mix is used and the gap is reported. C1 depends on Opus labels, and this is disclosed together with the J-P1 validity numbers.

## 6. S8: 8B abstention guard, completing the registered seeds and adding seed 0 (2 GPUs, ~7.5 h)

The 8B guard is already registered (A4 Addition 1, dirA3/queue_a4_8b.txt, seeds 1-2). s2 r3 exists and is being probed; s1 r3 is queued on a <HOST> GPU pair. Finish it under the registered PF thresholds and add UMWP probes for s1/s2 r3. Addendum, hashed before its data: seed 0 r1-r3 from Qwen3-8B-Base, paired with the 8B own s0 raw proposals (own s0 r1-r3 are on <HOST>), one GPU pair, about 2.5 h per round.

Predictions:
- Seeds 1-2: PF1-PF4 as registered in A4 Addition 1, unchanged.
- G8-P1 (new, all 3 seeds): r3 UMWP CA guard <= own r3 - 0.04 in >= 2 of 3 seeds (8B own r3 0.242/0.203/0.362).
- G8-P2: r3 twin Q1u guard <= own r3 in >= 2 of 3 seeds (8B own Q1u is only 0.07-0.13, so the room is small).
- G8-P3: lenient accuracy >= own r3 - 2 in >= 2 of 3 seeds; the tracer ratio at r3 >= 1.1 in >= 2 of 3 (drift persists).

## 7. T5 then T-L (conditional): tri-gate at round 5, then from round 1 with a prompt-matched loop control (6 GPUs, ~6 h)

T5 runs only if the TG gate passes. It uses the count-unknown guard pass with the T4 solver on the own r5 raw proposals (abstention prompt), then the tri-gate filter, own_uv training and full evals, for 3 seeds.
T-L runs only if T5 passes. It runs tri-gate lineages from Qwen3-4B-Base for seeds 0-2, r1-r3, alongside ABS-L, a prompt-matched loop: abstention-prompt training, 'unknown' with zero reward, and the same count-unknown band admission. Both are paired on the own raw proposals. T is disclosed as a post-hoc variant in every registration.

Predictions:
- T5-P1 (no re-drift, the analogue of HM2): the invented-answer target share at r5 <= the T4 value + 5 points in >= 2 of 3 seeds.
- T5-P2: T r5 Q1u <= own r5 - 0.10 in >= 2 of 3 seeds; lenient accuracy >= own r5 - 2 and intact abstention <= 0.15 in 3 of 3.
- TL-P1: r3 Q1u T-L <= ABS-L - 0.05 and <= guard r3 of the same seed in >= 2 of 3 seeds.
- TL-P2: lenient accuracy >= guard r3 - 2; at every round, intact abstention <= 0.15 and unknown as the target on <= 5% of audited-solvable admitted items, in 3 of 3 seeds.

## 8. MiP filler: second external unanswerable-math probe (eval-only, conditional on staging) (2 GPUs, ~1 h)

MiP-GSM8K and MiP-SVAMP (missing-premise problems) plus their answerable sources, run through dirA3/code/umwp_probe.py (plain and abstention prompts, G=8). Checkpoints: base, own r3 seeds 0-4, clean r3 seeds 0-3, guard r3 seeds 0-2. The data are staged from the local PC. If not staged by 10:30, the probe is reported as not run.

Predictions:
- M-P1: CA own r3 - base >= +0.10 in >= 4 of 5 Qwen3-4B seeds.
- M-P2: clean r3 CA < own r3 CA in >= 3 of 4 seeds.
- M-P3: guard r3 CA <= own r3 - 0.05 in >= 2 of 3 seeds; answerable accuracy within 3 points of own.

## Dropped at design time (for the record)

- D1-L1 / D5-E2 (own_uv w=1 full loop from round 1, 29-54 GPU-h). The most expensive line, and the one most likely to be mixed:
- A4j round 5 shows the gain eroding (invented-answer targets 75.0/62.5 vs 49.0/56.0 at r4; HM2 fails 2/2).
- The rescue Q1u gain is matched by the prompt-matched zero-reward control rescue_abs (0.335/0.197/0.357 vs 0.347/0.233/0.367).
- 'unknown' becomes the target on 7-17% of solvable items.
Replaced by the 7 GPU-h TG gate.
- D1-L2 and D2-E3 (Phi-4-mini uv/AoA full loops). The go/no-go rule could not be hashed before the Phi rescue was read at about 08:05. The guard already fails on Phi externally (UMWP CA 0.787/0.763 vs own 0.787/0.785), and Phi's instruct-tuned abstention makes a refusal spiral likely.
- D1-S1 and D3-E2a as new experiments. The 8B guard is already registered (A4 Addition 1, seeds 1-2; s2 r3 done, s1 r3 queued). Converted into S8: finish the registered seeds and add a hashed seed-0 addendum.
- D1-C1 (one-round difficulty-matched reject). Folded into CTRL-L as the LOWP loop arm, which tests the claimed full-loop effect, with the Jaccard overlap validity rule measured at r1.
- D2-E1 (AoA full loop launched at 08:45). It was a post-hoc variant presented as derived from the law, with no cheap gate before 31 GPU-h. Replaced by the disclosed, gated chain TG -> T5 -> T-L, with the prompt-matched ABS-L control.
- D2-E4 (threshold phase boundary). Its 'predicted' boundary f* is the geometric mean of the two observed anchors, so it is a postdiction.
- D5-E1 and D5-E3 (mean-field critical weight w*). Forking paths: the kappa fallback was chosen against the only two anchors, boundary-zone runs were unscored, and no null interpolation predictor was registered, so a pass would show nothing beyond the anchors. The new model code would also have had to be validated by 08:55.
- D5-E4 (abstain-bonus and entropy-gated reward baselines). These are baselines for a reward variant that is no longer the headline. Revisit only if T passes T5.
- D5-E5 and the D3-E3 GPU lineages (no-guard, AZR and TTRL loops). Already covered: A3e own_azr seeds 0-1 exist and KA holds (unsolvable 36.5 -> 52.5 -> 55.5). Under GRPO a no-band TTRL loop gets gradient only from non-unanimous groups, which is the AZR admission set. A CPU re-banding may be reported as exploratory.
- D2-E5 (40 GPU-h AoA horizon to r5) and D2-E6 (8B AoA). Ungated. Replaced by HZN for the headline method and T5 for the variant.
- D3-E2b/c (8B and Phi uv round 4 started from own r3). Start-checkpoint mismatch with A4i, which started from rescue_uv r3. It also compared against own r4 admission read from abstention-prompt samples and against own r3, one round fewer.
- D3-E4 (SQuAD passage-QA self-play domain). A new pipeline, normalizer and tracers in one day, with a high risk of a null or surface-form artifact.
- D3-E5 (1.7B size ladder). Adds little. The 14B r2 checkpoint is only copied and probed if time allows.
- D3-E1 full battery (TreeCut generation, AbstentionBench non-math). The server's internet access is unverified and the runtime was about 2x underestimated. Kept only as the conditional MiP filler.
- D4-E2 (round-5 continuation of own/own_uv). Already running as the registered A4j; finished, not re-registered.
- D4-E3 (8B one-round swap). Superseded by the registered 8B guard loop.
- D4-E4 and D4-E6 as written. Kept as TG and T-L, with two fixes: the primary comparator is the prompt-, start- and item-matched ABS4 (own_abs), not own r4; and T-L gets its own prompt-matched loop control, ABS-L.
- Pre-08:30 GPU passes on <HOST> (D1, D4, D5). All 8 of its GPUs are finishing the registered A4k Phi evals until the 08:28 release.
- Copying 100-200 GB of checkpoints to NFS (D1-D5). Unnecessary: copy host to host into the kept hosts' /dev/shm (345 and 447 GB free) with /root/copy_ck.sh on 11067. NFS stays at results-only size.
- The 8-GPU plans in D1, D2, D3 and D5, and judge 3's objection to D4's 16 GPUs. The fleet has three hosts. [internal note removed]

## Addendum A5b (2026-09-24 08:33 CST, before any trained-arm frontier is computed or viewed)

Motivation (disclosed): F-P1 and F-P3 failed. At its default point, the guard is dominated by the drifted own r3 model with a
test-time abstention prompt and a label-free rule. So the fair test of a training-time method is whether it moves the whole
selective-answering frontier under the abstention prompt, not only the default point. Using the a5_frontier.py definitions
(abstention-prompt dumps, registered rule set, strict Pareto frontier, linear interpolation), compare CA_U at FA_A = 0.05 on
UMWP-answerable, and at FA_A = 0.10 on intact twins, for arm vs own r3 of the same seed.
- FB1 (method, UMWP): rescue_uv r3 CA_U@0.05 <= own r3 CA_U@0.05 - 0.03 in >= 3 of 4 seeds (0, 1, 2, 4).
- FB2 (method in the loop, UMWP): loop_uv r4 CA_U@0.05 <= own r3 CA_U@0.05 - 0.03 in >= 2 of 3 seeds. Comparator is own r3, not r4, because no own r4 UMWP dump survives; disclosed.
- FB3 (twins): the same two comparisons at FA_A = 0.10 on the twins, same seed counts.
- Descriptive only, no prediction: guard r3, clean r3, rescue_abs r3, Phi rescue_uv r3, TG abs4/t4 r4 when available.
Script: dirA5/code/a5_frontier_b.py.
