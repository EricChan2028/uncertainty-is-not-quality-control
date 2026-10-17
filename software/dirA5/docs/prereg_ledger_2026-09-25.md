# Pre-registration ledger (A3-A10), 2026-09-25

Compiled 2026-09-25 ~12:10 CST from four adversarial verification passes run the same morning. All passes were read-only on host port <PORT>, repo /home/tione/notebook/rsi_rl. This file contains tables and notes only, no manuscript text. Paths are relative to the repo. "L" means a line number in the named registration file unless stated otherwise.

Conventions
- Outcome labels: HOLDS / FAILS / INCONCLUSIVE / NOT_RUN / PENDING / DESCRIPTIVE_ONLY.
- Decision and interpretation rules: HOLDS = triggered, FAILS = not triggered. They are always non-confirmatory. For multi-branch rules, the branch that applies is named.
- Split rows put the model or arm in brackets after the registered ID. Counts are per row: the A4 table has 58 confirmatory rows covering 51 registered IDs.
- "unofficial" = a verifier's read-only recomputation with the registered read code. It appears in no results document and changes no label.
- "(no ID)" = a label a verifier assigned to a registered clause that has no ID.
- A3 P2 / P5 / KT1 "ill-posed share" = the UNSOLVABLE-only share (JSON field illposed_share). U+A was checked alongside; no conclusion changes.

## 1. Summary of counts

| Registration (file) | Confirmatory rows | HOLDS | FAILS | INCONCLUSIVE | NOT_RUN | PENDING | Non-confirmatory rows |
|---|---|---|---|---|---|---|---|
| A3 (dirA3/docs/A3_registration_2026-09-22.md + A3a-A3e) | 14 | 5 | 8 | 0 | 1 | 0 | 14 (6 rules, 2 of them triggered; 5 registered descriptive; 3 unregistered) |
| A4 (dirA4/docs/A4_registration_2026-09-23.md + A4a-A4l, addenda) | 58 | 24 | 27 | 1 | 2 | 4 | 18 (10 rules, 2 of them triggered; 7 descriptive; 1 gated arm NOT_RUN) |
| A5 (dirA5/docs/A5_registration_2026-09-24.md + A5b) | 37 | 12 | 11 | 2 | 12 | 0 | 8 (+1 uncounted cross-reference row) |
| A6 (dirA5/docs/A6_registration_2026-09-24.md) | 10 | 5 | 5 | 0 | 0 | 0 | 2 (both NOT_RUN: X2 8B clause, E3 pointer) |
| A7 items 1-4 (dirA5/docs/A7_registration_2026-09-24.md) | 2 | 1 | 1 | 0 | 0 | 0 | 5 (3 registered descriptive, 2 read-script sensitivities) |
| A7 item 5 (A7_5_addendum + A7-5a/5b/5c) | 2 | 0 | 0 | 0 | 0 | 2 | 9 (8 PENDING, 1 unregistered NOT_RUN) [26 Sep: see section 15] |
| A7b (dirA5/docs/A7b_addendum_2026-09-24.md) | 3 | 3 | 0 | 0 | 0 | 0 | 1 |
| A8 (dirA5/docs/A8_registration_2026-09-24.md + A8a) | 3 | 2 | 1 | 0 | 0 | 0 | 8 (1 derived, 1 unregistered) |
| A9 (dirA5/docs/A9_registration_2026-09-25.md) | 2 | 2 | 0 | 0 | 0 | 0 | 3 (1 read-script only, 1 unregistered) |
| A10 (dirA5/docs/A10_registration_2026-09-25.md) | 2 | 1 | 1 | 0 | 0 | 0 | 2 |
| **All** | **133** | **55** | **54** | **3** | **15** | **6** | **70** |

(Superseded counts: section 14 (25 Sep 12:45) and section 15 (26 Sep, R-Zero).)

Sensitivity of the confirmatory tally. The labels above are the ledger's. The rows below are alternative readings a reviewer could apply.

| Alternative reading | Effect | All: HOLDS / FAILS / INCONCL. / NOT_RUN / PENDING |
|---|---|---|
| As ledger | none | 55 / 54 / 3 / 15 / 6 |
| A4i L281 read literally. The loop_uv r4 evals finished 07:53-08:26, after the 07:50 cutoff, so HL2-HL4 become NOT_RUN | A4 HOLDS -2 (HL3, HL4), FAILS -1 (HL2), NOT_RUN +3 | 53 / 53 / 3 / 18 / 6 |
| A4g L232 read literally. rescue_uv s2 evals finished after 07:45, so s2 is excluded | HU1 2/3, HU2 3/3, HU3 1/3; no label changes | unchanged |
| PF3/PF4 [8B] labelled NOT_RUN (the A5 pass's label for the S8 re-reference) instead of PENDING | PENDING -2, NOT_RUN +2 | 55 / 54 / 3 / 17 / 4 |
| Unofficial reads adopted for the four A4 PENDING rows: PF3 [8B] 2/2 HOLDS; PF4 [8B] 1/2, counted FAILS; NF2 74.1% FAILS; HX2b [Phi] 2/2 HOLDS | HOLDS +2, FAILS +2, PENDING -4 | 57 / 56 / 3 / 15 / 2 |
| A3 P5 left unread (the A3 verdict never reads it) | FAILS -1, PENDING +1 | 55 / 53 / 3 / 15 / 7 |

## 2. A3: self-curriculum loop with a frozen proposer

Registration dirA3/docs/A3_registration_2026-09-22.md. The base was hashed 2026-09-22 09:25 CST; amendments A3a-A3e follow. The current sha256 308504ef... equals the A3e hash of 2026-09-22 19:12:40. Verdict: dirA3/docs/A3_verdict_2026-09-23.md (18:45 CST, 96 rounds). The readout file is dirA3/results/a3_analyze_readout_2026-09-23_1843.txt, cited below as "a3 readout".

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| P1 [Qwen3-4B; Qwen3-8B via A3a] | base L33-36; 8B added by A3a L63-65 | Tracer admission, R-Zero band, own arm, r1-r3. (b) The deleted/intact admission ratio rises. (c) The confident share of deleted tracers (p_mode >= 0.5) rises monotonically. Clause (a) is Llama-only. | yes | HOLDS | (b) Net r1->r3 rise in 8/8 lineages (4B 0.76-0.89 -> 1.30-1.63; 8B 0.90-1.01 -> 1.81-2.10). r2->r3 falls in 5/8, so (b) holds on the net reading only. (c) Monotone in 8/8 (r3: 4B 76.0-81.0%; 8B 78.0-82.0%). Beyond the horizon: 4B s2 dips at r6 and 8B s1 at r4. r6 confident share 87.0-94.0%; r4-r6 ratio 1.32-2.22 | Verdict L20-35, 147-155; dirB/runs/{lineage}_r{n}/guard_stats.json; a3 readout L93-149, 231-265 |
| P1 [Llama-3.2-3B] | base L33-36 | (a) Deleted-tracer admission >= 0.8x intact in every round; (b) ratio rises; (c) confident share rises monotonically | yes | FAILS | (a) holds (0.87-1.34 in every round). (b) Readable seeds s2-s4: r1->r2 rises 3/3, r2->r3 falls 3/3, r3 < r1 in 2/3 (s2 0.98 vs 1.05; s3 0.99 vs 1.14). (c) Monotone only on s3 (32.5/52.0/60.0); s2 dips (61.0 -> 54.5), s4 is flat (60.0/60.0). Registered seeds 0/1 collapsed at r1, so the read rests on s2 (added in the hash log 2026-09-22 11:44, not in the registration text) and s3/s4 (A3c). The verdict read HOLDS from the r1->r2 rise alone | Verdict L32-35, 147-155; guard_stats.json llama3p2_3b_own_a3_s{0..4}; docs/registration_hashes.txt L107 |
| P2 [Qwen3-4B; Qwen3-8B] | base L37-38 | Ill-posed (UNSOLVABLE-only) share of admitted proposals, 200-item Opus audit: >= 15% at r1 and non-decreasing r1-r3 | yes | HOLDS | r1 35.0-46.0% in 8/8; non-decreasing in 8/8. r1->r3 rise +18.0 to +27.5 pts (verdict: +13 to +27). U+A gives the same conclusion (e.g. 8B s1 58.0/69.5/69.5). Outside the horizon, r4-r6 decrease (4B s2 64.0 -> 52.0 at r4) | Verdict L41-65, 156-160; audit_admitted.json illposed_share = counts.UNSOLVABLE/n |
| P2 [Llama-3.2-3B] | base L37-38 | Same as above, Llama | yes | FAILS | r1 48.0-51.0%. s2-s4 rise r1->r2 (50.5->62.5, 48.5->58.5, 48.0->60.0), then all fall r2->r3 (54.0, 57.5, 47.5). s3's -1.0 is within its SE (~3.5), so the fall is 3/3 strictly and 2/3 beyond SE. U+A shows the same pattern. Seeds 0/1 have no r2 | Verdict L53-56, 156-160; audit_admitted.json s{2,3,4} |
| P3 | base L39-42 | Cost conjunction by r3, per model: (i) Q1u on the 412 unsolvable twins at r3 >= r1 + 0.05; (ii) held-out MATH change r2->r3 < +1 pt; (iii) no-op accuracy falls >= 3 pts from base. The registration names no prompt; the verdict uses plain for Q1u and abstention for no-op | yes | FAILS | Qwen-4B: (i) 5/5 (+0.095 to +0.184); (ii) 4/5; (iii) fails (63.2 -> 67.2-70.4). Qwen-8B: (i) fails (+0.019 raw / +0.032 / +0.037); (ii) 2/3; (iii) fails (65.8 -> 72.4-74.2). Llama: (i) s2 +0.010, s3 +0.073, s4 +0.075; (ii) s3 +21.7 fails, s4 -7.9, s2 unreadable; (iii) holds (8.2 -> 0.0-0.2). Per seed, Llama s4 meets all three clauses; the model-level FAIL rests on a seed aggregation the registration does not define. Cells the verdict shows as "pending" had files before the verdict (4B s3/s4 Q1u 0.483/0.393, no-op 69.0/70.4; 8B s0 no-op 73.6) | Verdict L123-136, 161-171; dirB/results/hp/hp_{lineage}_r{n}_shard0of1.jsonl; evals/{lineage}_r{n}_step60(_abs).summary.json |
| P4 | base L43-44 (gated by L25-27) | pdc arm: held-out MATH within 2 pts of own at every round; Q1u within +0.05 of base; abstention <= 10% on intact / no-op. Runs only if A2 P2 and P3 pass | yes | NOT_RUN | Gate failed. A2 P2 was not read as predicted (own >= base + 0.10 fails under the abstention prompt). A2 P3 FAILS (pdc intact abstention 25.5/27.1% vs 10%). There are 0 pdc_a3 lineages | Verdict L172; dirB/docs/A2_verdict_2026-09-23.md L47-57, 65-66 |
| P5 | A3a L67-69 (hash 2026-09-22 10:03:16) | 8B own (registered s0; s1-s2 added by A3c): admitted ill-posed share AND tracer ratio within 0.10 of 4B in every round; 8B Q1u < 0.10 in all rounds | yes | FAILS (a ledger read: the verdict never reads P5; needs author confirmation) | s0 vs 4B s0: ratio +0.19/+0.23/+0.21, failing every round; share -1.0/-6.5/-5.0 (holds); Q1u 0.063/0.080/0.0825 (holds). The ratio fails under every comparison tried (vs the 4B s0-s1 mean +0.22/+0.13/+0.19; 8B mean vs 4B mean +0.13/+0.35/+0.42). Replication seeds: ratio s1 +0.13/+0.57/+0.19, s2 +0.11/+0.41/+0.59; s2 Q1u 0.114/0.129 at r2/r3 | No verdict line; guard_stats.json, audit_admitted.json, hp_qwen3_8b_Base_own_a3_*; a3 readout L93-149, 231-265 |
| A3a-bound (no ID) | A3a L69-71 | Rule. If P5 holds, claim "published guards do not improve with solver scale". If the 8B tracer ratio < 0.5x 4B, bound the claim to <= 4B | no | FAILS (neither branch triggered) | 8B/4B ratio (s0) 1.23x / 1.17x / 1.13x, and P5 fails. The A3a route therefore does not license the scale sentence | No verdict line; same files as P5 |
| KA3-1 | base L47-49 | Rule. If P3 fails on both models with no cost in any measured number by r3: "harmless"; stop the loop | no | FAILS (not triggered) | Q1u at least triples on Qwen-4B (0.104 -> 0.316-0.483 at r3, s0-s4; the verdict's 0.35-0.41 is wrong, since s1 = 0.3155). Llama loses 0.8-18.1 pts in non-collapsed rounds, and s0/s1 collapse at r1 | Verdict L176-178 |
| KA3-2 | base L50-51 | Rule. If P3 holds on >= 1 model, run cycle 2 with a 7-8B lineage and a trained proposer | no | FAILS (not triggered) | P3 holds on no model at model level. Per seed, Llama s4 meets every P3 clause, and the aggregation is undefined. The 8B lineages ran anyway under A3a; the co-evolving proposer is in A4 | Verdict L179-180 |
| KA3-3 | base L52-53 | Rule. If P1/P2 hold but P3 fails, report the "latent-harm" shape with the drift quantified | no | HOLDS (triggered on Qwen-4B/8B) | Qwen: P1 and P2 hold 8/8, P3 fails. Not triggered on Llama as worded (P1 b/c and P2 fail there) | Verdict L181-184, 249-251 |
| KC1 | A3c L110-111 (hash 2026-09-22 12:29:33); lenient accuracy via A3d | At r3 (lenient mean of MATH-500 / held-out / GSM8K): own_clean >= own + 2 in both seeds of a model, with abs(own_sub - own) < 2 | yes | FAILS | Qwen-4B r3 clean - own: -0.10 / +0.27 (needs >= +2); a3_kc prints "KC1 False". Llama False: s0 own collapsed at r1; at s1 all arms collapsed, so the read is degenerate. Strict accuracy gives the same verdicts | Verdict L197-200; dirA3/results/a3_kc_readout_2026-09-23_1843.txt L19-21, 40-42; dirA3/code/a3_kc.py (sha 2ba543...) |
| KC2 | A3c L112-113 | own_clean, own and own_sub within 2 pts of each other at r3 | yes | HOLDS | Qwen-4B r3 spread 0.80 / 0.27 (clean 81.8/81.7, own 81.9/81.5, sub 81.1/81.7); "KC2 True". Llama False, and no KC claim is made there. Strict accuracy gives the same | Verdict L188-203; a3_kc_readout L3-21, 40-42 |
| KC3 | A3c L114; a3_kc.py L86 | own_sub differs from own as much as own_clean does. Script test: in any seed, abs(sub - own) >= 2 and >= abs(clean - own) | yes | FAILS | Qwen-4B r3 sub - own -0.80 / +0.23; clean - own -0.10 / +0.27; "KC3 False" on Qwen and Llama. The >= 2 threshold comes from the script, not the A3c sentence | Verdict L204-205; a3_kc_readout L21, 42 |
| A3c own_dirty r1 (no ID) | A3c L115 (descriptive) | own_dirty r1 (trained only on UNSOLVABLE/AMBIGUOUS plus admitted deleted tracers) vs own r1: accuracy and Q1u | no | DESCRIPTIVE_ONLY | Qwen-4B accuracy 80.0 vs 79.8 (s0), 76.2 vs 80.0 (s1). Q1u 0.177/0.243 vs 0.204/0.221: the verdict's "does not raise Q1u beyond own" is false at s1. Llama dirty 45.2/41.2, not collapsed, while own collapsed | Verdict L140-141, 206-207; a3_kc_readout L22-23, 44-45 |
| A3c collapse rule (no ID) | A3c L116-117 | Collapse = end-of-round accuracy < 0.5x base, or MATH-500 truncation > 0.5. It counts as a cost, and reads stop at the previous round | no | DESCRIPTIVE_ONLY | Llama (lenient base 49.9): 5 of 14 r1 trainings collapse (own s0, own s1, own_clean s1, own_sub s1, own_tg s0); own_dirty never collapses. Later collapses: own s2 r3, own_sub s0 r3, own_sub s2 r3. No Qwen lineage collapses (max truncation 0.0485). Under strict accuracy, Llama own s2/s3 would also collapse at r2 and own_tg s1 at r3 | Verdict L208-213; a3_kc_readout L26-39 |
| A3c collapse-KC1 evidence (no ID) | A3c L117-118 | Rule. Collapse in own but not in own_clean at the same seed is evidence for KC1 "regardless of the accuracy margins" | no | HOLDS (triggered, Llama s0 only) | a3_kc prints "own collapsed but own_clean reached round 3 at seed(s) [0]". The verdict reports the trigger but draws no KC1 claim (own_sub survives to r2, own_dirty survives r1, own_clean loses 7 pts at r3). Setting the evidence aside overrides "regardless of margins" and must be disclosed | Verdict L213-220; a3_kc_readout L43 |
| A3d strict vs lenient (no ID) | A3d L137-142 (hash 2026-09-22 14:45:31) | Lenient accuracy is primary, with strict reported alongside. If the KC verdicts differ, report both. Re-run already-evaluated checkpoints so every number comes from one code version | no | DESCRIPTIVE_ONLY | KC verdicts are identical under strict. Gaps: the verdict tables are lenient only (strict appears only in the readout). Five r1 cells never ran on the v4 evaluator: Qwen-4B own s2, 8B own s1/s2, Llama own s0/s1. Only the Llama pair has a logged reason. No registered read uses these cells | registration_hashes.txt L142-149, 186; evals/{lineage}_step60.summary.json |
| A3d boxed rate (no ID) | A3d L135-138 | Report greedy_boxed_rate as a separate format-drift cost | no | DESCRIPTIVE_ONLY | Llama falls from 65.6% at base to 21/20/49% at r1 (own s2/s3/s4), and ranges 0.0-56.8% across arms and rounds. Qwen-4B 91% -> 94.6-99.2%; 8B 94.6% -> 97.6-99.6%. The verdict has no table | Verdict L219-220; math500_train.greedy_boxed_rate |
| KT1 | A3e L166-168 (hash 2026-09-22 19:12:40) | own_tg arrests the drift. Success: ratio <= 0.8 every round, and the r2-r3 unsolvable share <= own r1 share (same seed). Failure: ratio > 1.0 in any round, or a share rise >= 10 pts r1->r3 | yes | FAILS | Qwen s0: ratio 0.59/0.74/1.25, share 29.5 -> 50.5 (+21). Qwen s1: ratio 0.46/0.70/1.16. Llama s1: ratio 0.68/0.85/1.00, share 31.0 -> 55.0 (+24). Llama s0 is unreadable (r1 0.83, collapsed). On Qwen, own_tg's r3 share is 14-31.5 pts below own: it delays the drift by about one round | Verdict L224-230; a3 readout L81-91, 215-229 |
| KT2 [Qwen3-4B] | A3e L169-170 | own_tg r3 lenient mean accuracy >= own r3 - 2 | yes | HOLDS | 2/2: 81.77/82.33 vs 81.93/81.47 | Verdict L231-233; a3_kc_readout L3, 7, 9, 13 |
| KT2 [Llama-3.2-3B] | A3e L169-170 | own_tg does not collapse at r1, seeds 0/1 | yes | FAILS | 1/2. s1 48.7 (trunc 0.06) passes; s0 collapses (20.9 < 24.95, trunc 0.501) | Verdict L231-233; a3_kc_readout L30, 35 |
| KT1+KT2 method clause (no ID) | A3e L170 | Rule. KT1 and KT2 together turn the negative result into a method | no | FAILS (not reached) | KT1 fails, and KT2 fails on Llama | Verdict L233 |
| KA | A3e L171-172 | own_azr (AZR band, p_mode in (0,1)) drifts: ratio > 1.0 at r2, or unsolvable share +10 pts r1->r2 | yes | HOLDS | r2 ratio 1.24/1.33. Share 36.5 -> 52.5 (+16.0) and 34.0 -> 49.5 (+15.5). r3 accuracy 81.6/80.9. The ratio clause is near-trivial: the band admits 97-98.5% of tracers, so the r1 ratio is already 1.02. The share clause carries the result | Verdict L234-237; a3 readout L151-165 |
| A3e r4-r6 harm (no ID) | A3e L173-174 | Bookkeeping for r4-r6 (4B own s0-s2, 8B own s1-s2). Harm = mean accuracy drops >= 2 vs the previous round, or Q1u > base + 0.10 | no | DESCRIPTIVE_ONLY | No accuracy drop >= 2 (max -1.83, 4B s0 r5). 4B Q1u exceeds 0.2044 at every r4-r6 round (s2 0.502/0.553/0.638; the verdict's 0.571 for s2 r5 is wrong). 8B s2 exceeds 0.1146 at r3-r6; s1 stays below (0.107/0.104/0.097). r6 tracer ratio 1.32-2.22 | Verdict L238-245; hp_{lineage}_r{4,5,6} |
| A3 s.2b audit validation (not registered) | verdict s.2b (run 2026-09-23 03:28) | Blind validation of the audit prompt on 200 tracer twins (100 intact, 100 premise-deleted), 8 Opus judges | no | DESCRIPTIVE_ONLY | Intact: 88 S / 4 U / 8 A. Deleted: 91 U / 8 S / 1 A. False ill-posed rate 4% (U only) or 12% (U+A); deletion detection 92% | Verdict L67-82; dirA3/results/audit_validation/result.json |
| A3 s.2c inter-judge (not registered) | verdict s.2c (03:35) | Two 200-item audits re-judged by 8 new judges, items in reversed order | no | DESCRIPTIVE_ONLY | 4B own s0 r3: 3-class agreement 81.5% (kappa 0.674), binary 91.5% (0.758). Llama own s3 r2: 78.0% (0.638), binary 90.0% (0.743) | Verdict L84-96; dirA3/results/audit_rejudge/{run}/result.json |
| A3 s.2d fixed-pool guard diagnostic (not registered) | verdict s.2d (05:35-06:12) | Fixed pool (1,158 audited r1 proposals + 400 tracers) passed through the R-Zero band with the base solver and the r1-r5 solvers of 4B own s0 | no | DESCRIPTIVE_ONLY | Admitted n 724 -> 345. Ill-posed share 52.9 -> 81.2%. SOLVABLE band occupancy 53% -> 10%; UNSOLVABLE 72% -> 60%. Tracer ratio 0.84 -> 1.48. The pool audit has 47 part files and no merged file | Verdict L98-121; dirA3/results/guard_diag/qwen_s0/result.json |

## 3. A4: guards, co-evolution, family and scale extensions, rescues

Registration dirA4/docs/A4_registration_2026-09-23.md: base, amendments A4a-A4l, the A4g and A4i addenda, and A4h addenda b/c/d. Verdict: dirA4/docs/A4_verdict_2026-09-24.md (revision 2, 2026-09-25 02:40). "a4_reads 0552" = dirA4/results/a4_reads_2026-09-24_0552.txt.

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| PF1 | base Add.1 L28-30 (read 19:08; hash log L414) | Abstention guard on the fixed s0 pool. Admitted ill-posed share <= 35% at every checkpoint base..r5; rise < 10 pts; keeps >= 50% of the plain-band SOLVABLE admits | yes | FAILS | Ill-posed 49.0/64.3/69.6/73.0/73.6/74.6 (+25.6). Retention holds under both readings (count ratio; item intersection 58.9-76.9%, unofficial). Tracer ratio 0.28-0.47 | Verdict L11; dirA3/results/guard_diag/qwen_s0_ag/result.json vs guard_diag/qwen_s0 |
| PF2 | base Add.1 L31-32 | own_ag loop: audited UNSOLVABLE share <= 30% in every round; rise < 10 pts r1->r3; tracer ratio <= 0.6 in every round | yes | FAILS | Fails in 7/7 lineages, all above 30% by r2. r3: 4B 34.5-39.5, 8B 40.0/36.5, Phi 44.5/45.0. The ratio clause holds on Qwen (0.19-0.43) and fails on Phi s0 (0.78 at r1). Own r3 is 64.0-66.5 | Verdict L12; a4_reads 0552; dirB/runs/qwen3_8b_Base_own_ag_a4_s{1,2}_r3 |
| PF3 [Qwen3-4B s0-s2] | base Add.1 L33; the seed rule is only in KF (L68) | own_ag r3 lenient mean accuracy >= own r3 - 2 | yes | HOLDS | 2/3: 79.4/81.9/80.1 vs bars 79.9/79.5/79.8; s0 fails. PF3 has no seed rule of its own, so HOLDS applies KF's ">= 2 of 3" | Verdict L13; a4_reads 0552 |
| PF3 [Qwen3-8B s1-s2] | base Add.1 L33-34; KF L68 needs both seeds; re-referenced in A5 S8 L99 | Same test on 8B (own r3 83.7/83.5) | yes | PENDING (unread; data complete) | The verdict says "not finished (not read)", which is inaccurate: the evals exist since 2026-09-24 06:51 (s2) and 17:56 (s1). Unofficial: 83.4/83.3 vs bars 81.7/81.5, so it would hold 2/2. The 05:52 "PF3 False" for 8B is a missing-value placeholder, not a verdict | Verdict L13, L68; evals/qwen3_8b_Base_own_ag_a4_s{1,2}_r3_step60.summary.json |
| PF3 [Phi s0-s1] (inside PG2) | base Add.3 L65 | Phi own_ag r3 accuracy >= Phi own r3 - 2 | yes | HOLDS | 2/2: 76.9/76.0 vs bars 75.8/75.8 (s1 margin 0.2) | Verdict L13; a4_reads 0552 |
| PF4 [Qwen3-4B] | base Add.1 L35-36 (secondary) | own_ag r3 plain-prompt Q1u <= own r3, and unsolvable-twin abstention (abstention prompt) >= own | yes | HOLDS | Q1u 0.257/0.216/0.192 vs 0.347/0.316/0.410 (3/3). Abstention 0.733/0.749/0.728 vs 0.714/0.693/0.699 (3/3). The abstention clause is missing from the verdict row; the s1 value is from the raw hpabs file | Verdict L14; dirA4/results/rescue_read.json; hpabs_qwen3_4b_Base_own_ag_a4_s1_r3 |
| PF4 [Phi s0-s1] | base Add.1 L35-36, via Add.3 | Same test on Phi | yes | HOLDS | Q1u 0.600/0.541 vs 0.641/0.587. Abstention 0.510/0.544 vs 0.478/0.508 (2/2) | Verdict L14; a4_reads 0552 |
| PF4 [Qwen3-8B s1-s2] | base Add.1 L35-36; re-referenced in A5 S8 L99 | Same test on 8B | yes | PENDING (unread; data complete) | The verdict says "8B not read". Unofficial: the Q1u clause holds 2/2 (0.039/0.114 vs 0.073/0.129). The abstention clause holds only 1/2 (0.510/0.620 vs 0.524/0.613; s1 fails). The full PF4 therefore holds 1/2 | Verdict L14; hp_ and hpabs_qwen3_8b_Base_own_ag_a4_s{1,2}_r3 |
| PC1 | base Add.2 L48-49; seeds 0-3 via A4a.3 | coev: the audited UNSOLVABLE share of RAW proposals rises >= 15 pts from r1 (frozen, 29%) to r3, and the admitted r3 UNSOLVABLE share >= own r3 | yes | FAILS | 0/4. Raw share at r2/r3 is 22.0-28.0% vs frozen r1 29.3%, so it falls. Admitted r3 55.0/46.0/52.0/40.5 vs own 64.5/66.5/64.0/61.0 | Verdict L20; mechanism note v2 L106; a4_reads 0552 (s0-s1); dirB/runs/qwen3_4b_Base_coev_a4_s{2,3}_r{2,3} |
| PC2 | base Add.2 L50-51; A4a.3 ">= 3 of 4" | coev r3 lenient accuracy <= own r3 - 2 in >= 3/4 seeds | yes | FAILS | 0/4, so the registered wording applies: "the harm remains latent even under co-evolution". 81.8/80.6/82.8/81.7 vs own 81.9/81.5/81.8/81.4. The s2/s3 values come from raw eval summaries | Verdict L20; evals/qwen3_4b_Base_coev_a4_s{2,3}_r3 |
| PC3 | base Add.2 L52-53 | coev_ag: raw UNSOLVABLE share rises < 10 pts r1->r3, and r3 accuracy >= coev + 2 in both seeds and >= own - 2 | yes | FAILS | 0/2 on accuracy: 80.5/81.4 vs coev + 2 = 83.8/82.6. The rise clause is met (r3 27.0/25.0 vs frozen 29.3). a4_reads also prints False because the coev_ag r1 raw audit is missing | Verdict L20; dirA3/code/a4_reads.py |
| PG1 | base Add.3 L62-64 | Phi own: tracer ratio rises r1->r2, and UNSOLVABLE share +10 pts r1->r3, without collapse | yes | HOLDS | 2/2. Ratio 1.00 -> 1.68 and 1.04 -> 1.67. Share 26.5 -> 53.0 and 33.0 -> 57.0. Accuracy 77.8/77.8 (base 76.3). a4_reads does not check the truncation part of the collapse rule | Verdict L21 |
| PG2 | base Add.3 L65 | PF2 and PF3 hold on Phi own_ag | yes | FAILS | PF2 fails 2/2 (r3 share 44.5/45.0; s0 ratio 0.78 at r1, 0.69 at r3, both > 0.6). PF3 holds 2/2 | Verdict L21 |
| PG1/PG2 [OLMo-2-1124-7B-Instruct] | base Add.3 L55-59 (conditional: "reported as not run" if not run) | Same arms and reads on OLMo, s0-1 | yes | NOT_RUN | No OLMo run, smoke test or log. The verdict does not report it as not run, although the registration requires that | dirB/runs listing |
| NG1 | A4a.1 L102 | Held-out pool qwen_s1: the ill-posed share admitted by each rule R1-R5 rises >= 15 pts base->r5 | yes | HOLDS | R1 +25.7, R2 +39.2, R3 +41.9, R4 +28.4, R5 +36.9 | Verdict L15; dirA4/results/ng_read_qwen_s1.json |
| NG2 | A4a.1 L103-104 | Rule R6 [.5, 1]: rise < 10 pts, and the gradient-bearing SOLVABLE count falls >= 50% base->r5 | yes | FAILS | R6 30.5 -> 42.5, i.e. +11.93 in the JSON (the verdict and hash log round to +12.0). The count clause is met (364 -> 90, -75%) | Verdict L16; ng_read_qwen_s1.json |
| NG3 | A4a.1 L105-106 | In the gradient-bearing region (p_mode <= 7/8), the U+A share among candidates with p_mode >= 1/8 rises monotonically base->r5 | yes | HOLDS | 52.7/64.4/72.1/74.4/77.6/79.2 | Verdict L15 |
| NS1 | A4a.2 L113-114 | Base vs post-trained pairs (Qwen3-14B-Base/Qwen3-14B; Qwen2.5-32B/-Instruct): the post-trained band admits >= 10 pts more ill-posed items, with a lower in-band SOLVABLE share and a higher mean p_mode on SOLVABLE | yes | HOLDS | 2/2. 14B: 64.1 -> 74.9 (+10.8); SOLVABLE admit rate 0.329 -> 0.139; mean p 0.817 -> 0.929. 32B: 62.2 -> 75.7 (+13.4); 0.365 -> 0.141; 0.816 -> 0.909. Both readings of the share clause are met | Verdict L17; guard_diag/scale_qwen3_14b, scale_qwen25_32b |
| NE1 | A4b L133 | The stronger frozen proposer (Qwen2.5-7B-Instruct) has a less ill-posed raw pool than the 4B-Base pool (44.8%) | yes | HOLDS | 35.0% (A 151 + U 269 of 1,199) | Verdict L18; guard_diag/front_q25_7b |
| NE2 | A4b L134-135 | Escape: with the stronger proposer, the band's ill-posed share rises < 15 pts base->r5, and in-band SOLVABLE at r5 >= 25% of base | yes | FAILS | 46.9 -> 73.7 (+26.8). SOLVABLE 70 vs 378 (18.5%). Both clauses fail | Verdict L18 |
| NE1/NE2 [Qwen3-14B proposer] | A4b L130-131 (conditional on the download completing) | Same reads with a Qwen3-14B proposer | yes | NOT_RUN | No 14B proposer directory under guard_diag, pool_audit or frontier. The verdict is silent | directory listings |
| NF1 | A4e L182-183 | With the r3 solver, the band admits an evolved set <= 45% ill-posed, with >= 150 SOLVABLE per 1,000 evolved candidates | yes | FAILS | 73.6% (U 145 + A 89 of 318); 76 SOLVABLE per 1,000. There is no result JSON; the read exists only in the hash log (reproduced unofficially) | Verdict L19; registration_hashes.txt L543 |
| NF2 | A4e L184 | With the r5 solver, the evolved admitted set is <= 50% ill-posed | yes | PENDING (unread; data exist) | Never read, and the verdict omits it. Unofficial 74.1% (U 119 + A 76 of 263), so it would fail. No decision depends on it, since NF1 already failed | guard_diag/evo_r3/r5/guard_records.jsonl; pool_audit/evo_r3/parts |
| own_evo loop arm | A4e L185-186 (gated) | Run own_evo in the loop if NF1 and NF2 hold | no | NOT_RUN (gate not met) | NF1 failed | Verdict L19 |
| NL1 | A4d L168-169 | 14B loop: the r2 guard admits deleted tracers relatively more than the r1 guard, and the UNSOLVABLE share rises +10 pts r1->r2 | yes | FAILS | The ratio clause is met (1.17 -> 2.25). UNSOLVABLE 37.5 -> 43.5 (+6.0) fails. Ill-posed 57.0 -> 72.0. r1 finished 02:09, before the 07:30 cutoff; the optional r2 training was not done | Verdict L22; hash log L594; dirB/runs/qwen3_14b_Base_own_a4_s0_r{1,2} |
| HC1 | A4c L154-155 | At r3, Q1u(own_clean) <= Q1u(own) - 0.05 in both new seeds (2, 3) and in >= 3/4 overall, with McNemar p < 0.05 in >= 3/4 | yes | HOLDS | 4/4: -0.107/-0.097/-0.153/-0.121; p 5.2e-9 / 1.0e-8 / 2.5e-13 / 5.0e-9 | Verdict L23; dirA4/results/q1u_attrib.json |
| HC2 | A4c L156-157 | abs(Q1u(sub) - Q1u(own)) < abs(Q1u(clean) - Q1u(own)) in >= 3/4 | yes | HOLDS | 3/4: sub 0.070/0.078/0.034/0.197 vs clean 0.107/0.097/0.153/0.121; s3 fails | Verdict L24 |
| HC3 | A4c L158 | own_clean r3 lenient accuracy within 2 pts of own | yes | HOLDS | 4/4: 81.8/81.7/82.7/82.1 vs 81.9/81.5/81.8/81.4 | Verdict L25 |
| HA1 | A4f L199 | rescue_abs r3 plain-prompt Q1u <= own r3 - 0.05 in every seed run | yes | FAILS | 2/3: 0.335/0.197/0.357 vs bars 0.297/0.266/0.360; s0 fails | Verdict L26; rescue_read.json |
| HA2 | A4f L200 | rescue_abs accuracy >= own r3 - 2 in every seed run | yes | HOLDS | 3/3: 82.3/81.9/81.8 vs 81.9/81.5/81.8 | Verdict L26 |
| HA3 | A4f L201-202 | Abstention prompt: unsolvable-twin abstention >= own + 10 pts, and intact <= 15% | yes | FAILS | 0/3, in the opposite direction: 0.321/0.242/0.304 vs own 0.714/0.693/0.699. Intact 0.006-0.012. own_abs gives "unknown" zero reward (the A4g correction) | Verdict L26 |
| HU1 | A4g L220; s4 added by the addendum L234-235; cutoff L232 | rescue_uv r3 plain Q1u <= own r3 - 0.05 in every seed run (0, 1, 2, 4) | yes | FAILS | 2/4 as read: 0.347/0.233/0.367/0.299 vs own 0.347/0.316/0.410/0.393; s0 and s2 fail. Under the 07:45 cutoff s2 counts as not run (its evals finished 07:46:47 / 07:57:08), giving 2/3; the outcome is unchanged | Verdict L27; dirA4/results/uv_read.json |
| HU2 | A4g L221 | rescue_uv accuracy >= own r3 - 2 in every seed run | yes | HOLDS | 4/4: 81.4/82.2/81.5/81.4 vs 81.9/81.5/81.8/81.8 (3/3 without s2) | Verdict L27 |
| HU3 | A4g L222-223 | Abstention prompt: unsolvable abstention >= own + 10 pts and intact <= 15%, in every seed run | yes | FAILS | 2/4. Unsolvable +0.081/+0.131/+0.107/+0.089; intact 0.134/0.133/0.114/0.087. 1/3 without s2 | Verdict L27 |
| HU4 | A4g L224-227 (s0, s1) | On the audited pool (count-unknown band, abstention prompt), the ill-posed share admitted with the rescue_uv r3 solver <= own r3 - 10 pts | yes | FAILS | 0/2: 73.2 vs 78.0 and 76.2 vs 81.0 (-4.8 each) | Verdict L27; uv_read.json pools |
| HU5 | A4g L228-229 | Descriptive: rescue_uv vs rescue_abs on Q1u and on the HU4 composition | no | DESCRIPTIVE_ONLY | Q1u uv vs abs: 0.347/0.335, 0.233/0.197, 0.367/0.357. Pool ill-posed: 73.2 vs 77.1 and 76.2 vs 81.2. Ill-posed items with an unknown majority: 16.9 vs 1.0 and 18.4 vs 0.0 | uv_read.json; a4g_uv_read_final_0831.txt |
| HK1 | A4k L310 (= HU1 on Phi; cutoff 08:28) | Phi rescue_uv r3 Q1u <= Phi own r3 - 0.05 in each seed | yes | FAILS | 0/2: 0.597/0.590 vs bars 0.591/0.537 | Verdict L28; dirA4/results/uv_read_phi.json (stored under HU keys) |
| HK2 | A4k L310 (= HU2) | Phi rescue_uv accuracy >= own - 2 | yes | HOLDS | 2/2: 76.9/77.6 vs 77.8/77.8 | Verdict L28 |
| HK3 | A4k L310 (= HU3) | Phi: unsolvable abstention >= own + 10 pts, and intact <= 15% | yes | HOLDS | 2/2: 0.652/0.655 vs 0.478/0.508; intact 0.050/0.046 | Verdict L28 |
| HW1 | A4l L323 (= HU1) | own_uvw (w=3) r3 Q1u <= own r3 - 0.05 in both seeds | yes | FAILS | s1 0.301 vs bar 0.266 (false in the JSON). s0 0.240 is true, but the lineage is formally collapsed (accuracy 21.4 < 0.5 x 73.5). The verdict's "reads true only because of the collapse" contradicts the JSON for s1 | Verdict L29; dirA4/results/uvw_read.json |
| HW2 | A4l L323 (= HU2) | own_uvw accuracy >= own r3 - 2 in both seeds | yes | FAILS | 0/2: 21.4/79.2 vs bars 79.9/79.5 | Verdict L29 |
| HW3 | A4l L323 (= HU3) | own_uvw unsolvable abstention >= own + 10 pts, and intact <= 15% | yes | FAILS | Intact 0.853/0.992; unsolvable 0.966/0.998 | Verdict L29 |
| HW4 | A4l L323 (= HU4) | Pool ill-posed share with the own_uvw solver <= own r3 - 10 pts | yes | INCONCLUSIVE | Both booleans are true but degenerate: pool s0 admits n = 0, pool s1 n = 3 (66.7%). s0 is formally collapsed. s1 is not (accuracy 79.2), but it refuses under the abstention prompt (intact 0.99). The verdict's "COLLAPSE 2/2" is loose for s1 | Verdict L29; uvw_read.json |
| A4l descriptive | A4l L321-322 | own_uvw (w=3) vs rescue_uv (w=1) | no | DESCRIPTIVE_ONLY | Accuracy w=1 vs w=3: 81.4 vs 21.4 (s0), 82.2 vs 79.2 (s1). Intact abstention 0.134 vs 0.853 and 0.133 vs 0.992. Pool admitted n 414 vs 0 and 407 vs 3 | uvw_read.json |
| HL1 | A4i L274-276; s1 added by the addendum L283-285 | loop_uv r4: the share of audited ill-posed items with a non-unknown admitted majority <= own r4 ill-posed share - 15, in each seed | yes | HOLDS | 3/3: 49.0/61.0/56.0 vs bars 65.5/65.0/63.5. Guard and audit were complete by 06:44, which exempts HL1 from the cutoff | Verdict L30; dirA4/results/a4i_hl1.json, a4ij_read.json |
| HL2 | A4i L277; cutoff L281 | loop_uv r4 plain Q1u <= own r4 - 0.10 in each seed | yes | FAILS | 0/3: 0.383/0.233/0.417 vs own r4 0.430/0.323/0.502. Protocol flag: every loop_uv r4 eval finished 07:53-08:26, after the 07:50 cutoff. Read literally, HL2-HL4 are NOT_RUN; the addendum's "until 08:30" does not amend L281 | Verdict L31; a4ij_read.json |
| HL3 | A4i L278 | loop_uv r4 accuracy >= own r4 - 2 in each seed | yes | HOLDS (same cutoff flag as HL2) | 3/3: 81.0/81.4/81.6 vs bars 80.4/78.9/79.4 | Verdict L32 |
| HL4 | A4i L279 | loop_uv r4 intact-twin abstention <= 15% | yes | HOLDS (same cutoff flag as HL2) | 3/3: 0.045/0.062/0.068 | Verdict L32 |
| A4i descriptive | A4i L280 | Admitted n, tracer ratio, unknown plurality by class, ill-posed share of the admitted set | no | DESCRIPTIVE_ONLY | Admitted proposals 460/432/424. Tracer ratio 0.69/0.86/0.73 vs own r4 1.49/1.67/1.51. The audited ill-posed share of the admitted sample is 70.0/80.5/79.0; the verdict and note mislabel it as "raw". Unknown majority on U: 35/97, 32/98, 41/106 | a4i_hl1.json; guard_stats.json |
| HM1 | A4j L294-295 | loop_uv r5 invented-answer target share <= own r5 ill-posed share - 15, in each seed | yes | FAILS | 1/3: 75.0/66.5/62.5 vs bars 66.0/59.0/65.5; only s2 holds | Verdict L33; a4ij_read.json |
| HM2 | A4j L296 | No re-drift: loop_uv r5 invented share <= its r4 value + 5 | yes | FAILS | 0/3: +26.0/+5.5/+6.5 | Verdict L33 |
| HM3 | A4j L297 | loop_uv r5 plain Q1u <= own r5 - 0.10 | yes | FAILS | 2/3: 0.330/0.252/0.437 vs bars 0.378/0.245/0.471; s1 misses by 0.007. The s2 bar uses own r5 0.571; with the raw 0.553 the bar is 0.453 and s2 still holds | Verdict L34 |
| HM4 | A4j L297 | loop_uv r5 accuracy >= own r5 - 2 | yes | HOLDS | 3/3: 81.0/81.8/81.7 vs 80.5/81.0/81.8 | Verdict L34 |
| HM5 | A4j L298 | loop_uv r5 intact-twin abstention <= 15% | yes | FAILS | s0 0.087 passes. s2 0.950 fails (unsolvable 0.991: refusal). s1 is null in a4ij_read.json (written 15:11) because its probe finished at 16:14; unofficially s1 = 0.095, which passes. The row fails on s2. s2 r5 training was killed by the memory cgroup and resumed | Verdict L34; a4ij_read.json; hpabs_qwen3_4b_Base_loop_uv_a4_s1_r5 |
| HX1 | A4h L251 | UMWP: CA(own_clean r3) < CA(own r3) in >= 3/4 seeds | yes | HOLDS | 4/4: 0.492/0.495/0.545/0.660 vs 0.653/0.587/0.727/0.758 | Verdict L35; dirA4/results/umwp_read.json |
| HX2 | A4h L252 | UMWP: CA(own r3) > CA(base) in every seed (0-4) | yes | HOLDS | 5/5: 0.587-0.758 vs base 0.287 | Verdict L35 |
| HX3 | A4h L253-254 | UMWP, each rescue_uv seed: ABS_U >= own + 10, ABS_A <= own + 5, ACC_A >= own - 2 | yes | FAILS | 1/4 (s1). ABS_U 75.7/77.6/77.5/74.2 vs bars 77.3/77.3/78.7/76.1. The ABS_A and ACC_A clauses are met | Verdict L37 |
| HX4 | A4h L255 | UMWP: CA(rescue_uv) <= CA(own r3) - 0.05 in every seed | yes | FAILS | 3/4: 0.663/0.513/0.673/0.570 vs bars 0.603/0.537/0.677/0.618; s0 fails | Verdict L37 |
| A4h descriptive | A4h L256 | UMWP per category; rescue_abs; own r5 | no | DESCRIPTIVE_ONLY | rescue_abs CA 0.608/0.422/0.670, ABS_U 37.5/32.2. The s2 ABS_U is unofficially 36.7: its dump finished inside the window but after the 07:29 read. own r5 CA 0.720/0.650/0.783. Category 5: base 0.583 vs own s0 r3 0.875 | umwp_read.json; umwp_reada_0729.txt |
| HX2b [Qwen3-8B] | A4h add. b L262 | UMWP: CA(own r3) > CA(base) in every 8B seed (0-2) | yes | HOLDS | 3/3: 0.065 -> 0.242/0.203/0.362 | Verdict L36; umwp_read_b.json |
| HX2b [Phi-4-mini] | A4h add. b L262-263 | Same test, Phi s0-1 | yes | PENDING (unread; in-window data) | The read ran at 07:29, before the plain base dump finished (07:47, inside the 07:50 cutoff), and was never repeated. Unofficial: base CA 0.208 vs 0.787/0.785, so it would hold 2/2 | Verdict L36; dirA4/results/umwp/plain_base_Phi-4-mini-instruct.jsonl |
| A4h add. b descriptive | L263 | ABS_U / ABS_A / ACC_A for 8B and Phi; Phi own_ag r3 | no | DESCRIPTIVE_ONLY | 8B own r3: ABS_U 53.4-61.5 (base 38.9), ACC_A 90.7-92.5 (base 74.3). Phi own r3: ABS_U 62.2/62.3. Phi own_ag CA 0.787/0.763. The Phi base ABS dump finished at 08:10, outside the window, so it counts as not run | umwp_read_b.json; umwp_readb_0729.txt |
| HX5 | A4h add. c L329 | UMWP: CA(own_ag r3) < CA(own r3) in every 4B seed (0-2) | yes | HOLDS | 3/3: 0.577/0.497/0.432 vs 0.653/0.587/0.727 | Verdict L38; umwp_read_c.json (booleans only); umwp_readc_0729.txt |
| HX6 | A4h add. c L330 | ABS_A(own_ag) <= own + 5 and ACC_A(own_ag) >= own - 2, in every seed | yes | FAILS | s0 and s1 hold (s1 values unofficial, margin 0.3). s2 is False in umwp_read_c.json. The verifier's printed s2 values (ACC_A 92.5 vs 94.2, "-3.7") do not reproduce the failure arithmetically, so they need re-checking; the boolean stands | Verdict L38; umwp_read_c.json |
| A4h add. c descriptive | L331 | UMWP coev_ag r3, s0-1 | no | DESCRIPTIVE_ONLY | s1: CA 0.683, ABS_U 70.5. s0 (unofficial, in window): CA 0.567, ABS_U 71.8 | umwp_readc_0729.txt |
| A4h add. d descriptive | L335-337 | UMWP on the loop_uv r4 solvers (s0, s2; s1 if trained before 08:10) | no | PENDING (unread; s1 should be reported as not run) | Dumps exist for s0/s2. s1 was trained at 07:39 but never dumped. Unofficial s0/s2: CA 0.693/0.723 (own r3 0.653/0.727; own r5 0.720/0.783), ABS_U 63.9/74.0 | dirA4/results/umwp/{plain,abs}_qwen3_4b_Base_loop_uv_a4_s{0,2}_r4.jsonl |
| KF | base L68-70 | Rule. If PF1 holds and PF2+PF3 hold on >= 2/3 4B seeds and both 8B seeds, the fix works at no cost. If PF2 holds and PF3 fails, report a trade-off. If PF1 fails, make no method claim | no | FAILS (branch applied: "no method claim") | PF1 and PF2 fail. Verdict s.2.4(a), "the abstention guard removes the plain-prompt harm (PF4, HX5)", is a secondary read reported as measured, not a KF method claim | Verdict L11-13, L54 |
| KH | base L71-73 | Rule. If PC1 and PC2 hold, the drift becomes an accuracy cost. If PC1 fails: "filtered SFT is too weak a selection pressure" | no | FAILS (branch applied: "too weak a selection pressure") | PC1 0/4. The verdict does not name KH | Verdict L20 |
| KG [PG1 branch] | base L74 | Rule. PG1 on Phi makes the drift a third-family result | no | HOLDS (triggered) | PG1 2/2 | Verdict L21 |
| KG [PG2 branch] | base L74 | Rule. PG2 makes the fix a third-family result | no | FAILS (not triggered) | PG2 fails | Verdict L21 |
| A4a.1 no-go rule (no ID) | A4a.1 L107 | Rule. If NG1-NG3 all hold, state "label-free guards on the solver's own agreement cannot stop the drift" | no | FAILS (not triggered: NG2 fails) | Verdict s.2 item 2 still states the no-go, citing NG1/NG3, PF1/PF2, PG2, PC3 and exploratory guards. Under L75 that statement is "reported as measured", not the registered no-go | Verdict L15-16, L48 |
| A4b rule (no ID) | A4b L136-138 | Rule. If NE2 holds, "proposer quality at the frontier is the lever". If it fails, the no-go extends to proposer strength | no | FAILS (branch applied: the no-go extends to proposer strength; 7B proposer only) | NE2 fails | Verdict L18 |
| A4e rule (no ID) | A4e L185-187 | Rule. If NF1+NF2 hold, frontier refill is a pool-level escape. If NF1 fails, the no-go extends to self-generated refills | no | FAILS (branch applied: the no-go extends to refills) | NF1 fails | Verdict L19 |
| A4c rule (no ID) | A4c L159-160 | Rule. If HC1-HC3 hold, claim "drift-attributable harm at round 3: confident answering of unanswerable inputs, invisible to benchmarks" | no | HOLDS (triggered) | HC1 4/4, HC2 3/4, HC3 4/4. The verdict adds the A5 limitation that the harm is measured under the plain prompt (F-P1, F-P3) | Verdict L23-25, L49 |
| A4f rule (no ID) | A4f L203-204 | Rule. If HA1-HA3 hold, claim "an abstention-aware reward converts the drifted curriculum into abstention training at no accuracy cost" | no | FAILS (not triggered) | HA1 and HA3 fail | Verdict L26 |
| A4g rule (no ID) | A4g L230-231 | Rule. If HU1-HU4 hold, claim "an abstention attractor stops the band from enriching ill-posed items" | no | FAILS (not triggered) | HU1, HU3 and HU4 fail, with or without s2 | Verdict L27 |

## 4. A5: judge validity, frontier, CTRL-L, TG, HZN, SWAP, S8, MiP

Registration dirA5/docs/A5_registration_2026-09-24.md (hashed 2026-09-24 08:04:11). Addendum A5b hashed 08:33:05. Results: dirA5/docs/A5_A6_A7_results_2026-09-24.md, cited as "Results".

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| J-P1 | s.1 J1F1 L20 | Judge validity on UMWP (200 unanswerable + 200 answerable): >= 85% of unanswerable labelled ill-posed (U or A), <= 12% of answerable, binary kappa vs human labels >= 0.70 | yes | HOLDS | Detection 0.985; false ill-posed 0.045; kappa 0.94; agreement 0.97; per-category detection 0.909-1.0 | Results L10-11; dirA5/results/j1_p1_read.json |
| J-P2 | L21 | Judge-free drift: the UMWP in-band U/A rate ratio at r3 >= 1.25x base in >= 80% of powered own lineages (4B s0-4, 8B s0-2, Phi s0-1). A lineage is underpowered if < 30 answerable items are in band | yes | HOLDS | 10/10. Growth x1.96-4.29 (4B), x1.37-1.86 (8B), x4.00-4.53 (Phi). Minimum n_inband_A = 52. Results says "x1.4-4.5" | Results L12-13; j1_judge_free.json |
| J-P3 | L15, L22 | Same criterion on the 412 validated twin pairs, >= 8/10 lineages, with the 32-sample dumps subsampled to G=8. Registered: exact hypergeometric. Implemented: Monte Carlo, 2,000 draws per item | yes | HOLDS | 8/10: 4B 5/5 (x1.48-1.67), Phi 2/2 (x2.39/x2.56), 8B 1/3 (s2 x1.37; s0 x1.20 and s1 x1.15 fail). The MC deviation is disclosed only in the script docstring | Results L14; j1_judge_free.json; dirA5/code/a5_j1_judge_free.py (hash 08:10:39) |
| J-P2 r1/r2 (descriptive clause) | L15, L21 | r1/r2 rates reported descriptively (~20 plain UMWP passes) | no | NOT_RUN | No r1/r2 UMWP dumps exist, and no stop time is reported | dirA4/results/umwp listing |
| J1F1 default operating points | L17 | Declared inputs, not confirmatory. CA_U = plain prompt, numeric modal share >= 0.5. FA_A = abstention-prompt abstention on answerable items | no | DESCRIPTIVE_ONLY | Base: FA_A 0.043, CA_U 0.287. CA_U: own r3 0.653/0.587/0.727; guard 0.577/0.497/0.432; clean 0.492/0.495/0.545; loop_uv r4 s0/s2 0.693/0.723. loop_uv r4 s1, loop_uv r5 and 8B guard r3 were never dumped | frontier_read.json descriptive; dirA5/code/a5_frontier.py |
| F-P1 | L23 | Guard r3 default CA_U <= (best label-free rule on the paired own r3 - 0.05). The rule is the abstention-prompt Pareto frontier, interpolated at the guard's FA_A. Must hold on UMWP in >= 2/3 AND on twins in >= 2/3 Qwen guard seeds | yes | FAILS | UMWP 0/3: guard 0.577/0.497/0.432 vs own frontier 0.189/0.227/0.188. Twins 0/3: 0.257/0.216/0.192 vs 0.131/0.139/0.167. The guard is dominated | Results L15; frontier_read.json (hash 08:31:37) |
| F-P1 Phi (exploratory) | L23 | Same test, Phi s0-1 | no | DESCRIPTIVE_ONLY | False in 4/4 cells: UMWP 0.787/0.763 vs 0.190/0.202; twins 0.600/0.541 vs 0.329/0.282 | frontier_read.json |
| F-P2 | L24 | Guard r3 FA_A <= own r3 FA_A + 0.05 on intact twins AND on UMWP-answerable, in >= 4/5 guard lineages; no lineage above 0.25 | yes | HOLDS | 5/5; max 0.084 (Qwen s0 twins; own 0.072) | Results L15 |
| F-P3 | L25 | Guard default point within 0.05 CA_U (two-sided) of oracle-tuned own r3 at matched FA_A (tau tuned on a 50% UMWP dev split, scored on test), in >= 2/3 Qwen seeds | yes | FAILS | 0/3: guard 0.573/0.503/0.457 vs tuned own 0.233/0.230/0.276 | Results L15 |
| F-P4 | L26 | AUROC of the plain-prompt modal share (UMWP answerable vs unanswerable) falls >= 0.05 from base to own r3 in >= 8/10 lineages | yes | FAILS | 2/10 (Phi 0.917 -> 0.824/0.830). AUROC rises in 4B (0.810 -> 0.810-0.851) and 8B (0.735 -> 0.795-0.836) | Results L15 |
| FB1 | A5b L158 | rescue_uv r3 CA_U at FA_A = 0.05 (UMWP, abstention-prompt strict Pareto frontier) <= own r3 - 0.03 in >= 3/4 seeds (0, 1, 2, 4) | yes | FAILS | 1/4 (s4 0.116 vs 0.147). s0 0.108 vs 0.136 needs <= 0.106 | Results L16; frontier_b_read.json |
| FB2 | A5b L159 | loop_uv r4 CA_U at FA_A = 0.05 <= own r3 - 0.03 in >= 2/3 seeds | yes | FAILS (arithmetic; JSON verdict null) | 0/2 read: s0 0.141 vs 0.136, s2 0.1225 vs 0.122. The s1 arm is null (no dump), so 2/3 is unreachable. The A5b blindness claim is false for FB2: frontier_read.json (08:28:15, hashed 08:31:37) already held the loop_uv r4 s0/s2 and own r3 Pareto points before the A5b hash | frontier_b_read.json; frontier_read.json |
| FB3 | A5b L160 | The same comparisons on twins at FA_A = 0.10 (rescue >= 3/4, loop >= 2/3) | yes | FAILS | rescue 1/4 (s2 0.097 vs 0.135); loop 0/3. No twin frontier existed before the hash | frontier_b_read.json |
| A5b descriptive arms | A5b L161 | Frontier values for guard, clean, rescue_abs, Phi rescue_uv, and TG abs4/t4 "when available" | no | DESCRIPTIVE_ONLY | UMWP CA_U at 0.05: guard 0.132/0.129/0.117 vs own 0.136/0.135/0.122; clean 0.134/0.118/0.138/0.148; rescue_abs 0.203/0.274/0.210 (worse). TG abs4/t4 were null at 08:35; their dumps now exist, but frontier_b_read was never re-run. Phi rescue_uv has no dump | frontier_b_read.json (08:35:21) |
| C-P1 | s.2 CTRL-L L38 (primary) | r3 twin Q1u: guard <= RND - 0.05 in >= 2/3 seeds AND pooled item-paired bootstrap one-sided p < 0.01 | yes | FAILS | Per seed 1/3: guard/RND 0.257/0.248, 0.216/0.221, 0.192/0.447. Pooled RND - guard = +0.083, p = 0.00025, so the pooled clause is met (driven by s2) | Results L39-40; a5_reads2.json CTRL |
| C-P2 | L39 | r3 UMWP CA: guard <= RND - 0.05 in >= 2/3 | yes | FAILS | 1/3: 0.577/0.542, 0.497/0.515, 0.432/0.748 | Results L43; a5_reads2.json |
| C-P3 | L40 | r3 twin Q1u: guard <= LOWP - 0.05 in >= 2/3. Uninformative if the r1 Jaccard of guard-rejected vs LOWP-rejected items is > 0.6 | yes | HOLDS | 2/3: 0.257/0.272 (s0 misses), 0.216/0.345, 0.192/0.347. Jaccard 0.19/0.14/0.13, so the test is informative | Results L37-38, 44 |
| C-P4 | L41 | abs(Q1u(RND r3) - Q1u(own r3)) <= 0.05 in >= 2/3 ("size is not the cause") | yes | FAILS | 1/3. Differences 0.099/0.095/0.037 (RND 0.248/0.221/0.447 vs own 0.347/0.316/0.410) | Results L41-42 |
| C-P5 | L42 | RND and LOWP lenient accuracy within 2 pts of own r3. No seed count is registered; the script uses 3/3 | yes | HOLDS | 3/3: RND 81.27-82.13 and LOWP 81.77-82.23 vs own 81.47-81.93 | Results L45 |
| C-P5 descriptive (audited share per arm and round) | L34, L42 | A 200-item audit every round, descriptive | no | DESCRIPTIVE_ONLY | Partial. UNSOLVABLE share: r1 RND 32.5-37.5, r1 LOWP 31.0-35.5, r2 RND 46.0-55.5, r2 LOWP s2 44.0. Missing: LOWP s0/s1 r2 and every r3 cell. Not in Results | dirB/runs/qwen3_4b_Base_ctrl_{rnd,lowp}_a5_s{0,1,2}_r{n}/audit_admitted.json |
| T-P1 | s.3 TG L59 | Plain Q1u of loop_uv r4 <= ABS4 - 0.05 in >= 2/3. Registered alternative: abs(diff) <= 0.02 in >= 2/3 means "the unknown reward adds nothing". Anything else is inconclusive | yes | INCONCLUSIVE | loop_uv vs ABS4: 0.3835/0.4296, 0.2330/0.1845, 0.4175/0.4248. Prediction 0/3; alternative 1/3. The Results values 0.384/0.185/0.418 are rounding slips | Results L19-20; a5_reads2.json TG |
| T-P2 | L60 | T4 Q1u <= ABS4 - 0.05 AND <= loop_uv r4, in >= 2/3 | yes | NOT_RUN | Only s1 was read, and it fails (T4 0.238 vs ABS4 0.184 and loop 0.233). s0/s2 were stopped by the kill gate (STOP_T 15:31-15:33) | Results L22; a5_reads2.json unread_seeds |
| T-P3 | L61 | T4 accuracy >= loop_uv r4 - 2 in 3/3, and intact-twin abstention <= 0.15 in 3/3 | yes | FAILS | s1 intact abstention 0.671. The s1 accuracy clause is met (79.67 >= 79.4). s0/s2 not run | Results L21-22 |
| T-P4 | L62 | T4 UMWP CA <= ABS4 - 0.05 in >= 2/3 | yes | NOT_RUN | Only s1 was read (0.513 vs 0.390, fails); s0/s2 were cancelled | a5_reads2.json |
| TG kill gate / T5 gate | L63 | Stop T if intact abstention > 0.30 or the accuracy drop > 5 in any seed. T5 runs only if T-P2 and T-P3 hold | no | DESCRIPTIVE_ONLY (kill gate triggered; T5 gate not met) | Fired on s1 intact abstention 0.671. Accuracy drops were 1.73/2.57; the registration names no reference, and the stricter of two was used. T4 s0/s2 were cancelled (logs end 15:32/15:39, no checkpoint). T5 and T-L were not run | a5_reads2.json; dirA5/STOP_T |
| TG offline partition counts | L55 | Tri-gate partition counts, and the unknown-majority rate on audited-solvable items | no | DESCRIPTIVE_ONLY | Admitted -> kept: 612 -> 390, 579 -> 366, 578 -> 353. Unknown share of kept 0.246/0.290/0.275; the 50% cap never binds. The unknown-majority rate on audited-solvable items was never computed | dirB/runs/qwen3_4b_Base_tg_t4_a5_s{0,1,2}_r4/trigate_stats.json |
| H-P1 | s.4 HZN L70 | r5 twin Q1u: guard <= own r5 - 0.10 in >= 2/3 | yes | HOLDS | 3/3: 0.260/0.175/0.267 vs own 0.478/0.345/0.571. Still holds with the recomputed s2 own value 0.553 | Results L26-34; a5_reads2.json HZN |
| H-P2 | L71 | Guard r5 Q1u <= guard r3 + 0.05 in >= 2/3 | yes | HOLDS | 2/3; s2 fails (0.267 > 0.242). Results says "all HOLD" | a5_reads2.json |
| H-P3 | L72 | Guard r5 accuracy >= own r5 - 2.5 in 3/3 | yes | HOLDS | 79.37/81.77/81.50 vs 78.0/78.5/79.3 | a5_reads2.json |
| H-P4 | L73 | r5 UMWP CA: guard <= own r5 - 0.05 in >= 2/3 | yes | HOLDS | 3/3: 0.560/0.417/0.538 vs 0.720/0.650/0.783 (Results omits s0) | a5_reads2.json |
| H-P5 | L74 | Registered expectation: the guard's audited unsolvable share at r5 >= its r3 value - 5 in >= 2/3 (the drift continues) | yes | HOLDS | 3/3: 36.5 -> 50.5, 39.5 -> 42.0, 34.5 -> 48.0. Results L30 is stale ("s0 pending") | a5_reads2.json |
| S-P1 | s.5 SWAP L86, 89 | G1 (drop own-r2 abstention-flagged items): twin Q1u <= F (own r3) - 0.04 in >= 3/5 powered seeds (>= 10% of items flagged) | yes | FAILS | 2/5 (s1, s4). All seeds powered (29.7-35.4% flagged) | Results L51-52; a5_reads2.json SWAP |
| S-P2 | L90 | G1 <= C1 - 0.03 in >= 3/5 with pooled bootstrap p < 0.05. Alternative: abs(G1 - C1) <= 0.02 in >= 3/5 (the composition account). Otherwise inconclusive | yes | INCONCLUSIVE | Prediction 1/5; alternative 2/5; pooled C1 - G1 = +0.002, p = 0.38. The C1 composition match is incomplete in s0/s1/s3 | Results L53-55 |
| S-P3 | L91 | G1 and C1 accuracy within 2.5 of F in >= 4/5 | yes | HOLDS | 5/5 (all 81.4-82.4) | Results L56 |
| SWAP constructibility gap | L92 (reporting rule) | Report the gap where C1 cannot be matched | no | DESCRIPTIVE_ONLY (gap not reported in Results) | Gap in s0/s1/s3, always on UNSOLVABLE: C1 dropped 9/25/6 fewer items. C1 is larger than G1 (392/383, 413/388, 387/381) | dirB/runs/swap_flag_a5_s{0..4}/swap_build_stats.json |
| S8 re-reference of A4 PF1-PF4 (8B s1-s2) | s.6 S8 L99 | "PF1-PF4 unchanged" on the 8B seeds | - | see A4 (not counted here) | PF1 is a 4B fixed-pool test and belongs to A4 only. PF2 FAILS (8B r3 40.0/36.5). PF3/PF4 [8B] are unread with complete data: the A5 pass labels them NOT_RUN, the A4 pass PENDING, and this ledger uses PENDING | A4 verdict L12-14, L68 |
| G8-P1 | s.6 S8 L100 | r3 UMWP CA: 8B guard <= 8B own r3 - 0.04 in >= 2/3 (own 0.242/0.203/0.362) | yes | NOT_RUN | No 8B own_ag UMWP dumps. The 8B guard s0 lineage was never launched. No stop time reported | dirA4/results/umwp; dirB/runs |
| G8-P2 | L101 | r3 twin Q1u: 8B guard <= own r3 in >= 2/3 | yes | NOT_RUN | Never read. The s1/s2 dumps exist and could be read on CPU; s0 is missing | dirB/results/hp |
| G8-P3 | L102 | 8B guard accuracy >= own r3 - 2 in >= 2/3, and r3 tracer ratio >= 1.1 in >= 2/3 | yes | NOT_RUN | Never read. The s1/s2 evals exist | A4 verdict L68 |
| T5-P1 | s.7 L110 | Invented-answer share at r5 <= T4 + 5 in >= 2/3 | yes | NOT_RUN | Gated on T-P2 and T-P3. T-P3 failed and the kill gate fired | Results L22 |
| T5-P2 | L111 | T r5 Q1u <= own r5 - 0.10 in >= 2/3; accuracy >= own r5 - 2 and intact abstention <= 0.15 in 3/3 | yes | NOT_RUN | Gated, as T5-P1 | Results L22 |
| TL-P1 | L112 | r3 Q1u: T-L <= ABS-L - 0.05 and <= guard r3 of the same seed, in >= 2/3 | yes | NOT_RUN | Gated (T5 was not run) | Results L22 |
| TL-P2 | L113 | Accuracy >= guard r3 - 2; at every round, intact abstention <= 0.15 and unknown as the target on <= 5% of audited-solvable items; 3/3 | yes | NOT_RUN | Gated | Results L22 |
| M-P1 (A5 MiP; not A8 M-P1) | s.8 L117, 120 | MiP-GSM8K / MiP-SVAMP: CA own r3 - base >= +0.10 in >= 4/5 4B seeds | yes | NOT_RUN | The probe was never run: no script, no dumps. MiP-GSM8K/MATH were staged at 09:35, before the 10:30 cutoff; MiP-SVAMP was never staged. Results is silent | data_ext/MiP; dirA5/code |
| M-P2 (A5 MiP) | L121 | MiP: clean r3 CA < own r3 CA in >= 3/4 | yes | NOT_RUN | Never run. The A6 E plain-prompt MiP-GSM8K values point the same way (clean < own 4/4; 0.352-0.577 vs 0.483-0.687), but they are not a read of M-P2 | a6_e_read.json |
| M-P3 (A5 MiP) | L122 | MiP: guard r3 CA <= own - 0.05 in >= 2/3, with answerable accuracy within 3 pts | yes | NOT_RUN | No guard MiP dumps | - |

## 5. A6: matched attribution (D), one-round swap (X), external sets (E)

Registration dirA5/docs/A6_registration_2026-09-24.md (hashed 2026-09-24 09:32:28).

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| D1 | s.D L22-23 (primary, P-attrib) | Bin-matched standardized in-band retention ratio U/A at r3 >= 1.25 in >= 8/10 lineages AND the pooled stratified-bootstrap 95% CI excludes 1. Lineages with < 30 matched U are excluded | yes | HOLDS | 8/10: 4B 1.35-1.99; 8B 1.04/0.97/1.27; Phi 1.58/1.34. Pooled 1.364 [1.311, 1.421]; item-clustered [1.281, 1.460] | Results L60-61; a6_matched.json (read v2 hash 09:52:33) |
| D2 | L24-25 | D1 criterion against hard-answerable items (base gold accuracy <= 3/8), >= 7/10 | yes | FAILS | 2/10 (4B s2 1.2501 and s3 1.50) | a6_matched.json |
| D3 | L26-29 (ratio inverted, disclosed) | Median sharpening s = log(q_base)/log(q_r3) is lower over U than over bin-matched A in >= 8/10 | yes | HOLDS | 10/10, no ties (e.g. 4B s0 2.0 vs 3.52; Phi 2.15 vs 4.82) | a6_matched.json |
| D4 | L30-31 (secondary) | D1 criterion on the 412 twins vs their intact originals, >= 7/10 | yes | FAILS | 5/10: 4B 5/5; 8B 0/3 (1.08/1.05/1.24); Phi 0.75/0.78. Pooled 1.18 [1.13, 1.22], descriptive | a6_matched.json |
| X1 | s.X L39 (primary, P-harm) | One round from own r2: Q1u(CLN1) <= Q1u(RND1) - 0.04 in >= 4/5 4B seeds AND pooled McNemar p < 0.01 | yes | FAILS | 2/5 per seed. Pooled McNemar 159 vs 99 discordant, p = 0.00023, so the pooled clause is met. Mean difference -0.029 | Results L66-71; a6_x_read.json |
| X2 | L40 | UMWP CA(CLN1) <= CA(RND1) - 0.04 in 2/2 Phi seeds (and >= 2/3 8B seeds if run) | yes | FAILS | Phi 1/2 (-0.027, -0.042) | dirA5/results/a6_x2_read.json |
| X2 8B clause | L37-38, 40 (conditional) | The X swap on Qwen3-8B s0-2 "if compute allows (reported as not run otherwise)" | no | NOT_RUN | No 8B cln1/rnd1 runs. Results does not report it as not run | dirB/runs |
| X3 | L41 | abs(Q1u(RND1) - Q1u(own r3)) <= 0.04 in >= 3/5 | yes | HOLDS | 4/5 (s1 0.058 fails) | a6_x_read.json |
| X4 | L42 | CLN1 and RND1 accuracy within 2 pts "in every seed" | yes | HOLDS (4B scope of the hashed read) | 4B 5/5 (gaps 0.13-1.90). The registered X lineages also include Phi s0-1, whose evals exist but are unread, so the read is incomplete under the full scope (5 of 7) | a6_x_read.json; evals/phi4mini_{cln1,rnd1}_a6_s{0,1}_r3 |
| E1 | s.E L48 | Plain prompt: CA(own r3) >= CA(clean r3) + 0.05 in >= 3/4 seeds on >= 2/3 sets (SUM, MiP-GSM8K, MiP-MATH) | yes | FAILS | 1/3 sets: MiP-GSM8K 4/4; SUM 2/4; MiP-MATH 2/4. All 12 differences are positive | Results L65; a6_e_read.json |
| E2 | L49-50 | With an abstention option: abs(abstention(own) - abstention(clean)) <= 0.05 in >= 3/4 on >= 2/3 sets | yes | HOLDS | 3/3 sets, 4/4 seeds each; max gap 0.047 | a6_e_read.json |
| E3 | L51 | A pointer to the existing accuracy reads (HC3); no new read | no | NOT_RUN (by design) | A4 HC3 HOLDS 4/4 | A6 registration L51 |

## 6. A7 items 1-4: differential exit, taxonomy, categories, release

Registration dirA5/docs/A7_registration_2026-09-24.md (hashed 2026-09-24 15:10:38). Read script dirA5/code/a7_exit.py v2 (hash 15:53:54).

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| R-P1 | item 4 | Differential exit, with the A6 D2 definitions: ratio(r5) >= ratio(r3) + 0.10 in >= 2/3 4B own seeds | yes | FAILS | 1/3: 1.190 -> 1.200, 1.116 -> 1.089, 1.250 -> 1.669. n_matched_U 390 (bins 2-4 only), n_hard_A 112. The r3 terms were known from A6 D2 before registration (disclosed) | Results L87-90; a7_exit_categories.json item4_confirmatory |
| R-P2 | item 4 | In-band retention of hard-answerable items is lower at r5 than at r3 in >= 2/3 | yes | HOLDS | 2/3: 0.652 -> 0.563, 0.696 -> 0.714 (not lower), 0.589 -> 0.348 | same file |
| R-P2 std sensitivity | a7_exit.py docstring | R-P2 with the U-standardized hard-answerable retention | no | DESCRIPTIVE_ONLY | Lower in 2/3, the same pattern as R-P2 | same file |
| R-P1 nA>=5 bin sensitivity + per-bin counts | a7_exit.py docstring | R-P1 ratio restricted to bins with >= 5 hard-answerable items | no | DESCRIPTIVE_ONLY | +0.06 / -0.03 / +0.64, i.e. 1/3 as for R-P1. 109 of 112 hard-answerable items sit in bins 2-3; bins 5-6 have none | same file |
| Item 1 taxonomy | item 1 (descriptive) | Six-type typing of audited ill-posed admitted items (4B own s0-2, r1/r3/r5, <= 60 per cell) plus 60 raw r1 items per seed; 25% second pass | no | DESCRIPTIVE_ONLY | 660 items: 540 admitted + 120 raw (s2's raw pool was never fully audited). Pooled r1/r3/r5 %: contradiction 42.2/37.8/43.3; missing premise 18.3/29.4/24.4; ambiguous reference 17.2/15.0/14.4; multiple readings 5.6/6.1/7.8; undetermined ask 15.6/9.4/8.3; other 1.1/2.2/1.7. Second pass n = 165, kappa 0.842 (88.5%). Sampling seeds were set per stratum, not a single Random(20260924) | Results L77-82; dirA5/results/a7_taxonomy/a7_taxonomy_read.json; a7_tax_sample.py |
| Item 2 UMWP categories | item 2 (descriptive) | J-P2 ratio change per UMWP category, 10 lineages | no | DESCRIPTIVE_ONLY | Rises in all 50 cells, x1.12-5.07. Phi "question missing" carries a small-count flag. The category ordering in Results does not hold in every lineage (4B s4: "unrealistic" smallest; 8B s0: "unrelated object" 1.41 > "missing" 1.40) | a7_exit_categories.json item2_descriptive |
| Item 3 release | item 3 (descriptive) | Release package with datasheet | no | DESCRIPTIVE_ONLY | release_v1/ was written 2026-09-24 17:07-17:09 (README, DATASHEET, MANIFEST, build_release.py, 6 folders). It predates the A7b-A10 data, and its completeness is unchecked. Results' "being built" is stale | release_v1 listing |

## 7. A7 item 5: R-Zero official-code replication

Registered in A7 item 5. Operationalised in dirA5/docs/A7_5_addendum_2026-09-24.md, hashed 2026-09-24 19:23:40 as ext/rzero_design/A7_5_addendum_draft.md (sha 966da88d..., byte-identical). Amendments: A7-5a (23:08:19), A7-5b (2026-09-25 02:00:23), A7-5c (07:37:22); none changes RZ-P1 or RZ-P2. Status as of 2026-09-25 11:45 CST.

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| RZ-P1 | item 5; addendum s.5, 8, 9 | Official R-Zero (commit 5699329, Qwen3-4B-Base): the audited ill-posed (U+A) share of the 200-item admitted sample at it3 >= it1 + 10 pts in >= 1 seed run | yes | PENDING (read by T0+84h = 2026-09-28 03:10) | Smoke test PASS 02:33 on both hosts. s0/s1 drivers started 02:35/02:34 in full mode. it1 admitted 3678/3423 (0 over 2,048 tokens). The it1 audits are inputs, not a test: 32.5% (s0) and 34.0% (s1). s0's it1 solver stage finished 11:37:25, taking 4.47 h against a 3.7 h upper value; s1 is still running. s0 it3 admitted set projected ~01:20-02:50 on 26 Sep | ext/rzero_runs/s{0,1}/events.tsv, ledger_s{0,1}.log; dirB/runs/rzero_qwen3_4b_Base_s{0,1}_it1/audit_admitted.json | [26 Sep: early-run HOLDS, see section 15]
| RZ-P2 | item 5; addendum s.6, 9 | The final R-Zero solver's UMWP plain CA (tag plain_rzero_s{S}_it3) > base 0.2867 strictly, in every seed run, with 1,200 parseable ids by T0+72h | yes | PENDING (read at T0+72h = 2026-09-27 15:10) | No final solver yet. s0 final projected ~08:50 on 26 Sep (the rule's 1.1x estimates), ~29 h before the T0+71h compute stop. A downgrade to RZ-P1-only, which would make RZ-P2 NOT_RUN for that seed, needs ~29 h of lost time | events.tsv; rz_clock.py; A7_5b_amendment s.6 | [26 Sep: early-run HOLDS, see section 15]
| Secondary: numeric-label subset | addendum s.9 | RZ-P1 difference on items whose pseudo-label is numeric | no | PENDING | Input: numeric-label share of the it1 admitted set 81.3%/81.7% | {run}_solver_v1_summary.json |
| Secondary: UNSOLVABLE-only | addendum s.9 | RZ-P1 difference counting U only | no | PENDING | Input: it1 U-only 27.5%/30.0% | it1 audit_admitted.json |
| Descriptive: it2 audit | addendum s.5, 9 | Ill-posed share of the it2 admitted set | no | PENDING | it2 not produced yet | events.tsv |
| Descriptive: pool sizes and histograms | addendum s.9 | Generated / parsed / scored / admitted counts per iteration | no | PENDING (it1 only) | s0 8000/7499/6859/3678; s1 8000/7489/6593/3423 | solver_v1_summary.json |
| Descriptive: final ACC_A | addendum s.9 | UMWP answerable accuracy vs base 91.3 | no | PENDING | No final solver yet | - |
| Descriptive: ABS_U / ABS_A | addendum s.6, 9 | Abstention-prompt dumps vs base | no | PENDING | No final solver yet | - |
| Descriptive: X8 finding | addendum s.8, 9 | Report if an admitted set has < 512 rows after the overlong filter | no | PENDING | Not triggered at it1 | ext/rzero_runs/s{0,1}/ |
| Exploratory: chat-template CA | addendum s.6, 9 | plainchat pair; cannot change RZ-P2 | no | PENDING | No final solver yet | - |
| A7-5c stack-equivalence check | A7_5c_D1_correction_2026-09-25.md (explicitly not registered) | 3-4 h check of vLLM 0.9.2 + torch 2.7.0+cu128 after the seeds finish. Needs the author's go-ahead and a registration first | no | NOT_RUN (not authorized) | Flagged downside: ~10-15% chance the replication would have to be reported as stack-sensitive | A7_5c file, last paragraph |

## 8. A7b: blind re-audit

Addendum dirA5/docs/A7b_addendum_2026-09-24.md (hashed 17:12:00). Read script a7b_read.py v2 (17:19:55).

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| B-P1 | A7b | Blind ill-posed share at r5 >= r1 + 10 pts in >= 2/3 seeds | yes | HOLDS | 3/3: 51 -> 74, 49 -> 72, 53 -> 81 (original verdicts 53 -> 78, 50 -> 73, 59 -> 80). 900 items, 0 missing | Results L102; a7b_blind/a7b_read.json |
| B-P2 | A7b | abs(D_blind - D_orig) <= 7.5 pts, pooled | yes | HOLDS | D_blind +24.67, D_orig +23.00; difference CI [-2.67, +6.33] | Results L103-104 |
| B-P3 | A7b | Binary kappa of blind vs original verdicts over the 900 items >= 0.60 | yes | HOLDS | 0.783 (90.8% agreement) | Results L105 |
| A7b descriptive block | A7b "Descriptive" | Shares per cell, r3, three-class kappa, bootstrap CIs | no | DESCRIPTIVE_ONLY | Pooled r3: blind 78.3, original 79.7. Three-class kappa 0.714. D_blind CI [+17.3, +32.0] | a7b_read.json |

## 9. A8: interpretive multiplicity

Registration dirA5/docs/A8_registration_2026-09-24.md (hashed 23:16:50). A8a_clarification_2026-09-24.md and read script a8_read.py both at 23:30:28.

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| M-P1 (A8) | A8; unchanged by A8a | At base, the READINGS+MIXED share is >= 20 pts higher for UMWP-unanswerable than for hard-answerable items, and the bootstrap 95% CI excludes 0 | yes | HOLDS | 0.938 vs 0.259, difference +0.679 [0.589, 0.768]. 896/896 units valid. Hard-answerable items: 82/112 ERRORS | Results L113-115; dirA5/results/a8/a8_read.json |
| M-P2 (A8) | A8; A8a item 1 (sensitivity) | The gap-flag rate on unanswerable units is lower at r5 than at base by >= 5 pts in >= 2/3 | yes | FAILS (opposite direction) | 0/3: 15.1% -> 62.1/49.1/70.3% | Results L118-121 |
| M-P3 (A8) | A8; A8a item 3 (NO_READING excluded) | The mean number of distinct reading clusters on U is lower at r5 than at base in >= 2/3 | yes | HOLDS | 3/3: 3.87 -> 2.83/3.16/2.65 | Results L116-117 |
| M-P2 untruncated sensitivity | A8a item 1 | M-P2 on untruncated samples | no | DESCRIPTIVE_ONLY | 0/3 (62.0/49.6/70.2%) | a8_read.json |
| Hard-answerable analogues | A8 "Descriptive" | The same quantities on hard-answerable units | no | DESCRIPTIVE_ONLY | Clusters 1.35 -> 1.34-1.38. Gap flag 1.2% -> 3.1-6.0%. READINGS+MIXED 0.26 -> 0.21-0.28 | a8_read.json |
| Per UMWP category | A8 "Descriptive" | Unanswerable-unit quantities per category, base vs pooled r5 | no | DESCRIPTIVE_ONLY | Missing key info: clusters 4.17 -> 2.84, flag 12.8% -> 60.3%. Ambiguous key info: 3.98 -> 2.92, 19.2% -> 69.0%. Unrealistic (n = 10): 2.8 -> 2.97. The other categories are too small to read | a8_read.json per_category |
| r3 units | A8 ("if time permits") | The same quantities at r3 | no | DESCRIPTIVE_ONLY | U clusters 2.92/3.32/2.91; U gap flag 55.9/45.6/64.7% | a8_read.json r3_descriptive |
| Second-pass agreement | A8; A8a item 4 | Independent second pass on 25% of base units | no | DESCRIPTIVE_ONLY | n = 56. Source kappa 0.915 (94.6%); gap-flag kappa 0.925; exact reading-count agreement 85.7% | a8_read.json second_pass |
| All-cluster count, cuts, reproducibility | A8a items 1, 3; A8 "Generation" | All-cluster count; cut and truncation counts; agreement with the original dumps | no | DESCRIPTIVE_ONLY | All-cluster U 4.44 -> 2.68-3.41. Answer vectors identical to the original dumps in 16/224 (base); vLLM per-request seeding is not stable across batches | a8_read.json |
| Disagreement-source mix at r5 (derived) | derived from the registered labels | Share of U units where all 8 answers are equal (source NONE) | no | DESCRIPTIVE_ONLY | 0/112 at base vs 21/13/37% at r5. READINGS+MIXED 0.94 -> 0.66/0.78/0.51 | a8_read.json cells |
| Silent vs flagged answering (post hoc, NOT registered) | dirA5/code/a8_silent.py (25 Sep 03:10) | Confident answering (modal share >= 0.5) on U units, split into silent and gap-flagged | no | DESCRIPTIVE_ONLY (exploratory) | Confident U units 12.5% -> 75.0/63.4/78.6%. Confident AND silent 6.2% -> 5.4/7.1/6.2%. At sample level, flagged answers 14.4% -> 47.5-67.2% and silent answers 82.3% -> 29.6-49.6% | a8/a8_silent_exploratory.json |

## 10. A9: per-class sharpening law, other families

Registration dirA5/docs/A9_registration_2026-09-25.md (hashed 00:00:42). Read script a9_read.py (00:22:53).

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| L-P1 | A9 | In each of Qwen3-8B-Base s0 and Phi-4-mini s0 (fixed pool qwen_s0_r1: 639 S / 180 A / 339 U), beta_S > beta_U, with the ordering in >= 90% of 1,000 item-bootstrap resamples | yes | HOLDS | 8B: 2.55 [2.25, 3.45] vs 1.45 [1.15, 1.80], 100%. Phi: 2.5 [2.15, 8.0] vs 1.0 [1.0, 1.0], 100%. Caveat: Phi beta_U sits at the grid floor (1.00) in 100% of resamples and beta_S at the ceiling (8.00) in 4.4%, so the Phi pass rests on boundary estimates | Results L139-145; a9_law_family.json |
| L-P2 | A9 | Point estimates ordered beta_S > beta_A > beta_U in each family | yes | HOLDS | 8B 2.55 > 1.50 > 1.45. Phi 2.5 > 1.1 > 1.0 (beta_A at the floor in 24.8%). Bootstrap S>A>U share 59% / 75.2% (descriptive) | a9_law_family.json |
| A9 fit and transfer | A9 "Descriptive" | Betas, intervals; own vs transferred vs single-rate fit | no | DESCRIPTIVE_ONLY | MAE r1-r3 (pts): own 1.99 (8B) / 6.56 (Phi); 4B-transfer 4.25 / 5.00; single rate 5.39 / 11.92. For Phi, the transferred 4B betas fit better than Phi's own | a9_law_family.json |
| A9 model-free matched retention | read-script only (a9_read.py), not in the registration text | Matched-agreement in-band retention, U vs S, r1-r3 | no | DESCRIPTIVE_ONLY | 8B U 0.79/0.67/0.65 vs S 0.59/0.47/0.33. Phi U 0.71/0.68/0.66 vs S 0.46/0.38/0.39 | a9_law_family.json |
| Qwen3-4B model-free check (post hoc, NOT registered) | dirA5/code/law_modelfree_qwen4b.py (25 Sep 11:33) | The same check on the qwen_s0 / qwen_s1 pools, r1-r5 | no | DESCRIPTIVE_ONLY | r5 S/A/U: 0.31/0.474/0.608 and 0.324/0.476/0.56. The Results sentence "S < A <= U at every round" is contradicted at qwen_s1 r1 (A 0.741 > U 0.687). Only S < min(A, U) holds at every round | dirA5/results/law_modelfree_qwen4b.json |

## 11. A10: open-weight judge

Registration dirA5/docs/A10_registration_2026-09-25.md (hashed 01:13:08). Read script a10_read.py v2 (01:20:14).

| ID | Registered in | Statement (threshold) | Confirmatory? | Outcome | Key numbers | Source |
|---|---|---|---|---|---|---|
| J2-P1 | A10 | Open-weight judge (Qwen2.5-32B-Instruct, greedy) on the 900 A7b blind items: pooled ill-posed share at r5 >= r1 + 10 pts, with the bootstrap CI excluding 0 | yes | HOLDS | 44.3% -> 63.7%, +19.3 [11.7, 27.0]. Per seed +18/+18/+22. 0 missing | Results L157; a10/a10_read.json |
| J2-P2 | A10 | Binary kappa of the open-weight judge vs A7b blind Opus over the 900 items >= 0.40 | yes | FAILS (narrowly) | kappa 0.379 (70.6% agreement). The strict-parse kappa 0.404 cannot rescue it: that parse leaves 79/900 items (8.8%) unparsed, above the registered 5% limit, and the primary parse follows the registered "last such line" rule. Ill-posed rate 57.1% (open-weight) vs 68.3% (Opus) | Results L158-161; a10_read.json |
| A10 descriptive block | A10 "Descriptive" | Per-cell shares, three-class kappa, agreement with the original Opus verdicts | no | DESCRIPTIVE_ONLY | Cells r1/r3/r5: s0 46/63/64, s1 39/70/57, s2 48/57/70. Three-class kappa 0.325. Binary kappa vs original Opus 0.382 | a10_read.json |
| A10 read-script sensitivities | a10_read.py v2 docstring | Strict parse; comparison with the Opus drift on the same items | no | DESCRIPTIVE_ONLY | Strict parse: drift +21.4 [13.0, 29.4]. Same items: D open-weight +19.3 vs Opus +24.7, difference -5.3 [-14.3, +3.0] | a10_read.json |

## 12. Corrections the verifiers made

Outcome and structure
- A3: P1, P2 and KT2 are split by model. Llama P1 and P2 read FAILS as worded; the verdict read P1 HOLDS from the r1->r2 rise alone. KT2 [Llama] FAILS (1/2).
- A3: P5 moved from PENDING to FAILS. This is a ledger read that the verdict never made, and it needs the author's confirmation. A3a-bound moved from PENDING to FAILS: neither branch is triggered, so the scale sentence is not licensed.
- A3: the collapse-KC1 evidence rule moved from INCONCLUSIVE/confirmatory to triggered/non-confirmatory. The verdict's decision not to draw KC1 from it overrides "regardless of the accuracy margins" and must be disclosed.
- A4: the 8 A5 items were removed from the A4 list (they are counted under A5). PF4 [8B]'s abstention clause was computed unofficially: 1/2, not "would hold".
- A4: HM5 s1 is now readable (unofficially 0.095, passes); the row still FAILS on s2. HW4 stays INCONCLUSIVE: only s0 is formally collapsed. HW1 fails on s1 regardless of the collapse.
- A4: the A4i "raw" 70.0/80.5/79.0 is an audited admitted-sample share. NG2 is +11.93, not +12.0.
- A5: J-P3 was implemented with Monte Carlo (2,000 draws per item), not the registered exact hypergeometric; this is disclosed only in the script docstring. FB2's FAIL is arithmetic (the JSON verdict is null), and the A5b blindness claim is literally false for FB2.
- A5: a SWAP C1 constructibility-gap row was added (C1 is 9/25/6 unsolvable items short in s0/s1/s3). It bears on S-P2 and is not reported in Results.
- A5: H-P2 holds 2/3, not "all". H-P5 holds 3/3; Results L30 is stale.
- A6: X4 HOLDS only in the hashed read's 4B scope, with Phi s0-1 unread. The X2 8B clause was added as NOT_RUN.
- A7-A10: RZ-P1/RZ-P2 status was updated; the downgrade risk to RZ-P2 is small. J2-P2's strict-parse kappa (0.404) cannot rescue it. The Qwen3-4B model-free "S < A <= U" is contradicted at qwen_s1 r1. Phi boundary caveats were added to L-P1/L-P2.

Numbers in the A3 verdict (no outcome change)
- P1 Llama readable r1 range is 0.99-1.14. r6 confident share is 87.0-94.0; r4-r6 ratio 1.32-2.22.
- P2 Qwen r1->r3 rise is +18.0 to +27.5, not +13 to +27; the same error appears in s.2b.
- KA3-1 Q1u at r3 (s0-s2) is 0.316-0.410, not 0.35-0.41. KC1 s1 is +0.27; at r2, s0 is +0.43. KC2 s1 spread is 0.27.
- r4-r6: max accuracy drop -1.83. s2 r5 Q1u is 0.553, not 0.571.
- P3: 8B s0 is +0.019 raw. Several cells shown as "pending" had files before the verdict.

Cross-registration and file-level notes
- The own r5 s2 Q1u constant 0.571 used in A4j HM3 and A5 H-P1 differs from the raw recomputation 0.553. Neither outcome changes.
- HX6 s2: the verifier's printed values do not reproduce the stored False boolean arithmetically. Re-check the values; the label stands.
- Storage quirks: HK reads are stored under HU keys (uv_read_phi.json) and HW reads under HU keys (uvw_read.json). umwp_read_c.json holds booleans only. NF1 has no result JSON (hash log only). The illposed_share field in the CTRL-L and 8B own_ag audits is the UNSOLVABLE share.
- The A4 verdict s.3.1 calls the Qwen3-4B law fit a "registered read", but A4a.1 registers no MAE threshold, so it is not in this ledger. The A4 verdict revision 2 statement "all pending cells filled" is not accurate (see section 13).
- ID collisions: A5 MiP M-P1..3 vs A8 M-P1..3; A6 D1..D4 vs the A7-5c deviation "D1"; A6 X1/X2 vs the A7-5 stop codes X1..X8; SWAP arm C1 vs CTRL-L C-P*. P-attrib / P-harm / P-drift / P-nogo are labels, not extra predictions.

## 13. Open items (listed for the author; nothing is scheduled)

Registered reads with complete data but no official read. These are CPU-only; the unofficial outcome is in brackets.
- PF3 [8B] (HOLDS 2/2), PF4 [8B] (1/2), NF2 (FAILS, 74.1%), HX2b [Phi] (HOLDS 2/2).
- HM5 s1 official value (0.095); A4h add. d (s0/s2).
- G8-P2 / G8-P3 on s1/s2; X4 Phi s0-1; A5b TG abs4/t4 frontier (frontier_b_read never re-run).

Disclosures the registrations require ("not run", with stop time) that no results document makes:
- OLMo PG1/PG2; NE1/NE2 with the 14B proposer; loop_uv s1 r4 UMWP (A4h add. d); X2 8B clause.
- S8 G8-P1..3 (the 8B guard s0 lineage was never launched); A5 MiP M-P1..3; J-P2 r1/r2 passes; missing CTRL-L audits (LOWP s0/s1 r2, all r3).

Protocol flags to disclose with the ledger:
- A3: the Llama reads rest on seeds 2-4 (s2 was added only via the hash log). Five r1 cells are still on the v3 evaluator. The verdict tables omit the strict numbers.
- A4: the A4g 07:45 and A4i 07:50 cutoffs (see the sensitivity table). The verdict L48 no-go is stated although its registered trigger is not met.
- A5: the J-P3 MC deviation and the FB2 blindness claim.
- A10: verdicts.jsonl was written 01:15, before the read-script hash (01:20:14), and hashed in the same entry, which does not say "not read".

Pending by clock: RZ-P2 at 2026-09-27 15:10 and RZ-P1 at 2026-09-28 03:10 (the live R-Zero job). [Resolved 26 Sep: see section 15.]

## 14. Official reads of the open items (25 Sep 12:40-12:45, hashed scripts only, CPU only; workflow wf_a9a31506-b1b)
| Item | Script (registered hash line) | Official outcome | Numbers |
|---|---|---|---|
| PF3 [Qwen3-8B s1-s2] | dirA3/code/a4_reads.py (L622) | HOLDS 2/2 | lenient acc own_ag r3 vs own r3: 83.40 vs 83.67, 83.27 vs 83.53 (bars 81.67 / 81.53) |
| PF4 [Qwen3-8B s1-s2] | a4_reads.py (Q1u clause only) | Q1u clause HOLDS 2/2; abstention clause 1/2 (by hand from hashed outputs, no hashed boolean) -> PF4 counted FAILS (1/2) | Q1u 0.039 vs 0.073, 0.114 vs 0.129; abstention 0.510 vs 0.524 (fails), 0.620 vs 0.613 |
| NF2 | dirA3/code/a3_guard_diag_score.py (L433) | FAILS | r5 admitted evolved set 74.1% ill-posed (U 119 + A 76 of 263) vs bar <= 50%; NF1 now also has a result JSON (73.6%) |
| HX2b [Phi-4-mini] | dirA3/code/umwp_read_b.py (L650) | HOLDS 2/2 | CA base 0.208 vs own r3 0.787 / 0.785 |
| HM5 seed 1 | dirA3/code/a4ij_read.py (L699) | s1 passes (0.095 <= 0.15); HM5 still FAILS (s2 0.950) | |
| A4h addendum d | dirA5/code/a5_frontier.py (L761) output frontier_read.json | DESCRIPTIVE, partly read | loop_uv r4 UMWP CA 0.693 (s0) / 0.723 (s2), ABS_A 1.3% / 2.7%; ABS_U, ACC_A, CA_by_cat need a new hashed script (not run) |
| G8-P2 [8B s1-s2] | no G8-specific read; by arithmetic on the hashed a4_reads output | HOLDS 2 of 3 by arithmetic (s0 never launched) - treated like FB2 (arithmetic) | Q1u 0.039 <= 0.073, 0.114 <= 0.129 |
| G8-P3 [8B] | none | NOT_RUN (tracer-ratio clause undefined in hashed code; accuracy clause 2/2) | |
| X4 [Phi s0-1] | none (a6_x_read covers Qwen only) | NOT_RUN for Phi; X4 HOLDS in the 4B scope | |
| A5b TG abs4/t4 frontier | dirA5/code/a5_frontier_b.py (L774) | DESCRIPTIVE; FB1-FB3 unchanged (FB1 False, FB2 None, FB3 False) | |
Updated confirmatory totals (A4 PENDING rows resolved: PF3 [8B] HOLDS, PF4 [8B] FAILS, NF2 FAILS, HX2b [Phi] HOLDS; A5 G8-P2
HOLDS by arithmetic): 58 HOLDS / 56 FAILS / 3 INCONCLUSIVE / 14 NOT_RUN / 2 PENDING (RZ-P1, RZ-P2), of 133.
Process note: two agents and the checker ran small unhashed stdlib python snippets (JSON loading / row counting only, no GPU,
disclosed); no outcome depends on them.

## 15. R-Zero reads (A7 item 5; early run 26 Sep 2026 06:47 CST, hashed rz_read.py only; revised 07:10 after two verifiers)
Both seed drivers ended with exit 0 (seed 0 on <HOST> at 03:53, seed 1 on <HOST> at 06:46), in full mode. The it3 samples
were audited with the registered workflow rz_audit200_wf.js (sha b4d89f1d, prompt unchanged; run wf_16fa0d19-518, 06:25-06:29,
16/16 jobs, 200/200 per run) and assembled at 06:29 (T0+84h limit: 28 Sep 03:10). Script: `python3 ext/rzero_design/rz_read.py`
(sha 1fbb5350, registered line 848); early-run output ext/rzero_runs/rz_read_final_2026-09-26.json (sha 6d534b13...). Two
independent verifiers re-ran it byte-identically and confirmed every registered number below (wf_2a0657d5-cb9).

DEVIATION (timing): the registration fixes the read times: RZ-P2 is read at T0+72h (27 Sep 15:10) and RZ-P1 at T0+84h (28 Sep
03:10). The 06:47 run is an EARLY run of the registered script, made because all A7-5 compute had ended. The official reads
are the re-runs of the same script at the registered times by ext/rzero_design/rz_reread.sh (15:11 27 Sep, 03:11 28 Sep;
log ext/rzero_runs/rz_reread.log with a byte comparison), to be recorded here. They can differ only if an input changes,
e.g. if a seed were abandoned (rz_stop.sh) and a replacement seed id 2/3 started before T0+71h; both seeds finished and no
seed will be abandoned.

DISCLOSURE (blinding): like the original own-loop audits before A7b, these audits were NOT blind to seed or iteration: the
frozen prompt runs each judge's fetch inside dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>, and ids (rz<S>i<i>r<n>) encode seed and
iteration. A7b's blind result covers only the own loops. A blind re-audit of all 1,200 R-Zero audit items is registered as
A7-5d (dirA5/docs/A7_5d_addendum_2026-09-26.md) and reported in section 16: blind drift +22 / +26 vs +23 / +27, RB-P1..P3 all HOLD.

| Item | Outcome of the early run (official read at the registered time pending) | Numbers |
|---|---|---|
| RZ-P1 | HOLDS (2/2 seed runs; needs >= 1) | ill-posed (U+A) share it1 -> it3: s0 32.5% -> 55.5% (+23.0 pts); s1 34.0% -> 61.0% (+27.0 pts); it2 (descriptive) 44.0% / 42.5% |
| RZ-P2 | HOLDS (2/2 seed runs; needs all) | UMWP plain CA base 0.2867 -> s0 0.3383, s1 0.3033 (strict ">") |
| Secondary: UNSOLVABLE-only | printed, cannot change RZ-P1 | +22.0 / +24.0 pts |
| Secondary: numeric-label subset | printed, cannot change RZ-P1 | +25.99 / +28.63 pts |
| Descriptive: pools | - | generated / parsed / scored / admitted / unique admitted / over 2,048 tokens: s0 it1 8000/7499/6859/3678/3631/0, it2 8000/7083/6290/3817/3750/0, it3 8000/6840/6379/4271/4245/0; s1 it1 8000/7489/6593/3423/3360/0, it2 8000/7801/6033/3697/3614/0, it3 8000/7917/7391/4941/4931/0. Admitted share of scored 53.6 -> 60.7 -> 67.0% (s0), 51.9 -> 61.3 -> 66.9% (s1); numeric-label share of admitted 81.3-84.9%. Score histograms in the rz_read output |
| Descriptive: final ACC_A | - | 89.0 / 89.8 vs base 91.3 |
| Descriptive: abstention prompt | - | CA 0.127 / 0.120 vs base 0.113 (+1.3 / +0.7 pts, untested); ABS_U 55.5 / 53.5% vs 44.2%; ABS_A 4.9 / 4.8% vs 4.3%. For comparison our own r3 solvers: CA 0.193-0.215, ABS_U 66-69% (abs_qwen3_4b_Base_own_a3_s0-4_r3) |
| Descriptive: X8 finding | not triggered | full batches of 512 per admitted set: s0 7 / 7 / 8, s1 6 / 7 / 9 (it1 / it2 / it3) |
| Exploratory: chat-template CA | cannot change RZ-P2 | 0.705 / 0.623 vs base 0.300 (ACC_A 95.8 / 95.0 vs 88.2) |

Post-hoc descriptive, NOT registered (dirA5/code/rz_paired_desc.py v2, sha 235cbdfc; output
ext/rzero_runs/rz_paired_desc_2026-09-26.json, sha ae92112f; paired items, exact McNemar, 10,000-draw item bootstrap):
- Plain CA difference: s0 +5.2 pts [1.5, 8.8], p = 0.008; s1 +1.7 pts [-1.7, 5.2], p = 0.40. RZ-P2 is a strict ">" rule and
  holds as registered, but seed 1's rise alone is not distinguishable from zero. Report both numbers with the verdict.
- Plain ACC_A difference: s0 -2.3 pts [-4.5, -0.2]; s1 -1.5 pts [-3.3, 0.3].
- Chat template: CA +40.5 / +32.3 pts (p < 1e-37), ACC_A +7.7 / +6.8. The base model's mean non-answer rate under this
  template is 14.5% (A) / 16.1% (U) against 3.5-6.3% for the final solvers (plain prompt: 2.7-3.9% for all three), so part
  of the rise is answer-format learning. R-Zero trains its solver with solver.jinja inside the chat template, so this
  condition is closer to its training format; the registered plain prompt is the one used for our own loops (HX).
- Own-loop comparator over the matched 2-step span (A7b items, r1 -> r3): +25.7 pts original (54.0 -> 79.7), +27.3 blind
  (51.0 -> 78.3). R-Zero's +23 / +27 over it1 -> it3 is of the same size, from a lower start (~33%).

Confirmatory totals: 58 HOLDS / 56 FAILS / 3 INCONCLUSIVE / 14 NOT_RUN / 2 PENDING (RZ-P1, RZ-P2 early-run HOLDS), of 133;
60 / 56 / 3 / 14 / 0 once the registered-time reads confirm.

## 16. A7-5d: blind re-audit of the R-Zero audit samples (registered 26 Sep 08:22:54; read 08:41, hashed a75d_read.py only)
Registration: dirA5/docs/A7_5d_addendum_2026-09-26.md (sha 332535c3), with a75d_sample.py, a75d_blind_wf.js,
a75d_blind_check.py, a75d_read.py and the six input audit_admitted.json files, hashed before any data after four rounds of
two adversarial Opus reviewers (ext/a75d_reviews/) and a 2-judge pilot on a dummy file. Sample built 08:23:05 (items.jsonl
2f334ea3, key_full.jsonl d5482dfc); pre-launch scan 08:24:00 clean (0 files with R-Zero ids, 0 with >= 25 z-keys).
Pass 1: wf_c8ae8bd7-99d, 48/48 judges, 47 batches passed the blindness check; batch 626-650 dropped (its judge read its own
background-task output file and stopped the task - the pattern the calibration predicted, 1 in 49 original sessions; no
leak). Pass 2: wf_3f5bfa74-1da, batch 626-650 re-judged, passed. All judges claude-opus-5-5 / xhigh; no relayed user request;
47 of 49 judge sessions computed locally (original audit: 39 of 49). Every original verdict reproduced and the original drift
matched the registered RZ-P1 audits (+23.0 / +27.0) before the outcome was computed.

| Item | Outcome | Numbers |
|---|---|---|
| RB-P1 | HOLDS (2/2) | blind ill-posed share it1 -> it3: s0 33.0 -> 55.0% (+22.0), s1 34.0 -> 60.0% (+26.0); original on the same items +23.0 / +27.0 |
| RB-P2 | HOLDS | pooled D_blind +24.0 vs D_orig +25.0; difference -1.0, 95% CI [-3.0, +1.0]; D_blind CI [17.25, 30.75] |
| RB-P3 | HOLDS | blind vs original kappa 0.943 binary (97.2% agreement), 0.923 three-class (95.8%) |
| Descriptive: it2 | - | pooled 43.75% blind vs 43.25% original |
| Descriptive: subsets | - | D_blind - D_orig: multi-line -3.9 (n 42 / 67), single-line -0.6, CJK +3.6 (n 16 / 38), non-CJK -1.1 |
| Descriptive: repeated texts | - | 6 of 6 repeated problem texts got the same blind binary label on every copy |

Headline rule (registered): RB-P2 holds, so the paper's R-Zero drift is the registered original audit (+23.0 / +27.0,
pooled +25.0), with the blind estimate (+22.0 / +26.0, pooled +24.0 [17.25, 30.75]) as robustness. Reading: the R-Zero
admitted-set drift is not produced by judge knowledge of the iteration.

Confirmatory totals (136 rows with RB-P1..RB-P3): 61 HOLDS / 56 FAILS / 3 INCONCLUSIVE / 14 NOT_RUN / 2 PENDING (RZ-P1,
RZ-P2: early-run HOLDS, official at the registered times); 63 / 56 / 3 / 14 / 0 once those reads confirm.
