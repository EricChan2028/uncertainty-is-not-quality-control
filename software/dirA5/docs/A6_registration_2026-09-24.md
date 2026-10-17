# A6 registration (2026-09-24 09:32 CST, before any of its data or statistics)

[internal note removed] A5 lines are read as
registered, unchanged. Interim A5 numbers already seen before this hash (disclosed): TG admitted counts and training
logs; HZN round-4 guard counts and round-4 audits (guard lineage unsolvable 42.0 / 53.5 / 40.0%); CTRL-L round-1 postfilter
counts and round-1 audits; SWAP flag counts are NOT yet seen.

## Primary endpoints of the paper (declared now)
- P-drift: the registered drift results (A3 P1/P2, A4 PG1, NS1) and the judge-free J-P2 (A5).
- P-nogo: the registered label-free no-go (A4 NG1/NG3, PF1/PF2).
- P-attrib (new, D1 below): ill-posed questions stay in the band at matched difficulty.
- P-harm (new, X1 below): the drift-attributable harm holds against a size-matched random curriculum.
All other A6 predictions are secondary. Nothing already read is re-registered.

## D. Matched-difficulty retention on UMWP human labels (CPU; script dirA5/code/a6_matched.py)
Data: the existing G = 8 plain-prompt UMWP dumps (600 unanswerable U, 600 answerable A) for base and own r3 of Qwen3-4B
s0-4, Qwen3-8B s0-2, Phi-4-mini s0-1 (10 lineages). m = numeric modal share of the 8 samples; base bin = round(8 m_base)/8.
Retention at r3 = share of items in the R-Zero band (m in [0.25, 0.75]) at r3, computed within base bins 2/8..6/8
(items in the band at base). Standardized ratio = sum_b wU_b rU(b) / sum_b wU_b rA(b), wU = U's base-bin distribution over
bins that contain at least one A item (renormalized); a lineage with fewer than 30 matched U items is underpowered (excluded).
- D1 (primary, P-attrib): ratio >= 1.25 in >= 8 of 10 lineages, and the pooled stratified-bootstrap 95% CI (2,000 resamples
  within lineage x bin x class, seed 20260924) of the pooled standardized ratio excludes 1.
- D2: the D1 criterion (ratio >= 1.25) against the hard-answerable subset (A items with base gold accuracy <= 3/8, i.e. at
  most 3 of the 8 base samples equal the reference answer) in >= 7 of 10 lineages (underpowered rule as D1).
- D3: per-item sharpening s = log(q_base) / log(q_r3) with q = m clipped to [1/16, 15/16] (larger s = faster sharpening;
  the plan's "log q_r3 / log q_base" is the reciprocal and is inverted so that "lower" means "sharpens less", matching the
  registered hypothesis beta_U < beta_S; disclosed): the median s over U items is lower than the median over bin-matched A
  items (same standardization) in >= 8 of 10 lineages.
- D4 (secondary): the D1 criterion on the 412 validated unsolvable twins vs their intact originals (32 samples,
  MC-subsampled to 8 as in J-P3; base bin from the 32-sample modal share) in >= 7 of 10 lineages.

## X. Size-matched one-round harm swap (GPU; script dirA5/code/a6_cln.sh)
From each lineage's own r2, one round of 60 GRPO steps (arm own, plain prompt, trainer seed = seed) on either CLN1 = the
audited-solvable proposals of that seed's own r3 admitted set plus its intact tracers (as A3 own_clean), or RND1 = a random
subset of the same admitted set of exactly |CLN1| items (RNG 20260924 + seed). Outcomes: plain-prompt twin Q1u, UMWP CA,
lenient accuracy. Qwen3-4B seeds 0-4 (full audits exist); Phi-4-mini seeds 0-1 after a full Opus audit of their r3 admitted
sets; Qwen3-8B only if compute allows (reported as not run otherwise).
- X1 (primary, P-harm): Q1u(CLN1) <= Q1u(RND1) - 0.04 in >= 4 of 5 Qwen3-4B seeds, with pooled paired McNemar p < 0.01.
- X2: UMWP CA(CLN1) <= CA(RND1) - 0.04 in 2 of 2 Phi seeds (and >= 2 of 3 8B seeds if run).
- X3 (size is not the cause): |Q1u(RND1) - Q1u(own r3)| <= 0.04 in >= 3 of 5 Qwen3-4B seeds.
- X4: lenient accuracy of CLN1 and RND1 within 2 points in every seed.

## E. Extraction sweep (GPU, conditional on staging the data; script dirA5/code/a6_sweep.sh)
Sets: SUM (Hallucination Tax), MiP-GSM8K, MiP-MATH (Missing Premise), G = 8; prompt conditions: plain (the probe's), and the
explicit abstention option (the probe's abstention prompt). Checkpoints: own r3 s0-3, clean r3 s0-3 (Qwen3-4B). If a set cannot
be staged by 14:00 CST it is reported as not run.
- E1: CA(own r3) >= CA(clean r3) + 0.05 under the plain prompt in >= 3 of 4 seeds on >= 2 of the 3 sets.
- E2 (scope, expected): with the abstention option, |abstention(own r3) - abstention(clean r3)| <= 0.05 in >= 3 of 4 seeds on
  >= 2 of 3 sets.
- E3: covered by existing accuracy reads (HC3); not re-read.
