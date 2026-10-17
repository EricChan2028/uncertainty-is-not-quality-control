# A9 registration: the per-class sharpening law in other families (00:00 CST 25 Sep 2026, before any of its data)

Motivation [internal note removed]: the per-class law (beta_S 3.75 > beta_A 1.60 > beta_U 1.40, ordering in 100% of bootstrap
resamples) was fitted on one fully audited fixed pool scored by Qwen3-4B checkpoints only. This item tests it in two other
families on the SAME audited pool.

Data: the fixed pool dirA3/results/pool_audit/qwen_s0_r1 (1,158 Qwen3-4B round-1 proposals, every item Opus-audited
SOLVABLE / AMBIGUOUS / UNSOLVABLE) plus the same planted tracers (seed 1), scored by the R-Zero band guard exactly as the
Qwen3-4B law data (a3_guard.py --seed 1 --abstain_prompt 0 --guard rzero; G = 8) with the base model and the own-loop
checkpoints r1, r2, r3 of Qwen3-8B-Base seed 0 (qwen3_8b_Base_own_a3_s0_r*) and Phi-4-mini-instruct seed 0
(phi4mini_own_a4_s0_r*). Output dirA3/results/guard_diag/fam_qwen3_8b_s0/<ck> and fam_phi4mini_s0/<ck>.

Fit: identical to dirA3/code/a4_law_stable.py (log-space model, q0 = (8 p_mode + 0.5) / 9 at base, in-band share per class
per checkpoint, grid 1.00-8.00 step 0.05, squared error over t = 1..3), per class and per family.
Predictions (confirmatory; read with a reviewed script hashed before its run):
- L-P1: in each of the two families, beta_SOLVABLE > beta_UNSOLVABLE, and the ordering holds in >= 90% of 1,000 item-bootstrap
  resamples (items resampled within class, random seed 20260925).
- L-P2: in each of the two families the point estimates are ordered beta_S > beta_A > beta_U.
Descriptive: the betas and their bootstrap intervals; predicted vs observed ill-posed share of the in-band set per checkpoint
under each family's own betas and under the Qwen3-4B betas (transfer); the single-shared-rate comparison.
Downside flagged before data: Phi-4-mini is an instruction-tuned model with a different base agreement distribution, and only
3 training rounds are available per family; the ordering may not hold, in which case the law is reported as Qwen-specific.
