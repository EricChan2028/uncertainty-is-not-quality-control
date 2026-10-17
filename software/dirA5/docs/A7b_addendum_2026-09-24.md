# A7b addendum: blind re-audit of the round-wise drift (registered 17:12 CST 24 Sep 2026, before any of its data)

Why: the release critic (A7 item 3) found that the Opus audit judges were not blind to round or condition. Each judge's fetch
command ran inside dirB/runs/<RUN> (the run name encodes model, arm, seed and round), returned run="<RUN>", and proposer ids
(prop<seed*100+round>:n) encode seed and round. The round-wise rise of the audited ill-posed share could therefore partly be
judge expectation. This addendum tests that with a blind re-audit. Validity evidence that does not depend on it already
exists (J-P1 kappa 0.94 vs UMWP human labels; judge-free J-P2 10/10 and J-P3 8/10; fixed-pool audits).

Sample: Qwen3-4B-Base own_a3 seeds 0, 1, 2 at rounds 1, 3 and 5; from each (seed, round) 200-item audit sample, 100 items
drawn with random.Random(20260926 + 100*seed + round) (so every item has an original, non-blind verdict). The 900 items are
shuffled together (random.Random(20260926)) and given neutral keys b0000..b0899. The judges see only the problem text, in a
file whose path and name carry no run information (dirA5/results/a7b_blind/items.jsonl), fetched by line range.

Judging: the frozen audit prompt, verbatim in its definitions (SOLVABLE / UNSOLVABLE / AMBIGUOUS, "Be strict: a default
assumption you would have to invent makes it UNSOLVABLE", confidence, note <= 20 words, the same "written by a Qwen3-4B-Base
language model and ADMITTED by a self-play filter" sentence, which is identical for all items), 25 items per judge (36 Opus
judges), verdicts returned through a structured schema instead of the push protocol. Ill-posed = UNSOLVABLE or AMBIGUOUS.

Predictions (confirmatory; read with a script reviewed and hashed before its run):
- B-P1 (the drift survives blinding): blind ill-posed share at r5 >= blind share at r1 + 10 points in >= 2 of 3 seeds.
- B-P2 (no round-dependent judge bias): pooled over the 3 seeds, the blind drift D_blind = share(r5) - share(r1) is within
  7.5 points of the original (non-blind) drift D_orig on the same items: |D_blind - D_orig| <= 7.5.
- B-P3 (item agreement): binary (ill-posed vs solvable) Cohen's kappa between blind and original verdicts, pooled over the 900
  items, >= 0.60.
Descriptive: shares per seed and round (blind and original), r3, three-class kappa, a paired item bootstrap CI (4,000
replicates, seed 20260926, items resampled within seed x round) for D_blind and D_blind - D_orig.
Downside if B-P1 or B-P2 fails: the audited round-wise drift is reported as partly judge-dependent, and the headline drift
evidence becomes the judge-free reads (J-P2/J-P3), the fixed-pool audits and the blind estimate itself.
