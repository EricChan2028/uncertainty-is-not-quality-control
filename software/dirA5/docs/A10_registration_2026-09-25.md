# A10 registration: a second, open-weight judge family on the blind re-audit items (01:13 CST 25 Sep 2026, before its data)

Motivation [internal note removed]: the audited drift rests on one proprietary judge family (Opus). A10 re-judges the 900 A7b
blind items (Qwen3-4B own s0-2 x r1/r3/r5 x 100; neutral keys; judged blind by Opus in A7b) with an open-weight judge.

Judge: Qwen2.5-32B-Instruct (/dev/shm/models/Qwen2.5-32B-Instruct on host A), vLLM, tensor parallel 2, chat template, greedy
decoding (temperature 0), max 400 new tokens, one item per request. Prompt: the frozen audit definitions verbatim (SOLVABLE /
UNSOLVABLE / AMBIGUOUS, the "Be strict ..." sentence, the "written by a Qwen3-4B-Base language model and ADMITTED by a
self-play filter" sentence), asking for a one-line note and then a final line "VERDICT: <SOLVABLE|UNSOLVABLE|AMBIGUOUS>";
the last such line is parsed; an item without a parsable verdict is re-asked once with temperature 0.7 and seed 1, then
counted as missing. Ill-posed = UNSOLVABLE or AMBIGUOUS. Disclosed limitations: the judge is from the same model family
provider as the proposer (Qwen), is much smaller than Opus, and judges one item per request (Opus judged 25 per request).

Predictions (confirmatory; read with a reviewed script hashed before its run):
- J2-P1 (drift with an open-weight judge): pooled over seeds, the ill-posed share at r5 is >= the share at r1 + 10 points,
  and the item-bootstrap 95% CI of the difference (4,000 replicates, seed 20260925, items resampled within seed x round)
  excludes 0.
- J2-P2 (agreement): binary Cohen's kappa between the open-weight verdicts and the A7b blind Opus verdicts over the 900 items
  >= 0.40.
Descriptive: per seed and round shares, three-class kappa, agreement with the original (non-blind) Opus verdicts, the
open-weight judge's overall ill-posed rate. More than 5% missing verdicts makes both predictions None.
Downside flagged before data: a smaller judge may be lenient (calling ill-posed items solvable) or noisy; J2-P1 may then
fail although the Opus drift is real, and the paper would report the drift as judge-dependent in magnitude.
