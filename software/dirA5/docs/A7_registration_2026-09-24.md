# A7 registration (2026-09-24 15:10 CST, before any of its data or statistics)

[internal note removed] Items 1-3 are descriptive (no
prediction can fail); items 4-5 are confirmatory with bounded downside, reported whatever the outcome.

## 1. Linguistic taxonomy of ill-posedness (descriptive)
An Opus typing pass labels audited ill-posed (UNSOLVABLE or AMBIGUOUS) admitted problems with one primary type: missing
premise / quantity; contradictory or impossible condition; ambiguous reference or scope; multiple readings of the question;
unanswerable question form (asks for something not determined); other. Sample: for Qwen3-4B own seeds 0-2 at rounds 1, 3, 5,
up to 60 audited ill-posed admitted items per (seed, round) (Random(20260924)), plus 60 ill-posed items from each seed's raw
round-1 proposals (pool audits). A second independent typing pass on a 25% subsample gives type agreement (kappa). Reported:
the type mix per round and its change; no thresholds.

## 2. Judge-free UMWP category breakdown (descriptive)
The J-P2 in-band unanswerable / answerable ratio change (base -> r3) computed per UMWP unanswerability category (missing key
information, ambiguous key information, unrealistic conditions, unrelated object, question missing) for the 10 own lineages.

## 3. Resource release (descriptive)
Package, for release, the audited admitted-problem pools per round and lineage (problem text, Opus verdicts, base-model
agreements), the fixed-pool full audits, the planted twins, the judge prompt, and the judge-free probe, with a datasheet.

## 4. Differential exit over time on UMWP (confirmatory)
Using the D2 definitions of A6 (bin-matched standardized retention ratio of unanswerable vs hard-answerable items, base gold
accuracy <= 3/8, reference required), computed at r3 and at r5 for Qwen3-4B own seeds 0-2 (the lineages with r5 UMWP dumps).
- R-P1: ratio(r5) >= ratio(r3) + 0.10 in >= 2 of 3 seeds (hard answerable items leave the band over time; unanswerable stay).
- R-P2: the in-band retention of hard-answerable items at r5 is lower than at r3 in >= 2 of 3 seeds.
Downside if both fail: the paper states that hard answerable and ill-posed items are retained alike through round 5.

## 5. Replication in the official R-Zero code (confirmatory; design now, exact commands in an addendum before data)
Run the public R-Zero implementation with its default challenger / solver and uncertainty filter on Qwen3-4B-Base (or the
closest supported base model), 1-2 seeds, 3 iterations, with the repository's defaults except compute settings. Audit, with the
frozen Opus prompt, 200 filtered (admitted) questions per iteration.
- RZ-P1: the audited ill-posed share of admitted questions at iteration 3 >= iteration 1 + 10 points in >= 1 of the seeds run.
- RZ-P2: the final solver's UMWP CA (plain prompt, as HX) exceeds the base model's in every seed run.
If the official code cannot be run on this cluster within 3 days, this item is reported as not run with the reason.
