# A8 registration: interpretive multiplicity of in-band ill-posed questions (23:16 CST 24 Sep 2026, before any of its data)

Motivation [internal note removed]: the band selects "hard or ill-posed" questions (A6 D1 holds, D2 fails). This item
asks whether the disagreement that keeps an unanswerable question in the band is of a different, language-level kind -
several internally consistent READINGS (a value assumed for a missing premise, a referent chosen for an ambiguous phrase, a
constraint dropped from a contradiction, a target chosen for an underdetermined question) - rather than scattered computation
errors, and whether self-play collapses the readings and suppresses explicit gap-flagging.

Items (fixed before generation): from the Qwen3-4B-Base UMWP plain-prompt dump (plain_qwen3_4b_base, G = 8), base bins
k = round(8 m) in 2..6 (the D1/D2 bins). Hard answerable = answerable with reference and base gold accuracy <= 3/8 (the D2
subset), all of them in bins 2..6; unanswerable = the same number per bin drawn with random.Random(20260925) from the
unanswerable items of that bin (all of the bin if fewer). About 110 + 110 items. The same items are used for every model.

Generation: base, own_a3 r3 and r5 of Qwen3-4B seeds 0-2 (7 models); the plain prompt and sampling of umwp_probe.py exactly
(T = 1.0, top_p 1.0, max_new 1024, G = 8, per-request seed 17), with the full completion texts saved
(dirA3/code/umwp_probe_text.py = umwp_probe.py plus --ids and text saving; answers are extracted by the same function, and
their agreement with the existing dumps is reported as a reproducibility check).

Labelling (blind): each (item, model) unit gets a neutral key; units of base and r5 (4 models, ~880 units) are shuffled and
labelled by Opus agents (5 units per agent) who see the problem and its 8 solutions (each truncated to its last 1,800
characters) but not the model, round, seed or answerability. Per unit: reading clusters (groups of sample indices that adopt
the same interpretation / assumption, with a one-line description); per sample: gap_flag (the solution explicitly notes missing,
ambiguous or contradictory information, or states an assumption it had to make); per unit: disagreement_source in {NONE (all
final answers equal), READINGS (non-modal samples follow different consistent readings), ERRORS (same reading, different
computation), MIXED}. A 25% random subsample of base units (Random(20260926)) is labelled a second time by independent agents
(agreement reported). r3 units are labelled only if time permits (descriptive).

Predictions (confirmatory; read with a reviewed script hashed before its run; ill-posed side = UMWP unanswerable):
- M-P1 (disagreement kind): at base, the share of units with disagreement_source in {READINGS, MIXED} is >= 20 points higher
  for unanswerable than for hard answerable items, and the item-bootstrap 95% CI of the difference (4,000 replicates, seed
  20260925) excludes 0.
- M-P2 (gap-flagging falls): the gap_flag rate over the samples of unanswerable units is lower at r5 than at base by >= 5
  points in >= 2 of 3 seeds.
- M-P3 (readings collapse): the mean number of distinct reading clusters on unanswerable units is lower at r5 than at base in
  >= 2 of 3 seeds.
Descriptive: the same quantities for hard answerable units, per UMWP category, r3 (if labelled), second-pass agreement.
Downside flagged before data: M-P1 may fail on contradiction items (samples may err rather than choose a constraint), and M-P2/
M-P3 may fail if self-play leaves readings unchanged; each outcome is reported as registered.
