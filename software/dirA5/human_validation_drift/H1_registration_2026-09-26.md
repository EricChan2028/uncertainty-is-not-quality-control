# H1 registration: human read of the self-play drift (26 Sep 2026, ~23:10 CST; revised 27 Sep 2026, ~00:15 CST)

**Revision, made before the read.** Two independent reviews of the 23:08 version (sha256 9b98d197...) led to these changes,
all made before any join of the labels with the key:
- The account of file access is now exact.
- The file table lists the nine inputs that `human_read.py` checks, plus `annotators.md` and `hv_drift_packet.py` for the record.
- Full-word labels are accepted.
- U-vs-A disagreements without an adjudicated label have a pre-stated fallback.
- The packet text is checked against the key.
- The logistic trend no longer stops the run under separation.
- A second exploratory estimator (two-phase) is added.
- NOT RUN messages give counts only.

The statistics of the 23:08 version and their definitions are unchanged. On synthetic data the revised script reproduces
every number of the 23:08 version.

## Status when this plan was written

- This plan was written after the annotation was completed and before the labels were joined with the key. The four
  annotation files were placed in this folder on 26 Sep 2026 (22:36-22:37 CST). The labels have not been joined with
  `key.json`.
- Access to the label files and the key before the read, stated exactly:
  - The lead author received the four label files from the annotators and the adjudicator and handled them on their own
    machine. `labels_ann2.csv` was open in Excel there from 22:32:37 CST until between 23:38 and 00:01. That copy was not
    modified: its size, its write time (22:29:56) and its sha256 are unchanged and equal the registered hash. This document
    does not record what the lead author viewed. A filename search of the lead author's local project folder found no copy
    of the drift `key.json`, of `key_full.jsonl` or of the drift `packet.md` (the only `packet.md` there is the earlier A4
    packet, with a different hash). So the labels could not be joined with rounds from that folder.
  - The lead analysis session displayed each local label file's header line and its line count (`wc -l`) once, at
    22:35 CST, before the upload. It displayed nothing else from these files. The line count of `labels_adjudicated.csv`
    gives the number of adjudicated rows. If there is one row per disagreement, that is also the number of three-class
    disagreements. It carries no information on round, seed or judge. The ledger entry of 22:37:34 ("stored unopened")
    should be read with this qualification.
  - The lead analysis session opened `_key_do_not_open/key.json` once, at 22:14 CST. It printed only the file's type and
    length, the field names of its first entry, the counts of `round` and `seed`, and the names of its label fields. It
    printed no per-item value. The file's access time (06:54:50 CST, 26 Sep) comes from a recursive text search of `rsi_rl`
    at 06:54:19 for an unrelated workflow id, a string that cannot occur in its fields.
  - Apart from these two displays, no analysis session displayed any content of the label files or of `key.json`. The
    sessions that wrote and reviewed this plan and `human_read.py` ran the script and its test only on synthetic files
    under `/tmp`. All hashes below come from `sha256sum`, which displays no content.
- Annotation, as stated by the lead author in `annotators.md`:
  - Two external mathematics PhD students labelled independently, without LLM assistance, following `packet.md`. They were
    blind to round, seed and judge labels.
  - An external mathematics postdoc adjudicated the disagreements (`labels_adjudicated.csv`, with reasons in
    `adjudication_rationale.csv`).
  - `annotators.md` does not state whether the adjudicator was blind to round, seed and judge labels. It also does not say
    which files the annotators and the adjudicator received, or whether they had access to the server.
- Blinding by construction:
  - `packet.md` shows the 240 items in shuffled order under neutral ids D001..D240, with only the problem text and the S/U/A
    definitions. The shuffle used `random.Random(20260927)` in `dirA5/code/hv_drift_packet.py`. The packet contains no judge
    verdicts.
  - Within this folder, round, seed, the A7b item id and the two Opus verdicts exist only in `_key_do_not_open/key.json`.
    The open-weight verdicts were never in this folder.
  - Outside this folder, `results/a7b_blind/key_full.jsonl` holds the seed, round and problem text of all 900 items. Anyone
    with server access could therefore rebuild the round of each hid from `packet.md`.
- Sample: 240 of the 900 A7b blind re-audit items. These are Qwen3-4B-Base own-proposal self-play questions, seeds 0-2 x
  rounds 1/3/5 x 100. The sample has 80 per round, split 27/27/26 across seeds 0/1/2.

## Files (sha256)

| file (relative to `/home/tione/notebook/rsi_rl/dirA5/`) | sha256 |
|---|---|
| human_validation_drift/labels_ann1.csv | 0d6b3e2b24ba0a25d87f777df18733608823cea2460c46a84c06201da2a925ae |
| human_validation_drift/labels_ann2.csv | b98bf4160fc70bf93bd3f51f2a7cd9aa22a36f091e499eec1b9d740f43b48674 |
| human_validation_drift/labels_adjudicated.csv | cd4e9003cb671f17cc618578c95553bdf958d78840a5ef0af0a282912ce75dc9 |
| human_validation_drift/adjudication_rationale.csv | a87702b199f543ce1cb9f3e7b3b6045f5a3fc09195b4da4b95e76640794b72d4 |
| human_validation_drift/_key_do_not_open/key.json | 91d86caaf4978fc0e3740bb34b5ac7681efa94b87cc30cc1286ee74259e2af5f |
| human_validation_drift/packet.md | fe667ec21c5542d900a977dec09df69daeb80c7620a8c7aa5f912e0fab4b6f95 |
| results/a7b_blind/key_full.jsonl | c4f16b0b9488f55c60c2e2b11e5065dd2e3035e7a2f558f23efc1014ae774bff |
| results/a7b_blind/blind_verdicts.json | 67957b08064db94482612c4534629874489f99eae2fa99276847f74373210815 |
| results/a10/verdicts.jsonl | 574b222f5dbd160bbe9e0d91cc88cb84b75b20b2f553f95e6a782904fde68915 |
| human_validation_drift/labels_template.csv | 166091c53705515eab2fd9fe0987db5c0614bf5f0b65d1d82fa303ea947629ff |
| human_validation_drift/annotators.md | eaf5468cb14bf71ea61fb232c695756887cd153b8966ee2911731e109404088b |
| code/hv_drift_packet.py | 31bbb8330e578db272b706941c43689856ee4014799b35195d1d9de136edaa4c |
| human_validation_drift/human_read.py | 2c8ab779429efbbbb7669114c8e13f45112e8eb3e8681ff436a1e62f09de9411 |
| human_validation_drift/dev_human_read_test.py | 155af60a46dbe5926f861e235436cf06dadf480b859b3554e872dfcc64c52870 |

The first nine rows are built into `human_read.py`: the four label files, `key.json`, `packet.md`, `key_full.jsonl`,
`blind_verdicts.json` and `a10/verdicts.jsonl`. The script stops with NOT RUN (writes nothing) if any of them differs.
`labels_template.csv`, `annotators.md` and `hv_drift_packet.py` are listed for the record; the script does not read them.
The hashes of this file, `human_read.py` and `dev_human_read_test.py` are appended to `docs/registration_hashes.txt` before
the read. The binding version is the last ledger entry made before `human_read.json` is created.

## Open-weight judge and the join

The open-weight labels come from A10 (registered 01:13 25 Sep): Qwen2.5-32B-Instruct, greedy decoding, one item per request,
using the frozen audit definitions. They are in `results/a10/verdicts.jsonl`: 900 rows keyed by `key` (b0000..b0899), with
the registered parse in the field `verdict`. That `key` is the neutral A7b item id from `results/a7b_blind/items.jsonl`, the
same id that `hv_drift_packet.py` stored as `a7b_key` in `key.json`. The join is therefore direct (`key == a7b_key`). A10
reported 0 missing verdicts, and the file hash equals the one recorded in `results/a10/a10_read.json`. An item without a
valid verdict counts as not joined, and the number joined is reported. (`results/a12_open` is a different item set, with keys
`prop1:N`, and is not used.)

## Conventions

- Labels are normalised with `strip().upper()`. Annotator and adjudicated labels must be S, U or A, or the full words
  SOLVABLE, UNSOLVABLE or AMBIGUOUS, which map to S/U/A as for the judges. Any other value is invalid. Ill-posed = U or A.
  Judge labels SOLVABLE/UNSOLVABLE/AMBIGUOUS map to S/U/A. Shares and differences are in percentage points.
- Bootstrap: 4,000 replicates. Every replicate matrix is drawn from a fresh `random.Random(20260926 + 21)`. The interval is
  the percentile interval: sorted finite replicates at positions int(0.025 nv) and int(0.975 nv) - 1, with nv reported.
  - "Item" bootstrap: the 240 items resampled with replacement, in hid order.
  - "Stratified" bootstrap: items resampled within seed x round, with cells in (seed, round) order and items in hid order.
  - One item matrix and one stratified matrix are shared by all statistics, so differences between label sources are paired.
  - The exploratory 900-item resample uses `random.Random(20260926 + 22)`.
- Wilson 95% intervals use z = 1.959963984540054.

## Analysis (implemented exactly by `human_read.py`)

1. **Quality.**
   - Completeness: each annotator file must have exactly 240 data rows, D001..D240 once each (blank rows ignored), with valid
     labels. The numbers of invalid or empty labels are reported.
   - Inter-annotator agreement, each with an item-bootstrap 95% CI: raw agreement, Cohen's kappa for 3 classes, and
     Cohen's kappa for the binary split (U or A vs S).
   - The 3x3 confusion matrix.
   - Adjudication bookkeeping: rows used, extra rows on agreed items (ignored, counted), and agreement of the adjudicated
     label with each annotator.
   - A consistency check of `adjudication_rationale.csv` against the label files (hid set and columns; free text not
     coded).
   - Integrity of the key: the text under each hid in `packet.md` must equal the `key_full.jsonl` problem of that hid's
     `a7b_key`. All 900 problem texts are distinct, and a format-only parse of the real `packet.md` gives the 240 sections,
     each equal to one of them. The key's Opus labels are also compared with `key_full.jsonl` and `blind_verdicts.json`,
     with counts reported.
2. **Final label.**
   - Where the annotators agree (3-class), the final label is their common label. Otherwise it is the adjudicated label.
   - A disagreement between S and ill-posed (U or A) without an adjudicated label gives NOT RUN.
   - Fallback: suppose the only disagreements without an adjudicated label are U vs A. Those items then take the binary
     final label ill-posed. The primary and all binary statistics are computed as usual. The statistics that need a 3-class
     final label are reported as not available, with the number of such items. These are the final U/A counts, the 3-class
     judge kappas and confusion matrices, the U-only robustness line and the descriptive U/A counts.
3. **PRIMARY.** The human ill-posed share (final label U or A) at rounds 1, 3 and 5, pooled over seeds, each with a Wilson
   95% CI. The primary effect is D_h = share(r5) - share(r1), with a stratified-bootstrap 95% CI. Pre-stated reading:
   - **RISES** if the CI lower bound is above 0;
   - **FALLS** if the upper bound is below 0;
   - **FLAT** if the CI lies within [-10, +10] and contains 0;
   - **INCONCLUSIVE** otherwise.

   A lower bound of exactly 0.0 is not above 0.

   Secondary: D_h for r3 - r1 and D_h per seed (stratified CIs), plus a logistic-regression trend over round. The model is
   logit P(ill-posed) = b0 + b1 * round, with round in {1, 3, 5}, fitted at item level by Newton-Raphson ML. It reports b1
   with its Wald 95% CI and the odds ratio per round. The trend is reported as not estimable, and the rest of the run is
   unaffected, in two cases:
   - the labels are (quasi-)separated in round: every solvable item is at a round <= every ill-posed item's round, or the
     reverse, so the ML slope is infinite;
   - the fit fails numerically.
4. **SECONDARY (judge validity).** Each of three judges is compared with the final human label: opus_original and
   opus_blind (from key.json) and the open-weight judge (joined as above). For each judge:
   - n joined;
   - kappa for 3 classes and for binary, with item-bootstrap CIs;
   - accuracy (raw agreement);
   - the human-by-judge confusion matrix;
   - precision, recall and F1 of "ill-posed" against the humans, overall and per round;
   - the false-positive rate (judge ill-posed among human-S items) per round with Wilson CIs, and its r5 - r1 difference
     with a stratified CI;
   - the drift D measured by the judge on the same items, next to D_h on the same items, with the paired difference.
5. **Robustness.**
   - D_h from each annotator alone.
   - D_h with AMBIGUOUS counted as solvable (U only).
   - EXPLORATORY (not a test), as planned: a misclassification-corrected Opus blind drift on the 900 A7b items, using
     Rogan-Gladen per round.
     - p_obs_r is the blind-Opus ill-posed share among the round-r items of the 900.
     - Se_r and Sp_r are taken from the 80 round-r items of the 240, against the final human label.
     - pi_r = (p_obs_r + Sp_r - 1)/(Se_r + Sp_r - 1), clipped to [0, 1], and D = 100 (pi_5 - pi_1).
     - Each bootstrap replicate pairs a stratified resample of the 900 with one of the 240, treated as independent. This
       is an approximation, since the 240 are a subset of the 900.
     - A replicate with Se + Sp - 1 <= 0 is dropped.
   - EXPLORATORY (not a test), second estimator, added before the read: the two-phase (double-sampling) estimator. The 240
     are a stratified random subsample of the 900, 26-27 per 100 in every seed x round cell.
     - pi_r = PPV_r * p_obs_r + (1 - NPV_r) * (1 - p_obs_r).
     - PPV_r is the human ill-posed share among the sampled round-r items that blind Opus calls ill-posed. NPV_r is the
       human solvable share among those it calls solvable.
     - D = 100 (pi_5 - pi_1).
     - The estimate stays in [0, 1] and needs no J <= 0 rule.
     - Its CI uses one stratified resample of the 900 per replicate (`random.Random(20260948)`), carrying the human labels
       of the sampled items, so no independence approximation is needed.
     - A replicate with an undefined PPV or NPV in a round is dropped.
6. **Descriptive.** Annotator notes are free text and are not coded. Only the counts of final U and A per round are
   reported.

## Abort rules and output

NOT RUN (exit 2, nothing written) if any of these holds:
- an input file is missing;
- an input sha256 differs from the table above;
- an annotator file lacks a hid or label column, has the wrong row count, or has unknown, duplicate or missing hids;
- an annotator label is invalid or empty, including on an adjudicated item;
- an adjudicated row has an invalid label, an unknown hid or a conflicting duplicate;
- an S-vs-ill-posed disagreement has no adjudicated label (U-vs-A: see the fallback in item 2);
- key.json is not 240 valid entries with 80 per round, or its a7b_key/seed/round does not match key_full.jsonl;
- `packet.md` does not parse into D001..D240, or its text under a hid differs from the `key_full.jsonl` problem of that
  hid's `a7b_key`;
- `key_full.jsonl`, `blind_verdicts.json` or `a10/verdicts.jsonl` is malformed.

NOT RUN messages give counts only, never hids or label values. The real run is made by the lead author.

The output `human_validation_drift/human_read.json` is created once (exclusive create; exit 3 if it exists). It holds every
number with its n, the readings, the input hashes and the script hash, and the script prints a compact summary. The run is
deterministic: two runs on the same inputs give identical bytes. Only the D_h reading in item 3 is confirmatory. Items 3
(secondary), 4 and 5 are secondary or descriptive, and the exploratory lines are marked as such in the output. Any change
after the read goes into a separate, dated amendment file.

## Design properties and downside flagged before the read

- **Width of the primary CI.** With 80 items per round, the 95% CI of D_h is about +-15 points when the shares are near 50%.
  - FLAT is practically unreachable unless both shares are near 0 or 100%.
  - The replicate shares lie on a 1.25-point lattice, so a lower bound of exactly 0.0 is possible. It reads INCONCLUSIVE (or
    FLAT if the upper bound is at most 10), not RISES.
  - Synthetic data with the ill-posed items spread evenly over seeds give the smallest D_h that reads RISES:

    | round-1 share | smallest D_h that reads RISES |
    |---|---|
    | 45-50% | 17.5 points |
    | 37.5-40% | 16.25 points |
    | 30% | 15 points |
    | 10% | 12.5 points |

    A D_h of 15 near 50% reads INCONCLUSIVE.
  - For comparison, the judge drifts on all 900 items were +24.7 [17.3, 32.0] (A7b blind Opus) and +19.3 [11.7, 27.0] (A10
    open-weight). A human drift of that size would read RISES; one near 10 points would read INCONCLUSIVE.
- **Scope of the intervals.** The stratified bootstrap treats the three seeds as fixed strata.
  - The CIs therefore cover item-sampling uncertainty within these three runs, not seed-to-seed variability. The per-seed
    D_h is the only view of heterogeneity across seeds, and the result is stated in these terms.
  - The percentile bootstrap on 26-27-item cells uses the plug-in variance (factor (n-1)/n, about 0.96). The CIs are
    therefore about 2% too narrow, as in the A7b and A10 reads.
- **Humans may call fewer items ill-posed than Opus** (Opus blind: 68% ill-posed on the 900). The judge false-positive rates
  would then be high, and the drift magnitude would be judge-dependent.
- **Rising judge false-positive rate.** If the false-positive rate rises with round, part of the judge-measured drift is a
  judge effect. The per-round false-positive rates and their r5 - r1 difference measure this directly.
- **How each outcome is reported.**
  - RISES: the human labels confirm the direction of the drift.
  - INCONCLUSIVE: the drift is reported as judge-measured, with the human estimate and its CI.
  - FALLS: reported as a contradiction of the judge drift.

## Test

`dev_human_read_test.py` builds synthetic files in `/tmp/hr_claude_h1/`: labels, key, packet, A7b and open-judge files. It
checks the output against hand-computed values and against an independent pure-Python recomputation of the bootstrap
intervals, including both exploratory estimators. It covers kappa (closed form), Wilson bounds, the logistic slope and SE,
designed false-positive rates and drifts, and bootstrap reproducibility. It checks the runs that must succeed:
- full-word labels;
- U-vs-A disagreements without adjudication;
- five (quasi-)separated round patterns.

It checks 13 abort paths, confirms that no NOT RUN message contains a hid, and checks the overwrite refusal. It also
unit-tests the reading rule at its boundaries and the logistic fit, then deletes the folder.

Result at this revision: 254 checks, 0 failures. The 23:08 version detected four planted bugs: a wrong bootstrap seed, the
wrong final-label rule, the wrong bootstrap type for kappa, and a rounded z. Eight more were each detected at this revision:
- no separation guard;
- no word mapping;
- no U-vs-A fallback;
- PPV and NPV swapped;
- the packet check disabled;
- the item bootstrap in place of the stratified one in Rogan-Gladen;
- a hid in a NOT RUN message;
- a wrong 900-item seed.

On the base synthetic data set the revised script reproduces every number of the 23:08 version; only added or renamed
fields differ.
