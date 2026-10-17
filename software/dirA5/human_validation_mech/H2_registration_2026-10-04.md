# H2 registration: human check of the LLM mechanism labels (2026-10-04, before any human label exists)

## 1. Purpose

Two statements of the paper rest on LLM labels from the A8 readings annotation (`dirA5/results/a8/`, labeller prompt
`dirA5/code/a8_label_prompt.txt`, read `dirA5/code/a8_read.py`):

- (a) at base, unanswerable UMWP questions (U) disagree over readings (label READINGS or MIXED) in 94% of units, against 26% for
  hard answerable questions (A), which disagree through computation errors (section 4.4; difference +0.68);
- (b) on unanswerable questions, the share of solutions that state an assumption or flag a gap (gap_flag) rises from about 15%
  at base to 49-70% at round five (section 5, Figure 3).

Two reviewers asked for a human check of these labels. H2 collects human labels for a random sample of the A8 items and
compares them with the LLM labels on the same items. Like H1 and the four further analyses of the paper, H2 is registered with
a pre-stated interpretation for each outcome rather than as pass/fail predictions, and it is not counted among the 136
predictions.

## 2. Materials (built by `h2_packet.py`, `random.Random(20261005)`, before this registration)

- **Part A**: 40 base-model units, 20 U and 20 A, drawn at random within class from the 112 + 112 A8 base units. Each unit shows
  the problem and its 8 solutions, exactly the texts the A8 labellers saw (each at most its last 1,800 characters), with the
  automatically parsed final answer under each solution; the annotators are told that the parsed answer may be wrong and that
  their own reading of the final answer prevails.
- **Part B**: 120 single solutions: 48 U and 12 A questions not used in part A; for each, one base-model solution and one
  round-5 solution of the full-curriculum lineages (seeds 0, 1, 2 in rotation: 16 U and 4 A questions per seed); the sample
  index is drawn uniformly from 0-7. The A questions are included so that not every item comes from an unanswerable question.
- Items are shuffled under neutral ids (A01-A40, B001-B120) in `packet_A.html` and `packet_B.html` (instructions in English,
  which is authoritative, with a Chinese translation), with `labels_A_template.csv`, `labels_B_template.csv` and
  `README_annotators.md`. The key is in `_key_do_not_open/key.json` and is not given to the annotators.
- LLM labels on these items, computed at registration from `labels_main.json`: part A, READINGS-or-MIXED share 0.90 for U (18 of
  20) and 0.20 for A (4 of 20), difference 0.70 (all 224 base units: 0.94, 0.26, 0.68); part B, gap_flag rate on U 0.208 at base
  and 0.646 at round five (rise 0.438), on A 0 and 0.

## 3. Annotators and procedure

- The lead author asks the two external annotators and the adjudicator of H1 (PhD students in mathematics, fourth and third
  year; a postdoctoral mathematician). If one of them is unavailable, an external annotator with at least a master's degree in
  mathematics or a related field replaces them; the replacement is documented before the labels are joined with the key.
- The annotators work independently, without AI assistance and without discussing the items, and are blind to class, model,
  round, seed and the LLM labels. They receive only the two HTML packets, the two templates and `README_annotators.md`.
- The return is requested by 2026-10-08 20:00 CST. **Inclusion cut-off**: label files received by 2026-10-10 12:00 CST; later
  files are not used in the submitted paper. On receipt, each file is saved as `labels_A_ann1.csv`, `labels_A_ann2.csv`,
  `labels_B_ann1.csv` and `labels_B_ann2.csv` (ann1 and ann2 fixed when the packets are sent) and its SHA-256 is appended to
  `docs/registration_hashes.txt` before `h2_read.py` runs.
- **Adjudication** (secondary, optional): if the adjudicator is available before the cut-off, they see only the items on which
  the two annotators disagree (part A: R or M against N or E; part B: flag or answer), each with the two labels and nothing
  else, and record their label in `labels_A_adjudicated.csv` or `labels_B_adjudicated.csv`.

## 4. Labels (the packets contain the full instructions)

- **Part A** (per unit): N / R / E / M, the human counterpart of NONE / READINGS / ERRORS / MIXED in Steps 1, 2 and 4 of the A8
  prompt: the most common final answer (ties: the answer whose reading more solutions share); a departure is a solution with a
  different final answer; a reading departure resolves some element of the problem differently in a way the wording admits; an
  error departure shares the reading or has no determinable reading. Optional: the indices of the reading departures.
- **Part B** (per solution): flag Y/N (Step 3 of the A8 prompt, gap_flag: the text explicitly says that information is missing,
  ambiguous, contradictory or unrealistic, or explicitly states an assumption it had to make) and answer NUM / DECLINE / NONE.

## 5. Analysis (`h2_read.py`; tested only on simulated labels with `h2_dev_test.py`)

- Normalisation and missing labels as in the script's docstring. An annotator with valid labels for fewer than 90% of a part's
  items is left out of that part's primary estimates and reported descriptively.
- Each item's human value is the mean of the included annotators' binary labels (0, 0.5 or 1).
- **H2a**: D_h = share(R or M | U) - share(R or M | A) over the 40 part-A units, with a 95% CI from an item bootstrap within class.
- **H2b**: on the 48 U questions of part B, the flag rate of the round-5 solution minus that of the base solution, with a 95% CI
  from a bootstrap over questions.
- **H2c**: Cohen's kappa of each annotator with the LLM label (part A: R-or-M binary and four-way; part B: flag) and between the
  two annotators, with item-bootstrap 95% CIs.
- All bootstraps use 4,000 replicates and `random.Random(20261005)`; the same draws give the LLM value on the same items and the
  human-minus-LLM difference.
- Secondary: per-annotator estimates; consensus labels (agreeing annotators, else the adjudicated label); the Figure-3
  categories (flagged answer, silent answer, no numeric answer) at base and round five, with the LLM analogue on the same
  solutions; the 12 answerable pairs.

## 6. Pre-stated interpretations and their consequences for the paper

**H2a (readings contrast).**
- CONFIRMED (lower end of the 95% CI above 0.20): section 4.4 reports the human contrast next to the LLM contrast, and the
  sentence relating readings to the class rates stays as worded (consistent with the rates, not causal).
- DIRECTION_CONFIRMED_WEAKER (lower end in (0, 0.20]): section 4.4 reports both and describes the LLM contrast as possibly
  overstated.
- NOT_CONFIRMED (lower end at or below 0): section 4.4 states that human annotation does not confirm the readings contrast; the
  sentence relating readings to the class rates is removed from the main text, and statements elsewhere that the band selects
  problems "whose text supports several readings" are rephrased to rest on the fixed-pool exit alone.

**H2b (rise in stated assumptions or flags).**
- CONFIRMED (lower end above 0.10): section 5 and the Figure 3 caption report the human rise next to the LLM rise.
- DIRECTION_CONFIRMED_WEAKER (lower end in (0, 0.10]): both are reported, the LLM rise described as possibly overstated.
- NOT_CONFIRMED (lower end at or below 0): section 5 states that the rise in stated assumptions is not confirmed by human
  labels, and statements that sharpening turns readings into "one stated assumption" are rephrased in terms of answer
  concentration (exact-match statistics), which do not use these labels.

**H2c (agreement).** Kappa bands: above 0.60 substantial, 0.41-0.60 moderate, 0.21-0.40 fair, at most 0.20 slight. If an
annotator's kappa with the LLM label is fair or slight for a part, the paper says that the LLM labels of that part agree only
weakly with human labels, whatever H2a or H2b show.

Whatever the outcome, H2 is reported in the appendix (readings annotation) with every estimate above, and the Limitations
sentence on LLM-only labels is updated accordingly.

## 7. Deviations

Any departure from this document is listed among the deviations in the paper's pre-registration appendix.

## 8. Registered files

`H2_registration_2026-10-04.md` (this file), `h2_packet.py`, `h2_read.py`, `h2_dev_test.py`, `README_annotators.md`,
`packet_A.html`, `packet_B.html`, `labels_A_template.csv`, `labels_B_template.csv`, `_key_do_not_open/key.json`, all in
`dirA5/human_validation_mech/`; their SHA-256 values are appended to `docs/registration_hashes.txt` at registration.
