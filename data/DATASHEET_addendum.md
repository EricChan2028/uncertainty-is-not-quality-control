# Datasheet addendum (release v2)

This addendum extends `release_v1/DATASHEET.md` (24 September 2026) to the material added for the paper after that date.
Section numbers follow Gebru et al., "Datasheets for Datasets" (CACM 2021). Sizes and hashes of all files are in
`MANIFEST.tsv`.

## 1. Motivation

The additions test whether the drift measured by the original audits depends on the judge (blind re-judging, an
open-weight judge, human annotation), extend it to the official R-Zero code (admitted and raw challenger questions), and
record the behavioural probes, the attribution contrasts and the readings annotation on which the paper's analyses rely.

## 2. Composition

- `labels/qwen3_4b_900_labels.jsonl`: 900 admitted Qwen3-4B problems (seeds 0-2; rounds 1, 3, 5; 100 per seed and
  round) with four or seven labels each: original Opus audit, blind Opus re-judging, Qwen2.5-32B-Instruct, and for 240
  problems (80 per round) two annotators and the adjudicated label.
- `labels/rzero_admitted_labels.jsonl`: 1,200 questions admitted by the official R-Zero code (two seeds, iterations 1-3,
  200 per seed and iteration) with the original and the blind Opus label.
- `labels/rzero_raw_challenger_labels.jsonl`: R-Zero challenger questions sampled before pseudo-labelling and filtering,
  with blind Opus labels by judging pass (later passes re-judge batches whose judges failed the blindness check; the
  registered read defines which pass counts).
- `labels/fixed_pool_verifier_labels.jsonl`: the fixed pool `qwen_s0_r1` (1,158 problems) with the original Opus label
  and the verdict of Qwen2.5-32B-Instruct used as a verifier.
- `probes/`, `runs/`, `reads/`, `raw/`: see `README.md`. Probe rows contain question ids, the eight parsed answers and
  their summary statistics, not question texts.

## 3. Collection process

The judges labelled problems under the prompts in `software.zip/prompts/judge/`. Blind re-judging showed each judge a
batch of problems under neutral ids and a framing sentence shared by every problem; a check of the judges' tool calls
confirmed that no judge read a file that could reveal an item's origin, and batches that failed were re-judged (paper,
Appendix C, deviation 12). The human annotators labelled a packet of 240 problems (`raw/dirA5/human_validation_drift/
packet.md`) independently and blind to round, seed and judge labels; an adjudicator resolved their disagreements. The
annotators were external mathematicians (two PhD students and a postdoctoral researcher), were paid USD 500 each for
about five hours of work, and consented to the use and anonymised release of their labels.

## 4. Preprocessing

None beyond the anonymisation listed in `REDACTIONS.tsv` (infrastructure details; internal notes in registration
documents; annotators' free-text notes withheld during review) and the withholding of GSM-Symbolic-derived texts.

## 5. Uses

Intended: auditing label-free self-play curricula; studying how admission rules select ill-posed problems; evaluating
well-posedness detectors against several label sources. Not intended: training or evaluating models as if the judge
labels were ground truth for answerability (see `README.md`).

## 6. Distribution and 7. Maintenance

As in `release_v1/DATASHEET.md`: anonymous during review, public with the paper; errata and new versions get a new
release directory and manifest.
