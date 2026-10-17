# Supplementary data

Anonymous submission: *Uncertainty Is Not Quality Control: Label-Free Self-Play Drifts Toward Ill-Posed Problems*.

This archive contains the labelled self-play curricula, label tables that give the source of every label, the outputs
of the behavioural probes, the official R-Zero runs and the outputs of every registered read. The code is in
`software.zip`. `release_v1/` is the release of the audited curricula built on 24 September 2026; everything added since
then is in the other folders (summary in `DATASHEET_addendum.md`).

## Layout

| path | contents |
|---|---|
| `release_v1/` | admitted problems per lineage and round with the original Opus labels and the solver's agreement at admission (`audited_pools/`), exhaustively labelled fixed pools with every checkpoint's agreement (`fixed_pool_audits/`), the judge prompts (`judge/`), the planted premise-deletion twins as patches (`twins/`), the judge-free UMWP probe (`probe/`); see `release_v1/README.md` and `release_v1/DATASHEET.md` |
| `labels/` | one table per labelled set; every label carries its source (next section) |
| `raw/` | the registered files those tables are built from, at their repository paths: blind re-judging, open-weight judge and verifier, human annotation, R-Zero audits, readings annotation, fixed-pool guard records |
| `probes/umwp/` | UMWP probe outputs per checkpoint, one row per question with the eight parsed answers, the modal answer and its share: `plain_*` (plain prompt), `abs_*` (original abstention prompt), `absfix_*` (corrected abstention prompt; A11 and A13) |
| `probes/external_sets/` | SUM, MiP-GSM8K and MiP-MATH probe outputs (A6 and A11), same row format |
| `probes/rev6/` | the re-measurement with saved texts (paper, Appendix G): `dumps_original/` and `dumps_strict/` hold every rev6 run in the row format above, with the eight answers read by the probes' first parser and by the strict reading of Section 2 (a solution that declines in words, answers in a variable or gives no final number is a non-answer; otherwise the same number); `texts/` holds the full solutions of the main Qwen3-4B checkpoints (base, full curriculum rounds 3 and 5, solvable-only round 3); the other texts will be released with the camera-ready version (the full set exceeds the size limit) |
| `runs/rzero/` | official R-Zero runs (two seeds): the questions admitted at each solver iteration, with summaries |
| `runs/a13/` | the planted-free contrast: the five seeds' curricula and benchmark evaluations |
| `reads/` | outputs of the registered reads (A3-A13, R-Zero, human annotation), benchmark-accuracy summaries (`reads/dirB/results/evals/`) and the outputs of the exploratory analyses, at their repository paths |
| `MANIFEST.tsv` | SHA-256 and size of every file |
| `PATHMAP.tsv`, `REDACTIONS.tsv` | which files the registration log covers, and every anonymising replacement (copies of the files in `software.zip`) |

To rerun a read from `software.zip`, copy `raw/` and `reads/` into the project root so that the repository paths are
restored (for example `raw/dirA5/results/a7b_blind/` becomes `dirA5/results/a7b_blind/`).

## Label sources

| source | where | coverage |
|---|---|---|
| Claude Opus, original audit (the judge could see the round) | `release_v1/audited_pools/`, `release_v1/fixed_pool_audits/`, field `opus_original` in `labels/` | 200 admitted problems per lineage and round, or all of them; the two fixed pools in full; R-Zero, 200 admitted questions per seed and iteration |
| Claude Opus, blind re-judging (neutral ids, round hidden, shared framing) | `labels/*` field `opus_blind*` | 900 Qwen3-4B problems (100 per seed at rounds 1, 3, 5); all 1,200 labelled R-Zero questions; 1,200 raw R-Zero challenger questions (by pass) |
| Qwen2.5-32B-Instruct, same prompt | `labels/qwen3_4b_900_labels.jsonl`, field `open_weight` | the same 900 problems |
| Qwen2.5-32B-Instruct as a verifier on top of the band | `labels/fixed_pool_verifier_labels.jsonl` | the fixed pool `qwen_s0_r1` |
| two external mathematicians and an adjudicator (blind to round, seed and judge labels) | `labels/qwen3_4b_900_labels.jsonl`, fields `human_*` | 240 of the 900 problems (80 per round) |
| two external annotators, H2 check of the mechanism labels (blind to model, round and LLM labels; two differently ordered versions) | `raw/dirA5/human_validation_mech/v2/` (labels without notes, packets, `key_v2.json`), read in `reads/dirA5/human_validation_mech/v2/h21_read.json` | 184 base units of eight solutions (source of disagreement) and 600 single solutions (gap flag, answer status) |

Labels are SOLVABLE, UNSOLVABLE or AMBIGUOUS. A problem is ill-posed if it is UNSOLVABLE or AMBIGUOUS. Rows of different
tables can be joined on `problem_sha256`.

Model-written labels are not ground truth: on problems the human annotators call solvable, the judges' false-positive
rate rises in later rounds (paper, Appendix B). Do not use these labels as answerability ground truth without human
review.

## Withheld during review

- The free-text notes of the human annotators (labels are complete); they will be released with the camera-ready version.
- The problem texts of the planted GSM-Symbolic tracers (CC BY-NC-ND 4.0 does not allow sharing adapted material):
  `release_v1/twins/rebuild_twins.py` rebuilds them from a copy of GSM-Symbolic and checks each text's SHA-256.

## Licences (proposed)

Annotations, model-written problems and outputs: CC BY 4.0. UMWP-derived files (question ids in `probes/umwp/`, question
texts in `raw/dirA5/results/a8*/`): CC BY-SA 4.0, as UMWP. Llama-generated pools: Llama 3.2 Community License ("Built
with Llama"). Third-party benchmarks are referenced by id only.

## Parser note

The files in `probes/umwp/` and `probes/external_sets/` were scored by the probes' first parser, which reads the last number of a solution and so counts many verbal declines ("cannot be determined", "Answer: N/A") as answers. The paper's UMWP and external-set numbers come from `probes/rev6/` under the strict reading; the registered reads keep the first parser.
