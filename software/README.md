# Supplementary software

Anonymous submission: *Uncertainty Is Not Quality Control: Label-Free Self-Play Drifts Toward Ill-Posed Problems*.

This archive contains the code, prompts and pre-registration records behind the paper. The companion archive
`data.zip` contains the labelled curricula, the label tables (with the source of every label), the probe outputs and
the outputs of every registered read. Both archives are anonymised for review (last section).

Files keep their paths in the project repository, so that paths in the registration log and in the scripts resolve
directly. Each script starts with a docstring that names its registration and inputs.

## What is where

| path | contents |
|---|---|
| `dirA3/code/` | the self-play loop: proposer (`a3_propose.py`, with the 40 topic prompts and the instruction template), admission band and planted tracers (`a3_guard.py`), round drivers (`a3_round*.sh`, `a4_round*.sh`), co-evolving proposer (`a4_proposer_sft.py`), UMWP probe (`umwp_probe.py`, `umwp_read*.py`), audits and the A3/A4 reads |
| `dirB/code/` | GRPO trainer with the majority-vote reward (`b_train.py`) and benchmark evaluation (`b_eval.py`), shared with an earlier study |
| `dirA/code/` | premise-deleted probe (`c0_probe.py`) |
| `dirA4/code/` | patches that turn the A3 loop into the A4 variants |
| `dirA5/code/` | studies A5-A10: judge-free statistic (`a5_j1_*`), frontier (`a5_frontier*`), controls (`a5_postfilter.py`), clean and matched curricula (`a6_*`), exits and taxonomy (`a7_*`), blind re-judging (`a7b_*`), readings annotation (`a8_*`), sharpening model in other families (`a9_*`), open-weight judge (`a10_*`), blind re-audit of R-Zero (`a75d_*`), human-annotation packet (`hv_drift_packet.py`) |
| `dirA5/human_validation_drift/` | human-annotation read (`human_read.py`) and its tests |
| `ext/rzero_design/` | official R-Zero replication: our patch to commit 5699329 (`rzero_A7_5.patch`), environment, launchers, audit workflow, read (`rz_read.py`) |
| `ext/a11_draft/` | probes with the corrected abstention prompt and their read (A11) |
| `ext/a12_draft/` | blind audit of R-Zero's raw challenger output, R-Zero accuracy, open-weight verifier (A12) |
| `ext/a13_draft/` | planted-free same-size contrast (A13) |
| `ext/a75d_draft/` | blind re-audit of R-Zero's admitted questions (A7-5d) |
| `paper_arr/figures/` | figure scripts |
| `paper_arr/analyses/` | exploratory analyses marked with a dagger in the paper, including the robustness checks in `rev3/` (exact modal-count refit of the sharpening model; stricter self-consistency thresholds) |
| `host_scripts/` | helper scripts that lived on a host's local disk and were registered from there |
| `*/docs/`, `ext/*_draft/*.md`, `dirA5/human_validation_drift/*.md` | registration documents, amendments and results documents |
| `docs/registration_hashes.txt` | the registration log |
| `preregistration/` | `predictions.pdf` (all 136 predictions with thresholds and outcomes), `PREREGISTRATION_INDEX.md`, `PATHMAP.tsv`, `REDACTIONS.tsv` |
| `prompts/` | the judge prompts and the readings-labeller prompt in one place (the proposer and probe prompts are in the scripts named above) |
| `verify_hashes.py` | recomputes the SHA-256 of every packaged file that the log registers |

## Paper sections and their main scripts

| paper | scripts (outputs in `data.zip`) |
|---|---|
| §2 loop and probes | `dirA3/code/a3_propose.py`, `a3_guard.py`, `a3_round.sh`; `dirB/code/b_train.py`, `b_eval.py`; `dirA3/code/umwp_probe.py`; `dirA/code/c0_probe.py`; judge prompts in `prompts/judge/` |
| §3 Table 1, drift | `dirA3/code/a3_analyze.py`, `a4_analyze.py`, `a4_reads.py`; blind rows `dirA5/code/a7b_*`; open-weight judge `dirA5/code/a10_*`; human annotation `dirA5/code/hv_drift_packet.py`, `dirA5/human_validation_drift/human_read.py`; judge-free statistic `dirA5/code/a5_j1_*` |
| §3 R-Zero | `ext/rzero_design/` (patch, launchers, `rz_read.py`); blind re-audit `ext/a75d_draft/`; raw challenger `ext/a12_draft/a12_p1_*`; accuracy `a12_p2_read.py`, `a12_plain_eval.py`, `a12_rzchat_eval.py` |
| §4 fixed pool, odds, sharpening model | `dirA3/code/a3_guard_diag_score.py`, `a4_nt_analyses.py`, `a4_law_stable.py`, `a4_law_bootstrap.py`; robustness `paper_arr/analyses/rev3/law_exact_modal.py`; other families `dirA5/code/a9_read.py`; model-free check `dirA5/code/law_modelfree_qwen4b.py` |
| §4 hard vs ill-posed, exits, readings | `dirA5/code/a6_matched.py`, `a7_exit.py`, `a8_*` |
| §5 behaviour, attribution | `dirA3/code/umwp_read*.py`, `dirA5/code/a6_*`, `ext/a13_draft/`, `paper_arr/analyses/attribution.py`, `convergence.py`, `rev3/ca_thresholds.py`, `dirA5/code/a8_silent.py` |
| §5-6 abstention | `ext/a11_draft/` (corrected abstention prompt), `dirA5/code/a5_frontier*.py` |
| §6 filters and remedies | `dirA3/code/a4_signal_search.py`, `a4_program_closure.py`, `a4_nt_analyses.py`, `a4_rescue*`; verifier `ext/a12_draft/a12_open_judge.py`, `a12_p3_read.py` |
| figures | `paper_arr/figures/fig*.py` |

## Pre-registration records

- `docs/registration_hashes.txt` is the append-only log we kept: for each study, the SHA-256 of its registration text and
  analysis scripts, written before the data were generated, and later of its inputs and outputs, with local timestamps. It
  is not timestamped by a third party. It lets a reader check that the packaged documents and scripts are the ones whose
  hashes were logged before the corresponding results appear in the log. The log also covers an earlier, unrelated study
  of the same group (entries before 21 September 2026; paths under `code/`, `runs/`, `results_notes/`, `paper_loto/`)
  and two exploratory directions not reported in the paper (`dirB/` results, `dirC/`); those files are not included.
- `preregistration/PATHMAP.tsv` lists every registered file in the two archives: its location, its last registered hash,
  the hash of the packaged copy and a status. *Verbatim* files match their last (or an earlier) registration;
  *anonymised copies* differ by the replacements listed in `REDACTIONS.tsv`. Nine shell launchers were edited after
  registration. Six were edited on 24 September to claim GPUs atomically and to skip evaluations that had already
  finished; their registered versions are included in `dirA5/code/registered_versions/` and match the registered hashes,
  and the training and probe commands are the same in both versions. For the other three (`dirA3/code/a3_round.sh`,
  `dirA5/code/a6_sweep_launch.sh`, `dirB/code/b_runner.sh`) no copy of the registered version survives.
- `python3 verify_hashes.py --data ../data` recomputes all hashes after unpacking both archives side by side.
- `preregistration/predictions.pdf` lists all 136 confirmatory predictions with their thresholds and outcomes (S1-S136),
  the deviations and the topic prompts. `dirA5/docs/prereg_ledger_2026-09-25.md` maps the registration clauses of A3-A10
  to these outcomes; A11-A13 and the human annotation (H1) are read by their own registered scripts, whose outputs are in
  `data.zip` (`reads/`, `raw/`).

## Running the code

- Python 3.10 with torch 2.8, vLLM 0.11 and transformers 4.57; numpy and scipy for the reads. The R-Zero replication runs
  the official code at commit 5699329 with `ext/rzero_design/rzero_A7_5.patch` in the environment of
  `ext/rzero_design/rz_env.sh` (newer library versions than the release pins; see the paper's appendix).
- Models: Qwen3-4B/8B/14B-Base, Phi-4-mini-instruct, Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct (stronger proposer) and
  Qwen2.5-32B-Instruct (open-weight judge and verifier).
- Scripts refer to the project root `/home/tione/notebook/rsi_rl`; we kept this path so that the registered hashes
  verify. To rerun a read, create that root (a symbolic link suffices), unpack `data.zip` into it so that `raw/` and
  `reads/` restore the repository paths (see `data/README.md`), or edit the root constant at the top of the script.
- The LLM judges (Claude Opus) ran as batches of agents through a workflow runner. The workflow scripts (`*_wf.js`)
  document the prompts, batching and blindness checks; rerunning them needs an equivalent runner and API access.
- Compute: about 1,400 GPU-hours on 72 GB cards.

## Anonymisation

- The server address, ssh ports, host numbers, cloud node names and local user paths are replaced with `<SERVER>`,
  `<PORT>`, `<HOST>`, `<NODEk>`, `<LOCAL_HOME>` and `<PROJECT>`. Exception: the blindness-check scripts
  (`*_blind_check*.py`) contain a regular expression for the temporary directory of the machine that ran the judges,
  whose account name is a hardware brand; we kept these scripts verbatim so that their registered hashes verify.
- A few sentences of internal scheduling and approval notes in registration documents, which are not part of any
  prediction, are replaced with `[internal note removed]`.
- The free-text notes of the human annotators are withheld during review; their labels are complete.
- Problem texts of the planted GSM-Symbolic tracers are not redistributed (licence); `data.zip` rebuilds them.

`preregistration/REDACTIONS.tsv` lists every replacement (file, line, category). Anonymised copies no longer match
their registered hash; `PATHMAP.tsv` gives the original hash. Code: MIT licence (proposed).

## Additions of 6 October 2026

- `dirA5/human_validation_mech/`: the H2 check of the mechanism labels (registration, amendment, packet builders `h2_packet.py` / `h21_packet.py`, reads `h2_read.py` / `h21_read.py`, simulated tests). The first packet's label files were set aside unread; the read uses the second packet only.
- `paper_arr/analyses/rev5/strict_parse.py`: the check of the probes' first parser against the annotators' answer status.
- `paper_arr/rev6/`: the re-measurement with saved texts. `rev6_plan_2026-10-06.md` (plan), `rev6_read.py` (the main read; `--check` reproduces the paper's earlier numbers on the original dumps), `rev6_reruns.py`, `rev6_reruns_extra.py` and `rev6_reruns_a6.py` (the registered and exploratory UMWP and external-set reads re-executed on the new runs under both parsers, from copies that change only their input directory and output path; the copies are in `reruns_original/` and `reruns_strict/`), the text-saving probe copies (`umwp_probe_text_absfix.py`, `a6_probe_text.py`), launchers and job lists, and the figure inputs (`fig4_categories.py`, `a8_strict_v2.py`, `a8_exact.py`). The registration log lists every hash and the order of the steps, including one amendment (the strict reading keeps the first parser's number) made before any new output was aggregated.
