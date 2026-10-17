# probe/

Judge-free probe (A5 J1) and the UMWP / twin dumps it reads.

- `reproduce_j1.py`: recomputes J-P2 (UMWP) and J-P3 (premise-deletion twins) from `umwp_dumps/`, `twin_dumps/` and
  `../twins/probe543_validation.jsonl` alone and compares the result with `results/j1_judge_free.json`
  (`cd probe && python3 reproduce_j1.py`; J-P3 by exact hypergeometric subsampling, `--mc` for the original Monte Carlo).
- `code/`: copies of the project scripts: `umwp_probe.py` (wrote the UMWP dumps), `c0_probe.py` (the twin dumps),
  `a5_j1_judge_free.py` (the original J-P2 / J-P3 read), `a5_j1_umwp400.py` (the UMWP-400 judge validation),
  `a3_propose.py` / `a3_guard.py` (proposer and admission guard), `loto_common.py` / `pdc_util.py` (helpers). The only
  edit to these otherwise verbatim copies replaces the absolute project root by
  `__import__("os").environ.get("RSI_REPO", "<REPO>")` (in a string literal) or `<REPO>` (elsewhere); `code/SOURCES.tsv`
  gives each source file's SHA-256 and the number of replacements (the same holds for `../judge/code/`). The scripts
  expect the project tree (`RSI_REPO`), vLLM and GPUs; `reproduce_j1.py` needs neither. `results/` holds the original
  read-outs, with the absolute project root replaced by `<REPO>`.
- `umwp_subset1200.jsonl`: the 1,200-item UMWP subset (600 unanswerable questions, a Random(0) sample of those whose
  answerable originals exist, and their 600 answerable originals): UMWP ids, labels and pairing, no question text.
- `umwp_dumps/<prompt>_<checkpoint>.jsonl`: our G = 8 samples per UMWP item (`umwp_id`, `answerable`, `mode`, `p_mode`,
  `nonanswer`, `abstain`, `correct`, `answers`); `umwp_dump_summary.tsv` summarises each dump.
- `twin_dumps/<checkpoint>.jsonl`: our G = 32 samples per twin and form (`fam`, `task`, `form`, ..., `answers`; no text).

UMWP (Sun et al., LREC-COLING 2024) is third-party data: https://github.com/Yuki-Asuuna/UMWP, file `data/StandardDataset.jsonl`, retrieved 2026-09-24 05:30 CST from branch
main (head commit 1b3ae0fae01a6c6f178361c716bbb4f7d7d54c3c, 2023-10-21; the file was last changed in commit f5eb429a4a0761fe151d55ec4ef7ff10f5434953;
git blob 690944cc038ca45924181f30c1a28076b09eee12; SHA-256 e8840e8383357238a08e9c5028e4758ceceb369e1db31c64678b0f851c9c9e73; 5200 rows). Licence: data CC BY-SA 4.0, code Apache-2.0 (repository README). No UMWP question text is included. The UMWP
ids and labels (answerable, category, source, pairing) are UMWP data under CC BY-SA 4.0; the files that carry them
(`umwp_subset1200.jsonl`, `umwp_dumps/`, `../judge/validation/umwp400_opus_vs_human_labels.jsonl`) should therefore be
shared under CC BY-SA 4.0 with attribution (to be confirmed by the authors).
