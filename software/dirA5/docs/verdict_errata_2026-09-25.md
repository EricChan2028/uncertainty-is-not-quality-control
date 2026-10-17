# Errata to the A3 and A4 verdicts (25 Sep 2026), from the verified pre-registration ledger

Source: dirA5/docs/prereg_ledger_2026-09-25.md (four adversarial verification passes against the raw result files). The
ledger is authoritative where it differs from the verdict documents; these errata list the differences that change a label
or a quoted number.

## A3 verdict (dirA3/docs/A3_verdict_2026-09-23.md)
- P1 [Llama-3.2-3B]: FAILS as registered (was read HOLDS from the r1 -> r2 rise alone). Clause (a) holds; (b) the
  deleted/intact ratio falls r2 -> r3 in 3/3 readable seeds and ends below r1 in 2/3; (c) monotone only in s3.
- P2 [Llama-3.2-3B]: FAILS as registered (non-decreasing fails at r3 in 3/3 readable seeds strictly, 2/3 beyond SE).
- P2 [Qwen]: HOLDS 8/8 (unchanged); the r1 -> r3 rise is +18.0 to +27.5 points (not "+13 to +27"; also in s.2b).
- P5: the ledger reads it FAILS from the existing data (the verdict never read it) - needs the author's confirmation.
- A3a scale sentence: neither registered branch is triggered, so the scale sentence is not licensed.
- KC1: the collapse-evidence rule is triggered and non-confirmatory; the verdict's decision not to draw KC1 from it must be
  disclosed.
- Numbers: P1 Llama readable r1 range 0.99-1.14; r6 confident share 87.0-94.0; r4-r6 ratio 1.32-2.22; KA3-1 Q1u r3
  0.316-0.410; own r5 s2 Q1u 0.553 (the constant 0.571 was used in A4j HM3 and A5 H-P1; neither outcome changes); P3 8B s0
  +0.019. Several cells shown as "probe pending" had complete files before the verdict.
- The Llama reads rest on seeds 2-4 (seed 2 was added via the hash log, not the registration text).

## A4 verdict (dirA4/docs/A4_verdict_2026-09-24.md, revision 2)
- "All pending cells filled" is not accurate: four registered reads had complete data but no official read (PF3 [8B],
  PF4 [8B], NF2, HX2b [Phi]); they are being run officially with their hashed scripts on 25 Sep (results appended to the
  ledger).
- HM5: seed 1 is readable (0.095, passes); HM5 still FAILS on seed 2 (0.95).
- NG2 is +11.93, not +12.0; the A4i "raw" 70.0 / 80.5 / 79.0 are audited admitted-sample shares.
- The Qwen3-4B law fit is called a "registered read", but A4a.1 registers no MAE threshold: it is descriptive.
- Cut-off sensitivity: A4i registered a 07:50 cutoff; the loop_uv r4 evals finished 07:53-08:26, so a literal reading makes
  HL2-HL4 NOT_RUN (the ledger keeps them read; both readings are reported).
- The verdict's no-go sentence (L48) is stated although its registered trigger is not met.

## Required "not run" disclosures (no results document made them)
OLMo PG1/PG2; NE1/NE2 with the 14B proposer; loop_uv s1 r4 UMWP (A4h addendum d); the A6 X2 8B clause; S8 G8-P1..P3 (the
8B guard s0 lineage was never launched); A5 MiP M-P1..M-P3; the J-P2 r1/r2 passes; CTRL-L audits of LOWP s0/s1 r2 and all
r3 (descriptive).

## Other disclosures
- A5 J-P3 used Monte Carlo (2,000 draws per item) instead of the registered exact hypergeometric subsampling.
- A5b FB2: the JSON verdict is null (FAIL is arithmetic); the A5b blindness claim is literally false for FB2.
- SWAP C1 could not be composition-matched exactly (9 / 25 / 6 unsolvable items short in s0 / s1 / s3); this bears on S-P2.
- A10: verdicts.jsonl was written before the read-script hash and hashed in the same entry (it was not read before the read).
- ID collisions to disambiguate in any write-up: A5 MiP M-P1..3 vs A8 M-P1..3; A6 D1..D4 vs deviation D1; A6 X1/X2 vs the
  A7-5 stop codes X1..X8; SWAP arm C1 vs CTRL-L C-P*.
