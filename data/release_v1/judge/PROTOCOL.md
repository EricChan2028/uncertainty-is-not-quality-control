# Opus audit protocol (frozen 2026-09-22)

Judge: Claude Opus (Anthropic), run as independent subagents of Claude Code workflows with the model alias `opus`
(recorded as `claude-opus-5` in the files of the first audits). Each judge instance starts from an empty context, fetches
20-25 items with a fixed shell command, labels every item, and returns (or pushes) one JSON part file. Judges never see
solver outputs, reference answers or other judges' labels; they are told which model family wrote the items.

What the judges could see. The judges were NOT blind to run, round or condition. Every judge ran a fetch command that
names the item directory, and the file-pushing template also asks the judge to return `run="<RUN>"`:
- admitted-pool audits (200-item and full) and raw-proposal audits: `cd <REPO>/dirB/runs/<RUN>`; the run name encodes
  model, condition, seed and round (for example `qwen3_4b_Base_own_a3_s0_r3`); extension audits ran in the pseudo-run
  directories `dirB/runs/swap_audit_a5_s<seed>` / `phi_audit_a6_s<seed>`; in all of them the printed proposer ids
  (`prop<seed*100+round>:<index>`) encode seed and round as well;
- fixed-pool audits: the source directory (`dirB/runs/qwen3_4b_Base_own_a3_s0_r1`, `..._s1_r1`,
  `dirA3/results/frontier/q25_7b`, `dirA3/results/frontier/evo_r3`) and `dirA3/results/pool_audit/<POOL>/`;
- UMWP-400: `dirB/runs/umwp400_j1` and ids `umwp:<UMWP id>`;
- tracer-200: `dirA3/results/audit_validation` and ids `trd:<fam>:<task>` / `tri:<fam>:<task>`, whose prefix tells
  whether the premise was deleted (trd) or kept (tri);
- re-judges: `dirB/runs/<RUN>` and `dirA3/results/audit_rejudge/<RUN>/`.
Whether this information changed any verdict was not measured; no round-blind or condition-blind audit exists.

Label set (all admitted-pool, raw-pool and fixed-pool audits, and the UMWP / tracer validations):
- SOLVABLE: the text determines a unique numeric answer (the judge states it in the note);
- UNSOLVABLE: a needed quantity or relation is missing, or the text is self-contradictory, so no unique answer exists;
- AMBIGUOUS: two reasonable readings give different answers.
Strictness rule: a default assumption the judge would have to invent makes the item UNSOLVABLE. Confidence in [0, 1],
note of at most 20 words. "Ill-posed" in the paper = UNSOLVABLE + AMBIGUOUS; the `illposed_share` field of the original
audit files is the UNSOLVABLE-only share (INDEX.tsv gives both, named explicitly).

Sampling: per audited run, the admitted proposer questions (kind == proposed; planted tracers excluded) are shuffled
with Python `random.Random(0)`; items k = 0..199 form the 200-item audit (`in_audit_sample200`). `audit_k` is every
row's position in this order (the 200-item and full audits of a run print exactly this k; INDEX.jsonl `audit_k_check`
counts mismatches). Full audits extend the same order to every admitted proposer question (a3_audit_merge.py; for
Qwen3-4B own r3 seeds 0-4 and Phi own r3 seeds 0-1 the remainder was audited later through pseudo-runs swap_audit_a5_s* /
phi_audit_a6_s*, whose own k is not the pool position). Fixed pools are audited exhaustively in file order. The first
verdict of an item is kept (no item was judged twice inside one audit; `duplicate_verdict_conflicts` in INDEX.jsonl
counts disagreements if it happened).

Prompt files (verbatim workflow text; only host, port, user and local paths replaced by placeholders). The four-line
definition block and the "Be strict" sentence are identical in all of them except the two twin validations:
- `prompt_admitted_pool_audit.txt`: every admitted-pool audit (200-item and full), the extension audits and the UMWP-400
  validation (pseudo-run `umwp400_j1` with MODEL = Qwen3-4B-Base).
- `prompt_admitted_pool_audit_round1_v2.txt`: the 4 first round-1 200-item audits (llama3p2_3b_own_a3_s0_r1, llama3p2_3b_own_a3_s1_r1, qwen3_4b_Base_own_a3_s0_r1, qwen3_4b_Base_own_a3_s1_r1; the
  copies qwen3_4b_Base_coev_a4_s0_r1, qwen3_4b_Base_coev_a4_s1_r1 inherit them): identical definitions and strictness sentence, 10 judges x 20 items, verdicts
  returned in the structured output instead of pushed part files; their full-audit extensions used the main template.
- `prompt_tracer200_validation.txt`: the tracer-200 validation. Same text and framing as the main template (MODEL =
  Qwen3-4B-Base, "ADMITTED by a self-play filter"), but the fetch command reads `sample.jsonl` of
  `<REPO>/dirA3/results/audit_validation` in file order, without the Random(0) shuffle.
- `prompt_rejudge200_reversed.txt`: the two 200-item re-judges. The main template plus the line `P=P[:200][::-1]`
  after the shuffle (the first audit's sample in reverse order; 400 of
  400 re-judged items carry k = 199 - first k), parts pushed to `<REPO>/dirA3/results/audit_rejudge/<RUN>/parts/`.
- `prompt_raw_pool_audit.txt`: fixed-pool and raw-proposal audits ("as self-play training candidates" instead of
  "ADMITTED by a self-play filter"; items in file order or Random(0) order).
- `prompt_twin_validation_pass1.txt`, `prompt_twin_validation_pass2.txt`: the two twin validations (other label sets).

Provenance of the prompt text: the prompts are the authors' own text. They exist in the Claude Code workflow scripts on
the authors' workstation and were not stored on the compute host; the files here were transcribed verbatim from those
scripts (and from the recorded workflow runs) when this release was built. The four-line definition block and the
strictness sentence are identical in all 51 audit workflow scripts (45 file-pushing, 6 first-round structured-return)
and in every recorded workflow run that contains them.

Human validation packet: a sealed 150-item packet (50 items per Opus class, drawn from drifted rounds; instructions
mirror the judge definitions: S / U / A, "be strict") is planned for non-author annotation. Its item key (which run,
id and Opus class each packet item has) is withheld from this release until those labels are in; summary.json gives
only the class mix.
