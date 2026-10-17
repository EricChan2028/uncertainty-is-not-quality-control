# A8a clarification (23:30 CST 24 Sep 2026): fixed after a two-reviewer check, before any unit is built or labelled

1. Solution text shown to labellers: the completion is first cut at its first "\nQuestion:" (the plain prompt has no stop
   sequence, so a base-model sample can continue with an invented follow-on problem), then truncated to its last 1,800
   characters. The number of cut and of truncated texts per model is recorded in the key file (descriptive; M-P2 restricted to
   untruncated samples is reported as a sensitivity).
2. Labelling instructions: fixed verbatim in dirA5/code/a8_label_prompt.txt (hashed with this note); solutions are shown as
   [0]..[7] and indices are 0-based. Solutions that decline because information is missing/ambiguous/contradictory form one
   cluster whose reading starts with "DECLINES:"; solutions whose reading cannot be determined form one cluster whose reading
   starts with "NO_READING". disagreement_source is defined through departures from the modal answer as in the prompt.
3. M-P3 counts reading clusters EXCLUDING the NO_READING cluster (DECLINES counts as a reading); the count including every
   cluster is descriptive. M-P1 and M-P2 are unchanged.
4. Batching: 5 units per labeller, formed from the shuffled keys so that no batch holds two units of the same item or of an
   unanswerable item and its answerable original; the second pass uses fresh labellers, the identical prompt, the registered
   25% base subsample, and never sees first-pass labels. A returned unit that fails validation is re-asked up to 2 times; the
   first valid label per key is kept; after 2 failed retries the unit counts as missing.
5. Read rules (in the hashed read script): invalid/missing labels count as missing; a cell with > 5% missing makes the
   predictions that use it None; 2-of-3 verdicts are three-state (True if >= 2 hold, False if >= 2 fail, else None); labels
   whose sample indices are exactly {1..8} are shifted to 0-based (such a set can never be a valid 0-based partition).
