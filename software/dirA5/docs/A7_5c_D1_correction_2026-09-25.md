# A7-5c note (07:37 CST 25 Sep 2026): corrected statement of deviation D1 (documentation only; no change to the runs)

Prompted by the author's question. An independent CPU-only review (workflow wf_cff68d69-dba: binary inspection of the real
PyPI wheels, documentation search, adversarial critique) re-checked the judgment behind D1.

Corrected statement (replaces the D1 wording of the addendum):
"The exact pinned environment cannot run on compute capability 12.0 (Blackwell) under any configuration. Three pinned
binaries each block it on their own: (i) torch 2.7.0 from PyPI is the CUDA 12.6 build, with GPU code only up to sm_90 and no
PTX; (ii) vLLM 0.9.1's KV-cache write and core layer kernels (reshape_and_cache[_flash], rms_norm, rotary_embedding,
act_and_mul, paged_attention), which every V0 and V1 attention backend calls, exist only as sm_70-sm_100 machine code
without PTX, so no attention-backend setting helps; (iii) Triton 3.3.0 (required by torch 2.7.0) cannot compile tensor-core
kernels for cc 12.0 (fixed in 3.3.1, triton PR #6771). Rebuilding vLLM 0.9.1 for sm_120 needs GitHub-hosted sources
(CUTLASS, the vLLM flash-attention fork) that are unreachable here. We therefore ran the unmodified R-Zero algorithm on the
cluster stack (torch 2.8.0+cu128, vLLM 0.11.0, transformers 4.57.1, flash-attn 2.8.1, triton 3.4.0, CUDA 12.8 libraries;
127 of 208 pinned packages differ) with three API-compatibility edits (A7-5b)."

Correction to the earlier reasoning: "the flash-attention library has no sm_120 kernels" was not sufficient on its own (other
attention backends exist); the missing core/KV-cache kernels and the Triton failure are what rule out every backend.

Disclosure of a closer option not evaluated before T0: vLLM 0.9.2 (the first release with SM120 support, vLLM PR #19794;
available on the reachable mirror; requires the same torch 2.7.0 / xformers 0.0.30) with torch 2.7.0+cu128 and flash-attn
2.7.4.post1 built from its PyPI sdist would have kept most pins (transformers, ray, tensordict, triton, numpy, datasets) and
would not need edits F1/F2. It is untested (Triton 3.3.0 would remain a risk). D1 is therefore the closest TESTED stack, not
the least-deviation stack.

Why the running seeds continue unchanged: the CPU side of the algorithm (tokenization, the questioner rewards incl. BLEU
clustering, grade_answer, data order) was checked bit-identical under the pinned and the running packages on this run's own
data, so D1 can change only GPU numerics and the vLLM engine; the RZ-P2 comparison (final solver vs base on UMWP) uses the
same probe on the same stack for both sides (base dump 24 Sep 06:05, vLLM 0.11.0), so the stack largely cancels; restarting
under the untested option would cost ~1 day of the registered window and would look like a forking path.
Optional check (not registered; needs the author's go-ahead and a registration before any data): after the seeds finish, a
3-4 h equivalence check of the untested closer stack on one host. Flagged downside: ~10-15% chance that the replication would
then have to be reported as stack-sensitive.
