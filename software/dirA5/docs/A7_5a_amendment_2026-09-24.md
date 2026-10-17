# A7-5a amendment (23:08 CST 24 Sep 2026): run-control fix, before any R-Zero data

What happened: the first smoke test on host B (<HOST>) started 23:03:56 and was stopped by its own monitor at 23:05:26 with
exit 6 ("foreign GPU use"). The 4 processes it flagged (pids 1253911/1253916/1253926/1254009, VLLM::EngineCore, ~59 GB each on
GPUs 4-7) were R-Zero's own reward-server engines (started 23:04:05, working directory /run/ti/rzero/R-Zero). vLLM engine
processes rename themselves with setproctitle, which blanks their /proc/<pid>/environ view, so the monitor could not read
RZ_A75_TAG there; kill_tag, which also selects by that environ, left them orphaned (ppid 1). No co-tenant was involved, and
the smoke test produced no data (it stopped during the first questioner stage).

Change (rz_common.sh only; no change to the R-Zero code, its patch, defaults, seeds, reads or verdict rules):
- new helpers rz_cwd (working directory inside $RZ = /run/ti/rzero), anc_tagged (a tagged ancestor), rz_engine_pids (GPU
  compute or VLLM-titled processes with cwd inside $RZ);
- foreign_gpu: a GPU process is ours if its environ carries the tag, or its cwd is inside $RZ, or an ancestor is tagged;
- kill_tag: also stops rz_engine_pids (so no R-Zero engine can outlive its run).
Every R-Zero process runs with its cwd inside $RZ (rz_smoke.sh and rz_run.sh cd to $RZ/R-Zero; rz_umwp.sh inherits it); no
other job of ours and no co-tenant process uses that directory, and co-tenant processes of other containers remain foreign as
before. Verified on host B: the patched functions classify the 4 engines as ours and report no foreign use.

Actions: the orphaned engines are stopped with the patched kill_tag; both waiters are restarted so that they source the
patched file; the smoke test then reruns from its start. The T0 clock and the projected-finish rule are unchanged.
New hash of rz_common.sh is recorded with this amendment in docs/registration_hashes.txt.
