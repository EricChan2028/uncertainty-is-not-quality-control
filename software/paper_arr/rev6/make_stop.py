"""rev6 stop-sequence check (exploratory, 7 Oct 00:55, after the registered read): the strict reading cuts a solution at its
first line starting with 'Question:' (amendment 2), as the loop's own filter stops generation there (a3_guard.py STOP). This
check re-runs the UMWP probe with that stop sequence at generation time (umwp_probe_text.py unchanged otherwise) on the three
base models, Qwen3-8B round 3 seed 0 and Qwen3-4B round 3 seeds 0-1, and compares strict() on the stopped texts with the
amended reading (strict_v3.read) on the rev6 texts. Expected difference: samples whose continuation ran to the length limit
count as non-answers under the cut but not under the stop. Output: paper_arr/rev6/texts_stop/stopq_<tag>.jsonl.
Writes umwp_probe_text_stop.py, rev6_launch_stop.sh, jobs_stop.txt."""
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/paper_arr/rev6"
s = open(f"{R}/dirA3/code/umwp_probe_text.py").read()
old = "sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, seed=17)"
assert s.count(old) == 1
s = s.replace(old, "sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_new, n=args.G, seed=17, stop=[\"\\nQuestion:\"])  # rev6 stop-sequence check")
s = '"""rev6 stop-sequence copy of umwp_probe_text.py: generation stops at a new line starting with Question: (as a3_guard.py); otherwise identical."""\n' + s
open(f"{D}/umwp_probe_text_stop.py", "w").write(s)
L = open(f"{D}/rev6_launch_fast.sh").read()
for a, b in (("OUT=$D/texts;", "OUT=$D/texts_stop;"),
             ("    absfix) script=$D/umwp_probe_text_absfix.py; flag=1;;", "    absfix) script=$D/umwp_probe_text_absfix.py; flag=1;;\n    stopq) script=$D/umwp_probe_text_stop.py; flag=0;;")):
    assert L.count(a) == 1, a; L = L.replace(a, b)
open(f"{D}/rev6_launch_stop.sh", "w").write(L)
jobs = ["stopq qwen3_4b_base /root/models/Qwen3-4B-Base", "stopq base_Qwen3-8B-Base /root/models/Qwen3-8B-Base",
        "stopq base_Phi-4-mini-instruct /root/models/Phi-4-mini-instruct",
        "stopq qwen3_8b_Base_own_a3_s0_r3 /dev/shm/ckpt_b/qwen3_8b_Base_own_a3_s0_r3/final",
        "stopq qwen3_4b_Base_own_a3_s0_r3 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s0_r3/final",
        "stopq qwen3_4b_Base_own_a3_s1_r3 /dev/shm/ckpt_b/qwen3_4b_Base_own_a3_s1_r3/final"]
open(f"{D}/jobs_stop.txt", "w").write("\n".join(jobs) + "\n")
print(open(f"{D}/jobs_stop.txt").read())
