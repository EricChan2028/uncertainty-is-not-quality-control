"""rev6: text-saving copy of dirA5/code/a6_probe.py (external sets SUM, MiP-GSM8K, MiP-MATH), made the way
dirA3/code/umwp_probe_text.py was made from umwp_probe.py: only --ids (accepted, unused: all items) / --outdir and the saved
texts differ. Also writes the rev6 launcher variant for it (kind a6plain, outputs in paper_arr/rev6/texts_a6)."""
R = "/home/tione/notebook/rsi_rl"; D = f"{R}/paper_arr/rev6"
s = open(f"{R}/dirA5/code/a6_probe.py").read()
reps = [('ap.add_argument("--n_uns", type=int, default=600)\n',
         'ap.add_argument("--n_uns", type=int, default=600)\nap.add_argument("--ids", default=None); ap.add_argument("--outdir", required=True)\n'),
        ('R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA5/results/a6_sweep"; os.makedirs(OUT, exist_ok=True)',
         'R = "/home/tione/notebook/rsi_rl"; OUT = args.outdir; os.makedirs(OUT, exist_ok=True)'),
        ('"correct": int(mode is not None and ref is not None and mode == ref), "answers": ans}) + "\\n")',
         '"correct": int(mode is not None and ref is not None and mode == ref), "answers": ans, "texts": texts, "finish": fin}) + "\\n")')]
for a, b in reps:
    assert s.count(a) == 1, a; s = s.replace(a, b)
s = '"""rev6 text-saving copy of dirA5/code/a6_probe.py (only --ids/--outdir and the saved texts differ)."""\n' + s
open(f"{D}/a6_probe_text.py", "w").write(s)
L = open(f"{D}/rev6_launch_xfer.sh").read()
reps = [("OUT=$D/texts;", "OUT=$D/texts_a6;"),
        ("    absfix) script=$D/umwp_probe_text_absfix.py; flag=1;;", "    absfix) script=$D/umwp_probe_text_absfix.py; flag=1;;\n    a6plain) script=$D/a6_probe_text.py; flag=0;;"),
        ('[ "$(wc -l < $out)" -ge 1200 ]', '[ "$(wc -l < $out)" -ge 1552 ]')]
for a, b in reps:
    assert L.count(a) == 1, a; L = L.replace(a, b)
L = L.replace("# Variant for the checkpoints copied from host B", "# a6 variant (external sets, kind a6plain, outputs in texts_a6, complete at 1,552 rows). Variant for the checkpoints copied from host B")
open(f"{D}/rev6_launch_a6.sh", "w").write(L)
B = "/dev/shm/ckpt_b/qwen3_4b_Base_%s_s%d_r3/final"
jobs = ["a6plain qwen3_4b_base /root/models/Qwen3-4B-Base"] + [f"a6plain qwen3_4b_Base_{a}_s{s}_r3 {B % (a, s)}" for a in ("own_a3", "own_clean_a3") for s in range(4)]
open(f"{D}/jobs_a6.txt", "w").write("\n".join(jobs) + "\n")
print(open(f"{D}/jobs_a6.txt").read())
