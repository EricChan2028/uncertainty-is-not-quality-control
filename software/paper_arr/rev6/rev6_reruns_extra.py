"""rev6 (exploratory): the registered UMWP reads not covered by rev6_reruns.py, on the same pseudo-dumps (run rev6_reruns.py
first): dirA3/code/umwp_read.py (HX1-HX4), dirA3/code/umwp_read_c.py (HX5, HX6), dirA5/code/a5_frontier_b.py (FB1-FB3), so
that the registered UMWP predictions can be re-scored under both parsers. As in rev6_reruns.py, each script runs from a copy
whose only changes are the UMWP directory (also inside the a5_frontier.py source that a5_frontier_b.py executes) and the
output path (paper_arr/rev6/reruns_<P>/).   python3 rev6_reruns_extra.py"""
import os, subprocess, sys
R = "/home/tione/notebook/rsi_rl"; D6 = f"{R}/paper_arr/rev6"
JOBS = {
    "dirA3/code/umwp_read.py": [('open(f"{R}/dirA4/results/umwp_read.json", "w")', 'open(f"{R}/paper_arr/rev6/reruns_{P}/umwp_read.json", "w")')],
    "dirA3/code/umwp_read_c.py": [('open(f"{R}/dirA4/results/umwp_read_c.json", "w")', 'open(f"{R}/paper_arr/rev6/reruns_{P}/umwp_read_c.json", "w")')],
    "dirA5/code/a5_frontier_b.py": [('open(f"{R}/dirA5/results/frontier_b_read.json", "w")', 'open(f"{R}/paper_arr/rev6/reruns_{P}/frontier_b_read.json", "w")'),
                                    ('src = open(f"{R}/dirA5/code/a5_frontier.py").read()',
                                     'src = open(f"{R}/dirA5/code/a5_frontier.py").read().replace("/dirA4/results/umwp", "/paper_arr/rev6/dumps_{P}")')],
}
for script, reps in JOBS.items():
    for P in ("original", "strict"):
        s = open(f"{R}/{script}").read()
        for old, new in reps:
            assert s.count(old) == 1, (script, old, s.count(old)); s = s.replace(old, new.replace("{P}", P))
        k = s.count("/dirA4/results/umwp"); s = s.replace("/dirA4/results/umwp", f"/paper_arr/rev6/dumps_{P}")
        cp = f"{D6}/reruns_{P}/{os.path.basename(script)}"
        open(cp, "w").write(f"# rev6 rerun copy of {script} (UMWP dir -> dumps_{P}, {k} occurrence(s); output -> reruns_{P})\n" + s)
        r = subprocess.run([sys.executable, cp], cwd=os.path.dirname(f"{R}/{script}"), capture_output=True, text=True)
        open(f"{cp}.log", "w").write(r.stdout + "\n--- stderr\n" + r.stderr)
        print(f"{script:32s} {P:9s} rc={r.returncode} (UMWP dir replaced {k}x)" + ("" if r.returncode == 0 else " :: " + r.stderr.strip().splitlines()[-1][:200]))
