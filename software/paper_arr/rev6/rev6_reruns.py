"""rev6 (exploratory, after the plan): the paper's other UMWP analyses on the rev6 samples under both parsers.

1. Pseudo-dumps. For each rev6 dump paper_arr/rev6/texts/<kind>_<tag>.jsonl a copy without texts is written to
   paper_arr/rev6/dumps_<P>/ for P = original (the stored answers) and strict (a sample is a non-answer where strict(), or
   strict_abs() under the abstention prompt, finds none; otherwise the probe's number, exactly as in rev6_read.py), with mode, p_mode, nonanswer and correct recomputed by the
   probe's own rule (Counter over the 8 answers, None included; correct = mode == ref) and every other field unchanged.
   Names follow dirA4/results/umwp: plain_<tag>, absfix_<tag> (also linked as abs_<tag>, so that scripts reading the
   abstention-prompt slot read the corrected dumps, as ext/a11_draft/a11_read.py does), plainchat_<tag> for the chat-format
   runs (rev6 names plain_chat_<tag>). The A8 texts (dirA5/results/a8/texts) get the same treatment (dumps_<P>/a8/).
2. Reruns. Each script below is executed from a copy whose only changes are: the UMWP directory dirA4/results/umwp -> the
   pseudo-dump directory (and dirA5/results/a8 -> dumps_<P>/a8 where the A8 texts are read), and its output path ->
   paper_arr/rev6/reruns_<P>/. The copies are kept next to their outputs. Inputs the rev6 runs do not cover are missing from
   the pseudo-dump directory, so each script treats them as it treats any missing dump.
python3 rev6_reruns.py"""
import json, os, re, subprocess, sys
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; D6 = f"{R}/paper_arr/rev6"
sys.path.insert(0, f"{R}/code"); from loto_common import last_boxed  # noqa: E402
sys.path.insert(0, f"{R}/dirB/code"); from pdc_util import is_abstain  # noqa: E402
SP = f"{R}/paper_arr/analyses/rev5/strict_parse.py"
src = open(SP).read(); sp = {}; exec(compile(src[:src.index("# ---- validation against the human answer status")], SP, "exec"), sp)
strict, norm_ans = sp["strict"], sp["norm_ans"]
sys.path.insert(0, D6); import strict_v3  # noqa: E402  (amendment 2: the strict reading of rev6_read.py)
ABS_RE = re.compile(r"Answer:\s*(unknown|unanswerable|insufficient information)", re.I)
def strict_abs(t, f):
    if f == "length": return None
    b = last_boxed(t)
    if b is not None and is_abstain(b): return None
    if b is not None: return norm_ans(b)
    if ABS_RE.search(t): return None
    return strict(t, f)
def rewrite(rows_in, out_path, parser, prompt):
    rows = {}
    for l in open(rows_in):
        try: r = json.loads(l); rows[r["id"]] = r
        except ValueError: continue
    with open(out_path, "w") as fh:
        for r in rows.values():
            if parser == "original": ans = list(r["answers"])
            else:  # strict()'s status, the probe's number (as rev6_read.py, amendment)
                fn = strict if prompt == "plain" else strict_abs
                ans = [strict_v3.read(t, f, prompt) for t, f in zip(r["texts"], r["finish"])]  # amendment 2
            mode, n = Counter(ans).most_common(1)[0]
            o = {k: v for k, v in r.items() if k not in ("texts", "finish")}
            o.update({"answers": ans, "mode": mode, "p_mode": n / len(ans), "nonanswer": sum(a is None for a in ans) / len(ans),
                      "correct": int(mode is not None and r.get("ref") is not None and mode == r["ref"])})
            fh.write(json.dumps(o) + "\n")
    return len(rows)
for P in ("original", "strict"):
    d = f"{D6}/dumps_{P}"; os.makedirs(f"{d}/a8/texts", exist_ok=True); os.makedirs(f"{D6}/reruns_{P}", exist_ok=True)
    for f in sorted(os.listdir(f"{D6}/texts")):
        if not f.endswith(".jsonl"): continue
        kind = "absfix" if f.startswith("absfix_") else "plain"
        name = "plainchat_" + f[len("plain_chat_"):] if f.startswith("plain_chat_") else f
        n = rewrite(f"{D6}/texts/{f}", f"{d}/{name}", P, kind)
        if n < 1200: print("INCOMPLETE", P, f, n)
        if kind == "absfix":
            ln = f"{d}/abs_{f[len('absfix_'):]}"
            if os.path.lexists(ln): os.remove(ln)
            os.symlink(f"{d}/{name}", ln)
    for f in sorted(os.listdir(f"{R}/dirA5/results/a8/texts")):
        if f.endswith(".jsonl"): rewrite(f"{R}/dirA5/results/a8/texts/{f}", f"{d}/a8/texts/{f}", P, "plain")
    for f in os.listdir(f"{R}/dirA5/results/a8"):
        if f == "items.json": os.system(f"cp {R}/dirA5/results/a8/items.json {d}/a8/items.json")
JOBS = {  # script -> [(old, new)] applied besides the UMWP directory; {P} = parser
    "dirA5/code/a5_j1_judge_free.py": [('f"{R}/dirA5/results/j1_judge_free.json"', 'f"{R}/paper_arr/rev6/reruns_{P}/j1_judge_free.json"')],
    "dirA5/code/a6_matched.py": [('f"{R}/dirA5/results/a6_matched.json"', 'f"{R}/paper_arr/rev6/reruns_{P}/a6_matched.json"')],
    "dirA5/code/a7_exit.py": [('f"{R}/dirA5/results/a7_exit_categories.json"', 'f"{R}/paper_arr/rev6/reruns_{P}/a7_exit_categories.json"'),
                              ('src = open(f"{R}/dirA5/code/a6_matched.py").read()',
                               'src = open(f"{R}/dirA5/code/a6_matched.py").read().replace("/dirA4/results/umwp", "/paper_arr/rev6/dumps_{P}")')],
    "paper_arr/analyses/hardtraj.py": [('OUT = f"{R}/paper_arr/analyses/hardtraj.json"', 'OUT = f"{R}/paper_arr/rev6/reruns_{P}/hardtraj.json"'),
                                       ('A8 = f"{R}/dirA5/results/a8"', 'A8 = f"{R}/paper_arr/rev6/dumps_{P}/a8"')],
    "paper_arr/analyses/convergence.py": [('open(f"{R}/paper_arr/analyses/convergence.json", "w")', 'open(f"{R}/paper_arr/rev6/reruns_{P}/convergence.json", "w")')],
    "paper_arr/analyses/check_convergence.py": [('open(R + "/paper_arr/analyses/check_convergence.json", "w")', 'open(R + "/paper_arr/rev6/reruns_{P}/check_convergence.json", "w")')],
    "paper_arr/analyses/rev3/ca_thresholds.py": [('open(f"{R}/paper_arr/analyses/rev3/ca_thresholds_2026-10-04.json", "w")', 'open(f"{R}/paper_arr/rev6/reruns_{P}/ca_thresholds.json", "w")')],
    "dirA5/code/a5_frontier.py": [('os.makedirs(f"{R}/dirA5/results", exist_ok=True)', 'os.makedirs(f"{R}/paper_arr/rev6/reruns_{P}", exist_ok=True)'),
                                  ('open(f"{R}/dirA5/results/frontier_read.json", "w")', 'open(f"{R}/paper_arr/rev6/reruns_{P}/frontier_read.json", "w")')],
    "dirA5/code/rz_paired_desc.py": [],
    "paper_arr/figures/fig1_drift_harm.py": [('OUT = f"{ROOT}/paper_arr/figures/fig1_drift_harm.pdf"', 'OUT = f"{ROOT}/paper_arr/rev6/reruns_{P}/fig1_drift_harm.pdf"')],
}
only = sys.argv[1:]
for script, reps in JOBS.items():
    if only and not any(o in script for o in only): continue
    for P in ("original", "strict"):
        s = open(f"{R}/{script}").read()
        for old, new in reps:
            assert s.count(old) == 1, (script, old, s.count(old)); s = s.replace(old, new.replace("{P}", P))
        k = s.count("/dirA4/results/umwp"); s = s.replace("/dirA4/results/umwp", f"/paper_arr/rev6/dumps_{P}")
        cp = f"{D6}/reruns_{P}/{os.path.basename(script)}"
        open(cp, "w").write(f"# rev6 rerun copy of {script} (UMWP dir -> dumps_{P}, {k} occurrence(s); output -> reruns_{P})\n" + s)
        r = subprocess.run([sys.executable, cp], cwd=os.path.dirname(f"{R}/{script}"), capture_output=True, text=True)
        open(f"{D6}/reruns_{P}/{os.path.basename(script)}.log", "w").write(r.stdout + "\n--- stderr\n" + r.stderr)
        print(f"{script:45s} {P:9s} rc={r.returncode} (UMWP dir replaced {k}x)" + ("" if r.returncode == 0 else " :: " + r.stderr.strip().splitlines()[-1][:200]))
