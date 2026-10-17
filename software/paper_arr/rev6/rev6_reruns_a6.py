"""rev6 (exploratory, after the plan): the external-set read dirA5/code/a6_e_read.py (E1: the full curriculum answers
self-consistently more often than the solvable-only curriculum without the abstention option; E2: abstention with it) on the
rev6 external-set samples under both parsers, as rev6_reruns.py does for UMWP.
Pseudo-dumps paper_arr/rev6/dumps_<P>/a6/plain_<tag>.jsonl from paper_arr/rev6/texts_a6/a6plain_<tag>.jsonl (P = original:
stored answers; strict: non-answer where strict() finds none, otherwise the probe's number), with mode, p_mode, nonanswer and
correct recomputed by the probe's rule. The abstention-prompt slot abs_<tag> links to the corrected abstention dumps of A11
(dirA5/results/a6_sweep/absfix_<tag>.jsonl; abstention does not depend on the numeric parser), as ext/a11_draft/a11_read.py
reads them. a6_e_read.py is executed from a copy whose only changes are its input directory and its output path.
python3 rev6_reruns_a6.py"""
import json, os, subprocess, sys
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; D6 = f"{R}/paper_arr/rev6"; A6 = f"{R}/dirA5/results/a6_sweep"
SP = f"{R}/paper_arr/analyses/rev5/strict_parse.py"
src = open(SP).read(); sp = {}; exec(compile(src[:src.index("# ---- validation against the human answer status")], SP, "exec"), sp)
strict = sp["strict"]
sys.path.insert(0, D6); import strict_v3  # noqa: E402  (amendment 2)
for P in ("original", "strict"):
    d = f"{D6}/dumps_{P}/a6"; os.makedirs(d, exist_ok=True); os.makedirs(f"{D6}/reruns_{P}", exist_ok=True)
    for f in sorted(os.listdir(f"{D6}/texts_a6")):
        if not f.startswith("a6plain_") or not f.endswith(".jsonl"): continue
        tag = f[len("a6plain_"):-len(".jsonl")]
        rows = {}
        for l in open(f"{D6}/texts_a6/{f}"):
            try: r = json.loads(l); rows[r["id"]] = r
            except ValueError: continue
        with open(f"{d}/plain_{tag}.jsonl", "w") as fh:
            for r in rows.values():
                ans = list(r["answers"]) if P == "original" else \
                    [strict_v3.read(t, fi, "plain") for t, fi in zip(r["texts"], r["finish"])]
                mode, n = Counter(ans).most_common(1)[0]
                o = {k: v for k, v in r.items() if k not in ("texts", "finish")}
                o.update({"answers": ans, "mode": mode, "p_mode": n / len(ans), "nonanswer": sum(a is None for a in ans) / len(ans),
                          "correct": int(mode is not None and r.get("ref") is not None and mode == r["ref"])})
                fh.write(json.dumps(o) + "\n")
        if os.path.exists(f"{A6}/absfix_{tag}.jsonl"):
            ln = f"{d}/abs_{tag}.jsonl"
            if os.path.lexists(ln): os.remove(ln)
            os.symlink(f"{A6}/absfix_{tag}.jsonl", ln)
        print(P, tag, len(rows), "rows")
    s = open(f"{R}/dirA5/code/a6_e_read.py").read()
    reps = [('D = f"{R}/dirA5/results/a6_sweep"', f'D = f"{{R}}/paper_arr/rev6/dumps_{P}/a6"'),
            ('open(f"{R}/dirA5/results/a6_e_read.json", "w")', f'open(f"{{R}}/paper_arr/rev6/reruns_{P}/a6_e_read.json", "w")')]
    for a, b in reps:
        assert s.count(a) == 1, a; s = s.replace(a, b)
    cp = f"{D6}/reruns_{P}/a6_e_read.py"; open(cp, "w").write(f"# rev6 rerun copy of dirA5/code/a6_e_read.py (input dir -> dumps_{P}/a6; output -> reruns_{P})\n" + s)
    r = subprocess.run([sys.executable, cp], cwd=f"{R}/dirA5/code", capture_output=True, text=True)
    open(f"{cp}.log", "w").write(r.stdout + "\n--- stderr\n" + r.stderr)
    print("a6_e_read", P, "rc", r.returncode, r.stderr.strip().splitlines()[-1][:200] if r.returncode else "")
