"""A7-5 clock (addendum section 8): stage times and the projected-finish rule. CPU only; writes only
ext/rzero_runs/s<S>/events.tsv and ext/rzero_runs/s<S>/mode.
T0 = 2026-09-24 15:10 CST (A7 registration). All A7-5 compute stops at T0 + 71 h; RZ-P2 is read at T0 + 72 h; audits of
admitted sets that exist by T0 + 71 h may finish by T0 + 84 h.
Stages in order: q1 a1 s1 q2 a2 s2 q3 a3 s3 u (q = questioner stage, a = generation + pseudo-labels + filter, ending with the
admitted set, s = solver training and merge, u = the final solver's UMWP dumps). A stage's completion time is the mtime of
its marker (A75_DONE, the admitted parquet, A75_UMWP_DONE). Upper estimates before any measurement (h): q 3.5, a 1.1, s 3.7,
u 1.0; smoke test 2.5. A stage is measured when it started and ended inside one driver attempt (the event before it is its
predecessor stage; for q1, the seed's first attempt); once a stage type is measured in any seed, its estimate becomes
1.1 x its longest measurement.
Rule: mode "full" if now (+ the smoke test) + the estimates of all remaining stages end by T0 + 71 h; else "p1only" (the run
stops once the iteration-3 admitted set exists, and RZ-P2 is not run for this seed) if the remaining stages up to a3 end by
T0 + 71 h; else "none" (the seed is not started). A committed mode is never upgraded.
  rz_clock.py attempt --seed S --storage DIR     record a driver attempt start and the stages completed so far
  rz_clock.py observe --seed S --storage DIR     record newly completed stages
  rz_clock.py mode --seed S --storage DIR [--smoke 0|1] [--commit]   print "<mode> <details without spaces>" """
import argparse, glob, os, time

R = "/home/tione/notebook/rsi_rl"
T0 = 1790233800; STOP = T0 + 71 * 3600
EST = {"q": 3.5, "a": 1.1, "s": 3.7, "u": 1.0}; SMOKE_H = 2.5
PIPE = ["q1", "a1", "s1", "q2", "a2", "s2", "q3", "a3", "s3", "u"]
RANK = {"none": 0, "p1only": 1, "full": 2}

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["attempt", "observe", "mode"])
ap.add_argument("--seed", type=int, required=True, choices=[0, 1, 2, 3]); ap.add_argument("--storage", required=True)
ap.add_argument("--smoke", type=int, default=0, choices=[0, 1]); ap.add_argument("--commit", action="store_true")
ap.add_argument("--now", type=float, default=None)  # for the design tests only
a = ap.parse_args()
now = time.time() if a.now is None else a.now
X = f"{R}/ext/rzero_runs"; E = f"{X}/s{a.seed}"; os.makedirs(E, exist_ok=True)
abbr = f"rzero_qwen3_4b_Base_s{a.seed}"


def marker(st):
    if st == "u":
        return f"{a.storage}/A75_UMWP_DONE"
    i = st[1]
    return {"q": f"{a.storage}/models/{abbr}_questioner_v{i}/A75_DONE",
            "a": f"{a.storage}/generated_question/{abbr}_solver_v{i}_admitted.parquet",
            "s": f"{a.storage}/models/{abbr}_solver_v{i}/A75_DONE"}[st[0]]


def read_events(path):
    ev = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 2 and (p[1] in PIPE or p[1] == "attempt"):
                ev.append((float(p[0]), p[1]))
    return ev


def observe(attempt):
    path = f"{E}/events.tsv"
    seen = {n for _, n in read_events(path)}
    new = [(os.path.getmtime(marker(st)), st) for st in PIPE if st not in seen and os.path.exists(marker(st))]
    if attempt:
        new.append((now, "attempt"))
    if new:
        with open(path, "a") as fh:
            for t, n in new:
                fh.write(f"{t:.0f}\t{n}\n")


def measured():
    m = {k: [] for k in EST}
    for f in sorted(glob.glob(f"{X}/s[0-9]/events.tsv")):
        ev = sorted(read_events(f))
        for j, (t, n) in enumerate(ev):
            if n == "attempt" or j == 0:
                continue
            k = PIPE.index(n); pt, pn = ev[j - 1]
            if k == 0:
                ok = pn == "attempt" and not any(x == "attempt" for _, x in ev[:j - 1])
            else:
                ok = pn == PIPE[k - 1]
            if ok and t >= pt:
                m[n[0]].append(t - pt)
    return m


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M", time.localtime(t))


if a.cmd in ("attempt", "observe"):
    observe(a.cmd == "attempt")
else:
    observe(False)
    m = measured()
    est = {k: (1.1 * max(v) / 3600 if v else EST[k]) for k, v in m.items()}
    rem = [st for st in PIPE if not os.path.exists(marker(st))]
    base = now + (SMOKE_H * 3600 if a.smoke else 0)
    fin = base + sum(est[st[0]] * 3600 for st in rem)
    a3 = base + sum(est[st[0]] * 3600 for st in rem if PIPE.index(st) <= PIPE.index("a3"))
    md = "full" if fin <= STOP else ("p1only" if a3 <= STOP else "none")
    mf = f"{E}/mode"
    old = open(mf).read().split()[0] if os.path.exists(mf) else None
    if old in RANK and RANK[old] < RANK[md]:
        md = old
    detail = (f"now={iso(now)},final={iso(fin)},it3_admitted={iso(a3)},stop={iso(STOP)},smoke={a.smoke},"
              f"remaining={'+'.join(rem) or '-'},est_h=" + "/".join(f"{k}:{est[k]:.2f}" for k in EST) +
              ",measured_n=" + "/".join(f"{k}:{len(m[k])}" for k in EST) + (f",committed={old}" if old else ""))
    if a.commit:
        with open(mf + ".tmp", "w") as fh:
            fh.write(f"{md} {detail}\n")
        os.replace(mf + ".tmp", mf)
    print(md, detail)
