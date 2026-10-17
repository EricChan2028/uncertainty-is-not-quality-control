"""EXPLORATORY (post hoc) attribution table, Qwen3-4B-Base, round 3 (+ one-round A6 swap).
Read-only over dirB/runs, dirA4/results/umwp, dirB/results/evals; writes only paper_arr/analyses/attribution.json.

Per run: final-round training file = config.json "data" (admitted.jsonl, or admitted_clean.jsonl for own_clean);
n_train = its line count; n_unique_proposer_problems = distinct whitespace-normalised "problem" texts among kind == proposed;
passes_per_item = steps*qps/n_train (60*32/n); n_deleted_tracers = rows with kind == tracer_deleted in the training file
(final round), plus a lineage-cumulative count over the r1..r3 training files (swap arms: own_a3 r1, r2 + swap r3).
UMWP (plain prompt, dirA4/results/umwp/plain_<tag>.jsonl, rows de-duplicated by id as in dirA3/code/umwp_read.py):
CA = share of the 600 answerable == False rows with mode not None and p_mode >= 0.5; ACC_A = 100*mean of "correct" over
the 600 answerable rows. Benchmark accuracy = 100*mean greedy_acc_lenient over math500_train, math_heldout1000,
gsm8k_test500 in dirB/results/evals/<run>_step60.summary.json (base: qwen3_4b_base_step0).
Paired seed-level differences (same seed index), mean +- t_{n-1,0.975}*sd/sqrt(n).
Usage: python3 attribution.py
"""
import json, math, os, statistics
from collections import Counter
from scipy import stats

R = "/home/tione/notebook/rsi_rl"
RUNS = R + "/dirB/runs"
UMWP = R + "/dirA4/results/umwp"
EV = R + "/dirB/results/evals"
OUT = R + "/paper_arr/analyses/attribution.json"
BENCH = ("math500_train", "math_heldout1000", "gsm8k_test500")

ARMS = {  # arm -> (run-name infix, seeds, lineage type)
    "own": ("own_a3", range(5), "loop"),
    "clean": ("own_clean_a3", range(4), "loop"),
    "rnd": ("ctrl_rnd_a5", range(3), "loop"),
    "lowp": ("ctrl_lowp_a5", range(3), "loop"),
    "guard": ("own_ag_a4", range(3), "loop"),
    "cln1": ("cln1_a6", range(5), "swap"),
    "rnd1": ("rnd1_a6", range(5), "swap"),
}


def rn(infix, s, r):
    return "qwen3_4b_Base_%s_s%d_r%d" % (infix, s, r)


def train_file(run):
    c = json.load(open(os.path.join(RUNS, run, "config.json")))
    p = c["data"]
    assert os.path.dirname(p) == os.path.join(RUNS, run), (run, p)
    return p, c["steps"], c["qps"]


def load_rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def norm(t):
    return " ".join(str(t).split())


def train_stats(run):
    p, steps, qps = train_file(run)
    rows = load_rows(p)
    kinds = Counter(r["kind"] for r in rows)
    props = [r for r in rows if r["kind"] == "proposed"]
    last = None
    done = False
    for l in open(os.path.join(RUNS, run, "log.jsonl")):
        try:
            d = json.loads(l)
        except ValueError:
            continue
        if "step" in d:
            last = d
        if d.get("done"):
            done = True
    n = len(rows)
    return {"train_file": os.path.basename(p), "n_train": n, "kinds": dict(kinds),
            "n_proposed_items": len(props),
            "n_unique_proposer_problems": len({norm(r["problem"]) for r in props}),
            "n_unique_ids": len({r["id"] for r in rows}),
            "n_deleted_tracers": kinds.get("tracer_deleted", 0),
            "steps": steps, "qps": qps, "passes_per_item": steps * qps / n,
            "log_last_step": None if last is None else last["step"],
            "log_last_epoch_index": None if last is None else last.get("epoch"), "log_done": done}


def umwp(tag):
    f = os.path.join(UMWP, "plain_%s.jsonl" % tag)
    rows = {}
    for l in open(f):
        try:
            r = json.loads(l)
            rows[r["id"]] = r
        except ValueError:
            pass
    u = [r for r in rows.values() if r["answerable"] is False]
    a = [r for r in rows.values() if r["answerable"] is True]
    assert len(u) == 600 and len(a) == 600, (tag, len(u), len(a))
    ca_k = sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u)
    return {"n_unanswerable": len(u), "n_answerable": len(a), "CA": ca_k / len(u), "CA_count": ca_k,
            "ACC_A": 100 * sum(r["correct"] for r in a) / len(a)}


def bench(name):
    d = json.load(open(os.path.join(EV, name + ".summary.json")))
    per = {b: 100 * d[b]["greedy_acc_lenient"] for b in BENCH}
    return {"acc_mean": sum(per.values()) / 3, "per_set": per, "n_per_set": {b: d[b]["n"] for b in BENCH}}


out = {"_note": "EXPLORATORY post hoc analysis; not a registered read. Definitions in the docstring of attribution.py.",
       "base": {"umwp": umwp("qwen3_4b_base"), "bench": bench("qwen3_4b_base_step0")}, "runs": {}}

for arm, (infix, seeds, kind) in ARMS.items():
    for s in seeds:
        run = rn(infix, s, 3)
        rec = {"arm": arm, "seed": s, "run": run, "final_round": train_stats(run)}
        if kind == "loop":
            lin = [rn(infix, s, r) for r in (1, 2, 3)]
        else:
            lin = [rn("own_a3", s, 1), rn("own_a3", s, 2), run]
        per = [train_stats(x) for x in lin]
        rec["lineage_runs"] = lin
        rec["lineage_n_train_per_round"] = [q["n_train"] for q in per]
        rec["lineage_deleted_tracers_per_round"] = [q["n_deleted_tracers"] for q in per]
        rec["lineage_deleted_tracers_cum_r1_r3"] = sum(q["n_deleted_tracers"] for q in per)
        rec["umwp"] = umwp(run)
        rec["bench"] = bench(run + "_step60")
        out["runs"]["%s_s%d" % (arm, s)] = rec


def val(arm, s, metric):
    r = out["runs"]["%s_s%d" % (arm, s)]
    return {"CA": r["umwp"]["CA"], "ACC_A": r["umwp"]["ACC_A"], "bench": r["bench"]["acc_mean"],
            "n_train": r["final_round"]["n_train"],
            "n_deleted_tracers": r["final_round"]["n_deleted_tracers"]}[metric]


def paired(a, b, seeds, metric):
    d = [val(a, s, metric) - val(b, s, metric) for s in seeds]
    n = len(d)
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / math.sqrt(n)
    t = float(stats.t.ppf(0.975, n - 1))
    tstat = m / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(tstat), n - 1)) if se > 0 else float("nan")
    return {"contrast": "%s - %s" % (a, b), "metric": metric, "seeds": list(seeds), "n": n, "diffs": d,
            "mean": m, "sd": sd, "t_crit": t, "ci95": [m - t * se, m + t * se], "t": tstat, "p_two_sided": p,
            "n_negative": sum(x < 0 for x in d), "n_positive": sum(x > 0 for x in d)}


CONTRASTS = [("clean", "own", range(4)), ("rnd", "own", range(3)), ("guard", "own", range(3)),
             ("cln1", "rnd1", range(5)), ("lowp", "own", range(3))]
out["paired"] = {}
for a, b, seeds in CONTRASTS:
    for metric in ("CA", "bench", "ACC_A", "n_train", "n_deleted_tracers"):
        out["paired"]["%s-%s:%s" % (a, b, metric)] = paired(a, b, seeds, metric)


def arm_summary(arm):
    ks = [k for k in out["runs"] if out["runs"][k]["arm"] == arm]
    g = lambda f: [f(out["runs"][k]) for k in ks]
    return {"seeds": [out["runs"][k]["seed"] for k in ks],
            "n_train": g(lambda r: r["final_round"]["n_train"]),
            "n_unique_proposer_problems": g(lambda r: r["final_round"]["n_unique_proposer_problems"]),
            "passes_per_item": [round(x, 2) for x in g(lambda r: r["final_round"]["passes_per_item"])],
            "n_deleted_tracers_final": g(lambda r: r["final_round"]["n_deleted_tracers"]),
            "n_deleted_tracers_cum_r1_r3": g(lambda r: r["lineage_deleted_tracers_cum_r1_r3"]),
            "CA": [round(x, 4) for x in g(lambda r: r["umwp"]["CA"])],
            "ACC_A": [round(x, 2) for x in g(lambda r: r["umwp"]["ACC_A"])],
            "bench": [round(x, 2) for x in g(lambda r: r["bench"]["acc_mean"])],
            "CA_mean": statistics.mean(g(lambda r: r["umwp"]["CA"])),
            "ACC_A_mean": statistics.mean(g(lambda r: r["umwp"]["ACC_A"])),
            "bench_mean": statistics.mean(g(lambda r: r["bench"]["acc_mean"]))}


out["arm_summary"] = {a: arm_summary(a) for a in ARMS}
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1)

print("base  CA %.4f ACC_A %.2f bench %.2f" % (out["base"]["umwp"]["CA"], out["base"]["umwp"]["ACC_A"],
                                              out["base"]["bench"]["acc_mean"]))
print("%-9s %5s %5s %6s %4s %5s %7s %7s %7s" % ("run", "n", "uniqP", "pass", "del", "delC", "CA", "ACC_A", "bench"))
for k, r in out["runs"].items():
    f = r["final_round"]
    print("%-9s %5d %5d %6.2f %4d %5d %7.4f %7.2f %7.2f  %s ids=%d done=%s step=%s lin_n=%s lin_del=%s" % (
        k, f["n_train"], f["n_unique_proposer_problems"], f["passes_per_item"], f["n_deleted_tracers"],
        r["lineage_deleted_tracers_cum_r1_r3"], r["umwp"]["CA"], r["umwp"]["ACC_A"], r["bench"]["acc_mean"],
        f["kinds"], f["n_unique_ids"], f["log_done"], f["log_last_step"], r["lineage_n_train_per_round"],
        r["lineage_deleted_tracers_per_round"]))
for k, p in out["paired"].items():
    print("%-26s n=%d mean=%+.4f ci95=[%+.4f, %+.4f] p=%.3g neg=%d diffs=%s" % (
        k, p["n"], p["mean"], p["ci95"][0], p["ci95"][1], p["p_two_sided"], p["n_negative"],
        [round(x, 4) for x in p["diffs"]]))
