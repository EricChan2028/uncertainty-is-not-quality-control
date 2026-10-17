"""Independent check of paper_arr/analyses/attribution.py (EXPLORATORY, post hoc). Read-only except check_attribution.json.
python3 check_attribution.py"""
import glob, json, math, os
from collections import Counter

R = "/home/tione/notebook/rsi_rl"; RUNS = f"{R}/dirB/runs"; UM = f"{R}/dirA4/results/umwp"; EV = f"{R}/dirB/results/evals"
TCRIT = {2: 4.302652729911275, 3: 3.182446305284263, 4: 2.7764451051977987}

def jl(p):
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

def name(infix, s, r=3):
    return f"qwen3_4b_Base_{infix}_s{s}_r{r}"

def trainset(run):
    cfg = json.load(open(f"{RUNS}/{run}/config.json"))
    path = cfg["data"]
    rows = jl(path)
    k = Counter(r["kind"] for r in rows)
    probs = [r["problem"] for r in rows if r["kind"] == "proposed"]
    logs = []
    for line in open(f"{RUNS}/{run}/log.jsonl"):
        try: logs.append(json.loads(line))
        except ValueError: pass
    steps = [d for d in logs if "step" in d]
    last = max(steps, key=lambda d: d["step"])
    n = len(rows)
    total = cfg["steps"] * cfg["qps"]
    return {"cfg_data": os.path.relpath(path, RUNS), "cfg_model": cfg["model"], "n": n, "kinds": dict(k),
            "uniq_prop_ws": len({" ".join(p.split()) for p in probs}),
            "uniq_prop_raw": len(set(probs)), "uniq_prop_lower": len({" ".join(p.lower().split()) for p in probs}),
            "uniq_prompt_all": len({r["prompt"] for r in rows}),
            "passes": total / n, "steps": cfg["steps"], "qps": cfg["qps"],
            "last_step": last["step"], "last_epoch": last.get("epoch"), "expected_epoch": (total - 1) // n,
            "done": any(d.get("done") for d in logs), "n_tdel": k.get("tracer_deleted", 0),
            "ids": [r["id"] for r in rows]}

def umwp(tag, keep="last"):
    rows = jl(f"{UM}/plain_{tag}.jsonl")
    d = {}
    for r in rows:
        if keep == "last" or r["id"] not in d:
            d[r["id"]] = r
    U = [r for r in d.values() if r["answerable"] is False]
    A = [r for r in d.values() if r["answerable"] is True]
    ca = sum(1 for r in U if r["mode"] is not None and r["p_mode"] >= 0.5)
    return {"n_lines": len(rows), "n_ids": len(d), "nU": len(U), "nA": len(A), "CA_k": ca, "CA": ca / len(U),
            "ACC_A": 100.0 * sum(1 for r in A if r["correct"]) / len(A),
            "correct_vals": sorted({str(r["correct"]) for r in A})}

def bench(tag):
    d = json.load(open(f"{EV}/{tag}.summary.json"))
    sets = ["math500_train", "math_heldout1000", "gsm8k_test500"]
    v = [100.0 * d[s]["greedy_acc_lenient"] for s in sets]
    models = sorted({d[s].get("model") for s in sets})
    return {"mean": sum(v) / 3, "per": v, "n": [d[s]["n"] for s in sets], "models": models,
            "abstain_prompt": [d[s].get("abstain_prompt") for s in sets]}

ARMS = {"own": ("own_a3", range(5)), "clean": ("own_clean_a3", range(4)), "rnd": ("ctrl_rnd_a5", range(3)),
        "lowp": ("ctrl_lowp_a5", range(3)), "guard": ("own_ag_a4", range(3)), "cln1": ("cln1_a6", range(5)),
        "rnd1": ("rnd1_a6", range(5))}

res = {"base": {"umwp": umwp("qwen3_4b_base"), "bench": bench("qwen3_4b_base_step0")}, "runs": {}}
problems = []
for arm, (inf, seeds) in ARMS.items():
    for s in seeds:
        run = name(inf, s)
        t = trainset(run)
        u = umwp(run); u_first = umwp(run, keep="first")
        b = bench(run + "_step60")
        exp_model = f"/dev/shm/ckpt_b/{run}/final"
        if b["models"] != [exp_model]: problems.append(f"{run}: bench model {b['models']}")
        if b["abstain_prompt"] != [0, 0, 0]: problems.append(f"{run}: bench abstain_prompt {b['abstain_prompt']}")
        if (u["nU"], u["nA"], u["n_lines"]) != (600, 600, 1200): problems.append(f"{run}: umwp counts {u['nU']},{u['nA']},{u['n_lines']}")
        if u["CA_k"] != u_first["CA_k"]: problems.append(f"{run}: dedup order matters")
        if t["last_step"] != 60 or not t["done"] or t["last_epoch"] != t["expected_epoch"]:
            problems.append(f"{run}: log step {t['last_step']} epoch {t['last_epoch']} vs {t['expected_epoch']} done {t['done']}")
        if t["cfg_data"] != f"{run}/{'admitted_clean.jsonl' if arm == 'clean' else 'admitted.jsonl'}":
            problems.append(f"{run}: data {t['cfg_data']}")
        if arm in ("cln1", "rnd1"):
            lin = [name("own_a3", s, 1), name("own_a3", s, 2), run]
            if t["cfg_model"] != f"/dev/shm/ckpt_b/{name('own_a3', s, 2)}/final": problems.append(f"{run}: init {t['cfg_model']}")
        else:
            lin = [name(inf, s, 1), name(inf, s, 2), run]
            for r in (2, 3):
                m = json.load(open(f"{RUNS}/{name(inf, s, r)}/config.json"))["model"]
                if m != f"/dev/shm/ckpt_b/{name(inf, s, r - 1)}/final": problems.append(f"{name(inf, s, r)}: init {m}")
        lt = [trainset(x) for x in lin]
        for x, q in zip(lin, lt):
            if q["last_step"] != 60 or not q["done"]: problems.append(f"{x}: lineage run not done")
        rec = {k: v for k, v in t.items() if k != "ids"}
        rec.update({"umwp": u, "bench": b, "lin_tdel": [q["n_tdel"] for q in lt], "lin_tdel_cum": sum(q["n_tdel"] for q in lt),
                    "lin_n": [q["n"] for q in lt]})
        if arm in ("cln1", "rnd1"):
            A = jl(f"{RUNS}/{name('own_a3', s)}/admitted.jsonl"); Aid = {a["id"] for a in A}
            lab = {v["id"]: v["verdict"] for v in json.load(open(f"{RUNS}/{name('own_a3', s)}/audit_admitted.json"))["verdicts"]}
            for f in glob.glob(f"{RUNS}/swap_audit_a5_s{s}/audit_parts/*.json"):
                for v in json.load(open(f))["verdicts"]: lab[v["id"]] = v["verdict"]
            rows = jl(f"{RUNS}/{run}/admitted.jsonl")
            props = [r for r in rows if r["kind"] == "proposed"]
            c = Counter(lab.get(r["id"], "MISSING") for r in props)
            rec["subset_of_own_r3"] = all(r["id"] in Aid for r in rows)
            rec["prop_verdicts"] = dict(c)
            rec["n_illposed_UA"] = c.get("UNSOLVABLE", 0) + c.get("AMBIGUOUS", 0)
        res["runs"][f"{arm}_s{s}"] = rec

for s in range(5):
    A = jl(f"{RUNS}/{name('own_a3', s)}/admitted.jsonl")
    lab = {v["id"]: v["verdict"] for v in json.load(open(f"{RUNS}/{name('own_a3', s)}/audit_admitted.json"))["verdicts"]}
    for f in glob.glob(f"{RUNS}/swap_audit_a5_s{s}/audit_parts/*.json"):
        for v in json.load(open(f))["verdicts"]: lab[v["id"]] = v["verdict"]
    c = Counter(lab.get(r["id"], "MISSING") for r in A if r["kind"] == "proposed")
    res["runs"][f"own_s{s}"]["prop_verdicts_full"] = dict(c)
    res["runs"][f"own_s{s}"]["n_illposed_UA"] = c.get("UNSOLVABLE", 0) + c.get("AMBIGUOUS", 0)

def get(k, m):
    r = res["runs"][k]
    return {"CA": r["umwp"]["CA"], "ACC_A": r["umwp"]["ACC_A"], "bench": r["bench"]["mean"], "n": r["n"],
            "tdel": r["n_tdel"], "ill": r.get("n_illposed_UA")}[m]

def pair(a, b, seeds, m):
    d = [get(f"{a}_s{s}", m) - get(f"{b}_s{s}", m) for s in seeds]
    n = len(d); mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n); t = TCRIT[n - 1]
    out = {"diffs": [round(x, 4) for x in d], "mean": mu, "sd": sd, "ci": [mu - t * se, mu + t * se], "neg": sum(x < 0 for x in d)}
    try:
        from scipy import stats
        out["p"] = float(2 * stats.t.sf(abs(mu / se), n - 1)) if se > 0 else None
    except Exception:
        out["p"] = None
    return out

res["paired"] = {}
for a, b, seeds in (("clean", "own", range(4)), ("rnd", "own", range(3)), ("guard", "own", range(3)),
                    ("cln1", "rnd1", range(5)), ("lowp", "own", range(3))):
    for m in ("CA", "bench", "ACC_A", "n", "tdel"):
        res["paired"][f"{a}-{b}:{m}"] = pair(a, b, seeds, m)
res["paired"]["cln1-rnd1:ill"] = pair("cln1", "rnd1", range(5), "ill")
res["arm_means"] = {arm: {m: sum(get(f"{arm}_s{s}", m) for s in seeds) / len(seeds) for m in ("CA", "ACC_A", "bench")}
                    for arm, (_, seeds) in ARMS.items()}
res["problems"] = problems
json.dump(res, open(f"{R}/paper_arr/analyses/check_attribution.json", "w"), indent=1)

b = res["base"]
print("BASE CA %.4f (%d/600) ACC_A %.2f bench %.2f lines %d" % (b["umwp"]["CA"], b["umwp"]["CA_k"], b["umwp"]["ACC_A"], b["bench"]["mean"], b["umwp"]["n_lines"]), b["bench"]["models"])
for k, r in res["runs"].items():
    print("%-8s n=%3d uq=%3d/%3d/%3d prop=%3d pass=%5.2f tdel=%3d cum=%3d lin_n=%s CA=%.4f(%d) ACC=%.2f bench=%.2f ep=%s/%s ill=%s %s" % (
        k, r["n"], r["uniq_prop_ws"], r["uniq_prop_raw"], r["uniq_prop_lower"], r["kinds"].get("proposed", 0), r["passes"],
        r["n_tdel"], r["lin_tdel_cum"], r["lin_n"], r["umwp"]["CA"], r["umwp"]["CA_k"], r["umwp"]["ACC_A"], r["bench"]["mean"],
        r["last_epoch"], r["expected_epoch"], r.get("n_illposed_UA"), r.get("prop_verdicts", r.get("prop_verdicts_full", ""))))
for k, p in res["paired"].items():
    print("%-20s mean=%+.4f ci=[%+.4f,%+.4f] neg=%d p=%s diffs=%s" % (k, p["mean"], p["ci"][0], p["ci"][1], p["neg"],
          None if p["p"] is None else round(p["p"], 4), p["diffs"]))
print("arm means", {a: {m: round(v, 4) for m, v in d.items()} for a, d in res["arm_means"].items()})
print("PROBLEMS", problems)
