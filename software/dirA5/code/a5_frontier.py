"""A5 F-P1..F-P4 (registered 08:04, A5 J1F1 part (c)); v2 after a three-reviewer adversarial check (all flagged the v1
envelope, rule set, default points and F-P3 matching; fixed here BEFORE any run; v1 was never run).
Definitions (fixed before running):
- UMWP dumps: dirA4/results/umwp/{plain,abs}_<tag>.jsonl, exactly 600 unanswerable (U) + 600 answerable (A), G = 8.
  Twin dumps: dirB/results/hp/{hp,hpabs}_<run>_shard*.jsonl, one shard set; U = the 412 validated unsolvable deleted twins,
  A = all intact twins (543), as in a4_rescue_read.py. Twin samples (32) are subsampled to G = 8 for every RULE (Monte Carlo,
  2,000 draws per item, Random(20260924); the count of abstaining samples is round(abstain * 32), assigned to None answers).
- "Answered" always requires at least one numeric answer among the samples used.
- DEFAULT operating points (the already-observed A4 definitions): UMWP CA_U = share of U whose plain-prompt numeric mode
  holds >= 0.5 of samples (umwp_read CA); UMWP FA_A = mean per-sample abstention share over A in the abstention-prompt dump
  (ABS_A). Twins CA_U = Q1u (plain hp dump, p_mode >= 0.5 on U; a4_rescue_read.q1u); twins FA_A = mean hpabs abstention
  share over all intact rows (a4_rescue_read.abst).
- PRIMARY own-r3 frontier (registered rule set, abstention-prompt dumps only): single sample (in expectation: P(a random
  sample is numeric)), modal-share threshold (answer iff numeric modal share >= tau, tau in {2/8..8/8}), abstention count
  (abstain iff >= k of 8 samples abstain, k = 1..8). Frontier = strict Pareto points (lower CA_U at higher FA_A), linear
  interpolation between neighbours; below the smallest FA_A: flat at the first Pareto point (conservative); above the largest:
  not read. SENSITIVITY frontier: the union with the same rules on the plain-prompt dump.
- F-P1: guard r3 default CA_U <= primary frontier(own r3) at the guard's default FA_A - 0.05, in >= 2 of 3 Qwen seeds on UMWP
  AND >= 2 of 3 on twins. Phi guard seeds 0-1: exploratory.
- F-P2: guard FA_A <= own FA_A + 0.05 (default points) on intact twins AND UMWP answerable, in >= 4 of 5 guard lineages,
  and no guard lineage with FA_A > 0.25.
- F-P3: oracle-tuned tau only (abstention-prompt modal share, tau grid above) on a 50% dev split of UMWP ids
  (Random(20260924) shuffle of the sorted ids; dev = even positions): mix the two thresholds that bracket the guard's dev
  FA_A so that own's dev FA_A matches it exactly (w), score the same mixture on test; holds if |CA_U(guard, test) -
  CA_U(own, test)| <= 0.05 in >= 2 of 3 Qwen seeds; not read if the guard's dev FA_A is outside own's tau range.
- F-P4: AUROC of the plain-prompt numeric modal share for separating A (higher) from U on UMWP, base vs own r3; holds if it
  falls by >= 0.05 in >= 8 of 10 own lineages. Inclusive thresholds use a 1e-9 tolerance.
- Descriptive: default points and frontier points for base, own r3, guard r3, clean r3, loop_uv r4 (where dumps exist).
python3 a5_frontier.py"""
import glob, json, os, random, re
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; UM = f"{R}/dirA4/results/umwp"; H = f"{R}/dirB/results/hp"
EPS = 1e-9
V = json.load(open(f"{R}/dirA/results/twin_validation_probe543.json"))
UNS = {(str(v["fam"]), int(v["task"])) for v in V["verdicts"] if v["verdict"] == "UNSOLVABLE"}
assert len(UNS) == 412 and len({(str(v["fam"]), int(v["task"])) for v in V["verdicts"]}) == 543
TAUS = [k / 8 for k in range(2, 9)]


def nshare(ans):
    c = Counter(a for a in ans if a is not None and a != "unknown")
    return (c.most_common(1)[0][1] if c else 0) / len(ans)


def load_umwp(tag, prompt):
    f = f"{UM}/{prompt}_{tag}.jsonl"
    if not os.path.exists(f): return None
    rows = {}
    for l in open(f):
        try: r = json.loads(l)
        except ValueError: continue
        if "abstain" not in r or "answers" not in r or len(r["answers"]) != 8: return None
        rows[r["id"]] = r
    U = [r for r in rows.values() if not r["answerable"]]; A = [r for r in rows.values() if r["answerable"]]
    if len(U) != 600 or len(A) != 600: return None
    return {i: {"U": not r["answerable"], "samples": [("NUM", a) if a is not None else ("NONE", None) for a in r["answers"]],
                "abst": r["abstain"], "m": nshare(r["answers"]), "p_mode": r["p_mode"], "mode": r["mode"]} for i, r in rows.items()}


def load_twins(run, prompt):
    fs = glob.glob(f"{H}/{prompt}_{run}_shard*.jsonl")
    sets = {re.search(r"of(\d+)\.jsonl$", f).group(1) for f in fs}
    if len(sets) != 1: return None
    N = int(sets.pop())
    if len(fs) != N: return None
    rows = {}
    for f in fs:
        for l in open(f):
            try: r = json.loads(l)
            except ValueError: continue
            if "answers" not in r or (prompt == "hpabs" and "abstain" not in r): return None
            rows[(str(r["fam"]), int(r["task"]), r["form"])] = r
    U = [r for k, r in rows.items() if k[2] == "deleted" and k[:2] in UNS]; A = [r for k, r in rows.items() if k[2] == "intact"]
    if len(U) != 412 or len(A) != 543: return None
    out = {}
    for r in U + A:
        ans = r["answers"]; nabs = round(r.get("abstain", 0) * len(ans)) if prompt == "hpabs" else 0
        lab, k = [], 0
        for a in ans:
            if a is not None: lab.append(("NUM", a))
            elif k < nabs: lab.append(("ABS", None)); k += 1
            else: lab.append(("NONE", None))
        out[(str(r["fam"]), int(r["task"]), r["form"])] = {"U": r["form"] == "deleted", "samples": lab, "abst": r.get("abstain", 0), "p_mode": r["p_mode"], "m": nshare(ans)}
    return out


def subsample_draws(items, G=8, D=2000, seed=20260924):
    """For twins: per item, D random 8-subsets (as label lists)."""
    rng = random.Random(seed); out = {}
    for k in sorted(items):
        s = items[k]["samples"]
        out[k] = [rng.sample(s, G) for _ in range(D)] if len(s) > G else [s]
    return out


def draw_stats(samples_list):
    """Per draw: (n_numeric, modal numeric count, n_abstain, G)."""
    out = []
    for s in samples_list:
        nums = [a for t, a in s if t == "NUM"]
        out.append((len(nums), Counter(nums).most_common(1)[0][1] if nums else 0, sum(1 for t, _ in s if t == "ABS"), len(s)))
    return out


def rule_answer_prob(stats, rule, param):
    """Probability (over draws) that the rule answers; answering needs >= 1 numeric sample."""
    n = 0.0
    for nn, mc, na, g in stats:
        if nn == 0: continue
        if rule == "single": n += nn / g
        elif rule == "tau": n += 1.0 if mc / g >= param - EPS else 0.0
        elif rule == "k": n += 0.0 if na >= param - EPS else 1.0
    return n / len(stats)


def curve(items, draws=None, umwp_abs=True):
    """Rule points (name, FA_A, CA_U) on one dump. For UMWP (G = 8) abstaining samples are marked from the abstain share."""
    U = [k for k in items if items[k]["U"]]; A = [k for k in items if not items[k]["U"]]
    def samp(k):
        if draws is not None: return draws[k]
        s = items[k]["samples"]; nabs = round(items[k]["abst"] * 8) if umwp_abs else 0; out, j = [], 0
        for t, a in s:
            if t == "NONE" and j < nabs: out.append(("ABS", None)); j += 1
            else: out.append((t, a))
        return [out]
    st = {k: draw_stats(samp(k)) for k in items}
    pts = []
    for rule, p in [("single", None)] + [("tau", t) for t in TAUS] + [("k", k) for k in range(1, 9)]:
        ca = sum(rule_answer_prob(st[k], rule, p) for k in U) / len(U)
        fa = sum(1 - rule_answer_prob(st[k], rule, p) for k in A) / len(A)
        pts.append((f"{rule}{'' if p is None else p}", fa, ca))
    return pts


def pareto(pts):
    env, best = [], float("inf")
    for _, x, y in sorted(pts, key=lambda p: (p[1], p[2])):
        if y < best - 1e-12: env.append((x, y)); best = y
    return env


def frontier_at(pts, fa):
    env = pareto(pts)
    if not env: return None
    if fa <= env[0][0] + EPS: return env[0][1]
    if fa > env[-1][0] + EPS: return None
    for (x0, y0), (x1, y1) in zip(env, env[1:]):
        if x0 - EPS <= fa <= x1 + EPS:
            return y0 + (y1 - y0) * (fa - x0) / (x1 - x0)
    return None


def umwp_default(plain, absd, ids=None):
    ids = ids or set(plain)
    U = [i for i in ids if plain[i]["U"]]; A = [i for i in ids if not plain[i]["U"]]
    ca = sum(1 for i in U if plain[i]["mode"] is not None and plain[i]["p_mode"] >= 0.5 - EPS) / len(U)
    fa = sum(absd[i]["abst"] for i in A) / len(A)
    return fa, ca


def twin_default(hp, hpabs):
    U = [k for k in hp if hp[k]["U"]]; A = [k for k in hpabs if not hpabs[k]["U"]]
    ca = sum(1 for k in U if hp[k]["p_mode"] >= 0.5 - EPS) / len(U)
    fa = sum(hpabs[k]["abst"] for k in A) / len(A)
    return fa, ca


def auroc(items):
    A = [i["m"] for i in items.values() if not i["U"]]; U = [i["m"] for i in items.values() if i["U"]]
    s = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in A for y in U)
    return s / (len(A) * len(U))


res = {"F_P1": [], "F_P1_phi_exploratory": [], "F_P2": [], "F_P3": [], "F_P4": [], "descriptive": {}}
def f1_for(own, grd, key):
    for bench in ("umwp", "twins"):
        if bench == "umwp":
            op, oa, gp, ga = load_umwp(own, "plain"), load_umwp(own, "abs"), load_umwp(grd, "plain"), load_umwp(grd, "abs")
            if not (op and oa and gp and ga):
                res[key].append({"own": own, "bench": bench, "read": None, "why": "dump missing/incomplete"}); print(f"{key} {own} {bench}: not read"); continue
            gfa, gca = umwp_default(gp, ga)
            prim = curve(oa); sens = prim + curve(op, umwp_abs=False)
        else:
            oh, oa, gh, ga = load_twins(own, "hp"), load_twins(own, "hpabs"), load_twins(grd, "hp"), load_twins(grd, "hpabs")
            if not (oh and oa and gh and ga):
                res[key].append({"own": own, "bench": bench, "read": None, "why": "dump missing/incomplete"}); print(f"{key} {own} {bench}: not read"); continue
            gfa, gca = twin_default(gh, ga)
            prim = curve(oa, draws=subsample_draws(oa)); sens = prim + curve(oh, draws=subsample_draws(oh))
        fr, frs = frontier_at(prim, gfa), frontier_at(sens, gfa)
        holds = None if fr is None else gca <= fr - 0.05 + EPS
        res[key].append({"own": own, "guard": grd, "bench": bench, "guard_FA_A": gfa, "guard_CA_U": gca, "own_frontier_CA_U": fr,
                         "own_frontier_CA_U_sensitivity": frs, "holds": holds, "pareto": pareto(prim)})
        print(f"{key} {own} {bench}: guard (FA_A {gfa:.3f}, CA_U {gca:.3f}) vs own frontier {fr} (sensitivity {frs}) -> {holds}")
for s in range(3):
    f1_for(f"qwen3_4b_Base_own_a3_s{s}_r3", f"qwen3_4b_Base_own_ag_a4_s{s}_r3", "F_P1")
for s in range(2):
    f1_for(f"phi4mini_own_a4_s{s}_r3", f"phi4mini_own_ag_a4_s{s}_r3", "F_P1_phi_exploratory")
lin = [(f"qwen3_4b_Base_own_a3_s{s}_r3", f"qwen3_4b_Base_own_ag_a4_s{s}_r3") for s in range(3)] + [(f"phi4mini_own_a4_s{s}_r3", f"phi4mini_own_ag_a4_s{s}_r3") for s in range(2)]
for own, grd in lin:
    oa, ga, ot, gt = load_umwp(own, "abs"), load_umwp(grd, "abs"), load_twins(own, "hpabs"), load_twins(grd, "hpabs")
    if not (oa and ga and ot and gt):
        res["F_P2"].append({"guard": grd, "read": None}); print(f"F-P2 {grd}: not read"); continue
    fu_o = sum(v["abst"] for v in oa.values() if not v["U"]) / 600; fu_g = sum(v["abst"] for v in ga.values() if not v["U"]) / 600
    ft_o = sum(v["abst"] for v in ot.values() if not v["U"]) / 543; ft_g = sum(v["abst"] for v in gt.values() if not v["U"]) / 543
    ok = fu_g <= fu_o + 0.05 + EPS and ft_g <= ft_o + 0.05 + EPS
    res["F_P2"].append({"guard": grd, "umwp_FA_A_own_guard": [fu_o, fu_g], "twins_FA_A_own_guard": [ft_o, ft_g], "holds": ok, "max_FA_A": max(fu_g, ft_g)})
    print(f"F-P2 {grd}: UMWP FA_A own {fu_o:.3f} guard {fu_g:.3f}; twins own {ft_o:.3f} guard {ft_g:.3f} -> {ok}")
for s in range(3):
    own, grd = f"qwen3_4b_Base_own_a3_s{s}_r3", f"qwen3_4b_Base_own_ag_a4_s{s}_r3"
    oa, gp, ga = load_umwp(own, "abs"), load_umwp(grd, "plain"), load_umwp(grd, "abs")
    if not (oa and gp and ga):
        res["F_P3"].append({"seed": s, "read": None}); print(f"F-P3 s{s}: not read"); continue
    ids = sorted(ga); rng = random.Random(20260924); rng.shuffle(ids)
    dev, test = set(ids[0::2]), set(ids[1::2])
    gfa_dev = umwp_default(gp, ga, dev)[0]; gfa_t, gca_t = umwp_default(gp, ga, test)
    sub = lambda d, S: {k: v for k, v in d.items() if k in S}
    tau_dev = [p for p in curve(sub(oa, dev)) if p[0].startswith("tau")]; tau_test = {p[0]: p for p in curve(sub(oa, test)) if p[0].startswith("tau")}
    tau_dev.sort(key=lambda p: p[1])
    lo = [p for p in tau_dev if p[1] <= gfa_dev + EPS]; hi = [p for p in tau_dev if p[1] >= gfa_dev - EPS]
    if not lo or not hi:
        res["F_P3"].append({"seed": s, "read": None, "why": "guard dev FA_A outside own tau range", "guard_dev_FA_A": gfa_dev}); print(f"F-P3 s{s}: not read (outside range)"); continue
    p0, p1 = lo[-1], hi[0]
    w = 0.0 if p1[1] == p0[1] else (gfa_dev - p0[1]) / (p1[1] - p0[1])
    ca_o = (1 - w) * tau_test[p0[0]][2] + w * tau_test[p1[0]][2]; fa_o = (1 - w) * tau_test[p0[0]][1] + w * tau_test[p1[0]][1]
    ok = abs(gca_t - ca_o) <= 0.05 + EPS
    res["F_P3"].append({"seed": s, "rules": [p0[0], p1[0]], "w": w, "guard_dev_FA_A": gfa_dev, "guard_test": [gfa_t, gca_t], "own_tuned_test": [fa_o, ca_o], "holds": ok})
    print(f"F-P3 s{s}: mix {p0[0]}/{p1[0]} w={w:.2f}; test guard (FA {gfa_t:.3f}, CA {gca_t:.3f}) vs own tuned (FA {fa_o:.3f}, CA {ca_o:.3f}) -> {ok}")
bases = {"qwen3_4b": "qwen3_4b_base", "qwen3_8b": "base_Qwen3-8B-Base", "phi4mini": "base_Phi-4-mini-instruct"}
runs = [("qwen3_4b", f"qwen3_4b_Base_own_a3_s{s}_r3") for s in range(5)] + [("qwen3_8b", f"qwen3_8b_Base_own_a3_s{s}_r3") for s in range(3)] + [("phi4mini", f"phi4mini_own_a4_s{s}_r3") for s in range(2)]
bA = {}
for k, v in bases.items():
    d = load_umwp(v, "plain"); bA[k] = auroc(d) if d else None
for fam, run in runs:
    d = load_umwp(run, "plain")
    if d is None or bA[fam] is None:
        res["F_P4"].append({"run": run, "read": None}); print(f"F-P4 {run}: not read"); continue
    a = auroc(d); ok = a <= bA[fam] - 0.05 + EPS
    res["F_P4"].append({"run": run, "auroc_base": bA[fam], "auroc_r3": a, "holds": ok})
    print(f"F-P4 {run}: AUROC base {bA[fam]:.3f} -> r3 {a:.3f} -> {ok}")
for tag in ["qwen3_4b_base"] + [f"qwen3_4b_Base_{a}_s{s}_r{r}" for a, r in (("own_a3", 3), ("own_ag_a4", 3), ("own_clean_a3", 3), ("loop_uv_a4", 4)) for s in range(3)]:
    p, a = load_umwp(tag, "plain"), load_umwp(tag, "abs")
    if p and a:
        res["descriptive"][tag] = {"default": umwp_default(p, a), "pareto_abs": pareto(curve(a))}
def verdict(lst, need, of):
    rd = [x for x in lst if x.get("holds") is not None]
    return None if len(rd) < of else sum(x["holds"] for x in rd) >= need
p1u = [x for x in res["F_P1"] if x.get("bench") == "umwp"]; p1t = [x for x in res["F_P1"] if x.get("bench") == "twins"]
vu, vt = verdict(p1u, 2, 3), verdict(p1t, 2, 3)
p2 = [x for x in res["F_P2"] if "holds" in x]
res["verdicts"] = {"F_P1": None if vu is None or vt is None else (vu and vt),
                   "F_P2": None if len(p2) < 5 else (sum(x["holds"] for x in p2) >= 4 and all(x["max_FA_A"] <= 0.25 + EPS for x in p2)),
                   "F_P3": verdict(res["F_P3"], 2, 3), "F_P4": verdict(res["F_P4"], 8, 10)}
print("verdicts:", res["verdicts"])
os.makedirs(f"{R}/dirA5/results", exist_ok=True)
json.dump(res, open(f"{R}/dirA5/results/frontier_read.json", "w"), indent=1, default=str)
