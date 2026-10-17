# EXPLORATORY (post hoc, not registered): held-out calibration of the a5_frontier.py rule set on corrected (absfix) dumps.
import json, random, statistics
from collections import defaultdict
R = "/home/tione/notebook/rsi_rl"
src = open(f"{R}/dirA5/code/a5_frontier.py").read()
ns = {}
exec(src[:src.index("res = {")], ns)          # definitions only; nothing is written
load_umwp, curve, frontier_at, umwp_default, EPS = ns["load_umwp"], ns["curve"], ns["frontier_at"], ns["umwp_default"], ns["EPS"]
D = {x["id"]: x for x in (json.loads(l) for l in open(f"{R}/data_ext/UMWP/data/StandardDataset.jsonl"))}

def named_pareto(pts):
    env, best = [], float("inf")
    for name, x, y in sorted(pts, key=lambda p: (p[1], p[2])):
        if y < best - 1e-12: env.append((name, x, y)); best = y
    return env

def calibrate(pts, target):
    env = named_pareto(pts)
    if target <= env[0][1] + EPS: mix = [(env[0][0], 1.0)]
    elif target > env[-1][1] + EPS: mix = [(env[-1][0], 1.0)]
    else:
        mix = None
        for (n0, x0, y0), (n1, x1, y1) in zip(env, env[1:]):
            if x0 - EPS <= target <= x1 + EPS:
                w = (target - x0) / (x1 - x0); mix = [(n0, 1 - w), (n1, w)]; break
    cons = [p for p in env if p[1] <= target + EPS]
    det = [(cons[-1][0] if cons else env[0][0], 1.0)]
    return mix, det

def evaluate(pts, mix):
    d = {n: (x, y) for n, x, y in pts}
    return sum(w * d[n][0] for n, w in mix), sum(w * d[n][1] for n, w in mix)

def split(items, seed):
    U = sorted(i for i in items if items[i]["U"])
    strata = defaultdict(list)
    for u in U: strata[D[u]["source"]].append(u)
    rng = random.Random(seed); h = [set(), set()]
    for s in sorted(strata):
        us = sorted(strata[s]); rng.shuffle(us)
        for j, u in enumerate(us):
            a = D[u]["relevant_ids"][0]
            assert a in items and not items[a]["U"]
            h[j % 2].update((u, a))
    return h

def one(tag, target, seed):
    items = load_umwp(tag, "absfix")
    h = split(items, seed); out = []
    for c, e in ((0, 1), (1, 0)):
        cal = {k: v for k, v in items.items() if k in h[c]}; ev = {k: v for k, v in items.items() if k in h[e]}
        pc, pe = curve(cal), curve(ev)
        mix, det = calibrate(pc, target)
        out.append({"mix": mix, "det": det, "eval_mix": evaluate(pe, mix), "eval_det": evaluate(pe, det),
                    "insample_cal": frontier_at(pc, target), "n_cal": (sum(v["U"] for v in cal.values()), sum(not v["U"] for v in cal.values()))})
    return items, out

tags = {s: f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(3)}
guards = {s: f"qwen3_4b_Base_own_ag_a4_s{s}_r3" for s in range(3)}
res = {}
for s in range(3):
    gp, ga = load_umwp(guards[s], "plain"), load_umwp(guards[s], "absfix")
    gfa, gca = umwp_default(gp, ga)
    items = load_umwp(tags[s], "absfix")
    full = curve(items)
    for tname, target in (("guardFA", gfa), ("FA5", 0.05)):
        ins = frontier_at(full, target)
        _, dirs = one(tags[s], target, 20260927)
        ca_mix = statistics.mean(d["eval_mix"][1] for d in dirs); fa_mix = statistics.mean(d["eval_mix"][0] for d in dirs)
        ca_det = statistics.mean(d["eval_det"][1] for d in dirs); fa_det = statistics.mean(d["eval_det"][0] for d in dirs)
        rep = []
        for sd in range(200):
            _, dd = one(tags[s], target, 1000 + sd)
            rep.append((statistics.mean(d["eval_mix"][1] for d in dd), statistics.mean(d["eval_mix"][0] for d in dd),
                        statistics.mean(d["eval_det"][1] for d in dd), statistics.mean(d["eval_det"][0] for d in dd)))
        q = lambda xs: (round(statistics.mean(xs), 4), round(sorted(xs)[4], 4), round(sorted(xs)[194], 4))
        res[f"s{s}|{tname}"] = {"target_FA": round(target, 4), "guard_plain_CA": round(gca, 4), "insample_CA": round(ins, 4),
            "primary_split": {"CA_mix": round(ca_mix, 4), "FA_mix": round(fa_mix, 4), "CA_det": round(ca_det, 4), "FA_det": round(fa_det, 4),
                              "dirs": [{"mix": [(n, round(w, 3)) for n, w in d["mix"]], "det": d["det"][0][0], "eval_mix": [round(x, 4) for x in d["eval_mix"]],
                                        "eval_det": [round(x, 4) for x in d["eval_det"]], "insample_cal": round(d["insample_cal"], 4), "n_cal_UA": d["n_cal"]} for d in dirs]},
            "200_splits_mean_p2.5_p97.5": {"CA_mix": q([r[0] for r in rep]), "FA_mix": q([r[1] for r in rep]), "CA_det": q([r[2] for r in rep]), "FA_det": q([r[3] for r in rep])}}
        print(f"s{s} {tname}: target {target:.4f} insample {ins:.4f} | held-out mix CA {ca_mix:.4f} FA {fa_mix:.4f} | det CA {ca_det:.4f} FA {fa_det:.4f} | 200 splits mix CA {q([r[0] for r in rep])} FA {q([r[1] for r in rep])} det CA {q([r[2] for r in rep])} FA {q([r[3] for r in rep])}")
# base model at 5%
items = load_umwp("qwen3_4b_base", "absfix"); full = curve(items); ins = frontier_at(full, 0.05)
_, dirs = one("qwen3_4b_base", 0.05, 20260927)
rep = []
for sd in range(200):
    _, dd = one("qwen3_4b_base", 0.05, 1000 + sd)
    rep.append((statistics.mean(d["eval_mix"][1] for d in dd), statistics.mean(d["eval_mix"][0] for d in dd), statistics.mean(d["eval_det"][1] for d in dd), statistics.mean(d["eval_det"][0] for d in dd)))
q = lambda xs: (round(statistics.mean(xs), 4), round(sorted(xs)[4], 4), round(sorted(xs)[194], 4))
print(f"base FA5: insample {ins:.4f} | held-out mix CA {statistics.mean(d['eval_mix'][1] for d in dirs):.4f} FA {statistics.mean(d['eval_mix'][0] for d in dirs):.4f} det CA {statistics.mean(d['eval_det'][1] for d in dirs):.4f} FA {statistics.mean(d['eval_det'][0] for d in dirs):.4f} | 200 splits mix CA {q([r[0] for r in rep])} FA {q([r[1] for r in rep])} det CA {q([r[2] for r in rep])} FA {q([r[3] for r in rep])}")
print(json.dumps(res, indent=0)[:6000])
sources = defaultdict(int)
for u in (i for i in items if items[i]["U"]): sources[D[u]["source"]] += 1
print("U by source", dict(sources))
