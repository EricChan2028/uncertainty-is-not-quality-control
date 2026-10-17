"""Independent verification of parsing.py (EXPLORATORY, read-only on inputs). Writes only paper_arr/analyses/check_parsing.json.
python3 check_parsing.py
"""
import glob, json, math, random, re
from collections import Counter, defaultdict

R = "/home/tione/notebook/rsi_rl"
OUT = f"{R}/paper_arr/analyses/check_parsing.json"
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
G = 8
IPC = {"UNSOLVABLE", "AMBIGUOUS"}
res = {"a": {}, "b": {}}

lab = {}
for f in sorted(glob.glob(f"{R}/dirA3/results/pool_audit/qwen_s0_r1/parts/*.json")):
    for v in json.load(open(f))["verdicts"]:
        if v["id"] in lab and lab[v["id"]] != v["verdict"]:
            print("DUP-CONFLICT", v["id"])
        lab[v["id"]] = v["verdict"]
rel = {}
for l in open(f"{R}/release_v1/fixed_pool_audits/qwen_s0_r1.jsonl"):
    x = json.loads(l); rel[x["id"]] = x["opus_verdict"]
res["a"]["labels"] = {"n": len(lab), "counts": dict(Counter(lab.values())),
                      "release_agree": sum(rel.get(i) == v for i, v in lab.items()), "release_n": len(rel)}

data = {}
for ck in CKS:
    rows = [json.loads(l) for l in open(f"{R}/dirA3/results/guard_diag/qwen_s0/{ck}/guard_records.jsonl")]
    its = []
    for r in rows:
        k8 = r["p_mode"] * G; u8 = r["abstain"] * G
        assert abs(k8 - round(k8)) < 1e-9 and abs(u8 - round(u8)) < 1e-9
        c = lab.get(r["id"], "NOLABEL") if r["kind"] == "proposed" else r["kind"]
        its.append(dict(id=r["id"], c=c, k=round(k8), u=round(u8), mode=r["mode"], adm=r["admit"]))
    data[ck] = its


def inband(k, m):
    return m > 0 and 4 * k >= m and 4 * k <= 3 * m


reg = {}
desc = {}
for ck in CKS:
    P = [x for x in data[ck] if x["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")]
    assert len(P) == 1158, len(P)
    A = [x for x in P if inband(x["k"], 8)]
    assert {x["id"] for x in A} == {x["id"] for x in P if x["adm"]}
    reg[ck] = {"n": len(A), "ip": sum(x["c"] in IPC for x in A) / len(A)}
    d = {}
    for c in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE", "tracer_intact", "tracer_deleted"):
        X = [x for x in data[ck] if x["c"] == c]
        n = len(X)
        d[c] = {"n": n, "modeNone": sum(x["k"] == 0 for x in X), "unk_share": round(sum(x["u"] for x in X) / (8 * n), 5),
                "LB": round(sum(8 - x["u"] for x in X if x["k"] == 0) / (8 * n), 5),
                "UB": round(sum(8 - x["k"] - x["u"] for x in X) / (8 * n), 5),
                "inband": round(sum(inband(x["k"], 8) for x in X) / n, 4)}
    d["proposals_unk_share"] = round(sum(x["u"] for x in P) / (8 * len(P)), 5)
    desc[ck] = d
res["a"]["registered"] = {ck: {"n": reg[ck]["n"], "ip_pct": round(100 * reg[ck]["ip"], 2)} for ck in CKS}
res["a"]["descriptives"] = desc

prox = {}
for i, ck in enumerate(CKS):
    L = [json.loads(l) for l in open(f"{R}/dirB/runs/qwen3_4b_Base_own_a3_s0_r{i+1}/log.jsonl")]
    L = [x for x in L if "fmt_rate" in x]
    s1 = [x for x in L if x["step"] == 1]; assert len(s1) == 1; s1 = s1[0]
    s10 = [x for x in L if 1 <= x["step"] <= 10]; assert len(s10) == 10
    prox[ck] = {"s1_none": 1 - s1["fmt_rate"], "s1_trunc": s1["trunc"], "s1_ndel": s1["n_del"], "s1_nseq": s1["n_seqs"],
                "s10_none": sum(1 - x["fmt_rate"] for x in s10) / 10, "s10_trunc": sum(x["trunc"] for x in s10) / 10,
                "s10_ndel_max": max(x["n_del"] for x in s10)}
res["a"]["proxy"] = {ck: {k: round(v, 4) for k, v in prox[ck].items()} for ck in CKS}


def p_adm(k, u, th, unk_in_denom=False):
    if k == 0:
        return 0.0
    ns = 8 - k - u
    tot = 0.0
    for j in range(ns + 1):
        m = (8 - j) if unk_in_denom else (8 - u - j)
        if inband(k, m):
            tot += math.comb(ns, j) * th ** j * (1 - th) ** (ns - j)
    return tot


def comp(P, thS, thI, unk=False):
    eS = sum(p_adm(x["k"], x["u"], thS, unk) for x in P if x["c"] == "SOLVABLE")
    eI = sum(p_adm(x["k"], x["u"], thI, unk) for x in P if x["c"] in IPC)
    return eI / (eS + eI), eS + eI


def seqd(v):
    return {"pct": [round(100 * x, 1) for x in v], "drift_pts": round(100 * (v[-1] - v[0]), 1),
            "monotone": all(b >= a - 1e-12 for a, b in zip(v, v[1:]))}


scen = defaultdict(list)
ns_ = defaultdict(list)
for ck in CKS:
    P = [x for x in data[ck] if x["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")]
    a0 = sum(8 - x["u"] for x in P if x["k"] == 0)
    sS = sum(8 - x["k"] - x["u"] for x in P if x["k"] > 0 and x["c"] == "SOLVABLE")
    sI = sum(8 - x["k"] - x["u"] for x in P if x["k"] > 0 and x["c"] in IPC)
    N8 = 8 * len(P)
    for nm, q in (("s1", prox[ck]["s1_none"]), ("s10", prox[ck]["s10_none"])):
        B = q * N8 - a0
        th = B / (sS + sI)
        s, n = comp(P, th, th); scen[f"cal_uniform_{nm}"].append(s); ns_[f"cal_uniform_{nm}"].append(round(n, 1))
        s, n = comp(P, th, th, True); scen[f"cal_uniform_{nm}_unk_in_denom"].append(s)
        tS = min(1.0, B / sS); tI = min(1.0, B / sI)
        scen[f"cal_S_only_{nm}"].append(comp(P, tS, 0.0)[0]); scen[f"cal_IP_only_{nm}"].append(comp(P, 0.0, tI)[0])
        scen[f"thetaS_{nm}"].append(round(B / sS, 3))
        scen[f"thetaU_{nm}"].append(round(th, 4))
    for th in (0.0, 0.10, 0.25, 0.5):
        scen[f"uniform_{th}"].append(comp(P, th, th)[0])
        scen[f"implied_none_{th}"].append(round((a0 + th * (sS + sI)) / N8, 4))
    scen["S_only_0.5"].append(comp(P, 0.5, 0.0)[0]); scen["IP_only_0.5"].append(comp(P, 0.0, 0.5)[0])
drift = {k: seqd(v) for k, v in scen.items() if not k.startswith(("thetaS", "thetaU", "implied"))}
drift["registered"] = seqd([reg[ck]["ip"] for ck in CKS])
res["a"]["redenominated"] = drift
res["a"]["theta"] = {k: v for k, v in scen.items() if k.startswith(("thetaS", "thetaU"))}
res["a"]["implied_none"] = {k: v for k, v in scen.items() if k.startswith("implied")}
res["a"]["expected_n_cal_s1"] = ns_["cal_uniform_s1"]
adv = {}
for nm, (kS, kI) in {"s1": ("cal_S_only_s1", "cal_IP_only_s1"), "s10": ("cal_S_only_s10", "cal_IP_only_s10"),
                     "th0.5": ("S_only_0.5", "IP_only_0.5")}.items():
    b = max(scen[kS][0], scen[kI][0], scen["uniform_0.0"][0]); e = min(scen[kS][-1], scen[kI][-1], scen["uniform_0.0"][-1])
    adv[nm] = {"base_max": round(100 * b, 1), "r5_min": round(100 * e, 1), "drift": round(100 * (e - b), 1)}
res["a"]["class_level_adversarial"] = adv

rng = random.Random(1)
mc = []
for ck in CKS:
    P = [x for x in data[ck] if x["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")]
    a0 = sum(8 - x["u"] for x in P if x["k"] == 0)
    sl = sum(8 - x["k"] - x["u"] for x in P if x["k"] > 0)
    th = (prox[ck]["s1_none"] * 8 * len(P) - a0) / sl
    vals = []
    for _ in range(300):
        nI = nA = 0
        for x in P:
            if x["k"] == 0:
                continue
            j = sum(rng.random() < th for _ in range(8 - x["k"] - x["u"]))
            if inband(x["k"], 8 - x["u"] - j):
                nA += 1; nI += x["c"] in IPC
        vals.append(nI / nA)
    vals.sort()
    mc.append([round(100 * sum(vals) / len(vals), 2), round(100 * vals[7], 2), round(100 * vals[292], 2)])
res["a"]["mc_cal_uniform_s1_mean_p2.5_p97.5"] = mc


def cost_exit(k, u):
    ns = 8 - k - u
    for j in range(1, ns + 1):
        if not inband(k, 8 - u - j):
            return j
    return None


def cost_enter(k, u):
    if k == 0:
        return None
    ns = 8 - k - u
    for j in range(1, ns + 1):
        if inband(k, 8 - u - j):
            return j
    return None


def worst(P, budget, maximize_ip):
    tgt_out = (lambda c: c == "SOLVABLE") if maximize_ip else (lambda c: c in IPC)
    tgt_in = (lambda c: c in IPC) if maximize_ip else (lambda c: c == "SOLVABLE")
    A = [x for x in P if inband(x["k"], 8)]
    I0 = sum(x["c"] in IPC for x in A); S0 = len(A) - I0
    outc = sorted(c for c in (cost_exit(x["k"], x["u"]) for x in A if tgt_out(x["c"])) if c is not None)
    inc = sorted(c for c in (cost_enter(x["k"], x["u"]) for x in P if not inband(x["k"], 8) and tgt_in(x["c"])) if c is not None)
    best = None
    for na in range(len(inc) + 1):
        ca = sum(inc[:na])
        if ca > budget:
            break
        left = budget - ca; nr = 0
        for c in outc:
            if left >= c:
                left -= c; nr += 1
            else:
                break
        if maximize_ip:
            I, S = I0 + na, S0 - nr
        else:
            I, S = I0 - nr, S0 + na
        f = I / (I + S) if I + S else float("nan")
        if best is None or (f > best[0] if maximize_ip else f < best[0]):
            best = (f, na, nr, I + S)
    return best


wc = {}
for nm in ("s1", "s10"):
    out = {}
    for ck, mx in (("base", True), ("r5", False)):
        P = [x for x in data[ck] if x["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE")]
        a0 = sum(8 - x["u"] for x in P if x["k"] == 0)
        B = int(prox[ck][f"{nm}_none"] * 8 * len(P) - a0)
        f, na, nr, n = worst(P, B, mx)
        out[ck] = {"budget_samples": B, "ip_pct": round(100 * f, 1), "n_entered": na, "n_exited": nr, "n_admitted": n}
    out["drift_pts"] = round(out["r5"]["ip_pct"] - out["base"]["ip_pct"], 1)
    wc[nm] = out
res["a"]["item_level_worst_case"] = wc
res["a"]["admitted_k_hist"] = {ck: {c: dict(sorted(Counter(x["k"] for x in data[ck] if x["c"] in cs and inband(x["k"], 8)).items()))
                                    for c, cs in (("S", {"SOLVABLE"}), ("IP", IPC))} for ck in ("base", "r5")}

NUM = re.compile(r"-?[\d.,]+(/\d+)?")
nn = []
for ck in CKS:
    A = [x for x in data[ck] if x["c"] in ("SOLVABLE", "AMBIGUOUS", "UNSOLVABLE") and inband(x["k"], 8)]
    keep = [x for x in A if NUM.fullmatch(str(x["mode"]))]
    nn.append((len(A) - len(keep), sum(x["c"] in IPC for x in keep) / len(keep)))
res["a"]["nonnumeric"] = {"n_dropped": [a for a, _ in nn], **seqd([b for _, b in nn])}

UD = f"{R}/dirA4/results/umwp"
tags = {"base": "qwen3_4b_base", **{f"r3s{s}": f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)},
        **{f"r5s{s}": f"qwen3_4b_Base_own_a3_s{s}_r5" for s in range(3)}}
feat = {}
for key, t in tags.items():
    rows = {}
    for l in open(f"{UD}/plain_{t}.jsonl"):
        r = json.loads(l); rows[r["id"]] = r
    U = [r for r in rows.values() if r["answerable"] is False]
    A = [r for r in rows.values() if r["answerable"] is True]
    assert len(U) == 600 and len(A) == 600 and len(rows) == 1200
    for r in rows.values():
        assert len(r["answers"]) == 8
        assert abs(r["nonanswer"] - sum(a is None for a in r["answers"]) / 8) < 1e-9
    f = {}
    for r in U:
        nums = [a for a in r["answers"] if a is not None]
        c = Counter(nums).most_common(1)
        c1 = c[0][1] if c else 0
        f[r["id"]] = {"reg": r["mode"] is not None and r["p_mode"] >= 0.5, "n8": c1 >= 4,
                      "po": len(nums) > 0 and 2 * c1 >= len(nums), "po4": len(nums) >= 4 and 2 * c1 >= len(nums)}
    feat[key] = f
    m = {"na_U": sum(r["nonanswer"] for r in U) / 600, "na_A": sum(r["nonanswer"] for r in A) / 600,
         "any_na_U": sum(r["nonanswer"] > 0 for r in U) / 600, "all_na_U": sum(r["nonanswer"] == 1 for r in U),
         "all_na_A": sum(r["nonanswer"] == 1 for r in A)}
    for v in ("reg", "n8", "po", "po4"):
        m[f"CA_{v}"] = sum(x[v] for x in f.values()) / 600
    m["dis_reg_n8"] = sum(x["reg"] != x["n8"] for x in f.values()); m["dis_reg_po"] = sum(x["reg"] != x["po"] for x in f.values())
    m["ACC_A"] = 100 * sum(r["correct"] for r in A) / 600
    acc2 = 0
    for r in A:
        nums = [a for a in r["answers"] if a is not None]
        c = Counter(nums).most_common(1)
        acc2 += bool(c) and r["ref"] is not None and c[0][0] == r["ref"]
    m["ACC_A_numeric_mode"] = 100 * acc2 / 600
    res["b"][key] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}


def boot(fb, fo, v, nb=2000, seed=12345):
    ids = sorted(fb)
    d = [int(fo[i][v]) - int(fb[i][v]) for i in ids]
    g = random.Random(seed)
    bs = sorted(sum(d[g.randrange(len(d))] for _ in d) / len(d) for _ in range(nb))
    return [round(sum(d) / len(d), 4), round(bs[50], 4), round(bs[1949], 4)]


res["b"]["paired"] = {f"r3s{s}": {v: boot(feat["base"], feat[f"r3s{s}"], v) for v in ("reg", "po")} for s in range(5)}
json.dump(res, open(OUT, "w"), indent=1, default=str)
print("ok")
