import json, os, re, glob
from collections import defaultdict
R = "/home/tione/notebook/rsi_rl/dirB/runs"
T = defaultdict(dict)
for d in sorted(glob.glob(f"{R}/qwen3_4b_Base_*_s*_r*")):
    if d.endswith(".claim") or not os.path.exists(f"{d}/config.json"): continue
    m = re.search(r"qwen3_4b_Base_(.+)_s(\d+)_r(\d+)$", d)
    arm, s, r = m.group(1), int(m.group(2)), int(m.group(3))
    cfg = json.load(open(f"{d}/config.json")); data = cfg["data"]
    comp = {"prop": 0, "int": 0, "del": 0}
    for l in open(data):
        i = str(json.loads(l).get("id", ""))
        comp["del" if i.startswith("trd") else "int" if i.startswith("tri") else "prop"] += 1
    n = sum(comp.values())
    gs = json.load(open(f"{d}/guard_stats.json")) if os.path.exists(f"{d}/guard_stats.json") else {}
    T[arm][(s, r)] = dict(n=n, passes=cfg["steps"] * cfg["qps"] / n, comp=comp, band=gs.get("n_admitted"), proposed=gs.get("proposed_n"))
for arm in ["own_a3", "own_clean_a3", "own_ag_a4", "ctrl_rnd_a5", "coev_a4", "coev_ag_a4", "own_azr_a3", "cln1_a6", "rnd1_a6", "loop_uv_a4"]:
    rs = sorted({r for (_, r) in T[arm]})
    for r in rs:
        cells = sorted((s, v) for (s, rr), v in T[arm].items() if rr == r)
        ns = [v["n"] for _, v in cells]; ps = [v["passes"] for _, v in cells]
        dl = [v["comp"]["del"] for _, v in cells]; it = [v["comp"]["int"] for _, v in cells]; pr = [v["comp"]["prop"] for _, v in cells]
        bd = [v["band"] for _, v in cells]
        print(f"{arm:13s} r{r} seeds {[s for s,_ in cells]} n {min(ns)}-{max(ns)} mean {sum(ns)/len(ns):.0f} | passes {min(ps):.2f}-{max(ps):.2f} mean {sum(ps)/len(ps):.2f} | prop {min(pr)}-{max(pr)} int {min(it)}-{max(it)} del {min(dl)}-{max(dl)} | band {bd}")
        print("      per-seed n:", {s: v["n"] for s, v in cells}, "passes:", {s: round(v["passes"], 2) for s, v in cells})
