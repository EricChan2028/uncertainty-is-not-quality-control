import json, os, re, glob
from collections import defaultdict
R = "/home/tione/notebook/rsi_rl/dirB/runs"
arms = ["own_a3", "own_clean_a3", "own_ag_a4", "ctrl_rnd_a5", "coev_a4", "coev_ag_a4", "own_azr_a3", "cln1_a6", "rnd1_a6", "own_sub_a3", "ctrl_lowp_a5", "own_tg_a3", "loop_uv_a4", "rescue_uv_a4", "rescue_abs_a4"]
rows = []
for arm in arms:
    for d in sorted(glob.glob(f"{R}/qwen3_4b_Base_{arm}_s*_r*")):
        if d.endswith(".claim"): continue
        m = re.search(r"_s(\d+)_r(\d+)$", d)
        if not m: continue
        s, r = int(m.group(1)), int(m.group(2))
        f = f"{d}/admitted.jsonl"
        n = sum(1 for _ in open(f)) if os.path.exists(f) else None
        cfg = json.load(open(f"{d}/config.json")) if os.path.exists(f"{d}/config.json") else {}
        gs = json.load(open(f"{d}/guard_stats.json")) if os.path.exists(f"{d}/guard_stats.json") else {}
        steps, qps = cfg.get("steps"), cfg.get("qps")
        last_ep, last_step, done = None, None, False
        if os.path.exists(f"{d}/log.jsonl"):
            for l in open(f"{d}/log.jsonl"):
                try: x = json.loads(l)
                except ValueError: continue
                if "step" in x and "epoch" in x: last_ep, last_step = x["epoch"], x["step"]
                if x.get("done"): done = True
        comp = None
        if n is not None and os.path.exists(f):
            kinds = defaultdict(int)
            for l in open(f):
                x = json.loads(l)
                k = x.get("kind") or x.get("src") or x.get("type") or ("tracer" if str(x.get("id", "")).startswith(("trd", "tri")) else "prop")
                idv = str(x.get("id", x.get("qid", "")))
                if idv.startswith("trd"): kinds["del"] += 1
                elif idv.startswith("tri"): kinds["int"] += 1
                else: kinds["prop"] += 1
            comp = dict(kinds)
        rows.append({"arm": arm, "seed": s, "round": r, "n_train": n, "comp": comp, "steps": steps, "qps": qps,
                     "passes": round(steps * qps / n, 2) if n and steps and qps else None, "last_step": last_step, "last_epoch": last_ep, "done": done,
                     "gs_n_admitted": gs.get("n_admitted"), "gs_prop_admit": gs.get("proposed_admit"), "gs_int": gs.get("tracer_intact_admit"), "gs_del": gs.get("tracer_deleted_admit"), "gs_proposed_n": gs.get("proposed_n")})
for x in rows: print(json.dumps(x))
