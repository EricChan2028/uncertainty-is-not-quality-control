import json, os, re, glob
R = "/home/tione/notebook/rsi_rl/dirB/runs"
out = []
for d in sorted(glob.glob(f"{R}/qwen3_4b_Base_*_s*_r*")):
    if d.endswith(".claim") or not os.path.exists(f"{d}/config.json"): continue
    cfg = json.load(open(f"{d}/config.json"))
    data = cfg.get("data")
    n = None; comp = {"prop": 0, "int": 0, "del": 0}
    if data and os.path.exists(data):
        for l in open(data):
            x = json.loads(l); i = str(x.get("id", ""))
            comp["del" if i.startswith("trd") else "int" if i.startswith("tri") else "prop"] += 1
        n = sum(comp.values())
    last_ep = None
    if os.path.exists(f"{d}/log.jsonl"):
        for l in open(f"{d}/log.jsonl"):
            try: x = json.loads(l)
            except ValueError: continue
            if "epoch" in x and "step" in x: last_ep = (x["step"], x["epoch"])
    adm = f"{d}/admitted.jsonl"
    nadm = sum(1 for _ in open(adm)) if os.path.exists(adm) else None
    out.append((os.path.basename(d), os.path.relpath(data, "/home/tione/notebook/rsi_rl") if data else None, n, comp, nadm, cfg.get("steps"), cfg.get("qps"), cfg.get("arm"), last_ep))
for o in out:
    if o[2] != o[4] or "clean" in o[0] or "sub" in o[0]:
        print(o)
print("n_runs", len(out))
