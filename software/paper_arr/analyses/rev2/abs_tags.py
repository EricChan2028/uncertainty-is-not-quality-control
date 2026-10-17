import json
d = json.load(open("/home/tione/notebook/rsi_rl/ext/a11_runs/a11_read.json"))
u = d["descriptive"]["umwp_per_tag"]
for t, v in u.items():
    n = v.get("new") or {}
    if not n:
        print(t, "EMPTY", list(v.keys())); continue
    dp = n.get("default_point") or {}
    f = lambda x: None if x is None else round(x, 4)
    print(f"{t:45s} ABS_U {f(n.get('ABS_U'))} ABS_A {f(n.get('ABS_A'))} CAplain {f(n.get('CA_plain'))} CA@5% {f(n.get('CA_U_at_FA_A_0.05'))} FA_def {f(dp.get('FA_A'))}")
print(json.dumps(d["confirmatory_corrected_prompt"]["P1"])[:2500])
