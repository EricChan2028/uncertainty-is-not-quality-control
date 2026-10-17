"""A8 (registered 23:16 24 Sep): (1) select the fixed item set; (2) create dirA3/code/umwp_probe_text.py = umwp_probe.py with
--ids (restrict to listed UMWP ids) and --outdir, saving the full completion texts. Everything else (prompt, sampling, seed,
answer extraction) is byte-identical. python3 a8_prep.py"""
import json, os, random
from collections import defaultdict, Counter
R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA5/results/a8"; os.makedirs(OUT, exist_ok=True)
# (1) items
rows = {}
for l in open(f"{R}/dirA4/results/umwp/plain_qwen3_4b_base.jsonl"):
    try: r = json.loads(l); rows[r["id"]] = r
    except ValueError: pass
def nshare(ans):
    c = Counter(a for a in ans if a is not None and a != "unknown"); return (c.most_common(1)[0][1] if c else 0) / len(ans)
BINS = [2, 3, 4, 5, 6]; hardA, U = defaultdict(list), defaultdict(list)
for i, r in sorted(rows.items()):
    k = round(8 * nshare(r["answers"]))
    if k not in BINS: continue
    if r["answerable"]:
        if r.get("ref") is None: continue
        gold = sum(1 for a in r["answers"] if a is not None and a == r["ref"]) / len(r["answers"])
        if gold <= 3 / 8 + 1e-9: hardA[k].append(i)
    else: U[k].append(i)
rng = random.Random(20260925); sel = {"hard_A": [], "U": [], "per_bin": {}}
for k in BINS:
    a = sorted(hardA[k]); u = sorted(U[k]); pu = rng.sample(u, min(len(a), len(u)))
    sel["hard_A"] += a; sel["U"] += sorted(pu); sel["per_bin"][k] = {"hard_A": len(a), "U_available": len(u), "U_drawn": len(pu)}
sel["n_hard_A"], sel["n_U"] = len(sel["hard_A"]), len(sel["U"])
json.dump(sel, open(f"{OUT}/items.json", "w"), indent=1)
print({k: v for k, v in sel.items() if k not in ("hard_A", "U")})
# (2) probe variant
src = open(f"{R}/dirA3/code/umwp_probe.py").read()
def rep(a, b):
    global src
    assert src.count(a) == 1, (a[:60], src.count(a)); src = src.replace(a, b)
rep('ap.add_argument("--n_uns", type=int, default=600)',
    'ap.add_argument("--n_uns", type=int, default=600)\nap.add_argument("--ids", required=True); ap.add_argument("--outdir", required=True)')
rep('R = "/home/tione/notebook/rsi_rl"; OUT = f"{R}/dirA4/results/umwp"; os.makedirs(OUT, exist_ok=True)',
    'R = "/home/tione/notebook/rsi_rl"; OUT = args.outdir; os.makedirs(OUT, exist_ok=True)')
rep('items = uns + [byid[i] for i in ans_ids]',
    'items = uns + [byid[i] for i in ans_ids]\n_keep = set(json.load(open(args.ids))["hard_A"] + json.load(open(args.ids))["U"])  # A8: restrict to the fixed item set\nitems = [x for x in items if x["id"] in _keep]')
rep('"correct": int(mode is not None and ref is not None and mode == ref), "answers": ans}) + "\\n")',
    '"correct": int(mode is not None and ref is not None and mode == ref), "answers": ans, "texts": texts, "finish": fin}) + "\\n")')
open(f"{R}/dirA3/code/umwp_probe_text.py", "w").write('"""A8 text-saving copy of umwp_probe.py (only --ids/--outdir and the saved texts differ)."""\n' + src)
print("probe variant written")
