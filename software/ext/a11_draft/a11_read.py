"""A11 read (registration A11_registration_2026-09-26.md in this directory): P1-P3 and the descriptive reads on the corrected
abstention-prompt dumps absfix_<tag>.jsonl, next to the same reads on the defective abs_<tag>.jsonl dumps (for the
old-vs-new differences and for a reproduction check against the stored outputs of the registered reads).

No metric is re-implemented. The registered read scripts are executed UNCHANGED (sha256 checked), split at a fixed marker the
way dirA5/code/a5_frontier_b.py already splits a5_frontier.py: definitions first, then the loader is wrapped, then the body.
The wrapper changes only which file an abstention-prompt tag opens (abs_<tag> -> absfix_<tag>); each script's final
json.dump line (and a5_frontier.py's os.makedirs) is not executed, so no registered output is touched; their prints are
captured into this JSON.
  dirA3/code/umwp_read.py      load() [1,200-id rule], metrics() [CA, ACC_A, ABS_U, ABS_A, ACC_A_abs], body [M, HX1-HX4]
  dirA3/code/umwp_read_c.py    load(), metrics(), body [HX5, HX6]
  dirA5/code/a6_e_read.py      load() [1,552-id rule], ns(), body [per set x seed CA / ABS_U, E1, E2]
  dirA5/code/a5_frontier.py    nshare, load_umwp [600 U + 600 A, 8 answers], curve, pareto, frontier_at, umwp_default, ...,
                               body [F-P1 .. F-P4, descriptive default points and abstention-prompt Pareto sets]
  dirA5/code/a5_frontier_b.py  ca_at(), body [FB1-FB3, descriptive CA_U at FA_A = 0.05]
Mapping to A11: P1 <- umwp_read M[...]["ABS_U"]; P2 <- a6_e_read res["values"][set_s]["ABS_<arm>"] with its own E2 cell
test; P3 <- a5_frontier res["F_P1"] (UMWP rows): own_frontier_CA_U vs guard_CA_U. "Abstention-prompt CA" (CA_abs) = the same
registered metrics() / a6 body with the plain-prompt slot pointed at the abstention-prompt dump (the umwp_read L29 / a6_e_read
L25 CA expression on that dump, as ext/rzero_design/rz_read.py L64 does). The only new formula is ABS_A on the answerable
MiP items (the a6_e_read L26 expression with `not r["answerable"]` -> `r["answerable"]`), marked as such.
A new dump counts only when (a) the registered loader accepts it and (b) its sidecar absfix_<tag>.a11.json (written by
a11_launch.sh) has state "complete", its first event shows an empty dump, and every event names the sha256 of the
registered probe. Otherwise the cell is NOT RUN (reason reported, with the launchers' status lines and, per cell, whether
each registered loader accepts the file); a NOT RUN cell never counts as a pass.
P2 is the registered E2 aggregation of a6_e_read.py (L29 cell test, L31-32 >= 3 of 4 seeds per set, L33-36 >= 2 of 3 sets),
taken verbatim from that script's own verdict on the corrected dumps; the 12-cell count is descriptive.
The confirmatory read is run ONCE, at the latest 12 h after T0 (ext/a11_runs/launch_epoch). It writes one JSON object to
ext/a11_runs/a11_read.json and refuses to overwrite an existing file (O_EXCL); the author then hashes that file into
docs/registration_hashes.txt at once.   /usr/bin/python3 a11_read.py [--out PATH]   (the registered read uses the default)"""
import argparse, contextlib, glob, hashlib, io, json, os, sys, time
R = "/home/tione/notebook/rsi_rl"; HERE = os.path.dirname(os.path.abspath(__file__)); X = f"{R}/ext/a11_runs"
UM = f"{R}/dirA4/results/umwp"; A6D = f"{R}/dirA5/results/a6_sweep"; REG = f"{R}/docs/registration_hashes.txt"
EPS = 1e-9; CUTOFF_H = 12
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{X}/a11_read.json"); ARGS = ap.parse_args()
if os.path.exists(ARGS.out):
    sys.exit(f"refusing to run: {ARGS.out} exists (the confirmatory read is run once)")
T_START = time.time()
SRC = {"umwp_read": ("dirA3/code/umwp_read.py", "36a9f1239e11156cda2f38f06c755614e3171dd91f140fbf068a283d05a2b51a", "names = {"),
       "umwp_read_c": ("dirA3/code/umwp_read_c.py", "4240ae0fffb92c1e2e97ab9d4d3e617df4110128e90be55afc7cbd34d315b70c", "res = {"),
       "a6_e_read": ("dirA5/code/a6_e_read.py", "c1ab4803037545e299275a537ac19a34b3cccdf4a0da6dcb9f7181ac5321c4ee", "res = {"),
       "a5_frontier": ("dirA5/code/a5_frontier.py", "afdcafac2e7f45e9c12e30a435ecf7f5bd3798e455f3c6931bf19a9d4497762a", "res = {"),
       "a5_frontier_b": ("dirA5/code/a5_frontier_b.py", "fba8ff2be089c93f7863e900f9b3ff01d0d7d7535a3c2cbbb2b479f78d9048da", "EPS = ns[")}
STORED = {"umwp_read": "dirA4/results/umwp_read.json", "umwp_read_c": "dirA4/results/umwp_read_c.json",
          "a6_e_read": "dirA5/results/a6_e_read.json", "a5_frontier": "dirA5/results/frontier_read.json",
          "a5_frontier_b": "dirA5/results/frontier_b_read.json"}


def sha(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


for _k, (_rel, _s, _) in SRC.items():
    assert sha(f"{R}/{_rel}") == _s, f"{_rel} is not the registered version"
REGTXT = open(REG).read()
PROBE_SHA = {"umwp": sha(f"{HERE}/umwp_probe_absfix.py"), "a6": sha(f"{HERE}/a6_probe_absfix.py")}
PROBE_REGISTERED = {b: s in REGTXT for b, s in PROBE_SHA.items()}
Q = "qwen3_4b_Base"
TIER1_UMWP = (["rzero_s0_it3", "rzero_s1_it3", "qwen3_4b_base"] + [f"{Q}_own_a3_s{s}_r3" for s in range(5)]
              + [f"{Q}_own_clean_a3_s{s}_r3" for s in range(4)] + [f"{Q}_own_ag_a4_s{s}_r3" for s in range(3)])
TIER1_A6 = [f"{Q}_{a}_s{s}_r3" for a in ("own_a3", "own_clean_a3") for s in range(4)]
TIER2_UMWP = ([f"{Q}_rescue_uv_a4_s{s}_r3" for s in (0, 1, 2, 4)] + [f"{Q}_rescue_abs_a4_s{s}_r3" for s in range(3)]
              + [f"{Q}_loop_uv_a4_s{s}_r4" for s in (0, 2)] + [f"{Q}_coev_ag_a4_s{s}_r3" for s in range(2)]
              + [f"{Q}_tg_abs4_a5_s{s}_r4" for s in range(3)] + [f"{Q}_tg_t4_a5_s1_r4"])
SETS = ("sum", "mip_gsm8k", "mip_math")
_side = {}


def sidecar(bench, name):
    """(accepted, reason) for a corrected dump; the registered loader still has to accept the file itself."""
    k = (bench, name)
    if k in _side:
        return _side[k]
    f = f"{UM if bench == 'umwp' else A6D}/absfix_{name}.a11.json"
    if not os.path.exists(f):
        v = (False, "no A11 sidecar (never launched, still running, or checkpoint missing)")
    else:
        try:
            m = json.load(open(f))
        except ValueError:
            m = None
        if m is None:
            v = (False, "sidecar unreadable")
        elif m.get("state") != "complete":
            v = (False, f"sidecar state {m.get('state')!r}")
        elif not PROBE_REGISTERED[bench]:
            v = (False, "the absfix probe in this directory is not in docs/registration_hashes.txt")
        elif m.get("probe_sha256") != PROBE_SHA[bench]:
            v = (False, "dump written by a probe other than the registered one")
        elif (not m.get("history") or m["history"][0].get("n_ids") != 0
              or any(e.get("probe_sha256") != PROBE_SHA[bench] for e in m["history"])):
            v = (False, "dump did not start empty under the registered probe (rows of unknown origin)")
        else:
            copies = {json.dumps({n: x[1] for n, x in e["model_fingerprint"].get("files", {}).items() if n.endswith(".safetensors")})
                      for e in m.get("history", []) if e.get("model_fingerprint")}
            v = (True, "complete" + ("" if len(copies) <= 1 else " (WARNING: resumed on checkpoint copies with differing weights)"))
    _side[k] = v
    return v


def gate(bench, P, name):
    return P == "abs" or sidecar(bench, name)[0]


def run_registered(key, patch=None, body=True):
    """exec the registered script: definitions [0, marker), patch(ns), body [marker, final dump line); tail not executed."""
    rel, _, stop = SRC[key]
    src = open(f"{R}/{rel}").read()
    assert src.count(stop) == 1, (rel, stop)
    i = src.index(stop)
    j = src.rindex("os.makedirs(") if key == "a5_frontier" else src.rindex("json.dump(")
    tail = [l for l in src[j:].splitlines() if l.strip()]
    assert j > i and tail and all(l.startswith(("json.dump(", "os.makedirs(")) for l in tail), (rel, tail)
    ns = {"__name__": f"a11_{key}"}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src[:i], rel, "exec"), ns)
        if patch:
            patch(ns)
        if body:
            exec(compile("\n" * src[:i].count("\n") + src[i:j], rel, "exec"), ns)
    return ns, buf.getvalue()


def p_load(bench, P, mode="abs"):
    """umwp_read / umwp_read_c / a6_e_read load(tag). mode 'abs': abs_<n> -> <P>_<n>; mode 'ca': the plain-prompt slot reads
    <P>_<n> (abstention-prompt CA) and the abstention slot is empty."""
    def patch(ns):
        orig = ns["load"]

        def load(tag):
            if mode == "abs":
                if tag.startswith("abs_"):
                    n = tag[4:]
                    return orig(f"{P}_{n}") if gate(bench, P, n) else None
                return orig(tag)
            if tag.startswith("plain_"):
                n = tag[6:]
                return orig(f"{P}_{n}") if gate(bench, P, n) else None
            return None
        ns["load"] = load
    return patch


def p_frontier(P, inner=False):
    """a5_frontier load_umwp(tag, prompt): prompt 'abs' -> <P>; a5_frontier_b keeps a5_frontier's namespace in its `ns`."""
    def patch(ns):
        tgt = ns["ns"] if inner else ns
        orig = tgt["load_umwp"]

        def load_umwp(tag, prompt):
            if prompt == "abs":
                return orig(tag, P) if gate("umwp", P, tag) else None
            return orig(tag, prompt)
        tgt["load_umwp"] = load_umwp
    return patch


def norm(x):
    return json.loads(json.dumps(x, default=str))


def one_prompt(P):
    r, logs = {}, {}
    ur, logs["umwp_read"] = run_registered("umwp_read", p_load("umwp", P))
    r["umwp_read"] = {"metrics": ur["M"], "HX3": ur["hx3"], "HX1": ur["hx1"], "HX2": ur["hx2"], "HX4": ur["hx4"]}
    uc, logs["umwp_read_c"] = run_registered("umwp_read_c", p_load("umwp", P))
    r["umwp_read_c"] = uc["res"]
    ae, logs["a6_e_read"] = run_registered("a6_e_read", p_load("a6", P))
    r["a6_e_read"] = ae["res"]
    aec, _ = run_registered("a6_e_read", p_load("a6", P, "ca"))
    fr, logs["a5_frontier"] = run_registered("a5_frontier", p_frontier(P))
    r["a5_frontier"] = fr["res"]
    fb, logs["a5_frontier_b"] = run_registered("a5_frontier_b", p_frontier(P, inner=True))
    r["a5_frontier_b"] = fb["out"]
    um, _ = run_registered("umwp_read", p_load("umwp", P), body=False)          # metrics() only
    umc, _ = run_registered("umwp_read", p_load("umwp", P, "ca"), body=False)   # metrics() with the plain slot -> abs dump
    per = {}
    for t in TIER1_UMWP + TIER2_UMWP:
        m, c = um["metrics"](t), umc["metrics"](t)
        a, p = fr["load_umwp"](t, "abs"), fr["load_umwp"](t, "plain")
        per[t] = {"ABS_U": m.get("ABS_U"), "ABS_A": m.get("ABS_A"), "ACC_A_abs": m.get("ACC_A_abs"), "CA_abs": c.get("CA"),
                  "CA_abs_by_cat": c.get("CA_by_cat"), "CA_plain": m.get("CA"), "ACC_A_plain": m.get("ACC_A"),
                  "CA_U_at_FA_A_0.05": fb["ca_at"](t, "umwp", 0.05),
                  "default_point": dict(zip(("FA_A", "CA_U"), fr["umwp_default"](p, a))) if (a and p) else None,
                  "pareto_abs": fr["pareto"](fr["curve"](a)) if a else None}
    r["umwp_per_tag"] = per
    a6 = {}
    for t in TIER1_A6:
        arm = "own_clean_a3" if "_own_clean_a3_" in t else "own_a3"; s = int(t.split("_s")[-1].split("_")[0])
        rows = ae["load"](f"abs_{t}")
        for st in SETS:
            v, vc = r["a6_e_read"]["values"].get(f"{st}_s{s}", {}), aec["res"]["values"].get(f"{st}_s{s}", {})
            A = [x for x in rows.values() if x["source"] == st and x["answerable"]] if rows else []
            a6[f"{t}|{st}"] = {"ABS_U": v.get(f"ABS_{arm}"), "CA_abs": vc.get(f"CA_{arm}"),
                               "ABS_A_new_formula": (sum(x["abstain"] for x in A) / len(A)) if A else None}
    r["a6_per_tag_set"] = a6
    return norm(r), logs


def loader_acceptance():
    """per cell: does each registered loader, unpatched, accept the corrected file absfix_<tag>.jsonl (None: no file)?"""
    um, _ = run_registered("umwp_read", None, body=False)
    ae, _ = run_registered("a6_e_read", None, body=False)
    fr, _ = run_registered("a5_frontier", None, body=False)
    acc = {}
    for t in TIER1_UMWP + TIER2_UMWP:
        acc[("umwp", t)] = ({"umwp_read.load (1,200 ids)": um["load"](f"absfix_{t}") is not None,
                             "a5_frontier.load_umwp (600 U + 600 A, 8 answers)": fr["load_umwp"](t, "absfix") is not None}
                            if os.path.exists(f"{UM}/absfix_{t}.jsonl") else None)
    for t in TIER1_A6:
        acc[("a6", t)] = ({"a6_e_read.load (1,552 ids)": ae["load"](f"absfix_{t}") is not None}
                          if os.path.exists(f"{A6D}/absfix_{t}.jsonl") else None)
    return acc


def verdict(cells, need):
    v = [c["holds"] for c in cells]
    if any(x is False for x in v):
        return "FAILS"
    if len(v) == need and all(x is True for x in v):
        return "HOLDS"
    return "NOT_READ"


def predictions(r):
    M = r["umwp_read"]["metrics"]; p1 = []
    for s in range(4):
        o, c = M.get(f"own_a3_s{s}_r3", {}), M.get(f"own_clean_a3_s{s}_r3", {})
        if "ABS_U" in o and "ABS_U" in c:
            d = (o["ABS_U"] - c["ABS_U"]) / 100
            p1.append({"seed": s, "ABS_U_own": o["ABS_U"] / 100, "ABS_U_clean": c["ABS_U"] / 100, "diff": d, "holds": abs(d) <= 0.05 + EPS})
        else:
            p1.append({"seed": s, "holds": None, "why": "own or clean corrected dump not read"})
    p2c = []   # descriptive only: the 12 set x seed cells (the E2 cell test of a6_e_read.py L29, verbatim)
    for st in SETS:
        for s in range(4):
            v = r["a6_e_read"]["values"].get(f"{st}_s{s}", {})
            if "ABS_own_a3" in v and "ABS_own_clean_a3" in v:
                p2c.append({"set": st, "seed": s, "ABS_own": v["ABS_own_a3"], "ABS_clean": v["ABS_own_clean_a3"],
                            "diff": v["ABS_own_a3"] - v["ABS_own_clean_a3"],
                            "within_0.05": abs(v["ABS_own_a3"] - v["ABS_own_clean_a3"]) <= 0.05 + 1e-9})
            else:
                p2c.append({"set": st, "seed": s, "within_0.05": None, "why": "own or clean corrected dump not read"})
    e2 = r["a6_e_read"]["verdicts"]["E2"]   # a6_e_read.py L31-36 on the corrected dumps: True / False / None
    rd = [x for x in p2c if x["within_0.05"] is not None]
    p2 = {"rule": "registered E2 aggregation (a6_e_read.py L29 cell test, L31-32 >= 3 of 4 seeds per set, L33-36 >= 2 of 3 sets), verbatim",
          "per_set": r["a6_e_read"]["E2"], "verdict": {True: "HOLDS", False: "FAILS"}.get(e2, "NOT_READ"),
          "descriptive_12_cells": {"cells": p2c, "n_read": len(rd), "n_within_0.05": sum(x["within_0.05"] for x in rd),
                                   "all_12_within_0.05": all(x["within_0.05"] for x in rd) if len(rd) == 12 else None}}
    p3 = []
    rows = {x["own"]: x for x in r["a5_frontier"]["F_P1"] if x.get("bench") == "umwp"}
    for s in range(3):
        x = rows.get(f"{Q}_own_a3_s{s}_r3", {})
        fr_, g = x.get("own_frontier_CA_U"), x.get("guard_CA_U")
        if fr_ is None or g is None:
            p3.append({"seed": s, "holds": None, "why": x.get("why", "own frontier undefined at the guard's FA_A")})
        else:
            p3.append({"seed": s, "guard_FA_A": x["guard_FA_A"], "guard_CA_U_plain": g, "own_frontier_CA_U": fr_,
                       "own_frontier_CA_U_sensitivity": x.get("own_frontier_CA_U_sensitivity"), "margin": g - fr_,
                       "holds": fr_ <= g - 0.05 + EPS, "registered_F_P1_cell": x.get("holds")})
    return {"P1": {"cells": p1, "verdict": verdict(p1, 4)}, "P2": p2, "P3": {"cells": p3, "verdict": verdict(p3, 3)}}


def diffs(new, old):
    if isinstance(new, dict) and isinstance(old, dict):
        return {k: diffs(new.get(k), old.get(k)) for k in new if k in old}
    if isinstance(new, (int, float)) and isinstance(old, (int, float)) and not isinstance(new, bool) and not isinstance(old, bool):
        return new - old
    return None


def compare(a, b, path="", out=None):
    """leaves that differ between a recomputed old read and its stored output (tolerance 1e-9), and keys present on one
    side only, reported in both directions"""
    out = out if out is not None else {"n_compared": 0, "n_differ": 0, "max_abs_dev": 0.0, "examples": [],
                                       "n_only_in_recomputed": 0, "only_in_recomputed": [], "n_only_in_stored": 0, "only_in_stored": []}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b:
                compare(a[k], b[k], f"{path}/{k}", out)
            else:
                out["n_only_in_recomputed"] += 1; out["only_in_recomputed"] = (out["only_in_recomputed"] + [f"{path}/{k}"])[:50]
        for k in b:
            if k not in a:
                out["n_only_in_stored"] += 1; out["only_in_stored"] = (out["only_in_stored"] + [f"{path}/{k}"])[:50]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["n_differ"] += 1; out["examples"].append(f"{path}: length {len(a)} vs stored {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out)
    elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        out["n_compared"] += 1; out["n_differ"] += 1
        out["examples"].append(f"{path}: type {type(a).__name__} vs stored {type(b).__name__}")
    elif isinstance(a, bool) or isinstance(b, bool) or a is None or b is None or isinstance(a, str):
        out["n_compared"] += 1
        if a != b:
            out["n_differ"] += 1; out["examples"].append(f"{path}: {a} vs stored {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out["n_compared"] += 1; d = abs(a - b); out["max_abs_dev"] = max(out["max_abs_dev"], d)
        if d > 1e-9:
            out["n_differ"] += 1; out["examples"].append(f"{path}: {a} vs stored {b}")
    out["examples"] = out["examples"][:25]
    return out


def launcher_status():
    st = {}
    for f in sorted(glob.glob(f"{X}/status_*.tsv")):
        for l in open(f):
            p = l.rstrip("\n").split("\t")
            if len(p) >= 5:
                st.setdefault(f"{p[2]}:{p[3]}", {})[p[1]] = {"state": p[4], "detail": p[5] if len(p) > 5 else "", "epoch": p[0]}
    return st


new, logs_new = one_prompt("absfix")
old, logs_old = one_prompt("abs")
LS = launcher_status()
LA = loader_acceptance()
cells = {}
for tier, bench, tags in (("tier1", "umwp", TIER1_UMWP), ("tier1", "a6", TIER1_A6), ("tier2", "umwp", TIER2_UMWP)):
    for t in tags:
        ok, why = sidecar(bench, t)
        f = f"{UM if bench == 'umwp' else A6D}/absfix_{t}.jsonl"
        hosts = LS.get(f"{bench}:{t}", {})
        missing = bool(hosts) and len(hosts) >= 2 and all(h["state"] == "missing_here" for h in hosts.values())
        la = LA.get((bench, t))
        if ok and la and all(la.values()):
            state = "RUN"
        elif ok and la and any(la.values()):
            state = "PARTIAL (rejected by " + ", ".join(k for k, v in la.items() if not v) + ")"
        elif ok:
            state = "NOT RUN (the registered loader rejects the dump)"
        else:
            state = "NOT RUN (checkpoint missing on both hosts)" if missing else "NOT RUN"
        cells[f"{bench}:{t}"] = {"tier": tier, "required": True, "state": state, "sidecar": why, "loader_accepted": la,
                                 "file_exists": os.path.exists(f), "launcher_status": hosts}
stored_repro = {}
for key, rel in STORED.items():
    if os.path.exists(f"{R}/{rel}"):
        s = json.load(open(f"{R}/{rel}"))
        mine = {"umwp_read": {k: old["umwp_read"][k] for k in ("metrics", "HX1", "HX2", "HX3", "HX4")},
                "umwp_read_c": old["umwp_read_c"], "a6_e_read": old["a6_e_read"],
                "a5_frontier": old["a5_frontier"], "a5_frontier_b": old["a5_frontier_b"]}[key]
        stored_repro[rel] = compare(mine, s)
try:
    T0 = int(open(f"{X}/launch_epoch").read().strip())
except (OSError, ValueError):
    T0 = None
clock = {"T0": T0, "T0_str": time.strftime("%F %T %Z", time.localtime(T0)) if T0 else None,
         "read_started": time.strftime("%F %T %Z", time.localtime(T_START)),
         "hours_after_T0": round((T_START - T0) / 3600, 3) if T0 else None,
         "within_cutoff": (T_START - T0 <= CUTOFF_H * 3600) if T0 else None, "cutoff_hours": CUTOFF_H}
out = {"registration": "A11 (corrected abstention-prompt re-run), read a11_read.py", "read_sha256": sha(os.path.abspath(__file__)),
       "clock": clock, "probe_sha256": PROBE_SHA, "probe_registered": PROBE_REGISTERED, "cells": cells,
       "confirmatory_corrected_prompt": predictions(new),
       "same_rules_on_defective_prompt_for_comparison_only": predictions(old),
       "descriptive": {
           "umwp_per_tag": {t: {"new": new["umwp_per_tag"][t], "old": old["umwp_per_tag"][t],
                                "new_minus_old": diffs(new["umwp_per_tag"][t], old["umwp_per_tag"][t])} for t in new["umwp_per_tag"]},
           "a6_per_tag_set": {k: {"new": new["a6_per_tag_set"][k], "old": old["a6_per_tag_set"][k],
                                  "new_minus_old": diffs(new["a6_per_tag_set"][k], old["a6_per_tag_set"][k])} for k in new["a6_per_tag_set"]},
           "registered_rules_reapplied_to_corrected_dumps": {
               "E2_registered_aggregation": {"new": {"per_set": new["a6_e_read"]["E2"], "verdict": new["a6_e_read"]["verdicts"]["E2"]},
                                             "old": {"per_set": old["a6_e_read"]["E2"], "verdict": old["a6_e_read"]["verdicts"]["E2"]}},
               "HX3": {"new": new["umwp_read"]["HX3"], "old": old["umwp_read"]["HX3"]},
               "HX6": {"new": new["umwp_read_c"].get("HX6"), "old": old["umwp_read_c"].get("HX6")},
               "F_P1": {"new": new["a5_frontier"]["F_P1"], "old": old["a5_frontier"]["F_P1"]},
               "F_P2": {"new": new["a5_frontier"]["F_P2"], "old": old["a5_frontier"]["F_P2"]},
               "F_P3": {"new": new["a5_frontier"]["F_P3"], "old": old["a5_frontier"]["F_P3"]},
               "F_P1_phi_exploratory": {"new": new["a5_frontier"]["F_P1_phi_exploratory"], "old": old["a5_frontier"]["F_P1_phi_exploratory"],
                                        "note": "Phi-4-mini abstention dumps are not re-run: the new value is empty by construction"},
               "F_P4": {"new": new["a5_frontier"]["F_P4"], "old": old["a5_frontier"]["F_P4"], "note": "plain-prompt AUROC; unaffected"},
               "frontier_verdicts": {"new": new["a5_frontier"]["verdicts"], "old": old["a5_frontier"]["verdicts"]},
               "FB": {"new": new["a5_frontier_b"], "old": old["a5_frontier_b"]},
               "frontier_descriptive": {"new": new["a5_frontier"]["descriptive"], "old": old["a5_frontier"]["descriptive"]}},
           "note": "re-applications of registered rules are descriptive; the registered verdicts keep their labels"},
       "reproduction_of_stored_old_reads": stored_repro,
       "captured_prints": {"new": logs_new, "old": logs_old}}
fd = os.open(ARGS.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)   # never overwrites
with os.fdopen(fd, "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print(json.dumps({"wrote": ARGS.out, "sha256": sha(ARGS.out), "clock": clock,
                  "verdicts": {k: out["confirmatory_corrected_prompt"][k]["verdict"] for k in ("P1", "P2", "P3")},
                  "next": "append its sha256 line to docs/registration_hashes.txt now (the author does this)"}, indent=1))
