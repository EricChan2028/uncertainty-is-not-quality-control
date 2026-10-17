"""A7-5 reads. Prints one JSON object; safe at any time (an incomplete input is reported as missing and never read).
Seeds: ids 0-3 (2 and 3 only replace abandoned ids, section 4.1); an abandoned seed is reported and never counts; at most two
seeds can count. Clock (section 8): T0 = 2026-09-24 15:10 CST.
RZ-P1, per seed: ill(it3) >= ill(it1) + 10 points, ill = (UNSOLVABLE + AMBIGUOUS) / verdicts of the audit of that
       iteration's sample (dirB/runs/rzero_qwen3_4b_Base_s<S>_it<i>/audit_admitted.json with a verdict for every sampled
       item). An iteration's audit counts only if its admitted set existed by T0 + 71 h and the audit was assembled by
       T0 + 84 h. The item HOLDS if it holds in >= 1 seed run (a seed counts once both its it1 and it3 audits count).
RZ-P2, per seed: CA(plain_rzero_s<S>_it3) > CA(plain_qwen3_4b_base); CA and the 1,200-id completeness rule as in
       dirA3/code/umwp_read.py; the final dump must be complete by T0 + 72 h, and a seed run in RZ-P1-only mode
       (projected-finish rule) has no RZ-P2. HOLDS if it holds in every seed run (a seed counts once its dump counts).
Secondary and descriptive reads are printed next to them and never change the two verdicts.   python3 rz_read.py"""
import json, os, re
R = "/home/tione/notebook/rsi_rl"; U = f"{R}/dirA4/results/umwp"; X = f"{R}/ext/rzero_runs"; ILL = ("UNSOLVABLE", "AMBIGUOUS")
T0 = 1790233800; ADMITTED_BY = T0 + 71 * 3600; P2_BY = T0 + 72 * 3600; AUDIT_BY = T0 + 84 * 3600


def is_numeric(label):  # same rule as rz_extract.py
    t = re.sub(r"\s|\$|\\[,!;]", "", str(label)).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    t = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", t)
    return bool(re.fullmatch(r"-?\d+(\.\d+)?|-?\d+/\d+|-?\\frac\{\d+\}\{\d+\}", t))


def jload(f):
    return json.load(open(f)) if os.path.exists(f) else None


def audit(S, it):
    d = f"{R}/dirB/runs/rzero_qwen3_4b_Base_s{S}_it{it}"
    if not os.path.exists(f"{d}/audit_admitted.json"):
        return None
    summ = jload(f"{X}/s{S}/rzero_qwen3_4b_Base_s{S}_solver_v{it}_summary.json") or {}
    t_adm = summ.get("admitted_parquet_mtime")
    if t_adm is None or t_adm > ADMITTED_BY:
        return {"complete": False, "reason": "admitted set missing from the summary or created after T0+71h", "admitted_mtime": t_adm}
    if os.path.getmtime(f"{d}/audit_admitted.json") > AUDIT_BY:
        return {"complete": False, "reason": "audit assembled after T0+84h"}
    rows = {}
    for l in open(f"{d}/admitted.jsonl"):
        r = json.loads(l); rows[r["id"]] = r
    v = [x for x in json.load(open(f"{d}/audit_admitted.json"))["verdicts"] if x["id"] in rows]
    if len(v) < len(rows):
        return {"complete": False, "n": len(v), "expected": len(rows)}
    n = len(v); c = {k: sum(x["verdict"] == k for x in v) for k in ("SOLVABLE",) + ILL}
    num = [x for x in v if is_numeric(rows[x["id"]].get("rz_label", ""))]
    return {"complete": True, "n": n, "counts": c, "ill": (c["UNSOLVABLE"] + c["AMBIGUOUS"]) / n, "uns_only": c["UNSOLVABLE"] / n,
            "n_numeric_label": len(num), "ill_numeric_label": sum(x["verdict"] in ILL for x in num) / len(num) if num else None}


def umwp(tag, by=None):
    f = f"{U}/{tag}.jsonl"
    if not os.path.exists(f):
        return None
    rows = {}
    for l in open(f, errors="replace"):
        try:
            r = json.loads(l); rows[r["id"]] = r
        except (ValueError, KeyError, TypeError):
            pass
    if len(rows) < 1200:
        return {"complete": False, "unique_parseable_ids": len(rows)}
    if by is not None and os.path.getmtime(f) > by:
        return {"complete": False, "reason": "dump completed after T0+72h"}
    p = list(rows.values()); u = [r for r in p if not r["answerable"]]; s = [r for r in p if r["answerable"]]
    return {"complete": True, "CA": sum(r["mode"] is not None and r["p_mode"] >= 0.5 for r in u) / len(u),
            "ACC_A": 100 * sum(r["correct"] for r in s) / len(s),
            "ABS_U": 100 * sum(r["abstain"] for r in u) / len(u), "ABS_A": 100 * sum(r["abstain"] for r in s) / len(s)}


ok = lambda m: bool(m) and m.get("complete")
out = {"per_seed": {}, "abandoned": {}, "not_started": []}
base = umwp("plain_qwen3_4b_base")
p1, p2 = {}, {}
for S in (0, 1, 2, 3):
    host = open(f"{X}/s{S}.host").read().split() if os.path.exists(f"{X}/s{S}.host") else None
    if host is None:
        out["not_started"].append(S); continue
    A = {it: audit(S, it) for it in (1, 2, 3)}
    mode = open(f"{X}/s{S}/mode").read().split()[0] if os.path.exists(f"{X}/s{S}/mode") else None
    e = {"host": host[0], "mode": mode, "audit": A, "pool_summaries": {},
         "finding_admitted_lt512": jload(f"{X}/s{S}/finding_admitted_lt512.json")}
    for it in (1, 2, 3):
        f = f"{X}/s{S}/rzero_qwen3_4b_Base_s{S}_solver_v{it}_summary.json"
        if os.path.exists(f):
            e["pool_summaries"][it] = json.load(open(f))
    if os.path.exists(f"{X}/s{S}.abandoned"):
        e["abandoned"] = open(f"{X}/s{S}.abandoned").read().strip()
        out["abandoned"][S] = e; continue
    if ok(A[1]) and ok(A[3]):
        dp = 100 * (A[3]["ill"] - A[1]["ill"])
        e["RZ-P1"] = {"ill_it1": A[1]["ill"], "ill_it3": A[3]["ill"], "diff_points": round(dp, 2), "holds": dp >= 10 - 1e-9}
        p1[S] = e["RZ-P1"]["holds"]
        e["secondary"] = {"unsolvable_only_diff_points": round(100 * (A[3]["uns_only"] - A[1]["uns_only"]), 2),
                          "numeric_label_subset_diff_points": round(100 * (A[3]["ill_numeric_label"] - A[1]["ill_numeric_label"]), 2)
                          if A[1]["ill_numeric_label"] is not None and A[3]["ill_numeric_label"] is not None else None}
    m = umwp(f"plain_rzero_s{S}_it3", by=P2_BY)
    if mode == "p1only":
        e["RZ-P2"] = {"not_run": "RZ-P1-only mode under the projected-finish rule (section 8)"}
    elif ok(m) and ok(base):
        e["RZ-P2"] = {"CA_base": base["CA"], "CA_final": m["CA"], "holds": m["CA"] > base["CA"],
                      "ACC_A_base": base["ACC_A"], "ACC_A_final": m["ACC_A"]}
        p2[S] = e["RZ-P2"]["holds"]
    e["descriptive_abstention_prompt"] = {"final": umwp(f"abs_rzero_s{S}_it3"), "base": umwp("abs_qwen3_4b_base")}
    e["exploratory_chat_template"] = {"final": umwp(f"plainchat_rzero_s{S}_it3"), "base": umwp("plainchat_qwen3_4b_base")}
    out["per_seed"][S] = e
out["counted_seed_ids"] = sorted(out["per_seed"])
if len(out["per_seed"]) > 2:
    out["ERROR"] = "more than two non-abandoned seed ids: the seed rule of section 4.1 was broken; no verdict"
    p1, p2 = {}, {}
out["RZ-P1"] = {"verdict": ("HOLDS" if any(p1.values()) else "FAILS") if p1 else "NOT RUN", "seeds_run": sorted(p1)}
out["RZ-P2"] = {"verdict": ("HOLDS" if all(p2.values()) else "FAILS") if p2 else "NOT RUN", "seeds_run": sorted(p2)}
print(json.dumps(out, indent=1))
