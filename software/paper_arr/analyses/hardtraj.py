"""hardtraj (EXPLORATORY, post hoc, 26 Sep 2026): trajectory of hard answerable vs other answerable vs unanswerable UMWP items.
Read-only over dirA4/results/umwp/plain_<tag>.jsonl (G = 8 plain prompt, 600 answerable + 600 unanswerable).
Definitions (a6_matched.py where applicable):
- m = numeric modal share = count of the most common answer that is not None and not "unknown", divided by 8 (a6 nshare).
- band = 0.25 <= m <= 0.75 (k = 2..6 of 8); above = m > 0.75 (k >= 7); below = m < 0.25 (k <= 1).
- hard answerable (per family, fixed at the base model of that family) = answerable, ref parsed (not None), base gold accuracy
  (share of the 8 base samples equal to ref) <= 3/8. other answerable = the remaining answerable items (ref None included).
- acc_plurality = mean of the dump field correct (plurality mode == ref; this is the ACC_A convention);
  acc_sample = mean share of the 8 samples equal to ref (items with ref None skipped).
- CA (unanswerable only) = share with dump mode not None and p_mode >= 0.5. conf_rate = the same rule applied to answerable groups
  (descriptive; not called CA).
- Two populations: all = every item of the group; base_band = the group restricted to items in the band at base
  (base k = round(8 m) in 2..6, the a6 cells population). For base_band items, the checkpoint state is decomposed into
  in_band / above_correct (m > 0.75 and plurality correct: solved) / above_wrong (m > 0.75, plurality wrong; for U every
  confident answer is wrong) / below (m < 0.25).
- fresh_resample_check (Qwen3-4B only): the hard group is selected on the same 8 base samples it is scored on, so part of any
  later rise can be regression to the mean. dirA5/results/a8/texts/a8_<ckpt>.jsonl holds 8 FRESH plain-prompt samples for the
  112 base-band hard answerable items and 112 bin-matched unanswerable items (A8 item draw), at base and own s0-s2 r3/r5.
  The same decomposition is applied to those fresh samples.
python3 hardtraj.py
"""
import json, os
from collections import Counter
R = "/home/tione/notebook/rsi_rl"; UM = f"{R}/dirA4/results/umwp"; OUT = f"{R}/paper_arr/analyses/hardtraj.json"
A8 = f"{R}/dirA5/results/a8"
FAM = {
    "Qwen3-4B": ("qwen3_4b_base", [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)] + [f"qwen3_4b_Base_own_a3_s{s}_r5" for s in range(3)]),
    "Qwen3-8B": ("base_Qwen3-8B-Base", [f"qwen3_8b_Base_own_a3_s{s}_r3" for s in range(3)]),
    "Phi-4-mini": ("base_Phi-4-mini-instruct", [f"phi4mini_own_a4_s{s}_r3" for s in range(2)]),
}


def nshare(ans):
    c = Counter(a for a in ans if a is not None and a != "unknown")
    return (c.most_common(1)[0][1] if c else 0) / len(ans)


def read_rows(f):
    rows = {}
    for l in open(f):
        try: r = json.loads(l); rows[int(r["id"])] = r
        except ValueError: pass
    return rows


def load(tag):
    f = f"{UM}/plain_{tag}.jsonl"
    if not os.path.exists(f): return None, "missing"
    rows = read_rows(f)
    nU = sum(not r["answerable"] for r in rows.values()); nA = sum(r["answerable"] for r in rows.values())
    if nU != 600 or nA != 600: return None, f"counts U={nU} A={nA}"
    return rows, "ok"


def mean(x):
    x = list(x); return (sum(x) / len(x)) if x else None


def stats(rows, ids, cls):
    rs = [rows[i] for i in ids if i in rows]
    ms = [nshare(r["answers"]) for r in rs]
    out = {"n": len(rs), "n_missing": len(ids) - len(rs), "mean_m": mean(ms), "mean_p_mode": mean(r["p_mode"] for r in rs),
           "in_band": mean(float(0.25 <= m <= 0.75) for m in ms), "above": mean(float(m > 0.75) for m in ms),
           "below": mean(float(m < 0.25) for m in ms)}
    conf = mean(float(r["mode"] is not None and r["p_mode"] >= 0.5) for r in rs)
    if cls == "U":
        out["CA"] = conf
    else:
        out["acc_plurality"] = mean(r["correct"] for r in rs)
        out["acc_sample"] = mean(sum(1 for a in r["answers"] if a is not None and a == r["ref"]) / len(r["answers"]) for r in rs if r.get("ref") is not None)
        out["conf_rate"] = conf
    return out


def decomp(rows, ids, cls):
    st = Counter(); n = 0
    for i in ids:
        if i not in rows: continue
        r = rows[i]; m = nshare(r["answers"]); n += 1
        if 0.25 <= m <= 0.75: st["in_band"] += 1
        elif m < 0.25: st["below"] += 1
        elif cls != "U" and r["correct"] == 1: st["above_correct"] += 1
        else: st["above_wrong"] += 1
    ex = n - st["in_band"]
    return {"n": n, **{k: (st[k] / n if n else None) for k in ("in_band", "above_correct", "above_wrong", "below")},
            "counts": {k: st[k] for k in ("in_band", "above_correct", "above_wrong", "below")},
            "n_exits": ex, "share_of_exits_above_correct": (st["above_correct"] / ex if ex else None)}


res = {"analysis": "hardtraj", "status": "EXPLORATORY (post hoc)", "definitions": __doc__, "families": {}}
for fam, (btag, runs) in FAM.items():
    base, why = load(btag)
    if base is None:
        res["families"][fam] = {"base": btag, "read": False, "why": why}; print(fam, "base not read", why); continue
    gold = {}
    for i, r in base.items():
        if r["answerable"] and r.get("ref") is not None:
            gold[i] = sum(1 for a in r["answers"] if a is not None and a == r["ref"]) / len(r["answers"])
    groups = {"hard_A": sorted(i for i, g in gold.items() if g <= 3 / 8 + 1e-9),
              "other_A": sorted(i for i, r in base.items() if r["answerable"] and not (i in gold and gold[i] <= 3 / 8 + 1e-9)),
              "U": sorted(i for i, r in base.items() if not r["answerable"])}
    cls = {"hard_A": "A", "other_A": "A", "U": "U"}
    bk = {i: round(8 * nshare(r["answers"])) for i, r in base.items()}
    band_groups = {g: [i for i in ids if bk[i] in (2, 3, 4, 5, 6)] for g, ids in groups.items()}
    fr = {"base_tag": btag, "n_answerable_ref_none": sum(1 for r in base.values() if r["answerable"] and r.get("ref") is None),
          "group_sizes": {g: len(v) for g, v in groups.items()}, "base_band_group_sizes": {g: len(v) for g, v in band_groups.items()},
          "base_band_bins": {g: {k: sum(1 for i in v if bk[i] == k) for k in (2, 3, 4, 5, 6)} for g, v in band_groups.items()},
          "hard_A_base_bins_all": dict(sorted(Counter(bk[i] for i in groups["hard_A"]).items())),
          "checkpoints": {}}
    for tag in ["BASE"] + runs:
        rows, why = (base, "ok") if tag == "BASE" else load(tag)
        if rows is None:
            fr["checkpoints"][tag] = {"read": False, "why": why}; print(fam, tag, "not read", why); continue
        ck = {"read": True, "all": {g: stats(rows, ids, cls[g]) for g, ids in groups.items()},
              "base_band": {g: stats(rows, ids, cls[g]) for g, ids in band_groups.items()},
              "base_band_exit_decomposition": {g: decomp(rows, ids, cls[g]) for g, ids in band_groups.items()}}
        fr["checkpoints"][btag if tag == "BASE" else tag] = ck
        a, bb = ck["all"], ck["base_band_exit_decomposition"]
        print(f"{fam} {tag}: all in_band hard/other/U {a['hard_A']['in_band']:.3f}/{a['other_A']['in_band']:.3f}/{a['U']['in_band']:.3f}; "
              f"acc hard/other {a['hard_A']['acc_plurality']:.3f}/{a['other_A']['acc_plurality']:.3f}; CA {a['U']['CA']:.3f}; "
              f"base-band hard: in {bb['hard_A']['in_band']:.3f} solved {bb['hard_A']['above_correct']:.3f} wrong {bb['hard_A']['above_wrong']:.3f} below {bb['hard_A']['below']:.3f}; "
              f"U: in {bb['U']['in_band']:.3f} above {bb['U']['above_wrong']:.3f} below {bb['U']['below']:.3f}")
    agg = {}
    for rnd in ("r3", "r5"):
        tags = [t for t in runs if t.endswith("_" + rnd) and fr["checkpoints"].get(t, {}).get("read")]
        if not tags: continue
        a = {}
        for pop in ("all", "base_band"):
            a[pop] = {g: {k: mean(fr["checkpoints"][t][pop][g][k] for t in tags) for k in fr["checkpoints"][tags[0]][pop][g] if k not in ("n", "n_missing")} for g in groups}
        a["base_band_exit_decomposition"] = {g: {k: mean(fr["checkpoints"][t]["base_band_exit_decomposition"][g][k] for t in tags)
                                                 for k in ("in_band", "above_correct", "above_wrong", "below")} for g in groups}
        pe = {g: [fr["checkpoints"][t]["base_band_exit_decomposition"][g] for t in tags] for g in groups}
        a["pooled_exits"] = {g: {"n_exits": sum(x["n_exits"] for x in v), "n_above_correct": sum(x["counts"]["above_correct"] for x in v),
                                 "n_above_wrong": sum(x["counts"]["above_wrong"] for x in v), "n_below": sum(x["counts"]["below"] for x in v)} for g, v in pe.items()}
        a["seeds"] = tags
        agg[rnd] = a
    fr["seed_means"] = agg
    if fam == "Qwen3-4B":
        it = json.load(open(f"{A8}/items.json"))
        hid, uid = [int(i) for i in it["hard_A"]], [int(i) for i in it["U"]]
        fc = {"n_hard_A": len(hid), "n_U": len(uid), "hard_ids_equal_base_band_hard_A": sorted(hid) == sorted(band_groups["hard_A"]),
              "U_ids_subset_of_base_band_U": set(uid) <= set(band_groups["U"]), "checkpoints": {}}
        for ck in ["base"] + [f"own_s{s}_r{r}" for r in (3, 5) for s in range(3)]:
            f = f"{A8}/texts/a8_{ck}.jsonl"
            if not os.path.exists(f): fc["checkpoints"][ck] = {"read": False}; continue
            rows = read_rows(f)
            fc["checkpoints"][ck] = {"read": True, "hard_A": {**decomp(rows, hid, "A"), "acc_plurality": stats(rows, hid, "A")["acc_plurality"],
                                                              "acc_sample": stats(rows, hid, "A")["acc_sample"]},
                                     "U": {**decomp(rows, uid, "U"), "CA": stats(rows, uid, "U")["CA"]}}
            h, u = fc["checkpoints"][ck]["hard_A"], fc["checkpoints"][ck]["U"]
            print(f"fresh {ck}: hard n{h['n']} in {h['in_band']:.3f} solved {h['above_correct']:.3f} wrong {h['above_wrong']:.3f} below {h['below']:.3f} acc {h['acc_plurality']:.3f}/{h['acc_sample']:.3f}; "
                  f"U n{u['n']} in {u['in_band']:.3f} above {u['above_wrong']:.3f} below {u['below']:.3f} CA {u['CA']:.3f}")
        # same 112 + 112 items scored on the original dumps, for a like-for-like comparison with the fresh draws
        orig = {}
        for tag in ["BASE"] + runs[:3] + runs[5:]:
            rows = base if tag == "BASE" else load(tag)[0]
            if rows is None: continue
            orig[btag if tag == "BASE" else tag] = {"hard_A": decomp(rows, hid, "A"), "U": decomp(rows, uid, "U")}
        fc["same_items_original_dumps"] = orig
        fr["fresh_resample_check"] = fc
    res["families"][fam] = fr
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
