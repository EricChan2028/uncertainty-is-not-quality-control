"""Independent verification of the EXPLORATORY 'tracers' analysis (post hoc). Read-only on all inputs.
Writes only paper_arr/analyses/check_tracers.json. No GPU."""
import collections, glob, hashlib, json, os, random, re, statistics

ROOT = "/home/tione/notebook/rsi_rl"
RUNS = f"{ROOT}/dirB/runs"
OUT = f"{ROOT}/paper_arr/analyses/check_tracers.json"
GROUPS = [
    ("q4_own", r"^qwen3_4b_Base_own_a3_s(\d+)_r(\d+)$", True),
    ("q8_own", r"^qwen3_8b_Base_own_a3_s(\d+)_r(\d+)$", True),
    ("phi_own", r"^phi4mini_own_a4_s(\d+)_r(\d+)$", True),
    ("q4_clean", r"^qwen3_4b_Base_own_clean_a3_s(\d+)_r(\d+)$", True),
    ("llama_clean", r"^llama3p2_3b_own_clean_a3_s(\d+)_r(\d+)$", True),
    ("llama_own_supp", r"^llama3p2_3b_own_a3_s(\d+)_r(\d+)$", False),
    ("q14_own_supp", r"^qwen3_14b_Base_own_a4_s(\d+)_r(\d+)$", False),
]


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def kind_of_id(i):
    p = i.split(":")[0]
    if p == "trd":
        return "tracer_deleted"
    if p == "tri":
        return "tracer_intact"
    if re.fullmatch(r"prop\d+", p):
        return "proposed"
    return "OTHER:" + p


all_names = sorted(os.listdir(RUNS))
clean_like = [n for n in all_names if "clean" in n]

# tracer-pool validation sample (100 deleted + 100 intact pool twins, Random(0))
VD = f"{ROOT}/dirA3/results/audit_validation"
val_labels = json.load(open(f"{VD}/labels.json"))
val_verd = {}
for f in sorted(glob.glob(f"{VD}/parts/*.json")):
    for v in json.load(open(f))["verdicts"]:
        val_verd[v["id"]] = v["verdict"]

runs = []
for g, pat, scope in GROUPS:
    for n in all_names:
        m = re.match(pat, n)
        d = f"{RUNS}/{n}"
        if not m or not os.path.isdir(d):
            continue
        s, r = int(m.group(1)), int(m.group(2))
        rec = {"run": n, "group": g, "scope": scope, "s": s, "r": r}
        cfg = json.load(open(f"{d}/config.json"))
        rec["data"] = cfg["data"]
        rec["trained_flag"] = os.path.exists(f"{d}/.trained")
        rec["trained_val"] = open(f"{d}/.trained").read().strip() if rec["trained_flag"] else None
        adm = rj(f"{d}/admitted.jsonl")
        rec["adm_n"] = len(adm)
        ka = collections.Counter(x["kind"] for x in adm)
        rec["adm_kinds"] = dict(ka)
        rec["id_kind_mismatch"] = sum(1 for x in adm if kind_of_id(x["id"]) != x["kind"])
        rec["adm_dup_ids"] = len(adm) - len({x["id"] for x in adm})
        tr = rj(cfg["data"])
        kt = collections.Counter(x["kind"] for x in tr)
        rec["train_n"] = len(tr)
        rec["train_kinds"] = dict(kt)
        adm_ids = {x["id"] for x in adm}
        rec["train_subset_of_admitted"] = all(x["id"] in adm_ids for x in tr)
        nd, ni = kt.get("tracer_deleted", 0), kt.get("tracer_intact", 0)
        rec["del"], rec["int"] = nd, ni
        rec["sh_del"], rec["sh_int"], rec["sh_any"] = nd / len(tr), ni / len(tr), (nd + ni) / len(tr)
        gs = json.load(open(f"{d}/guard_stats.json")) if os.path.exists(f"{d}/guard_stats.json") else {}
        rec["gs_pool"] = (gs.get("tracer_deleted_n"), gs.get("tracer_intact_n"))
        rec["gs_match"] = (gs.get("proposed_admit"), gs.get("tracer_deleted_admit"), gs.get("tracer_intact_admit"), gs.get("n_admitted")) == (
            ka.get("proposed", 0), ka.get("tracer_deleted", 0), ka.get("tracer_intact", 0), len(adm))
        rec["gar_del"] = ka.get("tracer_deleted", 0) / gs["tracer_deleted_n"] if gs.get("tracer_deleted_n") else None
        rec["gar_int"] = ka.get("tracer_intact", 0) / gs["tracer_intact_n"] if gs.get("tracer_intact_n") else None
        # draws: global-index formulation, independent of the next_batch loop
        steps, qps, seed = int(cfg["steps"]), int(cfg["qps"]), int(cfg["seed"])
        N = len(tr)
        perms = {}

        def perm(e):
            if e not in perms:
                o = list(range(N))
                random.Random(seed * 1000 + e).shuffle(o)
                perms[e] = o
            return perms[e]
        logep = {}
        if os.path.exists(f"{d}/log.jsonl"):
            for x in rj(f"{d}/log.jsonl"):
                if "epoch" in x and "n_seqs" in x:
                    logep[int(x["step"])] = int(x["epoch"])
        rec["log_max_step"] = max(logep) if logep else None
        if logep:
            T = min(max(logep), steps)
            dc = collections.Counter()
            for gi in range(T * qps):
                dc[tr[perm(gi // N)[gi % N]]["kind"]] += 1
            rec["draw_n"] = T * qps
            rec["draw_del"] = dc["tracer_deleted"] / (T * qps)
            rec["draw_int"] = dc["tracer_intact"] / (T * qps)
            rec["epoch_ok"] = all(logep[t] == (t * qps - 1) // N for t in range(1, T + 1) if t in logep)
            rec["epoch_missing_steps"] = [t for t in range(1, T + 1) if t not in logep]
        # audits
        for af in ("audit_admitted.json", "audit_full.json"):
            p = f"{d}/{af}"
            if not os.path.exists(p):
                rec[af] = None
                continue
            A = json.load(open(p))
            ids = [v["id"] for v in A["verdicts"]]
            kmap = {x["id"]: x["kind"] for x in adm}
            c = collections.Counter(kmap.get(i, "NOT_IN_ADMITTED") for i in ids)
            recount = dict(collections.Counter(v["verdict"] for v in A["verdicts"]))
            info = {"n": len(ids), "dup": len(ids) - len(set(ids)), "by_kind": dict(c),
                    "tracer_prefix_ids": sum(1 for i in ids if i.split(":")[0] in ("trd", "tri")),
                    "counts_match_recount": recount == A.get("counts"), "counts": A.get("counts")}
            prop_ids = [x["id"] for x in adm if x["kind"] == "proposed"]
            if af == "audit_full.json":
                info["covers_all_proposed_exactly"] = set(ids) == set(prop_ids)
            else:
                P = [x for x in adm if x["kind"] == "proposed"]
                random.Random(0).shuffle(P)
                info["equals_first200_random0"] = set(ids) == {x["id"] for x in P[:200]}
            rec[af] = info
        # clean checks
        csp = f"{d}/admitted_clean_stats.json"
        if os.path.exists(csp):
            cs = json.load(open(csp))
            aud = json.load(open(cs["audit"]))
            V = {v["id"]: v["verdict"] for v in aud["verdicts"]}
            kept = [x for x in tr if x["kind"] == "proposed"]
            solv = [x for x in adm if x["kind"] == "proposed" and V.get(x["id"]) == "SOLVABLE"]
            rec["clean"] = {"audit": os.path.relpath(cs["audit"], ROOT),
                            "audit_in_own_dir": os.path.dirname(cs["audit"]) == d,
                            "kept_all_solvable": all(V.get(x["id"]) == "SOLVABLE" for x in kept),
                            "kept_eq_all_solvable": len(kept) == len(solv),
                            "intact_all_kept": ni == ka.get("tracer_intact", 0),
                            "del_kept": nd,
                            "prefilter_del": ka.get("tracer_deleted", 0),
                            "prefilter_del_share_adm": ka.get("tracer_deleted", 0) / len(adm)}
            if r == 1:
                own = n.replace("_own_clean_a3_", "_own_a3_")
                op = f"{RUNS}/{own}/admitted.jsonl"
                rec["clean"]["r1_admitted_md5_eq_own"] = (md5(f"{d}/admitted.jsonl") == md5(op)) if os.path.exists(op) else None
                rec["clean"]["r1_audit_is_own_run"] = cs["audit"].startswith(f"{RUNS}/{own}/")
        # validated-sample overlap (deleted tracers in train)
        dels = [x["id"] for x in tr if x["kind"] == "tracer_deleted"]
        rec["val_overlap_del"] = dict(collections.Counter(val_verd[i] for i in dels if i in val_labels))
        runs.append(rec)


def summ(rs):
    def rg(k):
        v = [x[k] for x in rs if x.get(k) is not None]
        return [min(v), max(v)] if v else None

    def mn(k):
        return statistics.mean(x[k] for x in rs)
    has_draw = [x for x in rs if x.get("draw_del") is not None]
    return {"n": len(rs), "seeds": sorted({x["s"] for x in rs}), "rounds": sorted({x["r"] for x in rs}),
            "train_n": rg("train_n"), "del": rg("del"), "int": rg("int"),
            "sh_del": rg("sh_del"), "sh_int": rg("sh_int"), "sh_any": rg("sh_any"),
            "mean_sh_del": mn("sh_del"), "mean_sh_int": mn("sh_int"),
            "pooled_sh_del": sum(x["del"] for x in rs) / sum(x["train_n"] for x in rs),
            "pooled_sh_int": sum(x["int"] for x in rs) / sum(x["train_n"] for x in rs),
            "n_del_gt_int_adm": sum(1 for x in rs if x["adm_kinds"].get("tracer_deleted", 0) > x["adm_kinds"].get("tracer_intact", 0)),
            "n_del_gt_int_train": sum(1 for x in rs if x["del"] > x["int"]),
            "gar_del": rg("gar_del"), "gar_int": rg("gar_int"),
            "draw_del": rg("draw_del"), "draw_int": rg("draw_int"),
            "max_abs_draw_minus_set": max(max(abs(x["draw_del"] - x["sh_del"]), abs(x["draw_int"] - x["sh_int"])) for x in has_draw) if has_draw else None,
            "adm_n": rg("adm_n")}


trained = [x for x in runs if x["trained_flag"]]
by_group, by_round = {}, {}
for g, _, _ in GROUPS:
    rs = [x for x in trained if x["group"] == g]
    if not rs:
        continue
    by_group[g] = summ(rs)
    by_round[g] = {f"r{r}": summ([x for x in rs if x["r"] == r]) for r in sorted({x["r"] for x in rs})}
    if "clean" in g:
        by_group[g]["prefilter_del"] = [min(x["clean"]["prefilter_del"] for x in rs), max(x["clean"]["prefilter_del"] for x in rs)]
        by_group[g]["prefilter_del_share_adm"] = [min(x["clean"]["prefilter_del_share_adm"] for x in rs), max(x["clean"]["prefilter_del_share_adm"] for x in rs)]

scope_own = [x for x in trained if x["group"] in ("q4_own", "q8_own", "phi_own")]
cross = {"own_in_scope_n": len(scope_own),
         "sh_del": [min(x["sh_del"] for x in scope_own), max(x["sh_del"] for x in scope_own)],
         "sh_int": [min(x["sh_int"] for x in scope_own), max(x["sh_int"] for x in scope_own)],
         "r2plus_n": sum(1 for x in scope_own if x["r"] >= 2),
         "r2plus_del_gt_int": sum(1 for x in scope_own if x["r"] >= 2 and x["del"] > x["int"]),
         "r1_n": sum(1 for x in scope_own if x["r"] == 1),
         "r1_del_gt_int": sum(1 for x in scope_own if x["r"] == 1 and x["del"] > x["int"]),
         "r1_rows": [(x["run"], x["del"], x["int"]) for x in scope_own if x["r"] == 1]}
clean_all = [x for x in trained if x["group"] in ("q4_clean", "llama_clean")]
cross["clean_sh_int"] = [min(x["sh_int"] for x in clean_all), max(x["sh_int"] for x in clean_all)]
cross["clean_del_total"] = sum(x["del"] for x in clean_all)

# audit exclusion over runs in this check
af_recs = [x[a] for x in runs for a in ("audit_admitted.json", "audit_full.json") if x.get(a)]
audit = {"n_files": len(af_recs),
         "n_audit_admitted": sum(1 for x in runs if x.get("audit_admitted.json")),
         "n_audit_full": sum(1 for x in runs if x.get("audit_full.json")),
         "n_verdicts": sum(a["n"] for a in af_recs),
         "verdicts_on_tracers": sum(a["by_kind"].get("tracer_deleted", 0) + a["by_kind"].get("tracer_intact", 0) for a in af_recs),
         "tracer_prefix_ids": sum(a["tracer_prefix_ids"] for a in af_recs),
         "not_in_admitted": sum(a["by_kind"].get("NOT_IN_ADMITTED", 0) for a in af_recs),
         "dups": sum(a["dup"] for a in af_recs),
         "counts_mismatch_recount": [x["run"] + "/" + a for x in runs for a in ("audit_admitted.json", "audit_full.json") if x.get(a) and not x[a]["counts_match_recount"]],
         "audit_full_not_exact": [x["run"] for x in runs if x.get("audit_full.json") and not x["audit_full.json"]["covers_all_proposed_exactly"]],
         "audit_admitted_not_first200": [x["run"] for x in runs if x.get("audit_admitted.json") and not x["audit_admitted.json"]["equals_first200_random0"]],
         "audit_admitted_n_not_200": [(x["run"], x["audit_admitted.json"]["n"]) for x in runs if x.get("audit_admitted.json") and x["audit_admitted.json"]["n"] != 200]}
# broader sweep: every audit_*.json directly under dirB/runs/<run>/, by id prefix only
sweep = collections.Counter()
nfiles = 0
for p in glob.glob(f"{RUNS}/*/audit_*.json"):
    try:
        A = json.load(open(p))
    except Exception:
        continue
    if not isinstance(A, dict) or "verdicts" not in A:
        continue
    nfiles += 1
    for v in A["verdicts"]:
        pre = str(v.get("id", "")).split(":")[0]
        sweep["prop" if re.fullmatch(r"prop\d+", pre) else pre] += 1
audit["all_dirB_audit_files_swept"] = nfiles
audit["all_dirB_prefix_counts"] = dict(sweep)

checks = {"n_runs_total": len(runs), "n_trained": len(trained),
          "n_trained_in_scope": sum(1 for x in trained if x["scope"]),
          "untrained": [x["run"] for x in runs if not x["trained_flag"]],
          "kinds_seen": sorted({k for x in runs for k in x["adm_kinds"]}),
          "id_kind_mismatch_total": sum(x["id_kind_mismatch"] for x in runs),
          "adm_dup_ids_total": sum(x["adm_dup_ids"] for x in runs),
          "pool_always_200_200": all(x["gs_pool"] == (200, 200) for x in runs),
          "gs_mismatch": [x["run"] for x in runs if not x["gs_match"]],
          "train_not_subset": [x["run"] for x in runs if not x["train_subset_of_admitted"]],
          "own_train_is_admitted": all(x["data"] == f"{RUNS}/{x['run']}/admitted.jsonl" for x in runs if "clean" not in x["group"]),
          "clean_train_is_admitted_clean": all(x["data"] == f"{RUNS}/{x['run']}/admitted_clean.jsonl" for x in runs if "clean" in x["group"]),
          "epoch_bad": [x["run"] for x in runs if x.get("epoch_ok") is False],
          "epoch_missing": {x["run"]: x["epoch_missing_steps"] for x in runs if x.get("epoch_missing_steps")},
          "trained_not_60": [(x["run"], x["trained_val"], x["log_max_step"]) for x in trained if x["trained_val"] != "60" or x["log_max_step"] != 60],
          "clean_all_ok": all(x["clean"]["kept_all_solvable"] and x["clean"]["kept_eq_all_solvable"] and x["clean"]["intact_all_kept"] and x["clean"]["del_kept"] == 0 for x in runs if x.get("clean")),
          "clean_r1": [(x["run"], x["clean"]["r1_admitted_md5_eq_own"], x["clean"]["r1_audit_is_own_run"]) for x in runs if x.get("clean") and x["r"] == 1],
          "clean_r2plus_audit_own_dir": all(x["clean"]["audit_in_own_dir"] for x in runs if x.get("clean") and x["r"] >= 2),
          "all_clean_like_run_names": clean_like}

# tracer-pool validation
vd = collections.Counter(val_verd[i] for i, l in val_labels.items() if l["kind"] == "tracer_deleted")
vi = collections.Counter(val_verd[i] for i, l in val_labels.items() if l["kind"] == "tracer_intact")
ov_tot = collections.Counter()
for x in trained:
    if x["group"] in ("q4_own", "q8_own", "phi_own"):
        ov_tot.update(x["val_overlap_del"])
tw = rj(f"{ROOT}/data/hp_twins.jsonl")
probe = [t for t in tw if t["task"] == 0 or t["split_kind"] == "template"]
pool = [t for t in tw if not (t["task"] == 0 or t["split_kind"] == "template")]
pv = json.load(open(f"{ROOT}/dirA/results/twin_validation_probe543.json"))
probe_fams = {str(t["fam"]) for t in probe}
validation = {"tracer_pool_sample_deleted": dict(vd), "tracer_pool_sample_intact": dict(vi),
              "val_sample_all_in_pool": all(("%s:%s" % (l["twin"]["fam"], l["twin"]["task"])) in {"%s:%s" % (t["fam"], t["task"]) for t in pool} for l in val_labels.values()),
              "n_twins": len(tw), "n_probe": len(probe), "n_pool": len(pool),
              "probe543_counts": pv["counts"],
              "pool_share_fam_in_probe_fams": sum(1 for t in pool if str(t["fam"]) in probe_fams) / len(pool),
              "own_in_scope_train_deleted_tracer_occurrences_in_validated_sample": dict(ov_tot)}

res = {"analysis": "check_tracers", "status": "EXPLORATORY (post hoc) verification", "checks": checks, "audit": audit,
       "cross": cross, "validation": validation, "by_group": by_group, "by_round": by_round,
       "runs": runs}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({"checks": checks, "audit": audit, "cross": cross, "validation": validation}))
for g in by_group:
    print("==", g, json.dumps(by_group[g]))
    for rr, sm in by_round[g].items():
        print("   ", rr, json.dumps({k: sm[k] for k in ("n", "seeds", "train_n", "del", "int", "sh_del", "sh_int", "n_del_gt_int_adm", "gar_del", "gar_int", "max_abs_draw_minus_set")}))
