"""EXPLORATORY (post hoc) analysis "tracers": planted tracers in the training sets.

For every own-loop run in scope, count admitted.jsonl items by 'kind'
(proposed / tracer_deleted / tracer_intact) and, for the file the trainer actually
read (config.json 'data': admitted.jsonl for own, admitted_clean.jsonl for own_clean),
the share of the training set that is admitted premise-deleted tracers and admitted
intact tracers. Also reproduces the exact 60 x 32 = 1,920 question draws of b_train.py
(epoch-shuffled order, random.Random(seed*1000 + epoch)) and cross-checks the reproduced
epoch counter against log.jsonl. Finally checks that the Opus audits (audit_admitted.json,
audit_full.json) judge only kind == 'proposed' items and never tracers.

Read-only on everything outside paper_arr/analyses/. No GPU.
Output: paper_arr/analyses/tracers.json
"""
import collections
import glob
import json
import os
import random
import re

ROOT = "/home/tione/notebook/rsi_rl"
RUNS = os.path.join(ROOT, "dirB", "runs")
OUT = os.path.join(ROOT, "paper_arr", "analyses", "tracers.json")

# (group label, glob pattern, in requested scope?)
GROUPS = [
    ("qwen3_4b_own", "qwen3_4b_Base_own_a3_s*_r*", True),
    ("qwen3_8b_own", "qwen3_8b_Base_own_a3_s*_r*", True),
    ("phi4mini_own", "phi4mini_own_a4_s*_r*", True),
    ("qwen3_4b_own_clean", "qwen3_4b_Base_own_clean_a3_s*_r*", True),
    ("llama3b_own_clean", "llama3p2_3b_own_clean_a3_s*_r*", True),
    ("llama3b_own", "llama3p2_3b_own_a3_s*_r*", False),
    ("qwen3_14b_own", "qwen3_14b_Base_own_a4_s*_r*", False),
]
NAME_RE = re.compile(r"_s(\d+)_r(\d+)$")
KINDS = ("proposed", "tracer_deleted", "tracer_intact")


def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def kinds(rows):
    c = collections.Counter(r.get("kind") for r in rows)
    return {k: c.get(k, 0) for k in sorted(set(c) | set(KINDS))}


def id_prefix_check(rows):
    """map id prefix -> Counter of kinds (to confirm kind labels are consistent with ids)"""
    m = collections.defaultdict(collections.Counter)
    for r in rows:
        m[str(r["id"]).split(":")[0]][r.get("kind")] += 1
    return {p: dict(c) for p, c in m.items()}


def reproduce_draws(rows, seed, steps, qps):
    """Exact replica of b_train.py _order()/next_batch(); returns per-step (row idx list, epoch after step)."""
    cur = {"epoch": 0, "pos": 0}
    cache = {}

    def order(e):
        if e not in cache:
            o = list(range(len(rows)))
            random.Random(seed * 1000 + e).shuffle(o)
            cache[e] = o
        return cache[e]

    per_step = []
    for _ in range(steps):
        out = []
        while len(out) < qps:
            o = order(cur["epoch"])
            if cur["pos"] >= len(o):
                cur["epoch"] += 1
                cur["pos"] = 0
                continue
            out.append(o[cur["pos"]])
            cur["pos"] += 1
        per_step.append((out, cur["epoch"]))
    return per_step


def logged_steps(run_dir):
    p = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(p):
        return {}
    ep = {}
    for r in jl(p):
        if "epoch" in r and "step" in r and "n_seqs" in r:
            ep[int(r["step"])] = int(r["epoch"])  # last write wins (resumes re-log a step)
    return ep


def audit_check(path, adm_by_id):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    ids = [v["id"] for v in d.get("verdicts", [])]
    kc = collections.Counter(adm_by_id.get(i, {}).get("kind", "NOT_IN_ADMITTED") for i in ids)
    return {"file": os.path.relpath(path, ROOT), "protocol": d.get("protocol"), "n_verdicts": len(ids),
            "n_unique_ids": len(set(ids)), "verdict_ids_by_admitted_kind": dict(kc),
            "counts": d.get("counts"), "n_proposed_admitted_field": d.get("n_proposed_admitted")}


def share(a, b):
    return (a / b) if b else None


runs = []
for gname, pat, in_scope in GROUPS:
    for rd in sorted(glob.glob(os.path.join(RUNS, pat))):
        if not os.path.isdir(rd):
            continue  # skips *.claim lock files
        name = os.path.basename(rd)
        m = NAME_RE.search(name)
        if not m or not os.path.exists(os.path.join(rd, "admitted.jsonl")):
            continue
        s, r = int(m.group(1)), int(m.group(2))
        cfg = json.load(open(os.path.join(rd, "config.json")))
        adm = jl(os.path.join(rd, "admitted.jsonl"))
        adm_by_id = {x["id"]: x for x in adm}
        data_path = cfg["data"]
        train = jl(data_path)
        k_adm, k_tr = kinds(adm), kinds(train)
        n_tr = len(train)
        gs_p = os.path.join(rd, "guard_stats.json")
        gs = json.load(open(gs_p)) if os.path.exists(gs_p) else {}
        gs_check = {
            "proposed_admit": gs.get("proposed_admit"), "tracer_deleted_admit": gs.get("tracer_deleted_admit"),
            "tracer_intact_admit": gs.get("tracer_intact_admit"), "n_admitted": gs.get("n_admitted"),
            "tracer_deleted_n": gs.get("tracer_deleted_n"), "tracer_intact_n": gs.get("tracer_intact_n"),
        }
        gs_match = (gs.get("proposed_admit") == k_adm["proposed"] and gs.get("tracer_deleted_admit") == k_adm["tracer_deleted"]
                    and gs.get("tracer_intact_admit") == k_adm["tracer_intact"] and gs.get("n_admitted") == len(adm))
        trained = os.path.exists(os.path.join(rd, ".trained"))
        lg = logged_steps(rd)
        steps, qps, seed = int(cfg["steps"]), int(cfg["qps"]), int(cfg["seed"])
        draws = None
        if lg:
            ps = reproduce_draws(train, seed, steps, qps)
            n_logged = min(max(lg), steps)
            ep_ok = all(lg[i + 1] == ps[i][1] for i in range(n_logged) if (i + 1) in lg)
            dc = collections.Counter()
            for idx, _ in ps[:n_logged]:
                for j in idx:
                    dc[train[j]["kind"]] += 1
            nd = sum(dc.values())
            draws = {"n_steps_logged": n_logged, "n_draws": nd, "draws_by_kind": dict(dc),
                     "share_draws_tracer_deleted": share(dc["tracer_deleted"], nd),
                     "share_draws_tracer_intact": share(dc["tracer_intact"], nd),
                     "epoch_after_last_step": ps[n_logged - 1][1] if n_logged else None,
                     "epoch_counter_matches_log": ep_ok}
        clean_stats_p = os.path.join(rd, "admitted_clean_stats.json")
        clean_stats = json.load(open(clean_stats_p)) if os.path.exists(clean_stats_p) else None
        clean_check = None
        if clean_stats:
            aud = json.load(open(clean_stats["audit"]))
            V = {v["id"]: v["verdict"] for v in aud["verdicts"]}
            kept_prop = [x for x in train if x["kind"] == "proposed"]
            clean_check = {"audit_used": os.path.relpath(clean_stats["audit"], ROOT),
                           "kept_proposed_all_SOLVABLE": all(V.get(x["id"]) == "SOLVABLE" for x in kept_prop),
                           "n_admitted_proposed_judged_SOLVABLE": sum(1 for x in adm if x["kind"] == "proposed" and V.get(x["id"]) == "SOLVABLE"),
                           "kept_tracer_deleted": k_tr.get("tracer_deleted", 0),
                           "all_admitted_intact_kept": k_tr.get("tracer_intact", 0) == k_adm["tracer_intact"]}
        runs.append({
            "run": name, "group": gname, "in_requested_scope": in_scope, "seed": s, "round": r,
            "train_file": os.path.relpath(data_path, ROOT), "trained": trained,
            "admitted_n": len(adm), "admitted_kinds": k_adm,
            "admitted_share_tracer_deleted": share(k_adm["tracer_deleted"], len(adm)),
            "admitted_share_tracer_intact": share(k_adm["tracer_intact"], len(adm)),
            "train_n": n_tr, "train_kinds": k_tr,
            "train_share_tracer_deleted": share(k_tr.get("tracer_deleted", 0), n_tr),
            "train_share_tracer_intact": share(k_tr.get("tracer_intact", 0), n_tr),
            "train_share_tracer_any": share(k_tr.get("tracer_deleted", 0) + k_tr.get("tracer_intact", 0), n_tr),
            "guard_admit_rate_tracer_deleted": share(k_adm["tracer_deleted"], gs.get("tracer_deleted_n")),
            "guard_admit_rate_tracer_intact": share(k_adm["tracer_intact"], gs.get("tracer_intact_n")),
            "guard_stats": gs_check, "guard_stats_match_admitted": gs_match,
            "id_prefix_by_kind": id_prefix_check(adm),
            "draws": draws, "clean_subset_check": clean_check,
            "audit_admitted": audit_check(os.path.join(rd, "audit_admitted.json"), adm_by_id),
            "audit_full": audit_check(os.path.join(rd, "audit_full.json"), adm_by_id),
        })


def rng_(vals):
    v = [x for x in vals if x is not None]
    return [min(v), max(v)] if v else None


def summarise(rs):
    return {"n_runs": len(rs), "seeds": sorted({x["seed"] for x in rs}),
            "train_n": rng_([x["train_n"] for x in rs]),
            "train_tracer_deleted": rng_([x["train_kinds"].get("tracer_deleted", 0) for x in rs]),
            "train_tracer_intact": rng_([x["train_kinds"].get("tracer_intact", 0) for x in rs]),
            "train_share_tracer_deleted": rng_([x["train_share_tracer_deleted"] for x in rs]),
            "train_share_tracer_intact": rng_([x["train_share_tracer_intact"] for x in rs]),
            "train_share_tracer_any": rng_([x["train_share_tracer_any"] for x in rs]),
            "admitted_n": rng_([x["admitted_n"] for x in rs]),
            "admitted_tracer_deleted": rng_([x["admitted_kinds"]["tracer_deleted"] for x in rs]),
            "admitted_tracer_intact": rng_([x["admitted_kinds"]["tracer_intact"] for x in rs]),
            "admitted_share_tracer_deleted": rng_([x["admitted_share_tracer_deleted"] for x in rs]),
            "admitted_share_tracer_intact": rng_([x["admitted_share_tracer_intact"] for x in rs]),
            "share_draws_tracer_deleted": rng_([x["draws"]["share_draws_tracer_deleted"] for x in rs if x["draws"]]),
            "share_draws_tracer_intact": rng_([x["draws"]["share_draws_tracer_intact"] for x in rs if x["draws"]]),
            "mean_train_share_tracer_deleted": sum(x["train_share_tracer_deleted"] for x in rs) / len(rs),
            "mean_train_share_tracer_intact": sum(x["train_share_tracer_intact"] for x in rs) / len(rs),
            "n_runs_admitted_deleted_gt_intact": sum(1 for x in rs if x["admitted_kinds"]["tracer_deleted"] > x["admitted_kinds"]["tracer_intact"]),
            "n_runs_admitted_deleted_eq_intact": sum(1 for x in rs if x["admitted_kinds"]["tracer_deleted"] == x["admitted_kinds"]["tracer_intact"]),
            "guard_admit_rate_tracer_deleted": rng_([x["guard_admit_rate_tracer_deleted"] for x in rs]),
            "guard_admit_rate_tracer_intact": rng_([x["guard_admit_rate_tracer_intact"] for x in rs])}


by_group_round, by_group = {}, {}
for g, _, _ in GROUPS:
    rs = [x for x in runs if x["group"] == g and x["trained"]]
    if not rs:
        continue
    by_group[g] = summarise(rs)
    by_group_round[g] = {f"r{r}": summarise([x for x in rs if x["round"] == r]) for r in sorted({x["round"] for x in rs})}

# audit exclusion summary over every audit file found
aud_files = [a for x in runs for a in (x["audit_admitted"], x["audit_full"]) if a]
tr_in_audit = sum(a["verdict_ids_by_admitted_kind"].get("tracer_deleted", 0) + a["verdict_ids_by_admitted_kind"].get("tracer_intact", 0) for a in aud_files)
notin = sum(a["verdict_ids_by_admitted_kind"].get("NOT_IN_ADMITTED", 0) for a in aud_files)
full_cov = [x["run"] for x in runs if x["audit_full"] and x["audit_full"]["verdict_ids_by_admitted_kind"].get("proposed", 0) != x["admitted_kinds"]["proposed"]]
audit_summary = {
    "n_audit_files_checked": len(aud_files),
    "n_audit_admitted_files": sum(1 for x in runs if x["audit_admitted"]),
    "n_audit_full_files": sum(1 for x in runs if x["audit_full"]),
    "total_verdicts": sum(a["n_verdicts"] for a in aud_files),
    "verdicts_on_tracer_items": tr_in_audit,
    "verdicts_on_ids_not_in_admitted": notin,
    "audit_full_runs_not_covering_all_admitted_proposals": full_cov,
    "code_evidence": ["dirA3/code/a3_audit200_assemble.py L16-17: P = [x for x in A if x['kind'] == 'proposed']; random.Random(0).shuffle(P)",
                      "audit_admitted.json 'protocol' strings (per run, below)",
                      "dirA3/code/a3_subset.py (clean = proposed judged SOLVABLE in audit_full.json + tracer_intact; tracer_deleted never kept)"],
    "protocol_strings": sorted({a["protocol"] for a in aud_files if a["protocol"]}),
}

def prefix_totals(rs):
    tot = collections.defaultdict(collections.Counter)
    for x in rs:
        for p, c in x["id_prefix_by_kind"].items():
            tot[p].update(c)
    return {p: dict(c) for p, c in tot.items()}


checks = {
    "all_guard_stats_match_admitted": all(x["guard_stats_match_admitted"] for x in runs if x["guard_stats"]["n_admitted"] is not None),
    "runs_guard_stats_mismatch": [x["run"] for x in runs if x["guard_stats"]["n_admitted"] is not None and not x["guard_stats_match_admitted"]],
    "all_epoch_counters_match_log": all(x["draws"]["epoch_counter_matches_log"] for x in runs if x["draws"]),
    "runs_epoch_mismatch": [x["run"] for x in runs if x["draws"] and not x["draws"]["epoch_counter_matches_log"]],
    "untrained_runs_excluded_from_summaries": [x["run"] for x in runs if not x["trained"]],
    "clean_runs_kept_proposed_all_SOLVABLE": all(x["clean_subset_check"]["kept_proposed_all_SOLVABLE"] for x in runs if x["clean_subset_check"]),
    "clean_runs_tracer_deleted_in_train": sum(x["clean_subset_check"]["kept_tracer_deleted"] for x in runs if x["clean_subset_check"]),
    "own_runs_train_file_is_admitted": all(x["train_file"].endswith("/admitted.jsonl") for x in runs if "clean" not in x["group"]),
    "kinds_seen": sorted({k for x in runs for k in x["admitted_kinds"]}),
    "id_prefix_by_kind_all_runs": prefix_totals(runs),
    "guard_pool_tracer_n_always_200_200": all(x["guard_stats"]["tracer_deleted_n"] == 200 and x["guard_stats"]["tracer_intact_n"] == 200 for x in runs if x["guard_stats"]["n_admitted"] is not None),
}

res = {"analysis": "tracers", "status": "EXPLORATORY (post hoc)",
       "definitions": {"train_share_tracer_deleted": "admitted premise-deleted tracers / items in the file the trainer read (config.json 'data')",
                       "train_share_tracer_intact": "admitted intact tracers / items in that file",
                       "share_draws_*": "share of the reproduced 60x32 = 1,920 training question draws (b_train.py next_batch)",
                       "pool": "each round's guard pool = proposer questions + 200 deleted + 200 intact tracers (guard_stats tracer_*_n)"},
       "checks": checks, "audit_exclusion": audit_summary,
       "summary_by_group": by_group, "summary_by_group_round": by_group_round, "runs": runs}
json.dump(res, open(OUT, "w"), indent=1)

print("checks", json.dumps(checks))
print("audit", json.dumps({k: v for k, v in audit_summary.items() if k != "code_evidence"}))
for g in by_group_round:
    print("==", g, "ALL", json.dumps(by_group[g]))
    for rr, sm in by_group_round[g].items():
        print("  ", rr, json.dumps(sm))
print("per-run: run | train_n | del | int | share_del | share_int | draw_del | draw_int")
for x in runs:
    d = x["draws"] or {}
    print(x["run"], x["train_n"], x["train_kinds"].get("tracer_deleted", 0), x["train_kinds"].get("tracer_intact", 0),
          round(x["train_share_tracer_deleted"], 4), round(x["train_share_tracer_intact"], 4),
          None if not d else round(d["share_draws_tracer_deleted"], 4), None if not d else round(d["share_draws_tracer_intact"], 4),
          "" if x["trained"] else "UNTRAINED")
