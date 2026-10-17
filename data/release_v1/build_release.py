#!/usr/bin/env python3
"""build_release.py -- regenerates every file of release_v1/ from the project tree (A7 item 3, registered 15:10 CST 24 Sep 2026).

Reads the run directories and result files of the project read-only and writes ONLY below release_v1/:
  audited_pools/        one JSONL per audited (lineage, round): admitted proposer problems + Opus verdicts + admission-time
                        solver agreement (G = 8); INDEX.tsv / INDEX.jsonl with one row per audited run
  raw_proposal_audits/  200-item audits of RAW (pre-guard) proposals (co-evolving proposer arms)
  fixed_pool_audits/    fully audited fixed candidate pools with the agreement of every scored checkpoint; CHECKPOINTS.tsv
  twins/                planted premise-deletion twins as patches against GSM-Symbolic (licence: see DATASHEET), rebuild
                        script, judge validations, tracer guard records
  judge/                exact Opus judge prompts (placeholders for host / local paths), protocol, validation sets
  probe/                judge-free probe code (copies; the only edit replaces the absolute project root by $RSI_REPO),
                        UMWP subset ids, our UMWP and twin dumps (ids only), reproduce_j1.py
  DATASHEET.md, README.md, MANIFEST.tsv (sha256, bytes, rows of every other file), build_release.py (this file)

  RSI_REPO=<REPO> python3 build_release.py
      <REPO> is the project root (default: the parent of this file's directory). CPU only, about 2 min. Deterministic: the
      same inputs give byte-identical outputs, except for the snapshot time (INDEX.jsonl, judge/validation/summary.json,
      DATASHEET.md). GSM_SYMBOLIC_DIR overrides the pinned GSM-Symbolic snapshot directory. A scratch directory
      release_v1/_work/ is left untouched and is not part of the release.
"""
import collections
import datetime
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys

R = os.environ.get("RSI_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (<REPO>)
B = f"{R}/dirB/runs"
OUT = f"{R}/release_v1"
GD = f"{R}/dirA3/results/guard_diag"
PA = f"{R}/dirA3/results/pool_audit"
GSM_SYM_REV = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
GSM_SYM = os.environ.get("GSM_SYMBOLIC_DIR") or f"{R}/hf_home/hub/datasets--apple--GSM-Symbolic/snapshots/{GSM_SYM_REV}"
# UMWP (Sun et al., LREC-COLING 2024), fetched 2026-09-24 05:30 CST from raw.githubusercontent.com (branch main). The local
# file's git blob equals the blob of data/StandardDataset.jsonl in the head commit of main (GitHub API, checked 2026-09-24);
# probe() re-checks the blob on every build.
UMWP = {"repo": "https://github.com/Yuki-Asuuna/UMWP", "file": "data/StandardDataset.jsonl", "branch": "main",
        "head_commit": "1b3ae0fae01a6c6f178361c716bbb4f7d7d54c3c", "head_commit_date": "2023-10-21",
        "file_last_changed_in": "f5eb429a4a0761fe151d55ec4ef7ff10f5434953", "git_blob": "690944cc038ca45924181f30c1a28076b09eee12",
        "retrieved": "2026-09-24 05:30 CST", "licence": "data CC BY-SA 4.0, code Apache-2.0 (repository README)"}
RELEASE_DATE = "2026-09-24"
SNAPSHOT = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
VERDICTS = ("SOLVABLE", "UNSOLVABLE", "AMBIGUOUS")
MODEL_NAMES = {"qwen3_4b_Base": "Qwen3-4B-Base", "qwen3_8b_Base": "Qwen3-8B-Base", "qwen3_14b_Base": "Qwen3-14B-Base",
               "llama3p2_3b": "Llama-3.2-3B-Instruct", "phi4mini": "Phi-4-mini-instruct"}
RUN_RE = re.compile(r"^(?P<m>qwen3_4b_Base|qwen3_8b_Base|qwen3_14b_Base|llama3p2_3b|phi4mini)_(?P<c>.+)_s(?P<s>\d+)_r(?P<r>\d+)$")
# condition -> (group, one-line description); groups: core_loop, guard_fix, variant, control, rescue
CONDITIONS = {
    "own_a3": ("core_loop", "R-Zero-style label-free loop: frozen proposer, R-Zero band guard p_mode in [0.25, 0.75] (plain prompt), GRPO on the solver's own majority answer (A3)"),
    "own_a4": ("core_loop", "same loop as own_a3 for Phi-4-mini-instruct and Qwen3-14B-Base (A4)"),
    "own_ag_a4": ("guard_fix", "abstention guard: abstention-option prompt, admit iff < 2/8 samples abstain and the band holds on the rest; training as own (A4)"),
    "coev_a4": ("variant", "co-evolving proposer: after each round the proposer is SFT-ed on the admitted proposals; R-Zero band; round 1 = own_a3 round 1 (A4)"),
    "coev_ag_a4": ("variant", "co-evolving proposer with the abstention guard (A4)"),
    "own_tg_a3": ("variant", "tracer-calibrated band: each round the band [lo, hi] maximises intact-minus-deleted tracer admission (A3e)"),
    "own_azr_a3": ("variant", "Absolute-Zero learnability guard: admit p_mode in (0, 1) (A3e)"),
    "own_clean_a3": ("control", "trains only on the admitted proposals judged SOLVABLE by the full Opus audit (+ intact tracers); round 1 = own round 1 (A3)"),
    "own_sub_a3": ("control", "trains on a uniformly random 50% of the admitted set, size-matched control (A3)"),
    "ctrl_rnd_a5": ("control", "plain band, then a random subsample down to the abstention-guard lineage's admitted count (A5 CTRL-L)"),
    "ctrl_lowp_a5": ("control", "plain band, then the lowest-modal-share in-band items dropped down to the guard lineage's count (A5 CTRL-L)"),
    "rescue_uv_a4": ("rescue", "own round-2 solver trained on own round-3 admitted set (copied) with reward own_uv ('unknown' can win the vote) (A4g)"),
    "rescue_uvw_a4": ("rescue", "as rescue_uv with weight w = 3 on 'unknown' votes (A4l)"),
    "rescue_abs_a4": ("rescue", "as rescue_uv but reward own_abs ('unknown' excluded from the vote, zero reward), prompt-matched control (A4f)"),
    "loop_uv_a4": ("rescue", "loop continued from the rescue_uv round-3 solver: abstention prompt, admission on max(p_mode, abstain) in the band, reward own_uv (A4i/A4j)"),
}
EXT_AUDITS = {**{f"qwen3_4b_Base_own_a3_s{s}_r3": f"swap_audit_a5_s{s}" for s in range(5)},
              **{f"phi4mini_own_a4_s{s}_r3": f"phi_audit_a6_s{s}" for s in range(2)}}
STATS = {"missing": [], "decisions": []}


# ------------------------------------------------------------------ utils
def jl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                pass  # a truncated last line of a resumable dump
    return out


def jload(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def wjl(rel, rows):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def wtext(rel, text):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)


def wtsv(rel, header, rows):
    wtext(rel, "\t".join(header) + "\n" + "".join("\t".join("" if v is None else str(v) for v in r) + "\n" for r in rows))


def copy(src, rel):
    p = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    shutil.copyfile(src, p)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path_from_log(d):
    log = os.path.join(d, "guard.log")
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(r"'model': '([^']+)'", line)
                if m:
                    return m.group(1)
    return None


def ck_label(path):
    """<CKPT_DIR>/<run>/final -> '<run>' (the solver after that run's training); a base path -> its model name."""
    if not path:
        return None
    p = path.rstrip("/")
    m = re.search(r"/ckpt[^/]*/([^/]+)/(final|step\d+)$", p)
    if m:
        return m.group(1) if m.group(2) == "final" else f"{m.group(1)}/{m.group(2)}"
    return os.path.basename(p)


def fr(x, n=4):
    return None if x is None else round(x, n)


def verdict_row(v):
    return {"opus_verdict": v["verdict"], "opus_confidence": v.get("confidence"), "opus_note": v.get("note")}


def counts(vs):
    c = collections.Counter(v["verdict"] for v in vs)
    return {k: c.get(k, 0) for k in VERDICTS}


def sha_text(t):
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()


# Personal data in web text that base models sometimes copy after the problem: e-mail addresses, URLs, and the commenter
# names / @-handles of copied Stack Exchange comment threads. Only these spans are masked; everything else is verbatim.
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s\"'<>\[\]{}]*", re.I)
SE_MARKS = ("$\\begingroup$", "$\\endgroup$")
SE_SIG_RE = re.compile(r"(\$\\endgroup\$\s*[–—-]\s*)(.+?)"
                       r"(?=\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\b|\s*\$\\begingroup\$|$)")
SE_AT_RE = re.compile(r"@[A-Za-z]\w*")
TEXTS = collections.defaultdict(set)          # scope -> original problem texts
REDACTED = collections.defaultdict(dict)      # scope -> {original text: {kind: n masked spans}}
REDACTED_ROWS = collections.Counter()         # scope -> rows with redacted = true


def _url(m):
    s = m.group(0)
    core = s
    while core and core[-1] in ".,;:!?)'\"":
        if core[-1] == ")" and core.count("(") >= core.count(")"):
            break
        core = core[:-1]
    return "[URL]" + s[len(core):]


def redact(t):
    """-> (masked text, {kind: n}). [EMAIL] for e-mail addresses, [URL] for URLs and, only inside copied Stack Exchange
    comment threads, [USER] for commenter names and @-handles."""
    if not t:
        return t, {}
    hits = {}
    t, hits["email"] = EMAIL_RE.subn("[EMAIL]", t)
    t, hits["url"] = URL_RE.subn(_url, t)
    if any(s in t for s in SE_MARKS):
        t, n1 = SE_SIG_RE.subn(lambda m: m.group(1) + "[USER]", t)
        t, n2 = SE_AT_RE.subn("@[USER]", t)
        hits["se_user"] = n1 + n2
    return t, {k: v for k, v in hits.items() if v}


def release_text(t, scope):
    """Released form of a problem text: (masked text, SHA-256 of the masked text, redacted flag)."""
    TEXTS[scope].add(t or "")
    m, hits = redact(t)
    if hits:
        REDACTED[scope][t] = hits
        REDACTED_ROWS[scope] += 1
    return m, sha_text(m), bool(hits)


def redaction_summary():
    out = {}
    for scope in sorted(TEXTS):
        h = REDACTED[scope].values()
        out[scope] = {"unique_texts": len(TEXTS[scope]), "texts_redacted": len(REDACTED[scope]),
                      "texts_with_url": sum("url" in x for x in h), "texts_with_email": sum("email" in x for x in h),
                      "texts_with_se_thread": sum("se_user" in x for x in h), "rows_redacted": REDACTED_ROWS[scope]}
    allt = set().union(*TEXTS.values()) if TEXTS else set()
    allr = {t: x for s in REDACTED.values() for t, x in s.items()}
    out["all"] = {"unique_texts": len(allt), "texts_redacted": len(allr), "texts_with_url": sum("url" in x for x in allr.values()),
                  "texts_with_email": sum("email" in x for x in allr.values()),
                  "texts_with_se_thread": sum("se_user" in x for x in allr.values()), "rows_redacted": sum(REDACTED_ROWS.values())}
    return out


def admission_stat(recs, gst):
    """The statistic the guard compared with the band: 'p_mode' (a3_guard.py default) or 'max(p_mode,abstain)'
    (--count_unknown 1, A4i). The flag is not stored in guard_stats.json, so it is read off the records: the rule that
    reproduces every record's admit flag. -> (stat, basis)."""
    if not recs:
        return "p_mode", "no guard records; a3_guard.py default"
    gst = gst or {}
    g, lo, hi = gst.get("guard") or "rzero", gst.get("band_lo", 0.25), gst.get("band_hi", 0.75)
    ok_p = ok_m = sep = 0
    for r in recs:
        extra = g != "abstain" or r["abstain"] < 0.25
        ap = lo <= r["p_mode"] <= hi and extra
        am = lo <= max(r["p_mode"], r["abstain"]) <= hi and extra
        ok_p += ap == r["admit"]; ok_m += am == r["admit"]; sep += ap != am
    n = len(recs)
    if ok_m == n and ok_p < n:
        return "max(p_mode,abstain)", f"max rule reproduces {n}/{n} admit flags, p_mode rule {ok_p}/{n}"
    if ok_p == n:
        return "p_mode", (f"p_mode rule reproduces {n}/{n} admit flags, max rule {ok_m}/{n}" if sep else
                          f"p_mode rule reproduces {n}/{n} admit flags; no record separates the rules (a3_guard.py default)")
    raise SystemExit(f"admission rule not identified: p_mode rule {ok_p}/{n}, max rule {ok_m}/{n}")


def dup_stats(index_json, pool_text_rows):
    """Duplicate texts across the distinct pools and (model, id) pairs that name different texts in different runs."""
    distinct = {x["run"] for x in index_json if x["identical_pool_as"] is None}
    by_text = collections.defaultdict(lambda: (set(), set()))
    n_rows = 0
    for run, model, i, t, verdict in pool_text_rows:
        if run in distinct:
            n_rows += 1
            by_text[t][0].add(run); by_text[t][1].add(verdict)
    multi = [e for e in by_text.values() if len(e[0]) > 1]
    ids = collections.defaultdict(set)
    for run, model, i, t, verdict in pool_text_rows:
        ids[(model, i)].add(t)
    ex = {}
    ra, rb = "qwen3_4b_Base_own_a3_s0_r1", "qwen3_4b_Base_own_azr_a3_s0_r1"
    ta = {i: t for run, model, i, t, v in pool_text_rows if run == ra}
    tb = {i: t for run, model, i, t, v in pool_text_rows if run == rb}
    if ta and tb:
        ex = {"runs": [ra, rb], "shared_texts": len(set(ta.values()) & set(tb.values())),
              "shared_texts_same_id": sum(tb.get(i) == t for i, t in ta.items()),
              "common_ids_other_text": sum(i in tb and tb[i] != t for i, t in ta.items())}
    return {"distinct_pools": len(distinct), "rows": n_rows, "unique_texts": len(by_text), "texts_in_several_pools": len(multi),
            "several_pools_different_verdict": sum(len(e[1]) > 1 for e in multi),
            "several_pools_different_binary_verdict": sum(len({v == "SOLVABLE" for v in e[1]}) > 1 for e in multi),
            "model_id_pairs": len(ids), "model_id_pairs_with_several_texts": sum(len(s) > 1 for s in ids.values()), "example": ex}


# ------------------------------------------------------------------ 1. audited admitted pools
def audited_pools():
    runs = sorted(d for d in os.listdir(B) if os.path.exists(f"{B}/{d}/audit_admitted.json") and RUN_RE.match(d))
    info = {}
    for run in runs:
        m = RUN_RE.match(run).groupdict()
        A = jl(f"{B}/{run}/admitted.jsonl")
        prop = [a for a in A if a["kind"] == "proposed"]
        order = list(prop)
        random.Random(0).shuffle(order)  # exactly the list the judges' fetch command builds and shuffles
        info[run] = {"m": m, "A": A, "prop": prop, "pset": frozenset(a["id"] for a in prop),
                     "rpos": {p["id"]: k for k, p in enumerate(order)}}
        assert len(info[run]["rpos"]) == len(prop), f"duplicate proposer ids in {run}"
    # identical pools (same admitted proposer ids): copies made by construction (rescue arms, coev round 1)
    by_set = collections.defaultdict(list)
    for run, x in info.items():
        by_set[(x["m"]["m"], x["pset"])].append(run)
    order = ["own_a3", "own_a4"]
    canon = {}
    for key, rs in by_set.items():
        rs = sorted(rs, key=lambda r: (not os.path.exists(f"{B}/{r}/guard_records.jsonl"),
                                       order.index(info[r]["m"]["c"]) if info[r]["m"]["c"] in order else 9, r))
        for r in rs:
            canon[r] = rs[0]
    # verdicts: sample200 first, then full audit, then extension parts; first verdict of an id wins
    def collect(run):
        pset, rpos = info[run]["pset"], info[run]["rpos"]
        V, src, conflicts, extra = {}, collections.Counter(), 0, 0
        kseen, kbad = set(), set()
        def add(v, source):
            nonlocal conflicts, extra
            if v.get("verdict") not in VERDICTS or not v.get("id"):
                return
            if v["id"] not in pset:
                extra += 1
                return
            if source in ("sample200", "full_audit") and v.get("k") is not None:
                kseen.add(v["id"])
                if v["k"] != rpos[v["id"]]:
                    kbad.add(v["id"])
            if v["id"] in V:
                conflicts += int(V[v["id"]][0]["verdict"] != v["verdict"])
                return
            V[v["id"]] = (v, source)
            src[source] += 1
        a = jload(f"{B}/{run}/audit_admitted.json")
        for v in a["verdicts"]:
            add(v, "sample200")
        if os.path.exists(f"{B}/{run}/audit_full.json"):
            for v in jload(f"{B}/{run}/audit_full.json")["verdicts"]:
                add(v, "full_audit")
        for p in sorted(glob.glob(f"{B}/{run}/audit_parts/*.json")):
            try:
                for v in jload(p).get("verdicts", []):
                    add(v, "full_audit")
            except ValueError:
                pass
        if run in EXT_AUDITS:
            for p in sorted(glob.glob(f"{B}/{EXT_AUDITS[run]}/audit_parts/*.json")):
                for v in jload(p).get("verdicts", []):
                    add(v, f"extension:{EXT_AUDITS[run]}")
        return V, src, conflicts, extra, a, collections.Counter({"k_checked": len(kseen), "k_mismatch": len(kbad)})
    col = {run: collect(run) for run in runs}
    index, index_json = [], []
    tracer_rows, pool_text_rows = [], []
    total_rows = 0
    for run in runs:
        x = info[run]; m = x["m"]; c0 = canon[run]; rpos = x["rpos"]
        V, src, conflicts, extra, a, kchk = col[run]
        if c0 != run:  # inherit the canonical pool's extension verdicts (same items, same judges)
            for i, vs in col[c0][0].items():
                if i not in V:
                    V[i] = vs; src[vs[1] + f" (via {c0})"] += 1
        # solver agreement at admission: guard records of this run, else of the identical canonical pool
        rec_run = run if os.path.exists(f"{B}/{run}/guard_records.jsonl") else (c0 if os.path.exists(f"{B}/{c0}/guard_records.jsonl") else None)
        G, recs = {}, []
        solver_path = None
        if rec_run:
            recs = jl(f"{B}/{rec_run}/guard_records.jsonl")
            G = {g["id"]: g for g in recs}
            solver_path = model_path_from_log(f"{B}/{rec_run}")
        cfg = jload(f"{B}/{run}/config.json") if os.path.exists(f"{B}/{run}/config.json") else {}
        if not solver_path:
            cfg_src = jload(f"{B}/{rec_run}/config.json") if rec_run and os.path.exists(f"{B}/{rec_run}/config.json") else cfg
            solver_path = cfg_src.get("model")
        solver = ck_label(solver_path)
        solver_is_base = solver in MODEL_NAMES.values()
        gst = jload(f"{B}/{rec_run or run}/guard_stats.json") if os.path.exists(f"{B}/{rec_run or run}/guard_stats.json") else {}
        ast, ast_basis = admission_stat(recs, gst)
        k200 = {v["id"]: v.get("k") for v in a["verdicts"]}
        kchk["sample200_outside_first200"] += sum(rpos[i] >= 200 for i in k200 if i in rpos)
        ids = sorted(V, key=lambda i: rpos[i])
        textof = {p["id"]: (p.get("problem") or p.get("prompt")) for p in x["prop"]}
        rows = []
        for i in ids:
            v, source = V[i]
            g = G.get(i)
            text, tsha, red = release_text(textof[i], "audited_pools")
            row = {"lineage": f"{m['m']}_{m['c']}_s{m['s']}", "run": run, "model": MODEL_NAMES[m["m"]], "condition": m["c"],
                   "seed": int(m["s"]), "round": int(m["r"]), "id": i, "problem": text, "problem_sha256": tsha, "redacted": red,
                   **verdict_row(v), "in_audit_sample200": i in k200, "audit_k": rpos[i], "audit_source": source,
                   "solver_checkpoint": solver, "solver_p_mode": g["p_mode"] if g else None,
                   "solver_mode": g["mode"] if g else None, "solver_abstain": g["abstain"] if g else None,
                   "base_p_mode": (g["p_mode"] if g else None) if solver_is_base else None,
                   "base_mode": (g["mode"] if g else None) if solver_is_base else None,
                   "base_abstain": (g["abstain"] if g else None) if solver_is_base else None}
            rows.append(row)
            pool_text_rows.append((run, MODEL_NAMES[m["m"]], i, textof[i], v["verdict"]))
        total_rows += wjl(f"audited_pools/{run}.jsonl", rows)
        s200 = [V[i][0] for i in ids if i in k200]
        cs = counts(s200); ca = counts([V[i][0] for i in ids])
        kinds = collections.Counter(a_["kind"] for a_ in x["A"])
        n_missing_solver = sum(r["solver_p_mode"] is None for r in rows)
        grp = CONDITIONS.get(m["c"], ("other", ""))[0]
        rec = {"run": run, "lineage": f"{m['m']}_{m['c']}_s{m['s']}", "model": MODEL_NAMES[m["m"]], "condition": m["c"], "group": grp,
               "seed": int(m["s"]), "round": int(m["r"]), "solver_checkpoint": solver, "solver_records_from": rec_run,
               "guard": (gst.get("guard") or "rzero") if gst else None, "admission_stat": ast,
               "band_lo": gst.get("band_lo", 0.25 if gst else None), "band_hi": gst.get("band_hi", 0.75 if gst else None),
               "n_admitted": len(x["A"]), "n_admitted_proposed": kinds["proposed"], "n_admitted_tracer_deleted": kinds["tracer_deleted"],
               "n_admitted_tracer_intact": kinds["tracer_intact"], "n_audited": len(rows),
               "coverage": "full" if len(rows) == kinds["proposed"] else "sample200" if len(rows) == len(s200) else "partial",
               "sample200_SOLVABLE": cs["SOLVABLE"], "sample200_UNSOLVABLE": cs["UNSOLVABLE"], "sample200_AMBIGUOUS": cs["AMBIGUOUS"],
               "sample200_unsolvable_share": fr(cs["UNSOLVABLE"] / max(1, len(s200))),
               "sample200_illposed_share": fr((cs["UNSOLVABLE"] + cs["AMBIGUOUS"]) / max(1, len(s200))),
               "all_SOLVABLE": ca["SOLVABLE"], "all_UNSOLVABLE": ca["UNSOLVABLE"], "all_AMBIGUOUS": ca["AMBIGUOUS"],
               "identical_pool_as": None if c0 == run else c0, "audit_protocol": a.get("protocol"), "audit_judge": a.get("judge"),
               "verdict_sources": dict(src), "duplicate_verdict_conflicts": conflicts, "verdicts_outside_admitted_set": extra,
               "audit_k_check": {k: kchk.get(k, 0) for k in ("k_checked", "k_mismatch", "sample200_outside_first200")},
               "rows_redacted": sum(r["redacted"] for r in rows), "rows_without_solver_record": n_missing_solver,
               "admission_stat_basis": ast_basis, "snapshot_time": SNAPSHOT, "guard_stats": gst}
        index_json.append(rec)
        index.append([rec[k] for k in INDEX_COLS])
        if rec_run == run:
            for g in G.values():
                if g["kind"] != "proposed":
                    _, fam, task = g["id"].split(":")
                    tracer_rows.append({"source": f"run:{run}", "solver_checkpoint": solver, "id": g["id"], "kind": g["kind"],
                                        "fam": int(fam), "task": int(task), "p_mode": g["p_mode"], "mode": g["mode"],
                                        "abstain": g["abstain"], "admit": g["admit"]})
    wtsv("audited_pools/INDEX.tsv", INDEX_COLS, index)
    wjl("audited_pools/INDEX.jsonl", index_json)
    STATS["pools"] = index_json
    STATS["pool_rows"] = total_rows
    STATS["pool_text_rows"] = pool_text_rows
    STATS["dups"] = dup_stats(index_json, pool_text_rows)
    return tracer_rows


INDEX_COLS = ["run", "lineage", "model", "condition", "group", "seed", "round", "solver_checkpoint", "solver_records_from", "guard",
              "admission_stat", "band_lo", "band_hi", "n_admitted", "n_admitted_proposed", "n_admitted_tracer_deleted",
              "n_admitted_tracer_intact", "n_audited", "coverage", "sample200_SOLVABLE", "sample200_UNSOLVABLE", "sample200_AMBIGUOUS",
              "sample200_unsolvable_share", "sample200_illposed_share", "all_SOLVABLE", "all_UNSOLVABLE", "all_AMBIGUOUS",
              "identical_pool_as"]


# ------------------------------------------------------------------ 2. raw proposal audits (pre-guard)
def raw_audits():
    out = []
    for p in sorted(glob.glob(f"{B}/*/audit_raw.json")):
        run = os.path.basename(os.path.dirname(p))
        m = RUN_RE.match(run).groupdict()
        a = jload(p)
        P = {x["id"]: x for x in jl(f"{B}/{run}/proposed.jsonl")}
        rows = []
        for v in a["verdicts"]:
            if v["id"] not in P:
                continue
            text, tsha, red = release_text(P[v["id"]].get("problem"), "raw_proposal_audits")
            rows.append({"lineage": f"{m['m']}_{m['c']}_s{m['s']}", "run": run, "model": MODEL_NAMES[m["m"]], "condition": m["c"],
                         "seed": int(m["s"]), "round": int(m["r"]), "id": v["id"], "problem": text, "problem_sha256": tsha,
                         "redacted": red, **verdict_row(v), "audit_k": v.get("k")})
        wjl(f"raw_proposal_audits/{run}.jsonl", rows)
        out.append({"run": run, "n": len(rows), **counts(a["verdicts"]), "n_proposed": len(P), "protocol": a.get("protocol")})
    STATS["raw"] = out


# ------------------------------------------------------------------ 3. fixed-pool full audits
UV_NOTE = ("abstention prompt, admission on p_mode (R-Zero band): these guard passes ran without --count_unknown, so admit = "
           "p_mode in [0.25, 0.75]; the loop_uv rule max(p_mode, abstain) in the band can be recomputed from p_mode and abstain")
FIXED = {
    "qwen_s0_r1": {"texts": f"{B}/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl", "proposer": "Qwen3-4B-Base",
                   "what": "all round-1 raw proposals of the Qwen3-4B own seed-0 lineage (frozen proposer); mechanism train pool (A3 2d)",
                   "diags": [("qwen_s0", "plain"), ("qwen_s0_ag", "abstention"), ("scale_qwen3_14b", "plain"),
                             ("scale_qwen25_32b", "plain"), ("uv/pool_s0", "abstention")]},
    "qwen_s1_r1": {"texts": f"{B}/qwen3_4b_Base_own_a3_s1_r1/proposed.jsonl", "proposer": "Qwen3-4B-Base",
                   "what": "all round-1 raw proposals of the Qwen3-4B own seed-1 lineage; held-out validation pool (A4a.1)",
                   "diags": [("qwen_s1", "plain"), ("qwen_s1_ag", "abstention"), ("uv/pool_s1", "abstention")]},
    "front_q25_7b": {"texts": f"{R}/dirA3/results/frontier/q25_7b/proposed.jsonl", "proposer": "Qwen2.5-7B-Instruct",
                     "what": "{n}-candidate pool (from 1,200 requests) of a stronger frozen proposer (same prompt, seed 7); proposer-side escape test (A4c)",
                     "diags": [("front_q25_7b", "plain")]},
    "evo_r3": {"texts": f"{R}/dirA3/results/frontier/evo_r3/proposed.jsonl", "proposer": "Qwen3-4B-Base (rewriting)",
               "what": "harder variants written by the frozen proposer from the round-3 solver's unanimous qwen_s0_r1 candidates (A4e)",
               "diags": [("evo_r3", "plain")]},
}


def fixed_pools():
    tracer_rows, ck_rows, out = [], [], []
    for name, spec in FIXED.items():
        P = jl(spec["texts"])
        V = {}
        for p in sorted(glob.glob(f"{PA}/{name}/parts/*.json")):
            for v in jload(p)["verdicts"]:
                if v.get("verdict") in VERDICTS and v["id"] not in V:
                    V[v["id"]] = v
        agree = {}
        for diag, prompt in spec["diags"]:
            for d in sorted(glob.glob(f"{GD}/{diag}/*/guard_records.jsonl")):
                dd = os.path.dirname(d)
                ck = os.path.basename(dd)
                recs = jl(d)
                path = model_path_from_log(dd)
                label = ck_label(path)
                note = ""
                if diag.startswith("scale_") and ck != "base":
                    # the vendor's post-trained model, not a checkpoint of our training: key by its name, not by 'r1'
                    col = f"{diag}/{label}"
                    note = f"vendor post-trained model {label}, not a checkpoint of our training (source directory named {ck})"
                else:
                    col = f"{diag}/{ck}"
                if diag.startswith("uv/"):
                    note = UV_NOTE
                agree[col] = {r["id"]: r for r in recs}
                gst = jload(os.path.join(dd, "guard_stats.json")) if os.path.exists(os.path.join(dd, "guard_stats.json")) else {}
                ast, _ = admission_stat(recs, gst)
                ck_rows.append([name, col, os.path.relpath(dd, R), label, prompt, gst.get("guard") or "rzero", ast,
                                gst.get("band_lo", 0.25), gst.get("band_hi", 0.75), sum(r["kind"] == "proposed" for r in recs),
                                sum(r["kind"] != "proposed" for r in recs), note])
                for r in recs:
                    if r["kind"] != "proposed":
                        _, fam, task = r["id"].split(":")
                        tracer_rows.append({"source": f"guard_diag:{col}", "solver_checkpoint": label, "id": r["id"], "kind": r["kind"],
                                            "fam": int(fam), "task": int(task), "p_mode": r["p_mode"], "mode": r["mode"],
                                            "abstain": r["abstain"], "admit": r["admit"]})
        rows = []
        for k, p in enumerate(P):
            v = V.get(p["id"])
            if v is None:
                continue
            ag = {col: {"p_mode": a[p["id"]]["p_mode"], "mode": a[p["id"]]["mode"], "abstain": a[p["id"]]["abstain"],
                        "admit": a[p["id"]]["admit"]} for col, a in agree.items() if p["id"] in a}
            text, tsha, red = release_text(p.get("problem"), "fixed_pool_audits")
            row = {"pool": name, "proposer": spec["proposer"], "id": p["id"], "k": k, "problem": text, "problem_sha256": tsha,
                   "redacted": red, "stem": p.get("stem"), **verdict_row(v), "agreement": ag}
            if "seed_id" in p:
                row["evolved_from"] = p["seed_id"]
            rows.append(row)
        wjl(f"fixed_pool_audits/{name}.jsonl", rows)
        out.append({"pool": name, "n": len(rows), "n_candidates": len(P), **counts(rows_to_v(rows)), "checkpoints": list(agree),
                    "proposer": spec["proposer"], "what": spec["what"].replace("{n}", f"{len(P):,}")})
    wtsv("fixed_pool_audits/CHECKPOINTS.tsv", ["pool", "agreement_key", "source_dir", "solver_checkpoint", "prompt", "guard",
                                                "admission_stat", "band_lo", "band_hi", "n_candidates", "n_tracers", "note"], ck_rows)
    STATS["fixed"] = out
    STATS["checkpoints"] = ck_rows
    # agreement of the fixed-pool audit with the run-level audit of the same items (independent judge instances?)
    ag = []
    for name, run in (("qwen_s0_r1", "qwen3_4b_Base_own_a3_s0_r1"), ("qwen_s1_r1", "qwen3_4b_Base_own_a3_s1_r1")):
        F = {r["id"]: r["opus_verdict"] for r in jl(f"{OUT}/fixed_pool_audits/{name}.jsonl")}
        Rv = {r["id"]: r["opus_verdict"] for r in jl(f"{OUT}/audited_pools/{run}.jsonl")}
        both = [i for i in Rv if i in F]
        same = sum(F[i] == Rv[i] for i in both)
        ag.append({"pool": name, "run": run, "n_overlap": len(both), "identical_verdicts": same,
                   "kappa3": kappa([(Rv[i], F[i]) for i in both]),
                   "kappa_binary": kappa([(Rv[i] == "SOLVABLE", F[i] == "SOLVABLE") for i in both])})
    STATS["pool_vs_run"] = ag
    return tracer_rows


def rows_to_v(rows):
    return [{"verdict": r["opus_verdict"]} for r in rows]


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return fr((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


# ------------------------------------------------------------------ 4. twins (patches against GSM-Symbolic)
MINC = 6


def encode(src, tgt):
    """Greedy copy/literal patch of tgt against src: [start, end] copies src[start:end], a string is a literal."""
    ops, lit, j = [], [], 0
    while j < len(tgt):
        L = 0
        if j + MINC <= len(tgt) and tgt[j:j + MINC] in src:
            L = MINC
            while j + L < len(tgt) and tgt[j:j + L + 1] in src:
                L += 1
        if L:
            if lit:
                ops.append("".join(lit)); lit = []
            s = src.find(tgt[j:j + L])
            ops.append([s, s + L]); j += L
        else:
            lit.append(tgt[j]); j += 1
    if lit:
        ops.append("".join(lit))
    return ops


def decode(src, ops):
    return "".join(src[o[0]:o[1]] if isinstance(o, list) else o for o in ops)


def twins(tracer_rows):
    G = {}
    for cfg in ("main", "p1", "p2"):
        for x in jl(f"{GSM_SYM}/{cfg}/test.jsonl"):
            G[(cfg, str(x["id"]), str(x["instance"]))] = x
    T = jl(f"{R}/data/hp_twins.jsonl")
    probe = [t for t in T if t["task"] == 0 or t["split_kind"] == "template"]
    pidx = {(t["fam"], t["task"]): i for i, t in enumerate(probe)}
    rows, lit, tot = [], 0, 0
    for t in T:
        m = t["meta"]
        key = (m["cfg"], str(m["gsm_template"]), str(m["base_instance"]))
        src = G[key]["question"]
        rec = {"fam": t["fam"], "task": t["task"], "split": t["split"], "split_kind": t["split_kind"],
               "family_id_deleted": t["family_id_deleted"], "family_id_intact": t["family_id_intact"], "answer_intact": t["answer"],
               "in_probe543": (t["fam"], t["task"]) in pidx, "probe543_idx": pidx.get((t["fam"], t["task"])),
               "meta": {k: m[k] for k in ("template", "cfg", "gsm_template", "base_instance", "orig", "hidden_name", "H", "vars", "depth") if k in m},
               "source": {"dataset": "apple/GSM-Symbolic", "revision": GSM_SYM_REV, "config": m["cfg"], "id": int(m["gsm_template"]),
                          "instance": int(m["base_instance"])}}
        for form in ("intact", "deleted"):
            ops = encode(src, t[form])
            assert decode(src, ops) == t[form]
            rec[f"{form}_ops"] = ops
            rec[f"{form}_sha256"] = hashlib.sha256(t[form].encode("utf-8")).hexdigest()
            lit += sum(len(o) for o in ops if isinstance(o, str)); tot += len(t[form])
        rows.append(rec)
    wjl("twins/twin_patches.jsonl", rows)
    STATS["twins"] = {"n": len(rows), "probe543": len(probe), "literal_share": fr(lit / tot, 4),
                      "splits": dict(collections.Counter(t["split_kind"] for t in T))}
    # validations of the probe set: pass 1 (28 judges, intact shown) and pass 2 (solve-first, deleted only, 150 stratified)
    V1 = jload(f"{R}/dirA/results/twin_validation_probe543.json")
    V2 = {v["idx"]: v for v in jload(f"{R}/dirA/results/twin_validation_pass2.json")["verdicts"]}
    vrows = []
    for v in sorted(V1["verdicts"], key=lambda v: v["idx"]):
        t = probe[v["idx"]]
        assert str(t["fam"]) == str(v["fam"]) and int(t["task"]) == int(v["task"])
        p2 = V2.get(v["idx"])
        vrows.append({"probe543_idx": v["idx"], "fam": t["fam"], "task": t["task"], "split_kind": t["split_kind"],
                      "pass1_verdict": v["verdict"], "pass1_confidence": v["confidence"], "pass1_note": v["note"],
                      "pass2_solvable": p2["solvable"] if p2 else None, "pass2_answer_if_solvable": p2["answer_if_solvable"] if p2 else None,
                      "pass2_missing_if_not": p2["missing_if_not"] if p2 else None, "pass2_confidence": p2["confidence"] if p2 else None})
    wjl("twins/probe543_validation.jsonl", vrows)
    D5 = jl(f"{R}/dirC/data/hp_deleted500.jsonl")
    V3 = jload(f"{R}/dirC/results/twin_validation_200.json")
    r3 = []
    for v in V3["verdicts"]:
        d = D5[v["idx"]]
        r3.append({"hp_deleted500_idx": v["idx"], "fam": d["fam"], "task": d["task"], "verdict": v["verdict"],
                   "confidence": v["confidence"], "note": v["note"]})
    wjl("twins/random200_validation.jsonl", r3)
    STATS["twin_val"] = {"pass1": V1["counts"], "pass2_n": len(V2), "pass2_solvable": sum(v["solvable"] for v in V2.values()),
                         "random200": V3["counts"], "kappa_txt": open(f"{R}/dirC/results/twin_validation_kappa.txt").read().strip()}
    wjl("twins/tracer_guard_records.jsonl", tracer_rows)
    STATS["tracer_rows"] = len(tracer_rows)
    wtext("twins/rebuild_twins.py", REBUILD_TWINS)


REBUILD_TWINS = r'''#!/usr/bin/env python3
"""Rebuild the GSM-HP premise-deletion twins (intact + deleted texts) from your own copy of GSM-Symbolic.

The twins are adaptations of GSM-Symbolic (Apple; CC BY-NC-ND 4.0), which does not permit sharing adapted material, so this
release ships only patches: each text is a list of copy operations [start, end] into the GSM-Symbolic question it was built
from, plus short literals (numbers, number words, names; about 2% of the characters). Every rebuilt text is checked
against the SHA-256 stored in twin_patches.jsonl.

  pip install datasets   (or download main/p1/p2 test.jsonl of apple/GSM-Symbolic at the pinned revision)
  python3 rebuild_twins.py --patches twin_patches.jsonl --out twins_rebuilt.jsonl [--gsm_symbolic_dir DIR]

Output: one row per twin, in the order of the project's data/hp_twins.jsonl, with the fields c0_probe.py reads: fam, task,
split, split_kind, family_id_deleted, family_id_intact, answer (the intact problem's answer), intact, deleted, meta; plus
answer_intact (= answer), in_probe543, probe543_idx and canary (copied from GSM-Symbolic). The file is a drop-in
--twins_file for probe/code/c0_probe.py (its probe set = task 0 of every family + every template-split task = the 543
registered items, in probe543_idx order). Not rebuilt: the project file's deleted_sentence and meta.H_sentence (text).
Keep the canary: like GSM-Symbolic itself, the rebuilt twins should not appear in training corpora.
"""
import argparse, hashlib, json, os

REV = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
ap = argparse.ArgumentParser()
ap.add_argument("--patches", default="twin_patches.jsonl")
ap.add_argument("--out", default="twins_rebuilt.jsonl")
ap.add_argument("--gsm_symbolic_dir", default="", help="directory with main/test.jsonl, p1/test.jsonl, p2/test.jsonl")
a = ap.parse_args()
G = {}
for cfg in ("main", "p1", "p2"):
    if a.gsm_symbolic_dir:
        rows = [json.loads(l) for l in open(os.path.join(a.gsm_symbolic_dir, cfg, "test.jsonl"), encoding="utf-8")]
    else:
        from datasets import load_dataset
        rows = load_dataset("apple/GSM-Symbolic", cfg, revision=REV)["test"]
    for x in rows:
        G[(cfg, int(x["id"]), int(x["instance"]))] = x
bad = 0
with open(a.out, "w", encoding="utf-8") as fh:
    for l in open(a.patches, encoding="utf-8"):
        t = json.loads(l)
        s = t["source"]; x = G[(s["config"], s["id"], s["instance"])]
        src = x["question"]
        txt = {}
        for form in ("intact", "deleted"):
            txt[form] = "".join(src[o[0]:o[1]] if isinstance(o, list) else o for o in t[form + "_ops"])
            bad += hashlib.sha256(txt[form].encode("utf-8")).hexdigest() != t[form + "_sha256"]
        out = {"fam": t["fam"], "task": t["task"], "split": t["split"], "split_kind": t["split_kind"],
               "family_id_deleted": t["family_id_deleted"], "family_id_intact": t["family_id_intact"],
               "answer": t["answer_intact"], "intact": txt["intact"], "deleted": txt["deleted"], "meta": t["meta"],
               "answer_intact": t["answer_intact"], "in_probe543": t["in_probe543"], "probe543_idx": t["probe543_idx"],
               "canary": x.get("canary")}
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
print("rebuilt; sha256 mismatches:", bad)
raise SystemExit(1 if bad else 0)
'''


# ------------------------------------------------------------------ 5. judge prompts, protocol, validation sets
PROMPT_ADMITTED = """You are a careful mathematician auditing self-generated training questions.
Step 1: fetch questions <A>..<B-1> with this exact command (Bash tool, one call):
<SSH> 'cd <RUN_DIR>; python3 -c "
import json, random
A=[json.loads(l) for l in open(\\"admitted.jsonl\\")]
P=[a for a in A if a[\\"kind\\"]==\\"proposed\\"]
random.Random(0).shuffle(P)
for k in range(<A>,<B>):
    print(\\"=== k\\", k, \\"id\\", P[k][\\"id\\"]); print(P[k][\\"problem\\"])
"'
These were written by a <MODEL> language model and ADMITTED by a self-play filter. For EACH question decide:
  SOLVABLE    - the text determines a unique numeric answer (state it briefly in the note);
  UNSOLVABLE  - a needed quantity/relation is missing, or the text is self-contradictory, so no unique answer exists;
  AMBIGUOUS   - two reasonable readings give different answers.
Be strict: a default assumption you would have to invent makes it UNSOLVABLE. Confidence 0-1, note <= 20 words.
Step 2: write ALL <M> verdicts as one JSON object {"verdicts": [{"k": 0, "id": "...", "verdict": "SOLVABLE", "confidence": 0.9, "note": "..."}, ...]} to the local file <LOCAL_PART> with the Write tool (plain JSON, nothing else in the file).
Step 3: push and verify with this exact command (Bash tool, one call):
<SSH> "cat > <PART_FILE>" < <LOCAL_PART> && <SSH> "python3 <REPO>/dirA3/code/a3_part_check.py <PART_FILE>"
The printed number must equal <M>. If it does, set pushed=true and verified_n to that number; otherwise retry the push once, and if it still fails set pushed=false.
Step 4: delete the local file: rm <LOCAL_PART>
Return run="<RUN>", a=<A>, b=<B>, n=<M>, counts (SOLVABLE/UNSOLVABLE/AMBIGUOUS), pushed, verified_n. Include the verdicts array in your return ONLY if pushed is false.
"""
PROMPT_R1_V2 = """You are a careful mathematician auditing self-generated training questions. Fetch questions <A>..<B-1> of the sample with this exact command (Bash tool, one call):
<SSH> 'cd <RUN_DIR>; python3 -c "
import json, random
A=[json.loads(l) for l in open(\\"admitted.jsonl\\")]
P=[a for a in A if a[\\"kind\\"]==\\"proposed\\"]
random.Random(0).shuffle(P); S=P[:200]
for k in range(<A>,<B>):
    print(\\"=== k\\", k, \\"id\\", S[k][\\"id\\"]); print(S[k][\\"problem\\"])
"'
These were written by a <4B|3B> language model and ADMITTED by a self-play filter. For EACH question decide:
  SOLVABLE    - the text determines a unique numeric answer (state it briefly in the note);
  UNSOLVABLE  - a needed quantity/relation is missing, or the text is self-contradictory, so no unique answer exists;
  AMBIGUOUS   - two reasonable readings give different answers.
Be strict: a default assumption you would have to invent makes it UNSOLVABLE. Confidence 0-1, note <= 20 words. Return all <M> verdicts with the printed k and id.
"""
PROMPT_POOL = PROMPT_ADMITTED.replace(
    'A=[json.loads(l) for l in open(\\"admitted.jsonl\\")]\nP=[a for a in A if a[\\"kind\\"]==\\"proposed\\"]\nrandom.Random(0).shuffle(P)\n',
    'P=[json.loads(l) for l in open(\\"proposed.jsonl\\")]\n').replace(
    "language model and ADMITTED by a self-play filter.", "language model as self-play training candidates.")
PROMPT_TWIN1 = """You are a careful mathematician validating a benchmark of ILL-POSED math word problems. Fetch twins <A>..<B-1> (0-based index into the registered probe set) with this exact command (Bash tool, one call, do not modify it):
<SSH> 'cd <REPO>; python3 -c "
import json
tw=[json.loads(l) for l in open(\\"data/hp_twins.jsonl\\")]
P=[r for r in tw if r[\\"task\\"]==0 or r[\\"split_kind\\"]==\\"template\\"]
for i in range(<A>,<B>):
    r=P[i]; print(\\"=== idx\\", i, \\"fam\\", r[\\"fam\\"], \\"task\\", r[\\"task\\"]); print(\\"INTACT:\\", r[\\"intact\\"]); print(\\"DELETED:\\", r[\\"deleted\\"]); print(\\"ANSWER (of intact):\\", r[\\"answer\\"])
"'
For EACH twin decide, about the DELETED problem on its own (use the intact one only to see what was removed):
  UNSOLVABLE          - the deleted text does not determine a unique numeric answer (a needed quantity or relation is missing);
  SOLVABLE_SAME       - still uniquely solvable AND the same answer as the intact one (the removed sentence was redundant);
  SOLVABLE_DIFFERENT  - uniquely solvable but the answer differs (the deletion turned it into another well-posed problem, e.g. an additive fee/leg/discount was removed);
  AMBIGUOUS           - cannot tell / the intact problem itself is broken.
Be strict: "a reasonable default assumption would fill the gap" still counts as UNSOLVABLE. Give confidence 0-1 and a <= 20-word note. Return all <M> verdicts with the correct idx, fam and task exactly as printed.
"""
PROMPT_TWIN2 = """You are a strong mathematician. You will be shown math word problems ONE AT A TIME with no reference solution and no companion version. Fetch them with this exact command (Bash tool, one call):
<SSH> 'cd <REPO>; python3 -c "
import json
tw=[json.loads(l) for l in open(\\"data/hp_twins.jsonl\\")]
P=[r for r in tw if r[\\"task\\"]==0 or r[\\"split_kind\\"]==\\"template\\"]
for i in <IDX_LIST>:
    print(\\"=== idx\\", i); print(P[i][\\"deleted\\"])
"'
For EACH problem, TRY TO SOLVE IT. Report solvable=true only if the text as written determines a unique numeric answer without any assumption you had to invent (give the answer). Otherwise solvable=false and name in <= 15 words which quantity or relation is missing. Confidence 0-1. Return all <M> verdicts with the printed idx.
"""
PROMPT_TRACER = PROMPT_ADMITTED.replace(
    'A=[json.loads(l) for l in open(\\"admitted.jsonl\\")]\nP=[a for a in A if a[\\"kind\\"]==\\"proposed\\"]\nrandom.Random(0).shuffle(P)\n',
    'P=[json.loads(l) for l in open(\\"sample.jsonl\\")]\n')
PROMPT_REJUDGE = PROMPT_ADMITTED.replace("random.Random(0).shuffle(P)\n", "random.Random(0).shuffle(P)\nP=P[:200][::-1]\n")
PROMPT_NOTE = """# Placeholders (the only edits to the verbatim workflow text)
# <SSH>          ssh command to the compute host (host, port and user removed)
# <REPO>         project root on the compute host; <RUN_DIR> = the directory holding the item file (<REPO>/dirB/runs/<RUN>)
# <PART_FILE>    <RUN_DIR>/audit_parts/<A>-<B>.json (fixed pools: <REPO>/dirA3/results/pool_audit/<POOL>/parts/<A>-<B>.json)
# <LOCAL_PART>   a temporary local file of the judge (workstation path removed)
# <A>, <B>, <M>  item slice [A, B) of the ordered item list, M = B - A (20 or 25)
# <MODEL>        the proposer's model name as written in the workflow (e.g. Qwen3-4B-Base, Llama-3.2-3B, Phi-4-mini-instruct)
# The judge's structured return schema: verdict in {SOLVABLE, UNSOLVABLE, AMBIGUOUS}, confidence (number), note (string), k, id.
# The fetch command and the returned run="<RUN>" show the judge the run directory and name (see PROTOCOL.md).
"""
NOTE_MAIN = ("# Used by: every admitted-pool audit (200-item sample and full), the extension audits (pseudo-runs swap_audit_a5_s* /\n"
             "# phi_audit_a6_s*) and the UMWP-400 validation (<RUN> = umwp400_j1, <RUN_DIR> = <REPO>/dirB/runs/umwp400_j1,\n"
             "# <MODEL> = Qwen3-4B-Base; its admitted.jsonl holds the 400 UMWP questions with ids umwp:<UMWP id>).\n")
NOTE_TRACER = ("# Tracer-200 validation (workflow a3-audit-validation): <RUN> = audit_validation, <RUN_DIR> =\n"
               "# <REPO>/dirA3/results/audit_validation, <PART_FILE> = <RUN_DIR>/parts/<A>-<B>.json, <MODEL> = Qwen3-4B-Base.\n"
               "# The fetch command reads sample.jsonl in file order (no shuffle; k = line index); the printed ids trd:<fam>:<task>\n"
               "# (premise deleted) and tri:<fam>:<task> (intact) were visible to the judge.\n")
NOTE_REJUDGE = ("# The two 200-item re-judges (workflows a3-audit-rejudge-*): <RUN> = qwen3_4b_Base_own_a3_s0_r3 (<MODEL> = Qwen3-4B-Base)\n"
                "# and llama3p2_3b_own_a3_s3_r2 (<MODEL> = Llama-3.2-3B-Instruct); <RUN_DIR> = <REPO>/dirB/runs/<RUN>, <PART_FILE> =\n"
                "# <REPO>/dirA3/results/audit_rejudge/<RUN>/parts/<A>-<B>.json. The added line P=P[:200][::-1] shows the first audit's\n"
                "# 200-item sample in reverse order (k here = 199 - audit_k); the release matches the two panels by id.\n")


def judge():
    assert "proposed.jsonl" in PROMPT_POOL and "as self-play training candidates" in PROMPT_POOL and "random.Random(0)" not in PROMPT_POOL
    assert PROMPT_TRACER.count("sample.jsonl") == 1 and "random.Random(0)" not in PROMPT_TRACER and "admitted.jsonl" not in PROMPT_TRACER
    assert PROMPT_REJUDGE.count("P=P[:200][::-1]\n") == 1
    wtext("judge/prompt_admitted_pool_audit.txt", PROMPT_NOTE + NOTE_MAIN + "\n" + PROMPT_ADMITTED)
    wtext("judge/prompt_admitted_pool_audit_round1_v2.txt", PROMPT_NOTE + "\n" + PROMPT_R1_V2)
    wtext("judge/prompt_tracer200_validation.txt", PROMPT_NOTE + NOTE_TRACER + "\n" + PROMPT_TRACER)
    wtext("judge/prompt_rejudge200_reversed.txt", PROMPT_NOTE + NOTE_REJUDGE + "\n" + PROMPT_REJUDGE)
    wtext("judge/prompt_raw_pool_audit.txt", PROMPT_NOTE + "# Pool audits (fixed_pool_audits/): items in proposed.jsonl order, k = line index. Raw proposal audits\n# (raw_proposal_audits/): the same text after a Random(0) shuffle of proposed.jsonl, first 200.\n\n" + PROMPT_POOL)
    wtext("judge/prompt_twin_validation_pass1.txt", PROMPT_NOTE + "\n" + PROMPT_TWIN1)
    wtext("judge/prompt_twin_validation_pass2.txt", PROMPT_NOTE + "# <IDX_LIST>: 15 indices of the stratified 150-twin subsample (every 4th index of 0..542, then 1, 5, 9, ... until 150)\n\n" + PROMPT_TWIN2)
    # validation sets
    val = {"snapshot_time": SNAPSHOT}
    key = jload(f"{R}/dirA5/results/j1_umwp400_key.json")
    U = {}
    for p in sorted(glob.glob(f"{B}/umwp400_j1/audit_parts/*.json")):
        for v in jload(p)["verdicts"]:
            if v.get("verdict") in VERDICTS and v["id"] not in U:
                U[v["id"]] = v
    urows = [{"umwp_id": int(i.split(":")[1]), "umwp_answerable": key[i]["answerable"], "umwp_category": key[i]["category"],
              "umwp_source": key[i]["source"], **verdict_row(U[i]), "audit_k": U[i].get("k")} for i in key if i in U]
    urows.sort(key=lambda r: r["audit_k"] if r["audit_k"] is not None else 10 ** 6)
    wjl("judge/validation/umwp400_opus_vs_human_labels.jsonl", urows)
    val["umwp400"] = jload(f"{R}/dirA5/results/j1_p1_read.json")
    val["umwp400"]["n_rows_released"] = len(urows)
    lab = jload(f"{R}/dirA3/results/audit_validation/labels.json")
    TV = {}
    for p in sorted(glob.glob(f"{R}/dirA3/results/audit_validation/parts/*.json")):
        for v in jload(p)["verdicts"]:
            if v["id"] not in TV:
                TV[v["id"]] = v
    trows = [{"id": i, "kind": lab[i]["kind"], "fam": lab[i]["twin"]["fam"], "task": lab[i]["twin"]["task"], "audit_k": lab[i]["k"],
              **verdict_row(TV[i])} for i in sorted(lab, key=lambda i: lab[i]["k"]) if i in TV]
    wjl("judge/validation/tracer200_audit.jsonl", trows)
    val["tracer200"] = jload(f"{R}/dirA3/results/audit_validation/result.json")
    val["tracer200"]["judges_saw_id_prefix"] = "trd: (premise deleted) / tri: (intact); items fetched from sample.jsonl in file order"
    val["rejudge"] = []
    rk = collections.Counter()
    for d in sorted(glob.glob(f"{R}/dirA3/results/audit_rejudge/*/")):
        run = os.path.basename(d.rstrip("/"))
        first = {v["id"]: v for v in jload(f"{B}/{run}/audit_admitted.json")["verdicts"]}
        second = {}
        for p in sorted(glob.glob(f"{d}parts/*.json")):
            for v in jload(p)["verdicts"]:
                second.setdefault(v["id"], v)
        for i, v in second.items():
            if i in first and v.get("k") is not None and first[i].get("k") is not None:
                rk["checked"] += 1; rk["k_is_199_minus_first_k"] += int(v["k"] == 199 - first[i]["k"])
        rows = [{"run": run, "id": i, "audit_k": first[i].get("k"), "first_verdict": first[i]["verdict"], "first_confidence": first[i].get("confidence"),
                 "second_verdict": second[i]["verdict"], "second_confidence": second[i].get("confidence"), "second_note": second[i].get("note")}
                for i in sorted(first, key=lambda i: first[i].get("k", 0)) if i in second]
        wjl(f"judge/validation/rejudge_{run}.jsonl", rows)
        val["rejudge"].append(jload(f"{d}result.json"))
    STATS["rejudge_k"] = dict(rk)
    hk = jload(f"{R}/dirA4/human_validation/key.json")
    # the item key of the sealed 150-item packet is NOT released: it gives the Opus class of every item that non-author
    # annotators are to label blind; it stays in the project tree until those labels are in
    val["human_packet150"] = {"n": len(hk), "opus_mix": dict(collections.Counter(h["opus"] for h in hk)),
                              "labels_on_server": any(l.strip().split(",")[1:2] != [""] for l in open(f"{R}/dirA4/human_validation/labels_template.csv").readlines()[1:]),
                              "key_in_release": False,
                              "note": "sealed packet for non-author annotation (50 items per Opus class, drifted rounds); its item key "
                                      "is withheld from this release until those labels are in"}
    val["pool_vs_run_audit"] = STATS.get("pool_vs_run")
    d = STATS["dups"]
    val["same_text_in_several_audits"] = {
        "texts_in_several_distinct_pools": d["texts_in_several_pools"], "different_three_way_verdict": d["several_pools_different_verdict"],
        "different_binary_verdict": d["several_pools_different_binary_verdict"],
        "note": "problem texts that occur in more than one distinct admitted pool were judged by independent judge instances"}
    val["expert_annotation_user_reported"] = {
        "kappa_binary": 0.77, "kappa_three_way": 0.69,
        "status": "user-reported; comparison target (expert vs Opus or expert vs expert), item count, annotators and blinding "
                  "not yet documented; no label file exists on the compute host; not part of this release",
        "sources": ["dirA4/docs/A4_verdict_2026-09-24.md", "dirA4/docs/mechanism_note_2026-09-24.md"]}
    wtext("judge/validation/summary.json", json.dumps(val, indent=1, ensure_ascii=False) + "\n")
    STATS["val"] = val
    wtext("judge/PROTOCOL.md", protocol_md())


def protocol_md():
    p = STATS["pools"]
    n_v2 = [x["run"] for x in p if (x["audit_protocol"] or "").startswith("A3b parser") and x["identical_pool_as"] is None]
    n_v2c = [x["run"] for x in p if (x["audit_protocol"] or "").startswith("A3b parser") and x["identical_pool_as"] is not None]
    rk = STATS.get("rejudge_k", {})
    return f"""# Opus audit protocol (frozen 2026-09-22)

Judge: Claude Opus (Anthropic), run as independent subagents of Claude Code workflows with the model alias `opus`
(recorded as `claude-opus-5` in the files of the first audits). Each judge instance starts from an empty context, fetches
20-25 items with a fixed shell command, labels every item, and returns (or pushes) one JSON part file. Judges never see
solver outputs, reference answers or other judges' labels; they are told which model family wrote the items.

What the judges could see. The judges were NOT blind to run, round or condition. Every judge ran a fetch command that
names the item directory, and the file-pushing template also asks the judge to return `run="<RUN>"`:
- admitted-pool audits (200-item and full) and raw-proposal audits: `cd <REPO>/dirB/runs/<RUN>`; the run name encodes
  model, condition, seed and round (for example `qwen3_4b_Base_own_a3_s0_r3`); extension audits ran in the pseudo-run
  directories `dirB/runs/swap_audit_a5_s<seed>` / `phi_audit_a6_s<seed>`; in all of them the printed proposer ids
  (`prop<seed*100+round>:<index>`) encode seed and round as well;
- fixed-pool audits: the source directory (`dirB/runs/qwen3_4b_Base_own_a3_s0_r1`, `..._s1_r1`,
  `dirA3/results/frontier/q25_7b`, `dirA3/results/frontier/evo_r3`) and `dirA3/results/pool_audit/<POOL>/`;
- UMWP-400: `dirB/runs/umwp400_j1` and ids `umwp:<UMWP id>`;
- tracer-200: `dirA3/results/audit_validation` and ids `trd:<fam>:<task>` / `tri:<fam>:<task>`, whose prefix tells
  whether the premise was deleted (trd) or kept (tri);
- re-judges: `dirB/runs/<RUN>` and `dirA3/results/audit_rejudge/<RUN>/`.
Whether this information changed any verdict was not measured; no round-blind or condition-blind audit exists.

Label set (all admitted-pool, raw-pool and fixed-pool audits, and the UMWP / tracer validations):
- SOLVABLE: the text determines a unique numeric answer (the judge states it in the note);
- UNSOLVABLE: a needed quantity or relation is missing, or the text is self-contradictory, so no unique answer exists;
- AMBIGUOUS: two reasonable readings give different answers.
Strictness rule: a default assumption the judge would have to invent makes the item UNSOLVABLE. Confidence in [0, 1],
note of at most 20 words. "Ill-posed" in the paper = UNSOLVABLE + AMBIGUOUS; the `illposed_share` field of the original
audit files is the UNSOLVABLE-only share (INDEX.tsv gives both, named explicitly).

Sampling: per audited run, the admitted proposer questions (kind == proposed; planted tracers excluded) are shuffled
with Python `random.Random(0)`; items k = 0..199 form the 200-item audit (`in_audit_sample200`). `audit_k` is every
row's position in this order (the 200-item and full audits of a run print exactly this k; INDEX.jsonl `audit_k_check`
counts mismatches). Full audits extend the same order to every admitted proposer question (a3_audit_merge.py; for
Qwen3-4B own r3 seeds 0-4 and Phi own r3 seeds 0-1 the remainder was audited later through pseudo-runs swap_audit_a5_s* /
phi_audit_a6_s*, whose own k is not the pool position). Fixed pools are audited exhaustively in file order. The first
verdict of an item is kept (no item was judged twice inside one audit; `duplicate_verdict_conflicts` in INDEX.jsonl
counts disagreements if it happened).

Prompt files (verbatim workflow text; only host, port, user and local paths replaced by placeholders). The four-line
definition block and the "Be strict" sentence are identical in all of them except the two twin validations:
- `prompt_admitted_pool_audit.txt`: every admitted-pool audit (200-item and full), the extension audits and the UMWP-400
  validation (pseudo-run `umwp400_j1` with MODEL = Qwen3-4B-Base).
- `prompt_admitted_pool_audit_round1_v2.txt`: the {len(n_v2)} first round-1 200-item audits ({', '.join(n_v2)}; the
  copies {', '.join(n_v2c)} inherit them): identical definitions and strictness sentence, 10 judges x 20 items, verdicts
  returned in the structured output instead of pushed part files; their full-audit extensions used the main template.
- `prompt_tracer200_validation.txt`: the tracer-200 validation. Same text and framing as the main template (MODEL =
  Qwen3-4B-Base, "ADMITTED by a self-play filter"), but the fetch command reads `sample.jsonl` of
  `<REPO>/dirA3/results/audit_validation` in file order, without the Random(0) shuffle.
- `prompt_rejudge200_reversed.txt`: the two 200-item re-judges. The main template plus the line `P=P[:200][::-1]`
  after the shuffle (the first audit's sample in reverse order; {rk.get('k_is_199_minus_first_k', 0)} of
  {rk.get('checked', 0)} re-judged items carry k = 199 - first k), parts pushed to `<REPO>/dirA3/results/audit_rejudge/<RUN>/parts/`.
- `prompt_raw_pool_audit.txt`: fixed-pool and raw-proposal audits ("as self-play training candidates" instead of
  "ADMITTED by a self-play filter"; items in file order or Random(0) order).
- `prompt_twin_validation_pass1.txt`, `prompt_twin_validation_pass2.txt`: the two twin validations (other label sets).

Provenance of the prompt text: the prompts are the authors' own text. They exist in the Claude Code workflow scripts on
the authors' workstation and were not stored on the compute host; the files here were transcribed verbatim from those
scripts (and from the recorded workflow runs) when this release was built. The four-line definition block and the
strictness sentence are identical in all 51 audit workflow scripts (45 file-pushing, 6 first-round structured-return)
and in every recorded workflow run that contains them.

Human validation packet: a sealed 150-item packet (50 items per Opus class, drawn from drifted rounds; instructions
mirror the judge definitions: S / U / A, "be strict") is planned for non-author annotation. Its item key (which run,
id and Opus class each packet item has) is withheld from this release until those labels are in; summary.json gives
only the class mix.
"""


# ------------------------------------------------------------------ 6. judge-free probe
PROBE_CODE = [("dirA3/code/umwp_probe.py", "probe/code/umwp_probe.py"), ("dirA5/code/a5_j1_judge_free.py", "probe/code/a5_j1_judge_free.py"),
              ("dirA5/code/a5_j1_umwp400.py", "probe/code/a5_j1_umwp400.py"), ("dirA/code/c0_probe.py", "probe/code/c0_probe.py"),
              ("dirA3/code/a3_guard.py", "probe/code/a3_guard.py"), ("dirA3/code/a3_propose.py", "probe/code/a3_propose.py"),
              ("code/loto_common.py", "probe/code/loto_common.py"), ("dirB/code/pdc_util.py", "probe/code/pdc_util.py"),
              ("dirA3/code/a3_audit200_assemble.py", "judge/code/a3_audit200_assemble.py"), ("dirA3/code/a3_audit_merge.py", "judge/code/a3_audit_merge.py"),
              ("dirA3/code/a3_part_check.py", "judge/code/a3_part_check.py")]
ENV_ROOT = '__import__("os").environ.get("RSI_REPO", "<REPO>")'


def roots():
    return sorted({R, os.path.realpath(R)}, key=len, reverse=True)


def copy_code(src, dst):
    """Copy a code file verbatim except for one edit: every absolute project-root path becomes $RSI_REPO (default the
    placeholder "<REPO>"), i.e. "<root>" -> ENV_ROOT and "<root>/x" -> ENV_ROOT + "/x"; a root outside string literals
    becomes <REPO>. Returns the number of replacements."""
    text = open(src, encoding="utf-8").read()
    n = 0
    for root in roots():
        e = re.escape(root)
        text, k1 = re.subn(r"([\"'])" + e + r"\1", lambda m: ENV_ROOT, text)
        text, k2 = re.subn(r"([\"'])" + e + r"/", lambda m: ENV_ROOT + " + " + m.group(1) + "/", text)
        text, k3 = re.subn(e, "<REPO>", text)
        n += k1 + k2 + k3
    if dst.endswith(".py"):
        compile(text, dst, "exec")  # the edit must leave valid Python
    wtext(dst, text)
    return n


def copy_text(src, dst):
    text = open(src, encoding="utf-8").read()
    n = 0
    for root in roots():
        text, k = re.subn(re.escape(root), "<REPO>", text)
        n += k
    wtext(dst, text)
    return n


def inband(ans):
    c = collections.Counter(a for a in ans if a is not None and a != "unknown")
    k = c.most_common(1)[0][1] if c else 0
    return 0.25 <= k / len(ans) <= 0.75


def probe():
    src_rows = []
    for src, dst in PROBE_CODE:
        n = copy_code(f"{R}/{src}", dst)
        src_rows.append([dst, src, sha(f"{R}/{src}"), n])
    wtsv("probe/code/SOURCES.tsv", ["released_file", "source_file", "source_sha256", "root_replacements"], src_rows)
    STATS["code_sources"] = src_rows
    uf = f"{R}/data_ext/UMWP/{UMWP['file']}"
    raw = open(uf, "rb").read()
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert blob == UMWP["git_blob"], f"UMWP file differs from the pinned version: git blob {blob}"
    D = jl(uf)
    STATS["umwp_file"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "rows": len(D)}
    byid = {x["id"]: x for x in D}
    uns = [x for x in D if not x["answerable"] and x.get("relevant_ids") and all(i in byid and byid[i]["answerable"] for i in x["relevant_ids"])]
    uns = random.Random(0).sample(sorted(uns, key=lambda x: x["id"]), 600)
    ans_ids = sorted({x["relevant_ids"][0] for x in uns})
    pair = collections.defaultdict(list)
    for x in uns:
        pair[x["relevant_ids"][0]].append(x["id"])
    sub = [{"umwp_id": x["id"], "answerable": False, "category": x["category"], "source": x["source"], "original_id": x["relevant_ids"][0]} for x in uns]
    sub += [{"umwp_id": i, "answerable": True, "category": 0, "source": byid[i]["source"], "unanswerable_variants": pair[i]} for i in ans_ids]
    wjl("probe/umwp_subset1200.jsonl", sub)
    subset_ids = {r["umwp_id"] for r in sub}
    summ = []
    for f in sorted(glob.glob(f"{R}/dirA4/results/umwp/*.jsonl")):
        tag = os.path.basename(f)[:-6]
        rows = {}
        for r in jl(f):
            rows.setdefault(r["id"], r)
        keep = [{"umwp_id": r["id"], "answerable": r["answerable"], "mode": r["mode"], "p_mode": r["p_mode"], "nonanswer": r["nonanswer"],
                 "abstain": r.get("abstain"), "correct": r["correct"], "answers": r["answers"]} for r in rows.values()]
        wjl(f"probe/umwp_dumps/{tag}.jsonl", keep)
        u = [r for r in keep if not r["answerable"]]; a_ = [r for r in keep if r["answerable"]]
        mean = lambda xs: fr(sum(xs) / len(xs)) if xs else None
        prompt, ck = (tag.split("_", 1)[0], tag.split("_", 1)[1])
        ck = {"qwen3_4b_base": "Qwen3-4B-Base", "base_Qwen3-8B-Base": "Qwen3-8B-Base", "base_Phi-4-mini-instruct": "Phi-4-mini-instruct"}.get(ck, ck)
        ru, ra = mean([inband(r["answers"]) for r in u]), mean([inband(r["answers"]) for r in a_])
        summ.append([tag, prompt, ck, len(keep), int(len(keep) == 1200 and set(rows) == subset_ids), len(u), len(a_), mean([r["correct"] for r in a_]),
                     mean([r["abstain"] or 0 for r in u]), mean([r["abstain"] or 0 for r in a_]), mean([r["nonanswer"] for r in u]),
                     mean([r["nonanswer"] for r in a_]), ru, ra, fr(ru / ra, 3) if ru is not None and ra else None])
    wtsv("probe/umwp_dump_summary.tsv", ["tag", "prompt", "checkpoint", "n_items", "complete", "n_unanswerable", "n_answerable",
                                          "accuracy_answerable", "abstain_rate_unanswerable", "abstain_rate_answerable",
                                          "nonanswer_unanswerable", "nonanswer_answerable", "inband_rate_unanswerable",
                                          "inband_rate_answerable", "inband_ratio_U_over_A"], summ)
    STATS["umwp"] = {"n_files": len(summ), "complete": sum(s[4] for s in summ), "rows": sum(s[3] for s in summ)}
    # twin dumps used by J-P3 (ids + sampled answers only, no text)
    TW = {"Qwen3-4B-Base": f"{R}/dirA/results/qwen3_4b_base_shard*of8.jsonl", "Qwen3-8B-Base": f"{R}/dirA/results/qwen3_8b_base_shard*.jsonl",
          "Phi-4-mini-instruct": f"{R}/dirA/results/phi4mini_shard*of8.jsonl"}
    TW.update({run: f"{R}/dirB/results/hp/hp_{run}_shard*.jsonl" for run in
               [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)] + [f"qwen3_8b_Base_own_a3_s{s}_r3" for s in range(3)] + [f"phi4mini_own_a4_s{s}_r3" for s in range(2)]})
    tw = []
    for tag, pat in TW.items():
        rows = [r for f in sorted(glob.glob(pat)) for r in jl(f)]
        keep = [{k: r.get(k) for k in ("fam", "task", "form", "split_kind", "p_mode", "mode", "nonanswer", "correct", "trunc_rate", "abstain", "answers")} for r in rows]
        wjl(f"probe/twin_dumps/{tag}.jsonl", keep)
        tw.append((tag, len(keep)))
    STATS["twin_dumps"] = tw
    n = 0
    for f in ("j1_judge_free.json", "j1_judge_free.txt", "j1_p1_read.json"):
        n += copy_text(f"{R}/dirA5/results/{f}", f"probe/results/{f}")
    STATS["results_root_replacements"] = n
    wtext("probe/reproduce_j1.py", REPRODUCE_J1)
    wtext("probe/README.md", PROBE_README.format(**UMWP, **STATS["umwp_file"]))


REPRODUCE_J1 = r'''#!/usr/bin/env python3
"""Recompute the judge-free reads J-P2 (UMWP) and J-P3 (premise-deletion twins) of results/j1_judge_free.txt from the
released dumps alone: umwp_dumps/, twin_dumps/ and ../twins/probe543_validation.jsonl (no model, GPU, UMWP or GSM-Symbolic
text needed). Same definitions as code/a5_j1_judge_free.py, which reads the project tree instead.

  cd probe && python3 reproduce_j1.py [--mc] [--out FILE]

In-band = the modal numeric answer's share of G = 8 samples is in [0.25, 0.75] (unparsed / truncated / 'unknown' samples
count in the denominator and never vote; the admission statistic of a3_guard.py). Ratio = in-band rate on unanswerable
items / in-band rate on answerable items. J-P2: UMWP unanswerable questions vs their answerable originals, plain-prompt
G = 8 dumps; holds if ratio(r3) >= 1.25 x ratio(base) in >= 80% of the powered own lineages (>= 30 answerable items in
band at base and at r3). This part is exact. J-P3: the deleted forms of the twins that pass-1 validation judged
UNSOLVABLE vs the intact forms of the same tasks, 32-sample dumps subsampled to G = 8; holds in >= 8 of 10 own lineages.
By default the subsampling is the exact hypergeometric probability (the registered method); --mc repeats the original
Monte-Carlo approximation (2,000 draws per item, seed 20260924), whose values depend on the row order of the dumps and
therefore match the original read only approximately. The script ends by comparing its ratios with
results/j1_judge_free.json.
"""
import argparse, json, os, random
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--umwp_dir", default=os.path.join(HERE, "umwp_dumps"))
ap.add_argument("--twin_dir", default=os.path.join(HERE, "twin_dumps"))
ap.add_argument("--validation", default=os.path.join(HERE, "..", "twins", "probe543_validation.jsonl"))
ap.add_argument("--original", default=os.path.join(HERE, "results", "j1_judge_free.json"))
ap.add_argument("--mc", action="store_true", help="Monte-Carlo subsampling as in the original a5_j1_judge_free.py")
ap.add_argument("--out", default="")
a = ap.parse_args()
G, LO, HI = 8, 0.25, 0.75


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


UNS = {(str(v["fam"]), int(v["task"])) for v in jl(a.validation) if v["pass1_verdict"] == "UNSOLVABLE"}


def inband(ans):
    c = Counter(x for x in ans if x is not None and x != "unknown")
    k = c.most_common(1)[0][1] if c else 0
    return LO <= k / len(ans) <= HI


KS = [k for k in range(G + 1) if LO <= k / G <= HI]


def p_inband_exact(ans):
    """Probability that a uniformly random G-subset of ans is in band (multivariate hypergeometric, exact)."""
    n = len(ans)
    if n == G:
        return float(inband(ans))
    c = Counter(x for x in ans if x is not None and x != "unknown")
    n_null = n - sum(c.values())

    def ways_max_le(m):  # number of G-subsets in which no answer has more than m votes
        poly = [comb(n_null, j) for j in range(G + 1)]
        for nc in c.values():
            new = [0] * (G + 1)
            for i, pi in enumerate(poly):
                if pi:
                    for j in range(0, min(m, nc, G - i) + 1):
                        new[i + j] += pi * comb(nc, j)
            poly = new
        return poly[G]
    return (ways_max_le(max(KS)) - ways_max_le(min(KS) - 1)) / comb(n, G)


rng = random.Random(20260924)


def p_inband_mc(ans):
    return sum(inband(rng.sample(ans, G)) for _ in range(2000)) / 2000


def umwp(tag):
    f = os.path.join(a.umwp_dir, f"plain_{tag}.jsonl")
    if not os.path.exists(f):
        return None
    rows = {r["umwp_id"]: r for r in jl(f)}
    if len(rows) < 1200:
        return None
    iu = [inband(r["answers"]) for r in rows.values() if not r["answerable"]]
    ia = [inband(r["answers"]) for r in rows.values() if r["answerable"]]
    return {"rate_U": sum(iu) / len(iu), "rate_A": sum(ia) / len(ia), "n_inband_A": sum(ia)}


def twins(tag):
    f = os.path.join(a.twin_dir, f"{tag}.jsonl")
    d = jl(f) if os.path.exists(f) else []
    if len(d) < 1086:
        return None
    u = [r for r in d if r["form"] == "deleted" and (str(r["fam"]), int(r["task"])) in UNS]
    uk = {(str(r["fam"]), int(r["task"])) for r in u}
    ia = [r for r in d if r["form"] == "intact" and (str(r["fam"]), int(r["task"])) in uk]
    p = p_inband_mc if a.mc else p_inband_exact
    pu = [p(r["answers"]) for r in u]; pa = [p(r["answers"]) for r in ia]
    return {"rate_U": sum(pu) / len(pu), "rate_A": sum(pa) / len(pa), "n_inband_A": sum(pa), "n_U": len(u), "n_A": len(ia)}


LIN = [("qwen3_4b", "qwen3_4b_base", "Qwen3-4B-Base", [f"qwen3_4b_Base_own_a3_s{s}_r3" for s in range(5)]),
       ("qwen3_8b", "base_Qwen3-8B-Base", "Qwen3-8B-Base", [f"qwen3_8b_Base_own_a3_s{s}_r3" for s in range(3)]),
       ("phi4mini", "base_Phi-4-mini-instruct", "Phi-4-mini-instruct", [f"phi4mini_own_a4_s{s}_r3" for s in range(2)])]
out = {"JP2": [], "JP3": [], "subsampling": "monte_carlo" if a.mc else "exact_hypergeometric"}
for fam, base, _, runs in LIN:
    b = umwp(base)
    print(f"UMWP {fam} base: {b}")
    for run in runs:
        m = umwp(run)
        if b is None or m is None:
            out["JP2"].append({"run": run, "read": None}); print(f"  {run}: not read"); continue
        powered = b["n_inband_A"] >= 30 and m["n_inband_A"] >= 30
        rb, rm = b["rate_U"] / max(1e-9, b["rate_A"]), m["rate_U"] / max(1e-9, m["rate_A"])
        out["JP2"].append({"run": run, "ratio_base": rb, "ratio_r3": rm, "powered": powered, "holds": rm >= 1.25 * rb})
        print(f"  {run}: ratio base {rb:.3f} -> r3 {rm:.3f} (x{rm / max(1e-9, rb):.2f}) powered {powered} -> {rm >= 1.25 * rb}")
pw = [x for x in out["JP2"] if x.get("powered")]
out["JP2_verdict"] = None if not pw else sum(x["holds"] for x in pw) / len(pw) >= 0.8
print("J-P2:", out["JP2_verdict"], f"({sum(x['holds'] for x in pw)}/{len(pw)} powered lineages)")
bases = {}
for fam, _, tb, _ in LIN:
    bases[fam] = twins(tb)
    print(f"twins {fam} base ({tb}): {bases[fam]}")
for fam, _, _, runs in LIN:
    for run in runs:
        m, b = twins(run), bases.get(fam)
        if b is None or m is None:
            out["JP3"].append({"run": run, "read": None}); print(f"  twins {run}: not read"); continue
        rb, rm = b["rate_U"] / max(1e-9, b["rate_A"]), m["rate_U"] / max(1e-9, m["rate_A"])
        out["JP3"].append({"run": run, "ratio_base": rb, "ratio_r3": rm, "holds": rm >= 1.25 * rb})
        print(f"  twins {run}: ratio base {rb:.3f} -> r3 {rm:.3f} (x{rm / max(1e-9, rb):.2f})")
rd = [x for x in out["JP3"] if x.get("holds") is not None]
out["JP3_verdict"] = None if len(rd) < 10 else sum(x["holds"] for x in rd) >= 8
print("J-P3:", out["JP3_verdict"], f"({sum(x['holds'] for x in rd)}/{len(rd)} lineages read)")
if os.path.exists(a.original):
    o = json.load(open(a.original, encoding="utf-8"))
    for part in ("JP2", "JP3"):
        prev = {x["run"]: x for x in o.get(part, []) if x.get("ratio_r3") is not None}
        diffs = [max(abs(x["ratio_base"] - prev[x["run"]]["ratio_base"]), abs(x["ratio_r3"] - prev[x["run"]]["ratio_r3"]))
                 for x in out[part] if x.get("ratio_r3") is not None and x["run"] in prev]
        same = sum(x["holds"] == prev[x["run"]]["holds"] for x in out[part] if x.get("ratio_r3") is not None and x["run"] in prev)
        print(f"vs results/j1_judge_free.json {part}: {len(diffs)} lineages, max |ratio difference| {max(diffs) if diffs else float('nan'):.4f}, "
              f"same per-lineage outcome {same}/{len(diffs)}, verdict {out[part + '_verdict']} vs {o.get(part + '_verdict')}")
if a.out:
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
'''

PROBE_README = """# probe/

Judge-free probe (A5 J1) and the UMWP / twin dumps it reads.

- `reproduce_j1.py`: recomputes J-P2 (UMWP) and J-P3 (premise-deletion twins) from `umwp_dumps/`, `twin_dumps/` and
  `../twins/probe543_validation.jsonl` alone and compares the result with `results/j1_judge_free.json`
  (`cd probe && python3 reproduce_j1.py`; J-P3 by exact hypergeometric subsampling, `--mc` for the original Monte Carlo).
- `code/`: copies of the project scripts: `umwp_probe.py` (wrote the UMWP dumps), `c0_probe.py` (the twin dumps),
  `a5_j1_judge_free.py` (the original J-P2 / J-P3 read), `a5_j1_umwp400.py` (the UMWP-400 judge validation),
  `a3_propose.py` / `a3_guard.py` (proposer and admission guard), `loto_common.py` / `pdc_util.py` (helpers). The only
  edit to these otherwise verbatim copies replaces the absolute project root by
  `__import__("os").environ.get("RSI_REPO", "<REPO>")` (in a string literal) or `<REPO>` (elsewhere); `code/SOURCES.tsv`
  gives each source file's SHA-256 and the number of replacements (the same holds for `../judge/code/`). The scripts
  expect the project tree (`RSI_REPO`), vLLM and GPUs; `reproduce_j1.py` needs neither. `results/` holds the original
  read-outs, with the absolute project root replaced by `<REPO>`.
- `umwp_subset1200.jsonl`: the 1,200-item UMWP subset (600 unanswerable questions, a Random(0) sample of those whose
  answerable originals exist, and their 600 answerable originals): UMWP ids, labels and pairing, no question text.
- `umwp_dumps/<prompt>_<checkpoint>.jsonl`: our G = 8 samples per UMWP item (`umwp_id`, `answerable`, `mode`, `p_mode`,
  `nonanswer`, `abstain`, `correct`, `answers`); `umwp_dump_summary.tsv` summarises each dump.
- `twin_dumps/<checkpoint>.jsonl`: our G = 32 samples per twin and form (`fam`, `task`, `form`, ..., `answers`; no text).

UMWP (Sun et al., LREC-COLING 2024) is third-party data: {repo}, file `{file}`, retrieved {retrieved} from branch
{branch} (head commit {head_commit}, {head_commit_date}; the file was last changed in commit {file_last_changed_in};
git blob {git_blob}; SHA-256 {sha256}; {rows} rows). Licence: {licence}. No UMWP question text is included. The UMWP
ids and labels (answerable, category, source, pairing) are UMWP data under CC BY-SA 4.0; the files that carry them
(`umwp_subset1200.jsonl`, `umwp_dumps/`, `../judge/validation/umwp400_opus_vs_human_labels.jsonl`) should therefore be
shared under CC BY-SA 4.0 with attribution (to be confirmed by the authors).
"""


# ------------------------------------------------------------------ 7. datasheet, readme, manifest
def cstr(d):
    return ", ".join(f"{v} {k}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def lineage_table():
    lin = collections.defaultdict(dict)
    for x in STATS["pools"]:
        lin[(x["model"], x["condition"], x["seed"])][x["round"]] = x
    lines = ["| model | condition | seed | rounds audited | ill-posed share (U+A) of the 200-item sample by round |", "|---|---|---|---|---|"]
    for (model, cond, seed), rs in sorted(lin.items()):
        seq = ", ".join(f"r{r}: {rs[r]['sample200_illposed_share']:.2f}" for r in sorted(rs))
        lines.append(f"| {model} | {cond} | {seed} | {len(rs)} | {seq} |")
    return "\n".join(lines)


def file_table():
    lines = ["| file | rows | S / U / A (all audited rows) | coverage | identical pool as |", "|---|---|---|---|---|"]
    for x in STATS["pools"]:
        lines.append(f"| audited_pools/{x['run']}.jsonl | {x['n_audited']} | {x['all_SOLVABLE']} / {x['all_UNSOLVABLE']} / {x['all_AMBIGUOUS']} | "
                     f"{x['coverage']} | {x['identical_pool_as'] or ''} |")
    return "\n".join(lines)


DERIVED = {"own_clean_a3": 1, "own_sub_a3": 1, "own_dirty_a3": 1, "swap_c1_a5": 3, "swap_g1_a5": 3, "cln1_a6": 3, "rnd1_a6": 3,
           "tg_abs4_a5": 4, "tg_t4_a5": 4}


def derived_sets():
    """Training sets built from an audited pool and not audited themselves: the parent pool and how many of their
    proposer items carry a verdict in this release (matched on the problem text)."""
    audited = {x["run"]: x for x in STATS["pools"]}
    have = collections.defaultdict(set)
    for run, model, i, t, verdict in STATS["pool_text_rows"]:
        have[run].add(t)
    out = []
    for d in sorted(os.listdir(B)):
        mm = RUN_RE.match(d)
        if not mm or DERIVED.get(mm.group("c")) != int(mm.group("r")) or d in audited or not os.path.exists(f"{B}/{d}/config.json"):
            continue
        data = jload(f"{B}/{d}/config.json").get("data") or ""
        if not os.path.exists(data):
            data = f"{B}/{d}/{os.path.basename(data)}"
        if not os.path.exists(data):
            continue
        A = jl(data)
        own = "own_a4" if mm.group("m") in ("phi4mini", "qwen3_14b_Base") else "own_a3"
        parent = f"{mm.group('m')}_{'loop_uv_a4' if mm.group('c').startswith('tg_') else own}_s{mm.group('s')}_r{mm.group('r')}"
        texts = [(x.get("problem") or x.get("prompt")) for x in A if x.get("kind", "proposed") == "proposed"]
        pa = f"{B}/{parent}/admitted.jsonl"
        ptexts = {(x.get("problem") or x.get("prompt")) for x in jl(pa) if x["kind"] == "proposed"} if os.path.exists(pa) else set()
        out.append({"run": d, "condition": mm.group("c"), "round": int(mm.group("r")), "parent": parent,
                    "parent_coverage": audited.get(parent, {}).get("coverage"), "n_proposed": len(texts),
                    "n_in_parent": sum(t in ptexts for t in texts), "n_with_verdict": sum(t in have[parent] for t in texts),
                    "n_tracers": sum(x.get("kind") in ("tracer_deleted", "tracer_intact") for x in A)})
    return out


def derived_table(der):
    g = collections.OrderedDict()
    for x in der:
        g.setdefault(x["condition"], []).append(x)
    lines = ["| derived training sets | parent pools | sets | proposer items | with a verdict | not fully covered |", "|---|---|---|---|---|---|"]
    for c, xs in g.items():
        parents = sorted({RUN_RE.match(x["parent"]).group("c") for x in xs})
        part = [f"{x['run']}: {x['n_with_verdict']} of {x['n_proposed']} (parent {x['parent']}, {x['parent_coverage']})"
                for x in xs if x["n_with_verdict"] < x["n_proposed"]]
        miss = [f"{x['run']}: {x['n_proposed'] - x['n_in_parent']} items not in the parent pool" for x in xs if x["n_in_parent"] < x["n_proposed"]]
        lines.append(f"| {c} round {xs[0]['round']} | {' / '.join(parents)} round {xs[0]['round']} | {len(xs)} | "
                     f"{sum(x['n_proposed'] for x in xs):,} | {sum(x['n_with_verdict'] for x in xs):,} | {'; '.join(part + miss) or 'none'} |")
    return "\n".join(lines)


def datasheet(manifest):
    P = STATS["pools"]
    n_runs = len(P)
    uniq = [x for x in P if x["identical_pool_as"] is None]
    full = [x for x in P if x["coverage"] == "full"]
    models = collections.Counter(x["model"] for x in P)
    conds = collections.Counter(x["condition"] for x in P)
    total_rows = sum(x["n_audited"] for x in P)
    uniq_rows = sum(x["n_audited"] for x in uniq)
    alls = collections.Counter()
    for x in uniq:
        for k in VERDICTS:
            alls[k] += x[f"all_{k}"]
    no_solver = sum(x["rows_without_solver_record"] for x in P)
    base_rows = sum(x["n_audited"] for x in P if x["solver_checkpoint"] in MODEL_NAMES.values())
    maxrule = [x["run"] for x in P if x["admission_stat"] != "p_mode"]
    kc = collections.Counter()
    for x in P:
        kc.update(x["audit_k_check"])
    fx = STATS["fixed"]
    val = STATS["val"]
    u4 = val["umwp400"]
    rj = val["rejudge"]
    tv = val["tracer200"]
    pv = STATS["pool_vs_run"]
    tw = STATS["twins"]
    twv = STATS["twin_val"]
    dup = STATS["dups"]
    red = redaction_summary()
    STATS["redaction"] = red
    ra, rp = red["all"], red.get("audited_pools", {})
    der = derived_sets()
    STATS["derived"] = der
    uf = STATS["umwp_file"]
    scale = [(c[1], c[2]) for c in STATS["checkpoints"] if c[1].startswith("scale_") and not c[1].endswith("/base")]
    size = sum(int(m[1]) for m in manifest)
    by_dir = collections.OrderedDict()
    for m in manifest:
        d = m[0].split("/")[0] if "/" in m[0] else "(top level)"
        a = by_dir.setdefault(d, [0, 0, 0]); a[0] += 1; a[1] += int(m[1]); a[2] += int(m[2])
    dir_lines = "\n".join(f"| {d} | {a[0]} | {a[2]} | {a[1] / 1e6:.2f} |" for d, a in by_dir.items())
    cond_lines = "\n".join(f"| {c} | {CONDITIONS.get(c, ('other', ''))[0]} | {conds[c]} | {CONDITIONS.get(c, ('', ''))[1]} |" for c in sorted(conds))
    fixed_lines = "\n".join(f"| fixed_pool_audits/{f['pool']}.jsonl | {f['n']} of {f['n_candidates']} | {f['SOLVABLE']} / {f['UNSOLVABLE']} / {f['AMBIGUOUS']} | "
                            f"{f['proposer']} | {len(f['checkpoints'])} | {f['what']} |" for f in fx)
    raw_lines = "\n".join(f"| raw_proposal_audits/{r['run']}.jsonl | {r['n']} of {r['n_proposed']} | {r['SOLVABLE']} / {r['UNSOLVABLE']} / {r['AMBIGUOUS']} |" for r in STATS["raw"])
    rj_lines = "; ".join(f"{r['run']}: agreement {r['agreement']:.3f}, kappa {r['kappa3']:.2f} (three-way), {r['agreement_binary']:.3f} / {r['kappa_binary']:.2f} (binary)" for r in rj)
    pv_lines = "; ".join(f"{x['pool']} vs {x['run']}: {x['identical_verdicts']}/{x['n_overlap']} identical, kappa {x['kappa3']} (three-way), {x['kappa_binary']} (binary)" for x in pv)
    umwp_det = ", ".join(f"cat {k}: {v:.3f}" for k, v in sorted(u4["detection_by_category"].items()))
    dsame1 = f"{dup['several_pools_different_verdict']} ({dup['several_pools_different_verdict'] / max(1, dup['texts_in_several_pools']) * 100:.1f}%)"
    dsame = f"{dup['several_pools_different_verdict']} of {dup['texts_in_several_pools']} ({dup['several_pools_different_verdict'] / max(1, dup['texts_in_several_pools']) * 100:.1f}%)"
    dsame2 = f"{dup['several_pools_different_binary_verdict']} ({dup['several_pools_different_binary_verdict'] / max(1, dup['texts_in_several_pools']) * 100:.1f}%)"
    pending = []
    for d in sorted(os.listdir(B)):
        mm = RUN_RE.match(d)
        if not mm or os.path.exists(f"{B}/{d}/audit_admitted.json") or not os.path.exists(f"{B}/{d}/admitted.jsonl"):
            continue
        c, r = mm.group("c"), int(mm.group("r"))
        if c in CONDITIONS and not (c in ("own_clean_a3", "own_sub_a3") and r == 1):
            pending.append(d + ("" if os.path.exists(f"{B}/{d}/.trained") else " (training not finished)"))
    STATS["pending"] = pending
    return f"""# Datasheet: audited self-play curricula (release v1)

Release date {RELEASE_DATE}; the project tree was read at {SNAPSHOT} (snapshot time, also in INDEX.jsonl and
judge/validation/summary.json). Structure follows Gebru et al., "Datasheets for Datasets" (CACM 2021). Every number in
this file is computed by `build_release.py` from the files of this release; MANIFEST.tsv gives the SHA-256 of each file.

## 1. Motivation

**Purpose.** Label-free self-play for math reasoning (R-Zero-style loops) lets a proposer model write problems and admits
into the curriculum those on which the current solver's sampled answers disagree moderately (modal share in a band).
The loop itself never checks whether the admitted problems are well-posed. This release records, round by round and lineage by lineage,
which problems such loops admitted, how the admission-time solver answered them, and whether a strict LLM judge (Claude
Opus) finds them SOLVABLE, UNSOLVABLE or AMBIGUOUS. It also contains fixed candidate pools audited exhaustively (to
separate the guard's selection from the proposer's output), planted premise-deletion twins with their validation, the
exact judge prompts and a judge-free probe on an external human-labelled benchmark (UMWP).

**Creators and funding.** The authors of the accompanying submission (anonymous during review). Funding: to be stated
in the camera-ready version.

## 2. Composition

**Instances.** Grade-school math word problems in English, written by language models (the proposers), with one Opus
verdict per problem, the judge's confidence and a short note, and the admission-time solver statistics over G = 8 samples:
`solver_p_mode` (share of the modal parsed answer; unparsed or truncated samples count in the denominator, "unknown"
never votes), `solver_mode` (the modal answer) and `solver_abstain` (share of "unknown" answers; non-zero essentially
only under the abstention prompt). `solver_checkpoint` names the solver that scored the problem at admission: the base model in round
1, the previous round's trained checkpoint afterwards. `base_p_mode / base_mode / base_abstain` repeat these values when
that solver is the base model and are null otherwise (the base model did not score later-round proposals; base-model
agreement on fixed pools is in `fixed_pool_audits/`, key `<diag>/base`).

**Fields.** audited_pools rows: `lineage, run, model, condition, seed, round, id, problem, problem_sha256` (SHA-256 of the
released text), `redacted` (true if spans of the text were masked, see Confidentiality), `opus_verdict, opus_confidence,
opus_note, in_audit_sample200, audit_k` (the row's position in the run's Random(0) order of admitted proposer questions,
the order every audit of the run uses; set for every row, < 200 exactly for the 200-item sample; for the
{kc['k_checked']} items whose k the run's own 200-item or full audit recorded, {kc['k_mismatch']} differ), `audit_source`
(sample200, full_audit, or the extension pseudo-run), `solver_checkpoint, solver_p_mode, solver_mode, solver_abstain,
base_p_mode, base_mode, base_abstain`. INDEX.tsv: one row per audited run (admitted counts by kind, audited count,
coverage, verdict counts on the 200-item sample and on all audited rows, UNSOLVABLE-only and UNSOLVABLE + AMBIGUOUS
shares, guard rule, `admission_stat` and band, `identical_pool_as`). `admission_stat` is the statistic the guard compared
with the band: `p_mode` for all runs except the {len(maxrule)} loop_uv_a4 runs, which admitted on `max(p_mode,abstain)`
(`--count_unknown`); it is read off the guard records (the rule that reproduces every record's admit flag,
`admission_stat_basis` in INDEX.jsonl). INDEX.jsonl adds the verdict sources, the audit protocol string, the `audit_k`
check, `snapshot_time` and the raw guard_stats. fixed_pool_audits rows: `pool, proposer, id, k, problem,
problem_sha256, redacted, stem, opus_*`, `agreement` = map from `<diag>/<checkpoint>` to `p_mode, mode, abstain, admit`
(`evolved_from` for evo_r3). raw_proposal_audits rows mirror audited_pools rows without solver fields. Tracer, twin,
validation and probe files are described in their own sections; every file's row count is in MANIFEST.tsv (rows =
records for .jsonl, data rows for .tsv, lines otherwise).

**Counts.**

| directory | files | rows | MB |
|---|---|---|---|
{dir_lines}

Total without DATASHEET.md and MANIFEST.tsv: {len(manifest)} files, {size:,} bytes ({size / 1e6:.1f} MB; sum of file sizes).

*audited_pools/*: {n_runs} audited (lineage, round) pools, {total_rows} rows; {len(uniq)} distinct pools ({uniq_rows} rows)
after removing pools that are copies by construction (`identical_pool_as` in INDEX.tsv: the rescue arms reuse the own
round-3 admitted set, co-evolving round 1 reuses own round 1). Over the distinct pools: {alls['SOLVABLE']} SOLVABLE,
{alls['UNSOLVABLE']} UNSOLVABLE, {alls['AMBIGUOUS']} AMBIGUOUS. The {dup['rows']} rows of the distinct pools hold
{dup['unique_texts']} unique problem texts: {dup['texts_in_several_pools']} texts occur in more than one distinct pool
(see Relationships). {len(full)} pools are audited in full (every admitted proposer question), the others on the
200-item random sample. Models: {', '.join(f'{k} ({v} pools)' for k, v in sorted(models.items()))}.
{base_rows} rows were scored at admission by the base model itself; {no_solver} rows have no solver record.

| condition | group | pools | definition |
|---|---|---|---|
{cond_lines}

Ill-posed share (UNSOLVABLE + AMBIGUOUS) of the 200-item sample, per lineage and round:

{lineage_table()}

*fixed_pool_audits/* (every candidate audited; `agreement` holds, per scored checkpoint, p_mode / mode / abstain /
admit; CHECKPOINTS.tsv maps each key to its source directory, the solver checkpoint, prompt, guard, admission statistic
and band). Two groups of keys need care: {' and '.join(f'`{k}`' for k, _ in scale)} are the vendors' post-trained models
(Qwen3-14B and Qwen2.5-32B-Instruct, source directories named `r1`), not round-1 checkpoints of our training; the
`uv/pool_s0/*` and `uv/pool_s1/*` passes used the abstention prompt with admission on p_mode (R-Zero band), not the
loop_uv rule; admission on max(p_mode, abstain) can be recomputed from the stored `p_mode` and `abstain`.

| file | audited | S / U / A | proposer | checkpoints | pool |
|---|---|---|---|---|---|
{fixed_lines}

*raw_proposal_audits/* (200-item Random(0) sample of the raw, pre-guard proposals of the co-evolving proposer arms):

| file | audited | S / U / A |
|---|---|---|
{raw_lines}

*twins/*: {tw['n']} GSM-HP premise-deletion twins (an intact problem and the same problem with one premise sentence
removed; split kinds: {cstr(tw['splits'])}), of which {tw['probe543']} form the registered probe set. Texts are NOT
included (see Distribution); `twin_patches.jsonl` rebuilds them exactly from GSM-Symbolic ({tw['literal_share'] * 100:.1f}%
of the characters are literals: numbers, number words, names), and `rebuild_twins.py` writes a drop-in `--twins_file` for
`probe/code/c0_probe.py`. Validation: pass 1 (28 Opus judges, intact shown) on the probe set: {cstr(twv['pass1'])};
pass 2 (10 judges, solve-first, deleted text only) on 150 stratified probe items: {twv['pass2_solvable']} solvable
(pass 1 vs pass 2: {twv['kappa_txt']}); a random 200 of all 3,336 twins (the first 200 of the hp_deleted500 random
sample): {cstr(twv['random200'])}. `tracer_guard_records.jsonl` ({STATS['tracer_rows']} rows) gives the solver
statistics of the 200 deleted + 200 intact tracers planted in every audited round's pool and in every fixed-pool guard
pass (ids only).

*judge/*: exact prompts, PROTOCOL.md, and validation sets (Section 4). *probe/*: judge-free probe code, the 1,200-item
UMWP subset (ids, labels, pairing), {STATS['umwp']['n_files']} UMWP dumps ({STATS['umwp']['complete']} complete) with a
per-dump summary, the twin dumps used by J-P3, the J-P1 / J-P2 / J-P3 read-outs and `reproduce_j1.py`, which recomputes
J-P2 and J-P3 from the released dumps alone (probe/README.md).

**Per-file counts of audited_pools/** (INDEX.tsv has the full metadata; MANIFEST.tsv has rows and bytes of every file):

{file_table()}

**Missing information / gaps.** No gold answers exist (the loops are label-free); SOLVABLE notes often state the
judge's answer, which was not verified. Solver statistics are G = 8 samples (resolution 1/8). Later-round proposals were
not scored by the base model. Rounds were audited where compute allowed (see the lineage table); Qwen3-4B-Base own
lineages reach round 6 for seeds 0-2, Qwen3-8B-Base own seeds 1-2 reach round 6. Admitted pools of the conditions
above that had no Opus audit at the snapshot time: {', '.join(pending) or 'none'}; a later build will include them once
audited. Training sets built from audited pools and not audited themselves are not listed as pools; their items are rows
of the parent pool files where the parent is fully audited:

{derived_table(der)}

Not included at all: model checkpoints, full solver rollouts (only the per-item statistics), the twin and UMWP question
texts (licences, Section 6), the expert labels (Section 4) and the key of the sealed human-validation packet.

**Relationships.** Round t of a lineage is proposed by the lineage's proposer (frozen base model, except the coev arms)
and admitted by the solver trained in round t-1 (`solver_checkpoint`). Pools of different conditions with the same
model, seed and round draw on largely the same proposals: `a3_propose.py` samples with seed seed*100+round in every
condition, and its index counts only the outputs it keeps, so the same text often carries a different id. Ids
(`prop<seed*100+round>:<index>`) are therefore unique only within a run: the same (model, id) pair names different texts
in different runs in {dup['model_id_pairs_with_several_texts']} of {dup['model_id_pairs']} cases (for example, the admitted
pools of {' and '.join(dup['example'].get('runs', []))} share {dup['example'].get('shared_texts')} texts, only
{dup['example'].get('shared_texts_same_id')} of them under the same id, while {dup['example'].get('common_ids_other_text')}
ids name a different text in each). Join rows across runs on `problem_sha256`, not on `id`.
{dup['texts_in_several_pools']} texts occur in more than one distinct pool; each audit judged them separately
(Section 4). Tracer ids `trd:<fam>:<task>` / `tri:<fam>:<task>` point into `twins/twin_patches.jsonl`.

**Confidentiality, personal or sensitive content.** Names in the problems are mostly generic first names chosen by the
proposers, but base models sometimes continue a problem with copied web text, so not every name or string is
model-invented. A regex screen of the {ra['unique_texts']} unique problem texts of this release (audited, fixed and
raw pools) found {ra['texts_with_url']} texts with URLs, {ra['texts_with_email']} with an e-mail address and
{ra['texts_with_se_thread']} with a copied Stack Exchange comment thread that names its commenters (in audited_pools
alone: {rp.get('texts_with_url', 0)} with URLs, {rp.get('texts_with_email', 0)} with an e-mail address,
{rp.get('texts_with_se_thread', 0)} with a comment thread). These spans are masked in the released text: `[URL]`,
`[EMAIL]`, and `[USER]` for commenter names and @-handles inside the copied threads; {ra['rows_redacted']} rows carry
`redacted` = true, and `problem_sha256` is the hash of the masked text. The Opus verdicts were given on the unmasked
text. Personal names were not screened otherwise, and the texts were not screened systematically for offensive content
(none was observed in the audit notes).

## 3. Collection process

- Proposals: each round the proposer (base model, zero-shot) receives "Write one new, challenging grade-school math word
  problem about <stem>. It must require several arithmetic steps and have a single numeric answer. Output only the
  problem text, no solution." (40 stems, 1,200 requests, T = 1.0, top-p 0.95, 300 tokens; `probe/code/a3_propose.py`).
- Admission (`probe/code/a3_guard.py`): the current solver samples G = 8 answers (T = 1.0, top-p 1.0, 1,024 tokens) for
  every proposal and for 400 planted tracers; the R-Zero band admits p_mode in [0.25, 0.75]; the other guards and the
  loop_uv admission on max(p_mode, abstain) are listed in the condition table and in `admission_stat`.
- Audits: Claude Opus judges, frozen prompt (judge/PROTOCOL.md), 20-25 items per independent judge instance, 22-24
  September 2026. The audit sample is the first 200 of a Random(0) shuffle of the admitted proposer questions; merging
  scripts in `judge/code/`.
- Twins: rule-based deletion of one premise sentence from GSM-Symbolic instances with resampled numbers (GSM-HP recipe,
  CPU only, no LLM).
- UMWP probe (`probe/code/umwp_probe.py`): 600 unanswerable UMWP questions (Random(0) sample of those whose answerable
  originals exist) and their 600 answerable originals, G = 8 samples, T = 1.0, seed 17, plain and abstention prompts.
- No human subjects were involved in the data. An expert annotation is user-reported but not documented or released
  (Sections 4 and 8).

## 4. Preprocessing, cleaning, labelling

Problem texts are the proposers' outputs exactly as admitted (the training suffix "Let's think step by step and output
the final answer within \\boxed{{}}." is not included), except for the masked URL, e-mail and commenter spans
(`redacted`). No deduplication, spelling correction or filtering was applied; some texts contain artefacts of
base-model generation (a leaked "Result: 1058", a trailing note, a repeated question sentence, copied web text); the
judge labels such texts as written. Labels come only from the Opus audit; the raw part files are merged
first-verdict-wins. Released derived fields: shares in INDEX.tsv. UMWP rows keep only ids, labels and our model outputs;
the `ref` (gold answer) field of our dumps is dropped, `correct` is kept.

Judge validity (all in `judge/validation/`, summary.json). No audit was blind to run, round or condition (judge/PROTOCOL.md).
- UMWP-400 (200 unanswerable + 200 answerable, stratified): Opus detects {u4['detection_unanswerable'] * 100:.1f}% of the
  human-labelled unanswerable questions ({umwp_det}) and calls {u4['false_illposed_answerable'] * 100:.1f}% of the answerable
  ones ill-posed; binary kappa against the human labels {u4['kappa_binary']:.2f}, agreement {u4['agreement']:.3f}.
- Planted tracers (100 intact + 100 premise-deleted twins, same definitions, strictness sentence and framing; fetched
  from `sample.jsonl` in file order, and the judges could see the `trd:` / `tri:` id prefix that marks a deleted or
  intact premise): {tv['summary']['intact_judged_solvable'] * 100:.0f}% of intact judged SOLVABLE,
  {tv['summary']['deleted_judged_illposed'] * 100:.0f}% of deleted judged ill-posed (part of the deleted SOLVABLE verdicts
  are genuinely solvable deletions, see the twin validation).
- Opus vs Opus (8 new judges, reversed order, two 200-item samples): {rj_lines}.
- Fixed-pool audit vs run-level audit on the same admitted items: {pv_lines}.
- The same text in different audits: of the {dup['texts_in_several_pools']} problem texts that occur in more than one
  distinct pool, {dsame1} received different three-way verdicts and {dsame2} different binary verdicts (SOLVABLE vs
  ill-posed).
- Expert annotation: user-reported kappa 0.77 (binary) and 0.69 (three-way); comparison target (expert vs Opus or
  expert vs expert), item count, annotators and blinding not yet documented; no label file exists, and nothing of it is
  in this release.
- Human validation packet: a sealed 150-item packet (50 per Opus class, drifted rounds) awaits non-author labels; its
  item key is withheld from this release until those labels are in.

## 5. Uses

Intended: measuring curriculum drift of self-play loops; benchmarking ill-posedness detectors, abstention and admission
guards on model-written problems; studying the mechanism with the fixed pools (how one pool's admitted mix changes as the
solver sharpens); reproducing the judge-free UMWP and twin reads. Not intended: training math solvers on the problems
(many are ill-posed and none has a verified answer); treating Opus verdicts as ground truth without the error rates of
Section 4; any use of the rebuilt twins in training corpora (GSM-Symbolic canary). Composition caveats that affect use:
the audit is a 200-item sample for most pools (binomial SE about 3.5 points on a share near 50%); round-1 pools are shared
across several conditions; one judge family.

## 6. Distribution

Planned distribution: with the paper (anonymous repository during review, public afterwards). Proposed licences, to be
confirmed by the authors before release: annotations, generated problem texts and code CC BY 4.0 / MIT, subject to:
- GSM-Symbolic (Apple) is CC BY-NC-ND 4.0 (the pinned snapshot's dataset card: `license: cc-by-nc-nd-4.0`), which does
  not allow sharing adapted material; the twin texts are therefore distributed only as patches plus
  `twins/rebuild_twins.py`, which rebuilds them from the user's own copy of GSM-Symbolic (revision {GSM_SYM_REV}) and
  verifies each text's SHA-256. Whether patches with {tw['literal_share'] * 100:.1f}% literal characters are acceptable is
  for the authors to decide.
- UMWP (Sun et al., LREC-COLING 2024; {UMWP['licence']}): {UMWP['repo']}, file `{UMWP['file']}`, retrieved
  {UMWP['retrieved']} from branch {UMWP['branch']} (head commit {UMWP['head_commit']} of {UMWP['head_commit_date']}; the
  file was last changed in commit {UMWP['file_last_changed_in']}; SHA-256 {uf['sha256']}, {uf['rows']} rows). Only item
  ids, UMWP labels and our model outputs are included, no question text. The UMWP labels are CC BY-SA 4.0 data, so the
  files that carry them (`probe/umwp_subset1200.jsonl`, `probe/umwp_dumps/`,
  `judge/validation/umwp400_opus_vs_human_labels.jsonl`) should be shared under CC BY-SA 4.0 with attribution.
- Model-generated texts: Qwen3-4B/8B/14B-Base and Qwen2.5-7B-Instruct (Apache-2.0), Phi-4-mini-instruct (MIT),
  Llama-3.2-3B-Instruct (Llama 3.2 Community License and Acceptable Use Policy; we propose to mark the Llama-generated
  pools "Built with Llama"), Claude Opus notes (Anthropic usage policies). The authors should confirm these terms.
- The key of the sealed 150-item human-validation packet is not part of this release and must not be shared until the
  non-author labels are in.

## 7. Maintenance

Maintained by the authors through the paper's repository; errata and new versions get a new release directory and
MANIFEST. `build_release.py` regenerates every file from the project tree deterministically, apart from the snapshot
time.

## 8. Known limitations

- LLM-judge error: Opus misses {(1 - u4['detection_unanswerable']) * 100:.1f}% of human-labelled unanswerable UMWP questions
  and flags {u4['false_illposed_answerable'] * 100:.1f}% of answerable ones (Opus vs UMWP human labels: binary kappa
  {u4['kappa_binary']:.2f}); it flags {(1 - tv['summary']['intact_judged_solvable']) * 100:.0f}% of intact GSM-HP twins as ill-posed.
  A user-reported expert annotation gives kappa 0.77 (binary) / 0.69 (three-way), but its comparison target (expert vs
  Opus or expert vs expert), item count, annotators and blinding are not yet documented. Two independent Opus panels on
  the same items agree at kappa {min(r['kappa3'] for r in rj):.2f}-{max(r['kappa3'] for r in rj):.2f} (three-way) and {min(r['kappa_binary'] for r in rj):.2f}-{max(r['kappa_binary'] for r in rj):.2f}
  (binary) on two 200-item re-judged samples, and at {min(x['kappa3'] for x in pv):.2f}-{max(x['kappa3'] for x in pv):.2f} / {min(x['kappa_binary'] for x in pv):.2f}-{max(x['kappa_binary'] for x in pv):.2f}
  on the {sum(x['n_overlap'] for x in pv)} round-1 admitted items judged both in the run audit and in the fixed-pool audit;
  of the problem texts judged in several distinct pools, {dsame} got different three-way verdicts.
  The UNSOLVABLE / AMBIGUOUS boundary is the noisiest part: between the two re-judged panels the UNSOLVABLE-only share
  moves by {', '.join(f"{abs(r['unsolvable_share'][0] - r['unsolvable_share'][1]) * 100:.1f}" for r in rj)} points, the
  combined ill-posed share by {', '.join(f"{abs(r['illposed_share'][0] - r['illposed_share'][1]) * 100:.1f}" for r in rj)}.
- The strictness rule ("an invented default assumption makes it UNSOLVABLE") counts under-specified but conventionally
  answerable problems as ill-posed; results are about this strict notion.
- Framing and blinding: judges were told the model family and, for admitted pools, that a self-play filter admitted the
  items (the UMWP and tracer validation audits used the same framing, with Qwen3-4B-Base as the stated author). Judges
  were also not blind to round or condition: every admitted-pool judge ran `cd <REPO>/dirB/runs/<RUN>` (extension
  audits: a pseudo-run directory) and, in the file-pushing template, returned `run="<RUN>"`; the run name gives model,
  condition, seed and round (for example `qwen3_4b_Base_own_a3_s0_r3`), and the printed proposer ids encode seed and
  round too. The UMWP-400 judges saw `dirB/runs/umwp400_j1` and ids `umwp:<id>`; the
  tracer-200 judges saw `dirA3/results/audit_validation` and ids `trd:` / `tri:`, whose prefix tells whether the premise
  was deleted or kept, so the tracer validation is not a blind test. The framing is constant across rounds and
  conditions, but it may shift the level of the ill-posed share, and knowledge of the round or condition could bias
  the round-by-round comparison; no round-blind audit exists.
- One judge family (Claude Opus); no second LLM judge. The expert annotation is user-reported; its item list and labels
  are not in this release (not present on the compute host).
- Coverage: most pools have a 200-item sample; Llama-3.2-3B lineages s0-s1 are audited at round 1 only; Qwen3-14B-Base
  has one seed (rounds 1-2); later rounds exist only for some lineages; {len(pending)} admitted pools were unaudited at the
  snapshot time.
- Solver statistics come from 8 samples and depend on the answer parser (A3b: boxed answer, else '####', else last
  number); instruct models that rarely box are parsed from the last number.
- English, grade-school arithmetic word problems only.
"""


README = """# release_v1

Audited self-play curricula: admitted problems per round and lineage with Opus verdicts and solver agreement, fully
audited fixed pools, planted premise-deletion twins (as patches), the exact judge prompts and the judge-free UMWP probe.
Read DATASHEET.md first. MANIFEST.tsv lists every file with its SHA-256, size and row count; build_release.py rebuilds
everything from the project tree.

Quick start:
    import json
    rows = [json.loads(l) for l in open("audited_pools/qwen3_4b_Base_own_a3_s0_r3.jsonl")]
    share = sum(r["opus_verdict"] != "SOLVABLE" for r in rows if r["in_audit_sample200"]) / 200
    # ids are unique only within a run: join rows across runs on problem_sha256
    # judge-free reads J-P2 / J-P3 from the released dumps: cd probe && python3 reproduce_j1.py
    # twins: cd twins && python3 rebuild_twins.py [--gsm_symbolic_dir DIR]   (needs GSM-Symbolic; see the script header);
    #        the output is a drop-in --twins_file for probe/code/c0_probe.py
"""


def rows_of(path):
    if path.endswith(".jsonl"):
        return sum(1 for l in open(path, encoding="utf-8") if l.strip()), "records"
    if path.endswith(".tsv"):
        return max(0, sum(1 for _ in open(path, encoding="utf-8")) - 1), "data_rows"
    if path.endswith(".json"):
        d = jload(path)
        return (len(d) if isinstance(d, (list, dict)) else 1), "top_level_entries"
    return sum(1 for _ in open(path, encoding="utf-8", errors="replace")), "lines"


def manifest(exclude=("MANIFEST.tsv", "DATASHEET.md")):
    out = []
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, OUT)
            if rel in exclude or "__pycache__" in rel or rel.split(os.sep)[0] == "_work":
                continue
            n, unit = rows_of(p)
            out.append([rel, os.path.getsize(p), n, unit, sha(p)])
    return sorted(out)


def main():
    me = os.path.abspath(__file__)
    keep = {os.path.join(OUT, "build_release.py"), os.path.join(OUT, "_work")}
    tmp_self = open(me, encoding="utf-8").read()
    if os.path.isdir(OUT):
        for name in os.listdir(OUT):
            p = os.path.join(OUT, name)
            if os.path.abspath(p) in keep:
                continue
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(OUT, exist_ok=True)
    if os.path.abspath(me) != os.path.join(OUT, "build_release.py"):
        wtext("build_release.py", tmp_self)
    tracer_rows = audited_pools()
    raw_audits()
    tracer_rows += fixed_pools()
    twins(tracer_rows)
    judge()
    probe()
    wtext("README.md", README)
    m = manifest()
    wtext("DATASHEET.md", datasheet(m))
    m = manifest(exclude=("MANIFEST.tsv",))
    wtsv("MANIFEST.tsv", ["path", "bytes", "rows", "rows_unit", "sha256"], m)
    total = sum(x[1] for x in m) + os.path.getsize(os.path.join(OUT, "MANIFEST.tsv"))
    kc = collections.Counter()
    for x in STATS["pools"]:
        kc.update(x["audit_k_check"])
    print(json.dumps({"snapshot": SNAPSHOT, "files": len(m) + 1, "file_bytes": total, "pools": len(STATS["pools"]),
                      "pool_rows": STATS["pool_rows"], "fixed": [(f["pool"], f["n"], f["n_candidates"]) for f in STATS["fixed"]],
                      "raw": len(STATS["raw"]), "twins": STATS["twins"], "tracer_rows": STATS["tracer_rows"], "umwp": STATS["umwp"],
                      "twin_dumps": STATS["twin_dumps"], "pool_vs_run": STATS["pool_vs_run"], "dups": STATS["dups"],
                      "redaction": STATS["redaction"], "audit_k_check": dict(kc), "rejudge_k": STATS.get("rejudge_k"),
                      "admission_max_rule": [x["run"] for x in STATS["pools"] if x["admission_stat"] != "p_mode"],
                      "code_root_replacements": {r[0]: r[3] for r in STATS["code_sources"]},
                      "results_root_replacements": STATS["results_root_replacements"],
                      "derived_not_fully_covered": [(x["run"], x["n_with_verdict"], x["n_proposed"]) for x in STATS["derived"]
                                                    if x["n_with_verdict"] < x["n_proposed"]],
                      "pending": STATS["pending"]}, indent=1))


if __name__ == "__main__":
    main()
