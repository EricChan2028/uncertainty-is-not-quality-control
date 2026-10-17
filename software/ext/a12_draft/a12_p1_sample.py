"""A12 part 1 (registration A12_registration_2026-09-26.md, section 1): a sample of RAW R-Zero challenger questions.
RAW = every parsed question the questioner Q_i wrote in the 8 generation shards of iteration i (records with score 0 in
/run/ti/rzero/s<S>/generated_question/<abbr>_solver_v<i>_<k>.json.raw, k = 0..7, concatenated in shard order; score -1 records
have no <question> block or no boxed answer and are not questions). Nothing downstream of the questioner (pseudo-labelling,
its drops, the 0.3-0.8 uncertainty filter) has acted on this population.
  stage --seed S   CPU only, on the host that owns seed S (ext/rzero_runs/s<S>.host). Per iteration i = 1, 2, 3:
                   (a) checks the raw shards, the scored pool (<abbr>_solver_v<i>_all_scored.json) and the admitted set
                       (<abbr>_solver_v<i>_admitted.jsonl) against the registered A7-5 summaries (counts, admitted sha256);
                   (b) downstream status of every raw question by ordered alignment with the scored pool (upload.py
                       concatenates the shard results 0..7 in order; each shard's results keep the order of its parsed
                       questions): not_scored (dropped while pseudo-labelling: no boxed answer from S_(i-1), the "证明" / "box" /
                       "text" rules, or a killed shard) / scored_rejected / admitted, the filter being re-applied to the
                       aligned pool (0.3 <= score <= 0.8, label not '' / 'None'); the admitted rows so derived must equal
                       the admitted file row by row. If the alignment or that equality fails, status is assigned by text
                       membership instead and flagged (status_method "text"); the primary outcome does not use status;
                   (c) draws 200 positions: random.Random(2026092612 + 10 * S + i).sample(range(n_parsed), 200)
                       (a repeated question counts by multiplicity).
                   Writes dirA5/results/a12_key/stage_s<S>.jsonl (600 rows) and stage_s<S>_meta.json. Never redrawn.
  build            either host, once both stage files exist: pools the 1,200 items (s0 it1..3, then s1 it1..3), shuffles
                   them with random.Random(20260926 + 12), neutral keys y0000..y1199 (key yNNNN = line NNNN + 1); writes
                   dirA5/results/a12_blind/items.jsonl (key + problem only; UTF-8, ensure_ascii=False; U+2028, U+2029 and
                   U+0085 written as JSON escapes so that every item is exactly one line; the decoded text is unchanged) and
                   dirA5/results/a12_key/key_full.jsonl. Aborts (nothing is judged) if any 25-line batch exceeds 25,000
                   characters (the judges' fetch must fit in one tool result). Prints both sha256 for the hash log.
    /usr/bin/python3 a12_p1_sample.py stage --seed S     |     /usr/bin/python3 a12_p1_sample.py build"""
import hashlib, json, os, random, socket, sys
from collections import Counter

R = "/home/tione/notebook/rsi_rl"; RZ = "/run/ti/rzero"
KEY = f"{R}/dirA5/results/a12_key"; OUT = f"{R}/dirA5/results/a12_blind"
N_PER_CELL, ITERS, SEEDS = 200, (1, 2, 3), (0, 1)
SEP = (" ", " ", "\u0085")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def adm_rule(x):  # upload.py --min_score 0.3 --max_score 0.8, verbatim condition
    return x["score"] >= 0.3 and x["score"] <= 0.8 and x["answer"] != '' and x["answer"] != 'None'


def stage(S):
    hf = f"{R}/ext/rzero_runs/s{S}.host"
    owner = open(hf).read().split()[0]
    assert owner == socket.gethostname(), f"seed {S} belongs to host {owner}; this is {socket.gethostname()}"
    sf, mf = f"{KEY}/stage_s{S}.jsonl", f"{KEY}/stage_s{S}_meta.json"
    assert not os.path.exists(sf) and not os.path.exists(mf), f"{sf} exists: never redrawn"
    os.makedirs(KEY, exist_ok=True)
    abbr = f"rzero_qwen3_4b_Base_s{S}"; gq = f"{RZ}/s{S}/generated_question"
    rows, meta = [], {"seed": S, "host": socket.gethostname(), "cells": {}}
    for it in ITERS:
        name = f"{abbr}_solver_v{it}"
        summ = json.load(open(f"{R}/ext/rzero_runs/s{S}/{name}_summary.json", encoding="utf-8"))
        parsed, shard_sha = [], {}
        for k in range(8):
            f = f"{gq}/{name}_{k}.json.raw"
            shard_sha[k] = sha(f)
            parsed += [(k, j, r) for j, r in enumerate(json.load(open(f, encoding="utf-8"))) if r.get("score") == 0]
        pool = json.load(open(f"{gq}/{name}_all_scored.json", encoding="utf-8"))
        js = f"{gq}/{name}_admitted.jsonl"
        adm = [json.loads(l) for l in open(js, encoding="utf-8")]
        assert summ["n_raw_shards"] == 8 and summ["n_parsed"] == len(parsed), (name, "parsed count differs from the A7-5 summary")
        assert summ["n_scored"] == len(pool) and summ["n_admitted"] == len(adm), (name, "pool / admitted counts differ")
        assert summ["sha256_admitted_jsonl"] == sha(js), (name, "admitted set differs from the one A7-5 audited")
        # (b) ordered alignment of the scored pool onto the parsed questions
        status, pidx, method = ["not_scored"] * len(parsed), [None] * len(parsed), "ordered"
        try:
            p = 0
            for t, x in enumerate(pool):
                while p < len(parsed) and parsed[p][2]["question"] != x["question"]:
                    p += 1
                assert p < len(parsed), "pool item without a parsed question"
                pidx[p] = t; p += 1
            derived = [pool[t] for t in range(len(pool)) if adm_rule(pool[t])]
            assert len(derived) == len(adm) and all(d["question"] == a["problem"] and d["answer"] == a["answer"]
                                                     and abs(d["score"] - a["score"]) < 1e-9 for d, a in zip(derived, adm)), \
                "re-applied filter does not reproduce the admitted file"
            for q, t in enumerate(pidx):
                if t is not None:
                    status[q] = "admitted" if adm_rule(pool[t]) else "scored_rejected"
        except AssertionError as e:
            method = f"text ({e})"
            adm_t, pool_t = {a["problem"] for a in adm}, {x["question"] for x in pool}
            status = ["admitted" if r["question"] in adm_t else "scored_rejected" if r["question"] in pool_t else "not_scored"
                      for _, _, r in parsed]
            pidx = [None] * len(parsed)
        mult = Counter(r["question"] for _, _, r in parsed)
        idx = random.Random(2026092612 + 10 * S + it).sample(range(len(parsed)), N_PER_CELL)
        for rank, q in enumerate(idx):
            k, j, r = parsed[q]; t = pidx[q]
            rows.append({"seed": S, "iteration": it, "run": f"rzero_qwen3_4b_Base_s{S}_it{it}", "raw_id": f"a12s{S}i{it}k{k}j{j}",
                         "raw_pos": q, "rank": rank, "status": status[q], "status_method": method.split(" ")[0],
                         "pool_score": pool[t]["score"] if t is not None else None,
                         "pool_label": pool[t]["answer"] if t is not None else None,
                         "text_multiplicity": mult[r["question"]], "problem": r["question"]})
        meta["cells"][f"it{it}"] = {"n_generated": summ["n_generated"], "n_parsed": len(parsed),
                                    "n_scored": len(pool), "n_admitted": len(adm), "status_method": method,
                                    "status_counts_all_parsed": dict(Counter(status)), "shard_sha256": shard_sha,
                                    "pool_sha256": sha(f"{gq}/{name}_all_scored.json"), "admitted_sha256": sha(js),
                                    "sample_seed": 2026092612 + 10 * S + it,
                                    "sample_status_counts": dict(Counter(status[q] for q in idx))}
    with open(sf + ".tmp", "w", encoding="utf-8") as fh:
        for x in rows:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    json.dump(meta, open(mf + ".tmp", "w"), indent=1)
    os.replace(mf + ".tmp", mf); os.replace(sf + ".tmp", sf)
    print(json.dumps({"stage": sf, "sha256": sha(sf), "meta_sha256": sha(mf),
                      "cells": {c: {k: v[k] for k in ("n_parsed", "status_method")} for c, v in meta["cells"].items()}}))


def line_of(x):
    s = json.dumps({"key": x["key"], "problem": x["problem"]}, ensure_ascii=False)
    for c in SEP:
        s = s.replace(c, "\\u%04x" % ord(c))
    assert "\n" not in s and "\r" not in s
    return s


def build():
    for p in (f"{OUT}/items.jsonl", f"{KEY}/key_full.jsonl"):
        assert not os.path.exists(p), f"{p} exists: sample already built"
    items = []
    for S in SEEDS:
        st = [json.loads(l) for l in open(f"{KEY}/stage_s{S}.jsonl", encoding="utf-8")]
        c = Counter((x["seed"], x["iteration"]) for x in st)
        assert c == Counter({(S, i): N_PER_CELL for i in ITERS}), (S, c)
        items += st
    random.Random(20260926 + 12).shuffle(items)
    for k, x in enumerate(items):
        x["key"] = f"y{k:04d}"
    lines = [line_of(x) for x in items]
    for x, l in zip(items, lines):
        assert json.loads(l) == {"key": x["key"], "problem": x["problem"]}, x["key"]
    big = [(a + 1, a + 25, sum(len(l) + 1 for l in lines[a:a + 25])) for a in range(0, len(lines), 25)]
    worst = max(big, key=lambda t: t[2])
    assert worst[2] <= 25000, f"batch {worst[:2]} holds {worst[2]} characters (> 25,000): NOT RUN, nothing is judged"
    os.makedirs(OUT); os.makedirs(KEY, exist_ok=True)
    with open(f"{KEY}/key_full.jsonl", "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(f"{OUT}/items.jsonl", "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")
    print(json.dumps({"n": len(items), "largest_batch_chars": worst[2], "items_sha256": sha(f"{OUT}/items.jsonl"),
                      "key_sha256": sha(f"{KEY}/key_full.jsonl")}))


if __name__ == "__main__":
    if sys.argv[1:2] == ["stage"] and len(sys.argv) == 4 and sys.argv[2] == "--seed" and sys.argv[3] in ("0", "1"):
        stage(int(sys.argv[3]))
    elif sys.argv[1:] == ["build"]:
        build()
    else:
        sys.exit("usage: a12_p1_sample.py stage --seed 0|1  |  a12_p1_sample.py build")
