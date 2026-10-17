#!/usr/bin/env python3
"""H1 read: do human annotators see the self-play drift toward ill-posed questions?
Registered in H1_registration_2026-09-26.md (same folder), written after the annotation was completed and before the labels
were joined with the key (the registration states exactly which parts of which files were displayed before the read). This
script implements that registration; nothing else is computed.

Inputs (defaults = the registered files; their sha256 must equal REGISTERED unless --synthetic is given):
  labels_ann1.csv, labels_ann2.csv   hid,label,note; 240 data rows D001..D240; label S/U/A (or SOLVABLE/UNSOLVABLE/AMBIGUOUS)
                                     after strip().upper()
  labels_adjudicated.csv             hid,label; one row per three-class disagreement (same label vocabulary)
  adjudication_rationale.csv         hid,ann1,ann2,adjudicated,rationale; consistency check only (free text never coded)
  _key_do_not_open/key.json          240 x {hid, a7b_key, seed, round, opus_original, opus_blind}
  packet.md                          the annotators' packet; integrity check only: the text under each hid must equal the
                                     key_full.jsonl problem of that hid's a7b_key
  dirA5/results/a7b_blind/key_full.jsonl, blind_verdicts.json   the 900 A7b blind re-audit items (seed, round, Opus labels)
  dirA5/results/a10/verdicts.jsonl   A10 open-weight judge (Qwen2.5-32B-Instruct); its "key" is the A7b key (= a7b_key)
Ill-posed = U or A. Shares are in percentage points. Bootstrap: B = 4,000 replicates, percentile interval = sorted finite
replicates at positions int(0.025 nv) and int(0.975 nv) - 1 (nv = number of finite replicates, reported). Every replicate
matrix is drawn from a fresh random.Random(20260926 + 21): "item" = the 240 items resampled with replacement (hid order);
"stratified" = items resampled within seed x round (cells in (seed, round) order, items in hid order). One item matrix and one
stratified matrix are shared by all statistics, so differences between label sources are paired. The exploratory 900-item
stream uses random.Random(20260926 + 22) (stratified by seed x round, items in A7b key order).
Final label: the common label where the annotators agree (three-class), else the adjudicated label. If the only disagreements
without an adjudicated label are U vs A, those items take the binary final label ill-posed and every statistic that needs a
three-class final label is reported as not available; an S-vs-ill-posed disagreement without an adjudicated label is NOT RUN.
NOT RUN (exit 2, nothing written; messages give counts only, never hids or label values): a missing input file; a sha256 that
differs from the registration; an annotator file whose data rows are not exactly D001..D240 once each (240 rows); an invalid
or empty annotator label; an adjudicated row with an invalid label, an unknown hid or a conflicting duplicate; an
S-vs-ill-posed disagreement without an adjudicated label; a key.json that is not 240 valid entries with 80 per round, or whose
a7b_key/seed/round do not match key_full.jsonl; a packet.md that does not parse into D001..D240, or whose text under a hid
differs from the key_full.jsonl problem of that hid's a7b_key; a malformed key_full / blind-verdict / open-judge file.
Output human_read.json is created once (exclusive create; exit 3 if it exists). Deterministic (no clock, no unseeded RNG).
  python3 human_read.py
  dev tests: python3 human_read.py --synthetic --hv-dir D --key K --a7b-key-full F --a7b-blind V --open-judge O --out J"""
import argparse, csv, hashlib, io, json, math, os, random, re, sys
from collections import Counter
import numpy as np

A5 = "/home/tione/notebook/rsi_rl/dirA5"
HV = f"{A5}/human_validation_drift"
REGISTERED = {
    "ann1": "0d6b3e2b24ba0a25d87f777df18733608823cea2460c46a84c06201da2a925ae",
    "ann2": "b98bf4160fc70bf93bd3f51f2a7cd9aa22a36f091e499eec1b9d740f43b48674",
    "adjudicated": "cd4e9003cb671f17cc618578c95553bdf958d78840a5ef0af0a282912ce75dc9",
    "rationale": "a87702b199f543ce1cb9f3e7b3b6045f5a3fc09195b4da4b95e76640794b72d4",
    "key": "91d86caaf4978fc0e3740bb34b5ac7681efa94b87cc30cc1286ee74259e2af5f",
    "packet": "fe667ec21c5542d900a977dec09df69daeb80c7620a8c7aa5f912e0fab4b6f95",
    "a7b_key_full": "c4f16b0b9488f55c60c2e2b11e5065dd2e3035e7a2f558f23efc1014ae774bff",
    "a7b_blind": "67957b08064db94482612c4534629874489f99eae2fa99276847f74373210815",
    "open_judge": "574b222f5dbd160bbe9e0d91cc88cb84b75b20b2f553f95e6a782904fde68915"}
SEED, SEED_900, B = 20260926 + 21, 20260926 + 22, 4000
Z = 1.959963984540054
LABELS = ("S", "U", "A"); CODE = {"S": 0, "U": 1, "A": 2}
WORDS = {"SOLVABLE": "S", "UNSOLVABLE": "U", "AMBIGUOUS": "A"}
HIDS = [f"D{i:03d}" for i in range(1, 241)]; HSET = set(HIDS); N = len(HIDS)
ROUNDS, SEEDS = (1, 3, 5), (0, 1, 2)
READING_RULE = ("RISES if the CI lower bound > 0; FALLS if the upper bound < 0; FLAT if the CI lies within [-10, +10] and "
                "contains 0; INCONCLUSIVE otherwise")
PACKET_SECTION = re.compile(r"^## (D\d{3})\n\n(.*?)\n\nLabel \(S / A / U\): ____   Note: ____\n", re.S | re.M)


class NotRun(Exception):
    pass


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def read_rows(path):
    raw = open(path, "rb").read()
    try:
        text, repl = raw.decode("utf-8-sig"), False
    except UnicodeDecodeError:  # hid/label columns are ASCII; only free-text notes can be affected
        text, repl = raw.decode("utf-8-sig", errors="replace"), True
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        return [], [], 0, repl
    head = [h.strip().lower() for h in rows[0]]
    body = [r for r in rows[1:] if any(c.strip() for c in r)]
    return head, body, len(rows) - 1 - len(body), repl


def fld(r, i):
    return r[i] if i < len(r) else ""


def judge_label(v):
    s = str(v if v is not None else "").strip().upper()
    s = WORDS.get(s, s)
    return s if s in CODE else None


def human_label(raw):
    """Annotator / adjudicated label: strip().upper(); the full words SOLVABLE/UNSOLVABLE/AMBIGUOUS map to S/U/A.
    Returns (label or the normalised invalid string, given_as_word)."""
    s = raw.strip().upper()
    return (WORDS[s], True) if s in WORDS else (s, False)


def load_annotator(path, name, problems):
    head, body, blank, repl = read_rows(path)
    info = {"header": ",".join(head), "header_as_registered": head == ["hid", "label", "note"], "n_data_rows": len(body),
            "n_blank_rows_ignored": blank, "undecodable_bytes_replaced": repl}
    if "hid" not in head or "label" not in head:
        problems.append(f"{name}: header lacks a hid or label column (header not printed)")
        return None, info
    ih, il = head.index("hid"), head.index("label")
    lab, n_dup, n_unknown, n_bad, n_norm, n_word = {}, 0, 0, 0, 0, 0
    for r in body:
        hid, raw = fld(r, ih).strip().upper(), fld(r, il)
        lb, word = human_label(raw)
        if hid not in HSET:
            n_unknown += 1; continue
        if hid in lab:
            n_dup += 1; continue
        lab[hid] = lb
        if lb not in CODE:
            n_bad += 1
        else:
            n_norm += raw != lb; n_word += word
    n_missing = sum(h not in lab for h in HIDS)
    # messages give counts only (no hid, no label value)
    if len(body) != N:
        problems.append(f"{name}: {len(body)} data rows, expected {N}")
    if n_unknown:
        problems.append(f"{name}: {n_unknown} row(s) with an unknown or empty hid")
    if n_dup:
        problems.append(f"{name}: {n_dup} duplicate hid row(s)")
    if n_missing:
        problems.append(f"{name}: {n_missing} hid(s) without a row")
    if n_bad:
        problems.append(f"{name}: {n_bad} invalid or empty label(s)")
    info.update({"n_labels_normalised": n_norm, "n_labels_given_as_words": n_word, "n_invalid_or_empty_labels": n_bad,
                 "n_hids_without_row": n_missing, "label_counts": {x: sum(v == x for v in lab.values()) for x in LABELS}})
    return lab, info


def load_adjudicated(path, problems):
    head, body, blank, repl = read_rows(path)
    info = {"header": ",".join(head), "header_as_registered": head == ["hid", "label"], "n_data_rows": len(body),
            "n_blank_rows_ignored": blank}
    if "hid" not in head or "label" not in head:
        problems.append(f"labels_adjudicated.csv: header lacks a hid or label column (header not printed)")
        return None, info
    ih, il = head.index("hid"), head.index("label")
    adj, n_same, conflict, n_unknown, n_bad, n_word = {}, 0, set(), 0, 0, 0
    for r in body:
        hid, raw = fld(r, ih).strip().upper(), fld(r, il)
        lb, word = human_label(raw)
        if hid not in HSET:
            n_unknown += 1; continue
        if lb not in CODE:
            n_bad += 1; continue
        n_word += word
        if hid in adj:
            if adj[hid] == lb:
                n_same += 1
            else:
                conflict.add(hid)
            continue
        adj[hid] = lb
    if n_unknown:
        problems.append(f"labels_adjudicated.csv: {n_unknown} row(s) with an unknown or empty hid")
    if n_bad:
        problems.append(f"labels_adjudicated.csv: {n_bad} invalid or empty label(s)")
    if conflict:
        problems.append(f"labels_adjudicated.csv: {len(conflict)} hid(s) with conflicting duplicate rows")
    info.update({"n_duplicate_rows_same_label": n_same, "n_labels_given_as_words": n_word})
    return adj, info


def rationale_check(path, ann1, ann2, adj, disagreements):
    head, body, blank, repl = read_rows(path)
    want = ["hid", "ann1", "ann2", "adjudicated"]
    out = {"header_as_registered": head == want + ["rationale"], "n_data_rows": len(body)}
    if any(c not in head for c in want):
        out["checked"] = False
        return out
    ix = {c: head.index(c) for c in want}
    hids, mism = [], Counter()
    for r in body:
        hid = fld(r, ix["hid"]).strip().upper(); hids.append(hid)
        if hid not in HSET:
            mism["unknown_hid"] += 1; continue
        for col, ref in (("ann1", ann1), ("ann2", ann2), ("adjudicated", adj)):
            if judge_label(fld(r, ix[col])) != ref.get(hid):
                mism[col] += 1
    ds, hs = set(disagreements), set(hids)
    out.update({"checked": True, "hid_set_equals_disagreements": hs == ds, "n_disagreements_without_row": len(ds - hs),
                "n_rows_not_a_disagreement": len(hs - ds), "n_duplicate_hids": len(hids) - len(hs),
                "column_mismatches_vs_label_files": {c: mism[c] for c in ("ann1", "ann2", "adjudicated", "unknown_hid")}})
    return out


def load_key(path, problems):
    try:
        key = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        problems.append(f"key.json unreadable ({type(e).__name__})"); return None
    if not isinstance(key, list) or len(key) != N:
        problems.append(f"key.json: expected a list of {N} entries"); return None
    out = {}
    for e in key:
        try:
            hid, ak, s, r = str(e["hid"]).strip().upper(), str(e["a7b_key"]), int(e["seed"]), int(e["round"])
        except Exception:
            problems.append("key.json: an entry lacks hid/a7b_key/seed/round"); return None
        if hid not in HSET or hid in out:
            problems.append("key.json: an entry has an unknown or duplicate hid"); return None
        if s not in SEEDS or r not in ROUNDS:
            problems.append("key.json: an entry has seed/round outside 0-2 / 1,3,5"); return None
        out[hid] = {"a7b_key": ak, "seed": s, "round": r, "opus_original": judge_label(e.get("opus_original")),
                    "opus_blind": judge_label(e.get("opus_blind"))}
    per_round = Counter(v["round"] for v in out.values())
    if any(per_round[r] != 80 for r in ROUNDS):
        problems.append(f"key.json: items per round {dict(sorted(per_round.items()))}, expected 80 each")
    if len({v["a7b_key"] for v in out.values()}) != N:
        problems.append("key.json: duplicate a7b_key")
    return out


def load_packet(path, problems):
    """hid -> problem text as shown to the annotators (hv_drift_packet.py writes '## Dnnn', a blank line, the stripped problem,
    a blank line, then the fixed 'Label (S / A / U): ____   Note: ____' line)."""
    try:
        secs = PACKET_SECTION.findall(open(path, encoding="utf-8").read())
    except UnicodeDecodeError:
        problems.append("packet.md: not valid UTF-8"); return None
    out = {}
    for hid, text in secs:
        out.setdefault(hid, []).append(text)
    if len(secs) != N or sorted(out) != HIDS or any(len(v) != 1 for v in out.values()):
        problems.append(f"packet.md: {len(secs)} parsed sections, expected the 240 hids once each"); return None
    return {h: v[0] for h, v in out.items()}


def load_a7b(kf_path, bv_path):
    K = {}
    try:
        for line in open(kf_path, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                K[str(x["key"])] = {"seed": int(x["seed"]), "round": int(x["round"]), "orig": judge_label(x.get("orig_verdict")),
                                    "problem": str(x.get("problem", "")).strip()}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise NotRun([f"key_full.jsonl malformed ({type(e).__name__})"])
    seen, conflict = {}, set()
    try:
        for v in json.load(open(bv_path, encoding="utf-8")):
            k, lb = str(v.get("key")), judge_label(v.get("verdict"))
            if lb is None:
                continue
            if k in seen and seen[k] != lb:
                conflict.add(k)
            seen.setdefault(k, lb)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise NotRun([f"blind_verdicts.json malformed ({type(e).__name__})"])
    BO = {k: lb for k, lb in seen.items() if k in K and k not in conflict}
    return K, BO, {"n_items": len(K), "n_blind_valid": len(BO), "n_blind_conflicting_duplicates": len(conflict)}


def load_open(path):
    seen, conflict, n_lines, n_null = {}, set(), 0, 0
    try:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            n_lines += 1
            x = json.loads(line); k, lb = str(x.get("key")), judge_label(x.get("verdict"))
            if lb is None:
                n_null += 1; continue
            if k in seen and seen[k] != lb:
                conflict.add(k)
            seen.setdefault(k, lb)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise NotRun([f"open-judge verdicts.jsonl malformed ({type(e).__name__})"])
    return ({k: lb for k, lb in seen.items() if k not in conflict},
            {"n_lines": n_lines, "n_lines_without_valid_verdict": n_null, "n_conflicting_duplicate_keys": len(conflict)})


# ---------------------------------------------------------------- statistics
def item_idx(n, seed):
    rr = random.Random(seed).randrange
    return np.array([[rr(n) for _ in range(n)] for _ in range(B)], dtype=np.int64)


def strat_idx(cells, seed):
    rr = random.Random(seed).randrange
    rows = []
    for _ in range(B):
        row = []
        for c in cells:
            m = len(c); row.extend(c[rr(m)] for _ in range(m))
        rows.append(row)
    return np.array(rows, dtype=np.int64)


def pct(bs):
    v = np.asarray(bs, dtype=float); v = np.sort(v[np.isfinite(v)]); nv = len(v)
    if nv == 0:
        return None, 0
    return [float(v[int(0.025 * nv)]), float(v[int(0.975 * nv) - 1])], nv


def wilson(k, n):
    if n == 0:
        return None
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d; h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [100 * max(0.0, c - h), 100 * min(1.0, c + h)]


def share_block(k, n):
    return {"k": int(k), "n": int(n), "share_pct": 100 * k / n if n else None, "wilson95_pct": wilson(k, n)}


def gshare(y, m, cond, I):
    """100 * share of y among items with m & cond, for each row of the index matrix I (bool arrays over items)."""
    sel = (m & cond)[I]; num = (y[I] & sel).sum(1); den = sel.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, 100.0 * num / den, np.nan)


def kappa_vals(a, b, m, I, K):
    A, Bb, M = a[I], b[I], m[I]
    n = M.sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        po = ((A == Bb) & M).sum(1) / n
        pe = sum(((A == k) & M).sum(1).astype(float) * ((Bb == k) & M).sum(1) for k in range(K)) / (n * n)
        return np.where(pe < 1, (po - pe) / (1 - pe), np.nan), po


def reading(ci):
    if ci is None:
        return None
    lo, hi = ci
    if lo > 0:
        return "RISES"
    if hi < 0:
        return "FALLS"
    if lo >= -10 and hi <= 10:
        return "FLAT"
    return "INCONCLUSIVE"


def diff_block(pt, bs, n_hi, n_lo, with_reading):
    ci, nv = pct(bs)
    d = {"d_pts": pt, "ci95_pts": ci, "n_hi": int(n_hi), "n_lo": int(n_lo), "boot_valid_replicates": nv}
    if with_reading:
        d["reading"] = reading(ci)
    return d


def prf_counts(h_ill, j_ill, s):
    tp = int((h_ill & j_ill & s).sum()); fp = int((~h_ill & j_ill & s).sum())
    fn = int((h_ill & ~j_ill & s).sum()); tn = int((~h_ill & ~j_ill & s).sum())
    prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and prec + rec > 0 else None
    return {"n": tp + fp + fn + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1}


def sexp(v):
    """math.exp that returns None instead of raising OverflowError."""
    try:
        return math.exp(v)
    except OverflowError:
        return None


SEPARATION = ("not estimable: (quasi-)separation (every solvable item is at a round <= every ill-posed item's round, or the "
              "reverse), so the ML slope is infinite")


def logistic_trend(y, x):
    y = y.astype(float); xf = x.astype(float); X = np.column_stack([np.ones(len(xf)), xf])
    base = {"n": int(len(y)), "model": "logit P(ill-posed) = b0 + b1 * round, round in {1, 3, 5}, item level, Newton-Raphson ML, "
                                      "Wald 95% CI"}
    if y.min() == y.max():
        return dict(base, status="not estimable: all labels equal")
    x0, x1 = xf[y == 0], xf[y == 1]
    if x0.max() <= x1.min() or x1.max() <= x0.min():
        return dict(base, status=SEPARATION)
    beta, conv, it = np.zeros(2), False, 0
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for it in range(1, 101):
                p = 1 / (1 + np.exp(-(X @ beta))); H = X.T @ (X * (p * (1 - p))[:, None])
                step = np.linalg.solve(H, X.T @ (y - p)); beta = beta + step
                if np.max(np.abs(step)) < 1e-12:
                    conv = True; break
            p = 1 / (1 + np.exp(-(X @ beta))); H = X.T @ (X * (p * (1 - p))[:, None])
            se = float(np.sqrt(np.linalg.inv(H)[1, 1]))
        b1 = float(beta[1])
        if not (np.all(np.isfinite(beta)) and math.isfinite(se) and se > 0):
            raise FloatingPointError
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError):
        return dict(base, status="not estimable: numerical failure of the Newton-Raphson fit (near-separation)")
    z = b1 / se; ci = [b1 - Z * se, b1 + Z * se]
    return dict(base, status="estimated", b0=float(beta[0]), b1_per_round=b1, se_b1=se, wald_ci95_b1=ci, wald_z=z,
                wald_p_two_sided=math.erfc(abs(z) / math.sqrt(2)), odds_ratio_per_round=sexp(b1),
                odds_ratio_per_round_ci95=[sexp(ci[0]), sexp(ci[1])], odds_ratio_r5_vs_r1=sexp(4 * b1),
                iterations=it, converged=conv, separation_suspected=(not conv) or abs(b1) > 10)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o); return f if math.isfinite(f) else None
    return o


# ---------------------------------------------------------------- analysis
def run(a):
    paths = {"ann1": f"{a.hv_dir}/labels_ann1.csv", "ann2": f"{a.hv_dir}/labels_ann2.csv",
             "adjudicated": f"{a.hv_dir}/labels_adjudicated.csv", "rationale": f"{a.hv_dir}/adjudication_rationale.csv",
             "key": a.key, "packet": f"{a.hv_dir}/packet.md", "a7b_key_full": a.a7b_key_full, "a7b_blind": a.a7b_blind,
             "open_judge": a.open_judge}
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise NotRun([f"missing input file {p}" for p in missing])
    sha = {k: sha256(p) for k, p in paths.items()}
    if not a.synthetic:
        bad = [f"{k}: sha256 {sha[k]} differs from the registered {REGISTERED[k]}" for k in paths if sha[k] != REGISTERED[k]]
        if bad:
            raise NotRun(bad)
    problems = []
    ann1, q1 = load_annotator(paths["ann1"], "labels_ann1.csv", problems)
    ann2, q2 = load_annotator(paths["ann2"], "labels_ann2.csv", problems)
    adj, qa = load_adjudicated(paths["adjudicated"], problems)
    key = load_key(paths["key"], problems)
    packet = load_packet(paths["packet"], problems)
    if problems:
        raise NotRun(problems)
    disagreements = [h for h in HIDS if ann1[h] != ann2[h]]
    no_adj = [h for h in disagreements if h not in adj]
    no_adj_binary = [h for h in no_adj if (ann1[h] == "S") != (ann2[h] == "S")]
    if no_adj_binary:
        raise NotRun([f"{len(no_adj_binary)} S-vs-ill-posed disagreement(s) without an adjudicated label"])
    unres = set(no_adj)  # U-vs-A disagreements without an adjudicated label: binary final label ill-posed, no three-class label
    adjd = [h for h in disagreements if h in adj]
    K, BO, q900 = load_a7b(paths["a7b_key_full"], paths["a7b_blind"])
    n_nokey = sum(key[h]["a7b_key"] not in K for h in HIDS)
    n_cell = sum(key[h]["a7b_key"] in K and (K[key[h]["a7b_key"]]["seed"], K[key[h]["a7b_key"]]["round"])
                 != (key[h]["seed"], key[h]["round"]) for h in HIDS)
    bad = ([f"{n_nokey} hid(s): a7b_key not found in key_full.jsonl"] if n_nokey else []) + \
          ([f"{n_cell} hid(s): seed/round differ from key_full.jsonl"] if n_cell else [])
    if bad:
        raise NotRun(bad)
    n_text = sum(packet[h] != K[key[h]["a7b_key"]]["problem"] for h in HIDS)
    if n_text:
        raise NotRun([f"{n_text} hid(s): packet.md text differs from the key_full.jsonl problem of the hid's a7b_key"])
    OW, qow = load_open(paths["open_judge"])

    # ---- arrays over the 240 items in hid order
    rnd = np.array([key[h]["round"] for h in HIDS]); sd = np.array([key[h]["seed"] for h in HIDS])
    final = {h: ann1[h] if ann1[h] == ann2[h] else adj.get(h) for h in HIDS}  # None only for the U-vs-A items in unres
    a1 = np.array([CODE[ann1[h]] for h in HIDS]); a2 = np.array([CODE[ann2[h]] for h in HIDS])
    fc = np.array([CODE[final[h]] if final[h] is not None else -1 for h in HIDS]); h_ill = fc != 0
    three_ok = not unres
    NA3 = {"status": f"not available: {len(unres)} U-vs-A disagreement(s) without an adjudicated label have the binary final "
                     "label ill-posed but no three-class final label"}
    ALL = np.ones(N, dtype=bool); I0 = np.arange(N)[None, :]
    cells = [c for c in ([i for i in range(N) if sd[i] == s and rnd[i] == r] for s in SEEDS for r in ROUNDS) if c]
    P = item_idx(N, SEED); S = strat_idx(cells, SEED)
    r1, r3, r5 = (rnd == 1), (rnd == 3), (rnd == 5)

    def drift(y, m, c_hi, c_lo, Ib=S, Ip=I0):
        return float(gshare(y, m, c_hi, Ip)[0] - gshare(y, m, c_lo, Ip)[0]), gshare(y, m, c_hi, Ib) - gshare(y, m, c_lo, Ib)

    def drift_block(y, m, c_hi, c_lo, with_reading):
        pt, bs = drift(y, m, c_hi, c_lo)
        return diff_block(pt, bs, (m & c_hi).sum(), (m & c_lo).sum(), with_reading)

    def round_shares(y, m):
        return {f"r{r}": share_block((y & m & (rnd == r)).sum(), (m & (rnd == r)).sum()) for r in ROUNDS}

    def agree_block(x, y, m, Kc):
        kv, po = kappa_vals(x, y, m, P, Kc); k0, p0 = kappa_vals(x, y, m, I0, Kc)
        ci_k, nv_k = pct(kv); ci_p, nv_p = pct(po)
        return {"n": int(m.sum()), "raw_agreement": float(p0[0]), "raw_agreement_ci95": ci_p, "kappa": float(k0[0]),
                "kappa_ci95": ci_k, "boot_valid_replicates": nv_k}

    def confusion(x, y, m):
        return {lx: {ly: int(((x == CODE[lx]) & (y == CODE[ly]) & m).sum()) for ly in LABELS} for lx in LABELS}

    # ---- 1. quality
    extra = [h for h in adj if h not in set(disagreements)]
    quality = {
        "ann1": q1, "ann2": q2,
        "agreement": {
            "three_class": agree_block(a1, a2, ALL, 3),
            "binary_illposed_vs_solvable": agree_block((a1 != 0).astype(int), (a2 != 0).astype(int), ALL, 2),
            "confusion_ann1_rows_ann2_cols": confusion(a1, a2, ALL),
            "n_disagree_three_class": len(disagreements), "n_disagree_binary": int(((a1 != 0) != (a2 != 0)).sum()),
            "bootstrap": "item (240 items with replacement), random.Random(20260947), 4000 replicates, percentile"},
        "adjudication": dict(qa, n_disagreements=len(disagreements), n_used=len(adjd),
                             n_u_vs_a_disagreements_without_adjudication=len(unres),
                             n_extra_rows_on_agreed_items=len(extra),
                             n_extra_rows_conflicting_with_common_label=sum(adj[h] != ann1[h] for h in extra),
                             adjudicated_label_counts={x: sum(adj[h] == x for h in adjd) for x in LABELS},
                             adjudicated_equals={"ann1": sum(adj[h] == ann1[h] for h in adjd),
                                                 "ann2": sum(adj[h] == ann2[h] for h in adjd),
                                                 "neither": sum(adj[h] not in (ann1[h], ann2[h]) for h in adjd)}),
        "rationale_consistency": rationale_check(paths["rationale"], ann1, ann2, adj, disagreements)}

    # ---- 2. final label
    final_block = {"rule": ("common label where the annotators agree (three-class), else the adjudicated label; a U-vs-A "
                            "disagreement without an adjudicated label is ill-posed (binary) with no three-class label"),
                   "n": N, "n_agreed": N - len(disagreements), "n_from_adjudication": len(adjd),
                   "n_u_vs_a_without_adjudication_binary_illposed": len(unres), "three_class_final_label_available": three_ok,
                   "label_counts_binary": {"S": int((~h_ill).sum()), "U_or_A": int(h_ill.sum())},
                   "label_counts": {x: int((fc == CODE[x]).sum()) for x in LABELS} if three_ok else NA3}

    # ---- 3. primary + secondary human trend
    pt51, bs51 = drift(h_ill, ALL, r5, r1)
    primary = {"definition": "human ill-posed share (final label U or A) by round, pooled over seeds; D_h = share(r5) - share(r1)",
               "illposed_share_by_round": round_shares(h_ill, ALL),
               "D_h_r5_minus_r1": diff_block(pt51, bs51, r5.sum(), r1.sum(), True),
               "bootstrap": "stratified (items resampled within seed x round), random.Random(20260947), 4000 replicates, percentile",
               "ci_scope": ("the three seeds are fixed strata: the CI covers item-sampling uncertainty within these three runs, "
                            "not seed-to-seed variability (see the per-seed D_h)"),
               "reading_rule": READING_RULE,
               "reading_note": ("with 80 items per round the replicate shares lie on a 1.25-point lattice, so a lower bound of "
                                "exactly 0 is possible; it is not above 0")}
    per_seed = {}
    for s in SEEDS:
        per_seed[f"seed{s}"] = drift_block(h_ill, ALL, r5 & (sd == s), r1 & (sd == s), False)
    secondary_trend = {"D_h_r3_minus_r1": drift_block(h_ill, ALL, r3, r1, False), "D_h_r5_minus_r1_per_seed": per_seed,
                       "logistic_trend_over_round": logistic_trend(h_ill.astype(int), rnd)}

    # ---- 4. judge validity
    judges = {"opus_original": ({h: key[h]["opus_original"] for h in HIDS}, "key.json opus_original (non-blind Opus audit)"),
              "opus_blind": ({h: key[h]["opus_blind"] for h in HIDS}, "key.json opus_blind (A7b blind Opus re-audit)"),
              "open_weight_qwen2.5_32b": ({h: OW.get(key[h]["a7b_key"]) for h in HIDS},
                                          "dirA5/results/a10/verdicts.jsonl field verdict, joined on key == a7b_key")}
    judge_out = {}
    for name, (labs, src) in judges.items():
        m = np.array([labs[h] is not None for h in HIDS])
        jc = np.array([CODE[labs[h]] if labs[h] is not None else -1 for h in HIDS]); j_ill = jc > 0
        fpr_rounds = {f"r{r}": share_block((j_ill & ~h_ill & m & (rnd == r)).sum(), (~h_ill & m & (rnd == r)).sum()) for r in ROUNDS}
        pf, bf = drift(j_ill, m & ~h_ill, r5, r1)
        pj, bj = drift(j_ill, m, r5, r1); ph, bh = drift(h_ill, m, r5, r1)
        judge_out[name] = {
            "source": src, "n_joined": int(m.sum()), "n_not_joined": int(N - m.sum()),
            "agreement_three_class": agree_block(fc, jc, m, 3) if three_ok else NA3,
            "agreement_binary": agree_block(h_ill.astype(int), j_ill.astype(int), m, 2),
            "note_agreement": "raw_agreement = accuracy of the judge against the final human label",
            "confusion_human_rows_judge_cols": confusion(fc, jc, m) if three_ok else NA3,
            "illposed_detection_vs_humans": {"all": prf_counts(h_ill, j_ill, m),
                                             **{f"r{r}": prf_counts(h_ill, j_ill, m & (rnd == r)) for r in ROUNDS}},
            "false_positive_rate_by_round": fpr_rounds,
            "fpr_r5_minus_r1": diff_block(pf, bf, (m & ~h_ill & r5).sum(), (m & ~h_ill & r1).sum(), False),
            "judge_illposed_share_by_round": round_shares(j_ill, m),
            "drift_on_same_items": {"judge_D": diff_block(pj, bj, (m & r5).sum(), (m & r1).sum(), True),
                                    "human_D_h_same_items": diff_block(ph, bh, (m & r5).sum(), (m & r1).sum(), True),
                                    "judge_minus_human_paired": diff_block(pj - ph, bj - bh, (m & r5).sum(), (m & r1).sum(), False)}}
    secondary_judges = {"bootstrap": "kappa/agreement: item bootstrap; FPR difference and drifts: stratified; seed 20260947",
                        "fpr_definition": "judge ill-posed among items the humans label S", "judges": judge_out}

    # ---- 5. robustness
    def rob(y):
        return {"illposed_share_by_round": round_shares(y, ALL), "D_r5_minus_r1": drift_block(y, ALL, r5, r1, True)}
    robustness = {"annotator1_alone": rob(a1 != 0), "annotator2_alone": rob(a2 != 0),
                  "unsolvable_only_final": rob(fc == 1) if three_ok else NA3,
                  "note": "descriptive; the reading uses the pre-stated rule but only the primary D_h is confirmatory"}

    # exploratory: misclassification-corrected Opus (blind) drift on the 900
    keys9 = sorted(K); n9 = len(keys9)
    r9 = np.array([K[k]["round"] for k in keys9]); s9 = np.array([K[k]["seed"] for k in keys9])
    m9 = np.array([k in BO for k in keys9]); y9 = np.array([BO.get(k) in ("U", "A") for k in keys9])
    cells9 = [c for c in ([i for i in range(n9) if s9[i] == s and r9[i] == r] for s in SEEDS for r in ROUNDS) if c]
    S9 = strat_idx(cells9, SEED_900); I09 = np.arange(n9)[None, :]
    bl = [BO.get(key[h]["a7b_key"]) for h in HIDS]
    mj = np.array([x is not None for x in bl]); jb = np.array([x in ("U", "A") for x in bl])

    def rogan_gladen(Iq, I9):
        pis, det = {}, {}
        for r in ROUNDS:
            p = gshare(y9, m9, r9 == r, I9) / 100
            se = gshare(jb, mj & h_ill, rnd == r, Iq) / 100
            sp = 1 - gshare(jb, mj & ~h_ill, rnd == r, Iq) / 100
            J = se + sp - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = np.where(J > 0, (p + sp - 1) / J, np.nan)
            pis[r] = np.clip(raw, 0, 1); det[r] = (p, se, sp, raw)
        return 100 * (pis[5] - pis[1]), det, pis
    d0, det0, pis0 = rogan_gladen(I0, I09); dbs, _, _ = rogan_gladen(S, S9)
    pu, bu = drift(y9, m9, r9 == 5, r9 == 1, S9, I09)

    # second exploratory estimator: two-phase (double-sampling) on one joint resample of the 900 that carries the human labels
    pos9 = {k: i for i, k in enumerate(keys9)}
    has9 = np.zeros(n9, dtype=bool); hill9 = np.zeros(n9, dtype=bool)
    for i, h in enumerate(HIDS):
        has9[pos9[key[h]["a7b_key"]]] = True; hill9[pos9[key[h]["a7b_key"]]] = bool(h_ill[i])

    def two_phase(I9):
        pis, det = {}, {}
        for r in ROUNDS:
            c = r9 == r
            p = gshare(y9, m9, c, I9) / 100
            ppv = gshare(hill9, m9 & has9 & y9, c, I9) / 100          # human ill-posed among sampled judge-ill-posed
            fom = gshare(hill9, m9 & has9 & ~y9, c, I9) / 100         # 1 - NPV: human ill-posed among sampled judge-solvable
            with np.errstate(invalid="ignore"):
                pis[r] = np.where(p > 0, ppv * p, 0.0) + np.where(p < 1, fom * (1 - p), 0.0)
            det[r] = (p, ppv, fom)
        return 100 * (pis[5] - pis[1]), det, pis
    t0, tdet0, tpis0 = two_phase(I09); tbs, _, _ = two_phase(S9)
    exploratory = {
        "label": ("EXPLORATORY (not registered as a test): misclassification-corrected Opus blind drift on the 900 A7b items, "
                  "two estimators (Rogan-Gladen as planned; two-phase added before the read)"),
        "method": ("Rogan-Gladen per round: pi_r = (p_obs_r + Sp_r - 1) / (Se_r + Sp_r - 1), clipped to [0, 1]; p_obs_r = blind-Opus "
                   "ill-posed share among the round-r items of the 900; Se_r / Sp_r = blind-Opus sensitivity / specificity for "
                   "ill-posed against the final human label on the 80 round-r items of the 240; D = 100 (pi_5 - pi_1). CI: each "
                   "replicate pairs one stratified resample of the 900 (random.Random(20260948)) with one stratified resample of "
                   "the 240 (random.Random(20260947)); a replicate with Se + Sp - 1 <= 0 in a round is dropped. The 240 are a "
                   "subset of the 900; the two resamples are treated as independent (approximation)."),
        "per_round": {f"r{r}": {"n_900": int((m9 & (r9 == r)).sum()), "p_obs": float(det0[r][0][0]),
                                "n_human_illposed": int((mj & h_ill & (rnd == r)).sum()), "sensitivity": float(det0[r][1][0]),
                                "n_human_solvable": int((mj & ~h_ill & (rnd == r)).sum()), "specificity": float(det0[r][2][0]),
                                "pi_raw": float(det0[r][3][0]), "pi_clipped": float(pis0[r][0])} for r in ROUNDS},
        "D_uncorrected_900": diff_block(pu, bu, (m9 & (r9 == 5)).sum(), (m9 & (r9 == 1)).sum(), False),
        "D_corrected_900": diff_block(float(d0[0]), dbs, (m9 & (r9 == 5)).sum(), (m9 & (r9 == 1)).sum(), False),
        "two_phase_method": ("two-phase (double-sampling) estimator per round: pi_r = PPV_r * p_obs_r + (1 - NPV_r) * (1 - p_obs_r); "
                             "PPV_r = human ill-posed share among the sampled round-r items that blind Opus calls ill-posed, "
                             "1 - NPV_r = human ill-posed share among the sampled round-r items that blind Opus calls solvable "
                             "(the 240 were sampled 26-27 per 100 in every seed x round cell); D = 100 (pi_5 - pi_1). CI: one "
                             "stratified resample of the 900 per replicate (random.Random(20260948)), carrying the human labels of "
                             "the sampled items; a replicate with an undefined PPV or NPV in a round is dropped."),
        "per_round_two_phase": {f"r{r}": {"n_900": int((m9 & (r9 == r)).sum()), "p_obs": float(tdet0[r][0][0]),
                                          "n_sampled_judge_illposed": int((m9 & has9 & y9 & (r9 == r)).sum()),
                                          "ppv": float(tdet0[r][1][0]),
                                          "n_sampled_judge_solvable": int((m9 & has9 & ~y9 & (r9 == r)).sum()),
                                          "one_minus_npv": float(tdet0[r][2][0]), "pi": float(tpis0[r][0])} for r in ROUNDS},
        "D_two_phase_900": diff_block(float(t0[0]), tbs, (m9 & (r9 == 5)).sum(), (m9 & (r9 == 1)).sum(), False)}

    # ---- 6. descriptive
    def counts_r(r):
        c = {"n": int((rnd == r).sum()), "S": int(((fc == 0) & (rnd == r)).sum()), "U_or_A": int((h_ill & (rnd == r)).sum())}
        if three_ok:
            c.update(U=int(((fc == 1) & (rnd == r)).sum()), A=int(((fc == 2) & (rnd == r)).sum()))
        return c
    descriptive = {"note": "annotator notes are free text and are not coded; counts of final labels per round only",
                   "final_label_counts_by_round": {f"r{r}": counts_r(r) for r in ROUNDS},
                   "u_and_a_counts": "reported" if three_ok else NA3["status"],
                   "items_per_seed_round": {f"s{s}_r{r}": int(((sd == s) & (rnd == r)).sum()) for s in SEEDS for r in ROUNDS}}

    integrity = {"a7b_900": q900, "open_judge_file": qow,
                 "key_opus_original_differs_from_key_full": sum(key[h]["opus_original"] != K[key[h]["a7b_key"]]["orig"] for h in HIDS),
                 "key_opus_blind_differs_from_blind_verdicts": sum(key[h]["opus_blind"] != BO.get(key[h]["a7b_key"]) for h in HIDS),
                 "n_key_opus_original_invalid": sum(key[h]["opus_original"] is None for h in HIDS),
                 "n_key_opus_blind_invalid": sum(key[h]["opus_blind"] is None for h in HIDS),
                 "n_packet_text_differs_from_key_full_problem": n_text,
                 "open_judge_join": "direct: verdicts.jsonl 'key' == key.json 'a7b_key' (both the A7b neutral item id)"}
    res = {"analysis": "H1 human read of the self-play drift (registration H1_registration_2026-09-26.md)",
           "synthetic": bool(a.synthetic), "script_sha256": sha256(os.path.abspath(__file__)),
           "inputs": {k: {"path": paths[k], "sha256": sha[k]} for k in paths},
           "registered_sha256_check": "skipped (--synthetic dev run)" if a.synthetic else "passed",
           "conventions": {"illposed": "U or A", "units": "shares and differences in percentage points; kappa and agreement as fractions",
                           "labels": "strip().upper(); SOLVABLE/UNSOLVABLE/AMBIGUOUS -> S/U/A (annotator, adjudicated and judge labels)",
                           "bootstrap_seed": SEED, "bootstrap_replicates": B, "wilson_z": Z,
                           "percentile_positions": "int(0.025 nv), int(0.975 nv) - 1 of the sorted finite replicates"},
           "quality": quality, "final_label": final_block, "primary": primary,
           "secondary_human_trend": secondary_trend, "secondary_judge_validity": secondary_judges,
           "robustness": robustness, "exploratory": exploratory, "descriptive": descriptive, "integrity": integrity}
    return clean(res)


def f(x, nd=1, sign=False):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}")


def fci(ci, nd=1):
    return "[n/a]" if ci is None else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def summary(res):
    q = res["quality"]; g3, g2 = q["agreement"]["three_class"], q["agreement"]["binary_illposed_vs_solvable"]
    print(f"H1 human read{'  (SYNTHETIC dev run)' if res['synthetic'] else ''}")
    print(f" quality: rows ann1 {q['ann1']['n_data_rows']}, ann2 {q['ann2']['n_data_rows']}; agreement 3-class {f(g3['raw_agreement'], 3)}"
          f" kappa {f(g3['kappa'], 3)} {fci(g3['kappa_ci95'], 3)}; binary {f(g2['raw_agreement'], 3)} kappa {f(g2['kappa'], 3)}"
          f" {fci(g2['kappa_ci95'], 3)}; {res['final_label']['n_from_adjudication']} disagreements adjudicated"
          + (f"; {res['final_label']['n_u_vs_a_without_adjudication_binary_illposed']} U-vs-A without adjudication (binary only)"
             if not res['final_label']['three_class_final_label_available'] else ""))
    sh = res["primary"]["illposed_share_by_round"]
    print(" human ill-posed share: " + "; ".join(f"{r} {f(v['share_pct'])}% {fci(v['wilson95_pct'])} n={v['n']}" for r, v in sh.items()))
    d = res["primary"]["D_h_r5_minus_r1"]
    print(f" PRIMARY D_h (r5 - r1) = {f(d['d_pts'], 1, True)} pts {fci(d['ci95_pts'])} -> {d['reading']}")
    st = res["secondary_human_trend"]; lt = st["logistic_trend_over_round"]
    print(f" r3 - r1 {f(st['D_h_r3_minus_r1']['d_pts'], 1, True)} {fci(st['D_h_r3_minus_r1']['ci95_pts'])}; per seed "
          + ", ".join(f"{s} {f(v['d_pts'], 1, True)}" for s, v in st["D_h_r5_minus_r1_per_seed"].items())
          + f"; logistic b1/round {f(lt.get('b1_per_round'), 3)} {fci(lt.get('wald_ci95_b1'), 3)}"
          + ("" if lt.get("status") == "estimated" else f" ({lt.get('status', '')[:40]})"))
    for name, j in res["secondary_judge_validity"]["judges"].items():
        fr = j["false_positive_rate_by_round"]; dd = j["drift_on_same_items"]
        print(f" {name} (n={j['n_joined']}): kappa 3-class {f(j['agreement_three_class'].get('kappa'), 3)}, binary "
              f"{f(j['agreement_binary']['kappa'], 3)} {fci(j['agreement_binary']['kappa_ci95'], 3)}, acc {f(j['agreement_binary']['raw_agreement'], 3)};"
              f" F1 {f(j['illposed_detection_vs_humans']['all']['f1'], 3)}; FPR r1/r3/r5 {f(fr['r1']['share_pct'])}/{f(fr['r3']['share_pct'])}/"
              f"{f(fr['r5']['share_pct'])}; D judge {f(dd['judge_D']['d_pts'], 1, True)} vs D_h {f(dd['human_D_h_same_items']['d_pts'], 1, True)}")
    rb = res["robustness"]
    print(" robustness D: " + "; ".join(f"{k} {f(v['D_r5_minus_r1']['d_pts'], 1, True)} {fci(v['D_r5_minus_r1']['ci95_pts'])}"
                                        for k, v in rb.items() if isinstance(v, dict) and "D_r5_minus_r1" in v))
    ex = res["exploratory"]
    print(f" EXPLORATORY corrected Opus-blind D on 900: Rogan-Gladen {f(ex['D_corrected_900']['d_pts'], 1, True)} "
          f"{fci(ex['D_corrected_900']['ci95_pts'])}; two-phase {f(ex['D_two_phase_900']['d_pts'], 1, True)} "
          f"{fci(ex['D_two_phase_900']['ci95_pts'])} (uncorrected {f(ex['D_uncorrected_900']['d_pts'], 1, True)})")
    print(" counts per round: " + ", ".join(f"{r} " + (f"U{v['U']} A{v['A']}" if "U" in v else f"U-or-A {v['U_or_A']}")
                                         for r, v in res["descriptive"]["final_label_counts_by_round"].items()))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hv-dir", default=HV)
    p.add_argument("--key", default=f"{HV}/_key_do_not_open/key.json")
    p.add_argument("--a7b-key-full", default=f"{A5}/results/a7b_blind/key_full.jsonl")
    p.add_argument("--a7b-blind", default=f"{A5}/results/a7b_blind/blind_verdicts.json")
    p.add_argument("--open-judge", default=f"{A5}/results/a10/verdicts.jsonl")
    p.add_argument("--out", default=f"{HV}/human_read.json")
    p.add_argument("--synthetic", action="store_true", help="dev tests only: skip the registered sha256 check")
    a = p.parse_args()
    if os.path.exists(a.out):
        print(f"REFUSED: {a.out} exists; human_read.json is written once and never overwritten"); return 3
    try:
        res = run(a)
    except NotRun as e:
        print("NOT RUN (nothing written):")
        for line in e.args[0]:
            print("  -", line)
        return 2
    txt = json.dumps(res, indent=1, allow_nan=False) + "\n"
    try:
        with open(a.out, "x", encoding="utf-8") as fh:
            fh.write(txt)
    except FileExistsError:
        print(f"REFUSED: {a.out} appeared during the run; not overwritten"); return 3
    summary(res)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
