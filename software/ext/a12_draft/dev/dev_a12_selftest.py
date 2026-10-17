"""A12 provenance: CPU-only self-test of the A12 scripts on SYNTHETIC data under /tmp/a12test (nothing under the project tree
is written; no GPU; vLLM is replaced by a fake module). Each script is copied with its absolute roots rewritten
(/home/tione/notebook/rsi_rl -> /tmp/a12test/R, /run/ti/rzero -> /tmp/a12test/RZ); pinned input hashes are rewritten to the
synthetic files' hashes. Checks: part-1 stage/build (alignment, filter reproduction, refusal to redraw, wrong-host refusal,
text fallback, line escapes, batch sizes); the pilot build (keys q0000..q0049, no y-key) and the checker's pilot mode; the
blindness checker on synthetic transcripts whose task text is rendered from a12_p1_blind_wf.js (clean, banned call, leak,
local computation, a null structured return, wrong model, a task text without the framing sentence); the part-1 read
(items/key correspondence, pass 1 incomplete -> ranges incl. the null-return batch -> complete after pass 2; a passed batch
missing from the saved verdicts aborts; a changed or missing a75d_read.json aborts; pilot check refused; odds identity; Delta
log E CI, the secondary band state and the registered wording of every reading state, which names downstream selection); the
R-Zero-prompt wrapper, the plain runner (every SamplingParams stops at <|im_end|>, k = 1, and without it an invented next turn
would be scored) and the open judge through a fake vLLM; the part-2 read (model-path and b_eval checks, the exhaustive primary x
secondary mapping, the FORMAT-DEPENDENT wording gated by the format-contrast CI, the plain-truncation flag); the part-3 read
(pinned inputs, band numbers reproduce a3_guard_diag_score.py, every Corollary state and its wording).
    /usr/bin/python3 dev_a12_selftest.py <dir with the A12 scripts>"""
import ast, glob, hashlib, itertools, json, math, os, random, re, shutil, socket, subprocess, sys

SRC = os.path.abspath(sys.argv[1]); T = "/tmp/a12test"; RR = f"{T}/R"; RZZ = f"{T}/RZ"; REAL = "/home/tione/notebook/rsi_rl"
assert T.startswith("/tmp/")
shutil.rmtree(T, ignore_errors=True); os.makedirs(f"{T}/bin")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
HOST = socket.gethostname()


def cp(name, extra=()):
    s = open(f"{SRC}/{name}", encoding="utf-8").read().replace(REAL, RR).replace("/run/ti/rzero", RZZ)
    for a, b in extra:
        assert s.count(a) >= 1, (name, a); s = s.replace(a, b)
    open(f"{T}/bin/{name}", "w", encoding="utf-8").write(s)
    return f"{T}/bin/{name}"


def run(args, ok=True, env=None):
    p = subprocess.run(args, capture_output=True, text=True, env={**os.environ, **(env or {})})
    if ok and p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:]); raise SystemExit(f"FAILED: {args}")
    return p


def wj(p, x):
    os.makedirs(os.path.dirname(p), exist_ok=True); json.dump(x, open(p, "w", encoding="utf-8"), ensure_ascii=False)


def wl(p, xs):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def defs(path, names):
    """Top-level functions / assignments of a script, without running the script."""
    ns = {"math": math, "re": re}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        tgt = node.name if isinstance(node, ast.FunctionDef) else (node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if tgt in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    assert all(n in ns for n in names), names
    return ns


rule = lambda x: x["score"] >= 0.3 and x["score"] <= 0.8 and x["answer"] != '' and x["answer"] != 'None'
# ---------------------------------------------------------------- part 1 fixtures (synthetic R-Zero storage)
truth = {}
for S in (0, 1):
    os.makedirs(f"{RR}/ext/rzero_runs/s{S}"); open(f"{RR}/ext/rzero_runs/s{S}.host", "w").write(f"{HOST} {RZZ}/s{S} 0\n")
    gq = f"{RZZ}/s{S}/generated_question"; os.makedirs(gq)
    for i in (1, 2, 3):
        name = f"rzero_qwen3_4b_Base_s{S}_solver_v{i}"; rng = random.Random(100 * S + i); pool = []; nparsed = 0
        for k in range(8):
            rows = []
            for j in range(1000):
                if rng.random() < 0.08:
                    rows.append({"question": "a response without a question block", "answer": "", "score": -1}); continue
                q = f"Synthetic question {S}-{i}-{k}-{j}: add {rng.randint(1, 99)} and {rng.randint(1, 99)}."
                u = rng.random()
                if u < 0.01:
                    q = "A repeated synthetic question."
                elif u < 0.03:
                    q += "\n第二行 (second line)"
                elif u < 0.032:
                    q += " separator tail\u0085end"
                ill = rng.random() < 0.25 + 0.08 * (i - 1)
                truth[q] = ill
                rows.append({"question": q, "answer": str(rng.randint(1, 9)), "score": 0}); nparsed += 1
            json.dump(rows, open(f"{gq}/{name}_{k}.json.raw", "w"))
            for r in rows:
                if r["score"] != 0 or rng.random() < 0.07:
                    continue
                c = rng.randint(1, 9); a = rng.choice(["None", ""]) if rng.random() < 0.02 else str(rng.randint(1, 9))
                pool.append({"question": r["question"], "answer": a, "score": c / 9, "results": [a]})
        json.dump(pool, open(f"{gq}/{name}_all_scored.json", "w"))
        adm = [{"problem": x["question"], "answer": x["answer"], "score": x["score"]} for x in pool if rule(x)]
        wl(f"{gq}/{name}_admitted.jsonl", adm)
        wj(f"{RR}/ext/rzero_runs/s{S}/{name}_summary.json", {"n_raw_shards": 8, "n_generated": 8000, "n_parsed": nparsed,
                                                              "n_scored": len(pool), "n_admitted": len(adm),
                                                              "sha256_admitted_jsonl": sha(f"{gq}/{name}_admitted.jsonl")})
        # the registered A7-5 audit sample of this run (rz_extract.py) and a synthetic audit with the registered drifts
        idx = random.Random(20260924 + i).sample(range(len(adm)), 200); run_d = f"{RR}/dirB/runs/rzero_qwen3_4b_Base_s{S}_it{i}"
        wl(f"{run_d}/admitted.jsonl", [{"id": f"rz{S}i{i}r{j}", "problem": adm[j]["problem"]} for j in idx])
        n_ill = {(0, 1): 40, (0, 2): 60, (0, 3): 86, (1, 1): 50, (1, 2): 70, (1, 3): 104}[(S, i)]
        wj(f"{run_d}/audit_admitted.json", {"verdicts": [{"k": r, "id": f"rz{S}i{i}r{j}", "verdict": "UNSOLVABLE" if r < n_ill else "SOLVABLE"}
                                                         for r, j in enumerate(idx)]})
wj(f"{RR}/dirA5/results/a75d_blind/a75d_read.json", {"cells": {f"s{s}_it{i}": {"n": 200, "orig": 30.0, "blind": 29.0 + i} for s in (0, 1) for i in (1, 2, 3)},
                                                    "D_blind": 24.0, "per_seed": {"s0": {"D_blind": 22.0}, "s1": {"D_blind": 26.0}}})
samp = cp("a12_p1_sample.py")
run(["/usr/bin/python3", samp, "stage", "--seed", "0"]); run(["/usr/bin/python3", samp, "stage", "--seed", "1"])
assert run(["/usr/bin/python3", samp, "stage", "--seed", "0"], ok=False).returncode != 0, "a redraw must be refused"
meta0 = json.load(open(f"{RR}/dirA5/results/a12_key/stage_s0_meta.json"))
assert all(c["status_method"] == "ordered" for c in meta0["cells"].values()), meta0
# status must agree with the fixture: admitted iff the aligned pool row passes the filter (checked via counts over all parsed)
for it in (1, 2, 3):
    c = meta0["cells"][f"it{it}"]; assert c["status_counts_all_parsed"]["admitted"] == c["n_admitted"], c
    assert sum(c["status_counts_all_parsed"].values()) == c["n_parsed"]
    assert c["status_counts_all_parsed"]["admitted"] + c["status_counts_all_parsed"]["scored_rejected"] == c["n_scored"]
print("part1 stage ok", {k: v["sample_status_counts"] for k, v in meta0["cells"].items()})
run(["/usr/bin/python3", samp, "build"])
assert run(["/usr/bin/python3", samp, "build"], ok=False).returncode != 0, "a rebuild must be refused"
lines = open(f"{RR}/dirA5/results/a12_blind/items.jsonl", "rb").read().decode("utf-8").split("\n")
assert lines[-1] == "" and len(lines) == 1201 and all(re.fullmatch(r'\{"key": "y\d{4}", "problem": ".*"\}', l) for l in lines[:-1])
assert not any(c in "".join(lines) for c in "  \u0085\r")
K = [json.loads(l) for l in open(f"{RR}/dirA5/results/a12_key/key_full.jsonl", encoding="utf-8")]
assert [json.loads(l)["problem"] for l in lines[:-1]] == [x["problem"] for x in K] and [x["key"] for x in K] == [f"y{k:04d}" for k in range(1200)]
print("part1 build ok; escaped separators in", sum("\\u2028" in l or "\\u0085" in l for l in lines), "lines")
# wrong host and text fallback
bad = f"{T}/bad"; shutil.copytree(RR, f"{bad}/R"); shutil.copytree(RZZ, f"{bad}/RZ")
open(f"{bad}/R/ext/rzero_runs/s0.host", "w").write("some-other-host x 0\n")
for f in glob.glob(f"{bad}/R/dirA5/results/a12_key/stage_s*"):
    os.remove(f)
s2 = open(samp).read().replace(RR, f"{bad}/R").replace(RZZ, f"{bad}/RZ"); open(f"{T}/bin/samp_bad.py", "w").write(s2)
assert "belongs to host" in run(["/usr/bin/python3", f"{T}/bin/samp_bad.py", "stage", "--seed", "0"], ok=False).stderr
gp = f"{bad}/RZ/s1/generated_question/rzero_qwen3_4b_Base_s1_solver_v2_all_scored.json"; P = json.load(open(gp)); P.reverse()
json.dump(P, open(gp, "w"))
run(["/usr/bin/python3", f"{T}/bin/samp_bad.py", "stage", "--seed", "1"])
mb = json.load(open(f"{bad}/R/dirA5/results/a12_key/stage_s1_meta.json"))
assert mb["cells"]["it2"]["status_method"].startswith("text") and mb["cells"]["it1"]["status_method"] == "ordered", mb["cells"]["it2"]["status_method"]
print("part1 refusals and text fallback ok")

# ---------------------------------------------------------------- pilot build: q keys only
PILOT_DIR = f"{RR}/dirA5/results/a12_pilot"; pil = cp("a12_p1_pilot_build.py"); run(["/usr/bin/python3", pil, PILOT_DIR])
PILOT_FILE = f"{PILOT_DIR}/items.jsonl"; pl = open(PILOT_FILE, encoding="utf-8").read()
assert [json.loads(l)["key"] for l in pl.splitlines()] == [f"q{k:04d}" for k in range(50)] and not re.search(r"\by[01][0-9]{3}\b", pl)
print("pilot build ok: 50 dummy items, keys q0000..q0049, no y-key")

# ---------------------------------------------------------------- blindness check on synthetic transcripts
FILE_REG = f"{RR}/dirA5/results/a12_blind/items.jsonl"
chk = cp("a12_blind_check.py")
WF = open(f"{SRC}/a12_p1_blind_wf.js", encoding="utf-8").read()
TPL = re.search(r"agent\(`(.*?)`, \{ label", WF, re.S).group(1)
FRAMING = "These were written by a Qwen3-4B-Base language model in a self-play loop."
assert FRAMING in TPL
key2 = {x["key"]: x for x in K}
lab_rng = random.Random(5)


def render_task(a, b, file, port=<HOST>, framing=True):
    ssh = f"ssh -o BatchMode=yes -o ConnectTimeout=20 -p {port} root@<SERVER>"
    t = TPL.replace("${b - a + 1}", str(b - a + 1)).replace("${SSH}", ssh).replace("${a}", str(a)).replace("${b}", str(b)).replace("${FILE}", file)
    assert "${" not in t
    if not framing:
        t = t.replace(FRAMING, "These were written by a Qwen3-4B-Base language model and ADMITTED by a self-play filter.")
    return ("[Workflow harness — computed task] The task text below was computed at runtime by a workflow script. "
            "The computed task text follows:\n" + "\n".join("  " + l for l in t.split("\n"))), f"{ssh} \"sed -n '{a},{b}p' {file}\""


def verdict_for(key):
    x = key2[key]; p = (0.35 if truth.get(x["problem"], False) else 0.1) + (0.15 if x["status"] == "admitted" else 0)
    return "UNSOLVABLE" if lab_rng.random() < p else ("AMBIGUOUS" if lab_rng.random() < 0.05 else "SOLVABLE")


def transcript(d, n, a, b, file, extra=None, fetch_ok=True, model="claude-opus-5-5", kl="y", so_error=False, framing=True, body_sub=None):
    os.makedirs(d, exist_ok=True)
    wj(f"{d}/agent-{n}.meta.json", {"agentType": "workflow-subagent", "description": f"judge:{a}-{b}"})
    task, fetch = render_task(a, b, file, framing=framing)
    body = "\n".join(open(file, encoding="utf-8").read().split("\n")[a - 1:b]) if fetch_ok else "error"
    body = body_sub(body) if body_sub else body
    vs = [{"key": f"{kl}{ln - 1:04d}", "verdict": verdict_for(f"{kl}{ln - 1:04d}") if file == FILE_REG else "SOLVABLE", "confidence": 0.8, "note": "n"}
          for ln in range(a, b + 1)]
    L = [{"type": "user", "version": "2.1.290", "message": {"role": "user", "content": task}},
         {"type": "assistant", "effort": "xhigh", "perTurnEffort": "xhigh", "message": {"model": model, "content": [
             {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": fetch}}]}},
         {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": body, "is_error": not fetch_ok}]}}]
    if extra:
        cmd, res = extra
        L += [{"type": "assistant", "effort": "xhigh", "perTurnEffort": "xhigh", "message": {"model": model, "content": [
                  {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": cmd}}]}},
              {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "content": res, "is_error": False}]}}]
    L.append({"type": "assistant", "effort": "xhigh", "perTurnEffort": "xhigh", "message": {"model": model, "content": [
        {"type": "tool_use", "id": "t3", "name": "StructuredOutput", "input": {"verdicts": vs[:-1] if so_error else vs}}]}})
    if so_error:  # the harness rejected the return (schema), and the judge gave up: the workflow's result for this batch is null
        L.append({"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t3", "content": "schema validation failed", "is_error": True}]}})
    wl(f"{d}/agent-{n}.jsonl", L)
    return [] if so_error else [dict(v, lines=[a, b]) for v in vs]


D1 = f"{T}/transcripts/wf_pass1"; V1 = []
for n, a in enumerate(range(1, 1201, 25)):
    extra = None
    if a == 26:
        extra = (f"cat {REAL}/dirA5/results/a12_key/key_full.jsonl", "whatever")          # banned call -> dropped
    elif a == 51:
        extra = ("python3 -c \"print(1)\"", '{"raw_id": "' + "a12" + 's0i1k3j5"}')             # leak -> dropped (id split so this file never matches the pre-launch scan)
    elif a == 76:
        extra = ("python3 -c \"print(sum([3, 4]))\"", "7")                                  # allowed local computation
    V1 += transcript(D1, n, a, a + 24, FILE_REG, extra, so_error=(a == 101))                # 101-125: null structured return
out1 = f"{RR}/dirA5/results/a12_blind/blindness_check.json"
r = run(["/usr/bin/python3", chk, out1, "all", D1]); c1 = json.load(open(out1))
assert c1["dropped"] == [[26, 50], [51, 75]] and len(c1["passed"]) == 46 and c1["n_sessions_with_local_computation"] == 2, r.stdout  # the leaking session also computed
assert c1["items_file"] == FILE_REG and c1["pilot"] is False
assert c1["per_batch"]["101-125"]["transcripts"][0]["structured_output_accepted"] == [False]
assert c1["per_batch"]["126-150"]["transcripts"][0]["structured_output_accepted"] == [True]
json.dump({"pass": 1, "pilot": False, "verdicts": V1}, open(f"{RR}/dirA5/results/a12_blind/blind_verdicts.json", "w"))
# a task text without the registered framing sentence aborts the check
Dn = f"{T}/transcripts/wf_noframe"; transcript(Dn, 0, 1, 25, FILE_REG, framing=False)
assert "lacks the registered framing sentence" in run(["/usr/bin/python3", chk, f"{T}/nf.json", "[[1, 25]]", Dn], ok=False).stderr
# pilot mode: q keys, wrong model -> dropped
Dp = f"{T}/transcripts/wf_pilot"
transcript(Dp, 0, 1, 25, PILOT_FILE, kl="q"); transcript(Dp, 1, 26, 50, PILOT_FILE, kl="q", model="claude-sonnet-x")
cp_ = json.loads(run(["/usr/bin/python3", chk, f"{T}/pilot_check.json", "pilot", Dp]).stdout)
assert cp_["pilot"] is True and cp_["dropped"] == [[26, 50]] and cp_["items_file"] == PILOT_FILE, cp_
Dp2 = f"{T}/transcripts/wf_pilot_ykeys"; transcript(Dp2, 0, 1, 25, PILOT_FILE, kl="y", body_sub=lambda t: t.replace('"key": "q', '"key": "y'))
assert json.loads(run(["/usr/bin/python3", chk, f"{T}/pilot_check2.json", "pilot", Dp2]).stdout)["dropped"] == [[1, 25]], "pilot must want q keys"
print("checker ok (clean / banned / leak / local computation / null return / no framing sentence / pilot q keys / wrong model)")

# ---------------------------------------------------------------- part-1 read: pass 1 incomplete -> pass 2 -> complete
AS = {(s, i): sha(f"{RR}/dirB/runs/rzero_qwen3_4b_Base_s{s}_it{i}/audit_admitted.json") for s in (0, 1) for i in (1, 2, 3)}
real_audit = {(0, 1): "81ad37b312184dcb348ea99b2d2a070e62be129d47a9cbf9a22a3835975cbb68", (0, 2): "4cdc864798717369d741361c95135cd56dfd619d05f89bf784a7357eb6f7a29a",
              (0, 3): "957f294e40d0a3b9df3d44c5669a9ebe94c3a5c134551006d9275af02060962d", (1, 1): "301597a2607f4d5eab128529e8100084f3aaff19c16a28d05611e9dcc2c108b0",
              (1, 2): "685a9a1b54a4f798f7fd6385734adf164f3925e5d547bd31bc5b1d5f493d5669", (1, 3): "6700e71f2a9ca8f42e0b76438e80b719aad1e9768982860cd4d010f94a7d5a32"}
rd = cp("a12_p1_read.py", [(real_audit[k], AS[k]) for k in AS] + [("7c9bd8bd17d285c7b810450f2561d0c5566ceeb74e95d1e8bd7413e5deb87d7b",
                                                                   sha(f"{RR}/dirA5/results/a75d_blind/a75d_read.json"))])
BVF = f"{RR}/dirA5/results/a12_blind/blind_verdicts.json"; good = open(BVF).read()
# a passed batch whose judge returned verdicts but which is missing from the saved file aborts the read
json.dump({"pass": 1, "pilot": False, "verdicts": [v for v in V1 if v["lines"] != [126, 150]]}, open(BVF, "w"))
assert "passed the check and its judge returned verdicts" in run(["/usr/bin/python3", rd], ok=False).stderr
open(BVF, "w").write(good)
# items.jsonl must decode to key_full's pairs in order
ITF = FILE_REG; it_good = open(ITF, encoding="utf-8").read(); ls_ = it_good.split("\n"); ls_[0], ls_[1] = ls_[1], ls_[0]
open(ITF, "w", encoding="utf-8").write("\n".join(ls_))
assert "does not decode to key_full" in run(["/usr/bin/python3", rd], ok=False).stderr
open(ITF, "w", encoding="utf-8").write(it_good); assert sha(ITF) == hashlib.sha256(it_good.encode("utf-8")).hexdigest()
run(["/usr/bin/python3", rd]); p1 = json.load(open(f"{RR}/dirA5/results/a12_blind/a12_p1_read_pass1.json"))
assert p1["complete"] is False and p1["next_pass_ranges"] == [[26, 50], [51, 75], [101, 125]] and "primary" not in p1, p1["next_pass_ranges"]
assert p1["passes"][0]["passed_null_return"] == [[101, 125]], p1["passes"][0]["passed_null_return"]
D2 = f"{T}/transcripts/wf_pass2"; V2 = []
for n, a in enumerate((26, 51, 101)):
    V2 += transcript(D2, n, a, a + 24, FILE_REG)
run(["/usr/bin/python3", chk, f"{RR}/dirA5/results/a12_blind/blindness_check_rejudge.json", "[[26, 50], [51, 75], [101, 125]]", D2])
json.dump({"pass": 2, "pilot": False, "verdicts": V2}, open(f"{RR}/dirA5/results/a12_blind/blind_verdicts_rejudge.json", "w"))
run(["/usr/bin/python3", rd]); o = json.load(open(f"{RR}/dirA5/results/a12_blind/a12_p1_read.json"))
assert o["complete"] and o["n_blind"] == 1200 and o["descriptive"]["admitted_original"]["matches_registered_RZ_P1"], o.get("descriptive", {}).get("admitted_original")
ws = o["descriptive"]["within_sample"]; dec = ws["by_iteration"]["it1"]
lg = lambda p: math.log(p / (100 - p))
assert abs(lg(dec["s_adm_in"]) - (lg(dec["s_raw"]) + math.log(dec["E"]))) < 1e-9, "odds identity"
lo_ = ws["logodds"]
assert abs(lo_["delta_log_E"] - (math.log(ws["by_iteration"]["it3"]["E"]) - math.log(dec["E"]))) < 1e-9, "Delta log E identity"
fs = o["selection_share"]; assert fs["ci95"] is not None and fs["state"] in ("POSITIVE", "NEGATIVE", "NOT ESTABLISHED")
assert ("sits in downstream selection" in o["reading"]["text"]) == (fs["state"] == "POSITIVE" and o["reading"]["label"] in ("FLAT", "FALLS"))
assert "sits in the filter" not in o["reading"]["text"] and o["reading"]["selection_share"] == fs["state"]
bsec = o["band_share_secondary"]; bi = ws["by_iteration"]
assert bsec["state"] == o["reading"]["band_share_secondary"] and bsec["ci95"] == lo_["delta_log_E_scored_ci95"] and "SECONDARY" in bsec["note"]
assert abs(bsec["delta_log_E_scored"] - (math.log(bi["it3"]["E_scored"]) - math.log(bi["it1"]["E_scored"]))) < 1e-9, "Delta log E(scored)"
cc = o["descriptive"]["condition_check"]; assert cc["primary_reference"] == "A7-5d blind" and cc["flag"] in (True, False) and cc["pooled_ci95"]["pooled_diff_vs_blind"]
cov = o["descriptive"]["covariates"]["s0_it1"]; mc = meta0["cells"]["it1"]
assert abs(cov["parse_failure_rate"] - 100 * (8000 - mc["n_parsed"]) / 8000) < 1e-9 and "not_scored_rate_all_parsed" in cov
print("part1 read ok:", o["reading"], "D_raw", round(o["primary"]["D_raw_pooled"], 2), o["primary"]["ci95"], "dlogE", fs,
      "overlap n", o["descriptive"]["overlap_with_audited_admitted"]["n"], "cond flag", cc["flag"], cc["pooled_ci95"])
# registered wording of every reading state
F1 = defs(rd, ["share_state", "reading_text", "DS"])
assert F1["share_state"]([0.1, 0.5]) == "POSITIVE" and F1["share_state"]([-0.5, -0.1]) == "NEGATIVE"
assert F1["share_state"]([-0.1, 0.4]) == "NOT ESTABLISHED" and F1["share_state"](None) == "NOT ESTABLISHED"
cis = {"RISES": [2.0, 15.0], "FALLS": [-15.0, -2.0], "FLAT": [-4.0, 6.5], "INCONCLUSIVE": [-3.0, 12.0]}
for lab, st in itertools.product(cis, ("POSITIVE", "NOT ESTABLISHED", "NEGATIVE")):
    t = F1["reading_text"](lab, st, cis[lab], sum(cis[lab]) / 2, 25.0)
    assert ("the drift sits in downstream selection (pseudo-labelling and the 0.3-0.8 filter)" in t) == (st == "POSITIVE" and lab in ("FLAT", "FALLS")), (lab, st, t)
    assert ("both stages" in t) == (st == "POSITIVE" and lab == "RISES"), (lab, st, t)
    assert not any(w in t for w in ("sits in the filter", "the filter adds", "the filter's", "band-enrichment")), (lab, st, t)
    assert ("(pseudo-labelling and the 0.3-0.8 filter)" in t) == (lab == "RISES" or (lab in ("FLAT", "FALLS") and st == "POSITIVE")), (lab, st, t)
    if lab == "RISES":
        assert "consistent with, not proof of, the uncertainty reward" in t and "does not separate its cause" in t
    if lab == "FLAT":
        assert t.startswith("no raw drift detected; raw drift within the CI") and "< 6.5 points, i.e. < 26% of the admitted drift" in t, t
    if lab in ("FLAT", "FALLS") and st != "POSITIVE":
        assert ("raw flat; share of downstream selection not established" if lab == "FLAT" else "raw falls; share of downstream selection not established") in t
print("part1 reading wording ok (every label x downstream-selection state)")
# the A7-5d blind read is pinned like the six audits: a changed or a missing file aborts the read
A75F = f"{RR}/dirA5/results/a75d_blind/a75d_read.json"; a75_good = open(A75F, encoding="utf-8").read()
a75_bad = a75_good.replace('"D_blind": 24.0', '"D_blind": 24.5', 1); assert a75_bad != a75_good
open(A75F, "w", encoding="utf-8").write(a75_bad)
assert "is not the registered A7-5d blind read" in run(["/usr/bin/python3", rd], ok=False).stderr
os.remove(A75F); assert "is not the registered A7-5d blind read" in run(["/usr/bin/python3", rd], ok=False).stderr
open(A75F, "w", encoding="utf-8").write(a75_good); run(["/usr/bin/python3", rd])
print("part1 read aborts on a changed or missing a75d_read.json ok")
# a pilot check file is never accepted
shutil.copy(f"{T}/pilot_check.json", f"{RR}/dirA5/results/a12_blind/blindness_check_rejudge.json")
assert "not an a12_blind_check.py check of the registered items file" in run(["/usr/bin/python3", rd], ok=False).stderr
print("part1 read refuses a pilot check ok")

# ---------------------------------------------------------------- fake vLLM (for the wrapper, the plain runner and the judge)
fk = f"{T}/fake/vllm"; os.makedirs(fk)
open(f"{fk}/__init__.py", "w").write(r'''
import json, os, re
class SamplingParams:
    def __init__(self, **kw):
        self.kw = kw; self.__dict__.update(kw)
        if os.environ.get("FAKE_LOG"):
            open(os.environ["FAKE_LOG"], "a").write(json.dumps({k: v for k, v in kw.items() if k in ("temperature", "n", "stop_token_ids", "stop")}) + "\n")
class _C:
    def __init__(self, text, ids, fin, stop): self.text, self.token_ids, self.finish_reason, self.stop_reason = text, ids, fin, stop
class _O:
    def __init__(self, cs): self.outputs = cs
class LLM:
    def __init__(self, model, **kw):
        self.model = model; self.kw = kw; self._tok = None
    def get_tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(os.environ["FAKE_TOK"])
        return self._tok
    def generate(self, prompts, sp):
        t = self.get_tokenizer(); out = []; im = t.convert_tokens_to_ids("<|im_end|>")
        mode = os.environ["FAKE_MODE"]
        gold = json.load(open(os.environ["FAKE_GOLD"])) if mode in ("rz", "plain") else {}
        for i, p in enumerate(prompts):
            if mode == "rz":
                g = next((v for q, v in gold.items() if q in p), "0")
                ans = g if i % 3 else "999"
                txt = f"Reasoning.\nSo the answer is \\boxed{{{ans}}}."
                ids = t.encode(txt) + [im]
                out.append(_O([_C(txt, ids, "stop", im)]))
            elif mode == "plain":   # a chat-trained solver: ends its turn with <|im_end|>, then invents a next turn
                g = next((v for q, v in gold.items() if q in p), "0")
                ans = g if i % 3 else "999"; txt = f"Reasoning.\nSo the answer is \\boxed{{{ans}}}."
                cs = []
                for _ in range(sp.kw.get("n", 1)):
                    if im in (sp.kw.get("stop_token_ids") or []):
                        cs.append(_C(txt, t.encode(txt) + [im], "stop", im))
                    else:
                        tx = txt + "\nuser\nAnother question. The answer is \\boxed{999}."
                        cs.append(_C(tx, t.encode(tx), "stop", None))
                out.append(_O(cs))
            else:
                h = sum(map(ord, p)) % 7
                v = ["SOLVABLE", "UNSOLVABLE", "AMBIGUOUS", "SOLVABLE", "SOLVABLE", "UNSOLVABLE", None][h]
                if sp.kw.get("temperature", 0) > 0: v = "SOLVABLE"
                txt = "Short note." + (f"\nVERDICT: {v}" if v else "\nno verdict")
                out.append(_O([_C(txt, t.encode(txt), "stop", None)]))
        return out
''')
# ---------------------------------------------------------------- part-2 wrapper and plain runner through the fake vLLM
for sub in ("code", "dirB/code", "dirB/data", "dirB/results/evals"):
    os.makedirs(f"{RR}/{sub}", exist_ok=True)
for f in ("code/loto_common.py", "dirB/code/pdc_util.py"):
    shutil.copy(f"{REAL}/{f}", f"{RR}/{f}")
open(f"{RR}/dirB/code/b_eval.py", "w").write(open(f"{REAL}/dirB/code/b_eval.py").read().replace(REAL, RR))  # roots rewritten
BE_REAL = "5bf7c94ac1821c473e1826e9a2623b19af92673742dc169d12e28a0478f1f189"; assert sha(f"{REAL}/dirB/code/b_eval.py") == BE_REAL
BE_T = sha(f"{RR}/dirB/code/b_eval.py")
SUF = " Let's think step by step and output the final answer after \"####\"."
gold = {}
for name, kind in (("math500_train", "math"), ("math_heldout1000", "math"), ("gsm8k_test500", "gsm")):
    rows = []
    for j in range(6):
        a, b = 3 + j, 10 + 2 * j; q = f"What is {a} plus {b}?" + (" Answer with a number." if kind == "math" else "")
        gold[q] = str(a + b)
        rows.append({"id": f"{name}:{j}", "problem": q + (SUF if kind == "gsm" else ""), "prompt": q, "answer": str(a + b)})
    wl(f"{RR}/dirB/data/{name}.jsonl", rows)
json.dump(gold, open(f"{T}/gold.json", "w"))
wr = cp("a12_rzchat_eval.py", [(BE_REAL, BE_T)])
envf = {"PYTHONPATH": f"{T}/fake", "FAKE_TOK": "/root/models/Qwen3-4B-Base", "FAKE_MODE": "rz", "FAKE_GOLD": f"{T}/gold.json", "CUDA_VISIBLE_DEVICES": ""}
run(["/usr/bin/python3", wr, "--model", f"{T}/fake_model_Base", "--tag", "a12rz_test"], env=envf)
sm = json.load(open(f"{RR}/dirB/results/evals/a12rz_test.summary.json"))
for name in ("math500_train", "math_heldout1000", "gsm8k_test500"):
    assert abs(sm[name]["greedy_acc"] - 4 / 6) < 1e-9 and abs(sm[name]["greedy_acc_rzero_style"] - 4 / 6) < 1e-9 and sm[name]["im_end_rate"] == 1, sm[name]
assert run(["/usr/bin/python3", wr, "--model", "x", "--tag", "bad"], ok=False, env=envf).returncode != 0
wr_bad = cp("a12_rzchat_eval.py", [(BE_REAL, BE_T), ("901558113e95d58225ebe13abb44f37cbb58af9e97935a1145c30f9393f5ed11", "0" * 64)])
assert "is not the registered version" in run(["/usr/bin/python3", wr_bad, "--model", f"{T}/fake_model_Base", "--tag", "a12rz_x"], ok=False, env=envf).stderr
print("part2 wrapper ok (b_eval scorer and R-Zero scorer on fake outputs: 4/6 each; a changed loto_common.py is refused)")
pe = cp("a12_plain_eval.py", [(BE_REAL, BE_T)]); FL = f"{T}/sp_log.jsonl"
SETS3 = "math500_train,math_heldout1000,gsm8k_test500"
run(["/usr/bin/python3", pe, "--model", f"{T}/fake_model_Base", "--tag", "a12_plain_test", "--data", SETS3, "--k", "1"],
    env={**envf, "FAKE_MODE": "plain", "FAKE_LOG": FL})
sp_calls = [json.loads(l) for l in open(FL)]; IMID = 151645
assert len(sp_calls) == 6 and all(IMID in c["stop_token_ids"] and c["n"] == 1 for c in sp_calls), sp_calls
assert [c["temperature"] for c in sp_calls] == [0.0, 1.0] * 3 and all(c["stop"] == ["\nProblem:", "\nQuestion:", "\n\n\n\n"] for c in sp_calls)
sp_ = json.load(open(f"{RR}/dirB/results/evals/a12_plain_test.summary.json")); side = json.load(open(f"{RR}/dirB/results/evals/a12_plain_test.plain_stops.json"))
for name in SETS3.split(","):
    assert abs(sp_[name]["greedy_acc"] - 4 / 6) < 1e-9 and sp_[name]["k"] == 1 and side[name] == {"n": 6, "n_im_end_stop": 6, "n_length": 0, "im_end_id": IMID, "prompts_match": True}, (sp_[name], side)
# the same b_eval without the runner scores the invented next turn's box: the confound the stop removes
run(["/usr/bin/python3", f"{RR}/dirB/code/b_eval.py", "--model", f"{T}/fake_model_Base", "--tag", "direct_test", "--data", SETS3, "--k", "1"],
    env={**envf, "FAKE_MODE": "plain"})
sd = json.load(open(f"{RR}/dirB/results/evals/direct_test.summary.json"))
assert all(sd[n]["greedy_acc"] == 0 for n in SETS3.split(",")), sd
for bad_args in (["--k", "8"], ["--k", "1", "--chat", "1"]):
    assert run(["/usr/bin/python3", pe, "--model", f"{T}/fake_model_Base", "--tag", "a12_plain_x", "--data", SETS3] + bad_args, ok=False,
               env={**envf, "FAKE_MODE": "plain"}).returncode != 0
print("part2 plain runner ok (stop at <|im_end|> in every SamplingParams, k = 1, 4/6 vs 0/6 without the stop; other options refused)")

# ---------------------------------------------------------------- part-2 read on synthetic summaries
rng = random.Random(9); SETS = {"math500_train": 500, "math_heldout1000": 1000, "gsm8k_test500": 500}
PATH = {"qwen3_4b_base": "/root/models/Qwen3-4B-Base",
        **{f"rzero_s{s}_it{i}": f"{RZZ}/s{s}/models/rzero_qwen3_4b_Base_s{s}_solver_v{i}/global_step_15/actor/huggingface" for s in (0, 1) for i in (1, 2, 3)}}
for pre in ("a12_plain_", "a12rz_"):
    for m in PATH:
        summ = {}
        for name, n in SETS.items():
            pbase = 0.65 + (0.08 if pre == "a12rz_" and m != "qwen3_4b_base" else 0.0) + (0.02 * int(m[-1]) if m != "qwen3_4b_base" else 0)
            ok = [int(rng.random() < pbase) for _ in range(n)]
            if pre == "a12_plain_":
                rows = [{"id": f"{name}:{j}", "greedy_ok": ok[j], "greedy_ok_lenient": ok[j],
                         "greedy_trunc": int(j % 50 == 0 or (m == "rzero_s1_it3" and j % 10 == 0)),
                         "greedy_text": ("\\boxed{1} then \\boxed{2}" if j % 20 == 0 else "\\boxed{{3}}")} for j in range(n)]
            else:
                rows = [{"id": f"{name}:{j}", "greedy_ok": ok[j], "ok_rzero_style": ok[j]} for j in range(n)]
            wl(f"{RR}/dirB/results/evals/{pre}{m}.{name}.jsonl", rows)
            summ[name] = {"n": n, "greedy_acc": sum(ok) / n, "greedy_acc_lenient": sum(ok) / n, "greedy_acc_rzero_style": sum(ok) / n, "mean_len": 1,
                          "model": PATH[m], **({"b_eval_sha256": BE_REAL, "rendered_prompt_0_sha256": "c" * 64, "trunc_rate": 0.01}
                                               if pre == "a12rz_" else {"k": 1})}
        wj(f"{RR}/dirB/results/evals/{pre}{m}.summary.json", summ)
shutil.copy(f"{REAL}/dirB/results/evals/qwen3_4b_base_step0.summary.json", f"{RR}/dirB/results/evals/")
p2 = cp("a12_p2_read.py"); run(["/usr/bin/python3", p2]); o2 = json.load(open(f"{RR}/ext/a12_runs/a12_p2_read.json"))
assert not o2["not_run"] and "rzchat:it3:math500_train" in o2["labels"], o2["not_run"]
F2 = defs(p2, ["reading", "seed_label", "READING_TEXT", "TRUNC_MARGIN", "reading_text"])
assert o2["reading"]["label"] == F2["reading"](o2["reading"]["primary_labels"], o2["reading"]["secondary_labels"])
rr = o2["reading"]; trp = rr["plain_minus_rzchat_truncation_points_it3_math500"]
assert rr["plain_truncates_more"] is True and abs(trp["rzero_s1_it3"] - 9.0) < 1e-9 and abs(trp["qwen3_4b_base"] - 1.0) < 1e-9, trp
assert rr["format_contrast_it3_math500"]["s0"] == o2["format_contrast"]["rzero_s0_it3:math500_train"]
assert rr["format_contrast_ci_lower_gt0_both_seeds"] == all(o2["format_contrast"][f"rzero_s{s}_it3:math500_train"]["ci95"][0] > 0 for s in (0, 1))
assert rr["text"] == F2["reading_text"](rr["label"], rr["format_contrast_ci_lower_gt0_both_seeds"], rr["plain_truncates_more"])
FDT = "the gain is not shown in both seeds under the plain Base prompt (format contrast and plain truncation reported)"
for lab_, fcp, trm in itertools.product(F2["READING_TEXT"], (True, False, None), (True, False, None)):
    t = F2["reading_text"](lab_, fcp, trm); fd = lab_ == "REPRODUCES, FORMAT-DEPENDENT"
    assert ("depends on the format" in t) == (fd and fcp is True) and ("not called format adoption" in t) == (fd and trm is True), (lab_, fcp, trm, t)
    assert "depends on that format" not in t and t.startswith(F2["READING_TEXT"][lab_]) and ((FDT in t) == fd), (lab_, t)
assert o2["base_reproduction"]["stored_summary_registered"] and "plain_diagnostics" in o2 and o2["delta_vs_base_rzero_style"]
pdg = o2["plain_diagnostics"]["qwen3_4b_base:math500_train"]; assert abs(pdg["first_last_boxed_differ_rate"] - 0.05) < 1e-9 and abs(pdg["greedy_trunc_rate"] - 0.02) < 1e-9, pdg
# the registered table (section 2.4), written out independently: rows = primary seed labels, columns = number of secondary GAINs
ROW = {("GAIN", "GAIN"): ("REPRODUCES", "REPRODUCES, FORMAT-DEPENDENT", "REPRODUCES, FORMAT-DEPENDENT"),
       ("GAIN", "NO_GAIN"): ("REPRODUCES IN ONE SEED",) * 3, ("GAIN", "LOSS"): ("REPRODUCES IN ONE SEED",) * 3,
       ("NO_GAIN", "NO_GAIN"): ("PLAIN-ONLY", "PLAIN-ONLY", "NOT REPRODUCED"),
       ("NO_GAIN", "LOSS"): ("PLAIN-ONLY", "PLAIN-ONLY", "NOT REPRODUCED, LOSS"), ("LOSS", "LOSS"): ("PLAIN-ONLY", "PLAIN-ONLY", "NOT REPRODUCED, LOSS")}
LABS = ("GAIN", "NO_GAIN", "LOSS"); seen = set()
for P_ in itertools.product(LABS, repeat=2):
    for S_ in itertools.product(LABS, repeat=2):
        row = ROW[tuple(sorted(P_, key=LABS.index))]; want = row[{2: 0, 1: 1, 0: 2}[S_.count("GAIN")]]
        got = F2["reading"](list(P_), list(S_)); assert got == want, (P_, S_, got, want); seen.add(got)
    for S_ in ([None, "GAIN"], ["GAIN", None]):
        assert F2["reading"](list(P_), S_) == "NOT RUN"
assert F2["reading"]([None, "GAIN"], ["GAIN", "GAIN"]) == "NOT RUN" and F2["reading"](None, None) == "NOT RUN"
assert seen | {"NOT RUN"} == set(F2["READING_TEXT"]), seen
assert F2["seed_label"]([0.1, 3]) == "GAIN" and F2["seed_label"]([-3, -0.1]) == "LOSS" and F2["seed_label"]([-1, 1]) == "NO_GAIN"
# a summary that names another model aborts the read
sf = f"{RR}/dirB/results/evals/a12rz_rzero_s0_it3.summary.json"; sg = open(sf).read(); x = json.loads(sg); x["gsm8k_test500"]["model"] = PATH["rzero_s1_it3"]
json.dump(x, open(sf, "w")); assert "was run on" in run(["/usr/bin/python3", p2], ok=False).stderr; open(sf, "w").write(sg)
print("part2 read ok:", o2["reading"]["label"], {k: v["per_seed"] for k, v in o2["labels"].items() if "math500" in k and "it3" in k},
      "; mapping exhaustive over 81 label pairs + NOT RUN; wrong model refused")

# ---------------------------------------------------------------- part 3: fixture, judge through the fake vLLM, read
rng = random.Random(13); ids = [f"prop1:{j}" for j in range(1158)]
PROPF = f"{RR}/dirB/runs/qwen3_4b_Base_own_a3_s0_r1/proposed.jsonl"
wl(PROPF, [{"id": k, "problem": f"Proposal {k}: some text {rng.random()}", "prompt": "", "stem": ""} for k in ids])
lab = {k: rng.choices(["SOLVABLE", "AMBIGUOUS", "UNSOLVABLE"], [0.55, 0.16, 0.29])[0] for k in ids}
for a in range(0, 1158, 25):
    wj(f"{RR}/dirA3/results/pool_audit/qwen_s0_r1/parts/{a}-{a + 25}.json",
       {"verdicts": [{"k": j, "id": ids[j], "verdict": lab[ids[j]], "confidence": 0.9, "note": ""} for j in range(a, min(a + 25, 1158))]})
CKS = ["base", "r1", "r2", "r3", "r4", "r5"]
for t, ck in enumerate(CKS):
    recs = [{"id": k, "kind": "proposed", "p_mode": 0.5, "mode": "1", "abstain": 0.0,
             "admit": rng.random() < ((0.55 - 0.07 * t) if lab[k] == "SOLVABLE" else 0.45)} for k in ids]
    recs += [{"id": f"trd:{j}", "kind": "tracer_deleted" if j % 2 else "tracer_intact", "p_mode": 0.4, "mode": "1", "abstain": 0.0,
              "admit": rng.random() < 0.5} for j in range(400)]
    wl(f"{RR}/dirA3/results/guard_diag/qwen_s0/{ck}/guard_records.jsonl", recs)
gd = f"{T}/bin/a3_guard_diag_score.py"
open(gd, "w").write(open(f"{REAL}/dirA3/code/a3_guard_diag_score.py").read().replace(REAL, RR))
run(["/usr/bin/python3", gd])
jd = cp("a12_open_judge.py")
run(["/usr/bin/python3", jd], env={**envf, "FAKE_MODE": "judge"})
assert run(["/usr/bin/python3", jd], ok=False, env={**envf, "FAKE_MODE": "judge"}).returncode != 0, "the judge must refuse to overwrite"
V = [json.loads(l) for l in open(f"{RR}/dirA5/results/a12_open/verdicts.jsonl")]
assert len(V) == 1158 and all(v["verdict"] in ("SOLVABLE", "UNSOLVABLE", "AMBIGUOUS") for v in V) and any(v["retry"] for v in V)
GDR = f"{RR}/dirA3/results/guard_diag/qwen_s0"
REAL_PIN = {PROPF: "a97bf54ebae71447ae5d686222083745202f8516f090068d4386d8176b15acb9",
            f"{GDR}/result.json": "a250ce8824a7fcaa86871d77b773d05a85996335de71546ea5648d48dd36c7ad",
            f"{GDR}/base/guard_records.jsonl": "d7271ee75aa62713519f53f9344aa5a715c5f56fb08d64a45dc47dde99f781cb",
            f"{GDR}/r1/guard_records.jsonl": "b9aa0601ec2755f943afd57e12558143d10735cb66a85b065eff23adc3cc78ac",
            f"{GDR}/r2/guard_records.jsonl": "5da61cd028448aa6195be14b9755a5f9335f05647ae7ea28535ce6004d17cd03",
            f"{GDR}/r3/guard_records.jsonl": "9ccddc3394df8074d90739e66c21b40dd0161684accddf319bf8c579c4269443",
            f"{GDR}/r4/guard_records.jsonl": "ca5dc3d652a5fa559d25fb87dfa29ec28b4545cf8a99711f295e2182f0842228",
            f"{GDR}/r5/guard_records.jsonl": "6677ec70cfeb37680d39cfeffcf57fe9d47f611f0b8d74523644b929e60ce4bb"}
pf = sorted(glob.glob(f"{RR}/dirA3/results/pool_audit/qwen_s0_r1/parts/*.json")); assert len(pf) == 47
dig = hashlib.sha256("".join(f"{os.path.basename(f)} {sha(f)}\n" for f in pf).encode()).hexdigest()
pins = [(h, sha(p)) for p, h in REAL_PIN.items()] + [("d4718c01ee11de3cbeb624772ec507a003bdc67cb20788c265fc2aaf2aa194f7", dig)]
p3 = cp("a12_p3_read.py", pins); run(["/usr/bin/python3", p3]); o3 = json.load(open(f"{RR}/dirA5/results/a12_open/a12_p3_read.json"))
assert o3["inputs"]["band_alone_reproduces_result_json"] and all(v["odds_identity_check"] for v in o3["per_checkpoint"].values())
rd3 = o3["readings"]; assert rd3["corollary"] in ("CONSISTENT", "WEAKENS", "AMPLIFIES", "NOT DETERMINED") and rd3["direction_text"]
e = o3["per_checkpoint"]; orr = (e["r5"]["eps_over_k"] / e["base"]["eps_over_k"])
assert abs(o3["drift"]["OR_ver_over_OR_band"] - orr) < 1e-9, "OR ratio = ratio of eps/k"
p3_bad = cp("a12_p3_read.py", pins[1:])  # the proposals file no longer matches its pin
assert "is not the registered input" in run(["/usr/bin/python3", p3_bad], ok=False).stderr
F3 = defs(p3, ["corollary", "DIRECTION_TEXT"])
assert F3["corollary"]([0.4, 0.9], False)[0] == "WEAKENS" and F3["corollary"]([1.1, 2.0], False)[0] == "AMPLIFIES"
assert F3["corollary"]([0.8, 1.3], False)[0] == "CONSISTENT" and F3["corollary"](None, False)[0] == "NOT DETERMINED"
assert "escape condition" in F3["corollary"]([0.8, 1.3], True)[1] and "not evidence against Corollary 1" in F3["DIRECTION_TEXT"]["NOT SHOWN"]
ct = F3["corollary"]([0.8, 1.3], False)[1]; assert "consistent with a constant eps/k" in ct and "constant from base to r5" not in ct, ct
assert "fewer than 40" in F3["corollary"](None, False)[1] and o3["drift"]["n_d_replicates"] >= 40 and o3["drift"]["n_or_replicates"] >= 40
assert not any("contradict" in t for t in F3["DIRECTION_TEXT"].values())
print("part3 ok:", rd3["level"], rd3["direction"], rd3["corollary"], "D_band", round(o3["drift"]["D_band"], 1), "D_ver", round(o3["drift"]["D_ver"], 1),
      "OR ratio", round(o3["drift"]["OR_ver_over_OR_band"], 3), o3["drift"]["OR_ver_over_OR_band_ci95"], "miss", round(o3["pool_level"]["miss_rate_on_opus_illposed"], 3),
      "; pins enforced; every Corollary state worded")
print("ALL A12 SELF-TESTS PASSED")
