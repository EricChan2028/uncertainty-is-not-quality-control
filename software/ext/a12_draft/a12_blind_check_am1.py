"""[A12 amendment 1, passes 2-3 only: judges' own temp subfolders named a12* are exempt from the banned-name rule;
see A12_amendment1_2026-09-26.md. Otherwise identical to the registered a12_blind_check.py.]
A12 part-1 blindness check (A12_registration_2026-09-26.md, section 1.4): the registered A7-5d checker a75d_blind_check.py
with only the items file (plus a 2-batch pilot file), the key letter (y; q for the pilot), four banned names, the raw-sample
leak fields, a framing-sentence abort and a record of accepted structured returns changed (derived by dev_mk_check.py).
Reads the local transcripts of one pass of
a12_p1_blind_wf.js (its workflow transcript directory, plus every directory of its resume chain if it was interrupted) and
writes one verdict per judge batch.
Every workflow-subagent in the directories must be a judge "judge:<a>-<b>" with exactly one computed-task line
("[Workflow harness — computed task] ...") containing the registered fetch command for exactly that range (the port is read
from it, must be <HOST> or <HOST> and must be the same for all judges of the pass); else abort. Relayed user requests
("[Workflow harness — user request] ...") are recorded verbatim. Harness-inserted "<synthetic>" turns are not judge turns.
A transcript that made no structured return and no failing call is neutral (an interrupted attempt) and is ignored.
Every other transcript passes only if
 (1) it made a FETCH - the registered command for its own range, or a sed re-read of a sub-range of its own lines in the
     same form - whose tool result is not an error and contains all 25 of its keys ("key": "yNNNN"; "qNNNN" in the pilot);
 (2) every other call is allowed: StructuredOutput, TodoWrite; Write / Edit / Read of a file under a temp directory; a Bash /
     PowerShell command that names no transcript, task-output, key, audit or item file of this project, does not kill
     processes (taskkill, pkill, killall, kill, Stop-Process), and - outside temp-directory paths - uses no network tool,
     no server address and no user / home directory (no "..", "~", bare "cd"); local computation in a temp directory is
     allowed, as in the original audit;
 (3) no result of a non-fetch call contains key or audit information (rzero, rz<S>i<i>r<n> ids, orig_verdict,
     audit_admitted, key_full, "iteration": <n>, "run": "...", a12s<S>i<i>k<k> raw ids, "status": "admitted", "raw_id",
     "pool_score", "admitted":, scored_rejected, not_scored);
 (4) every assistant turn ran as claude-opus-5-5 with effort xhigh (the setting of all original R-Zero and A7b judges).
A batch passes if at least one non-neutral transcript exists for it and all its non-neutral transcripts pass; batches with
no non-neutral transcript are "absent". passed + dropped + absent = the expected batches. For each transcript the sha256 of
the sorted (key, verdict) pairs of each structured return is recorded, so the read can verify the saved verdict file.
The task text must also contain the registered framing sentence (FRAMING), else abort. For every structured return the
checker records whether the harness accepted it (no error result), so that the read can tell a null return (every return
rejected) from a batch that is missing from the saved verdict file.
    python3 a12_blind_check.py <out.json> all|pilot|<ranges json> <transcript dir> [<transcript dir> ...]
"pilot" checks the pre-registration pilot on dummy items (dirA5/results/a12_pilot/items.jsonl, keys q0000..q0049, lines
1-50); the read never accepts a pilot check."""
import glob, hashlib, json, os, re, sys

OUTF, EXP, DIRS = sys.argv[1], sys.argv[2], sys.argv[3:]
assert DIRS and all(os.path.isdir(d) for d in DIRS), "transcript directories missing"
PILOT = EXP == "pilot"
FILE = "/home/tione/notebook/rsi_rl/dirA5/results/" + ("a12_pilot" if PILOT else "a12_blind") + "/items.jsonl"
KL = "q" if PILOT else "y"  # key letter (A7-5d: z; the pilot's dummy items: q, so they never count as A12 keys)
FRAMING = "These were written by a Qwen3-4B-Base language model in a self-play loop."  # registration section 1.3
EXPECTED = ([[1, 25], [26, 50]] if PILOT else [[a, a + 24] for a in range(1, 1201, 25)] if EXP == "all"
            else [list(x) for x in json.loads(EXP)])
PROMPT_FETCH = re.compile(r"(ssh -o BatchMode=yes -o ConnectTimeout=20 -p (\d+) root@42\.192\.34\.154) \"sed -n '(\d+),(\d+)p' "
                          + re.escape(FILE) + "\"")
TEMP = re.compile(r"(?:(?:[a-z]:|/[a-z])?[/\\]+users[/\\]+rog[/\\]+appdata[/\\]+local[/\\]+temp|~[/\\]+appdata[/\\]+local[/\\]+temp"
                  r"|\$env:(?:temp|tmp)\b|\$\{?(?:tmpdir|temp|tmp)\b\}?|/tmp\b)(?:[/\\]+[^\s'\"|;&<>()`]*)?", re.I)
BAD_ANY = re.compile(r"\.claude|[/\\]subagents[/\\]|journal\.jsonl|[/\\]tasks[/\\]|audit_parts|rsi_audit_tmp|[/\\_]a75d|[/\\]a7b"
                     r"|key_full|rzero|[/\\_]a12|generated_question|all_scored|\.json\.raw|[/\\]dirb[/\\]|\btaskkill\b|\bpkill\b|\bkillall\b|stop-process", re.I)
BAD = re.compile(r"(?:^|[;&|(`\n]|\$\()\s*(?:ssh|scp|sftp|rsync|curl|wget)\b|42\.192\.34\.154|/home/tione|\btione\b|rsi_rl|appdata"
                 r"|(?:^|[\s'\"=:(<>])(?:[a-z]:|/[a-z])?[/\\]+users\b|(?:^|[\s'\"=:(<>])/(?:root|home)\b"
                 r"|(?:^|[\s'\"=(])~(?=$|[\s/\\'\"])|\$\{?home\b|\$\{?userprofile\b|%userprofile%|\$env:(?:userprofile|home)"
                 r"|(?:^|[;&|(\n])\s*cd\s*(?=$|[;&|)\n])|(?:^|[\s'\"=:(<>])\.\.(?=$|[/\\\s'\";&|)])|(?:^|[;&|(\n])\s*kill\b", re.I)
PORTS = {<HOST>, <HOST>}  # both hosts mount the same NFS volume, so the fetched bytes are identical
LEAK = re.compile(r"rzero|rz[01]i[1-3]r\d|orig_verdict|audit_admitted|key_full|\"iteration\"\s*:\s*\d|\"run\"\s*:\s*\""
                  r"|a12s[01]i[1-3]k\d|\"status\"\s*:\s*\"admitted\"|\"raw_id\"|\"pool_score\"|\"admitted\"\s*:|scored_rejected|not_scored", re.I)

JUDGE_DIR = re.compile(r"((?:scratchpad|temp|tmp)[/\\]+)a12(?!_(?:key|blind|open|runs|draft|pilot)(?![\w.-]))[\w.-]*", re.I)  # amendment 1


def am1_rename(s):  # amendment 1: a judge's own temp subfolder named a12* is not a banned name
    return JUDGE_DIR.sub(lambda m: m.group(1) + "judgetmp", s)


def temp_ok(p):
    return bool(TEMP.fullmatch(p) or TEMP.fullmatch(p.replace("\\", "/"))) and ".." not in p and not BAD_ANY.search(am1_rename(p))


def allowed(name, inp):
    if name in ("StructuredOutput", "TodoWrite"):
        return True
    if name in ("Write", "Edit", "Read"):
        return temp_ok(str(inp.get("file_path", "")))
    if name in ("Bash", "PowerShell"):
        cmd = str(inp.get("command", ""))
        if BAD_ANY.search(am1_rename(cmd)) or any(".." in m.group(0) for m in TEMP.finditer(cmd)):
            return False
        return not BAD.search(TEMP.sub(" TMPPATH ", cmd))
    return False


def result_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c)


def user_text(d):  # text of a user line that is not a tool result (task text or a relayed user request)
    c = (d.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list) and not any(isinstance(x, dict) and x.get("type") == "tool_result" for x in c):
        return result_text(c)
    return None


per, ports, n_comp_sessions, n_judged = {}, set(), 0, 0
for D in DIRS:
    for meta in sorted(glob.glob(os.path.join(D, "agent-*.meta.json"))):
        m = json.load(open(meta, encoding="utf-8"))
        if m.get("agentType") != "workflow-subagent":
            continue
        g = re.fullmatch(r"judge:(\d+)-(\d+)", m.get("description", ""))
        assert g, f"unexpected subagent in the judge run: {meta} {m.get('description')!r}"
        a, b = int(g.group(1)), int(g.group(2))
        jf = meta[:-len(".meta.json")] + ".jsonl"
        assert os.path.exists(jf), f"meta without transcript: {meta}"
        lines = [json.loads(l) for l in open(jf, encoding="utf-8") if l.strip()]
        strs = [s for s in (user_text(d) for d in lines if d.get("type") == "user") if s is not None]
        tasks = [s for s in strs if s.lstrip().startswith("[Workflow harness — computed task]")]
        relays = [s for s in strs if s.lstrip().startswith("[Workflow harness — user request]")]
        assert len(tasks) == 1, f"{jf}: expected exactly one computed-task line, found {len(tasks)}"
        task = tasks[0]
        pf = [mm for mm in PROMPT_FETCH.finditer(task)]
        assert len(pf) == 1 and (int(pf[0].group(3)), int(pf[0].group(4))) == (a, b), f"task text of {jf} lacks the registered fetch for {a}-{b}"
        assert FRAMING in task, f"task text of {jf} lacks the registered framing sentence"
        ssh, port = pf[0].group(1), int(pf[0].group(2)); ports.add(port)
        assert port in PORTS, f"{jf}: port {port} not in the registered set {sorted(PORTS)}"
        so_shas, so_ids = [], []
        fetch_re = re.compile(re.escape(ssh) + r" \"sed -n '(\d+)(?:,(\d+))?p' " + re.escape(FILE) + "\"")
        want = [f'"key": "{KL}{ln - 1:04d}"' for ln in range(a, b + 1)]
        calls, bad, fetch_ids, other_ids, results, models, efforts, versions, noncomp = 0, [], set(), set(), {}, set(), set(), set(), []
        returned, ncomp = False, 0
        for d in lines:
            if d.get("version"):
                versions.add(d["version"])
            msg = d.get("message") or {}
            if d.get("type") == "assistant" and str(msg.get("model")) != "<synthetic>":  # harness-inserted turns are not the judge
                models.add(str(msg.get("model"))); efforts.add(f"{d.get('effort')}/{d.get('perTurnEffort')}")
            c = msg.get("content")
            for blk in c if isinstance(c, list) else []:
                t = str(blk.get("type", ""))
                if d.get("type") == "assistant" and t.endswith("tool_use"):
                    calls += 1
                    name, inp = blk.get("name"), blk.get("input") or {}
                    fm = fetch_re.fullmatch(str(inp.get("command", ""))) if name == "Bash" else None
                    if fm and a <= int(fm.group(1)) <= int(fm.group(2) or fm.group(1)) <= b:
                        fetch_ids.add(blk.get("id")); continue
                    if name == "StructuredOutput":
                        returned = True
                        pairs = sorted([str(v.get("key")), str(v.get("verdict"))] for v in (inp.get("verdicts") or []) if isinstance(v, dict))
                        so_shas.append(hashlib.sha256(json.dumps(pairs).encode()).hexdigest()); so_ids.append(blk.get("id")); continue
                    other_ids.add(blk.get("id"))
                    rec = {"tool": name, "input": json.dumps(inp, ensure_ascii=False)[:200]}
                    noncomp.append(rec)
                    if not allowed(name, inp):
                        bad.append(rec)
                    elif name in ("Bash", "PowerShell"):
                        ncomp += 1
                elif t == "tool_result":
                    results[blk.get("tool_use_id")] = (bool(blk.get("is_error")), result_text(blk.get("content")))
        leaks = [i for i in other_ids if i in results and LEAK.search(results[i][1])]
        good_fetch = any(i in results and not results[i][0] and all(w in results[i][1] for w in want) for i in fetch_ids)
        setting_ok = models == {"claude-opus-5-5"} and efforts == {"xhigh/xhigh"}
        neutral = not returned and not bad and not leaks
        ok = good_fetch and not bad and not leaks and setting_ok
        e = per.setdefault(f"{a}-{b}", {"lines": [a, b], "transcripts": []})
        e["transcripts"].append({"file": os.path.relpath(jf, os.path.dirname(os.path.dirname(jf))), "neutral": neutral, "ok": ok,
                                 "returned": returned, "tool_calls": calls, "full_fetch": good_fetch, "bad_calls": bad,
                                 "n_leaking_results": len(leaks), "models": sorted(models), "efforts": sorted(efforts),
                                 "versions": sorted(versions), "task_sha256": hashlib.sha256(task.encode("utf-8")).hexdigest(),
                                 "relayed_user_requests": relays, "structured_output_pair_sha256": so_shas,
                                 "structured_output_accepted": [not (i in results and results[i][0]) for i in so_ids],
                                 "non_fetch_calls": noncomp})
        if not neutral:
            n_judged += 1; n_comp_sessions += ncomp > 0
assert len(ports) <= 1, f"judges of one pass used different ports: {ports}"
status = {}
for k, e in per.items():
    nn = [t for t in e["transcripts"] if not t["neutral"]]
    e["status"] = "absent" if not nn else ("passed" if all(t["ok"] for t in nn) else "dropped")
    status[tuple(e["lines"])] = e["status"]
unexpected = sorted(list(x) for x in status if list(x) not in EXPECTED)
assert not unexpected, f"batches outside the expected ranges: {unexpected}"
out = {"checker": "a12_blind_check.py", "amendment": "A12-am1 (a12_blind_check_am1.py)", "items_file": FILE, "pilot": PILOT, "transcript_dirs": [os.path.basename(os.path.normpath(d)) for d in DIRS], "port": next(iter(ports), None),
       "n_expected": len(EXPECTED), "n_batches_seen": len(per), "unexpected": [],
       "passed": sorted(x for x in EXPECTED if status.get(tuple(x)) == "passed"),
       "dropped": sorted(x for x in EXPECTED if status.get(tuple(x)) == "dropped"),
       "absent": sorted(x for x in EXPECTED if status.get(tuple(x), "absent") == "absent"),
       "n_judge_sessions": n_judged, "n_sessions_with_local_computation": n_comp_sessions,
       "per_batch": dict(sorted(per.items(), key=lambda kv: kv[1]["lines"][0]))}
with open(OUTF, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("items_file", "pilot", "transcript_dirs", "port", "n_expected", "n_batches_seen", "dropped", "absent",
                                      "n_judge_sessions", "n_sessions_with_local_computation")}))
