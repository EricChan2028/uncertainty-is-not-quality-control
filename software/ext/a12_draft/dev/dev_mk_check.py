"""A12 provenance: derives a12_blind_check.py from the registered A7-5d checker (dirA5/code/a75d_blind_check.py, sha256
e42e8905c4d833283aa3bed6cd8224f74fcaa769229c3716b86b13a365babd2b). Every replacement must match exactly once; the logic of the
check is otherwise byte-identical. Changes: the items file (and a pilot file with 2 batches), the key letter (y; q for the
pilot's dummy items), four names added to the ban list, raw-sample fields added to the leak list, an abort when the task text
lacks the registered framing sentence, a per-return record of whether the harness accepted each structured return, and three
fields in the output.
    python3 dev_mk_check.py <a75d_blind_check.py> <out a12_blind_check.py>"""
import hashlib, sys

SRC, DST = sys.argv[1:3]
b = open(SRC, "rb").read()
assert hashlib.sha256(b).hexdigest() == "e42e8905c4d833283aa3bed6cd8224f74fcaa769229c3716b86b13a365babd2b", "not the registered A7-5d checker"
s = b.decode("utf-8")
REPL = [
    ('"""A7-5d blindness check (addendum dirA5/docs/A7_5d_addendum_2026-09-26.md). Reads the local transcripts of one pass of\n'
     'a75d_blind_wf.js',
     '"""A12 part-1 blindness check (A12_registration_2026-09-26.md, section 1.4): the registered A7-5d checker a75d_blind_check.py\n'
     'with only the items file (plus a 2-batch pilot file), the key letter (y; q for the pilot), four banned names, the raw-sample\n'
     'leak fields, a framing-sentence abort and a record of accepted structured returns changed (derived by dev_mk_check.py).\n'
     'Reads the local transcripts of one pass of\n'
     'a12_p1_blind_wf.js'),
    ('contains all 25 of its keys ("key": "zNNNN");', 'contains all 25 of its keys ("key": "yNNNN"; "qNNNN" in the pilot);'),
    ('audit_admitted, key_full, "iteration": <n>, "run": "...");',
     'audit_admitted, key_full, "iteration": <n>, "run": "...", a12s<S>i<i>k<k> raw ids, "status": "admitted", "raw_id",\n'
     '     "pool_score", "admitted":, scored_rejected, not_scored);'),
    ('    python3 a75d_blind_check.py <out.json> all|<ranges json> <transcript dir> [<transcript dir> ...]"""',
     'The task text must also contain the registered framing sentence (FRAMING), else abort. For every structured return the\n'
     'checker records whether the harness accepted it (no error result), so that the read can tell a null return (every return\n'
     'rejected) from a batch that is missing from the saved verdict file.\n'
     '    python3 a12_blind_check.py <out.json> all|pilot|<ranges json> <transcript dir> [<transcript dir> ...]\n'
     '"pilot" checks the pre-registration pilot on dummy items (dirA5/results/a12_pilot/items.jsonl, keys q0000..q0049, lines\n'
     '1-50); the read never accepts a pilot check."""'),
    ('FILE = "/home/tione/notebook/rsi_rl/dirA5/results/a75d_blind/items.jsonl"\n'
     'EXPECTED = [[a, a + 24] for a in range(1, 1201, 25)] if EXP == "all" else [list(x) for x in json.loads(EXP)]\n',
     'PILOT = EXP == "pilot"\n'
     'FILE = "/home/tione/notebook/rsi_rl/dirA5/results/" + ("a12_pilot" if PILOT else "a12_blind") + "/items.jsonl"\n'
     'KL = "q" if PILOT else "y"  # key letter (A7-5d: z; the pilot\'s dummy items: q, so they never count as A12 keys)\n'
     'FRAMING = "These were written by a Qwen3-4B-Base language model in a self-play loop."  # registration section 1.3\n'
     'EXPECTED = ([[1, 25], [26, 50]] if PILOT else [[a, a + 24] for a in range(1, 1201, 25)] if EXP == "all"\n'
     '            else [list(x) for x in json.loads(EXP)])\n'),
    ('r"|key_full|rzero|[/\\\\]dirb[/\\\\]|',
     'r"|key_full|rzero|[/\\\\_]a12|generated_question|all_scored|\\.json\\.raw|[/\\\\]dirb[/\\\\]|'),
    ('LEAK = re.compile(r"rzero|rz[01]i[1-3]r\\d|orig_verdict|audit_admitted|key_full|\\"iteration\\"\\s*:\\s*\\d|\\"run\\"\\s*:\\s*\\"", re.I)',
     'LEAK = re.compile(r"rzero|rz[01]i[1-3]r\\d|orig_verdict|audit_admitted|key_full|\\"iteration\\"\\s*:\\s*\\d|\\"run\\"\\s*:\\s*\\""\n'
     '                  r"|a12s[01]i[1-3]k\\d|\\"status\\"\\s*:\\s*\\"admitted\\"|\\"raw_id\\"|\\"pool_score\\"|\\"admitted\\"\\s*:|scored_rejected|not_scored", re.I)'),
    ('        ssh, port = pf[0].group(1), int(pf[0].group(2)); ports.add(port)\n',
     '        assert FRAMING in task, f"task text of {jf} lacks the registered framing sentence"\n'
     '        ssh, port = pf[0].group(1), int(pf[0].group(2)); ports.add(port)\n'),
    ('        so_shas = []\n', '        so_shas, so_ids = [], []\n'),
    ('so_shas.append(hashlib.sha256(json.dumps(pairs).encode()).hexdigest()); continue',
     'so_shas.append(hashlib.sha256(json.dumps(pairs).encode()).hexdigest()); so_ids.append(blk.get("id")); continue'),
    ('"structured_output_pair_sha256": so_shas,\n',
     '"structured_output_pair_sha256": so_shas,\n'
     '                                 "structured_output_accepted": [not (i in results and results[i][0]) for i in so_ids],\n'),
    ("want = [f'\"key\": \"z{ln - 1:04d}\"' for ln in range(a, b + 1)]",
     "want = [f'\"key\": \"{KL}{ln - 1:04d}\"' for ln in range(a, b + 1)]"),
    ('out = {"transcript_dirs": ', 'out = {"checker": "a12_blind_check.py", "items_file": FILE, "pilot": PILOT, "transcript_dirs": '),
    ('print(json.dumps({k: out[k] for k in ("transcript_dirs", "port",',
     'print(json.dumps({k: out[k] for k in ("items_file", "pilot", "transcript_dirs", "port",'),
]
for old, new in REPL:
    assert s.count(old) == 1, ("replacement must match exactly once", old[:80], s.count(old))
    s = s.replace(old, new, 1)
open(DST, "wb").write(s.encode("utf-8"))
print("wrote", DST, hashlib.sha256(s.encode("utf-8")).hexdigest())
