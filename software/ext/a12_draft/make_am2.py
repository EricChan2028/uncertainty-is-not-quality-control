"""Build a12_blind_check_am2.py from a12_blind_check_am1.py (amendment 1, sha 8621965a...) with one further change
(amendment 2, pass 3 only): a judge's OWN background tasks are treated as local computation. A task id is the judge's own
if the result of one of its own earlier tool calls in the same transcript says "Command running in background with ID: <id>"
or "... was moved to the background (ID: <id>)".
For such ids only:
  - TaskStop with that task_id is allowed;
  - a Read of <temp dir>/.../tasks/<id>.output, or a Bash / PowerShell / Monitor command that names that path, is judged
    with the path rewritten to <temp dir>/.../judgetmp/<id>.output (so every other rule still applies to the command);
  - ToolSearch with a query "select:" of only TaskStop and/or Monitor is allowed (it only loads those tool definitions).
Monitor commands are judged like Bash commands. All other rules, the leak patterns (applied to the results of every
non-fetch call, including these), the fetch and model checks and the framing check are unchanged."""
import hashlib, io

SRC = "a12_blind_check_am1.py"
DST = "a12_blind_check_am2.py"
src = io.open(SRC, encoding="utf-8").read()
assert hashlib.sha256(src.encode("utf-8")).hexdigest().startswith("8621965a"), "not the amendment-1 checker"

def rep(s, a, b):
    assert s.count(a) == 1, (a[:70], s.count(a))
    return s.replace(a, b)

am = src
am = rep(am, '"""[A12 amendment 1, passes 2-3 only:',
         '"""[A12 amendment 2, pass 3 only: a judge\'s own background tasks (ids announced to it by its own tool results) count as\n'
         'local computation: TaskStop, Monitor and reading tasks/<own id>.output; see A12_amendment2_2026-09-26.md.]\n[A12 amendment 1, passes 2-3 only:')
am = rep(am, "\n\ndef allowed(name, inp):\n",
         "\nBG_ID = re.compile(r\"(?:Command running in background with ID: |moved to the background \(ID: )([A-Za-z0-9]+)\")  # amendment 2\n"
         "OWN_SELECT = re.compile(r\"select:(?:TaskStop|Monitor)(?:,(?:TaskStop|Monitor))?\")  # amendment 2\n"
         "\n\ndef am2_rewrite(s, own):  # amendment 2: tasks/<own id>.output is the judge's own computation output\n"
         "    return re.sub(r\"([/\\\\])tasks([/\\\\]+)([A-Za-z0-9]+)(\\.output)\", lambda m: m.group(1) + \"judgetmp\" + m.group(2) + m.group(3) + m.group(4)\n"
         "                  if m.group(3) in own else m.group(0), s)\n"
         "\n\ndef allowed2(name, inp, own):  # amendment 2\n"
         "    if name == \"TaskStop\":\n"
         "        return str(inp.get(\"task_id\") or inp.get(\"shell_id\") or \"\") in own\n"
         "    if name == \"ToolSearch\":\n"
         "        return bool(OWN_SELECT.fullmatch(str(inp.get(\"query\", \"\")).strip()))\n"
         "    if name == \"Monitor\":\n"
         "        return \"ws\" not in inp and allowed(\"Bash\", {\"command\": am2_rewrite(str(inp.get(\"command\", \"\")), own)})\n"
         "    if name == \"Read\":\n"
         "        return allowed(\"Read\", {\"file_path\": am2_rewrite(str(inp.get(\"file_path\", \"\")), own)})\n"
         "    if name in (\"Bash\", \"PowerShell\"):\n"
         "        return allowed(name, {\"command\": am2_rewrite(str(inp.get(\"command\", \"\")), own)})\n"
         "    return allowed(name, inp)\n"
         "\n\ndef allowed(name, inp):\n")
am = rep(am, "        returned, ncomp = False, 0\n", "        returned, ncomp = False, 0\n        own_bg = set()  # amendment 2\n")
am = rep(am, "                    if not allowed(name, inp):\n", "                    if not allowed2(name, inp, own_bg):\n")
am = rep(am, "                    results[blk.get(\"tool_use_id\")] = (bool(blk.get(\"is_error\")), result_text(blk.get(\"content\")))\n",
         "                    results[blk.get(\"tool_use_id\")] = (bool(blk.get(\"is_error\")), result_text(blk.get(\"content\")))\n"
         "                    if blk.get(\"tool_use_id\") in other_ids:  # amendment 2: ids announced by the judge's own calls\n"
         "                        own_bg.update(BG_ID.findall(results[blk.get(\"tool_use_id\")][1]))\n")
am = rep(am, '"amendment": "A12-am1 (a12_blind_check_am1.py)"', '"amendment": "A12-am1+am2 (a12_blind_check_am2.py)"')
io.open(DST, "w", encoding="utf-8", newline="\n").write(am)
print(DST, hashlib.sha256(open(DST, "rb").read()).hexdigest())
