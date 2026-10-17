"""Build a12_blind_check_am1.py from the registered a12_blind_check.py (sha 9f0b1eb0...) with exactly one change:
before the banned-name test (BAD_ANY) is applied to a Write/Edit/Read path or to a Bash/PowerShell command, a judge's own
temporary subfolder whose name starts with "a12" (directly under a scratchpad, temp or tmp directory, and not one of the
project directory names a12_key, a12_blind, a12_open, a12_runs, a12_draft, a12_pilot) is renamed to "judgetmp".
Every other rule is unchanged. The output records the amendment."""
import hashlib, io, sys

SRC = "a12_blind_check.py"
DST = "a12_blind_check_am1.py"
src = io.open(SRC, encoding="utf-8").read()
assert hashlib.sha256(src.encode("utf-8")).hexdigest().startswith("9f0b1eb0"), "not the registered checker"

def rep(s, a, b):
    assert s.count(a) == 1, (a[:70], s.count(a))
    return s.replace(a, b)

am = src
am = rep(am, '"""A12 part-1 blindness check', '"""[A12 amendment 1, passes 2-3 only: judges\' own temp subfolders named a12* are exempt from the banned-name rule;\n'
         'see A12_amendment1_2026-09-26.md. Otherwise identical to the registered a12_blind_check.py.]\nA12 part-1 blindness check')
am = rep(am, "\n\ndef temp_ok(p):\n",
         "\nJUDGE_DIR = re.compile(r\"((?:scratchpad|temp|tmp)[/\\\\]+)a12(?!_(?:key|blind|open|runs|draft|pilot)(?![\\w.-]))[\\w.-]*\", re.I)  # amendment 1\n"
         "\n\ndef am1_rename(s):  # amendment 1: a judge's own temp subfolder named a12* is not a banned name\n"
         "    return JUDGE_DIR.sub(lambda m: m.group(1) + \"judgetmp\", s)\n"
         "\n\ndef temp_ok(p):\n")
am = rep(am, 'return bool(TEMP.fullmatch(p) or TEMP.fullmatch(p.replace("\\\\", "/"))) and ".." not in p and not BAD_ANY.search(p)',
         'return bool(TEMP.fullmatch(p) or TEMP.fullmatch(p.replace("\\\\", "/"))) and ".." not in p and not BAD_ANY.search(am1_rename(p))')
am = rep(am, 'if BAD_ANY.search(cmd) or any(".." in m.group(0) for m in TEMP.finditer(cmd)):',
         'if BAD_ANY.search(am1_rename(cmd)) or any(".." in m.group(0) for m in TEMP.finditer(cmd)):')
am = rep(am, 'out = {"checker": "a12_blind_check.py", ',
         'out = {"checker": "a12_blind_check.py", "amendment": "A12-am1 (a12_blind_check_am1.py)", ')
io.open(DST, "w", encoding="utf-8", newline="\n").write(am)
print(DST, hashlib.sha256(open(DST, "rb").read()).hexdigest())
