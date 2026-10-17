"""One-off patch (2026-09-22): rounds after round 1 and trained-but-unevaluated runs are claimable only on the host that holds the checkpoint."""
import os
import re

R = "/home/tione/notebook/rsi_rl"
DEP = 'if [ "$rd" -gt 1 ] && [ ! -f $RUNS/${L}_r$((rd-1))/.done ]; then continue; fi'
LOC = 'if [ "$rd" -gt 1 ] && [ ! -e /dev/shm/ckpt_b/${L}_r$((rd-1))/final/config.json ]; then continue; fi   # previous final must be on this host'
for f in (f"{R}/dirA3/code/a3_runner1.sh", f"{R}/dirA3/code/a3_runner.sh"):
    s = open(f).read()
    if "previous final must be on this host" in s:
        print(f, "already patched")
        continue
    lines = s.split("\n")
    out, done = [], 0
    for ln in lines:
        out.append(ln)
        if DEP in ln:
            out.append(re.match(r"\s*", ln).group(0) + LOC)
            done += 1
    assert done == 1, (f, done)
    open(f + ".new", "w").write("\n".join(out))
    os.replace(f + ".new", f)
    os.chmod(f, 0o755)
    print(f, "patched")
f = f"{R}/dirB/code/b_runner2.sh"
s = open(f).read()
if "final lives on another host" in s:
    print(f, "already patched")
else:
    ck = re.search(r"\bCK=(\S+)", s)
    print("CK definition:", ck.group(0) if ck else "NOT FOUND")
    assert ck
    lines = s.split("\n")
    out, done = [], 0
    for ln in lines:
        if ln.strip() == "claim $name || continue":
            out.append(re.match(r"\s*", ln).group(0) + '[ -f $RUNS/$name/.trained ] && [ ! -e $CK/$name/final/config.json ] && continue   # final lives on another host')
            done += 1
        out.append(ln)
    assert done == 1, done
    open(f + ".new", "w").write("\n".join(out))
    os.replace(f + ".new", f)
    os.chmod(f, 0o755)
    print(f, "patched")
