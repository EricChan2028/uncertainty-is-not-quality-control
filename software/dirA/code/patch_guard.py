import io
p = "/home/tione/notebook/rsi_rl/dirA/code/c0_master.sh"
s = io.open(p).read()
old = '  log "$tag: launching $n shards on GPUs ${gpus[*]}"'
new = ('  for pid in $(pgrep -f "c0_probe.py --model $path --tag $tag "); do kill $pid 2>/dev/null; done; sleep 3   # never two writers per shard file\n'
       '  log "$tag: launching $n shards on GPUs ${gpus[*]}"')
if "never two writers" in s:
    print("guard present")
else:
    assert old in s
    io.open(p, "w").write(s.replace(old, new, 1))
    print("master: stale-shard guard added (takes effect on next start)")
