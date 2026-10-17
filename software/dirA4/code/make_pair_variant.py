"""Generate the A4 pair-GPU round/runner scripts from the A3 pair scripts (Qwen3-8B own_ag on host <HOST> pairs).
a4_round.sh: like a3_round.sh but with the A4 arm table (own | own_ag) and the abstention guard flags.
a4_runner.sh: like a3_runner.sh but reading queue_a4_8b.txt, logging a4pair, calling a4_round.sh."""
import os
R = "/home/tione/notebook/rsi_rl"
s = open(f"{R}/dirA3/code/a3_round.sh").read()
old_abs = 'ABS=0; [ "$ARM" = "pdc" ] && ABS=1\n'
assert old_abs in s
s = s.replace(old_abs, 'ABS=0; case "$ARM" in pdc|own_ag) ABS=1;; esac\nTARM=own; [ "$ARM" = pdc ] && TARM=pdc\nGUARD=rzero; [ "$ARM" = own_ag ] && GUARD=abstain      # A4 addition 1\n')
old_guard = '--abstain_prompt $ABS >> $OUT/guard.log'
assert old_guard in s
s = s.replace(old_guard, '--abstain_prompt $ABS --guard $GUARD --abs_max 0.25 >> $OUT/guard.log')
old_train = '--arm $ARM --seed $SEED --out $OUT --rollout_dir $RDIR'
assert old_train in s
s = s.replace(old_train, '--arm $TARM --seed $SEED --out $OUT --rollout_dir $RDIR')
s = s.replace('log(){ echo "$(date \'+%F %T\') a3 $NAME: $*" >> $LOG; }', 'log(){ echo "$(date \'+%F %T\') a4 $NAME: $*" >> $LOG; }')
s = s.replace("# One round of the A3 self-generated-curriculum loop", "# A4 pair variant (own | own_ag) of the A3 self-generated-curriculum loop round")
open(f"{R}/dirA3/code/a4_round.sh", "w").write(s)
r = open(f"{R}/dirA3/code/a3_runner.sh").read()
r = r.replace("queue_a3.txt", "queue_a4_8b.txt").replace("a3_round.sh", "a4_round.sh").replace("a3pair", "a4pair")
open(f"{R}/dirA3/code/a4_runner.sh", "w").write(r)
print("wrote a4_round.sh, a4_runner.sh")
