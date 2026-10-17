"""Amendment A3e: arms own_tg (tracer-calibrated guard) and own_azr (AZR learnability guard) in a3_round1.sh.
Both train with the own reward on their own guard's admitted set; the guard type is passed to a3_guard.py."""
p = "/home/tione/notebook/rsi_rl/dirA3/code/a3_round1.sh"
s = open(p).read()
assert "own_tg" not in s
old = 'TARM=$ARM; case "$ARM" in own_sub|own_clean|own_dirty) TARM=own;; esac\n'
assert old in s
new = ('TARM=$ARM; case "$ARM" in own_sub|own_clean|own_dirty|own_tg|own_azr) TARM=own;; esac\n'
       'GUARD=rzero; case "$ARM" in own_tg) GUARD=tracer;; own_azr) GUARD=azr;; esac      # amendment A3e\n')
s = s.replace(old, new, 1)
old = '--out_dir $OUT --seed $((SEED*100+RD)) --abstain_prompt $ABS >> $OUT/guard.log'
assert old in s
s = s.replace(old, '--out_dir $OUT --seed $((SEED*100+RD)) --abstain_prompt $ABS --guard $GUARD >> $OUT/guard.log', 1)
# the round-1 copy shortcut applies only to the subset arms, not to the guard arms
old = 'if [ "$RD" -eq 1 ] && [ "$TARM" != "$ARM" ] && [ ! -s $OUT/admitted.jsonl ]; then'
assert old in s
s = s.replace(old, 'if [ "$RD" -eq 1 ] && [ "$GUARD" = rzero ] && [ "$TARM" != "$ARM" ] && [ ! -s $OUT/admitted.jsonl ]; then', 1)
open(p, "w", newline="\n").write(s)
print("patched a3_round1.sh")
