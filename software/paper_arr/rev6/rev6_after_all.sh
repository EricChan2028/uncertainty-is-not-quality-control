#!/bin/bash
# rev6: wait until every output of jobs_all.txt is complete (1,200 rows), then run the registered read rev6_read.py ONCE and
# log the hash of its output; then wait for the remedy runs (jobs_tier2*.txt) and run the exploratory reruns.
R=/home/tione/notebook/rsi_rl; D=$R/paper_arr/rev6; cd $D
complete(){ while read -r k t m; do [ -z "$k" ] && continue; f=texts/${k}_${t}.jsonl; [ -f $f ] && [ "$(wc -l < $f)" -ge 1200 ] || return 1; done < $1; return 0; }
until complete jobs_all.txt; do sleep 20; done
sleep 10
echo "$(date '+%F %T') jobs_all complete; running rev6_read.py" > logs/after_all.status
python3 rev6_read.py > logs/rev6_read.out 2>&1; rc=$?
{ echo "$(date '+%F %T') rev6 registered read (rev6_read.py, sha256 $(sha256sum rev6_read.py | cut -d' ' -f1)) run once after all jobs_all outputs were complete; exit $rc:"; sha256sum $D/rev6_read.json | sed "s#$R/##"; } >> $R/docs/registration_hashes.txt
echo "$(date '+%F %T') rev6_read.py rc $rc" >> logs/after_all.status
until complete jobs_tier2.txt && complete jobs_tier2b.txt; do sleep 20; done
sleep 10
python3 rev6_reruns.py > logs/rev6_reruns.out 2>&1; python3 rev6_reruns_extra.py > logs/rev6_reruns_extra.out 2>&1; python3 rev6_reruns_a6.py > logs/rev6_reruns_a6.out 2>&1
{ echo "$(date '+%F %T') rev6 exploratory reruns run (rev6_reruns.py, rev6_reruns_extra.py, rev6_reruns_a6.py); outputs:"; sha256sum $D/reruns_*/*.json | sed "s#$R/##"; } >> $R/docs/registration_hashes.txt
python3 rev6_numbers.py > rev6_numbers.txt 2>&1
echo "$(date '+%F %T') reruns and numbers done" >> logs/after_all.status
