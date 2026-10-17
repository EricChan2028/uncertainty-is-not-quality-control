#!/bin/bash
# Final A3 readouts: run the registered readout scripts into dirA3/results/ and hash them with the verdict.
R=/home/tione/notebook/rsi_rl; cd $R; mkdir -p dirA3/results; TS=$(date +%Y-%m-%d_%H%M)
python3 dirA3/code/a3_kc.py > dirA3/results/a3_kc_readout_${TS}.txt 2>&1
python3 dirA3/code/a3_analyze.py > dirA3/results/a3_analyze_readout_${TS}.txt 2>&1
{ echo "# $(date '+%F %T') A3 final readouts + verdict"; sha256sum dirA3/results/a3_kc_readout_${TS}.txt dirA3/results/a3_analyze_readout_${TS}.txt dirA3/docs/A3_verdict_2026-09-23.md; } >> docs/registration_hashes.txt
echo "wrote dirA3/results/a3_kc_readout_${TS}.txt dirA3/results/a3_analyze_readout_${TS}.txt"; tail -4 docs/registration_hashes.txt | cut -c1-110
