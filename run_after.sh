#!/bin/sh
# Second pass (open-loop quality, blame ablation) followed by the supplementary experiments
cd "$(dirname "$0")"
for p in 0 1; do python3 -W ignore rerun.py $p 2 >> results/log_rerun_$p.txt 2>&1 & done
wait
./run_supp.sh
