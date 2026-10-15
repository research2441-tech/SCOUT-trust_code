#!/bin/sh
# Supplementary experiments (E3-E6, E8, E9); two worker processes
cd "$(dirname "$0")"
for p in 0 1; do
  ( for w in sens relia duty robust scale budget; do python3 -W ignore supp.py $w $p 2 >> results/log_supp_$p.txt 2>&1; done ) &
done
wait
