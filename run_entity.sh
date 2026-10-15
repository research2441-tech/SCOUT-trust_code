#!/bin/sh
# E2-E9 main entity-level runs: two worker processes, TON_IoT then CICIoT2023
cd "$(dirname "$0")"; mkdir -p results figures
for p in 0 1; do
  ( python3 -W ignore entity.py TON_IoT $p 2 > results/log_entity_ton_$p.txt 2>&1;
    while [ ! -f results/pool_CICIoT2023_Web-based_1.npz ]; do sleep 30; done;
    python3 -W ignore entity.py CICIoT2023 $p 2 > results/log_entity_cic_$p.txt 2>&1 ) &
done
wait
