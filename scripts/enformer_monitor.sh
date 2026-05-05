#!/usr/bin/env bash
set -eu

mkdir -p logs
while true; do
  echo "==== $(date) ===="
  ps aux | grep -E 'train_fm_rf.py|train_lra_regression.py|train_decomposition.py' | grep -v grep || echo 'No running experiment processes'
  echo '--- results/classification/records_rf.csv (tail 3) ---'
  tail -n 3 results/classification/records_rf.csv 2>/dev/null || true
  echo '--- results/regression_records.csv (tail 3) ---'
  tail -n 3 results/regression_records.csv 2>/dev/null || true
  echo '--- results/decomposition/records_decomposition.csv (tail 3) ---'
  tail -n 3 results/decomposition/records_decomposition.csv 2>/dev/null || true
  echo
  sleep 60
done
