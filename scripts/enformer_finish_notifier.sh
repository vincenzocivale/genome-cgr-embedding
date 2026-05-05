#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

echo "Starting enformer finish notifier: will check every 60s" >> logs/enformer_notifier.log 2>&1 || true

while true; do
  # wait a bit before checking
  sleep 60

  # if any of the experiment processes are running, continue waiting
  if ps aux | grep -E 'train_fm_rf.py|train_lra_regression.py|train_decomposition.py' | grep -v grep >/dev/null; then
    continue
  fi

  # No experiment processes found -> collect final summaries
  echo "==== DONE $(date) ====" > logs/enformer_done.log
  echo "Final classification results (tail 50):" >> logs/enformer_done.log
  tail -n 50 results/classification/records_rf.csv >> logs/enformer_done.log 2>/dev/null || true
  echo "\nFinal regression results (tail 50):" >> logs/enformer_done.log
  tail -n 50 results/regression_records.csv >> logs/enformer_done.log 2>/dev/null || true
  echo "\nFinal decomposition results (tail 50):" >> logs/enformer_done.log
  tail -n 50 results/decomposition/records_decomposition.csv >> logs/enformer_done.log 2>/dev/null || true

  # Create a flag file to indicate completion
  touch logs/enformer_done.flag

  # Also write a short message for quick inspection
  echo "Experiments with google/enformer finished at $(date). See logs/enformer_done.log" > /tmp/enformer_done_message.txt

  break
done
