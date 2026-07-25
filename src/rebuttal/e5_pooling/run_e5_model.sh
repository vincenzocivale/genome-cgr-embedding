#!/usr/bin/env bash
# Run E5 (pooling ablation) for one model, sharded by pooling across
# concurrent processes, then merge results back into the shared CSV.
#
# Usage: run_e5_model.sh <model_name> <conda_env> <output_csv> <pooling1> [pooling2 ...]
#
# Example (Evo2, from a machine where evo2 + transformer_engine are installed):
#   bash run_e5_model.sh evo2_1b_base evo2_bench \
#     results/rebuttal/E5_pooling/pooling_results.csv \
#     mean max mean_max attention
#
# Each pooling runs as its own process against its own seeded copy of the
# output CSV (avoids concurrent-write races), then merge_shards.py folds
# them back in, deduplicating by (dataset, model, pooling, probe). Shard
# files are only deleted if the merge actually grew the main CSV -- see
# merge_shards.py's docstring for why a naive `conda run python - <<EOF`
# merge silently loses data.
set -uo pipefail
cd "$(dirname "$0")/../../.."
MODEL="$1"; ENV="$2"; MAIN="$3"; shift 3
POOLINGS=("$@")

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=src/rebuttal/e5_pooling/run_pooling_comparison.py
MERGE_PY=src/rebuttal/e5_pooling/merge_shards.py
LOG="$(dirname "$MAIN")/run_e5.log"

echo "=== $(date -Is) starting $MODEL, parallel shards: ${POOLINGS[*]} ===" >> "$LOG"
shard_files=()
pids=()
for pooling in "${POOLINGS[@]}"; do
  shard="$(dirname "$MAIN")/pooling_results_shard_${pooling}.csv"
  cp "$MAIN" "$shard" 2>/dev/null || true
  shard_files+=("$shard")
  conda run -n "$ENV" python "$PY" \
    --models "$MODEL" --poolings "$pooling" --probe linear \
    --n-jobs 16 --batch-size 8 --output "$shard" >> "$LOG" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"

echo "=== $(date -Is) $MODEL shards done, merging ===" >> "$LOG"
before_lines=$(wc -l < "$MAIN" 2>/dev/null || echo 0)
conda run -n "$ENV" python "$MERGE_PY" "$MAIN" "${shard_files[@]}" >> "$LOG" 2>&1
rc=$?
after_lines=$(wc -l < "$MAIN" 2>/dev/null || echo 0)
if [ "$rc" -ne 0 ] || [ "$after_lines" -lt "$before_lines" ]; then
  echo "!!! $(date -Is) MERGE FAILED (rc=$rc, before=$before_lines, after=$after_lines) - NOT deleting shards: ${shard_files[*]} !!!" >> "$LOG"
  exit 1
fi
echo "=== $(date -Is) merge OK: $before_lines -> $after_lines lines ===" >> "$LOG"
rm -f "${shard_files[@]}"
echo "=== $(date -Is) finished $MODEL ===" >> "$LOG"
