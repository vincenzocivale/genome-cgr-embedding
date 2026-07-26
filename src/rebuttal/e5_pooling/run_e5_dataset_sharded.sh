#!/usr/bin/env bash
# Run E5 for ONE (model, pooling) across all 57 datasets, parallelized by
# SHARDING THE DATASETS across N concurrent processes (use this when there is
# only one pooling to run, so run_e5_model.sh's shard-by-pooling gives no
# parallelism). Datasets are size-balanced across shards (largest spread out).
#
# Usage: run_e5_dataset_sharded.sh <model> <pooling> <conda_env> <n_shards> <output_csv>
#
# Example (DNABERT-2 CLS, 8-way):
#   bash run_e5_dataset_sharded.sh zhihan1996/DNABERT-2-117M cls cgr_bench 8 \
#     results/rebuttal/E5_pooling/pooling_results.csv
set -uo pipefail
cd "$(dirname "$0")/../../.."
MODEL="$1"; POOLING="$2"; ENV="$3"; N="$4"; MAIN="$5"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=src/rebuttal/e5_pooling/run_pooling_comparison.py
MERGE_PY=src/rebuttal/e5_pooling/merge_shards.py
LOG="$(dirname "$MAIN")/run_e5.log"
DATA_ROOT=data/dna_foundation_benchmark

# Build N size-balanced dataset groups -> files (real .py, NOT a heredoc:
# conda run does not forward heredoc stdin).
rm -rf /tmp/e5_shards
conda run -n "$ENV" python src/rebuttal/e5_pooling/make_dataset_shards.py \
  "$DATA_ROOT" "$N" /tmp/e5_shards >> "$LOG" 2>&1

echo "=== $(date -Is) starting $MODEL / $POOLING, $N dataset-sharded processes ===" >> "$LOG"
shard_files=()
pids=()
for i in $(seq 0 $((N-1))); do
  group=$(cat /tmp/e5_shards/group_${i}.txt)
  [ -z "$group" ] && continue
  shard="$(dirname "$MAIN")/pooling_results_shard_ds${i}.csv"
  cp "$MAIN" "$shard" 2>/dev/null || true
  shard_files+=("$shard")
  conda run -n "$ENV" python "$PY" \
    --models "$MODEL" --poolings "$POOLING" --probe linear \
    --datasets $group --n-jobs 8 --batch-size 16 --output "$shard" >> "$LOG" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"

echo "=== $(date -Is) $MODEL / $POOLING shards done, merging ===" >> "$LOG"
before=$(wc -l < "$MAIN" 2>/dev/null || echo 0)
conda run -n "$ENV" python "$MERGE_PY" "$MAIN" "${shard_files[@]}" >> "$LOG" 2>&1
rc=$?
after=$(wc -l < "$MAIN" 2>/dev/null || echo 0)
if [ "$rc" -ne 0 ] || [ "$after" -lt "$before" ]; then
  echo "!!! $(date -Is) MERGE FAILED (rc=$rc, before=$before, after=$after) - keeping shards: ${shard_files[*]} !!!" >> "$LOG"
  exit 1
fi
echo "=== $(date -Is) merge OK: $before -> $after lines ===" >> "$LOG"
rm -f "${shard_files[@]}"
rm -rf /tmp/e5_shards
echo "=== $(date -Is) finished $MODEL / $POOLING ===" >> "$LOG"

echo "=== $(date -Is) regenerating E5 tables ===" >> "$LOG"
MPLCONFIGDIR=/tmp/mplconfig conda run -n "$ENV" python \
  src/rebuttal/analyze_rebuttal_experiments.py \
  --e5 "$MAIN" --e4 results/rebuttal/E4_probe_fairness/probe_matrix_ntv3_k6.csv \
  --out-root results/rebuttal >> "$LOG" 2>&1
echo "=== $(date -Is) done ($MODEL / $POOLING) ===" >> "$LOG"
