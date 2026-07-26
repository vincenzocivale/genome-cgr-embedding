#!/usr/bin/env bash
# Run last_token pooling for HyenaDNA + Caduceus, dataset-sharded and fully
# concurrent, writing to an ISOLATED staging CSV (never pooling_results.csv)
# so it cannot race the separately-running DNABERT-2 CLS orchestrator's merge.
# A final integration step (merge staging into pooling_results.csv + regen
# tables) is done manually once BOTH runs are complete.
set -uo pipefail
cd "$(dirname "$0")/../../.."
ENV="cgr_bench"
N=4                       # shards per model
STAGING=results/rebuttal/E5_pooling/staging_last_token.csv
LOG=results/rebuttal/E5_pooling/run_e5.log
PY=src/rebuttal/e5_pooling/run_pooling_comparison.py
MERGE_PY=src/rebuttal/e5_pooling/merge_shards.py
MK=src/rebuttal/e5_pooling/make_dataset_shards.py
DATA_ROOT=data/dna_foundation_benchmark
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

declare -A TAGS=(
  ["LongSafari/hyenadna-medium-160k-seqlen-hf"]="hyena"
  ["kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"]="cad"
)

echo "=== $(date -Is) starting last_token staging (HyenaDNA + Caduceus), $N shards each ===" >> "$LOG"
rm -f "$STAGING"
all_shards=()
pids=()
for model in "${!TAGS[@]}"; do
  tag="${TAGS[$model]}"
  gdir="/tmp/e5_lt_${tag}"
  conda run -n "$ENV" python "$MK" "$DATA_ROOT" "$N" "$gdir" >> "$LOG" 2>&1
  for i in $(seq 0 $((N-1))); do
    group=$(cat "$gdir/group_${i}.txt" 2>/dev/null)
    [ -z "$group" ] && continue
    shard="results/rebuttal/E5_pooling/pooling_results_lt_${tag}_${i}.csv"
    rm -f "$shard"       # seed empty: only last_token rows land here
    all_shards+=("$shard")
    conda run -n "$ENV" python "$PY" \
      --models "$model" --poolings last_token --probe linear \
      --datasets $group --n-jobs 6 --batch-size 16 --output "$shard" >> "$LOG" 2>&1 &
    pids+=("$!")
  done
done
wait "${pids[@]}"

echo "=== $(date -Is) last_token shards done, merging into staging ===" >> "$LOG"
conda run -n "$ENV" python "$MERGE_PY" "$STAGING" "${all_shards[@]}" >> "$LOG" 2>&1
if [ -s "$STAGING" ]; then
  rows=$(( $(wc -l < "$STAGING") - 1 ))
  echo "=== $(date -Is) staging OK: $rows last_token rows in $STAGING ===" >> "$LOG"
  rm -f "${all_shards[@]}"
  rm -rf /tmp/e5_lt_hyena /tmp/e5_lt_cad
else
  echo "!!! $(date -Is) STAGING EMPTY/FAILED - keeping shards: ${all_shards[*]} !!!" >> "$LOG"
  exit 1
fi
echo "=== $(date -Is) last_token staging complete (integrate manually when DNABERT-2 CLS is also done) ===" >> "$LOG"
