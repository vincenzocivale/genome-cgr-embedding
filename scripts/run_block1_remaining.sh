#!/usr/bin/env bash

set -uo pipefail

cd /data2/home/vcivale/genome-cgr-embedding

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_BIN="/data2/home/vcivale/miniconda3/bin/conda"
HG38="/data2/utility/references/genomics/Homo_sapiens_assembly38.fasta"

run() {
  echo
  echo "========================================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "========================================================================"
  "$CONDA_BIN" run --no-capture-output -n cgr_bench nice -n 10 "$@" || \
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN: command exited non-zero, continuing"
}

# Esp 2 — LRA RF: complete missing DNABERT-2 cell (variant_effect_pathogenic_clinvar)
run python src/scripts/regression/train_lra_benchmark.py \
  --model zhihan1996/DNABERT-2-117M \
  --hg38 "$HG38" \
  --fm-batch-size 16 \
  --n-workers 2

# Esp 5 — LRA linear probe FM (3 modelli × 3 task)
run python src/scripts/regression/train_lra_linear_probe.py \
  --model InstaDeepAI/NTv3_650M_pre \
  --hg38 "$HG38" \
  --fm-batch-size 8 \
  --n-workers 2

run python src/scripts/regression/train_lra_linear_probe.py \
  --model LongSafari/hyenadna-large-1m-seqlen-hf \
  --hg38 "$HG38" \
  --fm-batch-size 8 \
  --n-workers 2

run python src/scripts/regression/train_lra_linear_probe.py \
  --model zhihan1996/DNABERT-2-117M \
  --hg38 "$HG38" \
  --fm-batch-size 16 \
  --n-workers 2

# Esp 6 — decomposition ridge (pooling non-mean)
run python src/scripts/decomposition/train_decomposition.py \
  --model LongSafari/hyenadna-medium-160k-seqlen-hf \
  --pooling max --mapper ridge --k-values 4 5 6 \
  --fm-batch-size 16 --n-workers 2

run python src/scripts/decomposition/train_decomposition.py \
  --model zhihan1996/DNABERT-2-117M \
  --pooling max --mapper ridge --k-values 4 5 6 \
  --fm-batch-size 32 --n-workers 2

run python src/scripts/decomposition/train_decomposition.py \
  --model zhihan1996/DNABERT-2-117M \
  --pooling cls --mapper ridge --k-values 4 5 6 \
  --fm-batch-size 32 --n-workers 2

# Esp 2 retry — LRA RF DNABERT-2 (was skipped earlier due to OOM contention)
run python src/scripts/regression/train_lra_benchmark.py \
  --model zhihan1996/DNABERT-2-117M \
  --hg38 "$HG38" \
  --fm-batch-size 8 \
  --n-workers 2

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Block 1 remaining: DONE"
