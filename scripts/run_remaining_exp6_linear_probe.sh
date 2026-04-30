#!/usr/bin/env bash

set -euo pipefail

cd /data2/home/vcivale/genome-cgr-embedding

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_BIN="/data2/home/vcivale/miniconda3/bin/conda"

run() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  "$CONDA_BIN" run --no-capture-output -n cgr_bench nice -n 10 "$@"
}

run python src/scripts/classification/train_linear_probe.py \
  --mode fm \
  --data-root data/dna_foundation_benchmark \
  --model InstaDeepAI/NTv3_650M_pre \
  --pooling cls \
  --fm-batch-size 8

run python src/scripts/classification/train_linear_probe.py \
  --mode fm \
  --data-root data/dna_foundation_benchmark \
  --model LongSafari/hyenadna-medium-160k-seqlen-hf \
  --pooling max \
  --fm-batch-size 16

run python src/scripts/classification/train_linear_probe.py \
  --mode fm \
  --data-root data/dna_foundation_benchmark \
  --model zhihan1996/DNABERT-2-117M \
  --pooling cls \
  --fm-batch-size 32
