#!/usr/bin/env bash
set -euo pipefail

PY=/data2/home/vcivale/miniconda3/envs/cgr_bench/bin/python
LOG_DIR=results/exploratory/plan_runs
mkdir -p "$LOG_DIR"

echo "[INFO] Python: $PY"
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "[STEP] P5 linear probe pooling"
$PY src/scripts/classification/train_linear_probe.py --mode fm --model InstaDeepAI/NTv3_650M_pre --pooling max --fm-batch-size 16 --n-jobs-lr 2
$PY src/scripts/classification/train_linear_probe.py --mode fm --model InstaDeepAI/NTv3_650M_pre --pooling cls --fm-batch-size 16 --n-jobs-lr 2
$PY src/scripts/classification/train_linear_probe.py --mode fm --model LongSafari/hyenadna-medium-160k-seqlen-hf --pooling max --fm-batch-size 4 --n-jobs-lr 2
$PY src/scripts/classification/train_linear_probe.py --mode fm --model zhihan1996/DNABERT-2-117M --pooling max --fm-batch-size 8 --n-jobs-lr 2
$PY src/scripts/classification/train_linear_probe.py --mode fm --model zhihan1996/DNABERT-2-117M --pooling cls --fm-batch-size 8 --n-jobs-lr 2

echo "[STEP] P5+P6 decomposition ridge pooling"
$PY src/scripts/decomposition/train_decomposition.py --model InstaDeepAI/NTv3_650M_pre --pooling mean --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 16
$PY src/scripts/decomposition/train_decomposition.py --model InstaDeepAI/NTv3_650M_pre --pooling max --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 16
$PY src/scripts/decomposition/train_decomposition.py --model InstaDeepAI/NTv3_650M_pre --pooling cls --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 16
$PY src/scripts/decomposition/train_decomposition.py --model LongSafari/hyenadna-medium-160k-seqlen-hf --pooling mean --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 4
$PY src/scripts/decomposition/train_decomposition.py --model LongSafari/hyenadna-medium-160k-seqlen-hf --pooling max --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 4
$PY src/scripts/decomposition/train_decomposition.py --model zhihan1996/DNABERT-2-117M --pooling mean --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 8
$PY src/scripts/decomposition/train_decomposition.py --model zhihan1996/DNABERT-2-117M --pooling max --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 8
$PY src/scripts/decomposition/train_decomposition.py --model zhihan1996/DNABERT-2-117M --pooling cls --mapper ridge --k-values 4 5 6 --n-workers 1 --fm-batch-size 8

echo "[STEP] P6 decomposition MLP mean"
$PY src/scripts/decomposition/train_decomposition.py --model InstaDeepAI/NTv3_650M_pre --pooling mean --mapper mlp --k-values 4 5 6 --n-workers 1 --fm-batch-size 16 --mlp-epochs 20
$PY src/scripts/decomposition/train_decomposition.py --model LongSafari/hyenadna-medium-160k-seqlen-hf --pooling mean --mapper mlp --k-values 4 5 6 --n-workers 1 --fm-batch-size 4 --mlp-epochs 20
$PY src/scripts/decomposition/train_decomposition.py --model zhihan1996/DNABERT-2-117M --pooling mean --mapper mlp --k-values 4 5 6 --n-workers 1 --fm-batch-size 8 --mlp-epochs 20

echo "[STEP] P7 splice motif analysis"
$PY src/scripts/analysis/analyze_splice_residual_motifs.py --model InstaDeepAI/NTv3_650M_pre --k 6 --n-workers 1 --fm-batch-size 8 --max-seqs-per-motif 128

echo "[STEP] P8 timing symmetry"
$PY src/scripts/utils/benchmark_efficiency.py --seq-lens 100 250 500 1000 2000 --n-seqs 32 --n-workers 1 --methods fm_NTv3_650M_pre,fm_hyenadna-medium-160k-seqlen-hf,fm_DNABERT-2-117M,kmer_k6

echo "[DONE] Plan P5-P8 completed"
