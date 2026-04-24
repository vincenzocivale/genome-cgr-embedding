# genome-cgr-embedding

Benchmarking DNA sequence representations for genomic classification.

This project compares the **informational quality** of k-mer frequency vectors (derived from Chaos Game Representation) against embeddings produced by pre-trained **genomic Foundation Models** (NTv3, HyenaDNA, DNABERT-2), using Random Forest classifiers and linear probes as the downstream probe.

A supplementary experiment fits Ridge Regression from k-mer features to FM embeddings, measuring how much of the FM's information is already captured by k-mer statistics.

---

## Methods compared

| Method | Description |
|---|---|
| `kmer_k{K}` | L1-normalised k-mer frequency vector (4^K dims) from FCGR |
| `multi_k{4_5_6}` | Concatenation of k=4,5,6 vectors, each L2-normalised |
| `wms_k{…}` | Weighted multiscale k-mer (LogReg CV weight search) |
| `quadtree_{…}` | Adaptive QuadTree CGR features (chi-square splitting) |
| `wavelet_{…}` | Spatial pyramid pooling on FCGR at multiple resolutions |
| `onehot_{W}bp` | One-hot encoding of the central W base pairs |
| `fm_{model}` | Mean-pooled last hidden state of a pre-trained FM |
| `tok_{model}` | Mean-pooled embedding-layer output (pre-transformer) |
| `ridge_k{K}_{model}` | Ridge regression R²: k-mer → FM embedding |

---

## Documentation

- [Getting started and installation](docs/getting_started.md)
- [Experiment reference](docs/experiments.md)
- [Results file schema](docs/results_schema.md)
- [Missing / in-progress experiments](docs/missing_experiments.md)

---

## Repository structure

```
genome-cgr-embedding/
├── src/
│   ├── core/                # FCGR and QuadTree algorithms
│   ├── data/                # Dataset discovery and loading
│   ├── embedders/           # FM embedding extractors (NTv3, HyenaDNA, DNABERT-2)
│   ├── features/            # K-mer feature extraction
│   ├── training/            # RF, Ridge, Linear, MLP pipelines + efficiency logging
│   ├── analysis/            # FDR, AUROC consistency, mutual information
│   ├── records/             # Results persistence (CSV schema & helpers)
│   └── scripts/
│       ├── classification/  # RF + linear probe + canonical k-mer experiments
│       ├── decomposition/   # Ridge/MLP k-mer→FM decomposition
│       ├── regression/      # Long Range Arena benchmark
│       ├── analysis/        # Motif analysis, confusion matrix, truncation
│       └── utils/           # Benchmark runners, efficiency, migration
├── results/
│   ├── classification/      # records_rf.csv, records_linear_probe.csv, records_canonical_kmer.csv
│   ├── decomposition/       # records_decomposition.csv
│   ├── regression/          # lra_records_rf.csv, lra_records_linear_probe.csv
│   ├── analysis/            # fdr_results.csv, auroc_consistency.csv
│   ├── efficiency/          # efficiency.csv, efficiency_gpu_parallel.csv
│   ├── exploratory/         # fusion, MI, info_theory, splice motif overlap
│   ├── concat/              # FM + k-mer concatenation experiments
│   └── figures/             # Publication-ready PDF figures
├── docs/
│   ├── getting_started.md
│   ├── experiments.md
│   ├── results_schema.md
│   ├── missing_experiments.md
│   └── background/          # Theory, ablation plans
├── notebooks/               # Numbered Jupyter analysis notebooks
├── scripts/                 # Paper figure export and stats audit
├── tests/
├── cache/                   # Pre-computed embeddings (.npz, git-ignored)
└── environment.yml
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate cgr_bench
```

See [docs/getting_started.md](docs/getting_started.md) for full setup instructions.

---

## Quick start

```bash
# 1. k-mer RF (all 57 classification datasets)
python3 src/scripts/classification/train_kmer_rf.py --k-values 4 5 6 --n-workers 4

# 2. FM embeddings + RF
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_fm_rf.py \
    --model InstaDeepAI/NTv3_650M_pre

# 3. Ridge decomposition: how well do k-mers predict FM embeddings?
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model InstaDeepAI/NTv3_650M_pre --mapper ridge --k-values 4 5 6

# 4. Linear probe (verify RF results)
python3 src/scripts/classification/train_linear_probe.py --mode kmer --k-values 4 5 6

# 5. Long Range Arena benchmark (requires genome FASTA)
python3 src/scripts/regression/train_lra_benchmark.py \
    --k-values 4 5 6 --hg38 /path/to/hg38.fa
```

Results are written incrementally to `results/classification/records_rf.csv`; completed configurations are automatically skipped on re-runs.

See [docs/experiments.md](docs/experiments.md) for the full experiment reference with all scripts and output files.

---

## Dataset format

Datasets are expected as `train.csv` / `test.csv` files:

```
data/dna_foundation_benchmark/
├── <category>/<task_name>/
│   ├── train.csv   # columns: sequence, label
│   └── test.csv
```

---

## Supported FM models

| Model | HuggingFace ID |
|---|---|
| NTv3 650M | `InstaDeepAI/NTv3_650M_pre` |
| HyenaDNA medium 160k | `LongSafari/hyenadna-medium-160k-seqlen-hf` |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` |

> **Note:** Evo2 is excluded from all experiments.
