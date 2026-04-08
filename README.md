# genome-cgr-embedding

Benchmarking DNA sequence representations for genomic classification.

This project compares the **informational quality** of k-mer frequency vectors (derived from Chaos Game Representation) against embeddings produced by pre-trained **genomic Foundation Models** (NTv3, HyenaDNA, DNABERT-2), using Random Forest classifiers as the downstream probe.

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

## Repository structure

```
genome-cgr-embedding/
├── src/
│   ├── core/                # FCGR and QuadTree algorithms
│   │   ├── fcgr.py
│   │   └── quadtree.py
│   ├── data/                # Dataset discovery and loading
│   │   └── loader.py
│   ├── embedders/           # FM embedding extractors
│   │   ├── embedding_cache.py   # Shared device detection & cache I/O
│   │   ├── fm_embedder.py       # NTv3, HyenaDNA, DNABERT-2
│   │   ├── hyena_embedder.py    # HyenaDNA-specific wrapper
│   │   └── tokenizer_embedder.py
│   ├── features/            # Feature extraction
│   │   └── kmer_features.py
│   ├── training/            # Shared RF pipeline and utilities
│   │   ├── rf_pipeline.py       # train_rf, eval_rf (shared across all scripts)
│   │   ├── ridge_mapping.py
│   │   └── efficiency.py
│   ├── analysis/            # Post-hoc analysis
│   │   ├── compute_mi.py
│   │   ├── info_theory_pipeline.py
│   │   └── fill_dataset_info.py
│   ├── records/             # Results persistence
│   │   └── records.py
│   └── scripts/             # Runnable experiment scripts
│       ├── train_kmer_rf.py
│       ├── train_fm_rf.py
│       ├── train_multiscale_kmer_rf.py
│       ├── train_ridge.py
│       ├── train_ridge_multiscale.py
│       ├── train_lra_benchmark.py
│       ├── benchmark_all_methods.py
│       └── ...
├── results/
│   ├── classification/      # Main RF benchmark results
│   │   └── records.csv
│   ├── regression/          # Long Range Arena regression results
│   │   └── lra_records.csv
│   ├── concat/              # FM + k-mer concatenation experiments
│   │   └── records_concat_best.csv
│   └── exploratory/         # Exploratory analyses (MI, efficiency, fusion)
├── notebooks/               # Jupyter analysis notebooks
│   ├── results_overview.ipynb
│   ├── paper_plots.ipynb
│   └── linear__probe.ipynb
├── tests/
├── cache/                   # Pre-computed embeddings (.npz, git-ignored)
├── notes/                   # Theoretical background
└── environment.yml
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate cgr_bench
```

---

## Quick start

```bash
# 1. k-mer RF (all classification datasets)
python3 src/scripts/train_kmer_rf.py --k-values 4 5 6 --n-workers 8

# 2. FM embeddings + RF
python3 src/scripts/train_fm_rf.py --model InstaDeepAI/NTv3_650M_pre

# 3. Ridge mapping: how well do k-mers predict FM embeddings?
python3 src/scripts/train_ridge.py \
    --model InstaDeepAI/NTv3_650M_pre \
    --k-values 4 5 6

# 4. Run all methods at once
python3 src/scripts/benchmark_all_methods.py \
    --methods kmer,multiscale,fm,tok \
    --k-values 4 5 6 \
    --model InstaDeepAI/NTv3_650M_pre \
    --n-workers 8

# 5. Long Range Arena benchmark (requires genome FASTA)
python3 src/scripts/train_lra_benchmark.py \
    --k-values 4 5 6 \
    --hg38 /path/to/hg38.fa
```

Results are written incrementally to `results/classification/records.csv`; completed configurations are automatically skipped on re-runs.

---

## Dataset format

Datasets are expected as `train.csv` / `test.csv` files:

```
/data/genomic_bench/dna_foundation_benchmark/
├── task_name_1/
│   ├── train.csv   # columns: sequence, label
│   └── test.csv
└── task_name_2/
    ├── train.csv
    └── test.csv
```

---

## Supported FM models

| Model | HuggingFace ID |
|---|---|
| NTv3 650M | `InstaDeepAI/NTv3_650M_pre` |
| HyenaDNA medium 160k | `LongSafari/hyenadna-medium-160k-seqlen-hf` |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` |
| Evo2 7B | `evo2_7b` (requires `evo2` package) |
