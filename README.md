# genome-cgr-embedding

Benchmarking DNA sequence representations for genomic classification.

This repository compares fixed-resolution and multiscale k-mer features derived
from Chaos Game Representation against embeddings from three genomic foundation
models:

- `InstaDeepAI/NTv3_650M_pre`
- `LongSafari/hyenadna-medium-160k-seqlen-hf`
- `zhihan1996/DNABERT-2-117M`

The public repo is intentionally minimal: it retains the experiments used in the
paper and supplementary material, the canonical result tables, and the scripts
needed to regenerate publication figures and summary statistics.

## Retained experiments

- Classification with Random Forest on k-mer, multiscale k-mer, weighted multiscale, quadtree, wavelet, one-hot, FM and token-embedding features
- Linear probe validation
- Canonical reverse-complement k-mer benchmark
- PCA ablations on k-mer and FM features
- Orthogonal decomposition of FM embeddings into k-mer-explainable and residual components
- Statistical post-processing (`FDR`, `AUROC` consistency)
- Truncation audit across retained FM models
- Splice residual motif attribution
- FM + best k-mer concatenation benchmark
- Publication figure export and paper statistics audit

## Repository structure

```text
genome-cgr-embedding/
├── docs/
│   ├── getting_started.md
│   ├── experiments.md
│   └── results_schema.md
├── results/
│   ├── analysis/
│   ├── classification/
│   ├── concat/
│   ├── decomposition/
│   ├── efficiency/
│   ├── exploratory/
│   └── figures/
├── scripts/
│   ├── export_paper_figures.py
│   └── paper_stats_audit.py
├── src/
│   ├── analysis/
│   ├── core/
│   ├── data/
│   ├── embedders/
│   ├── features/
│   ├── records/
│   ├── scripts/
│   └── training/
├── tests/
├── .gitignore
└── environment.yml
```

## Installation

```bash
conda env create -f environment.yml
conda activate cgr_bench
```

## Quick start

```bash
# 1. Random Forest on fixed-resolution k-mers
python3 src/scripts/classification/train_kmer_rf.py --k-values 4 5 6

# 2. Random Forest on FM embeddings
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_fm_rf.py \
  --model InstaDeepAI/NTv3_650M_pre

# 3. Linear probe on FM embeddings
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
  --mode fm --model InstaDeepAI/NTv3_650M_pre --pooling mean

# 4. Decomposition benchmark
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
  --model InstaDeepAI/NTv3_650M_pre --mapper ridge --k-values 4 5 6

# 5. Export publication figures
python3 scripts/export_paper_figures.py

# 6. Print publication audit
python3 scripts/paper_stats_audit.py
```

## Canonical results

The retained single sources of truth are:

- `results/classification/records_rf.csv`
- `results/classification/records_linear_probe.csv`
- `results/classification/records_canonical_kmer.csv`
- `results/classification/records_pca.csv`
- `results/decomposition/records_decomposition.csv`
- `results/analysis/fdr_results.csv`
- `results/analysis/auroc_consistency.csv`
- `results/concat/records_concat_best.csv`
- `results/classification/truncation_analysis.csv`
- `results/exploratory/splice_residual_motif_overlap.csv`
- `results/efficiency/efficiency.csv`
- `results/efficiency/efficiency_gpu_parallel.csv`
- `results/figures/`

Derived tables such as “best k per dataset” are recomputed on the fly by the
public scripts and are not stored as separate tracked artifacts.

## Dataset layout

Classification datasets are expected as:

```text
data/dna_foundation_benchmark/
├── <group>/<dataset_name>/
│   ├── train.csv
│   └── test.csv
```

Each CSV must expose `sequence` and `label` columns.

## Documentation

- `docs/getting_started.md`
- `docs/experiments.md`
- `docs/results_schema.md`
