# Getting Started

## Environment

```bash
conda env create -f environment.yml
conda activate cgr_bench
```

The public environment only includes dependencies required by the retained
publication pipeline: model loading, feature extraction, training, plotting and
result analysis.

## Dataset layout

All retained experiments operate on classification datasets with the layout:

```text
data/dna_foundation_benchmark/
├── <group>/<dataset_name>/
│   ├── train.csv
│   └── test.csv
```

Both files must contain:

- `sequence`
- `label`

## Core commands

```bash
# Fixed-resolution k-mer RF
python3 src/scripts/classification/train_kmer_rf.py --k-values 4 5 6

# FM RF
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_fm_rf.py \
  --model InstaDeepAI/NTv3_650M_pre

# Linear probe
python3 src/scripts/classification/train_linear_probe.py --mode kmer --k-values 4 5 6

# Canonical reverse-complement k-mers
python3 src/scripts/classification/train_canonical_kmer.py --k-values 4 5 6

# PCA ablation
python3 src/scripts/classification/train_pca_rf.py --mode fm \
  --model zhihan1996/DNABERT-2-117M

# Decomposition
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
  --model InstaDeepAI/NTv3_650M_pre --mapper ridge --k-values 4 5 6

# Publication outputs
python3 scripts/export_paper_figures.py
python3 scripts/paper_stats_audit.py
```

## Canonical outputs

Use only the retained public tables:

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
