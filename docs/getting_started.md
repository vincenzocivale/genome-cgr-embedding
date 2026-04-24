# Getting Started

## Requirements

- Python 3.10+, CUDA-capable GPU (for FM embedding extraction)
- Conda (Miniconda or Anaconda)

## Installation

```bash
conda env create -f environment.yml
conda activate cgr_bench
```

## Data layout

Datasets are expected at `data/dna_foundation_benchmark/`.  
Each task has the structure:

```
data/dna_foundation_benchmark/<category>/<task_name>/
    train.csv   # columns: sequence, label
    test.csv
```

57 classification datasets and 2 LRA regression datasets are included.

## Quick start

### K-mer RF benchmark (CPU only)
```bash
python3 src/scripts/classification/train_kmer_rf.py --k-values 4 5 6 --n-workers 4
```

### FM embeddings RF (GPU required)
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_fm_rf.py \
    --model InstaDeepAI/NTv3_650M_pre --fm-batch-size 32
```

### Linear probe (verify RF results)
```bash
python3 src/scripts/classification/train_linear_probe.py --mode kmer --k-values 4 5 6
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model InstaDeepAI/NTv3_650M_pre
```

### Decomposition (k-mer → FM projection + residual)
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model InstaDeepAI/NTv3_650M_pre --mapper ridge --k-values 4 5 6
```

### LRA regression benchmark
```bash
python3 src/scripts/regression/train_lra_benchmark.py \
    --k-values 4 5 6 --hg38 /path/to/hg38.fa
```

### Run all P5–P8 experiments (GPU)
```bash
bash scripts/run_plan_p5_p8_cgr_bench.sh
```

## Supported foundation models

| Model | HuggingFace ID |
|---|---|
| NTv3 650M | `InstaDeepAI/NTv3_650M_pre` |
| HyenaDNA 160k | `LongSafari/hyenadna-medium-160k-seqlen-hf` |
| DNABERT-2 | `zhihan1996/DNABERT-2-117M` |

> **Note:** Evo2 is excluded from all experiments.

## Embedding cache

Extracted FM embeddings are cached in `cache/` as `.npz` files keyed by
`(model, dataset, split, pooling)`. Re-running a script reuses the cache,
making downstream ablations fast.

## Resource guidelines

- Default `--n-workers`: 4–8 for k-mer extraction; 1–2 on shared servers.
- Default `--fm-batch-size`: 32 for NTv3, 8 for HyenaDNA, 16 for DNABERT-2.
- All training scripts skip already-completed rows; safe to interrupt and resume.
