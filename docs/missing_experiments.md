# Missing / In-Progress Experiments

Status as of April 2026. Based on `docs/background/plan_p5_p8.md`.

---

## P5 — Pooling Robustness

**Status: Partial**

Mean pooling is complete for all models. Max and cls pooling have not been run
for all model × dataset combinations.

**What remains:**
- Re-extract FM embeddings with `--pooling max` and `--pooling cls` for all 3 models × 57 datasets.
- Run linear probe with max/cls pooling for HyenaDNA (max only; cls not applicable for char-level models).
- Add `fm_{model}_max_*` and `fm_{model}_cls_*` columns to `results/classification/records_linear_probe.csv`.
- Run decomposition with max/cls pooling for all models.

**Script:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model <model> --pooling max
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model <model> --pooling max --mapper ridge --k-values 4 5 6
```

---

## P6 — Non-linear Decomposition Upper Bound

**Status: Partial**

The MLP mapper is implemented (`src/training/mlp_mapping.py`). A single NTv3 mean-MLP
run is logged in `results/archive/` but not consolidated into the main results file.

**What remains:**
- Run MLP decomposition for all 3 models × mean pooling.
- Add `mlp_R2`, `mlp_proj_{model}_MCC`, `mlp_resid_{model}_MCC` columns to
  `results/decomposition/records_decomposition.csv`.

**Script:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model InstaDeepAI/NTv3_650M_pre --mapper mlp --k-values 4 5 6 --mlp-epochs 20
```

---

## P7 — Splice Residual Motif Attribution

**Status: Complete**

`results/exploratory/splice_residual_motif_overlap.csv` generated for 4 splice datasets.
PDF figures in `results/figures/splice_gpu_parallel/`.

---

## P8 — Timing Empirical Symmetry

**Status: Partial**

`results/efficiency/efficiency.csv` covers FM timing at multiple sequence lengths.
K-mer timing is printed to stdout by `train_kmer_rf.py` but not persisted to CSV.

**What remains:**
- Wire k-mer wall-clock + GFLOPS into `results/efficiency/efficiency.csv` via
  `src/training/efficiency.py:log_efficiency()`.
- Produce a unified table with FM and k-mer timing at the same sequence lengths.

**Script:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/utils/benchmark_efficiency.py \
    --seq-lens 100 250 500 1000 2000 --n-seqs 32 \
    --methods fm_NTv3_650M_pre,fm_hyenadna-medium-160k-seqlen-hf,fm_DNABERT-2-117M,kmer_k6
```

---

## LRA Linear Probe — FM Results Missing

**Status: Partial**

`results/regression/lra_records_linear_probe.csv` currently contains k-mer results only.
FM linear probe for the 2 LRA tasks has not been run.

**Script:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/regression/train_lra_linear_probe.py \
    --model InstaDeepAI/NTv3_650M_pre --hg38 /path/to/hg38.fa
```

---

## Operational Constraints

- Server is shared: default `--n-workers 1-2`, check GPU availability before running FM scripts.
- Evo2 is excluded from all experiments.
- All scripts resume from existing records (skip completed rows).
