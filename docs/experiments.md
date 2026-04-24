# Experiment Reference

All scripts are run from the **repository root**. Results are written incrementally;
completed rows are skipped on re-run.

---

## 1. Classification RF

Train Random Forest on 57 genomic classification datasets.

| Script | Result file |
|---|---|
| `src/scripts/classification/train_kmer_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_fm_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_onehot_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_quadtree_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_wavelet_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_multiscale_kmer_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_weighted_multiscale_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_fusion_multik_rf.py` | `results/exploratory/fusion_records.csv` |
| `src/scripts/classification/train_tok_rf.py` | `results/classification/records_rf.csv` |
| `src/scripts/classification/train_pca_rf.py` | `results/classification/records_pca.csv` |

**Run order:** `train_kmer_rf` → `train_fm_rf` (GPU) → remaining (CPU)

---

## 2. Linear Probe

Verify RF results are not driven by non-linearity. Uses LogisticRegressionCV + StandardScaler.

| Script | Result file |
|---|---|
| `src/scripts/classification/train_linear_probe.py` | `results/classification/records_linear_probe.csv` |
| `src/scripts/classification/train_lp_from_cache.py` | `results/classification/records_linear_probe.csv` |
| `src/scripts/regression/train_lra_linear_probe.py` | `results/regression/lra_records_linear_probe.csv` |

```bash
python3 src/scripts/classification/train_linear_probe.py --mode kmer --k-values 4 5 6
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model InstaDeepAI/NTv3_650M_pre --pooling mean
```

---

## 3. Canonical K-mer

Reverse-complement aware k-mer representation. Reduces feature dim (k=6: 4096 → 2080).

| Script | Result file |
|---|---|
| `src/scripts/classification/train_canonical_kmer.py` | `results/classification/records_canonical_kmer.csv` |

```bash
python3 src/scripts/classification/train_canonical_kmer.py --k-values 4 5 6
```

---

## 4. Decomposition

Orthogonal decomposition of FM embeddings: fit Ridge/MLP k-mer→FM, classify proj and residual.

| Script | Result file |
|---|---|
| `src/scripts/decomposition/train_ridge.py` | `results/decomposition/records_decomposition.csv` |
| `src/scripts/decomposition/train_ridge_multiscale.py` | `results/decomposition/records_decomposition.csv` |
| `src/scripts/decomposition/train_decomposition.py` | `results/decomposition/records_decomposition.csv` |

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model InstaDeepAI/NTv3_650M_pre --mapper ridge --k-values 4 5 6
# Non-linear upper bound (MLP):
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model InstaDeepAI/NTv3_650M_pre --mapper mlp --k-values 4 5 6 --mlp-epochs 20
```

---

## 5. LRA Regression Benchmark

Long Range Arena benchmark: variant effect prediction, CAGE, regulatory elements.

| Script | Result file |
|---|---|
| `src/scripts/regression/train_lra_benchmark.py` | `results/regression/lra_records_rf.csv` |
| `src/scripts/regression/train_lra_linear_probe.py` | `results/regression/lra_records_linear_probe.csv` |

```bash
python3 src/scripts/regression/train_lra_benchmark.py \
    --k-values 4 5 6 --hg38 /path/to/hg38.fa
```

---

## 6. Efficiency Benchmark

Wall-clock and GFLOPs comparison: FM vs k-mer at multiple sequence lengths.

| Script | Result file |
|---|---|
| `src/scripts/utils/benchmark_efficiency.py` | `results/efficiency/efficiency.csv` |

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/utils/benchmark_efficiency.py \
    --seq-lens 100 250 500 1000 2000 --n-seqs 32 \
    --methods fm_NTv3_650M_pre,kmer_k6
```

---

## 7. FDR and Statistical Analysis

Post-hoc multiple testing correction (Benjamini-Hochberg) and AUROC/MCC concordance.

| Script | Result file |
|---|---|
| `src/analysis/fdr_analysis.py` | `results/analysis/fdr_results.csv` |
| `src/analysis/auroc_consistency.py` | `results/analysis/auroc_consistency.csv` |

```bash
python3 src/analysis/fdr_analysis.py
python3 src/analysis/auroc_consistency.py
```

---

## 8. Splice Residual Motif Analysis

Attribute NTv3 residual signals to known splice motifs (GT-AG, branch point) on 4 splice datasets.

| Script | Result file |
|---|---|
| `src/scripts/analysis/analyze_splice_residual_motifs.py` | `results/exploratory/splice_residual_motif_overlap.csv` |

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/analysis/analyze_splice_residual_motifs.py \
    --model InstaDeepAI/NTv3_650M_pre --k 6 --n-workers 1 --fm-batch-size 8
```

---

## Supplementary

| Experiment | Script | Result |
|---|---|---|
| Concat best (FM + k-mer) | `src/scripts/utils/benchmark_concat_best.py` | `results/concat/records_concat_best.csv` |
| Truncation analysis | `src/scripts/analysis/truncation_analysis.py` | `results/classification/truncation_analysis.csv` |
| Confusion matrix | `src/scripts/analysis/confusion_matrix_analysis.py` | `results/classification/confusion_matrix_analysis.csv` |
| Mutual information | `src/analysis/compute_mi.py` | `results/exploratory/records_mi.csv` |
| Paper figures | `scripts/export_paper_figures.py` | `results/figures/` |
| Paper stats audit | `scripts/paper_stats_audit.py` | stdout markdown |
