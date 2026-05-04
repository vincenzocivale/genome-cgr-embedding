# Experiment Reference

All commands are run from the repository root.

## Classification benchmarks

| Experiment | Script | Canonical output |
| --- | --- | --- |
| Fixed-resolution k-mer RF | `src/scripts/classification/train_kmer_rf.py` | `results/classification/records_rf.csv` |
| FM RF | `src/scripts/classification/train_fm_rf.py` | `results/classification/records_rf.csv` |
| Token-embedding RF | `src/scripts/classification/train_tok_rf.py` | `results/classification/records_rf.csv` |
| One-hot RF | `src/scripts/classification/train_onehot_rf.py` | `results/classification/records_rf.csv` |
| Multiscale k-mer RF | `src/scripts/classification/train_multiscale_kmer_rf.py` | `results/classification/records_rf.csv` |
| Weighted multiscale RF | `src/scripts/classification/train_weighted_multiscale_rf.py` | `results/classification/records_rf.csv` |
| QuadTree RF | `src/scripts/classification/train_quadtree_rf.py` | `results/classification/records_rf.csv` |
| Wavelet RF | `src/scripts/classification/train_wavelet_rf.py` | `results/classification/records_rf.csv` |

## Validation and ablations

| Experiment | Script | Canonical output |
| --- | --- | --- |
| Linear probe | `src/scripts/classification/train_linear_probe.py` | `results/classification/records_linear_probe.csv` |
| Canonical reverse-complement k-mer | `src/scripts/classification/train_canonical_kmer.py` | `results/classification/records_canonical_kmer.csv` |
| PCA + RF | `src/scripts/classification/train_pca_rf.py` | `results/classification/records_pca.csv` |

## Decomposition and supplementary analyses

| Experiment | Script | Canonical output |
| --- | --- | --- |
| Decomposition | `src/scripts/decomposition/train_decomposition.py` | `results/decomposition/records_decomposition.csv` |
| FM + best k-mer concatenation | `src/scripts/utils/benchmark_concat_best.py` | `results/concat/records_concat_best.csv` |
| Efficiency benchmark | `src/scripts/utils/benchmark_efficiency.py` | `results/efficiency/efficiency_gpu_parallel.csv` |
| Truncation audit | `src/scripts/analysis/truncation_analysis.py` | `results/classification/truncation_analysis.csv` |
| Splice residual motifs | `src/scripts/analysis/analyze_splice_residual_motifs.py` | `results/exploratory/splice_residual_motif_overlap.csv` |

## Statistical post-processing

| Analysis | Script | Canonical output |
| --- | --- | --- |
| FDR correction | `src/analysis/fdr_analysis.py` | `results/analysis/fdr_results.csv` |
| MCC/AUROC consistency | `src/analysis/auroc_consistency.py` | `results/analysis/auroc_consistency.csv` |
| Dataset metadata fill | `src/analysis/fill_dataset_info.py` | `results/classification/records_rf.csv` |

## Publication assets

| Output | Script |
| --- | --- |
| Figures | `scripts/export_paper_figures.py` |
| Paper statistics audit | `scripts/paper_stats_audit.py` |
