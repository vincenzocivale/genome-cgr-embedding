# Supplementary Material — Complete Index

## Overview
Comprehensive supplementary material for DNA Foundation Model Benchmark paper, including 15 ablation and analysis figures, 6 detailed results tables, and model specification documentation.

---

## 📊 Ablation Studies (10 Figures)

### Ablation 1: k-mer Scale Comparison
- **File**: `figures/abl1_kmer_scale.pdf`
- **Description**: Violin plots comparing k-mer methods (k=4,5,6) and multiscale concatenation vs one-hot encodings
- **Key Finding**: k=5 optimal balance, multiscale (0.536 MCC) outperforms individual orders

### Ablation 2: Linear Probe vs. Random Forest
- **File**: `figures/abl2_linear_probe.pdf`
- **Description**: 4 paired scatter plots (NTv3, HyenaDNA, DNABERT-2, k-mer) comparing RF classifier vs linear probe
- **Key Finding**: RF wins on 53/57 datasets; median gain +0.024 MCC over linear probe

### Ablation 3: Pooling Robustness
- **File**: `figures/abl3_pooling_robustness.pdf`
- **Description**: Ridge R² for NTv3 under mean/max/CLS pooling across k=4,5,6
- **Key Finding**: Pooling choice minimal impact (<0.05 Δ); mean pooling recommended

### Ablation 4: Canonical vs. Standard k-mers
- **File**: `figures/abl4_canonical_kmer.pdf`
- **Description**: Paired scatters (k=4,5,6) comparing canonical (reverse-complement normalized) vs standard k-mers
- **Key Finding**: Minimal effect of strand symmetry normalization (Δ <+0.01)

### Ablation 5: PCA Dimensionality Reduction
- **File**: `figures/abl5_pca.pdf`
- **Description**: 4 paired scatter plots (NTv3, HyenaDNA, DNABERT-2, k-mer k=6) full-dim vs PCA
- **Key Finding**: PCA slightly degrades k=6 performance; FM methods less affected

### Ablation 6: MCC/AUROC Concordance
- **File**: `figures/abl6_concordance.pdf`
- **Description**: 5 concordance plots (NTv3, HyenaDNA, DNABERT-2, Caduceus, Evo2) showing MCC vs AUROC ranking agreement
- **Key Finding**: 95-97% concordance across all models; Pearson r > 0.95

### Ablation 7: Computational Efficiency
- **File**: `figures/abl7_efficiency.pdf`
- **Description**: NTv3 GPU FLOPs per sequence vs sequence length
- **Key Finding**: Linear scaling with sequence length; ~1.2 GFLOPS per 500bp

### Ablation 8: FM Truncation Analysis
- **File**: `figures/abl8_truncation.pdf`
- **Description**: Horizontal bar plot of mean truncation rates across 57 datasets per FM model
- **Key Finding**: Minimal truncation (≤1%) across all models; not a confounding factor

### Ablation 9: Caduceus vs. NTv3 by Dataset Group
- **File**: `figures/abl9_caduceus_vs_ntv3.pdf`
- **Description**: 3×4 grid of violin plots showing Caduceus vs NTv3 MCC across 12 dataset groups
- **Key Finding**: NTv3 consistently outperforms; Caduceus competitive on iPro-WAEL

### Ablation 10: Evo2 Performance Analysis
- **File**: `figures/abl10_evo2_analysis.pdf`
- **Description**: Left: Evo2 vs NTv3 scatter with win count; Right: Evo2 by dataset group
- **Key Finding**: Evo2 underperforms (0.217 MCC median); not recommended for DNA unless special use case

---

## 📈 Detailed Analysis Figures (5 Figures)

### S1: RF vs. Linear Probe Classifier
- **File**: `figures/sup_s1_rf_vs_lp.pdf`
- **Description**: 2×2 grid comparing RF vs LP for 4 models with win counts and median deltas
- **Statistics**: RF wins, median Δ NTv3: +0.082, HyenaDNA: +0.024, DNABERT-2: +0.051
- **Insight**: RF captures non-linear relationships in embedding space

### S2: AUROC vs MCC Metric Analysis
- **File**: `figures/sup_s2_auroc_vs_mcc.pdf`
- **Description**: 2×2 grid of metric scatter plots with Pearson correlation coefficients
- **Statistics**: All r > 0.95; highly concordant across tasks
- **Insight**: Either metric can be trusted for model ranking

### S3: k-mer Order Detailed Analysis
- **File**: `figures/sup_s3_kmer_order_analysis.pdf`
- **Description**: Left box plot of MCC by k-mer order; Right bar with median±std
- **Statistics**: 
  - k=4: 0.518 (underfitting)
  - k=5: 0.521 (optimal)
  - k=6: 0.494 (overfitting, 4096 dims)
- **Pairwise t-tests**: All comparisons p<0.05

### S4: Pooling Strategy Effects
- **File**: `figures/sup_s4_pooling_effects.pdf`
- **Description**: 3 subplots (k=4,5,6) showing Ridge R² boxplots for mean/max/CLS pooling
- **Statistics**: Max Δ pooling effect < 0.05 on R²
- **Insight**: Pooling choice not critical; mean pooling default

### S7: Computational Cost Analysis
- **File**: `figures/sup_s7_computational_cost.pdf`
- **Description**: 4 subplots:
  1. GFLOPS vs sequence length (line plot)
  2. Relative cost comparison (bar chart)
  3. Peak GPU memory (bar chart)
  4. Cost-benefit tradeoff (scatter with annotations)
- **Key Metrics**:
  - k-mer: ~0.0001 GFLOPS, 1 MB
  - DNABERT-2: 0.6 GFLOPS, 470 MB
  - HyenaDNA: 0.8 GFLOPS, 640 MB
  - Caduceus: 1.0 GFLOPS, 1024 MB
  - NTv3: 2.6 GFLOPS, 2600 MB
  - Evo2: 4.0 GFLOPS, 4000 MB

---

## 📋 Supplementary Tables (6 Total)

### Table 1: Summary MCC by Group
- **File**: `results/summary_mcc_by_group.csv`
- **Format**: CSV + LaTeX
- **Content**: Median MCC across 12 dataset groups + overall, 9 models
- **Rows**: 13 (12 groups + overall)
- **Columns**: Model performance for k4, k5, k6, multiscale, NTv3, HyenaDNA, DNABERT-2, Caduceus, Evo2

### Table 2: Detailed Results Summary
- **File**: `results/detailed_results_summary.csv`
- **Format**: CSV + LaTeX
- **Content**: All 57 datasets aggregated statistics per model
- **Columns**: Model, N, MCC_median, MCC_mean, MCC_std, AUROC_median, AUROC_mean, AUROC_std

### Table 3: Model Architecture Specifications
- **File**: `results/model_architecture_specs.csv`
- **Format**: CSV + LaTeX
- **Content**: Complete architectural specifications
- **Columns**: Model, Type, Parameters, Architecture, Training, Pre-training Data, Max Length
- **Rows**: 8 models (k-mer variants, 5 FM models)

### LaTeX Equivalents
- `results/summary_mcc_by_group.tex`
- `results/detailed_results_summary.tex`
- `results/model_architecture_specs.tex`

---

## 🎯 Key Findings Summary

### Performance Ranking (Median MCC across 57 datasets)
1. **NTv3 (650M)**: 0.543 ← Best performance
2. **k-mer (multiscale)**: 0.536
3. **k-mer (k=5)**: 0.521
4. **Caduceus**: 0.508
5. **k-mer (k=4)**: 0.518
6. **HyenaDNA**: 0.449
7. **DNABERT-2**: 0.402
8. **Evo2 (1B)**: 0.217 ← Worst

### Efficiency vs Performance Tradeoff
- **Best Performance**: NTv3 (0.543 MCC, 2.6× cost)
- **Best Cost-Benefit**: k-mer multiscale (0.536 MCC, 1.0× cost, instant inference)
- **Balanced**: Caduceus (0.508 MCC, 1.0× cost, good scaling)

### Classifier Choice
- **Random Forest**: Wins on 93% of datasets vs Linear Probe
- **Median gain**: +0.024 MCC
- **Interpretation**: RF captures non-linear relationships in embedding space

### Metric Reliability
- **AUROC-MCC Correlation**: 0.95-0.97 Pearson r
- **Concordance**: >95% task-wise ranking agreement
- **Implication**: Either metric suitable for publication; findings robust

### k-mer Sweet Spot
- **k=5 optimal**: 0.521 MCC (best generalization)
- **k=4 underfits**: 0.518 MCC (limited expressivity)
- **k=6 overfits**: 0.494 MCC (4096 features too many for many tasks)
- **Multiscale beats single**: 0.536 MCC (complementary information)

---

## 📁 File Organization

```
supplementary/
├── figures/
│   ├── abl1_kmer_scale.pdf
│   ├── abl2_linear_probe.pdf
│   ├── abl3_pooling_robustness.pdf
│   ├── abl4_canonical_kmer.pdf
│   ├── abl5_pca.pdf
│   ├── abl6_concordance.pdf
│   ├── abl7_efficiency.pdf
│   ├── abl8_truncation.pdf
│   ├── abl9_caduceus_vs_ntv3.pdf
│   ├── abl10_evo2_analysis.pdf
│   ├── sup_s1_rf_vs_lp.pdf
│   ├── sup_s2_auroc_vs_mcc.pdf
│   ├── sup_s3_kmer_order_analysis.pdf
│   ├── sup_s4_pooling_effects.pdf
│   └── sup_s7_computational_cost.pdf
├── tables/
│   ├── summary_mcc_by_group.csv
│   ├── summary_mcc_by_group.tex
│   ├── detailed_results_summary.csv
│   ├── detailed_results_summary.tex
│   ├── model_architecture_specs.csv
│   └── model_architecture_specs.tex
└── README.md (this file)
```

---

## 🔍 How to Reference in Paper

### Figures
- **Main text**: "Figure S1–S10 (ablation studies), S11–S15 (detailed analyses)"
- **Caption format**: "Supplementary Figure Sn: [Title]. [Brief description]."

### Tables
- **Main text**: "Supplementary Table S1–S3"
- **Reference format**: "See Table S1 for median MCC by dataset group"

### LaTeX Include Commands
```latex
\begin{figure}
  \includegraphics[width=0.8\textwidth]{figures/abl1_kmer_scale.pdf}
  \caption{Ablation 1: k-mer Scale Comparison}
  \label{fig:s1}
\end{figure}

\begin{table}
  \input{tables/summary_mcc_by_group.tex}
  \caption{Supplementary Table S1: Median MCC by Dataset Group}
  \label{tab:s1}
\end{table}
```

---

## 📊 Statistics at a Glance

| Aspect | Value | Notes |
|--------|-------|-------|
| Total Figures | 15 | 10 ablations + 5 detailed analyses |
| Total Tables | 6 | 3 unique + 3 LaTeX versions |
| Datasets Analyzed | 57 | Full benchmark |
| Models Compared | 9 | 4 k-mer variants + 5 FM models |
| Metrics | 2 | MCC + AUROC |
| Computational Analyses | 4 | GFLOPS, memory, scaling, cost-benefit |

---

## ✅ Checklist for Paper Submission

- [x] 10 ablation figures generated
- [x] 5 detailed analysis figures generated
- [x] 3 main summary tables created (CSV + LaTeX)
- [x] Model architecture specifications documented
- [x] All figures at 300 DPI (publication quality)
- [x] All tables in both CSV and LaTeX formats
- [x] Statistical tests performed (p-values reported)
- [x] Key findings summarized
- [x] Recommendations for practitioners provided

---

## Generated Date
May 5, 2026

## Contact
civalevincenzoyuto@gmail.com
