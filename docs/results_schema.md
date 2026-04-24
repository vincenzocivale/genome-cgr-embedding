# Results File Schema

All result files are CSV, one row per dataset (row key = `dataset` column), wide format.  
Paths are relative to the repository root.

---

## `results/classification/records_rf.csv`

Primary RF benchmark. One row per dataset (57 rows).

| Column pattern | Description |
|---|---|
| `kmer_k{K}_{metric}` | RF on standard k-mer; K ∈ {4,5,6,7}; metric ∈ {MCC, AUROC, F1, Accuracy} |
| `fm_{model_tag}_{metric}` | RF on FM mean-pool embedding |
| `tok_{model_tag}_{metric}` | RF on tokenizer (pre-transformer) embedding |
| `onehot_{W}bp_{metric}` | RF on one-hot encoded central W bp window |
| `quadtree_{depth}_{p}_{min}_{metric}` | RF on adaptive QuadTree CGR features |
| `wavelet_l{L}_{metric}` | RF on spatial pyramid pooling features |
| `multiscale_k{K}_{metric}` | RF on concatenated multiscale k-mer (k=4,5,6) |
| `n_train`, `n_test`, `n_classes`, … | Dataset metadata |

`model_tag` = last component of the HuggingFace ID (e.g. `NTv3_650M_pre`).

---

## `results/classification/records_linear_probe.csv`

Linear probe (LogisticRegressionCV + StandardScaler). Same column naming as `records_rf.csv`,
prefixed with `lp_`: `lp_kmer_k{K}_{metric}`, `lp_fm_{model_tag}_{metric}`.

Pooling variants use suffix `__pool_{pooling}`, e.g. `lp_fm_NTv3_650M_pre_MCC__pool_max`.

---

## `results/classification/records_canonical_kmer.csv`

Canonical (reverse-complement aware) k-mer. Reduces feature dim: k=4→136, k=5→512, k=6→2080.

| Column pattern | Description |
|---|---|
| `ckmer_k{K}_{metric}` | RF on canonical k-mer |
| `ckmer_lp_k{K}_{metric}` | Linear probe on canonical k-mer |

---

## `results/classification/records_pca.csv`

PCA (95% variance) applied before RF/linear probe.

| Column pattern | Description |
|---|---|
| `pca_kmer_k{K}_{metric}` | PCA + RF on k-mer |
| `pca_fm_{model_tag}_{metric}` | PCA + RF on FM embedding |

---

## `results/decomposition/records_decomposition.csv`

Orthogonal decomposition of FM embeddings.

| Column pattern | Description |
|---|---|
| `ridge_k{K}_{model_tag}_R2` / `_MSE` | Ridge regression R² of k-mer → FM |
| `proj_k{K}_{model_tag}_{metric}` | RF on projected (k-mer-explainable) component |
| `resid_k{K}_{model_tag}_{metric}` | RF on residual (k-mer-unexplained) component |

Non-default pooling or MLP mapper appended as `__pool_{pooling}__map_{mapper}`.

---

## `results/regression/lra_records_rf.csv`

LRA benchmark RF results. One row per LRA task.

| Column pattern | Description |
|---|---|
| `kmer_k{K}_R2`, `kmer_k{K}_Spearman` | k-mer RF on regression tasks |
| `kmer_k{K}_MCC`, `kmer_k{K}_AUROC` | k-mer RF on classification tasks |
| `fm_{model_tag}_R2`, `fm_{model_tag}_Spearman` | FM RF on regression tasks |

---

## `results/regression/lra_records_linear_probe.csv`

Same structure as `lra_records_rf.csv`, prefixed with `lp_`.  
Currently contains k-mer results only (FM results pending — see `docs/missing_experiments.md`).

---

## `results/analysis/fdr_results.csv`

Benjamini-Hochberg FDR correction on Wilcoxon signed-rank tests (57 datasets).

| Column | Description |
|---|---|
| `comparison` | FM model vs k-mer comparison |
| `n_datasets` | Number of datasets used |
| `mean_delta`, `median_delta` | Mean/median metric difference |
| `kmer_wins`, `fm_wins` | Count of datasets where each method wins |
| `raw_p` | Wilcoxon p-value |
| `adjusted_p` | BH-corrected p-value |
| `significant` | Boolean |

---

## `results/analysis/auroc_consistency.csv`

Concordance between MCC and AUROC verdicts (which method wins).

| Column | Description |
|---|---|
| `fm_model` | Foundation model |
| `concordance_rate` | Fraction of datasets where MCC and AUROC agree |
| `n_concordant`, `n_discordant` | Counts |
| `kmer_wins_mcc`, `kmer_wins_auroc` | Win counts per metric |

---

## `results/efficiency/efficiency.csv`

Efficiency logging via `src/training/efficiency.py:log_efficiency()`.  
Appended incrementally.

| Column | Description |
|---|---|
| `timestamp` | UTC ISO timestamp |
| `dataset` | Dataset name |
| `method` | Feature extraction method |
| `feat_dim` | Feature dimensionality |
| `time_sec` | Wall-clock seconds |

---

## `results/efficiency/efficiency_gpu_parallel.csv`

Broader efficiency sweep at multiple sequence lengths.

| Column | Description |
|---|---|
| `method` | Method identifier |
| `seq_len` | Input sequence length |
| `n_seqs` | Number of sequences timed |
| `params_M` | Model parameters (millions) |
| `duration_sec` | Wall-clock time |
| `GFLOPS_per_seq` | Measured GFLOPs per sequence |
| `GFLOPS_total` | Total GFLOPs for batch |

---

## `results/efficiency/records_efficiency.csv`

Per-dataset extraction timing summary used by `scripts/paper_stats_audit.py`.

| Column | Description |
|---|---|
| `method`, `seq_len`, `params_M` | Method metadata |
| `GFLOPS_per_seq`, `GFLOPS_total` | Compute cost |
| `AUROC` | Downstream task performance |
| `emissions`, `duration_sec` | Energy and time |

---

## `results/exploratory/splice_residual_motif_overlap.csv`

Splice motif attribution on 4 splice datasets (NTv3 residual component).

| Column | Description |
|---|---|
| `dataset` | Splice dataset name |
| `motif` | Motif pattern (e.g. GT-AG, branch point) |
| `n_matches` | Number of motif occurrences in test set |
| `mean_residual_norm_drop` | Average residual norm drop after motif perturbation |
| `overlap_rate` | Fraction of sequences with this motif |

---

## `results/concat/records_concat_best.csv`

Best k-mer + FM concatenation RF results. Columns mirror `records_rf.csv`.

---

## `results/classification/truncation_analysis.csv`

| Column | Description |
|---|---|
| `dataset`, `model`, `split` | Identifiers |
| `n_sequences` | Total sequences |
| `n_truncated` | Sequences exceeding model max tokens |
| `truncation_rate` | Fraction truncated |
| `mean_tokens`, `max_tokens` | Token length statistics |

---

## `results/archive/`

Run logs from earlier experiments. Not used by any downstream analysis.
