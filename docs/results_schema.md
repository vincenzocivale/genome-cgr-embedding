# Results Schema

All retained result files are CSV unless otherwise noted.

## `results/classification/records_rf.csv`

Wide table with one row per dataset.

- `kmer_k{K}_{metric}`
- `kmer_multi_k{4_5_6}_{metric}`
- `wms_k{...}_{weighting}_{metric}`
- `qt_d{depth}_p{threshold}_m{min_count}_{metric}`
- `wavelet_l{L}_{metric}`
- `onehot_{W}bp_{metric}`
- `fm_{model_tag}_{metric}`
- `tok_{model_tag}_{metric}`
- dataset metadata: `n_train`, `n_test`, `n_classes`, `class_balance`, `seq_len_mean`, `seq_len_std`, `seq_len_min`, `seq_len_max`, `gc_content_mean`

`metric ∈ {MCC, AUROC, F1, Accuracy}`.

## `results/classification/records_linear_probe.csv`

Wide linear-probe table with one row per dataset.

- `lp_kmer_k{K}_{metric}`
- `lp_fm_{model_tag}_{metric}`
- non-default pooling uses `__pool_{pooling}`, for example `lp_fm_NTv3_650M_pre_MCC__pool_max`

## `results/classification/records_canonical_kmer.csv`

- `ckmer_k{K}_{metric}`
- `ckmer_lp_k{K}_{metric}`

## `results/classification/records_pca.csv`

- `pca_kmer_k{K}_{metric}`
- `pca_fm_{model_tag}_{metric}`

## `results/decomposition/records_decomposition.csv`

Wide decomposition table with one row per dataset.

- `ridge_k{K}_{model_tag}_R2`
- `proj_k{K}_{model_tag}_{metric}`
- `resid_k{K}_{model_tag}_{metric}`
- optional non-default suffix: `__pool_{pooling}__map_{mapper}`

The canonical retained decomposition metrics are `R2`, `MCC` and `AUROC`.

## `results/analysis/fdr_results.csv`

- `probe_type`
- `metric`
- `comparison`
- `column`
- `n_datasets`
- `mean_delta`
- `median_delta`
- `kmer_wins`
- `fm_wins`
- `raw_p`
- `adjusted_p`
- `significant`

## `results/analysis/auroc_consistency.csv`

- `probe_type`
- `fm_model`
- `n_datasets`
- `concordance_rate`
- `n_concordant`
- `n_discordant`
- `kmer_wins_mcc`
- `kmer_wins_auroc`
- `discordant_datasets`

## `results/concat/records_concat_best.csv`

One row per dataset for the concatenation of the best FM representation and the
best retained k-mer representation.

- `dataset`
- `metric_select`
- `kmer_choice`
- `kmer_col`
- `fm_model`
- `fm_col`
- `feat_dim_fm`
- `feat_dim_kmer`
- `MCC`
- `AUROC`
- `F1`
- `Accuracy`

## `results/classification/truncation_analysis.csv`

One row per `(model, dataset, split)`.

- `model`
- `dataset`
- `split`
- `n_sequences`
- `n_truncated`
- `truncation_rate`
- `mean_tokens`
- `max_tokens`
- `fraction_retained_mean`
- `max_length`

## `results/exploratory/splice_residual_motif_overlap.csv`

One row per `(dataset, motif)`.

- `dataset`
- `motif`
- `n_matched`
- `n_total_test`
- `overlap_rate`
- `mean_delta_resid_l2`
- `std_delta_resid_l2`

## `results/efficiency/efficiency.csv`

Append-only timing log from training scripts.

- `timestamp`
- `dataset`
- `method`
- `feat_dim`
- `time_sec`

## `results/efficiency/efficiency_gpu_parallel.csv`

Synthetic sequence-length benchmark used for the cost-scaling figure.

- `method`
- `seq_len`
- `n_seqs`
- `params_M`
- `duration_sec`
- `GFLOPS_per_seq`
- `GFLOPS_total`
- `GFLOPS_theory_per_seq`

## `results/figures/`

Publication-ready PDF figures exported by `scripts/export_paper_figures.py`.
