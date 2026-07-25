# E5 — Pooling alternatives

The runner evaluates mean, max, mean+max, CLS (where a true special token is
available), and parameter-free content-attention pooling.  The latter uses the
masked sequence mean as its query, so it introduces no labels or additional
training budget.  `pooling_by_task_group.csv` specifically reports promoter,
splice-site and regulatory subsets; `best_pooling_by_task.csv` reports the
best pooling per individual dataset rather than averaged per model.
`pooling_significance.csv` / `pooling_significance_by_task_group.csv` test
each alternative pooling against mean with a paired Wilcoxon signed-rank test
(BH-corrected across all rows, same convention as `src/analysis/fdr_analysis.py`),
so "pooling X beats mean" is a statistical claim, not just a higher average.
Non-applicable CLS cells (char-level / autoregressive models) are logged
rather than silently substituted.

Models covered: NTv3, DNABERT-2, HyenaDNA. **Evo2 is not covered** — the
`evo2` Python package (StripedHyena/Vortex, custom CUDA kernels) is not
installed in any conda environment on this machine, so `Evo2Embedder` cannot
be instantiated to compute the non-mean poolings it needs. The pooling
extension for Evo2 was still implemented in `src/embedders/evo2_embedder.py`
(reusing `FMEmbedder.pool_hidden`) and validated on synthetic tensors; once
the package is installed, re-running `run_pooling_comparison.py --models
evo2_1b_base` will pick up where it left off (cache/results are resumable).

A bug was fixed in `src/embedders/embedding_cache.py` and
`src/rebuttal/common.py`: the pre-rebuttal "legacy" embedding cache (no
`pooling_*` subdirectory, mean-pooled only) was being used as a fallback for
*any* requested pooling, which would have silently returned mean-pooled
embeddings mislabeled as `mean_max`/`attention` for NTv3 and Evo2. The
fallback is now gated on `pooling == "mean"`.
