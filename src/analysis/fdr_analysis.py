"""
FDR (Benjamini-Hochberg) correction on kmer vs FM pairwise comparisons.

For each FM model, performs a Wilcoxon signed-rank test comparing best k-mer MCC
to FM MCC across all 57 classification datasets, then applies BH correction.

Runs on both records.csv (RF) and records_linear_probe.csv (linear probe).

Usage:
    python3 src/analysis/fdr_analysis.py
    python3 src/analysis/fdr_analysis.py --records results/classification/records.csv
    python3 src/analysis/fdr_analysis.py --records results/classification/records_linear_probe.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

FDR_RESULTS_CSV = "results/analysis/fdr_results.csv"


def _get_kmer_cols(df: pd.DataFrame, metric: str, prefix: str = "kmer") -> list[str]:
    """Find kmer columns for a given metric."""
    return [c for c in df.columns if c.startswith(f"{prefix}_k") and c.endswith(f"_{metric}")]


def _best_kmer(df: pd.DataFrame, metric: str, prefix: str = "kmer") -> pd.Series:
    """Per-dataset best k-mer metric (max across k values)."""
    cols = _get_kmer_cols(df, metric, prefix)
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].max(axis=1)


def _detect_fm_configs(df: pd.DataFrame, metric: str) -> list[tuple[str, str]]:
    """
    Detect all FM configurations in the dataframe.
    Returns list of (display_name, column_name).
    Handles both standard columns (ending _MCC) and pool-suffixed (ending _MCC__pool_X).
    """
    fm_configs = []
    seen = set()
    for col in df.columns:
        for prefix in ("lp_fm_", "fm_"):
            if not col.startswith(prefix):
                continue
            rest = col[len(prefix):]  # e.g. "NTv3_650M_pre_MCC__pool_max"
            # Standard: ends with _{metric}
            suffix = f"_{metric}"
            if rest.endswith(suffix):
                tag = rest[:-len(suffix)]
                if tag not in seen:
                    seen.add(tag)
                    fm_configs.append((tag, col))
            else:
                # Pool-suffixed: contains _{metric}__pool_
                pool_suffix = f"_{metric}__pool_"
                idx = rest.find(pool_suffix)
                if idx != -1:
                    tag = rest[:idx] + rest[idx + len(f"_{metric}"):]  # include __pool_X
                    if tag not in seen:
                        seen.add(tag)
                        fm_configs.append((tag, col))
    return fm_configs


def run_fdr_analysis(
    records_path: str,
    metric: str = "MCC",
    kmer_prefix: str = "kmer",
    output_path: str = FDR_RESULTS_CSV,
) -> pd.DataFrame:
    """
    Run FDR analysis for all FM models vs best k-mer.

    Returns DataFrame with columns:
        comparison, n_datasets, mean_delta, median_delta,
        kmer_wins, fm_wins, raw_p, adjusted_p, significant
    """
    if not os.path.exists(records_path):
        print(f"  Records not found: {records_path}")
        return pd.DataFrame()

    df = pd.read_csv(records_path, index_col=0)
    print(f"\n  Loaded {len(df)} datasets from {records_path}")

    best_kmer = _best_kmer(df, metric, kmer_prefix)
    if best_kmer.empty:
        print(f"  No k-mer columns found for metric={metric}")
        return pd.DataFrame()

    fm_configs = _detect_fm_configs(df, metric)
    if not fm_configs:
        print(f"  No FM columns found for metric={metric}")
        return pd.DataFrame()

    print(f"  FM configs found: {[tag for tag, _ in fm_configs]}")

    rows = []
    raw_pvals = []
    comparisons = []

    for tag, col in fm_configs:
        if col not in df.columns:
            continue
        fm_vals = df[col].dropna()
        common_idx = best_kmer.index.intersection(fm_vals.index)
        if len(common_idx) < 5:
            print(f"  Skipping {tag}: only {len(common_idx)} paired datasets")
            continue

        kmer_v = best_kmer.loc[common_idx].values
        fm_v = fm_vals.loc[common_idx].values
        delta = kmer_v - fm_v  # positive = kmer wins

        n = len(common_idx)
        mean_delta = float(np.mean(delta))
        median_delta = float(np.median(delta))
        kmer_wins = int(np.sum(delta > 0))
        fm_wins = int(np.sum(delta < 0))

        # Wilcoxon signed-rank test (two-sided)
        try:
            stat, p = wilcoxon(kmer_v, fm_v, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p = 1.0  # all differences zero

        comparisons.append(tag)
        raw_pvals.append(p)
        rows.append({
            "comparison": tag,
            "column": col,
            "n_datasets": n,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "kmer_wins": kmer_wins,
            "fm_wins": fm_wins,
            "raw_p": p,
        })

    if not rows:
        print("  No valid comparisons found.")
        return pd.DataFrame()

    # BH correction
    if len(raw_pvals) > 1:
        adjusted = false_discovery_control(raw_pvals, method="bh")
    else:
        adjusted = raw_pvals.copy()

    for row, adj_p in zip(rows, adjusted):
        row["adjusted_p"] = adj_p
        row["significant"] = bool(adj_p < 0.05)

    result = pd.DataFrame(rows)
    return result


def main():
    parser = argparse.ArgumentParser(description="FDR analysis: kmer vs FM comparisons")
    parser.add_argument(
        "--records",
        nargs="+",
        default=[
            "results/classification/records.csv",
            "results/classification/records_linear_probe.csv",
        ],
    )
    parser.add_argument("--metrics", nargs="+", default=["MCC", "AUROC"])
    parser.add_argument("--output", default=FDR_RESULTS_CSV)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results = []

    for records_path in args.records:
        probe_type = "linear_probe" if "linear_probe" in records_path else "rf"
        kmer_prefix = "lp_kmer" if probe_type == "linear_probe" else "kmer"

        print(f"\n{'='*60}")
        print(f"Analysis: {probe_type} | {records_path}")
        print(f"{'='*60}")

        for metric in args.metrics:
            print(f"\n  Metric: {metric}")
            result = run_fdr_analysis(records_path, metric=metric, kmer_prefix=kmer_prefix)
            if not result.empty:
                result.insert(0, "metric", metric)
                result.insert(0, "probe_type", probe_type)
                all_results.append(result)

    if not all_results:
        print("\nNo results generated.")
        return

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(args.output, index=False)
    print(f"\n\nFDR results saved to: {args.output}")
    print(f"\nSummary:")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
