"""
Rebuttal E3 (part 1/2) — Multi-k downstream baseline vs. best single k.

Indirectly requested by eZYW: does concatenating k-mer scales
[x_k4; x_k5; x_k6] change the k-mer-vs-FM picture? The multiscale downstream
classifier already exists (`kmer_multi_k4_5_6_*` in records_rf.csv, produced by
train_multiscale_kmer_rf.py — this is the paper's existing Ablation 1). This
script does NOT retrain anything; it just re-frames that existing result as a
paired comparison against best single-k and against each FM model, matching the
E1/E2 reporting format.

For the decomposition half (does the FM residual stay informative against a
multi-k baseline), see train_decomposition_multik.py.

Output:
    results/rebuttal/E3_multiscale/multik_vs_bestk_baseline.csv
    results/rebuttal/E3_multiscale/multik_vs_fm.csv

Usage:
    python3 src/rebuttal/e3_multiscale/analyze_multik_baseline.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.analysis.fdr_analysis import _detect_fm_configs
from src.records.records import RECORDS_CSV, best_kmer_metric

OUT_DIR = "results/rebuttal/E3_multiscale"
METRICS = ["MCC", "AUROC"]


def multik_vs_bestk():
    df = pd.read_csv(RECORDS_CSV, index_col=0)
    rows = []
    for metric in METRICS:
        multi_col = f"kmer_multi_k4_5_6_{metric}"
        if multi_col not in df.columns:
            continue
        best_single = best_kmer_metric(df, metric=metric, include_multiscale=False)
        valid_idx = df[multi_col].dropna().index.intersection(best_single.dropna().index)
        multi_v = df.loc[valid_idx, multi_col].values
        single_v = best_single.loc[valid_idx].values
        delta = multi_v - single_v
        try:
            _, p = wilcoxon(multi_v, single_v, alternative="two-sided", zero_method="wilcox")
        except ValueError:
            p = 1.0
        rows.append({
            "metric": metric, "n_tasks": len(valid_idx),
            "multi_median": float(np.median(multi_v)), "best_single_k_median": float(np.median(single_v)),
            "mean_delta_multi_minus_bestk": float(np.mean(delta)),
            "median_delta_multi_minus_bestk": float(np.median(delta)),
            "multi_wins": int(np.sum(delta > 0)), "single_k_wins": int(np.sum(delta < 0)),
            "raw_p": p,
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_p"] = false_discovery_control(result["raw_p"].values, method="bh") if len(result) > 1 else result["raw_p"]
        result["significant"] = result["adjusted_p"] < 0.05
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "multik_vs_bestk_baseline.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(result.to_string(index=False))
    return result


def multik_vs_fm():
    df = pd.read_csv(RECORDS_CSV, index_col=0)
    rows, raw_p = [], []
    for metric in METRICS:
        multi_col = f"kmer_multi_k4_5_6_{metric}"
        if multi_col not in df.columns:
            continue
        multi_vals = df[multi_col].dropna()
        for tag, col in _detect_fm_configs(df, metric):
            fm_vals = df[col].dropna()
            common = multi_vals.index.intersection(fm_vals.index)
            if len(common) < 5:
                continue
            multi_v = multi_vals.loc[common].values
            fm_v = fm_vals.loc[common].values
            delta = multi_v - fm_v
            try:
                _, p = wilcoxon(multi_v, fm_v, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                p = 1.0
            rows.append({
                "metric": metric, "comparison": tag, "n_tasks": len(common),
                "mean_delta_multi_minus_fm": float(np.mean(delta)),
                "median_delta_multi_minus_fm": float(np.median(delta)),
                "multi_wins": int(np.sum(delta > 0)), "fm_wins": int(np.sum(delta < 0)),
                "raw_p": p,
            })
            raw_p.append(p)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_p"] = false_discovery_control(raw_p, method="bh") if len(raw_p) > 1 else raw_p
        result["significant"] = result["adjusted_p"] < 0.05
    out_path = os.path.join(OUT_DIR, "multik_vs_fm.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(result.to_string(index=False))
    return result


def main():
    multik_vs_bestk()
    multik_vs_fm()


if __name__ == "__main__":
    main()
