"""
Rebuttal E2 — Canonical vs. standard (strand-specific) k-mers.

Reviewer HnDT asks whether the k-mer baseline's results depend on a biologically
inappropriate strand-specific encoding. `records_canonical_kmer.csv` already has
canonical (reverse-complement-aware) k-mer performance for all 57 datasets
(`train_canonical_kmer.py`, unchanged); this script does NOT retrain anything, it
adds the strand-invariance breakdown that was missing: canonicalization is only
expected to help (or be neutral) where task labels don't depend on strand
orientation (see task_strand_classification.py), and should not be judged on the
full 57-task pool alone.

Output:
    results/rebuttal/E2_canonical_kmer/canonical_vs_standard.csv   (per-task, per-k)
    results/rebuttal/E2_canonical_kmer/canonical_wilcoxon.csv      (paired tests per subset)
    results/rebuttal/E2_canonical_kmer/dimensionality.csv
    results/rebuttal/E2_canonical_kmer/E2_summary.md

Usage:
    python3 src/rebuttal/e2_canonical_kmer/analyze_canonical_vs_standard.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.analysis.paper_utils import infer_biocat, CATEGORY_LABELS
from src.features.kmer_features import _build_canonical_map
from src.records.records import RECORDS_CSV, RECORDS_CANONICAL_CSV
from src.rebuttal.e2_canonical_kmer.task_strand_classification import infer_strand_invariance

OUT_DIR = "results/rebuttal/E2_canonical_kmer"
K_VALUES = [4, 5, 6]
METRICS = ["MCC", "AUROC"]
SUBSETS = ["all", "strand_invariant", "strand_specific"]


def build_per_task_table() -> pd.DataFrame:
    rf = pd.read_csv(RECORDS_CSV, index_col=0)
    rf.index.name = "dataset"
    canon = pd.read_csv(RECORDS_CANONICAL_CSV, index_col=0)
    canon.index.name = "dataset"

    out = pd.DataFrame(index=rf.index)
    out["biocat"] = out.index.to_series().map(infer_biocat)
    out["biocat_label"] = out["biocat"].map(CATEGORY_LABELS)
    out["strand_class"] = out.index.to_series().map(infer_strand_invariance)

    for k in K_VALUES:
        for metric in METRICS:
            std_col = f"kmer_k{k}_{metric}"
            can_col = f"ckmer_k{k}_{metric}"
            out[std_col] = rf.get(std_col)
            out[can_col] = canon.get(can_col)
            out[f"delta_canon_minus_std_k{k}_{metric}"] = out[can_col] - out[std_col]

    out_path = os.path.join(OUT_DIR, "canonical_vs_standard.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(out_path)
    print(f"Wrote {out_path}")
    return out


def paired_wilcoxon_by_subset(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        for metric in METRICS:
            std_col = f"kmer_k{k}_{metric}"
            can_col = f"ckmer_k{k}_{metric}"
            for subset in SUBSETS:
                if subset == "all":
                    sub = table
                else:
                    sub = table[table["strand_class"] == subset]
                valid = sub[[std_col, can_col]].dropna()
                if len(valid) < 5:
                    continue
                std_v = valid[std_col].values
                can_v = valid[can_col].values
                delta = can_v - std_v  # positive = canonical wins
                try:
                    _, p = wilcoxon(can_v, std_v, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    p = 1.0
                rows.append({
                    "k": k, "metric": metric, "subset": subset, "n_tasks": len(valid),
                    "mean_delta_canon_minus_std": float(np.mean(delta)),
                    "median_delta_canon_minus_std": float(np.median(delta)),
                    "canonical_wins": int(np.sum(delta > 0)),
                    "standard_wins": int(np.sum(delta < 0)),
                    "raw_p": p,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["adjusted_p"] = false_discovery_control(df["raw_p"].values, method="bh")
        df["significant"] = df["adjusted_p"] < 0.05
    out_path = os.path.join(OUT_DIR, "canonical_wilcoxon.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def dimensionality_table() -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        _, n_canon = _build_canonical_map(k)
        rows.append({
            "k": k, "n_standard_kmers": 4 ** k, "n_canonical_kmers": n_canon,
            "reduction_pct": round(100.0 * (1 - n_canon / 4 ** k), 1),
        })
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "dimensionality.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def write_summary(table: pd.DataFrame, wilcoxon_df: pd.DataFrame, dim_df: pd.DataFrame):
    lines = ["# E2 — Canonical vs. standard k-mers: summary\n"]
    lines.append(
        "Success criterion: the paper's k-mer-vs-FM result does not depend on a "
        "biologically inappropriate strand-specific encoding — canonicalization "
        "should not systematically help on strand_specific tasks, and any effect "
        "on strand_invariant tasks should be small, consistent with the existing "
        "global Ablation 4 finding (Δ<0.01).\n"
    )
    lines.append("## Task counts by strand class\n")
    lines.append("```\n" + table["strand_class"].value_counts().to_string() + "\n```\n")

    lines.append("## Dimensionality (4^k vs canonical groups)\n")
    lines.append("```\n" + dim_df.to_string(index=False) + "\n```\n")

    lines.append("## Paired Wilcoxon (canonical vs standard), MCC, by subset\n")
    sub = wilcoxon_df[wilcoxon_df["metric"] == "MCC"]
    if not sub.empty:
        cols = ["k", "subset", "n_tasks", "canonical_wins", "standard_wins",
                "mean_delta_canon_minus_std", "median_delta_canon_minus_std",
                "adjusted_p", "significant"]
        lines.append("```\n" + sub[cols].to_string(index=False) + "\n```\n")

    ambiguous = table[table["strand_class"] == "ambiguous"]
    lines.append("## Ambiguous tasks (excluded from strand_invariant/strand_specific, reported separately)\n")
    lines.append("```\n" + ambiguous.index.to_series().to_string(index=False) + "\n```\n")

    out_path = os.path.join(OUT_DIR, "E2_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    table = build_per_task_table()
    wilcoxon_df = paired_wilcoxon_by_subset(table)
    dim_df = dimensionality_table()
    write_summary(table, wilcoxon_df, dim_df)


if __name__ == "__main__":
    main()
