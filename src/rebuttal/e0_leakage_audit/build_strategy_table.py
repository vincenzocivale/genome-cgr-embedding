"""
Rebuttal E0 — Step 3: dataset x splitting-strategy table + risk ranking.

Merges the sequence-duplication audit (Step 1, classification benchmark) and the
coordinate-overlap audit (Step 2, long-range benchmark) into the single
dataset x splitting-strategy table requested by reviewer e1HN, assigns a
recommended leakage-resistant split per dataset, and selects the high-risk
datasets to re-run in Steps 4-5.

Outputs:
  results/rebuttal/E0_data_leakage/dataset_splitting_strategy.csv
  results/rebuttal/E0_data_leakage/dataset_splitting_strategy.md
  results/rebuttal/E0_data_leakage/high_risk_datasets.txt

Usage:
    python3 src/rebuttal/e0_leakage_audit/build_strategy_table.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

RESULT_DIR = "results/rebuttal/E0_data_leakage"
SEQ_CSV = os.path.join(RESULT_DIR, "sequence_duplication_per_dataset.csv")
COORD_CSV = os.path.join(RESULT_DIR, "coordinate_overlap_per_dataset.csv")
OUT_CSV = os.path.join(RESULT_DIR, "dataset_splitting_strategy.csv")
OUT_MD = os.path.join(RESULT_DIR, "dataset_splitting_strategy.md")
OUT_HIGH = os.path.join(RESULT_DIR, "high_risk_datasets.txt")

# A dataset is re-run (Step 4/5) if its test contamination is at/above this cut.
HIGH_RISK_MIN_PCT = 1.0


def _to_markdown(df):
    """Render a DataFrame as a GitHub-flavoured Markdown table (no deps)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                v = "" if pd.isna(v) else (f"{v:.3f}" if v != int(v) else str(int(v)))
            cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _classification_rows(seq_df):
    rows = []
    for _, r in seq_df.iterrows():
        pct = float(r["contaminated_test_pct"])
        recommended = "dedup-test" if pct > 0 else "keep (as published)"
        rows.append({
            "dataset": r["dataset"],
            "benchmark": "classification (dna_foundation_benchmark)",
            "current_split": "random source split (train.csv / test.csv)",
            "split_axis": "sequence-level (no genomic coordinates)",
            "leakage_metric": "exact/revcomp/near-dup train-test overlap",
            "leakage_pct_before": round(pct, 3),
            "exact_cross_pct": r.get("exact_cross_pct", np.nan),
            "revcomp_cross_pct": r.get("revcomp_cross_pct", np.nan),
            "neardup_cross_pct": r.get("neardup_cross_pct", np.nan),
            "recommended_split": recommended,
            "risk_level": r["risk_level"],
        })
    return rows


def _longrange_rows(coord_df):
    rows = []
    for _, r in coord_df.iterrows():
        # Leakage figure = interval overlap (interval kind) or exact position % (point kind).
        iv = r.get("pct_test_interval_overlap_train", np.nan)
        pe = r.get("pct_test_pos_exact", np.nan)
        pct = iv if not pd.isna(iv) else pe
        pct = float(pct) if not pd.isna(pct) else np.nan
        safe = bool(r.get("leakage_safe", False))
        if r.get("shared_chrom_count", 0) == 0:
            current = "chromosome holdout (disjoint train/test chromosomes)"
        else:
            current = "region/window holdout (shared chromosomes, disjoint windows)"
        risk = "none" if safe and (pct == 0 or pd.isna(pct)) else "low"
        rows.append({
            "dataset": r["dataset"],
            "benchmark": "long-range (genomics-long-range-benchmark)",
            "current_split": current,
            "split_axis": r.get("kind", ""),
            "leakage_metric": "genomic coordinate overlap",
            "leakage_pct_before": round(pct, 4) if not pd.isna(pct) else np.nan,
            "exact_cross_pct": np.nan,
            "revcomp_cross_pct": np.nan,
            "neardup_cross_pct": np.nan,
            "recommended_split": "keep (already leakage-resistant)"
                if safe else "tighten to chromosome holdout",
            "risk_level": risk,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="E0 dataset x strategy table")
    ap.add_argument("--seq", default=SEQ_CSV)
    ap.add_argument("--coord", default=COORD_CSV)
    ap.add_argument("--min-pct", type=float, default=HIGH_RISK_MIN_PCT)
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.seq):
        rows += _classification_rows(pd.read_csv(args.seq))
    else:
        print(f"WARNING: {args.seq} not found (run Step 1 first)")
    if os.path.exists(args.coord):
        rows += _longrange_rows(pd.read_csv(args.coord))
    else:
        print(f"WARNING: {args.coord} not found (run Step 2 first)")

    df = pd.DataFrame(rows)
    df = df.sort_values(["benchmark", "leakage_pct_before"], ascending=[True, False])
    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # High-risk = classification datasets with contamination >= min-pct.
    high = df[(df["benchmark"].str.startswith("classification")) &
              (df["leakage_pct_before"] >= args.min_pct)].copy()
    high = high.sort_values("leakage_pct_before", ascending=False)
    high["dataset"].to_csv(OUT_HIGH, index=False, header=False)

    # Markdown table (compact columns) — manual writer (no tabulate dependency).
    md_cols = ["dataset", "benchmark", "current_split", "leakage_metric",
               "leakage_pct_before", "recommended_split", "risk_level"]
    with open(OUT_MD, "w") as f:
        f.write("# Dataset x splitting-strategy table (Rebuttal E0)\n\n")
        f.write(f"High-risk threshold: contaminated test >= {args.min_pct}% "
                f"(classification benchmark).\n\n")
        f.write(_to_markdown(df[md_cols]))
        f.write("\n")

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_HIGH}  ({len(high)} high-risk datasets)")
    print("\nHigh-risk datasets (contamination %):")
    print(high[["dataset", "leakage_pct_before", "exact_cross_pct",
                "neardup_cross_pct", "risk_level"]].to_string(index=False))


if __name__ == "__main__":
    main()
