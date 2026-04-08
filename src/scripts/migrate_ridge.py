"""
One-shot migration: move ridge columns from records.csv to records_decomposition.csv.

- Copies ridge_k{K}_* columns (single k only, excludes ridge_multi_*)
- Removes ALL ridge_* columns (including multi) from records.csv
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RECORDS_CSV = "results/classification/records.csv"
DECOMP_CSV = "results/classification/records_decomposition.csv"


def main():
    if not os.path.exists(RECORDS_CSV):
        print(f"File not found: {RECORDS_CSV}")
        return

    df = pd.read_csv(RECORDS_CSV, index_col=0)
    df.index.name = "dataset"

    all_ridge = [c for c in df.columns if c.startswith("ridge_")]
    single_k_ridge = [c for c in all_ridge if not c.startswith("ridge_multi_")]
    multi_ridge = [c for c in all_ridge if c.startswith("ridge_multi_")]

    if not all_ridge:
        print("No ridge columns found in records.csv — nothing to migrate.")
        return

    print(f"Found {len(all_ridge)} ridge columns in records.csv:")
    print(f"  Single-k (to migrate): {len(single_k_ridge)}")
    for c in single_k_ridge:
        n = df[c].notna().sum()
        print(f"    {c}: {n}/{len(df)} values")
    print(f"  Multi-scale (to discard): {len(multi_ridge)}")
    for c in multi_ridge:
        n = df[c].notna().sum()
        print(f"    {c}: {n}/{len(df)} values")

    # Create decomposition CSV with single-k ridge columns
    decomp_df = df[single_k_ridge].copy()
    decomp_df.index.name = "dataset"
    os.makedirs(os.path.dirname(DECOMP_CSV), exist_ok=True)
    decomp_df.to_csv(DECOMP_CSV, index=True)
    print(f"\nMigrated {len(single_k_ridge)} columns to {DECOMP_CSV}")

    # Remove ALL ridge columns from records.csv
    df = df.drop(columns=all_ridge)
    df.to_csv(RECORDS_CSV, index=True)
    print(f"Removed {len(all_ridge)} ridge columns from {RECORDS_CSV}")
    print(f"records.csv now has {len(df.columns)} columns")


if __name__ == "__main__":
    main()
