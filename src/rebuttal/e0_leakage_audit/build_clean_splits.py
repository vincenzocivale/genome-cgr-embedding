"""
Rebuttal E0 — Step 4: build de-contaminated (leakage-resistant) test splits.

For each high-risk classification dataset (selected in Step 3), the training set is
kept fixed and every test row that is an exact / reverse-complement / near-duplicate
(k-mer Jaccard >= 0.8) of any training sequence is removed. The surviving test-row
indices are saved so that both a CSV reload and the cached FM embeddings (stored in
CSV row order) can be re-indexed to the identical clean subset in Step 5.

Outputs:
  results/rebuttal/E0_data_leakage/clean_splits/<safe_ds>__test_keep_idx.npy
  results/rebuttal/E0_data_leakage/clean_splits_summary.csv

Usage:
    python3 src/rebuttal/e0_leakage_audit/build_clean_splits.py
    python3 src/rebuttal/e0_leakage_audit/build_clean_splits.py --datasets virus/covid_variants
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import load_dataset
from src.rebuttal.e0_leakage_audit.audit_sequence_duplication import (
    _reverse_complement, neardup_cross_mask,
)

RESULT_DIR = "results/rebuttal/E0_data_leakage"
CLEAN_DIR = os.path.join(RESULT_DIR, "clean_splits")
HIGH_TXT = os.path.join(RESULT_DIR, "high_risk_datasets.txt")
SUMMARY_CSV = os.path.join(RESULT_DIR, "clean_splits_summary.csv")
DATA_ROOT = "data/dna_foundation_benchmark"


def safe_ds(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")


def build_for(name, skip_neardup=False):
    """
    Build two leakage-resistant test masks and save both keep-index files:

      * Tier 1 ("exact"): remove exact + reverse-complement duplicates of train.
        This is the unambiguous leakage signal (identical sequences shared across
        the split) and is the primary de-contaminated test used in the re-run.
      * Tier 2 ("neardup"): additionally remove near-duplicates (k-mer Jaccard >= 0.8).
        Stricter; for tasks of intrinsically similar sequences (e.g. viral variants)
        this can remove most of the test set, so the re-run only uses it when enough
        rows survive.
    """
    train_path = os.path.join(DATA_ROOT, name, "train.csv")
    test_path = os.path.join(DATA_ROOT, name, "test.csv")
    train_seqs, _y_tr, test_seqs, _y_te = load_dataset(train_path, test_path)
    train_seqs = list(map(str, train_seqs))
    test_seqs = list(map(str, test_seqs))

    train_set = set(train_seqs)
    exact = np.array([s in train_set for s in test_seqs], dtype=bool)
    revcomp = np.array([(_reverse_complement(s) in train_set) and (s not in train_set)
                        for s in test_seqs], dtype=bool)
    tier1 = exact | revcomp
    if skip_neardup:
        near = np.zeros(len(test_seqs), dtype=bool)
    else:
        near = neardup_cross_mask(train_seqs, test_seqs)
    tier2 = tier1 | near

    keep_exact = np.nonzero(~tier1)[0]
    keep_neardup = np.nonzero(~tier2)[0]

    os.makedirs(CLEAN_DIR, exist_ok=True)
    sd = safe_ds(name)
    np.save(os.path.join(CLEAN_DIR, f"{sd}__test_keep_exact.npy"), keep_exact)
    np.save(os.path.join(CLEAN_DIR, f"{sd}__test_keep_neardup.npy"), keep_neardup)

    n = len(test_seqs)
    return {
        "dataset": name,
        "n_test": n,
        "n_removed_exact": int(exact.sum()),
        "n_removed_revcomp": int(revcomp.sum()),
        "n_removed_neardup_only": int((near & ~tier1).sum()),
        # Tier 1
        "tier1_n_removed": int(tier1.sum()),
        "tier1_pct_removed": round(100.0 * tier1.sum() / max(1, n), 3),
        "tier1_n_test_clean": int(len(keep_exact)),
        # Tier 2
        "tier2_n_removed": int(tier2.sum()),
        "tier2_pct_removed": round(100.0 * tier2.sum() / max(1, n), 3),
        "tier2_n_test_clean": int(len(keep_neardup)),
    }


def main():
    ap = argparse.ArgumentParser(description="E0 build de-contaminated test splits")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Explicit dataset names; default reads high_risk_datasets.txt")
    ap.add_argument("--skip-neardup", action="store_true",
                    help="Only remove exact + revcomp duplicates (faster)")
    args = ap.parse_args()

    if args.datasets:
        names = args.datasets
    elif os.path.exists(HIGH_TXT):
        names = [ln.strip() for ln in open(HIGH_TXT) if ln.strip()]
    else:
        raise SystemExit(f"No --datasets given and {HIGH_TXT} not found (run Step 3).")

    print(f"Building clean test splits for {len(names)} high-risk datasets\n")
    rows = []
    for name in names:
        r = build_for(name, skip_neardup=args.skip_neardup)
        rows.append(r)
        print(f"  {name:40s} n_test={r['n_test']:6d} | "
              f"Tier1(exact+rc) -{r['tier1_n_removed']} ({r['tier1_pct_removed']:.2f}%) "
              f"-> {r['tier1_n_test_clean']} | "
              f"Tier2(+neardup) -{r['tier2_n_removed']} ({r['tier2_pct_removed']:.2f}%) "
              f"-> {r['tier2_n_test_clean']}")

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote {SUMMARY_CSV}")
    print(f"Keep-index files in {CLEAN_DIR}/")


if __name__ == "__main__":
    main()
