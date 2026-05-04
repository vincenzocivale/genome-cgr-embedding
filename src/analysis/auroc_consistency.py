"""
AUROC consistency analysis: do MCC and AUROC verdicts agree?

For each (dataset, FM_model), computes:
  - Verdict by MCC: kmer_wins if best_kmer_MCC > fm_MCC, else fm_wins
  - Verdict by AUROC: same logic with AUROC

Reports concordance rate and lists discordant datasets.

Usage:
    python3 src/analysis/auroc_consistency.py
    python3 src/analysis/auroc_consistency.py --records results/classification/records_rf.csv
    python3 src/analysis/auroc_consistency.py --also-linear
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

AUROC_CSV = "results/analysis/auroc_consistency.csv"


def _best_kmer_metric(df: pd.DataFrame, metric: str, prefix: str = "kmer") -> pd.Series:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_k") and c.endswith(f"_{metric}")]
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].max(axis=1)


def _detect_fm_columns(df: pd.DataFrame, metric: str) -> list[tuple[str, str]]:
    """Returns list of (model_tag, col) for FM metric columns.
    Handles both standard (ending _MCC) and pool-suffixed (ending _MCC__pool_X) columns."""
    result = []
    seen = set()
    for col in df.columns:
        for prefix in ("lp_fm_", "fm_"):
            if not col.startswith(prefix):
                continue
            rest = col[len(prefix):]
            suffix = f"_{metric}"
            if rest.endswith(suffix):
                tag = rest[:-len(suffix)]
                if tag not in seen:
                    seen.add(tag)
                    result.append((tag, col))
            else:
                pool_suffix = f"_{metric}__pool_"
                idx = rest.find(pool_suffix)
                if idx != -1:
                    tag = rest[:idx] + rest[idx + len(f"_{metric}"):]
                    if tag not in seen:
                        seen.add(tag)
                        result.append((tag, col))
    return result


def analyze_consistency(
    records_path: str,
    kmer_prefix: str = "kmer",
    probe_type: str = "rf",
) -> pd.DataFrame:
    if not os.path.exists(records_path):
        print(f"  Not found: {records_path}")
        return pd.DataFrame()

    df = pd.read_csv(records_path, index_col=0)
    best_mcc = _best_kmer_metric(df, "MCC", kmer_prefix)
    best_auroc = _best_kmer_metric(df, "AUROC", kmer_prefix)

    if best_mcc.empty or best_auroc.empty:
        print(f"  No k-mer columns found in {records_path}")
        return pd.DataFrame()

    fm_mcc_configs = _detect_fm_columns(df, "MCC")
    fm_auroc_configs = {tag: col for tag, col in _detect_fm_columns(df, "AUROC")}

    rows = []
    for tag, mcc_col in fm_mcc_configs:
        auroc_col = fm_auroc_configs.get(tag)
        if auroc_col is None or auroc_col not in df.columns:
            continue
        if mcc_col not in df.columns:
            continue

        common = df.index.intersection(best_mcc.index)
        common = common[df[mcc_col].loc[common].notna() & df[auroc_col].loc[common].notna()]
        if len(common) < 5:
            continue

        kmer_mcc = best_mcc.loc[common].values
        kmer_auroc = best_auroc.loc[common].values
        fm_mcc = df[mcc_col].loc[common].values
        fm_auroc = df[auroc_col].loc[common].values

        verdict_mcc = np.where(kmer_mcc > fm_mcc, "kmer", "fm")
        verdict_auroc = np.where(kmer_auroc > fm_auroc, "kmer", "fm")

        concordant = (verdict_mcc == verdict_auroc)
        concordance_rate = float(concordant.mean())
        discordant_datasets = list(common[~concordant])

        n = len(common)
        kmer_wins_mcc = int(np.sum(verdict_mcc == "kmer"))
        kmer_wins_auroc = int(np.sum(verdict_auroc == "kmer"))

        print(
            f"  {probe_type} | {tag}: n={n}, concordance={concordance_rate:.3f} "
            f"kmer_wins(MCC)={kmer_wins_mcc}/{n} kmer_wins(AUROC)={kmer_wins_auroc}/{n}"
        )
        if discordant_datasets:
            print(f"    Discordant ({len(discordant_datasets)}): {discordant_datasets[:10]}")

        rows.append({
            "probe_type": probe_type,
            "fm_model": tag,
            "n_datasets": n,
            "concordance_rate": concordance_rate,
            "n_concordant": int(concordant.sum()),
            "n_discordant": int((~concordant).sum()),
            "kmer_wins_mcc": kmer_wins_mcc,
            "kmer_wins_auroc": kmer_wins_auroc,
            "discordant_datasets": "|".join(discordant_datasets),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="AUROC consistency: do MCC and AUROC verdicts agree?"
    )
    parser.add_argument("--records", default="results/classification/records_rf.csv")
    parser.add_argument("--also-linear", action="store_true",
                        help="Also analyze records_linear_probe.csv")
    parser.add_argument("--output", default=AUROC_CSV)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results = []

    print(f"\n{'='*60}")
    print(f"AUROC Consistency Analysis")
    print(f"{'='*60}")

    # RF records
    print(f"\n[RF] {args.records}")
    res = analyze_consistency(args.records, kmer_prefix="kmer", probe_type="rf")
    if not res.empty:
        all_results.append(res)

    if args.also_linear:
        lp_path = "results/classification/records_linear_probe.csv"
        print(f"\n[Linear Probe] {lp_path}")
        res = analyze_consistency(lp_path, kmer_prefix="lp_kmer", probe_type="linear_probe")
        if not res.empty:
            all_results.append(res)

    if not all_results:
        print("\nNo results generated.")
        return

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(args.output, index=False)
    print(f"\n\nResults saved to: {args.output}")
    print(f"\nSummary table:")
    cols = ["probe_type", "fm_model", "n_datasets", "concordance_rate",
            "kmer_wins_mcc", "kmer_wins_auroc", "n_discordant"]
    print(final[cols].to_string(index=False))


if __name__ == "__main__":
    main()
