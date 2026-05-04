"""
Generate a compact markdown audit of the retained paper statistics.

Usage:
    python3 scripts/paper_stats_audit.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.paper_utils import add_dataset_annotations

ROOT = Path(__file__).resolve().parents[1]
RF_PATH = ROOT / "results" / "classification" / "records_rf.csv"
DECOMP_PATH = ROOT / "results" / "decomposition" / "records_decomposition.csv"
CONCAT_PATH = ROOT / "results" / "concat" / "records_concat_best.csv"
TRUNC_PATH = ROOT / "results" / "classification" / "truncation_analysis.csv"
SPLICE_PATH = ROOT / "results" / "exploratory" / "splice_residual_motif_overlap.csv"


def _fmt(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"


def main() -> None:
    rf = add_dataset_annotations(pd.read_csv(RF_PATH).set_index("dataset"))
    decomp = pd.read_csv(DECOMP_PATH).set_index("dataset")
    concat = pd.read_csv(CONCAT_PATH).set_index("dataset")
    trunc = pd.read_csv(TRUNC_PATH)
    splice = pd.read_csv(SPLICE_PATH)

    best_k_counts = rf["best_k"].dropna().astype(int).value_counts().sort_index()
    delta_vs_best = {
        "NTv3-650M": pd.to_numeric(rf["fm_NTv3_650M_pre_MCC"], errors="coerce") - rf["best_kmer_MCC"],
        "HyenaDNA-160k": pd.to_numeric(rf["fm_hyenadna-medium-160k-seqlen-hf_MCC"], errors="coerce") - rf["best_kmer_MCC"],
        "DNABERT-2": pd.to_numeric(rf["fm_DNABERT-2-117M_MCC"], errors="coerce") - rf["best_kmer_MCC"],
    }

    concat_delta = pd.to_numeric(concat["MCC"], errors="coerce") - pd.concat(
        [
            rf["best_kmer_MCC"],
            pd.to_numeric(rf["fm_NTv3_650M_pre_MCC"], errors="coerce"),
            pd.to_numeric(rf["fm_hyenadna-medium-160k-seqlen-hf_MCC"], errors="coerce"),
            pd.to_numeric(rf["fm_DNABERT-2-117M_MCC"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)

    trunc_agg = (
        trunc.groupby(["model", "dataset"])
        .agg(
            truncation_rate=("truncation_rate", "max"),
            n_sequences=("n_sequences", "sum"),
            n_truncated=("n_truncated", "sum"),
        )
        .reset_index()
    )
    trunc_summary = trunc_agg.groupby("model").agg(
        datasets=("dataset", "count"),
        mean_truncation_rate=("truncation_rate", "mean"),
        max_truncation_rate=("truncation_rate", "max"),
        total_sequences=("n_sequences", "sum"),
        total_truncated=("n_truncated", "sum"),
    )
    trunc_summary["overall_truncation_pct"] = (
        trunc_summary["total_truncated"] / trunc_summary["total_sequences"] * 100.0
    )

    print("# Paper Statistics Audit")
    print()
    print("## Classification")
    print(f"- Datasets: {len(rf)}")
    print(f"- Mean best-k MCC: {_fmt(rf['best_kmer_MCC'].mean())}")
    print(f"- Mean MCC at k=5: {_fmt(pd.to_numeric(rf['kmer_k5_MCC'], errors='coerce').mean())}")
    print(
        "- Best-k distribution: "
        + ", ".join(f"k={k}: {count}" for k, count in best_k_counts.items())
    )
    for model_name, delta in delta_vs_best.items():
        print(f"- Mean Δ MCC ({model_name} − best k-mer): {_fmt(delta.mean())}")
    print()

    print("## Decomposition")
    for tag, label in (
        ("NTv3_650M_pre", "NTv3-650M"),
        ("hyenadna-medium-160k-seqlen-hf", "HyenaDNA-160k"),
        ("DNABERT-2-117M", "DNABERT-2"),
    ):
        r2 = pd.to_numeric(decomp[f"ridge_k6_{tag}_R2"], errors="coerce")
        proj = pd.to_numeric(decomp[f"proj_k6_{tag}_MCC"], errors="coerce")
        resid = pd.to_numeric(decomp[f"resid_k6_{tag}_MCC"], errors="coerce")
        print(f"- {label} mean R²: {_fmt(r2.mean())} | median R²: {_fmt(r2.median())}")
        print(f"- {label} mean projection MCC: {_fmt(proj.mean())}")
        print(f"- {label} mean residual MCC: {_fmt(resid.mean())}")
    print()

    print("## Supplementary")
    print(f"- Mean Δ MCC (concat − best single): {_fmt(concat_delta.mean())}")
    splice_summary = splice.groupby("motif").agg(
        datasets=("dataset", "count"),
        overlap_rate=("overlap_rate", "mean"),
        mean_delta_resid_l2=("mean_delta_resid_l2", "mean"),
    )
    print("- Splice motif summary:")
    print(splice_summary.round(4).to_string())
    print()

    print("## Truncation")
    print(
        trunc_summary[
            ["datasets", "mean_truncation_rate", "max_truncation_rate", "overall_truncation_pct"]
        ]
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
