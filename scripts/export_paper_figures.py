"""
Generate publication figures from the canonical retained result tables.

Usage:
    python3 scripts/export_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.analysis.paper_utils import CATEGORY_LABELS, add_dataset_annotations

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures"
RF_PATH = ROOT / "results" / "classification" / "records_rf.csv"
PCA_PATH = ROOT / "results" / "classification" / "records_pca.csv"
DECOMP_PATH = ROOT / "results" / "decomposition" / "records_decomposition.csv"
CONCAT_PATH = ROOT / "results" / "concat" / "records_concat_best.csv"
EFF_PATH = ROOT / "results" / "efficiency" / "efficiency_gpu_parallel.csv"

MODEL_COLS = {
    "NTv3-650M": "fm_NTv3_650M_pre_MCC",
    "HyenaDNA-160k": "fm_hyenadna-medium-160k-seqlen-hf_MCC",
    "DNABERT-2": "fm_DNABERT-2-117M_MCC",
}


def _load_table(path: Path, index_col: str = "dataset") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if index_col in df.columns:
        df = df.set_index(index_col)
    return df


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rf = add_dataset_annotations(_load_table(RF_PATH))
    pca = _load_table(PCA_PATH)
    decomp = _load_table(DECOMP_PATH)
    concat = _load_table(CONCAT_PATH)
    eff = pd.read_csv(EFF_PATH)
    return rf, pca, decomp, concat, eff


def fig1_scatter(rf: pd.DataFrame) -> None:
    palette = dict(zip(CATEGORY_LABELS.values(), sns.color_palette("tab10", n_colors=6)))
    best_k = pd.to_numeric(rf["best_kmer_MCC"], errors="coerce")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)

    for ax, (label, col) in zip(axes, MODEL_COLS.items()):
        x = pd.to_numeric(rf[col], errors="coerce")
        for biocat, sub in rf.groupby("biocat_label"):
            ax.scatter(
                x.loc[sub.index],
                best_k.loc[sub.index],
                label=biocat,
                s=35,
                alpha=0.85,
                color=palette[biocat],
                edgecolors="white",
                linewidth=0.4,
            )
        lo = np.nanmin([x.min(), best_k.min()]) - 0.02
        hi = np.nanmax([x.max(), best_k.max()]) + 0.02
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("FM MCC")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    axes[0].set_ylabel("Best k-mer MCC")
    axes[-1].legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", title="Category", fontsize=9)
    _save(fig, "01-fm-vs-kmer-by-model-multipanel.pdf")


def fig2_category_delta(rf: pd.DataFrame, decomp: pd.DataFrame, concat: pd.DataFrame) -> pd.DataFrame:
    cat = rf[["biocat_label", "seq_len_mean", "gc_content_mean", "best_k", "best_kmer_MCC"]].copy()
    cat["ntv3_MCC"] = pd.to_numeric(rf["fm_NTv3_650M_pre_MCC"], errors="coerce")
    cat["delta_ntv3_kmer"] = cat["ntv3_MCC"] - cat["best_kmer_MCC"]
    cat["proj_MCC"] = pd.to_numeric(decomp["proj_k6_NTv3_650M_pre_MCC"], errors="coerce")
    cat["resid_MCC"] = pd.to_numeric(decomp["resid_k6_NTv3_650M_pre_MCC"], errors="coerce")
    cat["R2"] = pd.to_numeric(decomp["ridge_k6_NTv3_650M_pre_R2"], errors="coerce")
    cat["synergy"] = cat["ntv3_MCC"] - np.maximum(cat["proj_MCC"], cat["resid_MCC"])
    cat["concat_MCC"] = pd.to_numeric(concat["MCC"], errors="coerce")
    cat["delta_concat"] = cat["concat_MCC"] - cat[["ntv3_MCC", "best_kmer_MCC"]].max(axis=1)

    order = cat.groupby("biocat_label")["delta_ntv3_kmer"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sns.boxplot(data=cat, y="biocat_label", x="delta_ntv3_kmer", order=order, color="#E7E7E7", fliersize=0, ax=ax)
    sns.stripplot(data=cat, y="biocat_label", x="delta_ntv3_kmer", order=order, color="#2B6CB0", size=4, alpha=0.75, ax=ax)
    ax.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Δ MCC (NTv3 − best k-mer)")
    ax.set_ylabel("")
    _save(fig, "02-category-delta-ntv3-kmer.pdf")
    return cat


def fig3_length_effect(cat: pd.DataFrame) -> None:
    valid = cat[["seq_len_mean", "delta_ntv3_kmer"]].dropna().rename(columns={"seq_len_mean": "seq_len"})
    bins = [0, 100, 300, 600, valid["seq_len"].max() + 1]
    labels = ["≤100", "101–300", "301–600", ">600"]
    valid["len_bin"] = pd.cut(valid["seq_len"], bins=bins, labels=labels, include_lowest=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sns.boxplot(data=valid, x="len_bin", y="delta_ntv3_kmer", color="#E7E7E7", fliersize=0, ax=ax)
    sns.stripplot(data=valid, x="len_bin", y="delta_ntv3_kmer", color="#2B6CB0", size=4, alpha=0.75, ax=ax)
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Sequence length bin (bp)")
    ax.set_ylabel("Δ MCC (NTv3 − best k-mer)")
    _save(fig, "03-sequence-length-effect-box-only.pdf")


def fig4_best_k(rf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    counts = rf["best_k"].dropna().astype(int).value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values, color="#4C78A8")
    axes[0].set_xlabel("Best k")
    axes[0].set_ylabel("Datasets")

    sns.boxplot(data=rf, x="best_k", y="gc_content_mean", color="#E7E7E7", fliersize=0, ax=axes[1])
    sns.stripplot(data=rf, x="best_k", y="gc_content_mean", color="#2B6CB0", size=4, alpha=0.75, ax=axes[1])
    axes[1].set_xlabel("Best k")
    axes[1].set_ylabel("Mean GC content")
    _save(fig, "04-best-k-distribution.pdf")


def fig5_decomposition(decomp: pd.DataFrame, rf: pd.DataFrame, cat: pd.DataFrame) -> None:
    r2 = pd.DataFrame(
        {
            "NTv3-650M": pd.to_numeric(decomp["ridge_k6_NTv3_650M_pre_R2"], errors="coerce"),
            "HyenaDNA-160k": pd.to_numeric(decomp["ridge_k6_hyenadna-medium-160k-seqlen-hf_R2"], errors="coerce"),
            "DNABERT-2": pd.to_numeric(decomp["ridge_k6_DNABERT-2-117M_R2"], errors="coerce"),
        }
    )
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    sns.boxplot(data=r2.melt(var_name="Model", value_name="R2"), x="Model", y="R2", color="#E7E7E7", fliersize=0, ax=ax)
    sns.stripplot(data=r2.melt(var_name="Model", value_name="R2"), x="Model", y="R2", color="#2B6CB0", size=4, alpha=0.6, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("R² (k=6 → FM)")
    _save(fig, "05-decomposition-r2.pdf")

    best_k = pd.to_numeric(rf["best_kmer_MCC"], errors="coerce")
    proj = pd.to_numeric(decomp["proj_k6_NTv3_650M_pre_MCC"], errors="coerce")
    resid = pd.to_numeric(decomp["resid_k6_NTv3_650M_pre_MCC"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True, sharey=True)
    for ax, values, title in (
        (axes[0], proj, "Projection vs best k-mer"),
        (axes[1], resid, "Residual vs best k-mer"),
    ):
        lo = np.nanmin([best_k.min(), values.min()]) - 0.02
        hi = np.nanmax([best_k.max(), values.max()]) + 0.02
        ax.scatter(best_k, values, s=35, alpha=0.8, color="#2B6CB0", edgecolors="white", linewidth=0.4)
        ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(title)
        ax.set_xlabel("Best k-mer MCC")
    axes[0].set_ylabel("MCC")
    _save(fig, "05-decomposition-projection-residual.pdf")

    order = cat.groupby("biocat_label")["synergy"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sns.boxplot(data=cat, y="biocat_label", x="synergy", order=order, color="#E7E7E7", fliersize=0, ax=ax)
    sns.stripplot(data=cat, y="biocat_label", x="synergy", order=order, color="#2B6CB0", size=4, alpha=0.75, ax=ax)
    ax.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Synergy (full − max(proj, resid))")
    ax.set_ylabel("")
    _save(fig, "06-synergy-by-category.pdf")


def fig7_components(rf: pd.DataFrame, decomp: pd.DataFrame) -> None:
    long_frames = []
    for label, fm_col in MODEL_COLS.items():
        tag = fm_col.replace("fm_", "").replace("_MCC", "")
        long_frames.append(pd.DataFrame({"Model": label, "Component": "Full", "MCC": pd.to_numeric(rf[fm_col], errors="coerce")}))
        long_frames.append(pd.DataFrame({"Model": label, "Component": "Projection", "MCC": pd.to_numeric(decomp[f"proj_k6_{tag}_MCC"], errors="coerce")}))
        long_frames.append(pd.DataFrame({"Model": label, "Component": "Residual", "MCC": pd.to_numeric(decomp[f"resid_k6_{tag}_MCC"], errors="coerce")}))

    comp = pd.concat(long_frames, ignore_index=True).dropna()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), sharey=True)
    for ax, label in zip(axes, MODEL_COLS):
        sub = comp[comp["Model"] == label]
        sns.boxplot(data=sub, x="Component", y="MCC", color="#E7E7E7", fliersize=0, ax=ax)
        sns.stripplot(data=sub, x="Component", y="MCC", color="#2B6CB0", size=4, alpha=0.6, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("MCC")
    _save(fig, "07_component_comparison_by_model.pdf")


def fig8_concat(cat: pd.DataFrame) -> None:
    delta = cat["delta_concat"].dropna()
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    sns.boxplot(x=delta, color="#E7E7E7", fliersize=0, ax=ax)
    sns.stripplot(x=delta, color="#2B6CB0", size=4, alpha=0.75, ax=ax)
    ax.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Δ MCC (concat − best single)")
    ax.set_yticks([])
    _save(fig, "08_concatenation_delta.pdf")


def fig9_pca(rf: pd.DataFrame, pca: pd.DataFrame) -> None:
    pairs = {
        "NTv3-650M": ("pca_fm_NTv3_650M_pre_MCC", "fm_NTv3_650M_pre_MCC"),
        "HyenaDNA-160k": ("pca_fm_hyenadna-medium-160k-seqlen-hf_MCC", "fm_hyenadna-medium-160k-seqlen-hf_MCC"),
        "DNABERT-2": ("pca_fm_DNABERT-2-117M_MCC", "fm_DNABERT-2-117M_MCC"),
        "k=4": ("pca_kmer_k4_MCC", "kmer_k4_MCC"),
        "k=5": ("pca_kmer_k5_MCC", "kmer_k5_MCC"),
        "k=6": ("pca_kmer_k6_MCC", "kmer_k6_MCC"),
    }
    rows = []
    for label, (pca_col, raw_col) in pairs.items():
        if pca_col in pca.columns and raw_col in rf.columns:
            delta = pd.to_numeric(pca[pca_col], errors="coerce") - pd.to_numeric(rf[raw_col], errors="coerce")
            rows.append(pd.DataFrame({"Method": label, "Delta": delta}))
    delta_df = pd.concat(rows, ignore_index=True).dropna()
    order = delta_df.groupby("Method")["Delta"].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    sns.boxplot(data=delta_df, y="Method", x="Delta", order=order, color="#E7E7E7", fliersize=0, ax=ax)
    sns.stripplot(data=delta_df, y="Method", x="Delta", order=order, color="#2B6CB0", size=4, alpha=0.6, ax=ax)
    ax.axvline(0.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Δ MCC (PCA − native)")
    ax.set_ylabel("")
    _save(fig, "09_pca_effect.pdf")


def fig10_efficiency(eff: pd.DataFrame) -> None:
    eff = eff.copy()
    eff["Method"] = eff["method"].map(
        {
            "fm_NTv3_650M_pre": "NTv3-650M",
            "fm_hyenadna-medium-160k-seqlen-hf": "HyenaDNA-160k",
            "fm_DNABERT-2-117M": "DNABERT-2",
            "kmer_k6": "k-mer (k=6)",
        }
    )
    eff = eff.dropna(subset=["Method"])
    eff["seq_len"] = pd.to_numeric(eff["seq_len"], errors="coerce")
    eff["GFLOPS_per_seq"] = pd.to_numeric(eff["GFLOPS_per_seq"], errors="coerce")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for method, sub in eff.groupby("Method"):
        sub = sub.sort_values("seq_len")
        ax.plot(sub["seq_len"], sub["GFLOPS_per_seq"], marker="o", linewidth=2, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (bp)")
    ax.set_ylabel("GFLOPS per sequence")
    ax.legend(fontsize=9)
    _save(fig, "10_efficiency_gflops_per_seq.pdf")


def fig11_correlations(rf: pd.DataFrame, decomp: pd.DataFrame, cat: pd.DataFrame) -> None:
    best_k = pd.to_numeric(rf["best_kmer_MCC"], errors="coerce")
    corr = pd.DataFrame(index=rf.index)
    corr["seq_len"] = pd.to_numeric(rf["seq_len_mean"], errors="coerce")
    corr["gc_content"] = pd.to_numeric(rf["gc_content_mean"], errors="coerce")
    corr["class_balance"] = pd.to_numeric(rf["class_balance"], errors="coerce")
    corr["delta_full_kmer"] = pd.to_numeric(rf["fm_NTv3_650M_pre_MCC"], errors="coerce") - best_k
    corr["delta_resid_kmer"] = pd.to_numeric(decomp["resid_k6_NTv3_650M_pre_MCC"], errors="coerce") - best_k
    corr["gap_full_proj"] = pd.to_numeric(rf["fm_NTv3_650M_pre_MCC"], errors="coerce") - pd.to_numeric(
        decomp["proj_k6_NTv3_650M_pre_MCC"], errors="coerce"
    )
    corr["synergy"] = pd.to_numeric(cat["synergy"], errors="coerce")
    corr["R2"] = pd.to_numeric(decomp["ridge_k6_NTv3_650M_pre_R2"], errors="coerce")
    corr = corr.dropna()

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    sns.heatmap(corr.corr(method="spearman"), cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, ax=ax)
    _save(fig, "11_cross_experiment_correlations.pdf")


def main() -> None:
    _setup_style()
    rf, pca, decomp, concat, eff = _base_tables()
    fig1_scatter(rf)
    cat = fig2_category_delta(rf, decomp, concat)
    fig3_length_effect(cat)
    fig4_best_k(rf)
    fig5_decomposition(decomp, rf, cat)
    fig7_components(rf, decomp)
    fig8_concat(cat)
    fig9_pca(rf, pca)
    fig10_efficiency(eff)
    fig11_correlations(rf, decomp, cat)
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
