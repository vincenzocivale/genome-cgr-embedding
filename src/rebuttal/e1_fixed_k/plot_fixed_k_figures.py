"""
Rebuttal E1 — Figure 2 / Figure 3 variants for a single global k.

Refactors notebooks/paper_plots_main.ipynb cells 7-8 (Figure 2a/2b) and 10-12
(Figure 3a/3b) into functions parameterized by k-mer selection mode, so the same
plots can be regenerated for k=4, k=5, k=6 fixed globally, next to the existing
best-k main figures (k=6 decomposition and best-k win/loss are the paper's current
Figure 2/3 and are not regenerated here).

No new training: reads records_rf.csv, records_decomposition.csv, and the
fdr_results_{mode}.csv tables written by analyze_fixed_k.py.

Output: results/rebuttal/E1_fixed_k/figures/fig{2a,2b}_{mode}.pdf
        results/rebuttal/E1_fixed_k/figures/fig{3a,3b}_k{K}.pdf

Usage:
    python3 src/rebuttal/e1_fixed_k/plot_fixed_k_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

RESULTS = "results"
FIG_DIR = "results/rebuttal/E1_fixed_k/figures"

FM_COLORS = {
    "NTv3": "#2563EB", "HyenaDNA": "#D97706", "DNABERT-2": "#059669",
    "Caduceus": "#7C3AED", "Evo2": "#DC2626",
}
KMER_COLOR = "#6B7280"
TARGET_MODELS = {
    "NTv3_650M_pre": "NTv3",
    "hyenadna-medium-160k-seqlen-hf": "HyenaDNA",
    "DNABERT-2-117M": "DNABERT-2",
    "caduceus-ph_seqlen-131k_d_model-256_n_layer-16": "Caduceus",
    "evo2_1b_base": "Evo2 (1B)",
}
MODEL_COLOR_KEY = {
    "NTv3_650M_pre": "NTv3", "hyenadna-medium-160k-seqlen-hf": "HyenaDNA",
    "DNABERT-2-117M": "DNABERT-2",
    "caduceus-ph_seqlen-131k_d_model-256_n_layer-16": "Caduceus",
    "evo2_1b_base": "Evo2",
}
N_DATASETS = 57

mpl_rc = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.alpha": 0.35, "grid.linewidth": 0.5,
}
plt.rcParams.update(mpl_rc)


def _savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {path}")


def sig_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def bootstrap_median_ci(a, b, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    deltas = (a - b).values
    boot = [np.median(rng.choice(deltas, len(deltas), replace=True)) for _ in range(n)]
    return np.percentile(boot, [2.5, 97.5])


def _kmer_col_for_mode(rf: pd.DataFrame, mode: str) -> str:
    if mode == "best":
        col = "best_kmer_MCC"
        rf[col] = rf[["kmer_k4_MCC", "kmer_k5_MCC", "kmer_k6_MCC"]].max(axis=1)
        return col
    return f"kmer_{mode}_MCC"


def plot_fig2a(rf: pd.DataFrame, fdr_mode: pd.DataFrame, mode: str):
    km_col = _kmer_col_for_mode(rf, mode)
    rf_fdr = fdr_mode[(fdr_mode["probe_type"] == "rf") & (fdr_mode["metric"] == "MCC")]

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    bar_data = []
    for mkey, mlabel in TARGET_MODELS.items():
        fm_col = f"fm_{mkey}_MCC"
        if fm_col not in rf.columns:
            continue
        row = rf_fdr[rf_fdr["comparison"] == mkey]
        if row.empty:
            valid = rf[[fm_col, km_col]].dropna()
            fm_w = int((valid[fm_col] > valid[km_col]).sum())
            km_w = int((valid[km_col] > valid[fm_col]).sum())
            sig = "—"
        else:
            row = row.iloc[0]
            fm_w, km_w, sig = int(row["fm_wins"]), int(row["kmer_wins"]), sig_label(row["adjusted_p"])
        bar_data.append({"label": mlabel, "fm_wins": fm_w, "km_wins": km_w,
                         "ties": N_DATASETS - fm_w - km_w, "sig": sig})

    y_pos = np.arange(len(bar_data))
    km_w = [d["km_wins"] for d in bar_data]
    fm_w = [d["fm_wins"] for d in bar_data]
    ax.barh(y_pos, km_w, color="#2563EB", height=0.55, label="k-mer wins")
    ax.barh(y_pos, fm_w, left=km_w, color=KMER_COLOR, height=0.55, label="FM wins")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([d["label"] for d in bar_data], fontsize=12)
    ax.set_xlabel(f"Number of datasets (N={N_DATASETS})")
    ax.set_xlim(0, 65)
    for i, d in enumerate(bar_data):
        ax.text(59, i, d["sig"], va="center", fontsize=12)
    ax.axvline(N_DATASETS / 2, color="gray", ls=":", lw=0.7)
    ax.legend(fontsize=13, loc="lower right")
    plt.tight_layout()
    _savefig(f"fig2a_win_loss_mcc_{mode}.pdf")


def plot_fig2b(rf: pd.DataFrame, mode: str):
    km_col = _kmer_col_for_mode(rf, mode)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    model_order = list(TARGET_MODELS.keys())
    for i, mkey in enumerate(model_order):
        fm_col = f"fm_{mkey}_MCC"
        if fm_col not in rf.columns:
            continue
        valid = rf[[fm_col, km_col]].dropna()
        if valid.empty:
            continue
        ci = bootstrap_median_ci(valid[fm_col], valid[km_col])
        med = np.median((valid[fm_col] - valid[km_col]).values)
        color = FM_COLORS[MODEL_COLOR_KEY[mkey]]
        ax.errorbar(med, i, xerr=[[med - ci[0]], [ci[1] - med]], fmt="o",
                    color=color, capsize=3, ms=5)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels(list(TARGET_MODELS.values()))
    ax.set_xlabel("Median Δ (FM − k-mer)")
    plt.tight_layout()
    _savefig(f"fig2b_effect_size_mcc_{mode}.pdf")


def plot_fig3a(dec: pd.DataFrame, K: int):
    fig, ax = plt.subplots(figsize=(5.1, 3.6))
    r2_items = []
    for mkey, mlabel in TARGET_MODELS.items():
        col = f"ridge_k{K}_{mkey}_R2"
        if col in dec.columns:
            r2_items.append((mlabel, FM_COLORS[MODEL_COLOR_KEY[mkey]], dec[col].dropna().values))
    for i, (label, color, vals) in enumerate(r2_items):
        parts = ax.violinplot(vals, positions=[i], showmedians=True, widths=0.65)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(2)
        for p in ("cbars", "cmins", "cmaxes"):
            parts[p].set_color(color)
            parts[p].set_linewidth(1)
        jit = np.random.default_rng(42 + i).uniform(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), i) + jit, vals, color=color, alpha=0.35, s=7, zorder=3)
    ax.set_xticks(range(len(r2_items)))
    ax.set_xticklabels([r[0] for r in r2_items], fontsize=7)
    ax.set_ylabel("Ridge R2 (k-mer → FM)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="gray", ls=":", lw=0.7)
    plt.tight_layout()
    _savefig(f"fig3a_r2_violin_k{K}.pdf")


def plot_fig3b(rf: pd.DataFrame, dec: pd.DataFrame, K: int):
    fig, ax = plt.subplots(figsize=(5.1, 3.6))
    kmer_ref_col = f"kmer_k{K}_MCC"
    labels_grp = ["Full FM", "Projected\n(k-mer)", "Residual\n(novel)", f"k-mer (k={K})"]
    colors_grp = ["#2563EB", "#93C5FD", "#FBBF24", KMER_COLOR]
    bw = 0.16

    for i, (mkey, mlabel) in enumerate(TARGET_MODELS.items()):
        fm_col = f"fm_{mkey}_MCC"
        proj_col = f"proj_k{K}_{mkey}_MCC"
        resid_col = f"resid_k{K}_{mkey}_MCC"
        has_decomp = proj_col in dec.columns and resid_col in dec.columns
        center = i * 1.0
        cols_to_plot = [fm_col, proj_col, resid_col, kmer_ref_col] if has_decomp else [fm_col, kmer_ref_col]
        gc_list = colors_grp if has_decomp else [FM_COLORS[MODEL_COLOR_KEY[mkey]], colors_grp[3]]
        n_boxes = len(cols_to_plot)
        for j, (col, gc) in enumerate(zip(cols_to_plot, gc_list)):
            src = rf if col in rf.columns else dec
            if col not in src.columns:
                continue
            vals = src[col].dropna().values
            if len(vals) == 0:
                continue
            offset = (j - (n_boxes - 1) / 2) * bw
            ax.boxplot(vals, positions=[center + offset], widths=bw * 0.82, patch_artist=True,
                      showfliers=False, medianprops={"color": "white", "linewidth": 1.5},
                      whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
                      boxprops={"facecolor": gc, "alpha": 0.85, "linewidth": 0})

    box_handles = [mpatches.Patch(facecolor=gc, label=gl, alpha=0.85) for gl, gc in zip(labels_grp, colors_grp)]
    ax.set_xticks([i * 1.0 for i in range(len(TARGET_MODELS))])
    ax.set_xticklabels(list(TARGET_MODELS.values()), fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("MCC")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(handles=box_handles, fontsize=8, loc="upper right", frameon=True, framealpha=0.9)
    plt.tight_layout()
    _savefig(f"fig3b_downstream_mcc_k{K}.pdf")


def main():
    rf = pd.read_csv(f"{RESULTS}/classification/records_rf.csv")
    dec = pd.read_csv(f"{RESULTS}/decomposition/records_decomposition.csv")

    for mode in ("k4", "k5", "k6", "best"):
        fdr_path = f"{RESULTS}/rebuttal/E1_fixed_k/fdr_results_{mode}.csv"
        if not os.path.exists(fdr_path):
            print(f"Skipping mode={mode}: {fdr_path} not found (run analyze_fixed_k.py first)")
            continue
        fdr_mode = pd.read_csv(fdr_path)
        plot_fig2a(rf.copy(), fdr_mode, mode)
        plot_fig2b(rf.copy(), mode)

    # Fig 3 (decomposition): k=6 is already the paper's main figure, generate k=4/k=5 variants.
    for K in (4, 5):
        plot_fig3a(dec, K)
        plot_fig3b(rf, dec, K)


if __name__ == "__main__":
    main()
