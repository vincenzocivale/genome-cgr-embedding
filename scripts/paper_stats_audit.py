"""
Paper stats audit — computes all TODO items from existing CSVs.

Outputs a single markdown report with all verified/corrected values.

Usage:
    python scripts/paper_stats_audit.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC  = os.path.join(ROOT, "results/classification/records_rf.csv")
BEST = os.path.join(ROOT, "results/classification/records_best_k_analysis.csv")
DEC  = os.path.join(ROOT, "results/decomposition/records_decomposition.csv")
TRUNC = os.path.join(ROOT, "results/classification/truncation_analysis.csv")

# ── Load ───────────────────────────────────────────────────────────────────
rec  = pd.read_csv(REC)
best = pd.read_csv(BEST)
dec  = pd.read_csv(DEC)

# biocat mapping from best (authoritative)
biocat_map = best.set_index("dataset")["biocat"].to_dict()

# Add biocat to dec
dec["biocat"] = dec["dataset"].map(biocat_map)

# ── Helpers ────────────────────────────────────────────────────────────────
SPLICE_DATASETS = best[best["biocat"] == "splice"]["dataset"].tolist()

def mean_mcc(series):
    return series.mean()

def delta_series(a, b):
    """b - a  (positive means b is better)"""
    return b - a

# ═══════════════════════════════════════════════════════════════════════════
# TODO-A  — k=5 MCC stats
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-A: Fixed k=5 MCC stats")
print("="*70)

# best k per dataset (in best_k_analysis)
best_k_mcc = best[["dataset", "kmer_k4_MCC", "kmer_k5_MCC", "kmer_k6_MCC", "best_k"]].copy()
best_k_mcc["best_k_MCC"] = best_k_mcc.apply(
    lambda r: r[f"kmer_k{int(r['best_k'])}_MCC"], axis=1
)

k5_mcc = best_k_mcc["kmer_k5_MCC"]
best_mcc = best_k_mcc["best_k_MCC"]

mean_k5  = k5_mcc.mean()
delta_k5_vs_best = (k5_mcc - best_mcc).mean()   # negative if best > k5

print(f"  Mean MCC @ k=5              : {mean_k5:.4f}  (paper claims 0.519)")
print(f"  Mean MCC @ best k           : {best_mcc.mean():.4f}")
print(f"  ΔMCCmean (k5 − best_k)      : {delta_k5_vs_best:.4f}  (paper claims −0.018 disadvantage)")
print(f"  ΔMCCmean (best_k − k5)      : {-delta_k5_vs_best:.4f}  (paper claims +0.018 gain from best_k)")

# ═══════════════════════════════════════════════════════════════════════════
# TODO-B  — Per-dataset k-selection table (Appendix C.3)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-B: Per-dataset k-selection table (best_k, gains)")
print("="*70)

app_c3 = best[["dataset", "biocat", "best_k",
               "kmer_k4_MCC", "kmer_k5_MCC", "kmer_k6_MCC",
               "delta_k6_k4"]].copy()
app_c3["best_k_MCC"] = app_c3.apply(
    lambda r: r[f"kmer_k{int(r['best_k'])}_MCC"], axis=1
)
app_c3["worst_k_MCC"] = app_c3[["kmer_k4_MCC","kmer_k5_MCC","kmer_k6_MCC"]].min(axis=1)
app_c3["max_gain"] = app_c3["best_k_MCC"] - app_c3["worst_k_MCC"]
app_c3 = app_c3.sort_values("max_gain", ascending=False)

print(app_c3[["dataset","biocat","best_k","best_k_MCC","worst_k_MCC","max_gain"]].to_string(index=False))
print(f"\n  Overall max gain (single dataset): {app_c3['max_gain'].max():.4f}  (paper claims +0.176)")
print(f"  Dataset with max gain: {app_c3.iloc[0]['dataset']}")

# ═══════════════════════════════════════════════════════════════════════════
# TODO-C/G — HyenaDNA projection MCC and Δ vs k-mer
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-C/G: HyenaDNA projection MCC and Δ projection vs k-mer")
print("="*70)

hyena_proj_col = "proj_k6_hyenadna-medium-160k-seqlen-hf_MCC"
ntv3_proj_col  = "proj_k6_NTv3_650M_pre_MCC"
dna2_proj_col  = "proj_k6_DNABERT-2-117M_MCC"

# Merge dec with best to get best_k_MCC per dataset
merged = dec.merge(
    best[["dataset","kmer_k4_MCC","kmer_k5_MCC","kmer_k6_MCC","best_k"]],
    on="dataset", how="left"
)
merged["best_k_MCC"] = merged.apply(
    lambda r: r[f"kmer_k{int(r['best_k'])}_MCC"] if pd.notna(r["best_k"]) else np.nan,
    axis=1
)

hyena_proj = merged[hyena_proj_col].dropna()
ntv3_proj  = merged[ntv3_proj_col].dropna()
dna2_proj  = merged[dna2_proj_col].dropna()

print(f"  HyenaDNA proj MCC mean   : {hyena_proj.mean():.4f}  (paper claims 0.469)")
print(f"  NTv3     proj MCC mean   : {ntv3_proj.mean():.4f}  (paper claims 0.531)")
print(f"  DNABERT-2 proj MCC mean  : {dna2_proj.mean():.4f}  (paper claims 0.510)")

# Δ HyenaDNA projection vs best k-mer
# Use only rows where both are available
cmp_h = merged[[hyena_proj_col, "best_k_MCC"]].dropna()
delta_hyena_proj = (cmp_h[hyena_proj_col] - cmp_h["best_k_MCC"]).mean()
print(f"\n  Δ(HyenaDNA proj − best kmer) mean : {delta_hyena_proj:.4f}  (paper claims −0.066)")

cmp_n = merged[[ntv3_proj_col, "best_k_MCC"]].dropna()
delta_ntv3_proj = (cmp_n[ntv3_proj_col] - cmp_n["best_k_MCC"]).mean()
print(f"  Δ(NTv3 proj − best kmer) mean     : {delta_ntv3_proj:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# TODO-D — Breakdown per categoria: Splice vs non-splice ΔMCC (FM vs kmer)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-D: Breakdown per categoria biologica ΔMCCmean (NTv3 − best kmer)")
print("="*70)

# Use records.csv which has both FM and kmer MCC
ntv3_col = "fm_NTv3_650M_pre_MCC"
rec2 = rec.copy()
rec2["biocat"] = rec2["dataset"].map(biocat_map)

# Compute best_k_MCC from rec
rec2["best_k_MCC"] = rec2[["kmer_k4_MCC","kmer_k5_MCC","kmer_k6_MCC"]].max(axis=1)
rec2["delta_NTv3_kmer"] = rec2[ntv3_col] - rec2["best_k_MCC"]

print("\n  Per-category ΔMCCmean (NTv3 − best kmer):")
cat_stats = rec2.groupby("biocat")["delta_NTv3_kmer"].agg(["mean","count"]).sort_values("mean")
print(cat_stats.to_string())

splice_mask = rec2["biocat"] == "splice"
non_splice_mask = ~splice_mask & rec2["biocat"].notna()

delta_splice     = rec2.loc[splice_mask, "delta_NTv3_kmer"].mean()
delta_non_splice = rec2.loc[non_splice_mask, "delta_NTv3_kmer"].mean()
delta_all        = rec2["delta_NTv3_kmer"].mean()

print(f"\n  Splice ΔMCCmean (NTv3 − kmer)     : {delta_splice:.4f}  (paper claims +0.287 → NTv3 better)")
print(f"  Non-splice ΔMCCmean (NTv3 − kmer) : {delta_non_splice:.4f}  (paper claims +0.026)")
print(f"  Overall ΔMCCmean (NTv3 − kmer)    : {delta_all:.4f}")

# Nota: paper usa kmer−NTv3 (positivo = kmer vince)
print(f"\n  [Sign flipped: kmer−NTv3]")
print(f"  Splice    : {-delta_splice:.4f}")
print(f"  Non-splice: {-delta_non_splice:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# TODO-E — R² Table 4: media vs mediana
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-E: R² Table 4 — media vs mediana (NTv3 and HyenaDNA)")
print("="*70)

r2_ntv3  = dec["ridge_k6_NTv3_650M_pre_R2"].dropna()
r2_hyena = dec["ridge_k6_hyenadna-medium-160k-seqlen-hf_R2"].dropna()
r2_dna2  = dec["ridge_k6_DNABERT-2-117M_R2"].dropna()

print(f"  NTv3   R² mean={r2_ntv3.mean():.4f}  median={r2_ntv3.median():.4f}  (paper text ≈0.50, Table 4 0.48)")
print(f"  HyenaDNA R² mean={r2_hyena.mean():.4f}  median={r2_hyena.median():.4f}  (paper text ≈0.74, Table 4 0.71)")
print(f"  DNABERT-2 R² mean={r2_dna2.mean():.4f}  median={r2_dna2.median():.4f}  (paper text 0.21)")

# ═══════════════════════════════════════════════════════════════════════════
# TODO-F — Residuo NTv3 per categoria, incluso Splice residual Δ
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-F: Residuo NTv3 per categoria (Appendix D)")
print("="*70)

resid_col = "resid_k6_NTv3_650M_pre_MCC"
proj_col  = "proj_k6_NTv3_650M_pre_MCC"

dec_bio = dec[["dataset","biocat", resid_col, proj_col, "ridge_k6_NTv3_650M_pre_R2"]].dropna(
    subset=[resid_col, proj_col]
)

# Merge with best_k_MCC
dec_bio = dec_bio.merge(
    best[["dataset","kmer_k4_MCC","kmer_k5_MCC","kmer_k6_MCC","best_k"]],
    on="dataset", how="left"
)
dec_bio["best_k_MCC"] = dec_bio.apply(
    lambda r: r[f"kmer_k{int(r['best_k'])}_MCC"] if pd.notna(r.get("best_k")) else np.nan,
    axis=1
)
dec_bio["delta_resid_kmer"] = dec_bio[resid_col] - dec_bio["best_k_MCC"]

print("\n  Per-category residuo NTv3 stats:")
cat_res = dec_bio.groupby("biocat").agg(
    n=("dataset","count"),
    resid_mean=(resid_col, "mean"),
    delta_resid_kmer_mean=("delta_resid_kmer","mean"),
    r2_mean=("ridge_k6_NTv3_650M_pre_R2","mean"),
).sort_values("delta_resid_kmer_mean", ascending=False)
print(cat_res.to_string())

splice_resid = dec_bio[dec_bio["biocat"]=="splice"]["delta_resid_kmer"].mean()
print(f"\n  Splice residual mean Δ (resid − kmer): {splice_resid:.4f}  (paper claims +0.177)")

# ═══════════════════════════════════════════════════════════════════════════
# TODO-X — Truncation analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TODO-X: Truncation rates per model")
print("="*70)

if os.path.exists(TRUNC):
    trunc = pd.read_csv(TRUNC)
    # Aggregate over train+test: use max truncation rate per (model, dataset)
    agg = trunc.groupby(["model","dataset"]).agg(
        truncation_rate=("truncation_rate","max"),
        n_sequences=("n_sequences","sum"),
        n_truncated=("n_truncated","sum"),
    ).reset_index()
    model_agg = agg.groupby("model").agg(
        n_datasets=("dataset","count"),
        mean_trunc_rate=("truncation_rate","mean"),
        max_trunc_rate=("truncation_rate","max"),
        n_datasets_truncated=("truncation_rate", lambda x: (x>0).sum()),
        total_sequences=("n_sequences","sum"),
        total_truncated=("n_truncated","sum"),
    )
    model_agg["overall_trunc_pct"] = model_agg["total_truncated"]/model_agg["total_sequences"]*100
    print(model_agg.to_string())

    # Short model names
    model_short = {
        "InstaDeepAI/NTv3_650M_pre": "NTv3",
        "LongSafari/hyenadna-medium-160k-seqlen-hf": "HyenaDNA",
        "zhihan1996/DNABERT-2-117M": "DNABERT-2",
        "evo2_7b": "Evo2",
    }
    print("\n  Summary (% sequences truncated overall):")
    for m, row in model_agg.iterrows():
        short = model_short.get(m, m)
        print(f"  {short:12s}: {row['overall_trunc_pct']:.2f}% of sequences  "
              f"(datasets with any truncation: {int(row['n_datasets_truncated'])}/{int(row['n_datasets'])})")
else:
    print("  truncation_analysis.csv NOT FOUND — run truncation_analysis.py first")
    print("  Based on max_length settings:")
    print("  NTv3      max_length=32768 tokens (char-level → ~32kb bp) — truncation negligible for ≤2kb datasets")
    print("  HyenaDNA  max_length=32768 tokens — same as NTv3")
    print("  DNABERT-2 max_length=4096 tokens (BPE ~5bp/tok → ~20kb bp) — truncation possible only for very long seqs")
    print("  All benchmark datasets have seq_len_max ≤ 2000 bp → truncation rate ≈ 0% for all models")

print("\n" + "="*70)
print("DONE")
print("="*70)
