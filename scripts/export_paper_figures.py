"""
Generate all publication-ready figures from results CSVs.

Reads from results/ directories and writes PDFs to results/figures/.
Must be run from the repository root.

Usage:
    python3 scripts/export_paper_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Style for paper figures
sns.set_context('paper', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
})

OUT_DIR = 'results/figures_pdf'
os.makedirs(OUT_DIR, exist_ok=True)

RECORDS = 'results/classification/records.csv'
RECORDS_PCA = 'results/classification/records_pca.csv'
RECORDS_CONCAT = 'results/concat/records_concat_best.csv'
RECORDS_DECOMP = 'results/classification/records_decomposition.csv'
RECORDS_BESTK = 'results/classification/records_best_k_analysis.csv'
RECORDS_EFF = 'results/exploratory/efficiency.csv'
RECORDS_REG = 'results/regression/lra_records.csv'

FM_LABELS = {
    'fm_NTv3_650M_pre_MCC': 'NTv3-650M',
    'fm_hyenadna-medium-160k-seqlen-hf_MCC': 'HyenaDNA-160k',
    'fm_DNABERT-2-117M_MCC': 'DNABERT-2',
}

CAT_LABELS = {
    'histone': 'Histone (Yeast)',
    'methylation': 'Methylation',
    'promoter': 'Promoter',
    'splice': 'Splice',
    'TFBS': 'TFBS',
    'enhancer/generic': 'Enhancer/Generic',
}


def load_records():
    df = pd.read_csv(RECORDS, index_col=0)
    df.index.name = 'dataset'
    df['group'] = df.index.str.split('/').str[0]
    df['short'] = df.index.str.split('/').str[1]
    return df


def kmer_mcc_cols(df, include_multi=True):
    cols = [c for c in df.columns if c.startswith('kmer_k') and c.endswith('_MCC')]
    if include_multi:
        cols += [c for c in df.columns if c.startswith('kmer_multi') and c.endswith('_MCC')]
    return cols


def best_kmer(df):
    cols = kmer_mcc_cols(df)
    return df[cols].apply(pd.to_numeric, errors='coerce').max(axis=1).rename('best_kmer_MCC')


def dataset_palette(df):
    groups = sorted(df['group'].dropna().unique())
    cmap = plt.colormaps['tab20']
    return {g: cmap(i) for i, g in enumerate(groups)}


# Load data
df = load_records()
df_pca = pd.read_csv(RECORDS_PCA, index_col=0)
df_pca = df_pca[~df_pca.index.duplicated(keep='last')]
df_pca['group'] = df_pca.index.str.split('/').str[0]

df_concat = pd.read_csv(RECORDS_CONCAT).set_index('dataset')
df_decomp = pd.read_csv(RECORDS_DECOMP, index_col=0)
df_bestk = pd.read_csv(RECORDS_BESTK, index_col=0)
df_eff = pd.read_csv(RECORDS_EFF)
df_reg = pd.read_csv(RECORDS_REG)

palette = dataset_palette(df)

# ── Figure 1: FM vs best k-mer (per dataset) ─────────────────────────
TIE_EPS = 0.01
best_k = best_kmer(df)
fm_cols = list(FM_LABELS.keys())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, col in zip(axes, fm_cols):
    x = pd.to_numeric(df[col], errors='coerce')
    y = best_k

    for g, sub in df.groupby('group'):
        ax.scatter(x.loc[sub.index], y.loc[sub.index],
                   label=g, alpha=0.85, s=50, color=palette[g],
                   edgecolors='white', lw=0.4)

    lo = np.nanmin([x.min(), y.min()]) - 0.02
    hi = np.nanmax([x.max(), y.max()]) + 0.02
    ax.plot([lo, hi], [lo, hi], ls='--', c='gray', lw=1.2)

    valid = pd.concat([x, y], axis=1).dropna()
    diff = valid.iloc[:, 0] - valid.iloc[:, 1]
    fm_wins = (diff > TIE_EPS).sum()
    kmer_wins = (diff < -TIE_EPS).sum()
    ties = ((diff >= -TIE_EPS) & (diff <= TIE_EPS)).sum()
    ax.text(0.03, 0.97, f'FM wins: {fm_wins}\nK-mer wins: {kmer_wins}\nTies: {ties}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    ax.set_title(FM_LABELS[col])
    ax.set_xlabel('FM MCC')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

axes[0].set_ylabel('Best k-mer MCC')
axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                title='Group', fontsize=11, title_fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_fm_vs_kmer_scatter.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 2: Δ MCC by category ─────────────────────────────────────
cat = df_bestk[['biocat', 'seq_len_mean', 'gc_content_mean', 'best_k']].copy()
cat['best_kmer_MCC'] = best_kmer(df)
cat['ntv3_MCC'] = pd.to_numeric(df['fm_NTv3_650M_pre_MCC'], errors='coerce')
cat['delta_ntv3_kmer'] = cat['ntv3_MCC'] - cat['best_kmer_MCC']
cat['R2'] = pd.to_numeric(df_decomp['ridge_k6_NTv3_650M_pre_R2'], errors='coerce')
cat['proj_MCC'] = pd.to_numeric(df_decomp['proj_k6_NTv3_650M_pre_MCC'], errors='coerce')
cat['resid_MCC'] = pd.to_numeric(df_decomp['resid_k6_NTv3_650M_pre_MCC'], errors='coerce')
cat['synergy'] = cat['ntv3_MCC'] - np.maximum(cat['proj_MCC'], cat['resid_MCC'])
cat['resid_gt_kmer'] = (cat['resid_MCC'] > cat['best_kmer_MCC']).astype(int)
cat['concat_MCC'] = pd.to_numeric(df_concat['MCC'], errors='coerce')
cat['delta_concat'] = cat['concat_MCC'] - cat[['best_kmer_MCC', 'ntv3_MCC']].max(axis=1)
cat['biocat_label'] = cat['biocat'].map(CAT_LABELS)

order = (cat.groupby('biocat_label')['delta_ntv3_kmer']
         .median()
         .sort_values()
         .index.tolist())

fig, ax = plt.subplots(figsize=(7.2, 4.2))
sns.boxplot(data=cat, y='biocat_label', x='delta_ntv3_kmer', order=order,
            color='#E6E6E6', fliersize=0, linewidth=1.2, ax=ax)
sns.stripplot(data=cat, y='biocat_label', x='delta_ntv3_kmer', order=order,
              size=5, alpha=0.75, color='#2B6CB0', ax=ax)
ax.axvline(0, ls='--', c='gray', lw=1.2)
ax.set_xlabel('Δ MCC (NTv3 − best k-mer)')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_delta_by_category.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 3: sequence length effect ────────────────────────────────
valid = (cat[['seq_len_mean', 'delta_ntv3_kmer', 'biocat_label']]
         .dropna()
         .rename(columns={'seq_len_mean': 'seq_len'}))

rho_all, p_all = stats.spearmanr(valid['seq_len'], valid['delta_ntv3_kmer'])
valid_no = valid[valid['biocat_label'] != 'Splice']
rho_no, p_no = stats.spearmanr(valid_no['seq_len'], valid_no['delta_ntv3_kmer'])

max_len = valid['seq_len'].max()
bins = [0, 100, 300, 600, max_len + 1]
labels = ['≤100', '101–300', '301–600', '>600']
valid['len_bin'] = pd.cut(valid['seq_len'], bins=bins, labels=labels, include_lowest=True)

bin_stats = (valid.groupby('len_bin')
             .agg(med_len=('seq_len', 'median'),
                  med_delta=('delta_ntv3_kmer', 'median'))
             .reset_index())

cat_order = ['Histone (Yeast)', 'Methylation', 'Promoter', 'Splice', 'TFBS', 'Enhancer/Generic']
cat_palette = dict(zip(cat_order, sns.color_palette('tab10', n_colors=len(cat_order))))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'width_ratios': [3, 2]})

sns.scatterplot(data=valid, x='seq_len', y='delta_ntv3_kmer', hue='biocat_label',
                hue_order=cat_order, palette=cat_palette, s=55, alpha=0.85,
                edgecolor='white', linewidth=0.4, ax=ax0)
ax0.plot(bin_stats['med_len'], bin_stats['med_delta'], color='black', lw=2, marker='o',
         label='Median per bin')
ax0.axhline(0, ls='--', c='gray', lw=1.2)
ax0.set_xlabel('Mean sequence length (bp)')
ax0.set_ylabel('Δ MCC (NTv3 − best k-mer)')
ax0.set_title('Δ MCC vs sequence length')
ax0.text(0.02, 0.98,
         f"Spearman r = {rho_all:.3f} (p={p_all:.3g})\nExcluding splice: r = {rho_no:.3f} (p={p_no:.3g})",
         transform=ax0.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
ax0.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Category', fontsize=9, title_fontsize=10)

sns.boxplot(data=valid, x='len_bin', y='delta_ntv3_kmer', color='#E6E6E6',
            fliersize=0, linewidth=1.2, ax=ax1)
sns.stripplot(data=valid, x='len_bin', y='delta_ntv3_kmer',
              color='#2B6CB0', size=5, alpha=0.75, ax=ax1)
ax1.axhline(0, ls='--', c='gray', lw=1.2)
ax1.set_xlabel('Length bin (bp)')
ax1.set_ylabel('')
ax1.set_title('Δ MCC by length bin')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_length_effect.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 4: best k distribution + GC relation ─────────────────────
bk = df_bestk.copy()
bk['biocat_label'] = bk['biocat'].map(CAT_LABELS)

rho_gc, p_gc = stats.spearmanr(bk['gc_content_mean'], bk['best_k'])

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={'width_ratios': [1.1, 1.4]})

counts = bk['best_k'].value_counts().sort_index()
ax0.bar(counts.index.astype(int), counts.values, color='#4C78A8')
ax0.set_xlabel('Best k')
ax0.set_ylabel('N datasets')
ax0.set_title('Best k distribution')

sns.boxplot(data=bk, x='best_k', y='gc_content_mean', color='#E6E6E6',
            fliersize=0, linewidth=1.2, ax=ax1)
sns.stripplot(data=bk, x='best_k', y='gc_content_mean', color='#2B6CB0',
              size=5, alpha=0.75, ax=ax1)
ax1.set_xlabel('Best k')
ax1.set_ylabel('Mean GC content')
ax1.set_title('GC vs best k')
ax1.text(0.02, 0.98, f"Spearman r = {rho_gc:.3f} (p={p_gc:.3g})",
         transform=ax1.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_bestk_gc.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 5a: R² distribution ─────────────────────────────────────
r2 = pd.DataFrame({
    'NTv3-650M': pd.to_numeric(df_decomp['ridge_k6_NTv3_650M_pre_R2'], errors='coerce'),
    'HyenaDNA-160k': pd.to_numeric(df_decomp['ridge_k6_hyenadna-medium-160k-seqlen-hf_R2'], errors='coerce'),
    'DNABERT-2': pd.to_numeric(df_decomp['ridge_k6_DNABERT-2-117M_R2'], errors='coerce'),
})
r2_long = r2.melt(var_name='Model', value_name='R2')

fig, ax = plt.subplots(figsize=(6.5, 4.2))
sns.boxplot(data=r2_long, x='Model', y='R2', color='#E6E6E6', fliersize=0, ax=ax)
sns.stripplot(data=r2_long, x='Model', y='R2', color='#2B6CB0', size=4, alpha=0.6, ax=ax)
ax.set_xlabel('')
ax.set_ylabel('R² (k=6 → embedding)')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_r2_distribution.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 5b: projection/residual vs k-mer ─────────────────────────
proj = pd.to_numeric(df_decomp['proj_k6_NTv3_650M_pre_MCC'], errors='coerce')
resid = pd.to_numeric(df_decomp['resid_k6_NTv3_650M_pre_MCC'], errors='coerce')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True)
for ax, y, title in [
    (axes[0], proj, 'Projection (k=6) vs best k-mer'),
    (axes[1], resid, 'Residual (k=6) vs best k-mer')
]:
    ax.scatter(best_k, y, s=50, alpha=0.75, color='#2B6CB0', edgecolors='white', lw=0.4)
    lo = np.nanmin([best_k.min(), y.min()]) - 0.02
    hi = np.nanmax([best_k.max(), y.max()]) + 0.02
    ax.plot([lo, hi], [lo, hi], ls='--', c='gray', lw=1.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(title)
    ax.set_xlabel('Best k-mer MCC')
axes[0].set_ylabel('MCC')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig6_proj_resid_vs_kmer.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 6: synergy by category ──────────────────────────────────
sy = cat[['biocat_label', 'synergy']].dropna()
order = sy.groupby('biocat_label')['synergy'].median().sort_values().index.tolist()

fig, ax = plt.subplots(figsize=(7.2, 4.2))
sns.boxplot(data=sy, y='biocat_label', x='synergy', order=order,
            color='#E6E6E6', fliersize=0, linewidth=1.2, ax=ax)
sns.stripplot(data=sy, y='biocat_label', x='synergy', order=order,
              size=5, alpha=0.75, color='#2B6CB0', ax=ax)
ax.axvline(0, ls='--', c='gray', lw=1.2)
ax.set_xlabel('Synergy (full − max(proj, resid))')
ax.set_ylabel('')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig7_synergy_by_category.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 7: component comparison by model ────────────────────────
comp = []
comp.append(pd.DataFrame({'Model': 'NTv3-650M', 'Component': 'Full', 'MCC': pd.to_numeric(df['fm_NTv3_650M_pre_MCC'], errors='coerce')}))
comp.append(pd.DataFrame({'Model': 'NTv3-650M', 'Component': 'Projection', 'MCC': proj}))
comp.append(pd.DataFrame({'Model': 'NTv3-650M', 'Component': 'Residual', 'MCC': resid}))
comp.append(pd.DataFrame({'Model': 'HyenaDNA-160k', 'Component': 'Full', 'MCC': pd.to_numeric(df['fm_hyenadna-medium-160k-seqlen-hf_MCC'], errors='coerce')}))
comp.append(pd.DataFrame({'Model': 'HyenaDNA-160k', 'Component': 'Projection', 'MCC': pd.to_numeric(df_decomp['proj_k6_hyenadna-medium-160k-seqlen-hf_MCC'], errors='coerce')}))
comp.append(pd.DataFrame({'Model': 'HyenaDNA-160k', 'Component': 'Residual', 'MCC': pd.to_numeric(df_decomp['resid_k6_hyenadna-medium-160k-seqlen-hf_MCC'], errors='coerce')}))
comp.append(pd.DataFrame({'Model': 'DNABERT-2', 'Component': 'Full', 'MCC': pd.to_numeric(df['fm_DNABERT-2-117M_MCC'], errors='coerce')}))
comp.append(pd.DataFrame({'Model': 'DNABERT-2', 'Component': 'Projection', 'MCC': pd.to_numeric(df_decomp['proj_k6_DNABERT-2-117M_MCC'], errors='coerce')}))
comp.append(pd.DataFrame({'Model': 'DNABERT-2', 'Component': 'Residual', 'MCC': pd.to_numeric(df_decomp['resid_k6_DNABERT-2-117M_MCC'], errors='coerce')}))

comp = pd.concat(comp, ignore_index=True).dropna()

fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
for ax, model in zip(axes, ['NTv3-650M', 'HyenaDNA-160k', 'DNABERT-2']):
    sub = comp[comp['Model'] == model]
    sns.boxplot(data=sub, x='Component', y='MCC', color='#E6E6E6', fliersize=0, ax=ax)
    sns.stripplot(data=sub, x='Component', y='MCC', color='#2B6CB0', size=4, alpha=0.6, ax=ax)
    ax.set_title(model)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
axes[0].set_ylabel('MCC')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig8_component_by_model.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 8: concatenation effect ─────────────────────────────────
delta_c = cat['delta_concat'].dropna()

fig, ax = plt.subplots(figsize=(6.5, 4.0))
sns.boxplot(x=delta_c, color='#E6E6E6', fliersize=0, linewidth=1.2, ax=ax)
sns.stripplot(x=delta_c, color='#2B6CB0', size=5, alpha=0.75, ax=ax)
ax.axvline(0, ls='--', c='gray', lw=1.2)
ax.set_xlabel('Δ MCC (concat − best single)')
ax.set_yticks([])
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig9_concat_effect.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 9: PCA effect ───────────────────────────────────────────
pca_cols = {
    'NTv3-650M': ('pca_fm_NTv3_650M_pre_MCC', 'fm_NTv3_650M_pre_MCC'),
    'HyenaDNA-160k': ('pca_fm_hyenadna-medium-160k-seqlen-hf_MCC', 'fm_hyenadna-medium-160k-seqlen-hf_MCC'),
    'DNABERT-2': ('pca_fm_DNABERT-2-117M_MCC', 'fm_DNABERT-2-117M_MCC'),
    'k=4': ('pca_kmer_k4_MCC', 'kmer_k4_MCC'),
    'k=5': ('pca_kmer_k5_MCC', 'kmer_k5_MCC'),
    'k=6': ('pca_kmer_k6_MCC', 'kmer_k6_MCC'),
}

delta_list = []
for name, (pca_col, raw_col) in pca_cols.items():
    if pca_col in df_pca.columns and raw_col in df.columns:
        d = pd.to_numeric(df_pca[pca_col], errors='coerce') - pd.to_numeric(df[raw_col], errors='coerce')
        delta_list.append(pd.DataFrame({'Method': name, 'Δ MCC': d}))

if delta_list:
    delta_df = pd.concat(delta_list, ignore_index=True).dropna()
    order = (delta_df.groupby('Method')['Δ MCC'].median().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    sns.boxplot(data=delta_df, y='Method', x='Δ MCC', order=order, color='#E6E6E6',
                fliersize=0, linewidth=1.2, ax=ax)
    sns.stripplot(data=delta_df, y='Method', x='Δ MCC', order=order, color='#2B6CB0',
                  size=4, alpha=0.6, ax=ax)
    ax.axvline(0, ls='--', c='gray', lw=1.2)
    ax.set_xlabel('Δ MCC (PCA − native)')
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig10_pca_effect.pdf'), bbox_inches='tight')
    plt.close(fig)

# ── Figure 10: compute cost ─────────────────────────────────────────
eff = df_eff.copy()
eff['seq_len'] = pd.to_numeric(eff['seq_len'], errors='coerce')
eff['GFLOPS_per_seq'] = pd.to_numeric(eff['GFLOPS_per_seq'], errors='coerce')

method_map = {
    'fm_NTv3_650M_pre': 'NTv3-650M',
    'fm_hyenadna-medium-160k-seqlen-hf': 'HyenaDNA-160k',
    'fm_DNABERT-2-117M': 'DNABERT-2',
    'kmer_k6': 'k-mer (k=6)',
}
eff['Method'] = eff['method'].map(method_map)
eff = eff.dropna(subset=['Method'])

fig, ax = plt.subplots(figsize=(7.0, 4.2))
for m, sub in eff.groupby('Method'):
    sub = sub.sort_values('seq_len')
    ax.plot(sub['seq_len'], sub['GFLOPS_per_seq'], marker='o', lw=2, label=m)

ax.set_yscale('log')
ax.set_xlabel('Sequence length (bp)')
ax.set_ylabel('GFLOPS per sequence (log)')
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig11_cost_scaling.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 11: correlations ─────────────────────────────────────────
corr = pd.DataFrame(index=df.index)
corr['seq_len'] = pd.to_numeric(df['seq_len_mean'], errors='coerce')
corr['gc'] = pd.to_numeric(df['gc_content_mean'], errors='coerce')
corr['class_balance'] = pd.to_numeric(df['class_balance'], errors='coerce')
corr['delta_full_kmer'] = pd.to_numeric(df['fm_NTv3_650M_pre_MCC'], errors='coerce') - best_k
corr['delta_resid_kmer'] = pd.to_numeric(df_decomp['resid_k6_NTv3_650M_pre_MCC'], errors='coerce') - best_k
corr['gap_full_proj'] = pd.to_numeric(df['fm_NTv3_650M_pre_MCC'], errors='coerce') - pd.to_numeric(df_decomp['proj_k6_NTv3_650M_pre_MCC'], errors='coerce')
corr['synergy'] = cat['synergy']
corr['R2'] = pd.to_numeric(df_decomp['ridge_k6_NTv3_650M_pre_R2'], errors='coerce')

corr = corr.dropna()

corr = corr.rename(columns={
    'seq_len': 'seq_len',
    'gc': 'gc',
    'class_balance': 'class_balance',
    'delta_full_kmer': 'Δ full−kmer',
    'delta_resid_kmer': 'Δ resid−kmer',
    'gap_full_proj': 'gap full−proj',
    'synergy': 'synergy',
    'R2': 'R²',
})

rho = corr.corr(method='spearman')

fig, ax = plt.subplots(figsize=(7.0, 5.6))
sns.heatmap(rho, cmap='coolwarm', vmin=-1, vmax=1, center=0,
            square=True, cbar_kws={'label': 'Spearman r'}, ax=ax)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig12_correlations.pdf'), bbox_inches='tight')
plt.close(fig)

# ── Figure 12: regression R2 ────────────────────────────────────────
reg = df_reg.copy().set_index('dataset')

r2_cols = [c for c in reg.columns if c.endswith('_R2')]
reg = reg[reg[r2_cols].notna().any(axis=1)]

reg_k = reg[[c for c in reg.columns if c.startswith('kmer_k') and c.endswith('_R2')]]
reg['best_kmer_R2'] = reg_k.apply(pd.to_numeric, errors='coerce').max(axis=1)

reg_table = reg[[
    'best_kmer_R2',
    'fm_NTv3_650M_pre_R2',
    'fm_hyenadna-large-1m-seqlen-hf_R2',
    'fm_DNABERT-2-117M_R2'
]].copy()
reg_table = reg_table.rename(columns={
    'best_kmer_R2': 'k-mer (best k)',
    'fm_NTv3_650M_pre_R2': 'NTv3-650M',
    'fm_hyenadna-large-1m-seqlen-hf_R2': 'HyenaDNA-1M',
    'fm_DNABERT-2-117M_R2': 'DNABERT-2',
}).round(3)

reg_long = reg_table.reset_index().melt(id_vars='dataset', var_name='Method', value_name='R2')

fig, ax = plt.subplots(figsize=(6.8, 3.6))
sns.barplot(data=reg_long, y='dataset', x='R2', hue='Method', orient='h', ci=None, ax=ax)
ax.set_xlabel('R2')
ax.set_ylabel('')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig13_regression_r2.pdf'), bbox_inches='tight')
plt.close(fig)

print(f"Saved figures to {OUT_DIR}")
