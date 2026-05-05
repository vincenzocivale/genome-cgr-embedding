#!/usr/bin/env python3
"""Generate summary MCC table: median MCC per model and dataset group."""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path('/home/oem/genome-cgr-embedding')
RESULTS = ROOT / 'results'

# Load RF results
rf = pd.read_csv(RESULTS / 'classification/records_rf.csv')

# Add dataset group
rf['group'] = rf['dataset'].apply(lambda x: x.split('/')[0])

# Define models: (column_name, display_label)
MODELS = [
    ('kmer_k4_MCC', 'k-mer (k=4)'),
    ('kmer_k5_MCC', 'k-mer (k=5)'),
    ('kmer_k6_MCC', 'k-mer (k=6)'),
    ('kmer_multi_k4_5_6_MCC', 'k-mer (multiscale)'),
    ('fm_NTv3_650M_pre_MCC', 'NTv3 (650M)'),
    ('fm_hyenadna-medium-160k-seqlen-hf_MCC', 'HyenaDNA'),
    ('fm_DNABERT-2-117M_MCC', 'DNABERT-2'),
    ('fm_caduceus-ph_seqlen-131k_d_model-256_n_layer-16_MCC', 'Caduceus'),
    ('fm_evo2_1b_base_MCC', 'Evo2 (1B)'),
]

# Get unique groups in order they appear
groups = sorted(rf['group'].unique())

# Build summary table
summary_data = []
for group in groups:
    group_data = rf[rf['group'] == group]
    row = {'Dataset Group': group, 'N': len(group_data)}

    for col, label in MODELS:
        if col in group_data.columns:
            median_mcc = group_data[col].median()
            row[label] = f'{median_mcc:.3f}'
        else:
            row[label] = 'N/A'

    summary_data.append(row)

# Create DataFrame
summary_df = pd.DataFrame(summary_data)

# Add overall median row
overall_row = {'Dataset Group': 'Overall (all 57)', 'N': len(rf)}
for col, label in MODELS:
    if col in rf.columns:
        median_mcc = rf[col].median()
        overall_row[label] = f'{median_mcc:.3f}'
    else:
        overall_row[label] = 'N/A'

summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

print("\n" + "="*180)
print("TABLE S1: Median MCC by Model and Dataset Group")
print("="*180)
print(summary_df.to_string(index=False))
print("="*180)

# Save as CSV
csv_path = RESULTS / 'summary_mcc_by_group.csv'
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")

# Also save as LaTeX table for paper
latex_path = RESULTS / 'summary_mcc_by_group.tex'
with open(latex_path, 'w') as f:
    f.write("\\begin{table}[!h]\n")
    f.write("\\centering\n")
    f.write("\\small\n")
    f.write("\\caption{Median MCC across models and dataset groups (N=57 total datasets).}\n")
    f.write("\\label{tab:summary-mcc}\n")
    f.write(summary_df.to_latex(index=False, escape=False))
    f.write("\\end{table}\n")

print(f"Saved → {latex_path}")
