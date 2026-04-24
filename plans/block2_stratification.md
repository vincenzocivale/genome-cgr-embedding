# Block 2 — Stratificazione

**Esperimenti:** 9, 10, 11, 12, 13
**GPU:** Solo per Esp 13 (few-shot con FM embedding)
**Priorità:** 🟡 Media

## Obiettivo

Produrre le analisi stratificate richieste dal paper: win/loss e ΔMCC per categoria biologica (Esp 9), bin di lunghezza sequenza (Esp 10), GC vs k ottimale (Esp 11), training size / class balance vs vantaggio FM (Esp 12), e analisi few-shot a 4 frazioni (Esp 13).

Esp 9–12 sono puri post-processing di `records_rf.csv` + `records_best_k_analysis.csv`. Esp 13 richiede un nuovo script di training.

---

## Nuovo script: `src/analysis/stratification_analysis.py`

Uno script unico che produce i 4 CSV per Esp 9–12. Tutti i confronti sono rispetto a k-mer k=6 (baseline del paper) per 3 FM: NTv3-650M, HyenaDNA-160k, DNABERT-2-117M.

### Esp 9 — Win/loss & ΔMCC per biocat

```python
import pandas as pd
import numpy as np

rf = pd.read_csv('results/classification/records_rf.csv', index_col=0)
meta = pd.read_csv('results/classification/records_best_k_analysis.csv').set_index('dataset')
df = rf.join(meta[['biocat']], how='left')

MODELS = ['NTv3_650M_pre', 'hyenadna-medium-160k-seqlen-hf', 'DNABERT-2-117M']
BASE = 'kmer_k6_MCC'

rows = []
for model in MODELS:
    fm_col = f'fm_{model}_MCC'
    df[f'delta_{model}'] = df[fm_col] - df[BASE]
    for biocat, g in df.groupby('biocat'):
        rows.append({
            'model': model,
            'biocat': biocat,
            'n_datasets': len(g),
            'mean_delta': g[f'delta_{model}'].mean(),
            'median_delta': g[f'delta_{model}'].median(),
            'fm_wins': (g[f'delta_{model}'] > 0).sum(),
            'kmer_wins': (g[f'delta_{model}'] < 0).sum(),
            'ties': (g[f'delta_{model}'] == 0).sum(),
        })
pd.DataFrame(rows).to_csv('results/analysis/stratification_biocat.csv', index=False)
```

### Esp 10 — Win/loss per bin di seq_len

```python
bins = [0, 100, 300, 600, np.inf]
labels = ['≤100', '101-300', '301-600', '>600']
df['seq_len_bin'] = pd.cut(df['seq_len_mean'], bins=bins, labels=labels, include_lowest=True)

rows = []
for model in MODELS:
    for bin_label, g in df.groupby('seq_len_bin', observed=True):
        rows.append({
            'model': model, 'seq_len_bin': bin_label,
            'n_datasets': len(g),
            'mean_delta': g[f'delta_{model}'].mean(),
            'median_delta': g[f'delta_{model}'].median(),
            'fm_wins': (g[f'delta_{model}'] > 0).sum(),
            'kmer_wins': (g[f'delta_{model}'] < 0).sum(),
        })
pd.DataFrame(rows).to_csv('results/analysis/stratification_seqlen.csv', index=False)
```

### Esp 11 — GC content vs k ottimale

```python
meta_full = pd.read_csv('results/classification/records_best_k_analysis.csv')
meta_full['gc_bin'] = pd.qcut(meta_full['gc_content_mean'], q=4,
                               labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
ct = pd.crosstab(meta_full['gc_bin'], meta_full['best_k'], margins=True, margins_name='total')
ct.to_csv('results/analysis/gc_vs_best_k.csv')

# Correlazione Spearman gc vs best_k
from scipy.stats import spearmanr
rho, p = spearmanr(meta_full['gc_content_mean'], meta_full['best_k'])
with open('results/analysis/gc_vs_best_k_correlation.txt', 'w') as f:
    f.write(f'Spearman rho={rho:.3f}, p={p:.4f}\n')
```

### Esp 12 — Class balance / n_train vs Δ FM

```python
# Scatter data + Spearman
from scipy.stats import spearmanr
out_rows = []
for model in MODELS:
    for xvar in ['n_train', 'class_balance']:
        rho, p = spearmanr(df[xvar], df[f'delta_{model}'], nan_policy='omit')
        out_rows.append({'model': model, 'feature': xvar, 'spearman_rho': rho, 'p_value': p})
pd.DataFrame(out_rows).to_csv('results/analysis/training_size_vs_delta.csv', index=False)

# Bin analysis
for xvar, bins_edges, labels_bin in [
    ('n_train', [0, 500, 2000, 10000, np.inf], ['≤500','501-2000','2001-10000','>10000']),
    ('class_balance', [0, 0.1, 0.3, 0.5, 1.01], ['≤0.1','0.1-0.3','0.3-0.5','≥0.5']),
]:
    df[f'{xvar}_bin'] = pd.cut(df[xvar], bins=bins_edges, labels=labels_bin, include_lowest=True)
    # groupby as above
```

### Run

```bash
python3 src/analysis/stratification_analysis.py
```

**Output:**
- `results/analysis/stratification_biocat.csv`
- `results/analysis/stratification_seqlen.csv`
- `results/analysis/gc_vs_best_k.csv`
- `results/analysis/gc_vs_best_k_correlation.txt`
- `results/analysis/training_size_vs_delta.csv`

---

## Esp 13 — Few-shot analysis

### Nuovo script: `src/scripts/classification/train_few_shot_rf.py`

**Protocollo:**
- Fractions: `[0.10, 0.25, 0.50, 1.00]`
- Subsampling stratificato con `StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=42)`.
- Training: `train_rf_fast` (nessun GridSearch — inutile su subset piccoli).
- Valutazione: test set completo (non sottocampionato).
- Dataset selection: ~15-18 dataset rappresentativi.

### Selezione dataset

Criteri (da `records_best_k_analysis.csv`):
- Almeno 2 dataset per biocat (6 categorie × 2 = 12 min)
- Mix di bin seq_len: ≤100, 101-300, 301-600, >600 bp
- Filtro: `n_train >= 500` (per garantire che 10% = ≥50 sample)

```python
import pandas as pd
meta = pd.read_csv('results/classification/records_best_k_analysis.csv')
meta = meta[meta['n_train'] >= 500]
# Per ciascun biocat, prendi i 2-3 dataset con copertura più larga di seq_len
selected = meta.groupby('biocat', group_keys=False).apply(
    lambda g: g.nlargest(3, 'n_train')
)['dataset'].tolist()
```

### Schema output

`results/classification/records_few_shot.csv`:
- Index: `dataset`
- Colonne per ogni `(method, frac, metric)`:
  - `kmer_k6_frac{10,25,50,100}_MCC`, `kmer_k6_frac{10,25,50,100}_AUROC`
  - `fm_{model}_frac{10,25,50,100}_MCC`, `fm_{model}_frac{10,25,50,100}_AUROC` per 3 FM

### Run

```bash
# Crea prima lo script, poi:
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_few_shot_rf.py \
    --methods kmer_k6 fm_NTv3_650M_pre fm_hyenadna-medium-160k-seqlen-hf fm_DNABERT-2-117M \
    --fractions 0.10 0.25 0.50 1.00 \
    --seed 42 --n-workers 1 --fm-batch-size 16
```

**Tempo stimato:** 15 dataset × 4 metodi × 4 frazioni × eval RF = ~3-5 h su 1 GPU.

---

## Verification

```bash
python3 -c "
import os, pandas as pd
files = [
    'results/analysis/stratification_biocat.csv',
    'results/analysis/stratification_seqlen.csv',
    'results/analysis/gc_vs_best_k.csv',
    'results/analysis/training_size_vs_delta.csv',
    'results/classification/records_few_shot.csv',
]
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f'{f}: rows={len(df)}, cols={len(df.columns)}')
    else:
        print(f'{f}: MISSING')

# Few-shot coverage
fs = pd.read_csv('results/classification/records_few_shot.csv', index_col=0)
expected_cols = [f'fm_NTv3_650M_pre_frac{f}_MCC' for f in [10,25,50,100]]
for c in expected_cols:
    assert c in fs.columns, f'{c} missing'
print('Few-shot datasets:', len(fs), '(expected ~15)')
"
```
