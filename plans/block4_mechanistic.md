# Block 4 — Analisi meccanicistica

**Esperimenti:** 24 (Esp 22, 23 già complete)
**GPU:** No
**Priorità:** 🟡 Media (dipende dal completamento di Block 3)

## Obiettivo

Produrre la matrice di correlazioni dataset-level che lega caratteristiche del dataset (seq_len, GC, class_balance) alle metriche chiave della decomposizione (Δ full-kmer, Δ resid-kmer, gap full-proj, synergy, R² ridge/MLP). È l'unica parte mancante di questo blocco.

---

## Stato di Esp 22–23

- **Esp 22 — Motif-level splice residuo:** ✅ `results/exploratory/splice_residual_motif_overlap.csv` contiene 12 righe (4 dataset splice × 3 motif: GT-AG donor/acceptor, branch point, polypyrimidine) con `overlap_rate`, `mean_delta_resid_l2`, `std_delta_resid_l2`.
- **Esp 23 — Confronto motif noti:** ✅ Incluso in Esp 22 via colonna `motif`. Non serve action aggiuntiva.

Nessun lavoro richiesto — solo documentazione in paper methods.

---

## Esp 24 — Dataset-level correlations

### Nuovo script: `src/analysis/dataset_correlations.py`

**Input:**
- `results/classification/records_rf.csv` — metadata dataset + MCC per k-mer/FM
- `results/classification/records_best_k_analysis.csv` — `best_k`, `biocat`
- `results/decomposition/records_decomposition.csv` — R² ridge/MLP, proj/resid MCC
- `results/concat/records_concat_best.csv` — synergy FM + k-mer

**Feature matrix per 57 dataset:**

```python
import pandas as pd
import numpy as np

rf   = pd.read_csv('results/classification/records_rf.csv', index_col=0)
meta = pd.read_csv('results/classification/records_best_k_analysis.csv').set_index('dataset')
dec  = pd.read_csv('results/decomposition/records_decomposition.csv', index_col=0)
cat  = pd.read_csv('results/concat/records_concat_best.csv', index_col=0)

MODELS = ['NTv3_650M_pre', 'hyenadna-medium-160k-seqlen-hf', 'DNABERT-2-117M']
# dopo Block 1: aggiungere 'caduceus-ph'

feat = rf[['seq_len_mean', 'gc_content_mean', 'class_balance', 'n_train']].copy()
feat = feat.join(meta[['best_k']], how='left')

for m in MODELS:
    fm_mcc   = f'fm_{m}_MCC'
    kmer_mcc = 'kmer_k6_MCC'
    feat[f'delta_full_{m}']    = rf[fm_mcc] - rf[kmer_mcc]
    feat[f'delta_resid_{m}']   = dec.get(f'resid_k6_{m}_MCC', pd.Series(index=rf.index)) - rf[kmer_mcc]
    feat[f'gap_full_proj_{m}'] = rf[fm_mcc] - dec.get(f'proj_k6_{m}_MCC', pd.Series(index=rf.index))
    feat[f'R2_ridge_{m}']      = dec.get(f'ridge_k6_{m}_R2', pd.Series(index=rf.index))
    feat[f'R2_mlp_{m}']        = dec.get(f'ridge_k6_{m}_R2__pool_mean__map_mlp', pd.Series(index=rf.index))

    cat_col = f'concat_best_fm_{m}_MCC'  # nome esatto da verificare in records_concat_best.csv
    if cat_col in cat.columns:
        feat[f'synergy_{m}'] = cat[cat_col] - np.maximum(rf[fm_mcc], rf[kmer_mcc])

# Salva feature matrix
feat.to_csv('results/analysis/dataset_feature_matrix.csv')

# Matrice di correlazione Spearman (feature × feature)
corr = feat.corr(method='spearman')
corr.to_csv('results/analysis/dataset_correlations.csv')

# Versione lunga (feat1, feat2, rho, p) per analisi puntuali
from scipy.stats import spearmanr
long_rows = []
for c1 in feat.columns:
    for c2 in feat.columns:
        if c1 >= c2:
            continue
        v1, v2 = feat[c1].dropna(), feat[c2].dropna()
        idx = v1.index.intersection(v2.index)
        if len(idx) < 10:
            continue
        rho, p = spearmanr(v1.loc[idx], v2.loc[idx])
        long_rows.append({'feat1': c1, 'feat2': c2, 'n': len(idx), 'rho': rho, 'p': p})
pd.DataFrame(long_rows).sort_values('p').to_csv('results/analysis/dataset_correlations_long.csv', index=False)
```

### Run

```bash
python3 src/analysis/dataset_correlations.py
```

**Output:**
- `results/analysis/dataset_feature_matrix.csv` — matrice 57 × ~25 colonne
- `results/analysis/dataset_correlations.csv` — matrice Spearman (quadrata)
- `results/analysis/dataset_correlations_long.csv` — versione lunga con p-value

---

## Dipendenze

**Block 4 dipende da Block 3 completato:**
- Colonne `ridge_k6_{m}_R2` per tutti e 3 i FM — oggi solo NTv3 complete.
- Colonne `ridge_k6_{m}_R2__pool_mean__map_mlp` — oggi solo 1/57 per NTv3.
- Colonne `proj_k6_{m}_MCC`, `resid_k6_{m}_MCC` — oggi HyenaDNA e DNABERT-2 hanno k=6 ma k=4/5 mancanti (Block 3 chiude questo gap; Esp 24 usa k=6).

Se Block 3 non è ancora completo, lo script produce comunque l'output ma con NaN per le colonne mancanti; documenta esplicitamente lo stato nel CSV.

---

## Verification

```bash
python3 -c "
import pandas as pd

m = pd.read_csv('results/analysis/dataset_feature_matrix.csv', index_col=0)
c = pd.read_csv('results/analysis/dataset_correlations.csv', index_col=0)
l = pd.read_csv('results/analysis/dataset_correlations_long.csv')

print('Feature matrix:', m.shape)
print('Correlation matrix:', c.shape)
print('Top-10 correlations (|rho| more extreme):')
print(l.reindex(l['rho'].abs().sort_values(ascending=False).index).head(10))

# Sanity: must contain at least delta_full, R2_ridge per 3 FM
expected_suffixes = ['NTv3_650M_pre', 'hyenadna-medium-160k-seqlen-hf', 'DNABERT-2-117M']
for s in expected_suffixes:
    assert f'delta_full_{s}' in m.columns, f'delta_full_{s} missing'
    assert f'R2_ridge_{s}' in m.columns, f'R2_ridge_{s} missing'
print('Block 4 OK')
"
```
