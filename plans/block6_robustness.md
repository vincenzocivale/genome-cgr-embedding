# Block 6 — Robustness statistica

**Esperimenti:** 31 (Esp 30 e 32 già complete)
**GPU:** No
**Priorità:** 🟢 Bassa (pura analisi)

## Obiettivo

Aggiungere la sensitivity analysis sulla soglia ε ∈ {0.005, 0.01, 0.02, 0.05} per robustezza dei claim win/tie/loss rispetto al baseline k-mer.

---

## Stato di Esp 30 e 32

- **Esp 30 — FDR (Benjamini-Hochberg):** ✅ `results/analysis/fdr_results.csv` contiene 14 righe con `raw_p`, `adjusted_p`, `significant` per ogni confronto FM vs k-mer.
- **Esp 32 — PCA compression:** ✅ `results/classification/records_pca.csv` 58 righe con `pca_kmer_k{K}_*` e `pca_fm_{model}_*`.

Nessun'azione richiesta — solo riferimento in paper methods.

---

## Esp 31 — Sensitivity su ε

### Modifica a `src/analysis/fdr_analysis.py`

Aggiungere flag `--sweep-eps` o creare nuovo script `src/analysis/sensitivity_eps.py`:

```python
import pandas as pd
import numpy as np

rf = pd.read_csv('results/classification/records_rf.csv', index_col=0)

EPS_GRID = [0.005, 0.01, 0.02, 0.05]
MODELS = ['NTv3_650M_pre', 'hyenadna-medium-160k-seqlen-hf', 'DNABERT-2-117M']
METRICS = ['MCC', 'AUROC']
BASELINE_K = 6

rows = []
for eps in EPS_GRID:
    for metric in METRICS:
        base_col = f'kmer_k{BASELINE_K}_{metric}'
        for m in MODELS:
            fm_col = f'fm_{m}_{metric}'
            delta = rf[fm_col] - rf[base_col]
            delta = delta.dropna()
            rows.append({
                'eps': eps,
                'metric': metric,
                'comparison': f'fm_{m}_vs_kmer_k{BASELINE_K}',
                'n_datasets': len(delta),
                'fm_wins':  int((delta >  eps).sum()),
                'kmer_wins': int((delta < -eps).sum()),
                'ties':      int(((delta >= -eps) & (delta <= eps)).sum()),
                'mean_delta':   float(delta.mean()),
                'median_delta': float(delta.median()),
            })

pd.DataFrame(rows).to_csv('results/analysis/sensitivity_analysis.csv', index=False)
```

### Run

```bash
python3 src/analysis/sensitivity_eps.py
```

**Output:** `results/analysis/sensitivity_analysis.csv` — 4 ε × 2 metriche × 3 FM = 24 righe (raddoppia a 32 se Caduceus è stato aggiunto in Block 1).

---

## Riferimento a Esp 3 (error bars, Block 1)

La sensitivity analysis per ε diventa più solida se disponiamo delle `_std` colonne prodotte in Block 1 Esp 3. Opzionale extension a `sensitivity_eps.py`:

```python
# Se le _std colonne esistono, calcola "significant wins":
# un win è robusto se |delta| > eps AND |delta| > 2*std_pooled
std_col = f'kmer_k{BASELINE_K}_{metric}_std'
fm_std_col = f'fm_{m}_{metric}_std'
if std_col in rf.columns and fm_std_col in rf.columns:
    std_pooled = np.sqrt(rf[std_col]**2 + rf[fm_std_col]**2)
    robust_wins = int(((delta > eps) & (delta.abs() > 2 * std_pooled)).sum())
    rows[-1]['robust_fm_wins'] = robust_wins
```

Questa è un'estensione "nice to have". Non bloccante se Esp 3 non è stato eseguito full.

---

## Verification

```bash
python3 -c "
import pandas as pd

df = pd.read_csv('results/analysis/sensitivity_analysis.csv')
print('Rows:', len(df))
print('Eps values:', sorted(df['eps'].unique()))
print('Comparisons:', df['comparison'].unique())
print()
print('Sample output:')
print(df[df['metric'] == 'MCC'].pivot_table(
    index='comparison', columns='eps', values='fm_wins'
))

# Sanity: 4 eps × 2 metrics × (3 or 4) comparisons = 24 or 32
assert len(df) in (24, 32), f'Unexpected rows: {len(df)}'
print('Block 6 OK')
"
```

---

## Note

- `records_rf.csv` è già 57/57 per le colonne MCC e AUROC dei 3 FM → lo script funziona subito.
- Dopo l'aggiunta di Caduceus (Block 1), rilanciare lo script per includere `fm_caduceus-ph_vs_kmer_k6` nel report.
