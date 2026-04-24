# Block 3 — Decomposizione

**Esperimenti:** 14, 15, 16, 17, 18, 19
**GPU:** Sì
**Priorità:** 🔴 Alta (è la sezione più vicina alla "novità" del paper)

## Obiettivo

Completare la decomposizione ortogonale k-mer → embedding FM per tutti e 3 i FM × 3 k × 57 dataset (Esp 14–18, mapper Ridge), e produrre l'upper bound non-lineare via MLP (Esp 19). Esp 20 (λ via CV) e Esp 21 (concat FM + k-mer) sono già complete.

---

## Stato attuale

Da `records_decomposition.csv`:

| Model | k=4 ridge | k=5 ridge | k=6 ridge | MLP k=4 | MLP k=5 | MLP k=6 |
|---|---|---|---|---|---|---|
| NTv3-650M | ✅ 57/57 | ✅ 57/57 | ✅ 57/57 | 🟡 1/57 | ⛔ 0/57 | ⛔ 0/57 |
| HyenaDNA-160k | ⛔ 1/57 | ⛔ 0/57 | ✅ 57/57 | ⛔ 0/57 | ⛔ 0/57 | ⛔ 0/57 |
| DNABERT-2-117M | ⛔ 0/57 | ⛔ 0/57 | ✅ 57/57 | ⛔ 0/57 | ⛔ 0/57 | ⛔ 0/57 |

Gap: 2 modelli × 2 k (4 e 5) × 57 dataset per ridge; tutti e 3 modelli × 3 k × 57 dataset per MLP.

---

## Esp 14–18 — Ridge decomposition (mancanti)

Script esistente `src/scripts/decomposition/train_decomposition.py` supporta già `--mapper ridge`, `--k-values`, `--pooling mean` e resume per riga. Basta riavviare sui modelli mancanti:

```bash
# HyenaDNA k=4 e k=5
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model LongSafari/hyenadna-medium-160k-seqlen-hf \
    --pooling mean --mapper ridge --k-values 4 5 \
    --n-workers 1 --fm-batch-size 8

# DNABERT-2 k=4 e k=5
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model zhihan1996/DNABERT-2-117M \
    --pooling mean --mapper ridge --k-values 4 5 \
    --n-workers 1 --fm-batch-size 16
```

Lo script salta righe già popolate (check `has_decomp_ridge`, `has_proj`, `has_resid`).

**Output:** in `records_decomposition.csv`:
- `ridge_k{4,5}_hyenadna-medium-160k-seqlen-hf_R2` (57/57)
- `ridge_k{4,5}_DNABERT-2-117M_R2` (57/57)
- `proj_k{4,5}_{m}_MCC`, `proj_k{4,5}_{m}_AUROC` per m∈{hyenadna, DNABERT-2}
- `resid_k{4,5}_{m}_MCC`, `resid_k{4,5}_{m}_AUROC`

Questo chiude Esp 14, 15 (R² distribution per modello derivabile), 16 (projection), 17 (residual), 18 (comparison full vs proj vs resid — pura post-processing).

---

## Esp 19 — MLP decomposition

Script esistente `train_decomposition.py` supporta `--mapper mlp`. Pipeline MLP in `src/training/mlp_mapping.py`.

```bash
for model in InstaDeepAI/NTv3_650M_pre \
             LongSafari/hyenadna-medium-160k-seqlen-hf \
             zhihan1996/DNABERT-2-117M; do
    bs=32
    [ "$model" = "LongSafari/hyenadna-medium-160k-seqlen-hf" ] && bs=8
    [ "$model" = "zhihan1996/DNABERT-2-117M" ] && bs=16

    CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
        --model $model --pooling mean --mapper mlp \
        --mlp-epochs 20 --mlp-hidden-dim 512 \
        --k-values 4 5 6 --n-workers 1 --fm-batch-size $bs
done
```

**Output:**
- `ridge_k{4,5,6}_{model}_R2__pool_mean__map_mlp`
- `proj_k{4,5,6}_{model}_{MCC,AUROC}__pool_mean__map_mlp`
- `resid_k{4,5,6}_{model}_{MCC,AUROC}__pool_mean__map_mlp`

### Post-processing

Aggiungi a `scripts/paper_stats_audit.py` (o nuovo script `src/analysis/mlp_vs_ridge_r2.py`):

```python
import pandas as pd
df = pd.read_csv('results/decomposition/records_decomposition.csv', index_col=0)
summary = []
for model in ['NTv3_650M_pre', 'hyenadna-medium-160k-seqlen-hf', 'DNABERT-2-117M']:
    for k in [4, 5, 6]:
        ridge_col = f'ridge_k{k}_{model}_R2'
        mlp_col = f'ridge_k{k}_{model}_R2__pool_mean__map_mlp'
        if ridge_col in df.columns and mlp_col in df.columns:
            delta = df[mlp_col] - df[ridge_col]
            summary.append({
                'model': model, 'k': k,
                'median_ridge_R2': df[ridge_col].median(),
                'median_mlp_R2': df[mlp_col].median(),
                'median_delta': delta.median(),
                'IQR_delta': delta.quantile(0.75) - delta.quantile(0.25),
                'pct_mlp_better_5pp': (delta > 0.05).mean() * 100,
            })
pd.DataFrame(summary).to_csv('results/decomposition/mlp_vs_ridge_summary.csv', index=False)
```

**Tempo stimato:** MLP 20 epoch × 57 dataset × 3 k × 3 modelli ≈ 8-15 h su 1 GPU.

---

## Verification

```bash
python3 -c "
import pandas as pd
d = pd.read_csv('results/decomposition/records_decomposition.csv', index_col=0)

models = ['NTv3_650M_pre', 'hyenadna-medium-160k-seqlen-hf', 'DNABERT-2-117M']

print('=== Ridge decomposition (Esp 14-18) ===')
for m in models:
    for k in [4, 5, 6]:
        col = f'ridge_k{k}_{m}_R2'
        n = d[col].notna().sum() if col in d.columns else 0
        flag = 'OK' if n == 57 else 'INCOMPLETE'
        print(f'  {col}: {n}/57 [{flag}]')

print()
print('=== MLP decomposition (Esp 19) ===')
for m in models:
    for k in [4, 5, 6]:
        col = f'ridge_k{k}_{m}_R2__pool_mean__map_mlp'
        n = d[col].notna().sum() if col in d.columns else 0
        flag = 'OK' if n == 57 else 'INCOMPLETE'
        print(f'  {col}: {n}/57 [{flag}]')

# MLP vs Ridge summary
import os
if os.path.exists('results/decomposition/mlp_vs_ridge_summary.csv'):
    print('\\nMLP vs Ridge summary:')
    print(pd.read_csv('results/decomposition/mlp_vs_ridge_summary.csv'))
"
```

Atteso: tutte le righe `ridge_k{K}_{m}_R2` e `ridge_k{K}_{m}_R2__pool_mean__map_mlp` = 57/57 per 3 modelli × 3 k = 18 celle ciascuno.

---

## Note operative

- NON lanciare simultaneamente run che scrivono su `records_decomposition.csv` per lo stesso (model, k, pooling, mapper) — rischio di race condition sul CSV. Diversi (model, k) sono OK.
- La cache embedding in `cache/` è condivisa per pooling: `--pooling mean` riusa le estrazioni già fatte. Non cancellare la cache tra ridge e MLP.
- Caduceus (Block 1 Esp 8) aggiunge un 4° FM: dopo Block 1, aggiungi anche la decomposizione Caduceus con comando analogo.
