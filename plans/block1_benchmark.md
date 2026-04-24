# Block 1 — Benchmark comparativo

**Esperimenti:** 2, 3, 5, 6, 8
**GPU:** Sì
**Priorità:** 🔴 Alta

## Obiettivo

Chiudere i gap del benchmark classificazione/regressione: std across seeds (Esp 3), linear probe FM su LRA (Esp 5), pooling max/cls per HyenaDNA e DNABERT-2 (Esp 6), aggiunta di Caduceus-Ph come FM alternativo (Esp 8), e completamento dell'unica cella LRA RF mancante (Esp 2).

Gli Esp 1, 4, 7 sono già complete e non sono oggetto di questo piano.

---

## Esp 2 — Completa LRA RF

**Stato attuale:** `lra_records_rf.csv` ha 3 righe; `fm_DNABERT-2-117M_R2` ha 2/3 valori. Rilanciare per completare la cella mancante.

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('results/regression/lra_records_rf.csv', index_col=0)
print('Missing DNABERT-2 R2 per task:')
print(df['fm_DNABERT-2-117M_R2'].isna())
"
# Trovato il task mancante, rilancia:
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/regression/train_lra_benchmark.py \
    --hg38 /raid/DATASETS/genomics-long-range-benchmark/Homo_sapiens_assembly38.fasta
```

Lo script resuma dai valori già presenti; completa solo le celle NaN.

**Output:** 3 righe × tutti i `fm_*_R2`, `fm_*_Spearman`, `fm_*_MSE`.

---

## Esp 3 — Error bars (std across folds)

**Strategia:** 5-seed rerun (decisione utente). Non tocco il protocollo CV (`StratifiedKFold(4)` interno + test held-out); eseguo `train_rf` con 5 random state diversi e salvo mean + std come colonne separate.

### Modifiche codice

1. **`src/training/rf_pipeline.py`** — `train_rf()` deve accettare `random_state`:
   ```python
   def train_rf(X_train, y_train, n_classes, random_state=42):
       cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
       rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
       return GridSearchCV(rf, _RF_PARAM_GRID, scoring=..., cv=cv, n_jobs=1, refit=True).fit(X_train, y_train).best_estimator_
   ```

2. **Nuovo script** `src/scripts/classification/train_rf_with_seeds.py`:
   - Cicla sui 5 seed `[42, 0, 1, 2, 3]` chiamando `train_rf(..., random_state=seed)` → `eval_rf` su test set.
   - Aggrega mean e std di MCC, AUROC, F1, Accuracy.
   - Scrive colonne `{method}_{metric}_mean`, `{method}_{metric}_std` in `records_rf.csv`.

3. **`src/records/records.py`** — nuovi helper `write_kmer_rf_std(ds, k, metrics_dict)` e `write_fm_rf_std(ds, model, metrics_dict)` che popolano entrambe le colonne `_mean` e `_std`.

### Run

```bash
# k-mer (CPU)
python3 src/scripts/classification/train_rf_with_seeds.py --mode kmer --k-values 4 5 6

# FM (GPU, un modello alla volta)
for model in InstaDeepAI/NTv3_650M_pre \
             LongSafari/hyenadna-medium-160k-seqlen-hf \
             zhihan1996/DNABERT-2-117M; do
    CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_rf_with_seeds.py \
        --mode fm --model $model --n-workers 1
done
```

**Output:** nuove colonne `kmer_k{4,5,6}_{metric}_std`, `fm_{model}_{metric}_std` in `records_rf.csv`. I valori `*_mean` sono ridondanti con i già esistenti `kmer_k{4,5,6}_{metric}` (media sui seed ≈ singolo run con seed=42) — da usare solo come sanity check.

**Tempo stimato:** ~5x tempo di run originale del benchmark = qualche giorno su 1 GPU per tutti i 3 FM. Plausibile limitare a un subset di 15-20 dataset rappresentativi se il compute è scarso; discutere prima di avviare full.

---

## Esp 5 — Linear probe FM su LRA

**Stato:** `lra_records_linear_probe.csv` contiene solo 1 riga (`variant_effect_pathogenic_clinvar`) con solo le colonne k-mer. Mancano le FM columns per tutti e 3 i task.

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/regression/train_lra_linear_probe.py \
    --model InstaDeepAI/NTv3_650M_pre \
    --hg38 /raid/DATASETS/genomics-long-range-benchmark/Homo_sapiens_assembly38.fasta

CUDA_VISIBLE_DEVICES=0 python3 src/scripts/regression/train_lra_linear_probe.py \
    --model LongSafari/hyenadna-large-1m-seqlen-hf \
    --hg38 /raid/DATASETS/genomics-long-range-benchmark/Homo_sapiens_assembly38.fasta

CUDA_VISIBLE_DEVICES=0 python3 src/scripts/regression/train_lra_linear_probe.py \
    --model zhihan1996/DNABERT-2-117M \
    --hg38 /raid/DATASETS/genomics-long-range-benchmark/Homo_sapiens_assembly38.fasta
```

**Output:** `lra_records_linear_probe.csv` esteso a 3 righe (variant_effect, bulk_rna, cage) con `lp_fm_{model}_R2`, `lp_fm_{model}_Spearman` per 3 modelli.

---

## Esp 6 — Pooling max/cls per HyenaDNA e DNABERT-2

**Stato:**
- NTv3: max=57/57 ✓, cls=22/57 (da completare)
- HyenaDNA: solo mean (max/cls assenti; cls non supportato per char-level → ValueError corretto)
- DNABERT-2: solo mean (max/cls assenti)

### Linear probe

```bash
# NTv3 cls (completare i 35 dataset mancanti)
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model InstaDeepAI/NTv3_650M_pre --pooling cls --n-workers 1

# HyenaDNA max
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model LongSafari/hyenadna-medium-160k-seqlen-hf --pooling max --n-workers 1

# DNABERT-2 max
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model zhihan1996/DNABERT-2-117M --pooling max --n-workers 1

# DNABERT-2 cls
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model zhihan1996/DNABERT-2-117M --pooling cls --n-workers 1
```

### Decomposition ridge (pooling non-mean)

```bash
# HyenaDNA max
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model LongSafari/hyenadna-medium-160k-seqlen-hf \
    --pooling max --mapper ridge --k-values 4 5 6 --n-workers 1

# DNABERT-2 max
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model zhihan1996/DNABERT-2-117M \
    --pooling max --mapper ridge --k-values 4 5 6 --n-workers 1

# DNABERT-2 cls
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model zhihan1996/DNABERT-2-117M \
    --pooling cls --mapper ridge --k-values 4 5 6 --n-workers 1
```

**Output:**
- `records_linear_probe.csv`: `lp_fm_{model}_{metric}__pool_{max,cls}`
- `records_decomposition.csv`: `ridge_k{K}_{model}_R2__pool_{max,cls}__map_ridge`, `proj_*`, `resid_*`

---

## Esp 8 — Caduceus-Ph

### Modifiche codice

1. **Nuovo embedder** `src/embedders/caduceus_embedder.py`
   - `class CaduceusEmbedder(FMEmbedder)`: eredita l'API di `FMEmbedder` (cache, embed_sequences).
   - `_load_model()`: `AutoModel.from_pretrained("kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16", trust_remote_code=True)`.
   - Pooling: mean/max su `last_hidden_state`; `cls` → `NotImplementedError` (BiMamba non ha CLS token).
   - Tokenizer: char-level (nucleotidi singoli) — verificare via `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.

2. **Registro modelli** — aggiungere `caduceus-ph` in:
   - `src/scripts/classification/train_fm_rf.py:FM_METHODS`
   - `src/scripts/classification/train_linear_probe.py` (dispatcher modello → embedder)
   - `src/scripts/decomposition/train_decomposition.py` (stesso dispatcher)
   - `src/scripts/utils/benchmark_efficiency.py` (per Esp 26)

3. **Sanity check** (1 dataset piccolo, prima del run full):
   ```bash
   python3 -c "
   from src.embedders.caduceus_embedder import CaduceusEmbedder
   emb = CaduceusEmbedder('kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16', 'cache/fm_embeddings')
   X = emb.embed_sequences(['ATCG'*20], dataset='test', split='train', pooling='mean', batch_size=1)
   print('OK shape=', X.shape)
   "
   ```

### Run (dopo sanity check)

```bash
MODEL="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"

CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_fm_rf.py \
    --model $MODEL --n-workers 1 --fm-batch-size 16

CUDA_VISIBLE_DEVICES=0 python3 src/scripts/classification/train_linear_probe.py \
    --mode fm --model $MODEL --pooling mean --n-workers 1

CUDA_VISIBLE_DEVICES=0 python3 src/scripts/decomposition/train_decomposition.py \
    --model $MODEL --pooling mean --mapper ridge --k-values 4 5 6 --n-workers 1
```

**Output:** nuove colonne `fm_caduceus-ph_{metric}` in `records_rf.csv`, `lp_fm_caduceus-ph_{metric}` in `records_linear_probe.csv`, e `ridge_k{4,5,6}_caduceus-ph_R2`, `proj_*`, `resid_*` in `records_decomposition.csv`.

---

## Verification

```bash
python3 -c "
import pandas as pd

rf = pd.read_csv('results/classification/records_rf.csv', index_col=0)
lp = pd.read_csv('results/classification/records_linear_probe.csv', index_col=0)
lra = pd.read_csv('results/regression/lra_records_rf.csv', index_col=0)
lra_lp = pd.read_csv('results/regression/lra_records_linear_probe.csv', index_col=0)
d = pd.read_csv('results/decomposition/records_decomposition.csv', index_col=0)

# Esp 2
assert lra['fm_DNABERT-2-117M_R2'].notna().sum() == 3, 'LRA DNABERT-2 incomplete'

# Esp 3
assert 'kmer_k6_MCC_std' in rf.columns, 'Esp 3 std kmer missing'
assert 'fm_NTv3_650M_pre_MCC_std' in rf.columns, 'Esp 3 std FM missing'

# Esp 5
for m in ['NTv3_650M_pre','hyenadna-large-1m-seqlen-hf','DNABERT-2-117M']:
    assert f'lp_fm_{m}_R2' in lra_lp.columns, f'LRA LP FM {m} missing'
assert len(lra_lp) == 3, f'LRA LP rows={len(lra_lp)}, expected 3'

# Esp 6
for m, pools in [('NTv3_650M_pre', ['max','cls']),
                 ('hyenadna-medium-160k-seqlen-hf', ['max']),
                 ('DNABERT-2-117M', ['max','cls'])]:
    for p in pools:
        col = f'lp_fm_{m}_MCC__pool_{p}'
        assert lp[col].notna().sum() == 57, f'{col}: {lp[col].notna().sum()}/57'

# Esp 8
assert 'fm_caduceus-ph_MCC' in rf.columns, 'Caduceus missing in RF'
assert 'lp_fm_caduceus-ph_MCC' in lp.columns, 'Caduceus missing in LP'
print('Block 1 OK')
"
```
