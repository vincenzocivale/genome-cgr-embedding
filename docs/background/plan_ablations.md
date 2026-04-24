# Piano di Implementazione — Esperimenti di Ablation

## Contesto

Il progetto dimostra che i vettori di frequenza k-mer (da FCGR) sono superiori agli embedding dei Foundation Model genomici per determinati task. Per rafforzare il paper contro obiezioni metodologiche di reviewer, servono 4 esperimenti di ablation:

1. **Linear probe** — verificare che i risultati non dipendano dalla non-linearita del Random Forest
2. **FDR correction** — correzione per test multipli sui 57 confronti dataset-level
3. **Canonical k-mer** — k-mer reverse-complement aware (standard bioinformatico)
4. **AUROC consistency** — verificare che i verdetti MCC siano consistenti con AUROC

**Esclusi:** Evo2, esperimenti multiscale k-mer.

---

## File critici esistenti (da riusare/estendere)

| File | Ruolo |
|------|-------|
| `src/training/rf_pipeline.py` | `train_rf`, `eval_rf`, `train_rf_regressor`, `eval_rf_regressor` |
| `src/records/records.py` | Persistenza CSV, colonne dinamiche, helper `has_*/write_*` |
| `src/features/kmer_features.py` | `extract_kmer_features`, `kmer_from_grids`, `_normalize_l1` |
| `src/core/fcgr.py` | `compute_fcgr`, `batch_fcgr`, `MAPPING = {'A':(0,0), 'C':(1,0), 'G':(1,1), 'T':(0,1)}` |
| `src/data/loader.py` | `discover_datasets`, `load_dataset` |
| `src/embedders/fm_embedder.py` | `FMEmbedder` (NTv3, DNABERT-2) |
| `src/embedders/hyena_embedder.py` | `HyenaEmbedder` |
| `src/scripts/train_kmer_rf.py` | Pattern di riferimento per script k-mer |
| `src/scripts/train_fm_rf.py` | Pattern di riferimento per script FM |
| `src/scripts/train_lra_benchmark.py` | Pattern LRA regression (records locali, `has_result`/`write_result`) |

---

## Nuovi file da creare

| File | Scopo |
|------|-------|
| `src/training/linear_pipeline.py` | Train/eval con LogisticRegressionCV e RidgeCV |
| `src/scripts/train_linear_probe.py` | Linear probe su 57 dataset classificazione (k-mer + FM) |
| `src/scripts/train_lra_linear_probe.py` | Linear probe su 2 dataset regressione LRA |
| `src/scripts/train_canonical_kmer.py` | Canonical k-mer con RF e linear probe |
| `src/analysis/fdr_analysis.py` | Correzione Benjamini-Hochberg (post-hoc) |
| `src/analysis/auroc_consistency.py` | Concordanza verdetti MCC vs AUROC (post-hoc) |

## File esistenti da estendere

| File | Modifiche |
|------|-----------|
| `src/records/records.py` | Aggiungere colonne/helper per linear probe e canonical k-mer |
| `src/features/kmer_features.py` | Aggiungere `canonicalize_kmer_features()` |

---

## Step 1 — `src/training/linear_pipeline.py` (nuovo)

Pipeline di training/eval con modelli lineari regolarizzati L2, analogo a `rf_pipeline.py`.

```python
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_linear_classifier(X_train, y_train, n_classes):
    """
    Pipeline: StandardScaler -> LogisticRegressionCV(L2).
    - Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    - solver = "lbfgs", max_iter = 5000
    - cv = StratifiedKFold(4, shuffle=True, random_state=42)
    - scoring = "roc_auc" (binary) o "accuracy" (multiclass)
    - multi_class = "multinomial"
    """

def eval_linear_classifier(model, X_test, y_test, n_classes):
    """Stesse metriche di eval_rf: {MCC, AUROC, F1, Accuracy}."""

def train_linear_regressor(X_train, y_train):
    """
    Pipeline: StandardScaler -> RidgeCV.
    - alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    """

def eval_linear_regressor(model, X_test, y_test):
    """Stesse metriche di eval_rf_regressor: {R2, MSE, Spearman}."""
```

**Nota critica:** StandardScaler e necessario (i modelli lineari sono sensibili alla scala, a differenza di RF). Va inserito come primo step di una `sklearn.Pipeline`.

---

## Step 2 — Estendere `src/records/records.py`

### Nuove costanti e CSV path
```python
RECORDS_LINEAR_CSV = "results/classification/records_linear_probe.csv"
RECORDS_CANONICAL_CSV = "results/classification/records_canonical_kmer.csv"
```

### Colonne linear probe
```python
def lp_kmer_col(k, metric) -> str:       # "lp_kmer_k{K}_{metric}"
def lp_fm_col(model_name, metric) -> str: # "lp_fm_{model_tag}_{metric}"
def load_lp_records(path) -> DataFrame
def has_lp_kmer(dataset, k, df) -> bool
def has_lp_fm(dataset, model_name, df) -> bool
def write_lp_kmer(dataset, k, rf_metrics, path) -> DataFrame
def write_lp_fm(dataset, model_name, rf_metrics, path) -> DataFrame
```

### Colonne canonical k-mer
```python
def canon_kmer_col(k, metric) -> str:       # "ckmer_k{K}_{metric}"
def canon_lp_kmer_col(k, metric) -> str:    # "ckmer_lp_k{K}_{metric}"
def load_canon_records(path) -> DataFrame
def has_canon_kmer(dataset, k, df) -> bool
def has_canon_lp_kmer(dataset, k, df) -> bool
def write_canon_kmer(dataset, k, metrics, path) -> DataFrame
def write_canon_lp_kmer(dataset, k, metrics, path) -> DataFrame
```

---

## Step 3 — Canonical k-mer in `src/features/kmer_features.py`

### Algoritmo

L'FCGR usa `MAPPING = {'A':(0,0), 'C':(1,0), 'G':(1,1), 'T':(0,1)}`. Il k-mer viene mappato a una cella `(i, j)` del grid `2^k x 2^k`, dove i/j sono le coordinate CGR discretizzate. L'indice flattened e `i * 2^k + j` (row-major).

Per costruire la mappa canonica:
1. Per ogni indice `idx` in `0..4^k-1`, determinare il k-mer corrispondente simulando il CGR walk inverso
2. Calcolare il reverse complement del k-mer
3. Determinare l'indice del reverse complement
4. L'indice canonico = `min(idx, rc_idx)`
5. Costruire matrice di aggregazione: per ogni canonical group, sommare le colonne

```python
def _fcgr_index_to_kmer(idx: int, k: int) -> str:
    """Converte indice FCGR flattened -> stringa k-mer."""
    # Decodifica (i, j) dalla row-major: i = idx // 2^k, j = idx % 2^k
    # Ricostruisce il k-mer dalla posizione nella griglia CGR

def _kmer_to_fcgr_index(kmer: str) -> int:
    """Converte stringa k-mer -> indice FCGR flattened."""

def _reverse_complement(seq: str) -> str:
    """ATCG <-> TAGC, reversed."""
    comp = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}

def _build_canonical_map(k: int) -> tuple[np.ndarray, int]:
    """
    Returns:
        canonical_map: (4^k,) array mapping each k-mer index to a canonical group index (0..n_canonical-1)
        n_canonical: number of canonical groups
    """

def canonicalize_kmer_features(kmer_freqs: np.ndarray, k: int) -> np.ndarray:
    """
    Input:  (N, 4^k)  standard k-mer frequency vectors (da kmer_from_grids)
    Output: (N, n_canonical) canonical k-mer frequency vectors, L1-normalizzati
    
    n_canonical per k: k=4 -> 136, k=5 -> 512, k=6 -> 2080
    """
```

**Nota:** La parte critica e la corrispondenza indice FCGR <-> k-mer. Serve un test unitario che verifichi per k=2 (16 k-mers) che l'assegnazione sia corretta contando direttamente i k-mers su sequenze note.

---

## Step 4 — `src/scripts/train_linear_probe.py` (nuovo)

Script unificato per linear probe su 57 dataset di classificazione.

```
Usage:
    python3 src/scripts/train_linear_probe.py --mode kmer --k-values 4 5 6 --n-workers 8
    python3 src/scripts/train_linear_probe.py --mode fm --model InstaDeepAI/NTv3_650M_pre
    python3 src/scripts/train_linear_probe.py --mode fm --model LongSafari/hyenadna-medium-160k-seqlen-hf
    python3 src/scripts/train_linear_probe.py --mode fm --model zhihan1996/DNABERT-2-117M
```

Struttura (identica al pattern `train_kmer_rf.py`/`train_fm_rf.py`):
1. `argparse` con `--mode {kmer, fm}`, `--k-values`, `--model`, `--data-root`, `--n-workers`
2. `discover_datasets()` loop
3. Skip check via `has_lp_kmer()` / `has_lp_fm()`
4. Feature extraction: riusa `kmer_from_grids()` / `FMEmbedder.embed_sequences()`
5. `train_linear_classifier()` + `eval_linear_classifier()`
6. `write_lp_kmer()` / `write_lp_fm()` -> `records_linear_probe.csv`

---

## Step 5 — `src/scripts/train_lra_linear_probe.py` (nuovo)

Analogo a `train_lra_benchmark.py` ma usa modelli lineari.

```
Usage:
    python3 src/scripts/train_lra_linear_probe.py --k-values 4 5 6
    python3 src/scripts/train_lra_linear_probe.py --k-values 4 5 6 --model InstaDeepAI/NTv3_650M_pre
```

- Riusa i task loader da `train_lra_benchmark.py` (importandoli o duplicando il minimo)
- Records in `results/regression/lra_records_linear_probe.csv`
- Per classificazione: `train_linear_classifier` / `eval_linear_classifier`
- Per regressione: `train_linear_regressor` / `eval_linear_regressor`

---

## Step 6 — `src/scripts/train_canonical_kmer.py` (nuovo)

Script per canonical k-mer con supporto sia RF che linear probe.

```
Usage:
    python3 src/scripts/train_canonical_kmer.py --k-values 4 5 6 --classifier rf
    python3 src/scripts/train_canonical_kmer.py --k-values 4 5 6 --classifier linear
```

Struttura:
1. `discover_datasets()` loop
2. Per ogni dataset: FCGR -> `kmer_from_grids()` -> `canonicalize_kmer_features()`
3. In base a `--classifier`:
   - `rf`: `train_rf` + `eval_rf` -> `write_canon_kmer()`
   - `linear`: `train_linear_classifier` + `eval_linear_classifier` -> `write_canon_lp_kmer()`
4. Tutto in `records_canonical_kmer.csv`

---

## Step 7 — `src/analysis/fdr_analysis.py` (nuovo, post-hoc)

Nessun training, solo analisi statistica su risultati esistenti.

```
Usage:
    python3 src/analysis/fdr_analysis.py
    python3 src/analysis/fdr_analysis.py --records results/classification/records_linear_probe.csv
```

Algoritmo:
1. Carica records CSV
2. Per ogni dataset, calcola `best_k_MCC = max(kmer_k4_MCC, ..., kmer_k6_MCC)`
3. Per ogni FM (NTv3, HyenaDNA, DNABERT-2):
   - Wilcoxon signed-rank test su coppie `(best_k_MCC, fm_MCC)` per 57 dataset
   - Raccogli p-value raw
4. Applica Benjamini-Hochberg via `scipy.stats.false_discovery_control()` o `statsmodels.stats.multitest.multipletests(method='fdr_bh')`
5. Salva in `results/analysis/fdr_results.csv` con colonne: `comparison, n_datasets, mean_delta, raw_p, adjusted_p, significant`
6. Ripeti con metrica AUROC

**Dipendenza:** `scipy` (gia presente). `statsmodels` potrebbe servire — verificare se installato, altrimenti usare `scipy.stats.false_discovery_control` (disponibile da scipy 1.11+, ma la versione nel progetto e 1.17).

---

## Step 8 — `src/analysis/auroc_consistency.py` (nuovo, post-hoc)

Analisi di concordanza tra verdetti MCC e AUROC. Nessun training.

```
Usage:
    python3 src/analysis/auroc_consistency.py
```

Algoritmo:
1. Carica `records.csv` (e opzionalmente `records_linear_probe.csv`)
2. Per ogni (dataset, FM_model):
   - `best_k_MCC` = max MCC tra k=4,5,6
   - `best_k_AUROC` = AUROC del k corrispondente al best MCC (o max AUROC, configurabile)
   - Verdetto MCC: `kmer_wins` se best_k_MCC > fm_MCC, `fm_wins` altrimenti
   - Verdetto AUROC: analogo
3. Calcola concordance rate
4. Elenca dataset discordanti
5. Output: `results/analysis/auroc_consistency.csv` e summary a stdout

---

## Ordine di esecuzione

```
Phase 1 (indipendenti, parallelizzabili):
  Step 1: src/training/linear_pipeline.py
  Step 2: src/records/records.py (estensioni)
  Step 3: src/features/kmer_features.py (canonical)

Phase 2 (dipendono da Phase 1):
  Step 4: src/scripts/train_linear_probe.py
  Step 5: src/scripts/train_lra_linear_probe.py
  Step 6: src/scripts/train_canonical_kmer.py

Phase 3 (post-hoc, eseguibili subito ma producono output utile dopo Phase 2):
  Step 7: src/analysis/fdr_analysis.py
  Step 8: src/analysis/auroc_consistency.py
```

---

## Nuovi file di risultati

| File | Contenuto |
|------|-----------|
| `results/classification/records_linear_probe.csv` | LP kmer + FM su 57 dataset |
| `results/regression/lra_records_linear_probe.csv` | LP kmer + FM su 2 dataset regressione |
| `results/classification/records_canonical_kmer.csv` | Canonical kmer (RF + LP) su 57 dataset |
| `results/analysis/fdr_results.csv` | P-value raw + BH-adjusted |
| `results/analysis/auroc_consistency.csv` | Concordanza MCC vs AUROC |

---

## Verifica end-to-end

1. **Unit test canonical k-mer:** per k=2, verificare che `_fcgr_index_to_kmer` e `_kmer_to_fcgr_index` siano inversi, e che la mappa canonica mergi correttamente le coppie RC (es. AA/TT, AC/GT)
2. **Dry run linear probe:** eseguire `train_linear_probe.py --mode kmer --k-values 4` su un sottoinsieme di dataset e verificare che `records_linear_probe.csv` venga creato con le colonne corrette
3. **Dry run canonical:** eseguire `train_canonical_kmer.py --k-values 4 --classifier rf` e verificare dimensioni feature ridotte (136 vs 256)
4. **FDR analysis:** dopo aver i risultati RF completi, eseguire `fdr_analysis.py` e verificare che i p-value adjusted siano >= raw
5. **AUROC consistency:** eseguire `auroc_consistency.py` su `records.csv` esistente e verificare output
