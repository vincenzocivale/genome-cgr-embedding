# Block 5 — Costo computazionale

**Esperimenti:** 25, 26, 27, 28 (Esp 29 già completa)
**GPU:** Sì (per FM profiling; k-mer è CPU)
**Priorità:** 🟡 Media

## Obiettivo

Produrre una tabella unificata di efficienza con **metodo × seq_len × (GFLOPS analitici, GFLOPS empirici, wall-clock)**, comparabile in maniera simmetrica tra k-mer (CPU, analitico ed empirico) e FM (GPU, empirico con profiler, 100 repliche).

---

## Stato attuale

- `results/efficiency/efficiency.csv`: 20 righe, **solo 4 colonne** (`method, seq_len, params_M, GFLOPS_per_seq`). Schema vecchio.
- `results/efficiency/records_efficiency.csv`: 56 righe per-dataset con `duration_sec`, `GFLOPS_total` — non direttamente confrontabile (granularità diversa).
- `results/efficiency/efficiency_gpu_parallel.csv`: 3 righe — scarsamente popolato.

Gap rispetto agli Esp 25–28:
- Manca `duration_sec` empirico per k-mer (Esp 27) e per FM nel CSV per-seq_len (Esp 28).
- Manca `GFLOPS_theory_per_seq` analitico per k-mer (Esp 25).
- Lo script `benchmark_efficiency.py` esiste ma scrive un file con schema diverso — verifica se scrive 8 col o 4.

---

## Preparazione: audit dello script

```bash
python3 -c "
import subprocess
subprocess.run(['python3', 'src/scripts/utils/benchmark_efficiency.py', '--help'])
"
```

Verifica che esistano i flag:
- `--methods` (lista)
- `--seq-lens` (lista)
- `--n-seqs` (batch size per benchmark)
- `--n-reps` (ripetizioni, target: 100 per Esp 26)

Se `--n-reps` non esiste, aggiungerlo: il loop di misura deve essere `for rep in range(n_reps): start=time.perf_counter(); extract(...); elapsed.append(...)`, poi riportare `duration_sec = median(elapsed)`.

Se `kmer_k4` e `kmer_k5` non sono registrati come metodi, aggiungerli in `_METHODS_DISPATCH` (o equivalente) insieme a `kmer_k6`.

---

## Esecuzione

### Step 1 — Backup

```bash
cp results/efficiency/efficiency.csv results/efficiency/efficiency_old_schema.csv
rm results/efficiency/efficiency.csv
```

### Step 2 — FM timing (GPU, 100 reps per stabilità)

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/utils/benchmark_efficiency.py \
    --methods fm_NTv3_650M_pre,fm_hyenadna-medium-160k-seqlen-hf,fm_DNABERT-2-117M \
    --seq-lens 100 250 500 1000 2000 \
    --n-seqs 32 --n-reps 100
```

Se Caduceus è stato aggiunto (Block 1 Esp 8), includerlo:

```bash
CUDA_VISIBLE_DEVICES=0 python3 src/scripts/utils/benchmark_efficiency.py \
    --methods fm_caduceus-ph \
    --seq-lens 100 250 500 1000 2000 \
    --n-seqs 32 --n-reps 100
```

### Step 3 — k-mer timing (CPU, può girare in parallelo)

```bash
python3 src/scripts/utils/benchmark_efficiency.py \
    --methods kmer_k4,kmer_k5,kmer_k6 \
    --seq-lens 100 250 500 1000 2000 \
    --n-seqs 32 --n-reps 100 --n-workers 1
```

---

## Output finale

`results/efficiency/efficiency.csv` con schema a 8 colonne:

| col | Descrizione |
|---|---|
| `method` | es. `fm_NTv3_650M_pre`, `kmer_k6`, `fm_caduceus-ph` |
| `seq_len` | 100, 250, 500, 1000, 2000 |
| `n_seqs` | batch size (32) |
| `params_M` | parametri in M (NaN per k-mer) |
| `duration_sec` | mediana wall-clock su 100 reps |
| `GFLOPS_per_seq` | empirico (da `torch.profiler` per FM, `4^k * L` per k-mer) |
| `GFLOPS_total` | `GFLOPS_per_seq * n_seqs` |
| `GFLOPS_theory_per_seq` | analitico: per FM ricalcolo dalla forward-pass closed form; per k-mer = stesso di empirico |

**Righe attese:** (3 FM o 4 con Caduceus) × 5 seq_len + 3 k-mer × 5 seq_len = 35-40 righe.

Questo copre:
- **Esp 25** — `GFLOPS_theory_per_seq` per k-mer a 5 seq_len.
- **Esp 26** — `GFLOPS_per_seq` empirico per FM su A100, batch=32 (vicino a batch=1 — discutere in paper se fare un secondo run a batch=1), 100 reps.
- **Esp 27** — `duration_sec` k-mer.
- **Esp 28** — `duration_sec` FM.

---

## Esp 29 — Truncation (già completo)

`results/classification/truncation_analysis.csv` ha 342 righe (57 dataset × 3 modelli × 2 split). Nessuna action.

---

## Verification

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('results/efficiency/efficiency.csv')
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Methods:', df['method'].unique())
print('Seq lens:', sorted(df['seq_len'].unique()))

# Controllo schema 8 colonne
expected = {'method','seq_len','n_seqs','params_M','duration_sec',
            'GFLOPS_per_seq','GFLOPS_total','GFLOPS_theory_per_seq'}
missing = expected - set(df.columns)
assert not missing, f'Missing columns: {missing}'

# Controllo che k-mer abbia duration_sec (Esp 27)
kmer_rows = df[df['method'].str.startswith('kmer_')]
assert kmer_rows['duration_sec'].notna().all(), 'k-mer duration_sec missing'
print('k-mer rows:', len(kmer_rows), '(expected 15 = 3 k × 5 seq_len)')

# Controllo che FM abbia duration_sec (Esp 28)
fm_rows = df[df['method'].str.startswith('fm_')]
assert fm_rows['duration_sec'].notna().all(), 'FM duration_sec missing'
print('FM rows:', len(fm_rows), '(expected 15 or 20)')

print('Block 5 OK')
"
```

---

## Note operative

- 100 reps su FM grandi (NTv3 650M a seq_len=2000, batch=32) può richiedere ~1-2h solo per NTv3. Plausibile ridurre a 20 reps per seq_len ≥ 1000 se il tempo è critico; documenta la scelta nel paper.
- Usa lo stesso GPU (A100) per tutti i run FM per comparabilità; evita di mischiare A100/V100/H100 nella stessa tabella.
- Per `batch=1` (richiesto esplicitamente da Esp 26): aggiungi un secondo run con `--n-seqs 1 --n-reps 100` e salva in `efficiency_batch1.csv`. Il batch=32 è più rappresentativo del throughput reale, il batch=1 della latenza per query singola.
