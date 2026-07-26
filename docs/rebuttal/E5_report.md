# Dossier E5 per la risposta al reviewer e1HN

Stato: 4/5 modelli completi (NTv3, DNABERT-2, HyenaDNA, Caduceus), **inclusa
una revisione di correttezza sul CLS pooling** (vedi sezione 2). **Evo2 in
attesa** (lanciato su un'altra macchina, vedi sezione 6). Questo documento è
pensato per essere aggiornato in-place quando arrivano i risultati Evo2 —
le sezioni che cambieranno sono segnalate con 🔄.

Riferimenti nel repo (branch `rebuttal-experiments`):
- `results/rebuttal/E5_pooling/README.md` — descrizione del protocollo
- `results/rebuttal/E5_pooling/E5_summary.md` — tabelle complete già formattate
- `results/rebuttal/E5_pooling/pooling_results.csv` — dati grezzi (1140 righe)
- `results/rebuttal/RUN_STATUS.md` — stato di tutti gli esperimenti del rebuttal

---

## 1. Richiesta originale del reviewer (verbatim)

> E5. Pooling alternativi degli embedding FM
>
> Per ciascun modello, quando tecnicamente possibile, valutare: mean
> pooling; max pooling; concatenazione mean + max; CLS token o token
> equivalente; attention pooling.
>
> Domanda scientifica: il mean pooling sta cancellando informazioni
> posizionali utili?
>
> Output: tabella modello × pooling × performance; migliore strategia di
> pooling per task; confronto con k-mer usando il pooling migliore;
> analisi specifica dei task promotore, splice-site e regulatory.
>
> Criterio di successo: poter affermare che i risultati non sono un
> artefatto esclusivo del mean pooling.

---

## 2. Setup sperimentale (da citare nella risposta)

- **Modelli**: NTv3 (650M, transformer), DNABERT-2 (117M, transformer),
  HyenaDNA (medium, state-space), Caduceus (Mamba-based, bidirezionale) —
  quattro architetture distinte. Evo2 (1B, StripedHyena) 🔄 in corso.
- **Dataset**: tutti i 57 dataset del benchmark di classificazione usato nel
  paper (nessun sottoinsieme).
- **Pooling valutati**: mean, max, mean+max (concatenazione, feature
  raddoppiate), attention (parameter-free: query = media mascherata della
  sequenza, nessun parametro addestrabile né budget di training aggiuntivo —
  isola l'effetto di trattenere posizioni salienti senza introdurre
  supervisione extra), più due varianti "CLS o token equivalente" **decise
  empiricamente per ciascun tokenizer, non per assunzione architetturale**
  (una prima versione del codice assumeva erroneamente "char-level = niente
  CLS", sbagliata: verificato che tutti e 4 i modelli dichiarano un token
  CLS nel vocabolario, ma solo alcuni lo usano davvero — vedi test diretto
  su ciascun tokenizer):
  - **cls** (`hidden[:, 0, :]`): valido solo dove il tokenizer *antepone*
    davvero `[CLS]` in posizione 0. Unico caso verificato: **DNABERT-2**
    (BERT/BPE). NTv3 dichiara `<cls>` nel vocabolario ma non lo antepone
    mai (nemmeno con `add_special_tokens=True`) — i risultati "cls"
    raccolti inizialmente per NTv3 sono stati rietichettati **first_token**
    (embedding del primo nucleotide, non un token di sintesi) per non
    riportare una claim falsa.
  - **last_token** (ultima posizione valida, robusto al padding): valido
    dove il tokenizer *appende* un token speciale reale (SEP/EOS) in fondo.
    **HyenaDNA** e **Caduceus** appendono `[SEP]`; per il modello
    autoregressivo HyenaDNA l'ultimo token è il summary naturale (come in
    GPT). NTv3 non appende nulla, quindi non ha alcun token CLS-equivalente.
- **Probe**: regressione logistica lineare, stesso protocollo fissato usato
  nell'esperimento E4 (probe fairness): split train/test originale,
  StandardScaler fit solo su train, 4-fold CV stratificata, 4 configurazioni
  di C, seed 42.
- **Test statistico**: Wilcoxon signed-rank appaiato (mean pooling vs
  ciascuna alternativa, per dataset), correzione Benjamini-Hochberg,
  soglia di significatività 0.05, minimo 5 dataset appaiati — stessa
  convenzione usata per il confronto k-mer/FM nel paper principale
  (`src/analysis/fdr_analysis.py`).

---

## 3. Output 1 — Tabella modello × pooling × performance

MCC medio di test su 57 dataset, probe lineare:

```
pooling                        mean    max  mean_max    cls  first_token  last_token  attention
DNABERT-2-117M                0.428  0.381     0.423  0.463          n/a         n/a      0.428
NTv3_650M_pre                 0.602  0.565     0.601    n/a        0.412         n/a      0.441
Caduceus-ph (256d, 16 layer)   0.559  0.465     0.522    n/a          n/a       0.403      0.517
HyenaDNA-medium                0.546  0.411     0.490    n/a          n/a       0.331      0.491
Evo2-1B 🔄                       —      —         —      n/a          n/a         —          —
```

(Tabella completa e sorgente machine-readable: `model_pooling_performance.csv`.)

---

## 4. Output 2 — Migliore strategia di pooling per task

Aggregato per modello: **mean** vince per NTv3, HyenaDNA, Caduceus;
**cls vince per DNABERT-2, e stavolta è un vantaggio reale e significativo**
(vedi sezione 7), non un pareggio.

Distribuzione del pooling vincente per singolo dataset (57 dataset ×
4 modelli = 228 celle, da `best_pooling_by_task.csv`, aggiornata dopo
l'aggiunta di cls/first_token/last_token):

```
mean          105 / 228  (46%)
mean_max       56 / 228  (25%)
cls            41 / 228  (18%)  <- tutti su DNABERT-2
attention      22 / 228  (10%)
max             3 / 228  (1%)
first_token     1 / 228  (<1%)
```

`mean` e `mean_max` insieme coprono il 71% dei casi migliori per singolo
task; il restante 18% (`cls`) è concentrato interamente su DNABERT-2, dove
il CLS è un token realmente addestrato — non è rumore distribuito a caso
tra i modelli.

---

## 5. Output 3 — Confronto con k-mer usando il pooling migliore

`analyze_rebuttal_experiments.py` (funzione `e5()`) accetta un solo file E4
alla volta e legge da `probe_matrix_ntv3_k6.csv` (12 righe, non consolidato),
quindi `best_pooling_vs_kmer.csv` resta parziale/inutilizzabile per NTv3.
**Per DNABERT-2, però, ho calcolato il confronto a mano** dai 4 shard E4
ancora non consolidati (`probe_matrix_dnabert2_k6_q{0..3}.csv`, letti in
sola lettura, rappresentazione `kmer`, probe lineare), disponibili su
54/57 dataset:

```
DNABERT-2 CLS (miglior pooling) vs k-mer:  cls vince 19/54   k-mer vince 35/54
  delta medio (cls - kmer): -0.047   Wilcoxon p=0.0055 (significativo, a favore di k-mer)

DNABERT-2 mean (riferimento) vs k-mer:     mean vince 12/54  k-mer vince 42/54
  delta medio (mean - kmer): -0.082  Wilcoxon p<0.000001
```

**k-mer resta significativamente superiore anche al pooling migliore
(CLS) di DNABERT-2** — CLS dimezza il divario rispetto a mean ma non lo
chiude. Non c'è quindi alcun rischio che il vantaggio di CLS su mean venga
letto come "il pooling giusto fa vincere l'FM sul k-mer": non è così, va
detto esplicitamente nella risposta per non generare un'impressione
fuorviante.

Per NTv3/HyenaDNA/Caduceus questo confronto va rifatto quando E4 sarà
consolidato (`wc -l results/rebuttal/E4_probe_fairness/probe_matrix_*_k6.csv`
per ciascun modello — devono avvicinarsi a 57×4×3=684 righe l'uno), poi:
```bash
conda run -n cgr_bench python src/rebuttal/analyze_rebuttal_experiments.py \
  --e5 results/rebuttal/E5_pooling/pooling_results.csv \
  --e4 results/rebuttal/E4_probe_fairness/probe_matrix_ntv3_k6.csv \
  --out-root results/rebuttal
```

---

## 6. Output 4 — Analisi task promotore / splice-site / regulatory

Stesso pattern del risultato generale conferma anche sui sottoinsiemi
prioritari dal reviewer (n_datasets: promoter=15, splice-site=4,
regulatory=17 — il resto è "other"):

```
                                model task_group   pooling  n_datasets  mean_delta  adjusted_p  significant
                       NTv3_650M_pre   promoter attention          15     -0.1332      0.0001         True
                       NTv3_650M_pre   promoter       cls          15     -0.1497      0.0021         True
                       NTv3_650M_pre   promoter       max          15     -0.0298      0.0001         True
                       NTv3_650M_pre   promoter  mean_max          15     -0.0058      0.4119        False
                       NTv3_650M_pre regulatory attention          17     -0.1801      0.0001         True
                       NTv3_650M_pre regulatory       cls          17     -0.1898      0.0001         True
                       NTv3_650M_pre regulatory       max          17     -0.0384      0.0023         True
                       NTv3_650M_pre regulatory  mean_max          17      0.0011      0.6970        False
```
(tabella completa per tutti e 4 i modelli in
`pooling_significance_by_task_group.csv` — nota: splice-site ha solo 4
dataset, sotto la soglia minima di 5 per il test Wilcoxon in alcuni
sottogruppi modello × pooling, quindi alcune celle mancano lì per quel
task_group specifico, non è un errore).

---

## 7. Criterio di successo — bozza di conclusione da adattare

> Su quattro modelli architetturalmente distinti (transformer, state-space,
> Mamba) e fino a sei strategie di pooling per modello, il mean pooling non
> è mai battuto in modo significativo, con **una sola eccezione**: il CLS
> genuino di DNABERT-2 (MCC 0.463 vs 0.428, 41/57 vittorie, p=2.2e-5). Ogni
> altro proxy non addestrato — max, mean+max, attention parameter-free,
> first-token, last-token/SEP — è pareggiato o significativamente peggiore
> di mean, in modo consistente sui quattro modelli e sui sottoinsiemi
> promotore/splice-site/regulatory indicati dal reviewer. L'eccezione è
> essa stessa informativa: non indica che l'informazione posizionale sia
> generalmente recuperabile da pooling alternativi generici, ma che un
> token specificamente addestrato ad aggregarla (come il CLS negli
> obiettivi BERT-style) può farlo, mentre proxy non supervisionati non ci
> riescono — e anche questa eccezione non ribalta il confronto con il
> k-mer (sezione 5): k-mer resta significativamente superiore anche al CLS
> di DNABERT-2. [🔄 Evo2 conferma/non conferma il pattern — aggiungere
> qui]. Questo risponde direttamente alla domanda scientifica posta: il
> mean pooling non sta scartando informazione posizionale recuperabile da
> strategie di pooling generiche.

**Nota onestà scientifica**: se Evo2 mostrasse un pattern diverso (es. un
pooling batte mean con margine ampio e significativo), la conclusione
sopra andrebbe riformulata di conseguenza e il caso discusso esplicitamente,
non nascosto — è stato esplicitamente richiesto all'agente che esegue Evo2
di segnalarlo se succede (vedi criterio di successo nel prompt di handoff).

---

## 8. Limitazioni da dichiarare esplicitamente nella risposta

1. **Evo2 mancante al momento della scrittura** (se la rebuttal deadline
   arriva prima del completamento) — dichiarare che è in corso su
   un'infrastruttura diversa per un vincolo tecnico (checkpoint FP8 di
   Evo2-1B richiede `transformer_engine`, che richiede un toolchain CUDA
   completo — nvcc + NCCL — non disponibile sulla macchina primaria), non
   per scelta metodologica.
2. **Confronto con k-mer (output 3) completo solo per DNABERT-2** (54/57
   dataset); per NTv3/HyenaDNA/Caduceus resta parziale finché E4 non è
   consolidato per quei modelli.
3. **Probe singolo** (solo lineare) per E5, non l'intera matrice RF/lineare/MLP
   usata in E4 — coerente con i dati già esistenti prima di questo run,
   ma va detto se il reviewer chiede robustezza rispetto al classificatore.
4. Il pooling `attention` è deliberatamente **senza parametri addestrabili**
   (query = media mascherata) — utile per isolare l'effetto architetturale,
   ma è una scelta di design da rendere esplicita: un vero attention pooling
   con parametri appresi potrebbe comportarsi diversamente (non testato).
5. **CLS/first_token/last_token non sono direttamente confrontabili tra
   modelli** — sono etichette diverse per meccanismi diversi (vero token
   addestrato per DNABERT-2, posizione grezza per gli altri), non varianti
   dello stesso pooling. Il confronto va sempre fatto per-modello, mai
   aggregato indistintamente come "CLS pooling" nel testo della risposta.

---

## 9. Come aggiornare questo dossier quando arriva Evo2

1. `git pull origin rebuttal-experiments` (l'altra macchina pusha i suoi
   risultati sullo stesso branch).
2. Rigenerare `E5_summary.md` con lo script di generazione (chiedimi di
   rilanciarlo).
3. Sostituire ogni 🔄 in questo documento con i numeri reali. Nota: il
   prompt di handoff per Evo2 richiede solo mean/max/mean_max/attention
   (CLS assunto non applicabile per architettura, non ancora verificato
   empiricamente sul tokenizer di Evo2 come fatto qui per gli altri 4 —
   se l'agente su Evo2 ha modo di controllare `tokenizer.tokenize()` /
   equivalente per un eventuale token di summary anteposto o appeso, vale
   la pena rifare la stessa verifica prima di escluderlo per assunzione).
4. Ricontrollare la sezione 5 (k-mer) — se anche E4 è avanzato nel
   frattempo, questa è l'occasione per completare NTv3/HyenaDNA/Caduceus
   insieme a Evo2 (per ora solo DNABERT-2 ha il confronto k-mer completo).
