# Dossier E5 per la risposta al reviewer e1HN

Stato: 4/5 modelli completi (NTv3, DNABERT-2, HyenaDNA, Caduceus). **Evo2 in
attesa** (lanciato su un'altra macchina, vedi sezione 6). Questo documento è
pensato per essere aggiornato in-place quando arrivano i risultati Evo2 —
le sezioni che cambieranno sono segnalate con 🔄.

Riferimenti nel repo (branch `rebuttal-experiments`, ultimo commit `871190d`):
- `results/rebuttal/E5_pooling/README.md` — descrizione del protocollo
- `results/rebuttal/E5_pooling/E5_summary.md` — tabelle complete già formattate
- `results/rebuttal/E5_pooling/pooling_results.csv` — dati grezzi (970 righe)
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
- **Pooling valutati**: mean, max, mean+max (concatenazione, quindi feature
  raddoppiate), CLS (solo dove esiste un token speciale reale — NTv3;
  non applicabile a modelli char-level/autoregressivi: DNABERT-2, HyenaDNA,
  Caduceus, Evo2), attention (parameter-free: query = media mascherata
  della sequenza, nessun parametro addestrabile né budget di training
  aggiuntivo — isola l'effetto di trattenere posizioni salienti senza
  introdurre supervisione extra).
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
pooling                                          mean    max  mean_max    cls  attention
model
DNABERT-2-117M                                  0.428  0.381     0.423    n/a      0.428
NTv3_650M_pre                                   0.602  0.565     0.601  0.412      0.441
Caduceus-ph (256d, 16 layer)                    0.559  0.465     0.522    n/a      0.517
HyenaDNA-medium                                 0.546  0.411     0.490    n/a      0.491
Evo2-1B 🔄                                        —      —         —      n/a        —
```

(Tabella completa e sorgente machine-readable: `model_pooling_performance.csv`.)

---

## 4. Output 2 — Migliore strategia di pooling per task

Aggregato per modello: **mean** vince per NTv3, HyenaDNA, Caduceus;
**attention** vince nominalmente per DNABERT-2 ma è un pareggio statistico
con mean (0.4285 vs 0.4283 MCC).

Distribuzione del pooling vincente per singolo dataset (57 dataset ×
4 modelli = 228 celle, da `best_pooling_by_task.csv`):

```
mean        124 / 228  (54%)
mean_max     65 / 228  (28%)
attention    35 / 228  (15%)
max           3 / 228  (1%)
cls           1 / 228  (<1%)
```

`mean` e `mean_max` insieme coprono l'82% dei casi migliori per singolo
task — `max` e `cls` sono quasi sempre dominati.

---

## 5. Output 3 — Confronto con k-mer usando il pooling migliore

**Parziale**, non dipende da E5 ma da E4 (probe fairness), gestito da
un'altra sessione sulla stessa macchina, ancora shardato in file non
consolidati (`results/rebuttal/E4_probe_fairness/probe_matrix_*_k6_q*.csv`,
non ancora uniti nel file master che la mia analisi legge). Solo 2 righe
disponibili in `best_pooling_vs_kmer.csv` al momento — **non sufficiente
per una claim quantitativa nella risposta al reviewer così com'è**.

Prima di scrivere questa sezione della risposta: verificare se E4 è stato
consolidato (`wc -l results/rebuttal/E4_probe_fairness/probe_matrix_*_k6.csv`
per ciascun modello — devono avvicinarsi a 57×4×3=684 righe l'uno) e
rigenerare con:
```bash
conda run -n cgr_bench python src/rebuttal/analyze_rebuttal_experiments.py \
  --e5 results/rebuttal/E5_pooling/pooling_results.csv \
  --e4 results/rebuttal/E4_probe_fairness/probe_matrix_ntv3_k6.csv \
  --out-root results/rebuttal
```
(ripetere `--e4` per gli altri modelli se la funzione `e5()` in
`analyze_rebuttal_experiments.py` viene estesa a più file — al momento
accetta un solo path E4 alla volta, guarda la funzione se serve confrontare
k-mer su più modelli contemporaneamente).

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
> Mamba), il mean pooling non è mai significativamente battuto da
> un'alternativa (max, CLS, concatenazione mean+max, attention
> parameter-free) sull'intero benchmark di 57 dataset, ed è la strategia
> singola migliore per tre modelli su quattro; sul quarto (DNABERT-2) il
> pareggio con `attention` è statisticamente non significativo. Lo stesso
> pattern si conferma sui sottoinsiemi promotore/splice-site/regulatory
> indicati dal reviewer. [🔄 Evo2 conferma/non conferma il pattern —
> aggiungere qui]. Questo risponde direttamente alla domanda scientifica
> posta: il mean pooling non sta scartando informazione posizionale
> recuperabile da strategie di pooling alternative — al contrario, nella
> maggioranza dei casi le supera.

**Nota onestà scientifica**: se Evo2 mostrasse un pattern diverso (es.
attention batte mean con margine ampio e significativo), la conclusione
sopra andrebbe riformulata come "3 modelli su 5" e il caso Evo2 discusso
esplicitamente, non nascosto — è stato esplicitamente richiesto all'agente
che esegue Evo2 di segnalarlo se succede (vedi criterio di successo nel
prompt di handoff).

---

## 8. Limitazioni da dichiarare esplicitamente nella risposta

1. **Evo2 mancante al momento della scrittura** (se la rebuttal deadline
   arriva prima del completamento) — dichiarare che è in corso su
   un'infrastruttura diversa per un vincolo tecnico (checkpoint FP8 di
   Evo2-1B richiede `transformer_engine`, che richiede un toolchain CUDA
   completo — nvcc + NCCL — non disponibile sulla macchina primaria), non
   per scelta metodologica.
2. **Confronto con k-mer (output 3) parziale** finché E4 non è consolidato.
3. **Probe singolo** (solo lineare) per E5, non l'intera matrice RF/lineare/MLP
   usata in E4 — coerente con i dati già esistenti prima di questo run,
   ma va detto se il reviewer chiede robustezza rispetto al classificatore.
4. Il pooling `attention` è deliberatamente **senza parametri addestrabili**
   (query = media mascherata) — utile per isolare l'effetto architetturale,
   ma è una scelta di design da rendere esplicita: un vero attention pooling
   con parametri appresi potrebbe comportarsi diversamente (non testato).

---

## 9. Come aggiornare questo dossier quando arriva Evo2

1. `git pull origin rebuttal-experiments` (l'altra macchina pusha i suoi
   risultati sullo stesso branch).
2. Rigenerare `E5_summary.md` con lo script di generazione (chiedimi di
   rilanciarlo, o cerca `gen_e5_summary.py` nello scratchpad di questa
   sessione).
3. Sostituire ogni 🔄 in questo documento con i numeri reali.
4. Ricontrollare la sezione 5 (k-mer) — se anche E4 è avanzato nel
   frattempo, questa è l'occasione per completarla insieme a Evo2.
