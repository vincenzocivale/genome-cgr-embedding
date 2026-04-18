# K-mer vs Foundation Model Embeddings — Sintesi trasversale degli esperimenti

## 1. Il risultato centrale

Su 57 dataset genomici, NTv3-650M è l'unico Foundation Model competitivo con i k-mer. Batte il best k-mer (k∈{4,5,6}) in 29/57 dataset (51%), con un Δ MCC medio di −0.027 a suo favore. HyenaDNA-160k e DNABERT-2 sono sistematicamente inferiori ai k-mer (k-mer vince nell'81% e 84% dei casi, rispettivamente).

Il quadro cambia radicalmente per categoria biologica e per lunghezza della sequenza. Escludendo i 4 dataset di splicing (dove NTv3 domina con Δ medio −0.275), il vantaggio complessivo di NTv3 si riduce a Δ = −0.008: sostanziale parità.

## 2. Tabella sinottica per categoria

| Categoria | N | Seq len | GC | Best k | MCC kmer | MCC NTv3 | NTv3 vince | R² | Resid > km | Sinergia | Δ concat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Istoni (Yeast) | 10 | 500 | 0.38 | 6 | 0.471 | 0.542 | 9/10 | 0.49 | 3/10 | +0.062 | −0.003 |
| Metilazione | 8 | 41 | 0.49 | 4 | 0.213 | 0.197 | 1/8 | 0.57 | 2/8 | +0.002 | −0.024 |
| Promotori | 15 | 437 | 0.53 | 4 | 0.707 | 0.730 | 9/15 | 0.44 | 1/15 | +0.006 | −0.008 |
| Splice | 4 | 500 | 0.45 | 4 | 0.388 | 0.662 | 4/4 | 0.51 | 4/4 | +0.076 | −0.003 |
| TFBS | 10 | 101 | 0.56 | 5 | 0.574 | 0.557 | 1/10 | 0.49 | 0/10 | −0.005 | −0.006 |
| Enhancer/generico | 10 | 357 | 0.44 | 4 | 0.617 | 0.584 | 5/10 | 0.43 | 3/10 | −0.010 | −0.017 |

## 3. Quando vince chi: il ruolo della lunghezza

La lunghezza della sequenza è il predittore più forte del vantaggio relativo (Spearman r = −0.464, p = 0.0003): sequenze più lunghe favoriscono NTv3, sequenze corte favoriscono i k-mer. La correlazione regge anche escludendo lo splice (r = −0.423, p = 0.002).

- **≤100 bp**: k-mer vince nel 47% dei dataset, NTv3 nel 27%. Δ mediano +0.009.
- **101–300 bp**: NTv3 vince nel 50%, k-mer nel 14%. Δ mediano −0.033.
- **301–600 bp**: NTv3 vince nel 71%, k-mer nel 7%. Δ mediano −0.056.
- **>600 bp**: NTv3 vince nel 67%, k-mer nello 0%. Δ mediano −0.152.

Altre variabili: GC content (r = +0.283, p = 0.033) e class balance (r = +0.278, p = 0.036) hanno correlazione marginale. La dimensione del training set non è significativa (r = +0.045, p = 0.74).

## 4. La scelta del k ottimale

k=4 è ottimale in 28/57 dataset, k=5 in 13, k=6 in 16. La distribuzione è fortemente associata alla categoria biologica:

- **Istoni (Yeast)**: k=6 in 9/10 casi. GC basso (~0.38), sequenze lunghe (500 bp).
- **Promotori**: k=4 in 11/15 casi. GC più alto (~0.53), alta pressione composizionale.
- **TFBS**: k=5 in 5/10 casi. Bilanciato.

GC content è inversamente correlato al k ottimale (r = −0.481, p = 0.0002): ad alto GC corrisponde k piccolo. La sensibilità alla scelta di k varia: il range MCC (best − worst k) è in media 0.050, ma raggiunge 0.204 nei promotori a 70 bp.

## 5. Decomposizione dell'embedding (proiezione + residuo)

### 5.1 Ridge R²: tre profili di ridondanza composizionale

La Ridge regression misura quanto dell'embedding FM è linearmente ricostruibile dalle frequenze k-mer (k=6):

| Modello | R² medio | R² min | R² max | Interpretazione |
|---|---|---|---|---|
| NTv3-650M | 0.48 | 0.14 (Arabidopsis TATA) | 0.92 (covid_variants) | Circa metà composizionale |
| HyenaDNA-160k | 0.71 | 0.37 (regulatory_region_type) | 0.97 (covid_variants) | Prevalentemente composizionale |
| DNABERT-2 | 0.21 | 0.05 | 0.76 | Prevalentemente non-composizionale |

La distribuzione del R² di NTv3 è relativamente omogenea tra categorie (range 0.43–0.57), con i dataset di metilazione leggermente più alti e quelli di enhancer/generico più bassi. Il R² non correla con nessuna variabile di outcome (Δ MCC, utilità del residuo, gap full−proiezione).

### 5.2 Proiezione ≈ k-mer: equivalenza statistica

La proiezione (componente dell'embedding spiegabile dai k-mer) ha performance identica ai k-mer diretti:

- Δ(proiezione − best k-mer) medio: −0.004, mediano: +0.006
- Wilcoxon non significativo (p = 0.35)
- Proiezione > k-mer in 21/57, k-mer > proiezione in 15/57, parità in 21/57

L'equivalenza è consistente tra categorie, senza eccezioni sistematiche. Il R² non correla con il gap proiezione−k-mer (r = −0.07, n.s.). Questo conferma che la parte composizionale dell'embedding NTv3 non contiene informazione aggiuntiva rispetto ai k-mer: è una trasformazione lineare dello stesso segnale.

### 5.3 Il residuo: informativo solo in casi specifici

Il residuo (componente non-composizionale) è globalmente inferiore ai k-mer (k-mer > residuo in 36/57, Δ medio −0.044), ma supera i k-mer in 13/57 dataset. I 13 dataset "residuo vince" si raggruppano in due cluster:

**Cluster 1 — Splice (4 dataset, Δ fino a +0.267).** Il residuo cattura segnali posizionali dei siti di splicing (branch point, sequenze consenso) che i k-mer per costruzione perdono. È il caso biologicamente più chiaro.

**Cluster 2 — Eterogeneo (9 dataset, Δ da +0.021 a +0.092).** Include coding regions, istoni, enhancer strength, promotori Arabidopsis. I guadagni sono modesti. I dataset "residuo vince" tendono ad avere sequenze più lunghe (mediana 400 bp vs 101 bp per "k-mer vince"), R² più basso (0.43 vs 0.50), e class balance inferiore (0.70 vs 0.94), ma nessuna di queste differenze raggiunge la significatività statistica (Mann-Whitney p = 0.11 per seq_len).

### 5.4 Sinergia: l'interazione tra componenti

Il full embedding supera max(proiezione, residuo) di +0.016 in media — un effetto di sinergia che indica che il RF cattura interazioni tra le due componenti quando sono presentate insieme nell'embedding completo. L'effetto è fortemente eterogeneo:

| Categoria | Sinergia media | Dataset con sinergia > 0.02 | Componente dominante |
|---|---|---|---|
| Splice | +0.076 | 4/4 | residuo (4/4) |
| Istoni (Yeast) | +0.062 | 8/10 | proiezione (8/10) |
| Promotori | +0.006 | 2/15 | proiezione (13/15) |
| Metilazione | +0.002 | 1/8 | proiezione (8/8) |
| TFBS | −0.005 | 1/10 | proiezione (10/10) |
| Enhancer/generico | −0.010 | 2/10 | proiezione (8/10) |

Due categorie mostrano sinergia forte: splice e istoni. Il meccanismo è però diverso:

- **Splice**: il residuo è la componente dominante e supera i k-mer da solo. La sinergia aggiunge ulteriore vantaggio (full 0.662 >> residuo 0.587 >> proiezione 0.417).
- **Istoni**: la proiezione domina, il residuo da solo è debole (0.411 < k-mer 0.471), eppure il full embedding è molto superiore (0.542). Qui la sinergia non viene dal residuo isolato, ma dall'interazione: il RF nell'embedding completo sfrutta pattern che emergono solo dalla combinazione delle due componenti.

La sinergia correla con la lunghezza della sequenza (r = +0.48, p = 0.0001) e, fortemente, con il vantaggio globale di NTv3 sui k-mer (r = +0.75, p < 0.0001). Questo chiude il cerchio: NTv3 vince quando la sinergia tra componenti composizionali e non-composizionali è alta, e questo accade prevalentemente su sequenze lunghe.

### 5.5 Confronto tra modelli: tre architetture, tre storie

La decomposizione rivela tre profili radicalmente diversi:

| | NTv3-650M | HyenaDNA-160k | DNABERT-2 |
|---|---|---|---|
| R² medio | 0.48 | 0.71 | 0.21 |
| MCC proiezione | 0.531 | 0.469 | 0.510 |
| MCC residuo | 0.491 | 0.302 | 0.332 |
| MCC full | 0.562 | 0.448 | 0.427 |
| Gap full−proj | +0.030 | −0.021 | −0.083 |
| Proiezione > full | 7/57 (12%) | 27/57 (47%) | 51/57 (89%) |

**NTv3** è l'unico modello dove il full embedding supera consistentemente la proiezione. La componente non-composizionale contribuisce positivamente: il gap full−proj è +0.030, e la proiezione batte il full solo nel 12% dei dataset.

**HyenaDNA** è prevalentemente composizionale (R² = 0.71), ma la proiezione è peggiore dei k-mer (−0.066): l'embedding rappresenta l'informazione composizionale in modo meno efficiente dei k-mer diretti. La proiezione batte il full nel 47% dei casi — il residuo è spesso rumore che danneggia la performance.

**DNABERT-2** è il caso più paradossale. Ha il R² più basso (0.21), cioè l'embedding apparentemente più "originale", ma la componente non-composizionale è massicciamente dannosa: la proiezione (0.510) batte il full (0.427) nell'89% dei dataset. Il gap full−proj è −0.083. Il tokenizzatore BPE di DNABERT-2 crea una rappresentazione diversa dai k-mer, ma questa diversità non è informativa — è rumore.

**Implicazione**: non basta che un FM catturi segnali diversi dai k-mer; quei segnali devono essere biologicamente rilevanti. Solo NTv3, con 650M di parametri e un pre-training su larga scala, riesce a codificare informazione non-composizionale utile.

### 5.6 R² non predice nulla

La ridondanza composizionale (R²) non correla con nessuna variabile di outcome:

| Correlazione | Spearman r | p |
|---|---|---|
| R² vs Δ(full − k-mer) | −0.128 | 0.34 |
| R² vs Δ(residuo − k-mer) | −0.162 | 0.23 |
| R² vs gap(full − proj) | −0.112 | 0.41 |
| R² vs Δ(proj − k-mer) | −0.073 | 0.59 |
| R² vs seq_len | −0.041 | 0.76 |

Sapere quanto l'embedding è composizionale non dice nulla sulla qualità della componente non-composizionale, né sul vantaggio dell'FM. DNABERT-2 è il controesempio perfetto: R² basso, performance bassa.

## 6. Concatenazione FM + k-mer

La concatenazione del best FM e del best k-mer non porta mai un guadagno sostanziale: 0/57 dataset con Δ > +0.02 rispetto al miglior singolo, 9/57 con peggioramento significativo. Il Wilcoxon è significativo nella direzione negativa (p < 0.0001).

I casi di peggioramento maggiore sono dataset dove k-mer domina (promoter_tata_70bp: −0.091, regulatory_region_type: −0.078): aggiungere features FM introduce rumore nella rappresentazione k-mer. Per lo splice, dove FM domina nettamente, la concatenazione è neutra (−0.003): il segnale FM è forte abbastanza da sopravvivere.

NTv3 viene scelto come best FM nella concatenazione in 50/57 dataset. La dimensionalità totale (FM 1536 + k-mer fino a 4096 = 5632 features) potrebbe contribuire al problema: con training set spesso sotto i 10.000 campioni, il RF fatica a gestire lo spazio ad alta dimensionalità senza che le features meno informative aggiungano rumore.

Il risultato è coerente con la decomposizione: poiché proiezione ≈ k-mer, la concatenazione aggiunge essenzialmente il residuo ai k-mer. Ma il residuo è globalmente più debole dei k-mer, quindi diluisce il segnale anziché rafforzarlo.

## 7. Effetto PCA

La PCA peggiora tutte le rappresentazioni, ma in modo asimmetrico: penalizza i k-mer (Δ medio da −0.027 per k=4 a −0.041 per k=6) più degli FM (Δ medio −0.013 per NTv3). La penalizzazione scala con la dimensionalità originale. Tutti gli effetti sono significativi (Wilcoxon p < 0.01). 13/57 verdetti si ribaltano sotto PCA. Il confronto sulle rappresentazioni native (senza PCA) è il più corretto.

L'effetto PCA su NTv3 è uniformemente negativo su tutte le categorie (da −0.003 per metilazione a −0.020 per enhancer/generico).

k=7 e k=8 sono disponibili solo con PCA e per 5 dataset: dati insufficienti per conclusioni.

## 8. Costo computazionale (GFLOPS)

### 8.1 Ordini di grandezza

Il costo per sequenza dei Foundation Models è superiore ai k-mer di 4–6 ordini di grandezza:

| Metodo | GFLOPS a 100 bp | GFLOPS a 500 bp | GFLOPS a 2000 bp | Scaling (2000/100) |
|---|---|---|---|---|
| k-mer (k=6) | 0.000017 | 0.000019 | 0.000026 | 1.5x |
| NTv3-650M | 0.916 | 3.665 | 14.673 | 16x |
| HyenaDNA-160k | 1.261 | 6.305 | 25.220 | 20x |
| DNABERT-2 | 5.619 | 26.926 | 103.020 | 18x |

A 500 bp (lunghezza tipica), NTv3 costa ~193.000x i k-mer, HyenaDNA ~332.000x, DNABERT-2 ~1.4 milioni di volte. I k-mer scalano in modo quasi costante con la lunghezza (conteggio lineare, nessun forward pass), mentre gli FM scalano linearmente (16–20x da 100 a 2000 bp).

### 8.2 Un paradosso architetturale

HyenaDNA ha solo 6.5M di parametri (100x meno di NTv3), eppure costa più GFLOPS per sequenza a ogni lunghezza testata. L'architettura Hyena è computazionalmente più pesante per token. Combinato con la performance inferiore (k-mer vince nell'81% dei dataset), HyenaDNA è inefficiente su entrambi i fronti.

DNABERT-2 (117M parametri, 6x meno di NTv3) costa 7x più GFLOPS di NTv3 a parità di lunghezza. Il tokenizzatore BPE, che produce più token per sequenza, moltiplica il costo. Il rapporto costo/performance è il peggiore tra i tre modelli: 1.4 milioni di volte più costoso dei k-mer, ma peggiore in 84% dei dataset.

### 8.3 Costo-efficacia per categoria biologica

| Categoria | Seq len tipica | Δ MCC (NTv3 − kmer) | Costo NTv3/kmer | Giustificato? |
|---|---|---|---|---|
| Metilazione | 41 bp | −0.016 | ~15.000x | No: NTv3 peggiore e più costoso |
| TFBS | 101 bp | −0.017 | ~37.000x | No: NTv3 peggiore e più costoso |
| Enhancer/generico | 357 bp | −0.033 | ~131.000x | No: NTv3 peggiore e più costoso |
| Promotori | 437 bp | +0.023 | ~160.000x | Marginale: +0.023 MCC per 160.000x costo |
| Istoni (Yeast) | 500 bp | +0.071 | ~183.000x | Dipende dall'applicazione |
| Splice | 500 bp | +0.275 | ~183.000x | Sì: guadagno netto e biologicamente motivato |

Solo lo splice offre un guadagno di performance che potrebbe giustificare il costo computazionale. Per istoni e promotori il guadagno esiste ma va bilanciato con un fattore ~10⁵ in costo. Per metilazione, TFBS e enhancer, NTv3 è sia più costoso sia peggiore.

Escludendo lo splice, il guadagno medio di NTv3 è +0.008 MCC per un fattore ~193.000x in GFLOPS: un rapporto costo/beneficio difficile da giustificare in qualsiasi scenario applicativo.

## 9. Correlazioni trasversali tra esperimenti

Le correlazioni più forti collegano la decomposizione dell'embedding al confronto globale:

| Correlazione | Spearman r | p |
|---|---|---|
| Gap(full−proj) vs Δ(kmer−NTv3) | −0.777 | < 0.0001 |
| Δ(resid−kmer) vs Δ(kmer−NTv3) | −0.765 | < 0.0001 |
| Sinergia vs Δ(full−kmer) | +0.752 | < 0.0001 |
| seq_len vs gap(full−proj) | +0.478 | 0.0002 |
| seq_len vs Δ(kmer−NTv3) | −0.464 | 0.0003 |
| seq_len vs sinergia | +0.482 | 0.0001 |

Queste sei correlazioni formano un quadro coerente: sequenze lunghe → il residuo e la sinergia tra componenti contribuiscono di più → il full embedding si distacca dalla proiezione → NTv3 vince. Il Ridge R² non partecipa a questa catena causale.

## 10. Discrepanze tra note e dati

1. **DNABERT-2**: le note riportavano 20/57 dataset disponibili, il CSV ne contiene 57. Esperimenti completati.
2. **Decomposizione**: le note riportavano 42 dataset completi, il CSV ne ha 57.
3. **Rappresentazioni aggiuntive nel CSV**: Evo2-7B (38/57), one-hot encoding (512/1024/2048 bp), k-mer multi-risoluzione (k4+5+6), token-level embeddings. Da escludere dal paper.
4. **iPro-WAEL/Promoter_B_amylolique**: presente solo in records_pca.csv. Da escludere.

## 11. Messaggi chiave per il paper

1. **NTv3 è l'unico FM competitivo**, ma il suo vantaggio è concentrato su sequenze lunghe e su un task specifico (splicing). Senza splice, la competizione è sostanzialmente pari.

2. **Il vantaggio di NTv3, quando esiste, è guidato dalla sinergia tra componenti composizionali e non-composizionali** dell'embedding. Il residuo isolato è più debole dei k-mer nella maggior parte dei casi, ma nell'embedding completo le due componenti interagiscono producendo un segnale superiore alla somma delle parti. Questo effetto è forte nello splice (dove il residuo porta informazione posizionale unica) e negli istoni (dove il meccanismo è meno trasparente).

3. **I k-mer restano una baseline fortissima**: semplici, interpretabili, e superiori a 2 FM su 3. La proiezione dell'embedding NTv3 sui k-mer è statisticamente equivalente ai k-mer stessi (Wilcoxon n.s.), confermando che la componente composizionale degli FM non aggiunge valore.

4. **Non basta essere diversi dai k-mer per essere migliori.** DNABERT-2 ha l'embedding meno ridondante (R² = 0.21), ma la sua componente non-composizionale è attivamente dannosa: la proiezione batte il full nell'89% dei casi. HyenaDNA è prevalentemente composizionale ma rappresenta l'informazione composizionale peggio dei k-mer diretti. Il R² non predice nessuna variabile di outcome.

5. **Combinare FM e k-mer non aiuta**: la concatenazione non aggiunge mai un guadagno >0.02 MCC. Il risultato è coerente con la decomposizione: concatenare equivale ad aggiungere il residuo ai k-mer, ma il residuo è globalmente più debole.

6. **La lunghezza della sequenza è il predittore unificante**: guida il vantaggio di NTv3 (r = −0.46), il gap full−proiezione (r = +0.48), e l'indice di sinergia (r = +0.48). Il filo conduttore è che sequenze più lunghe offrono più contesto posizionale, esattamente il tipo di informazione che i k-mer non catturano e che NTv3 codifica nella componente non-composizionale.

7. **La scelta di k è biologicamente informativa**: correlata al GC content e alla lunghezza, con pattern specifici per categoria.

8. **La dimensione del training set non predice quale metodo vince**: non si tratta di un problema di dati insufficienti per gli FM.

9. **Il costo computazionale degli FM è superiore di 5 ordini di grandezza**, senza un guadagno proporzionale. A 500 bp NTv3 costa ~193.000x i k-mer per un Δ MCC di +0.027 (o +0.008 senza splice). DNABERT-2 raggiunge 1.4 milioni di volte il costo dei k-mer con performance peggiore. HyenaDNA, nonostante 100x meno parametri di NTv3, è più costoso per sequenza a causa dell'architettura.
