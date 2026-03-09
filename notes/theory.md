# CGR-based Sequence Representation: Theoretical Notes

## La Chaos Game Representation (CGR)

La CGR è una trasformazione geometrica che mappa una sequenza DNA in un piano 2D.
Si parte dal centro del quadrato unitario e a ogni base si sposta il punto a metà strada verso il vertice corrispondente: A in basso a sinistra, C in alto a sinistra, G in alto a destra, T in basso a destra. Il punto finale dipende dall'intera storia della sequenza letta fino a quel momento.

La proprietà fondamentale della CGR è che ogni sotto-sequenza di lunghezza k occupa una regione ben definita e non sovrapposta della griglia. In altre parole, la CGR è una codifica spazialmente consistente dei k-mer: sub-sequenze simili cadono vicine nello spazio CGR, e sub-sequenze diverse cadono lontane. Questo la rende naturalmente adatta a rappresentare la struttura locale di una sequenza.

## FCGR: dalla CGR a una matrice di frequenze

La Frequency CGR (FCGR) discretizza la CGR: si sovrappone una griglia NxN alla regione unitaria e si conta quante volte il percorso CGR cade in ogni cella. Con una griglia 2^k × 2^k, ogni cella corrisponde esattamente a un k-mer specifico. La FCGR è quindi una rappresentazione visuale della distribuzione dei k-mer nella sequenza, preservando le relazioni di prossimità spaziale tra pattern simili.

Con grid_size=128 (=2^7), la griglia contiene implicitamente informazioni fino a k=7. Il k-mer di lunghezza desiderata si estrae tramite sum-pooling: le celle vengono aggregate a blocchi di dimensione (128/2^k)^2, ottenendo un vettore di 4^k frequenze normalizzate.

## K-mer fisso come caso speciale

Il k-mer convenzionale e la FCGR sono matematicamente equivalenti quando si usa sum-pooling uniforme. La differenza concettuale è che la FCGR permette di scegliere la risoluzione ex-post senza ricalcolare nulla, e che le feature mantengono la loro posizione spaziale (k-mer vicini nella gerarchia CGR sono vicini nel vettore), proprietà che un Random Forest non sfrutta direttamente ma che è rilevante per CNN o attention-based models.

## Il problema del k fisso

Con k fisso, tutte le regioni della griglia vengono aggregate alla stessa risoluzione, indipendentemente da quanto segnale contengano. Regioni dense (k-mer frequenti, indicativi di struttura biologica) e regioni sparse (k-mer rari o assenti) ricevono lo stesso trattamento. Questo è inefficiente: le regioni dense meritano maggiore risoluzione, quelle sparse possono essere compresse senza perdita informativa.

## QuadTree adattivo come soluzione

L'approccio QuadTree parte dalla griglia intera e la suddivide ricorsivamente solo dove è necessario. Il criterio di suddivisione risponde alla domanda: "questa cella contiene struttura interna che vale la pena risolvere, o la distribuzione tra i quattro figli è essenzialmente uniforme?"

Una distribuzione uniforme tra i quattro quadranti significa che la massa è distribuita senza pattern rilevante: non c'è informazione aggiuntiva da guadagnare suddividendo. Una distribuzione disomogenea invece indica che la struttura biologica preferisce alcune sotto-regioni, e vale la pena scendere di risoluzione per catturarla.

Il risultato è una rappresentazione a risoluzione variabile: alta nelle regioni della griglia dove si concentrano i k-mer biologicamente rilevanti, bassa dove la sequenza è uniforme o dove mancano osservazioni sufficienti.

## Perché non l'entropia come criterio

L'entropia di Shannon misura la disomogeneità della distribuzione ed è concettualmente corretta. Il problema pratico è che con griglie sparse (200bp su 128x128 = ~200 conteggi su 16384 celle) l'entropia tra quattro quadranti tende sempre verso 2.0 bit (il massimo per 4 categorie), semplicemente perché con pochi conteggi qualsiasi distribuzione empirica sembra quasi uniforme. Non esiste una soglia di entropia che funzioni sia per sequenze corte che lunghe.

## Il test chi-quadrato come criterio robusto

Il test chi-quadrato di Pearson per l'uniformità è più robusto perché normalizza la deviazione dall'atteso rispetto alla varianza statistica attesa sotto l'ipotesi nulla. Formalmente, dati i conteggi (o1, o2, o3, o4) nei quattro quadranti con totale N, il test verifica se la distribuzione si discosta significativamente dall'uniforme (N/4 ciascuno).

La statistica è:

    χ² = Σ (oi - N/4)² / (N/4)

Il p-value risultante è universalmente interpretabile come probabilità di osservare una distribuzione così disomogenea per caso sotto l'ipotesi nulla di uniformità. Con p < 0.05 si conclude con 95% di confidenza che esiste struttura nella cella, e vale la pena suddividere.

Questa scelta ha tre vantaggi pratici:
1. La soglia (p-value) non dipende dalla lunghezza della sequenza né dalla profondità.
2. Il parametro `min_count` esclude le celle con troppo pochi conteggi per avere un test affidabile (sotto ~8 conteggi la distribuzione chi-quadrato non è valida).
3. Con `split_threshold=1.0` e `min_count=0` il QuadTree degenera esattamente nel k-mer fisso (verificato).

## Rappresentazione a dimensione fissa

Il QuadTree produce un numero variabile di foglie per sequenza diversa, il che è incompatibile con un classificatore a input fisso. La soluzione adottata è il layout Z-order (Morton order): si pre-alloca un vettore di 4^max_depth slot, e ogni foglia a profondità d < max_depth propaga la propria densità equamente ai 4^(max_depth - d) slot discendenti. Le foglie terminali occupano esattamente uno slot ciascuna. Il vettore risultante ha sempre dimensione fissa, ma la distribuzione interna dei valori riflette la struttura adattiva: regioni uniformi producono blocchi di valori identici, regioni complesse producono valori eterogenei.

## Rappresentazione combinata [k-mer | QuadTree]

Il k-mer fisso e il QuadTree adattivo catturano aspetti complementari della stessa matrice FCGR. Il k-mer a risoluzione fissa garantisce una baseline robusta e uniforme su tutte le regioni. Il QuadTree aggiunge informazione sulla struttura gerarchica della distribuzione spaziale. La concatenazione [kmer_k | adaptive_qt] può essere vantaggiosa quando le due rappresentazioni sono ortogonali — il Random Forest con max_features='sqrt' esplora sottoinsiemi casuali di feature ad ogni split, permettendo al modello di sfruttare entrambe le fonti.

---

## Misurazioni per la scelta della soglia ottimale

La scelta del p-value ottimale per `split_threshold` non è banale. Elenchiamo le misurazioni che permettono di guidare la scelta in modo empiricamente fondato.

### 1. Sparsità adattiva media

Misurare la **frazione media di foglie premature** (celle che non vengono suddivise fino a max_depth) al variare della soglia. Una buona soglia lascia suddividere le regioni informative e comprime quelle vuote, producendo una sparsità moderata (~50-80% di foglie premature). Soglie troppo basse producono quasi tutto compresso (1 foglia), soglie troppo alte producono espansione totale.

```python
# Per ogni sequenza: contare quante posizioni nel vettore output
# hanno lo stesso valore del genitore (= cella non suddivisa)
```

### 2. Varianza inter-classe delle feature

Misurare la **varianza delle feature** separatamente per classe positiva e negativa (per dataset binari). Un buon vettore di feature ha alta varianza inter-classe (le classi si separano) e bassa varianza intra-classe. Questo si può quantificare con il rapporto F di Fisher per feature, o con l'analisi delle componenti principali.

### 3. Mutual information con le label

Calcolare la **mutual information** tra le feature estratte e le label. Testare diverse soglie e misurare la mutual information totale: la soglia ottimale è quella che massimizza l'informazione rilevante per la classificazione.

### 4. Stabilità cross-dataset

Per un iperparametro da usare in modo universale, misurare la **varianza delle prestazioni** (MCC, AUROC) al variare della soglia su un sottoinsieme di dataset di training. La soglia più stabile (minore varianza tra dataset diversi) è preferibile anche se non è il massimo assoluto su nessun dataset singolo.

### 5. Numero medio di regioni distinte per classe

Contare quanti **slot distinti del vettore** assumono valori diversi tra sequenze positive e negative. Una buona soglia produce vettori con regioni discriminative visibili — questo è verificabile visualmente con heatmap delle differenze medie per classe.

### 6. Correlazione con la lunghezza della sequenza

Verificare se la **profondità media di suddivisione** scala con la lunghezza della sequenza come atteso. Per sequenze più lunghe ci sono più conteggi, quindi il test chi-quadrato è più potente e si suddivide di più. La soglia ideale produce una scaling ragionevole: non troppo veloce (che produrrebbe k-mer fisso per qualsiasi sequenza lunga) né troppo lenta (che produrrebbe nessuna suddivisione per sequenze corte).
