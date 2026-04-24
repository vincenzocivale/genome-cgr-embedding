## Plan: Esperimenti mancanti paper (P5-P8)

Completare i quattro esperimenti ad alta priorita (pooling robustness, decomposizione non-lineare upper bound, motif-level sul residuo NTv3 per splice, timing empirico k-mer) riusando pipeline e records esistenti. Approccio consigliato: estensioni minime e modulari alle pipeline correnti, esecuzione su tutti i 57 dataset dove richiesto, vincoli operativi conservativi su risorse condivise (n_workers basso), esclusione totale di Evo2.

**Steps**
1. Fase 0: Allineamento operativo e guardrail (bloccante)
   - Definire una policy unica di risorse per tutti gli script sperimentali: default n_workers=1 o 2, nessun parallelismo aggressivo inter-dataset, batch FM moderato.
   - Aggiungere validazione esplicita dei modelli ammessi nei nuovi entrypoint: NTv3, HyenaDNA, DNABERT-2; rifiuto esplicito di Evo2.
   - Dipendenza: nessuna.
2. Fase 1: Robustness al pooling strategy (Esperimento 5)
   - Estendere l’estrazione embedding FM con pooling parametrico: mean (baseline), max, cls quando semanticamente supportato dal tokenizer/modello.
   - Per ogni FM e per tutti i 57 dataset: rigenerare embedding per pooling, rieseguire linear probe e decomposizione (ridge lineare) mantenendo protocollo invariato.
   - Scrivere risultati in records dedicati o con suffisso pooling nelle colonne, includendo R2 Ridge e metriche downstream (MCC/AUROC).
   - Preparare tabella sintetica finale con delta rispetto a mean: Delta MCC (full/proj/resid) e Delta R2.
   - Dipendenze: step 1 (guardrail). Parallelizzabile per modello, non per saturazione CPU.
3. Fase 2: Decomposizione non-lineare come upper bound (Esperimento 6)
   - Implementare predittore non-lineare MLP 2-layer per mapping k-mer -> embedding FM, separato dalla pipeline ridge attuale.
   - Eseguire su tutti i dataset e tutti i FM, con protocollo di split identico al lineare.
   - Confrontare sistematicamente R2 lineare vs R2 non-lineare per dataset/modello/pooling (almeno mean; opzionalmente estendere a max/cls in secondo passaggio).
   - Aggiungere sintesi statistica del gap (mediana, IQR, percentuale dataset con miglioramento > soglia predefinita).
   - Dipendenze: step 2 per coerenza di reporting; puo partire su mean anche in parallelo con step 2 per ridurre tempo.
4. Fase 3: Motif-level analysis sul residuo NTv3 in splice (Esperimento 7)
   - Limitare analisi ai 4 dataset splice identificati.
   - Costruire analisi di importanza sul residuo (posizionale e/o feature-level) con metodo riproducibile: attribution su input o importance sulle feature derivate residuo.
   - Mappare segnali salienti ai motif/splice consensus noti (GT-AG, branch point, polypyrimidine tract) con criterio esplicito di match.
   - Produrre output interpretabili: figure per-dataset e tabella di overlap con motif noti.
   - Dipendenze: richiede residui da decomposizione (step 2 baseline mean), ma non dipende da timing.
5. Fase 4: Timing empirico k-mer e simmetria metodologica (Esperimento 8)
   - Rendere persistenti nel CSV di efficienza i timing wall-clock k-mer gia misurati in script (no sola stampa a stdout).
   - Misurare e riportare per rappresentazione: FLOPS analitici, FLOPS empirici, wall-clock a diverse lunghezze sequenza.
   - Garantire comparabilita FM vs k-mer con stessa metrica, stesse condizioni hardware, stesso protocollo di ripetizione.
   - Aggiornare figura/tabella cost scaling per includere k-mer in modo completo e non asimmetrico.
   - Dipendenze: indipendente da step 2-3, parallelizzabile con step 3.
6. Fase 5: Reporting integrato paper-ready
   - Consolidare tabelle richieste dai 4 esperimenti: (modello, pooling, R2 lineare/non-lineare, Delta MCC componenti, timing tripla metrica).
   - Aggiungere sezione interpretativa su splice: perche il residuo NTv3 cattura segnali posizionali oltre k-mer.
   - Documentare esplicitamente robustezza rispetto a obiezioni reviewer (pooling artifact, non-linearita latente, confronto timing asimmetrico).
   - Dipendenze: completa tutti gli step precedenti.

**Relevant files**
- /data2/home/vcivale/genome-cgr-embedding/src/embedders/fm_embedder.py — estendere pooling parametrico in FMEmbedder.embed_sequences e gestione maschera/pad.
- /data2/home/vcivale/genome-cgr-embedding/src/embedders/hyena_embedder.py — allineare API pooling a quella FM generale (se gia separata per Hyena).
- /data2/home/vcivale/genome-cgr-embedding/src/scripts/train_decomposition.py — aggiungere opzioni pooling e backend mapping non-lineare mantenendo output compatibile.
- /data2/home/vcivale/genome-cgr-embedding/src/training/ridge_mapping.py — baseline lineare di riferimento per confronto R2.
- /data2/home/vcivale/genome-cgr-embedding/src/training/linear_pipeline.py — riuso pattern training/eval per parte lineare e coerenza metriche.
- /data2/home/vcivale/genome-cgr-embedding/src/scripts/train_linear_probe.py — rerun sistematico per modello/pooling con stessi split.
- /data2/home/vcivale/genome-cgr-embedding/src/features/kmer_features.py — base feature extraction per mapping k-mer -> embedding.
- /data2/home/vcivale/genome-cgr-embedding/src/training/efficiency.py — centralizzare logging timing/FLOPS e persistenza k-mer.
- /data2/home/vcivale/genome-cgr-embedding/src/scripts/train_kmer_rf.py — collegare timing k-mer a records di efficienza persistenti.
- /data2/home/vcivale/genome-cgr-embedding/src/records/records.py — estendere schema colonne per pooling e confronto lineare/non-lineare.
- /data2/home/vcivale/genome-cgr-embedding/scripts/paper_stats_audit.py — riuso per breakdown splice/non-splice e sanity check delta.
- /data2/home/vcivale/genome-cgr-embedding/scripts/export_paper_figures.py — aggiornare figure cost scaling e tabelle finali.
- /data2/home/vcivale/genome-cgr-embedding/PLAN1.md — mantenere coerenza con scope gia definito ed esclusione Evo2.

**Verification**
1. Verifica robustezza pooling: per ogni modello controllare completezza di run su 57 dataset e assenza buchi nei records per mean/max/cls.
2. Verifica upper bound non-lineare: confermare presenza colonne R2 lineare e R2 MLP con confronto per dataset e aggregati (mediana/IQR).
3. Verifica splice motif-level: confermare output per esattamente 4 dataset splice, con almeno una evidenza quantitativa di overlap motif.
4. Verifica timing simmetrico: records_efficiency deve contenere sia FM sia k-mer con duration_sec, GFLOPS_per_seq, GFLOPS_total a lunghezze multiple.
5. Verifica vincoli server condiviso: log di run con n_workers basso e nessun utilizzo di Evo2 in comandi, config o records.
6. Verifica riproducibilita: rerun di un subset fisso (almeno 3 dataset, inclusi 1 splice e 1 non-splice) produce differenze entro tolleranza definita.

**Decisions**
- Incluso: Esperimenti 5, 6, 7, 8 richiesti dal paper con priorita alta.
- Incluso: esecuzione pooling e decomposizione su tutti i 57 dataset.
- Incluso: metodo non-lineare scelto MLP 2-layer (no Kernel Ridge in prima istanza).
- Incluso: focus motif-level solo sui 4 dataset splice.
- Escluso: Evo2 in ogni fase.
- Vincolo operativo: n_workers basso (default 1-2) e pianificazione conservativa su server condiviso.

**Further Considerations**
1. Priorita run consigliata: prima pooling+timing (basso rischio), poi MLP decomposition (medio), infine motif-level splice (medio-alto interpretativo).
2. Per ridurre rischio computazionale: checkpoint per modello e resume robusto per dataset mancanti in ogni script.
3. Per reviewer-proof reporting: riportare sempre differenze rispetto a baseline mean+ridge, non solo valori assoluti.
