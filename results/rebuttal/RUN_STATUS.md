# Stato di esecuzione degli esperimenti di rebuttal

Ultimo aggiornamento: 2026-07-25 (E5 completo per NTv3+DNABERT-2+HyenaDNA; Caduceus in corso; Evo2 escluso definitivamente).

| Esperimento | Stato | Risultati attualmente presenti |
| --- | --- | --- |
| E4 | in esecuzione incrementale, prioritario | È in corso sulle 57 task per NTv3, HyenaDNA, DNABERT-2 ed Evo2-1B, con 4 rappresentazioni × RF/lineare/MLP. Sessione esterna, sharding e stato non gestiti da E5. |
| E5 | **NTv3/DNABERT-2/HyenaDNA completi (285+228+228 righe)**; **Caduceus in corso**; **Evo2 escluso definitivamente** | Pipeline: shard paralleli per pooling (uno per processo, `--n-jobs 16 --batch-size 64-128`), merge automatico via script Python dedicato (non più heredoc-in-`conda run`, che si era rivelato silenzioso e aveva causato una perdita di dati poi recuperata dalla cache embedding intatta — vedi `E5_pooling/README.md`). Risultati: mean pooling risulta il migliore o staticamente equivalente al migliore su tutti e 3 i modelli completati (test di Wilcoxon appaiato + correzione BH), incluse le analisi per promotore/splice-site/regulatory — vedi `pooling_significance.csv` e `pooling_significance_by_task_group.csv`. **Caduceus**: sbloccato (mamba_ssm ora installato in `cgr_bench`), 4 shard in corso (mean/max/mean_max/attention, CLS n/a). **Evo2**: il pacchetto `evo2` è ora disponibile nell'ambiente dedicato `evo2_bench` (compatibile con l'intera pipeline: pandas/sklearn/torch/transformers presenti), ma il checkpoint `evo2_1b_base` richiede anche `transformer_engine`, che **non è installabile** su questa macchina — mancano sia `nvcc` (compilatore CUDA) sia gli header NCCL, e non esiste una wheel precompilata per questa combinazione di CUDA/torch/Python. Tentata l'installazione (due volte, log in `/tmp/te_install.log` e `/tmp/te_install2.log`): primo tentativo fallito per disco pieno su `/` (`/tmp` è sullo stesso filesystem, 100% pieno), secondo tentativo (con TMPDIR su `/data2` e `--no-build-isolation`) arrivato fino alla compilazione C++, fallita per `fatal error: nccl.h: No such file or directory`. Servirebbe un intervento di sistema (CUDA toolkit + NCCL) fuori portata senza permessi admin — su decisione esplicita, Evo2 resta fuori da E5. |
| E6 | sospeso, da eseguire dopo E4/E5 | Il CSV contiene solo le righe già persistite per `iDHS-EL/DNase_I`; nessun processo E6 è attivo. |

Le tabelle e figure derivate per Caduceus/Evo2 devono essere considerate
descrittive del sottoinsieme indicato finché il CSV non copre Caduceus per
intero (Evo2 non è previsto). I runner sono riprendibili: le chiavi già
presenti nei CSV vengono saltate. Il confronto E5-vs-k-mer
(`best_pooling_vs_kmer.csv`) resta parziale finché E4 non è completo per il
modello corrispondente: va rigenerato con `analyze_rebuttal_experiments.py`
una volta che E4 ha finito.
