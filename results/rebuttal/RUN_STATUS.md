# Stato di esecuzione degli esperimenti di rebuttal

Ultimo aggiornamento: 2026-07-27 (E5 completo per NTv3+DNABERT-2+HyenaDNA+Caduceus con CLS/last_token corretti; Evo2 escluso definitivamente).

| Esperimento | Stato | Risultati attualmente presenti |
| --- | --- | --- |
| E4 | in esecuzione incrementale, prioritario | È in corso sulle 57 task per NTv3, HyenaDNA, DNABERT-2 ed Evo2-1B, con 4 rappresentazioni × RF/lineare/MLP. Sessione esterna, sharding e stato non gestiti da E5. |
| E5 | **4/4 modelli completi (1140 righe)**; **Evo2 escluso definitivamente** | Pipeline: shard paralleli per pooling/dataset, merge automatico via script Python dedicato (non più heredoc-in-`conda run`, che si era rivelato silenzioso e aveva causato una perdita di dati poi recuperata dalla cache embedding intatta). **Corretto un bug di applicabilità CLS**: la disponibilità del CLS pooling era legata a una lista hardcoded ("char-level = niente CLS"), errata in entrambe le direzioni — verificato empiricamente su ogni tokenizer che solo DNABERT-2 antepone davvero `[CLS]` in posizione 0 (era escluso per errore), mentre NTv3 dichiara `<cls>` ma non lo antepone mai (i risultati "cls" di NTv3 sono stati rietichettati `first_token`). Aggiunto anche `last_token` (posizione dell'ultimo `[SEP]`/EOS appeso) per HyenaDNA e Caduceus, che rispondono alla richiesta del reviewer "CLS **o token equivalente**". **Risultato**: il CLS reale di DNABERT-2 è l'unica alternativa a battere `mean` in modo significativo in tutta la matrice (MCC 0.463 vs 0.428, 41/57 vittorie, p=2.2e-5) — atteso, essendo l'unico token effettivamente addestrato a riassumere la sequenza; perde comunque contro il baseline k-mer (p=0.0055 su 54/57 dataset). Tutte le altre alternative (max, mean+max, attention, first_token, last_token) restano pareggiate o significativamente peggiori di mean su tutti e 4 i modelli — vedi `pooling_significance.csv`, `pooling_significance_by_task_group.csv` e `docs/rebuttal/E5_report.md` per il dossier completo. **Evo2**: il pacchetto `evo2` è disponibile nell'ambiente dedicato `evo2_bench`, ma il checkpoint `evo2_1b_base` richiede anche `transformer_engine`, **non installabile** su questa macchina — mancano sia `nvcc` (compilatore CUDA) sia gli header NCCL, nessuna wheel precompilata compatibile. Tentata l'installazione due volte (log in `/tmp/te_install.log` e `/tmp/te_install2.log`): primo tentativo fallito per disco pieno su `/`, secondo (TMPDIR su `/data2`, `--no-build-isolation`) arrivato alla compilazione C++, fallito per `fatal error: nccl.h: No such file or directory`. Serve un intervento di sistema (CUDA toolkit + NCCL) fuori portata senza permessi admin — su decisione esplicita, Evo2 resta fuori da E5, con handoff pronto per un'altra macchina (vedi prompt consegnato). |
| E6 | sospeso, da eseguire dopo E4/E5 | Il CSV contiene solo le righe già persistite per `iDHS-EL/DNase_I`; nessun processo E6 è attivo. |

Le tabelle e figure derivate per Caduceus/Evo2 devono essere considerate
descrittive del sottoinsieme indicato finché il CSV non copre Caduceus per
intero (Evo2 non è previsto). I runner sono riprendibili: le chiavi già
presenti nei CSV vengono saltate. Il confronto E5-vs-k-mer
(`best_pooling_vs_kmer.csv`) resta parziale finché E4 non è completo per il
modello corrispondente: va rigenerato con `analyze_rebuttal_experiments.py`
una volta che E4 ha finito.
