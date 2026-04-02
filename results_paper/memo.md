# Main Paper: Esperimenti e Plot

## Esperimenti (da includere)
- k-mer RF con k=4,5,6 su tutti i dataset (baseline principale)
- k-mer multi-scale k4_5_6 (baseline alternativa, stesso RF)
- FM RF: NTv3, HyenaDNA, DNABERT-2 (copertura completa)
- Confronto diretto k-mer vs FM con MCC come metrica primaria

## Plot (main paper)
- Box/violin MCC: `kmer_best(k4-6)` vs `fm_NTv3` vs `fm_hyenadna` vs `fm_DNABERT`
- Win/Loss bar: per ciascun FM vs `kmer_best(k4-6)` (MCC)
- Scatter `kmer_best` vs `fm_NTv3` con diagonale y=x + distribuzione di ΔMCC
- Plot ΔMCC vs `seq_len_mean` (mostra che NTv3 migliora con sequenze lunghe)

## Aggiunte Proposte (main o supplementare a seconda dello spazio)
- Efficienza: curva Pareto tempo vs MCC (k-mer vs FM) su subset rappresentativo
- Analisi “why”: ΔMCC vs `seq_len_mean` + gruppi short/mid/long per spiegare quando k‑mer perde
- Robustezza k: confronto k=4/5/6 (boxplot MCC) per mostrare che il k ottimale non è universale
