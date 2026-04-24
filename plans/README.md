# Experiment Plans

One plan file per paper block. Each plan is self-contained: scope, status, commands, outputs, verification. Esecuzione a cascata secondo le dipendenze elencate sotto.

## Stato dei 32 esperimenti del paper

Legenda: **✅ Complete** · **🟡 Partial** · **⛔ Absent**

### Blocco 1 — Benchmark comparativo
| # | Esperimento | Stato |
|---|---|---|
| 1 | Classification RF 57 × (k-mer k=4,5,6 + NTv3 + HyenaDNA + DNABERT-2) | ✅ |
| 2 | Regression benchmark LRA (3 task, HyenaDNA-1M) | 🟡 (1 cella DNABERT-2 manca) |
| 3 | Error bars (std across folds) | ⛔ |
| 4 | AUROC oltre a MCC | ✅ |
| 5 | Linear probe alternativo a RF | 🟡 (LRA LP FM mancante) |
| 6 | Pooling max/cls | 🟡 (solo NTv3 max completo) |
| 7 | Canonical k-mer | ✅ |
| 8 | FM aggiuntivo (Caduceus-Ph) | ⛔ |

### Blocco 2 — Stratificazione
| # | Esperimento | Stato |
|---|---|---|
| 9 | Win/loss & ΔMCC per biocat | ⛔ |
| 10 | Win/loss per bin seq_len | ⛔ |
| 11 | GC content vs k ottimale | ⛔ |
| 12 | Class balance / n_train vs vantaggio FM | ⛔ |
| 13 | Few-shot (10/25/50/100%) | ⛔ |

### Blocco 3 — Decomposizione
| # | Esperimento | Stato |
|---|---|---|
| 14 | Ridge k-mer → embedding | 🟡 (HyenaDNA, DNABERT-2 mancano k=4,5) |
| 15 | R² distribution | 🟡 (come 14) |
| 16 | Projection come input | 🟡 (come 14) |
| 17 | Residual come input | 🟡 (come 14) |
| 18 | Full vs Proj vs Resid | 🟡 (derivato) |
| 19 | MLP decomposition | 🟡 (solo 1/57) |
| 20 | λ Ridge via CV documentato | ✅ |
| 21 | Concat FM + k-mer | ✅ |

### Blocco 4 — Analisi meccanicistica
| # | Esperimento | Stato |
|---|---|---|
| 22 | Motif-level splice residuo | ✅ |
| 23 | Confronto motif noti | ✅ |
| 24 | Correlazioni dataset-level | ⛔ |

### Blocco 5 — Costo computazionale
| # | Esperimento | Stato |
|---|---|---|
| 25 | GFLOPS analitici k-mer | 🟡 (schema vecchio) |
| 26 | GFLOPS empirici FM (100 reps) | 🟡 (schema vecchio) |
| 27 | Wall-clock k-mer | ⛔ |
| 28 | Wall-clock FM | 🟡 |
| 29 | Truncation stats | ✅ |

### Blocco 6 — Robustness statistica
| # | Esperimento | Stato |
|---|---|---|
| 30 | FDR correction | ✅ |
| 31 | Sensitivity su ε | ⛔ |
| 32 | PCA compression | ✅ |

---

## Overview piani

| Plan | File | Esperimenti | GPU | Priorità |
|------|------|------------|-----|----------|
| Block 1 | `block1_benchmark.md` | 2, 3, 5, 6, 8 | Sì | 🔴 |
| Block 2 | `block2_stratification.md` | 9, 10, 11, 12, 13 | Sì (13) | 🟡 |
| Block 3 | `block3_decomposition.md` | 14–19 | Sì | 🔴 |
| Block 4 | `block4_mechanistic.md` | 24 | No | 🟡 |
| Block 5 | `block5_efficiency.md` | 25, 26, 27, 28 | Sì | 🟡 |
| Block 6 | `block6_robustness.md` | 31 | No | 🟢 |

## Ordine di esecuzione (1 GPU)

```
Block 3 (decomp gap) → Block 1 (pooling + Caduceus + std) → Block 2 (few-shot)
  → Block 5 (efficiency) → Block 4 (correlations, dipende da Block 3)
  → Block 6 (sensitivity ε)
```

## Decisioni di scope

- **CV protocol**: si mantiene GridSearchCV 4-fold + test held-out (già in `src/training/rf_pipeline.py:train_rf`). Per Esp 3 si aggiunge std across **5 random seed** rieseguendo RF.
- **LRA regression**: solo 3 task già presenti (`variant_effect_pathogenic_clinvar`, `bulk_rna_expression`, `cage_prediction`). No eqtl/enhancer/promoter.
- **FM aggiuntivo**: Caduceus-Ph (BiMamba, reverse-complement aware). Richiede nuova classe `src/embedders/caduceus_embedder.py`.
- **Evo2**: sempre escluso da tutte le pipeline.

## Constraint operativi

- Server condiviso → `--n-workers 1` di default per script di training, `2` solo per CPU-only di analisi.
- Tutti gli script di training sono idempotenti: saltano righe già popolate nel CSV di output.
- Cache embedding FM in `cache/` — riutilizzata tra script (mean/max/cls).
