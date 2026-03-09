#!/usr/bin/env python3
"""
Benchmark FCGR + Random Forest sui dataset del DNA Foundation Benchmark.

Metodi disponibili:
  kmer     — k-mer a risoluzione fissa dalla matrice FCGR         (dim = 4^k)
  adaptive — QuadTree adattivo: suddivide dove la distribuzione   (dim = 4^qt_depth)
             dei conteggi tra quadranti è significativamente
             non-uniforme (test chi-quadrato)
  kmer_qt  — concatenazione [k-mer | adaptive QuadTree]           (dim = 4^k + 4^qt_depth)
  all      — esegue tutti e tre

Classificatore: Random Forest con GridSearchCV a 4-fold sul training set,
identico alla strategia del paper (Feng et al., Nat Commun 2025).

Uso:
    python benchmark_dnafoundation.py --method all
    python benchmark_dnafoundation.py --method adaptive --qt-split-threshold 0.05
    python benchmark_dnafoundation.py --tasks enhancers/enhancer --k-values 4 6
    python benchmark_dnafoundation.py --list-tasks
"""

import argparse
import os

from pipeline.dnafoundation_loader import discover_datasets, load_dataset_csv
from pipeline.feature_extraction import extract_kmer_features
from pipeline.quadtree import extract_adaptive_features
from pipeline.combined_features import extract_combined_features
from pipeline.model import train_and_evaluate
from pipeline.results import save_result


def run_task(task_name, train_seqs, train_labels, test_seqs, test_labels, args):
    n_classes = len(set(train_labels))
    print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}, Classi: {n_classes}")

    methods = []
    if args.method in ("kmer", "all"):
        methods.append("kmer")
    if args.method in ("adaptive", "all"):
        methods.append("adaptive")
    if args.method in ("kmer_qt", "all"):
        methods.append("kmer_qt")

    for k in args.k_values:
        if args.grid_size < 2 ** k:
            print(f"  [SKIP] k={k} richiede grid_size >= {2 ** k}")
            continue

        for method in methods:
            if method == "kmer":
                dim = 4 ** k
                print(f"\n  --- kmer k={k} (dim={dim}) ---")
                X_train = extract_kmer_features(
                    train_seqs, k, args.grid_size, n_workers=args.n_workers,
                )
                X_test = extract_kmer_features(
                    test_seqs, k, args.grid_size, n_workers=args.n_workers,
                )

            elif method == "adaptive":
                dim = 4 ** args.qt_max_depth
                print(f"\n  --- adaptive depth={args.qt_max_depth} p<{args.qt_split_threshold} (dim={dim}) ---")
                X_train = extract_adaptive_features(
                    train_seqs, args.grid_size,
                    max_depth=args.qt_max_depth,
                    split_threshold=args.qt_split_threshold,
                    n_workers=args.n_workers,
                )
                X_test = extract_adaptive_features(
                    test_seqs, args.grid_size,
                    max_depth=args.qt_max_depth,
                    split_threshold=args.qt_split_threshold,
                    n_workers=args.n_workers,
                )

            else:  # kmer_qt
                dim = 4 ** k + 4 ** args.qt_max_depth
                print(f"\n  --- kmer_qt k={k} + depth={args.qt_max_depth} p<{args.qt_split_threshold} (dim={dim}) ---")
                X_train = extract_combined_features(
                    train_seqs, k, args.grid_size,
                    qt_max_depth=args.qt_max_depth,
                    qt_split_threshold=args.qt_split_threshold,
                    n_workers=args.n_workers,
                )
                X_test = extract_combined_features(
                    test_seqs, k, args.grid_size,
                    qt_max_depth=args.qt_max_depth,
                    qt_split_threshold=args.qt_split_threshold,
                    n_workers=args.n_workers,
                )

            metrics = train_and_evaluate(
                X_train, train_labels, X_test, test_labels,
                n_jobs=args.n_jobs,
            )

            k_label = k if method != "adaptive" else f"d{args.qt_max_depth}_p{args.qt_split_threshold}"
            result = {
                "task": task_name,
                "method": method,
                "k": k_label,
                "grid_size": args.grid_size,
                "feature_dim": dim,
                **metrics,
            }
            save_result(args.output, result)
            auroc_str = f"  AUROC: {metrics['auroc']:.4f}" if "auroc" in metrics else ""
            print(
                f"  MCC: {metrics['mcc']:.4f}"
                f"  F1: {metrics['f1_macro']:.4f}"
                f"  Acc: {metrics['accuracy']:.4f}"
                f"{auroc_str}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FCGR + RF su DNA Foundation Benchmark datasets",
    )
    parser.add_argument(
        "--method", choices=["kmer", "adaptive", "kmer_qt", "all"], default="all",
        help="Metodo di feature extraction (default: all)",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[4, 6],
        help="Valori di k per kmer e kmer_qt (default: 4 6)",
    )
    parser.add_argument("--grid-size", type=int, default=128)
    # QuadTree params
    parser.add_argument(
        "--qt-max-depth", type=int, default=6,
        help="Profondità massima QuadTree adattivo (default: 6 → 4096 feature)",
    )
    parser.add_argument(
        "--qt-split-threshold", type=float, default=0.05,
        help="P-value chi-quadrato per suddivisione (default: 0.05). "
             "Suddivide se la distribuzione tra i 4 figli è significativamente "
             "non-uniforme (p < soglia). 1.0 = suddivide sempre. 0.0 = mai.",
    )
    # Parallelism
    parser.add_argument(
        "--n-workers", type=int, default=os.cpu_count(),
        help="Worker per estrazione FCGR parallela (default: tutti i core)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Parallelismo Random Forest / GridSearchCV (default: -1 = tutti i core)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_dnafoundation_results.csv",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Dataset specifici (es. enhancers/enhancer). Default: tutti.",
    )
    parser.add_argument(
        "--data-root", type=str, default="data/dna_foundation_benchmark",
        help="Path alla directory con i dataset CSV",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="Elenca i dataset disponibili ed esce",
    )
    args = parser.parse_args()

    datasets = discover_datasets(args.data_root)

    if not datasets:
        print("Nessun dataset trovato in:", args.data_root)
        return

    if args.list_tasks:
        print(f"Dataset disponibili ({len(datasets)}):")
        for ds in datasets:
            print(f"  {ds['name']}")
        return

    if args.tasks:
        task_set = set(args.tasks)
        datasets = [d for d in datasets if d["name"] in task_set]
        if not datasets:
            print(f"Nessun dataset corrisponde a: {args.tasks}")
            return

    print(f"Dataset: {len(datasets)}")
    print(f"Metodo: {args.method}  |  k-values: {args.k_values}")
    if args.method in ("adaptive", "kmer_qt", "all"):
        print(f"QuadTree adattivo: max_depth={args.qt_max_depth}, split p<{args.qt_split_threshold}")
    print(f"Classificatore: Random Forest + GridSearchCV (4-fold)")
    print()

    for ds_info in datasets:
        task_name = ds_info["name"]
        print(f"{'=' * 60}")
        print(f"TASK: {task_name}")
        print(f"{'=' * 60}")

        train_seqs, train_labels, test_seqs, test_labels = load_dataset_csv(
            ds_info["train_path"], ds_info["test_path"],
        )
        run_task(task_name, train_seqs, train_labels, test_seqs, test_labels, args)

    print(f"\nRisultati salvati in: {args.output}")


if __name__ == "__main__":
    main()
