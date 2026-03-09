#!/usr/bin/env python3
"""
Benchmark FCGR k-mer + Random Forest sui dataset del DNA Foundation Benchmark.

Classificatore: Random Forest con GridSearchCV a 4-fold sul training set,
identico alla strategia usata nel paper (Feng et al., Nat Commun 2025).

Uso:
    python benchmark_dnafoundation.py
    python benchmark_dnafoundation.py --k-values 4 6
    python benchmark_dnafoundation.py --tasks enhancers/enhancer --k-values 4 6
    python benchmark_dnafoundation.py --list-tasks
"""

import argparse
import os

from pipeline.dnafoundation_loader import discover_datasets, load_dataset_csv
from pipeline.feature_extraction import extract_kmer_features
from pipeline.model import train_and_evaluate
from pipeline.results import save_result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FCGR k-mer + RF su DNA Foundation Benchmark datasets",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[4, 6],
        help="Valori di k per le frequenze k-mer (default: 4 6)",
    )
    parser.add_argument("--grid-size", type=int, default=128)
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
        help="Dataset specifici da eseguire (es. enhancers/enhancer). Default: tutti.",
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
    print(f"k-values: {args.k_values}")
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
        n_classes = len(set(train_labels))
        print(f"  Train: {len(train_seqs)}, Test: {len(test_seqs)}, Classi: {n_classes}")

        for k in args.k_values:
            if args.grid_size < 2 ** k:
                print(f"  [SKIP] k={k} richiede grid_size >= {2 ** k}")
                continue

            print(f"\n  --- kmer k={k} (dim={4 ** k}) ---")

            X_train = extract_kmer_features(
                train_seqs, k, args.grid_size, n_workers=args.n_workers,
            )
            X_test = extract_kmer_features(
                test_seqs, k, args.grid_size, n_workers=args.n_workers,
            )

            metrics = train_and_evaluate(
                X_train, train_labels, X_test, test_labels,
                n_jobs=args.n_jobs,
            )

            result = {
                "task": task_name,
                "method": "kmer_rf",
                "k": k,
                "grid_size": args.grid_size,
                "feature_dim": 4 ** k,
                **metrics,
            }
            save_result(args.output, result)
            auroc_str = f"  AUROC: {metrics['auroc']:.4f}" if "auroc" in metrics else ""
            print(f"  MCC: {metrics['mcc']:.4f}  F1: {metrics['f1_macro']:.4f}  Acc: {metrics['accuracy']:.4f}{auroc_str}")

    print(f"\nRisultati salvati in: {args.output}")


if __name__ == "__main__":
    main()
