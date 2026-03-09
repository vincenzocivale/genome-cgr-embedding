#!/usr/bin/env python3
"""
Benchmark FCGR k-mer MLP sui dataset del DNA Foundation Benchmark.

Uso:
    python benchmark_dnafoundation.py
    python benchmark_dnafoundation.py --k-values 4 6
    python benchmark_dnafoundation.py --tasks enhancers/enhancer --k-values 4 6
    python benchmark_dnafoundation.py --list-tasks
"""

import argparse

from pipeline.dnafoundation_loader import discover_datasets, load_dataset_csv
from pipeline.feature_extraction import extract_kmer_features
from pipeline.model import train_and_evaluate
from pipeline.results import save_result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FCGR k-mer su DNA Foundation Benchmark datasets",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[4, 6],
        help="Valori di k per le frequenze k-mer (default: 4 6)",
    )
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--output", type=str, default="benchmark_dnafoundation_results.csv",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Dataset specifici da eseguire (es. enhancers/enhancer). Default: tutti.",
    )
    parser.add_argument(
        "--data-root", type=str, default="data/dna_foundation_benchmark",
        help="Override del percorso data_processed",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="Elenca i dataset disponibili ed esce",
    )
    args = parser.parse_args()

    datasets = discover_datasets(args.data_root)

    if not datasets:
        print("Nessun dataset trovato. Esegui prima: python download_dnafoundation_data.py")
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
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            )

            result = {
                "task": task_name,
                "method": "kmer",
                "k": k,
                "grid_size": args.grid_size,
                "feature_dim": 4 ** k,
                **metrics,
            }
            save_result(args.output, result)
            print(f"  F1 macro: {metrics['f1_macro']:.4f}  Accuracy: {metrics['accuracy']:.4f}")

    print(f"\nRisultati salvati in: {args.output}")


if __name__ == "__main__":
    main()
