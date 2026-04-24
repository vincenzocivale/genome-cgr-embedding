"""
Linear probe (LogisticRegressionCV) on all 57 classification datasets.

Parallel execution: datasets processed in parallel using joblib.
Results are written atomically after each dataset.

Usage:
    python3 src/scripts/train_linear_probe.py --mode kmer --k-values 4 5 6
    python3 src/scripts/train_linear_probe.py --mode kmer --k-values 4 5 6 --n-parallel 32
    python3 src/scripts/train_linear_probe.py --mode fm --model InstaDeepAI/NTv3_650M_pre
    python3 src/scripts/train_linear_probe.py --mode fm --model LongSafari/hyenadna-medium-160k-seqlen-hf
    python3 src/scripts/train_linear_probe.py --mode fm --model zhihan1996/DNABERT-2-117M
"""

import argparse
import fcntl
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.core.fcgr import batch_fcgr
from src.data.loader import discover_datasets, load_dataset
from src.features.kmer_features import kmer_from_grids
from src.records.records import (
    RECORDS_LINEAR_CSV,
    has_lp_fm_pool,
    has_lp_kmer,
    load_lp_records,
    write_lp_fm_pool,
    write_lp_kmer,
)
from src.training.linear_pipeline import (
    eval_linear_classifier,
    train_linear_classifier,
)


def _process_kmer_dataset(ds, k_values, n_workers, n_jobs_lr):
    """Process one dataset (all k-values). Returns list of (name, k, metrics)."""
    name = ds["name"]
    records = load_lp_records(RECORDS_LINEAR_CSV)
    pending_ks = [k for k in k_values if not has_lp_kmer(name, k, records)]
    if not pending_ks:
        return []

    train_seqs, train_labels, test_seqs, test_labels = load_dataset(
        ds["train_path"], ds["test_path"]
    )
    n_classes = len(set(train_labels))

    max_k = max(pending_ks)
    grid_size = max(128, 2 ** max_k)
    grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=n_workers)
    grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=n_workers)

    results = []
    for k in k_values:
        if has_lp_kmer(name, k, records):
            continue
        X_train = kmer_from_grids(grids_train, k, normalize="l1")
        X_test = kmer_from_grids(grids_test, k, normalize="l1")
        t0 = time.perf_counter()
        clf = train_linear_classifier(X_train, train_labels, n_classes, n_jobs=n_jobs_lr)
        metrics = eval_linear_classifier(clf, X_test, test_labels, n_classes)
        elapsed = time.perf_counter() - t0
        results.append((name, k, metrics, elapsed))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe on k-mer or FM features for all classification datasets"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--mode", required=True, choices=["kmer", "fm"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--model", default=None,
                        help="FM model HuggingFace ID (only for --mode fm)")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--pooling", choices=["mean", "max", "cls"], default="mean")
    parser.add_argument("--n-workers", type=int, default=2,
                        help="Workers for FCGR computation per dataset")
    parser.add_argument("--n-parallel", type=int, default=2,
                        help="Parallel datasets (kmer mode only). Each uses n-jobs-lr threads.")
    parser.add_argument("--n-jobs-lr", type=int, default=2,
                        help="Threads for LogisticRegressionCV per dataset")
    args = parser.parse_args()

    if args.mode == "fm" and args.model is None:
        parser.error("--model is required when --mode fm")
    if args.mode == "fm" and args.model.lower().startswith("evo2"):
        parser.error("Evo2 is explicitly excluded in this experiment plan")
    if args.mode == "fm" and args.pooling == "cls" and "hyenadna" in args.model.lower():
        parser.error("CLS pooling is not supported for HyenaDNA")

    all_datasets = discover_datasets(args.data_root)
    n_datasets = len(all_datasets)

    if args.mode == "kmer":
        print(f"\nLinear Probe — k-mer features (parallel mode)")
        print(f"Datasets   : {n_datasets}")
        print(f"K-values   : {args.k_values}")
        print(f"n-parallel : {args.n_parallel}")
        print(f"n-jobs-lr  : {args.n_jobs_lr}")
        print(f"Results    : {RECORDS_LINEAR_CSV}\n")

        # Filter out already-completed datasets
        records = load_lp_records(RECORDS_LINEAR_CSV)
        pending = [
            ds for ds in all_datasets
            if any(not has_lp_kmer(ds["name"], k, records) for k in args.k_values)
        ]
        skipped = n_datasets - len(pending)
        print(f"Pending: {len(pending)} | Skipped (done): {skipped}\n")

        if not pending:
            print("All datasets already processed.")
            return

        completed = [0]
        total = len(pending)

        def _run_and_save(ds):
            results = _process_kmer_dataset(
                ds, args.k_values, args.n_workers, args.n_jobs_lr
            )
            for name, k, metrics, elapsed in results:
                write_lp_kmer(name, k, metrics)
                completed[0] += 1
                mcc_str = f"{metrics['MCC']:.3f}"
                print(
                    f"  [{completed[0]:3d}/{total}] {name} k={k} "
                    f"MCC={mcc_str} AUROC={metrics['AUROC']:.3f} ({elapsed:.0f}s)",
                    flush=True
                )
            return ds["name"]

        Parallel(n_jobs=args.n_parallel, backend="loky")(
            delayed(_run_and_save)(ds) for ds in pending
        )

    else:  # fm mode
        print(f"\nLinear Probe — FM embeddings: {args.model.split('/')[-1]}")
        print(f"Pooling  : {args.pooling}")
        print(f"Datasets : {n_datasets}")
        print(f"Results  : {RECORDS_LINEAR_CSV}\n")

        model_lc = args.model.lower()
        if "hyenadna" in model_lc:
            from src.embedders.hyena_embedder import HyenaEmbedder
            fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings", pooling=args.pooling)
        else:
            from src.embedders.fm_embedder import FMEmbedder
            fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings", pooling=args.pooling)

        records = load_lp_records(RECORDS_LINEAR_CSV)
        completed = 0
        for ds in all_datasets:
            name = ds["name"]
            completed += 1
            if has_lp_fm_pool(name, args.model, records, pooling=args.pooling):
                print(f"  [{completed:3d}/{n_datasets}] skip {name}", flush=True)
                continue

            train_seqs, train_labels, test_seqs, test_labels = load_dataset(
                ds["train_path"], ds["test_path"]
            )
            n_classes = len(set(train_labels))

            Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
            Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)

            t0 = time.perf_counter()
            clf = train_linear_classifier(Y_train, train_labels, n_classes, n_jobs=8)
            metrics = eval_linear_classifier(clf, Y_test, test_labels, n_classes)
            elapsed = time.perf_counter() - t0
            write_lp_fm_pool(name, args.model, metrics, pooling=args.pooling)
            records = load_lp_records(RECORDS_LINEAR_CSV)

            print(
                f"  [{completed:3d}/{n_datasets}] {name} "
                f"MCC={metrics['MCC']:.3f} AUROC={metrics['AUROC']:.3f} ({elapsed:.0f}s)",
                flush=True
            )

    print(f"\nDone! Results in {RECORDS_LINEAR_CSV}")


if __name__ == "__main__":
    main()
