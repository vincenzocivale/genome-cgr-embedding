"""
Train RF benchmarks across multiple random seeds and save mean/std metrics.

Usage:
    python3 src/scripts/classification/train_rf_with_seeds.py --mode kmer --k-values 4 5 6
    python3 src/scripts/classification/train_rf_with_seeds.py --mode fm --model InstaDeepAI/NTv3_650M_pre
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.core.fcgr import batch_fcgr
from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder
from src.embedders.hyena_embedder import HyenaEmbedder
from src.features.kmer_features import kmer_from_grids
from src.records.records import (
    RECORDS_CSV,
    load_records,
    kmer_col,
    fm_col,
    write_fm_rf_std,
    write_kmer_rf_std,
)
from src.training.rf_pipeline import eval_rf, train_rf

SEEDS = [42, 0, 1, 2, 3]
RF_METRICS = ["MCC", "AUROC", "F1", "Accuracy"]


def _has_std_entry(dataset: str, base_col: str, records) -> bool:
    std_col = f"{base_col}_std"
    return (
        (not records.empty)
        and (dataset in records.index)
        and (std_col in records.columns)
        and np.isfinite(records.loc[dataset, std_col])
    )


def _aggregate(metrics_list: list[dict]) -> dict:
    out = {}
    for metric in RF_METRICS:
        vals = np.array([m[metric] for m in metrics_list], dtype=np.float64)
        out[f"{metric}_mean"] = float(np.mean(vals))
        out[f"{metric}_std"] = float(np.std(vals, ddof=0))
    return out


def _fit_eval_across_seeds(X_train, y_train, X_test, y_test, n_classes, n_jobs_rf):
    all_metrics = []
    for seed in SEEDS:
        model = train_rf(
            X_train,
            y_train,
            n_classes,
            random_state=seed,
            n_jobs=n_jobs_rf,
        )
        all_metrics.append(eval_rf(model, X_test, y_test, n_classes))
    return _aggregate(all_metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Train RF across multiple seeds and save mean/std metrics"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--mode", required=True, choices=["kmer", "fm"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--model", default=None,
                        help="FM model HuggingFace ID (required for --mode fm)")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Workers for FCGR generation (k-mer mode)")
    parser.add_argument("--n-jobs-rf", type=int, default=1,
                        help="Threads for RF estimator in each seed run")
    args = parser.parse_args()

    if args.mode == "fm" and args.model is None:
        parser.error("--model is required when --mode fm")
    if args.mode == "fm" and args.model.lower().startswith("evo2"):
        parser.error("Evo2 is explicitly excluded in this experiment plan")

    all_datasets = discover_datasets(args.data_root)
    records = load_records(RECORDS_CSV)

    print(f"\nRF with seeds")
    print(f"Mode      : {args.mode}")
    print(f"Seeds     : {SEEDS}")
    print(f"Datasets  : {len(all_datasets)}")
    print(f"Results   : {RECORDS_CSV}\n")

    if args.mode == "fm":
        model_lc = args.model.lower()
        if "hyenadna" in model_lc:
            fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings")
        else:
            fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")
    else:
        fm = None

    pbar = tqdm(all_datasets, ncols=80, unit="ds")
    for ds in pbar:
        name = ds["name"]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        if args.mode == "kmer":
            pending = [k for k in args.k_values if not _has_std_entry(name, kmer_col(k, "MCC"), records)]
            if not pending:
                pbar.set_description(f"skip {name}")
                continue

            max_k = max(pending)
            grid_size = max(128, 2 ** max_k)
            grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
            grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)

            for k in args.k_values:
                if _has_std_entry(name, kmer_col(k, "MCC"), records):
                    continue
                X_train = kmer_from_grids(grids_train, k, normalize="l1")
                X_test = kmer_from_grids(grids_test, k, normalize="l1")
                agg = _fit_eval_across_seeds(
                    X_train, train_labels, X_test, test_labels, n_classes, args.n_jobs_rf
                )
                write_kmer_rf_std(name, k, agg)
                records = load_records(RECORDS_CSV)
                pbar.set_description(f"{name} k={k}")
        else:
            base = fm_col(args.model, "MCC")
            if _has_std_entry(name, base, records):
                pbar.set_description(f"skip {name}")
                continue
            Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
            Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)
            agg = _fit_eval_across_seeds(
                Y_train, train_labels, Y_test, test_labels, n_classes, args.n_jobs_rf
            )
            write_fm_rf_std(name, args.model, agg)
            records = load_records(RECORDS_CSV)
            pbar.set_description(f"{name} fm")

    print("\nDone.")


if __name__ == "__main__":
    main()
