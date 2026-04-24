"""
Linear probe on Genomics Long-Range Benchmark datasets.

Analogue to train_lra_benchmark.py but uses linear models.

Usage:
    python3 src/scripts/train_lra_linear_probe.py --k-values 4 5 6
    python3 src/scripts/train_lra_linear_probe.py --k-values 4 5 6 --model InstaDeepAI/NTv3_650M_pre
    python3 src/scripts/train_lra_linear_probe.py --k-values 4 5 6 --model LongSafari/hyenadna-medium-160k-seqlen-hf
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.core.fcgr import batch_fcgr
from src.features.kmer_features import kmer_from_grids
from src.training.linear_pipeline import (
    eval_linear_classifier,
    eval_linear_regressor,
    train_linear_classifier,
    train_linear_regressor,
)

# Reuse task loaders from the RF benchmark script
from src.scripts.regression.train_lra_benchmark import (
    ALL_TASKS,
    HG38_FA,
    LRA_RECORDS_CSV,
    TASK_META,
    _HG19_TASKS,
    _SUBSET_REQUIRED_TASKS,
    load_task,
)

LRA_LP_RECORDS_CSV = "results/regression/lra_records_linear_probe.csv"


def load_lra_lp_records(path: str = LRA_LP_RECORDS_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def save_lra_lp_records(df: pd.DataFrame, path: str = LRA_LP_RECORDS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def _ensure_row(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset not in df.index:
        new_row = pd.DataFrame([[pd.NA]], index=[dataset], columns=["_init"])
        df = pd.concat([df, new_row]) if not df.empty else new_row
        if "_init" in df.columns:
            df = df.drop(columns=["_init"])
    return df


def write_lra_lp_result(dataset: str, col_prefix: str, metrics: dict,
                         path: str = LRA_LP_RECORDS_CSV) -> pd.DataFrame:
    df = _ensure_row(load_lra_lp_records(path), dataset)
    for m, v in metrics.items():
        df.loc[dataset, f"{col_prefix}_{m}"] = v
    save_lra_lp_records(df, path)
    return df


def has_lra_lp_result(dataset: str, col_prefix: str, is_regression: bool = False,
                       path: str = LRA_LP_RECORDS_CSV) -> bool:
    df = load_lra_lp_records(path)
    if df.empty or dataset not in df.index:
        return False
    sentinel = f"{col_prefix}_R2" if is_regression else f"{col_prefix}_MCC"
    return sentinel in df.columns and pd.notna(df.loc[dataset, sentinel])


def main():
    all_supported = ALL_TASKS + _SUBSET_REQUIRED_TASKS + _HG19_TASKS
    parser = argparse.ArgumentParser(
        description="Linear probe on Genomics Long-Range Benchmark datasets"
    )
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        choices=all_supported)
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--model", default=None,
                        help="FM model (optional). E.g.: InstaDeepAI/NTv3_650M_pre")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--hg38", default=HG38_FA)
    parser.add_argument("--hg19", default=None)
    parser.add_argument("--with-subset-tasks", action="store_true")
    parser.add_argument("--no-subset", action="store_true")
    args = parser.parse_args()

    use_subset = not args.no_subset
    tasks_to_run = list(args.tasks)
    if args.with_subset_tasks:
        for t in _SUBSET_REQUIRED_TASKS:
            if t not in tasks_to_run:
                tasks_to_run.append(t)

    print(f"\nGenomics Long-Range Benchmark — Linear Probe")
    print(f"Tasks   : {tasks_to_run}")
    print(f"K-values: {args.k_values}")
    print(f"FM model: {args.model or 'none'}")
    print(f"Results : {LRA_LP_RECORDS_CSV}\n")

    fm = None
    if args.model:
        model_lc = args.model.lower()
        if "hyenadna" in model_lc:
            from src.embedders.hyena_embedder import HyenaEmbedder
            fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/lra_hyena_embeddings")
        else:
            from src.embedders.fm_embedder import FMEmbedder
            fm = FMEmbedder(model_name=args.model, cache_dir="cache/lra_fm_embeddings")

    from pyfaidx import Fasta
    print(f"Loading hg38 from {args.hg38} ...")
    hg38 = Fasta(args.hg38, one_based_attributes=False, rebuild=False)
    hg19 = None
    if args.hg19:
        print(f"Loading hg19 from {args.hg19} ...")
        hg19 = Fasta(args.hg19, one_based_attributes=False, rebuild=False)
    print()

    for task in tasks_to_run:
        is_regression = TASK_META[task][0]

        print(f"\n{'='*60}")
        print(f"Task: {task}  ({'regression' if is_regression else 'classification'})")
        print(f"{'='*60}")

        train_seqs, train_labels, test_seqs, test_labels = load_task(
            task, hg38, hg19, subset=use_subset
        )
        print(f"  Train: {len(train_seqs)} | Test: {len(test_seqs)}")

        if len(train_seqs) == 0 or len(test_seqs) == 0:
            print(f"  SKIP: no sequences extracted")
            continue

        n_classes = len(set(train_labels)) if not is_regression else None

        # ── K-mer ──
        pending_ks = [k for k in args.k_values
                      if not has_lra_lp_result(task, f"lp_kmer_k{k}", is_regression)]

        if pending_ks:
            max_k = max(pending_ks)
            grid_size = max(128, 2 ** max_k)
            print(f"  Computing FCGR grid={grid_size} ...")
            t0 = time.perf_counter()
            grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
            grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)
            print(f"  FCGR: {time.perf_counter()-t0:.1f}s")

            for k in args.k_values:
                col_prefix = f"lp_kmer_k{k}"
                if has_lra_lp_result(task, col_prefix, is_regression):
                    print(f"  [skip] lp kmer k={k}")
                    continue
                X_train = kmer_from_grids(grids_train, k, normalize="l1")
                X_test = kmer_from_grids(grids_test, k, normalize="l1")
                print(f"  Training linear probe kmer k={k} ...")
                t0 = time.perf_counter()
                if is_regression:
                    model = train_linear_regressor(X_train, train_labels)
                    metrics = eval_linear_regressor(model, X_test, test_labels)
                    print(f"  lp kmer k={k}: R2={metrics['R2']:.4f} Spearman={metrics['Spearman']:.4f} ({time.perf_counter()-t0:.1f}s)")
                else:
                    model = train_linear_classifier(X_train, train_labels, n_classes)
                    metrics = eval_linear_classifier(model, X_test, test_labels, n_classes)
                    print(f"  lp kmer k={k}: MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f} ({time.perf_counter()-t0:.1f}s)")
                write_lra_lp_result(task, col_prefix, metrics)
        else:
            print(f"  [skip] all k-mer already done")

        # ── FM ──
        if fm is not None:
            model_tag = args.model.split("/")[-1]
            col_prefix = f"lp_fm_{model_tag}"
            if has_lra_lp_result(task, col_prefix, is_regression):
                print(f"  [skip] FM {model_tag}")
            else:
                print(f"  Computing FM embeddings ({model_tag}) ...")
                t0 = time.perf_counter()
                Y_train = fm.embed_sequences(train_seqs, task, "train", args.fm_batch_size)
                Y_test = fm.embed_sequences(test_seqs, task, "test", args.fm_batch_size)
                print(f"  FM embedding: {time.perf_counter()-t0:.1f}s")
                print(f"  Training linear probe FM ...")
                t0 = time.perf_counter()
                if is_regression:
                    model = train_linear_regressor(Y_train.astype(np.float32), train_labels)
                    metrics = eval_linear_regressor(model, Y_test.astype(np.float32), test_labels)
                    print(f"  lp FM {model_tag}: R2={metrics['R2']:.4f} Spearman={metrics['Spearman']:.4f} ({time.perf_counter()-t0:.1f}s)")
                else:
                    model = train_linear_classifier(Y_train.astype(np.float32), train_labels, n_classes)
                    metrics = eval_linear_classifier(model, Y_test.astype(np.float32), test_labels, n_classes)
                    print(f"  lp FM {model_tag}: MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f} ({time.perf_counter()-t0:.1f}s)")
                write_lra_lp_result(task, col_prefix, metrics)

    print(f"\nDone! Results in {LRA_LP_RECORDS_CSV}")


if __name__ == "__main__":
    main()
