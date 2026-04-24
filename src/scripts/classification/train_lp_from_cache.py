"""
Linear probe using pre-computed cached embeddings (no model loading needed).

Directly loads .npz embedding files from cache, skipping FM model initialization.
Useful when all embeddings are already cached.

Usage:
    python3 src/scripts/train_lp_from_cache.py \
        --model LongSafari/hyenadna-medium-160k-seqlen-hf \
        --cache-dir cache/hyena_embeddings \
        --pooling mean

    python3 src/scripts/train_lp_from_cache.py \
        --model zhihan1996/DNABERT-2-117M \
        --cache-dir cache/fm_embeddings \
        --pooling mean
"""

import argparse
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.embedding_cache import load_cached_embeddings
from src.records.records import (
    RECORDS_LINEAR_CSV,
    has_lp_fm_pool,
    load_lp_records,
    write_lp_fm_pool,
)
from src.training.linear_pipeline import eval_linear_classifier, train_linear_classifier


def _process_one(ds, model_name, cache_dir, pooling, n_jobs_lr):
    name = ds["name"]
    records = load_lp_records(RECORDS_LINEAR_CSV)
    if has_lp_fm_pool(name, model_name, records, pooling=pooling):
        return name, None, 0.0, True  # skip

    # Load embeddings from cache
    Y_train = load_cached_embeddings(model_name, name, "train", pooling=pooling)
    Y_test = load_cached_embeddings(model_name, name, "test", pooling=pooling)

    if Y_train is None or Y_test is None:
        return name, None, 0.0, False  # cache miss

    _, train_labels, _, test_labels = load_dataset(ds["train_path"], ds["test_path"])
    n_classes = len(set(train_labels))

    t0 = time.perf_counter()
    clf = train_linear_classifier(Y_train, train_labels, n_classes, n_jobs=n_jobs_lr)
    metrics = eval_linear_classifier(clf, Y_test, test_labels, n_classes)
    elapsed = time.perf_counter() - t0
    write_lp_fm_pool(name, model_name, metrics, pooling=pooling)
    return name, metrics, elapsed, False


def main():
    parser = argparse.ArgumentParser(
        description="LP from cached embeddings (no model loading)"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True, help="FM model HuggingFace ID")
    parser.add_argument("--cache-dir", default="cache/fm_embeddings")
    parser.add_argument("--pooling", default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--n-parallel", type=int, default=8,
                        help="Parallel dataset workers")
    parser.add_argument("--n-jobs-lr", type=int, default=4,
                        help="Threads for LogisticRegressionCV per dataset")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_datasets = len(all_datasets)
    model_tag = args.model.split("/")[-1]

    print(f"\nLinear Probe from cache — {model_tag}")
    print(f"Pooling    : {args.pooling}")
    print(f"Datasets   : {n_datasets}")
    print(f"n-parallel : {args.n_parallel}")
    print(f"Results    : {RECORDS_LINEAR_CSV}\n")

    records = load_lp_records(RECORDS_LINEAR_CSV)
    pending = [
        ds for ds in all_datasets
        if not has_lp_fm_pool(ds["name"], args.model, records, pooling=args.pooling)
    ]
    skipped = n_datasets - len(pending)
    print(f"Pending: {len(pending)} | Skipped: {skipped}")

    if not pending:
        print("All done.")
        return

    cache_miss = []

    def _run(ds):
        name, metrics, elapsed, was_skip = _process_one(
            ds, args.model, args.cache_dir, args.pooling, args.n_jobs_lr
        )
        if was_skip:
            print(f"  [skip] {name}", flush=True)
        elif metrics is None:
            print(f"  [MISS] {name} — no cached embeddings", flush=True)
        else:
            print(
                f"  [done] {name} MCC={metrics['MCC']:.3f} "
                f"AUROC={metrics['AUROC']:.3f} ({elapsed:.0f}s)",
                flush=True
            )
        return name, metrics

    results = Parallel(n_jobs=args.n_parallel, backend="loky")(
        delayed(_run)(ds) for ds in pending
    )

    miss = [n for n, m in results if m is None]
    if miss:
        print(f"\nWARNING: {len(miss)} cache misses: {miss[:5]}")

    # Final count
    records = load_lp_records(RECORDS_LINEAR_CSV)
    col_prefix = f"lp_fm_{model_tag}"
    # Check any matching column
    done_cols = [c for c in records.columns if c.startswith(col_prefix) and c.endswith("_MCC")]
    if done_cols:
        total_done = records[done_cols[0]].notna().sum()
        print(f"\nDone! {total_done}/{n_datasets} — Results in {RECORDS_LINEAR_CSV}")
    else:
        print(f"\nDone! Results in {RECORDS_LINEAR_CSV}")


if __name__ == "__main__":
    main()
