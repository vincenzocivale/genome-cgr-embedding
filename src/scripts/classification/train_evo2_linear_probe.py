"""
Linear probe for Evo2 embeddings, reading directly from npz cache.

Usage:
    conda run -n cgr_bench python3 src/scripts/classification/train_evo2_linear_probe.py \
        --data-root /data/genomic_bench/dna_foundation_benchmark/
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.records.records import (
    RECORDS_LINEAR_CSV,
    has_lp_fm_pool,
    load_lp_records,
    write_lp_fm_pool,
)
from src.training.linear_pipeline import eval_linear_classifier, train_linear_classifier

MODEL_ID = "evo2_1b_base"
CACHE_DIR = "cache/fm_embeddings/evo2_1b_base"


def _dataset_key(name: str) -> str:
    return name.replace("/", "__")


def _load_cached(name: str, split: str) -> np.ndarray:
    path = os.path.join(CACHE_DIR, _dataset_key(name), f"{split}.npz")
    data = np.load(path)
    return data["embeddings"].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    args = parser.parse_args()

    datasets = discover_datasets(args.data_root)
    records = load_lp_records(RECORDS_LINEAR_CSV)

    print(f"Linear probe — Evo2 (from cache: {CACHE_DIR})")
    print(f"Datasets: {len(datasets)}")
    print(f"Results : {RECORDS_LINEAR_CSV}\n")

    for i, ds in enumerate(datasets, 1):
        name = ds["name"]
        if has_lp_fm_pool(name, MODEL_ID, records, pooling="mean"):
            print(f"  [{i:3d}/{len(datasets)}] skip {name}")
            continue

        cache_key = _dataset_key(name)
        train_npz = os.path.join(CACHE_DIR, cache_key, "train.npz")
        test_npz = os.path.join(CACHE_DIR, cache_key, "test.npz")
        if not os.path.exists(train_npz) or not os.path.exists(test_npz):
            print(f"  [{i:3d}/{len(datasets)}] MISSING cache for {name} — skipping")
            continue

        _, train_labels, _, test_labels = load_dataset(ds["train_path"], ds["test_path"])
        n_classes = len(set(train_labels))

        try:
            X_train = _load_cached(name, "train")
            X_test = _load_cached(name, "test")
        except Exception as e:
            print(f"  [{i:3d}/{len(datasets)}] ERROR loading cache for {name}: {e} — skipping")
            continue

        t0 = time.perf_counter()
        clf = train_linear_classifier(X_train, train_labels, n_classes, n_jobs=8)
        metrics = eval_linear_classifier(clf, X_test, test_labels, n_classes)
        elapsed = time.perf_counter() - t0

        write_lp_fm_pool(name, MODEL_ID, metrics, pooling="mean")
        records = load_lp_records(RECORDS_LINEAR_CSV)

        print(
            f"  [{i:3d}/{len(datasets)}] {name} "
            f"MCC={metrics['MCC']:.3f} AUROC={metrics['AUROC']:.3f} ({elapsed:.0f}s)",
            flush=True,
        )

    print(f"\nDone! Results in {RECORDS_LINEAR_CSV}")


if __name__ == "__main__":
    main()
