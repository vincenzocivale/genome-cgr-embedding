"""
PCA dimensionality reduction + Random Forest on k-mer and FM features.

PCA is fit on train only (no data leakage), then applied to test.
Number of components is chosen to retain 95% of the explained variance.

Usage:
    python3 src/scripts/classification/train_pca_rf.py --mode kmer --k-values 4 5 6
    python3 src/scripts/classification/train_pca_rf.py --mode fm --model InstaDeepAI/NTv3_650M_pre
    python3 src/scripts/classification/train_pca_rf.py --mode fm --model LongSafari/hyenadna-medium-160k-seqlen-hf
"""

import argparse
import sys
import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.features.kmer_features import kmer_from_grids
from src.core.fcgr import batch_fcgr
from src.embedders.fm_embedder import FMEmbedder, validate_supported_model
from src.embedders.hyena_embedder import HyenaEmbedder
from src.records.records import (
    load_pca_records, RECORDS_PCA_CSV,
    has_pca_kmer, write_pca_kmer,
    has_pca_fm, write_pca_fm,
)
from src.training.efficiency import log_efficiency
from src.training.rf_pipeline import train_rf, eval_rf

VARIANCE_THRESHOLD = 0.95


def fit_pca(X_train: np.ndarray, X_test: np.ndarray):
    """Standardize + PCA on train, transform both splits. Returns reduced arrays.

    StandardScaler is required because FM embeddings have highly non-uniform
    per-feature variance: without rescaling, the first PC captures >99% of the
    raw variance regardless of dtype precision.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    pca = PCA(n_components=VARIANCE_THRESHOLD, random_state=42)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)
    return X_train_pca.astype(np.float32), X_test_pca.astype(np.float32), pca.n_components_


# ── kmer mode ──────────────────────────────────────────────────────────────────

def run_kmer(args):
    all_datasets = discover_datasets(args.data_root)
    n_datasets = len(all_datasets)
    n_total = n_datasets * len(args.k_values)

    print(f"\nPCA + RF on k-mer features")
    print(f"Datasets : {n_datasets}")
    print(f"K-values : {args.k_values}")
    print(f"Variance : {VARIANCE_THRESHOLD}")
    print(f"Results  : {RECORDS_PCA_CSV}\n")

    records = load_pca_records()
    pbar = tqdm(total=n_total, desc="Progress", unit="task", ncols=70)

    for ds in all_datasets:
        name = ds["name"]
        pending_ks = [k for k in args.k_values if not has_pca_kmer(name, k, records)]

        if not pending_ks:
            pbar.update(len(args.k_values))
            pbar.set_description(f"skip {name}")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        max_k = max(pending_ks)
        grid_size = max(128, 2 ** max_k)
        pbar.set_description(f"[{name}] FCGR grid={grid_size}")
        grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
        grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)

        for k in args.k_values:
            if has_pca_kmer(name, k, records):
                pbar.update(1)
                continue

            t0 = time.perf_counter()
            X_train = kmer_from_grids(grids_train, k, normalize="l1")
            X_test = kmer_from_grids(grids_test, k, normalize="l1")
            X_train_pca, X_test_pca, n_comp = fit_pca(X_train, X_test)
            t1 = time.perf_counter()

            log_efficiency(name, f"pca_kmer_k{k}", n_comp, t1 - t0)

            rf = train_rf(X_train_pca, train_labels, n_classes)
            metrics = eval_rf(rf, X_test_pca, test_labels, n_classes)
            write_pca_kmer(name, k, metrics)
            records = load_pca_records()

            pbar.update(1)
            pbar.set_description(f"[{name} k={k}] PCA={n_comp}d MCC={metrics['MCC']:.3f}")

    pbar.close()
    print(f"\nDone!")


# ── fm mode ────────────────────────────────────────────────────────────────────

def run_fm(args):
    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets)

    print(f"\nPCA + RF on FM embeddings")
    print(f"Model    : {args.model.split('/')[-1]}")
    print(f"Datasets : {n_total}")
    print(f"Variance : {VARIANCE_THRESHOLD}")
    print(f"Results  : {RECORDS_PCA_CSV}\n")

    validate_supported_model(args.model)
    model_lc = args.model.lower()
    if "hyenadna" in model_lc:
        fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings")
    elif model_lc.startswith("google/enformer"):
        from src.embedders.enformer_embedder import EnformerEmbedder
        fm = EnformerEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")
    else:
        fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")

    records = load_pca_records()
    pbar = tqdm(all_datasets, desc="Progress", unit="ds", ncols=70)

    for ds in pbar:
        name = ds["name"]

        if has_pca_fm(name, args.model, records):
            pbar.set_description(f"skip {name}")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        t0 = time.perf_counter()
        Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
        Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)
        Y_train_pca, Y_test_pca, n_comp = fit_pca(Y_train, Y_test)
        t1 = time.perf_counter()

        log_efficiency(name, f"pca_fm_{args.model.split('/')[-1]}", n_comp, t1 - t0)

        rf = train_rf(Y_train_pca, train_labels, n_classes)
        metrics = eval_rf(rf, Y_test_pca, test_labels, n_classes)
        write_pca_fm(name, args.model, metrics)
        records = load_pca_records()

        pbar.set_description(f"[{name}] PCA={n_comp}d MCC={metrics['MCC']:.3f}")

    pbar.close()
    print(f"\nDone!")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCA + RF on k-mer or FM features"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--mode", choices=["kmer", "fm"], required=True,
                       help="Feature type: kmer or fm")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6],
                       help="K-mer sizes (kmer mode only)")
    parser.add_argument("--model", type=str, default=None,
                       help="FM model (fm mode only)")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "kmer":
        run_kmer(args)
    elif args.mode == "fm":
        if args.model is None:
            parser.error("--model is required for fm mode")
        run_fm(args)


if __name__ == "__main__":
    main()
