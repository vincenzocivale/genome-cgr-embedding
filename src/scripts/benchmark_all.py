"""
Benchmark script: misura le performance RF (k-mer e FM) su tutti i dataset.

Salva progressivamente i risultati in records.csv.
Salta i dataset già completati per quella configurazione (k, FM_model).

Usage:
    python3 src/fm_experiment/benchmark_all.py --k-values 6 --model InstaDeepAI/NTv3_650M_pre
    python3 src/fm_experiment/benchmark_all.py --k-values 4 6 8 --fm-batch-size 32
"""

import argparse
import sys
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder
from src.embedders.hyena_embedder import HyenaEmbedder
from src.embedders.evo2_embedder import Evo2Embedder
from src.features.kmer_features import extract_kmer_features
from src.records.records import (
    load_records, has_kmer, has_fm, has_ridge,
    write_kmer, write_fm, write_ridge,
    RECORDS_CSV,
)
from src.training.ridge_mapping import fit_and_evaluate
from src.training.rf_pipeline import train_rf, eval_rf


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RF su tutti i dataset — k-mer vs FM embeddings"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[6])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--model", default="InstaDeepAI/NTv3_650M_pre",
                       help="FM model: NTv3_650M_pre, hyenadna-medium-160k, etc.")
    parser.add_argument("--evo2-layer", type=str, default=None,
                       help="Evo2 layer name for embeddings (optional).")
    parser.add_argument("--evo2-max-length", type=int, default=None,
                       help="Optional max length (bp) for Evo2 tokenization.")
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="Filter datasets (exact match)")
    parser.add_argument("--fm-only", action="store_true",
                       help="Skip k-mer features and Ridge, run only FM embeddings")
    args = parser.parse_args()

    # Discover datasets
    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]

    print(f"📊 Benchmark: {len(all_datasets)} dataset/i")
    print(f"   k-values: {args.k_values}")
    print(f"   FM model: {args.model}")
    print(f"   Results: {RECORDS_CSV}\n")

    # Load FM una sola volta (scegli embedder basato su model name)
    model_lc = args.model.lower()
    if model_lc.startswith("evo2"):
        fm = Evo2Embedder(
            model_name=args.model,
            cache_dir="cache/fm_embeddings",
            layer_name=args.evo2_layer,
            max_length=args.evo2_max_length,
        )
    elif "hyenadna" in model_lc:
        fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings")
    else:
        fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")

    records = load_records(RECORDS_CSV)

    # Progress bar esterna per i dataset
    k_values = [] if args.fm_only else args.k_values
    total_work = len(all_datasets) * (1 + len(k_values))  # FM + k-mers per dataset
    pbar = tqdm(total=total_work, desc="Benchmark", unit="task", ncols=60)

    for ds in all_datasets:
        name = ds["name"]

        need_fm = not has_fm(name, args.model, records)
        pending_k: list[tuple[int, bool, bool]] = []
        need_any_ridge = False
        need_any_kmer = False
        for k in k_values:
            need_ridge = not has_ridge(name, k, args.model, records)
            need_kmer = not has_kmer(name, k, records)
            pending_k.append((k, need_ridge, need_kmer))
            need_any_ridge = need_any_ridge or need_ridge
            need_any_kmer = need_any_kmer or need_kmer

        # Skip dataset entirely if all requested outputs already exist in records.
        if not need_fm and not need_any_ridge and not need_any_kmer:
            pbar.update(1 + len(k_values))
            pbar.set_description(f"[skip] {name}")
            continue

        # Load sequences
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        Y_train = None
        Y_test = None
        if need_fm or need_any_ridge:
            # Compute/load FM embeddings only if needed for FM RF or Ridge.
            Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
            Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)

        # ── RF su FM (una sola volta per dataset) ──
        if need_fm:
            rf_fm = train_rf(Y_train.astype(np.float32), train_labels, n_classes)
            res_fm = eval_rf(rf_fm, Y_test.astype(np.float32), test_labels, n_classes)
            write_fm(name, args.model, res_fm)
            records = load_records(RECORDS_CSV)
        pbar.update(1)
        pbar.set_description(f"[{name}] FM")

        # ── Loop sui k-values ──
        for k, need_ridge, need_kmer in pending_k:
            if not need_ridge and not need_kmer:
                pbar.update(1)
                pbar.set_description(f"[skip] {name} k={k}")
                continue

            # Calcola feature k-mer una sola volta se servono (grid_size >= 2^k)
            if need_ridge or need_kmer:
                grid_size = max(128, 2 ** k)
                X_train = extract_kmer_features(train_seqs, k=k, grid_size=grid_size,
                                              n_workers=args.n_workers)
                X_test = extract_kmer_features(test_seqs, k=k, grid_size=grid_size,
                                             n_workers=args.n_workers)

            # Ridge
            if need_ridge:
                if Y_train is None or Y_test is None:
                    raise RuntimeError("Internal error: missing FM embeddings for Ridge.")
                ridge_raw = fit_and_evaluate(
                    X_train, Y_train.astype(np.float32),
                    X_test, Y_test.astype(np.float32),
                )
                write_ridge(name, k, args.model,
                           {"R2": ridge_raw["r2_global"], "MSE": ridge_raw["mse_global"]})
                records = load_records(RECORDS_CSV)

            # RF k-mer
            if need_kmer:
                rf_kmer = train_rf(X_train, train_labels, n_classes)
                res_kmer = eval_rf(rf_kmer, X_test, test_labels, n_classes)
                write_kmer(name, k, res_kmer)
                records = load_records(RECORDS_CSV)

            pbar.update(1)
            pbar.set_description(f"[{name}] k={k}")

    pbar.close()

    print(f"\n✅ Benchmark completato!")
    print(f"   Risultati salvati in: results/classification/records.csv")
    print(f"   Dataset: {len(records)}")
    print(f"   Colonne: {len(records.columns)}")


if __name__ == "__main__":
    main()
