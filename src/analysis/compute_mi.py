"""
Compute mean mutual information between features and labels for each dataset.

For k-mer features: average MI over all 4^k frequency dimensions.
For FM embeddings: average MI over all D embedding dimensions.
Computed on the training split.

Usage:
    python3 src/analysis/compute_mi.py --k-values 4 5 6
    python3 src/analysis/compute_mi.py --k-values 4 6 --fm-models InstaDeepAI/NTv3_650M_pre
    python3 src/analysis/compute_mi.py --k-values 4 6 --fm-models LongSafari/hyenadna-medium-160k-seqlen-hf
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.features.kmer_features import extract_kmer_features
from src.records.records import (
    load_mi_records, has_mi_kmer, has_mi_fm, has_mi_tok,
    write_mi_kmer, write_mi_fm, write_mi_tok, RECORDS_MI_CSV,
)
from src.embedders.embedding_cache import load_cached_embeddings


def load_tok_embeddings(model_name: str, dataset_name: str, split: str) -> np.ndarray | None:
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    path = os.path.join("cache/tok_embeddings", safe_model, safe_ds, f"{split}.npz")
    if os.path.exists(path):
        return np.load(path)["embeddings"].astype(np.float32)
    return None


def load_fm_embeddings(model_name: str, dataset_name: str, split: str) -> np.ndarray | None:
    return load_cached_embeddings(model_name, dataset_name, split)


def compute_mean_mi(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> float:
    """MI media tra tutte le feature e la label."""
    mi = mutual_info_classif(X, y, discrete_features=False,
                             n_neighbors=n_neighbors, random_state=42)
    return float(mi.mean())


def main():
    parser = argparse.ArgumentParser(
        description="Calcola Mutual Information feature→label per ogni dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[],
                        help="K-mer sizes per MI k-mer (es. 4 6)")
    parser.add_argument("--fm-models", nargs="+", default=[],
                        help="FM models per MI embeddings (es. InstaDeepAI/NTv3_650M_pre)")
    parser.add_argument("--tok-models", nargs="+", default=[],
                        help="FM models per MI tokenizer embeddings (da cache/tok_embeddings)")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="Subsample training set se più grande (0=nessun limite)")
    args = parser.parse_args()

    if not args.k_values and not args.fm_models and not args.tok_models:
        parser.error("Specifica almeno --k-values, --fm-models o --tok-models")

    all_datasets = discover_datasets(args.data_root)
    records = load_mi_records(RECORDS_MI_CSV)

    # Conta task totali
    n_total = 0
    for ds in all_datasets:
        name = ds["name"]
        for k in args.k_values:
            if not has_mi_kmer(name, k, records):
                n_total += 1
        for model in args.fm_models:
            if not has_mi_fm(name, model, records):
                n_total += 1
        for model in args.tok_models:
            if not has_mi_tok(name, model, records):
                n_total += 1

    print(f"\n📊 Mutual Information: feature → label")
    print(f"   k-values   : {args.k_values or '(nessuno)'}")
    print(f"   FM models  : {[m.split('/')[-1] for m in args.fm_models] or '(nessuno)'}")
    print(f"   Tok models : {[m.split('/')[-1] for m in args.tok_models] or '(nessuno)'}")
    print(f"   Dataset   : {len(all_datasets)}")
    print(f"   Task      : {n_total}\n")

    pbar = tqdm(total=n_total, unit="task", ncols=72)

    for ds in all_datasets:
        name = ds["name"]

        # Determina cosa serve
        pending_ks = [k for k in args.k_values if not has_mi_kmer(name, k, records)]
        pending_fms = [m for m in args.fm_models if not has_mi_fm(name, m, records)]
        pending_toks = [m for m in args.tok_models if not has_mi_tok(name, m, records)]

        if not pending_ks and not pending_fms and not pending_toks:
            continue

        # Load labels (always required)
        train_seqs, train_labels, _, _ = load_dataset(ds["train_path"], ds["test_path"])

        if args.max_train_samples > 0 and len(train_labels) > args.max_train_samples:
            n_orig = len(train_labels)
            rng = np.random.RandomState(42)
            idx = rng.choice(n_orig, args.max_train_samples, replace=False)
            idx.sort()
            train_seqs = [train_seqs[i] for i in idx]
            train_labels = np.asarray(train_labels)[idx]
            pbar.set_description(f"⚠️  {name} subsampled {args.max_train_samples}/{n_orig}")

        # MI per k-mer
        for k in pending_ks:
            grid_size = max(128, 2 ** k)
            X = extract_kmer_features(train_seqs, k=k, grid_size=grid_size,
                                      n_workers=args.n_workers)
            mi = compute_mean_mi(X, train_labels)
            write_mi_kmer(name, k, mi)
            records = load_mi_records(RECORDS_MI_CSV)
            pbar.update(1)
            pbar.set_description(f"[{name}] kmer k={k} MI={mi:.4f}")

        # MI per FM embeddings
        for model in pending_fms:
            emb = load_fm_embeddings(model, name, "train")
            if emb is None:
                pbar.update(1)
                pbar.set_description(f"⚠️  {name} (no {model.split('/')[-1]} cache)")
                continue

            mi = compute_mean_mi(emb, train_labels)
            write_mi_fm(name, model, mi)
            records = load_mi_records(RECORDS_MI_CSV)
            pbar.update(1)
            pbar.set_description(f"[{name}] {model.split('/')[-1]} MI={mi:.4f}")

        # MI per tokenizer embeddings
        for model in pending_toks:
            emb = load_tok_embeddings(model, name, "train")
            if emb is None:
                pbar.update(1)
                pbar.set_description(f"⚠️  {name} (no tok cache {model.split('/')[-1]})")
                continue

            mi = compute_mean_mi(emb, train_labels)
            write_mi_tok(name, model, mi)
            records = load_mi_records(RECORDS_MI_CSV)
            pbar.update(1)
            pbar.set_description(f"[{name}] tok_{model.split('/')[-1]} MI={mi:.4f}")

    pbar.close()
    print(f"\n✅ Done! Risultati in: {RECORDS_MI_CSV}")


if __name__ == "__main__":
    main()
