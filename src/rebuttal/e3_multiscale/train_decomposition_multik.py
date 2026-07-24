"""
Rebuttal E3 (part 2/2) — FM decomposition against a multi-k baseline.

Indirectly requested by eZYW: does the FM residual stay informative once the
compositional baseline integrates multiple k-mer scales at once, instead of a
single k? Adapted from src/scripts/decomposition/train_decomposition.py, the only
change is the ridge-regression input: X = extract_multiscale_kmer_features(seqs,
k_values=(4,5,6)) (per-scale L2-normalised, concatenated, 256+1024+4096=5376-dim)
instead of a single extract_kmer_features(seqs, k=k). Same pipeline otherwise:
  1. Embed sequences with the FM (from cache or on-the-fly)
  2. Fit Ridge: multi-k frequencies -> FM embedding
  3. Decompose: proj = Ridge.predict(multi-k), resid = FM - proj
  4. Train RF on proj, train RF on resid, record metrics

Writes into results/decomposition/records_decomposition.csv using the multi-k
columns (ridge_multi_k4_5_6_{model}_R2, proj_multi_k4_5_6_{model}_*,
resid_multi_k4_5_6_{model}_*) added in src/records/records.py, alongside the
existing single-k columns produced by train_decomposition.py.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 src/rebuttal/e3_multiscale/train_decomposition_multik.py \
        --model InstaDeepAI/NTv3_650M_pre
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.embedding_cache import load_cached_embeddings
from src.embedders.fm_embedder import validate_supported_model
from src.features.kmer_features import extract_multiscale_kmer_features
from src.records.records import (
    load_decomp_records,
    has_ridge_multi,
    has_proj_multi,
    has_resid_multi,
    write_ridge_multi,
    write_proj_multi,
    write_resid_multi,
    RECORDS_DECOMP_CSV,
)
from src.training.ridge_mapping import fit_and_evaluate
from src.training.rf_pipeline import train_rf_fast, eval_rf

# Reuse the original decomposition script's embedder/cache-cleanup machinery so the
# two scripts share identical cache paths and behave the same way for a given model.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "decomposition"))
from train_decomposition import _make_embedder, _delete_dataset_cache, _is_evo2  # noqa: E402

K_VALUES = (4, 5, 6)


def main():
    parser = argparse.ArgumentParser(
        description="E3: FM decomposition against multi-k (k=4,5,6 concatenated) baseline"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True,
                        help="FM model name (e.g. InstaDeepAI/NTv3_650M_pre)")
    parser.add_argument("--pooling", choices=["mean", "max", "cls"], default="mean")
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--no-cleanup", action="store_true",
                        help="Do not delete embedding cache after each dataset")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Filter datasets (exact match on name)")
    parser.add_argument("--reverse", action="store_true",
                        help="Process datasets in reverse order")
    args = parser.parse_args()

    if not _is_evo2(args.model):
        validate_supported_model(args.model)
    if args.pooling == "cls" and "hyenadna" in args.model.lower():
        raise ValueError("CLS pooling is not supported for HyenaDNA.")

    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]
    if args.reverse:
        all_datasets = list(reversed(all_datasets))

    model_tag = args.model.split("/")[-1]
    print(f"\nModel   : {model_tag}")
    print(f"Pooling : {args.pooling}")
    print(f"K-values: multi {K_VALUES} (concatenated)")
    print(f"Datasets: {len(all_datasets)}")
    print(f"Results : {RECORDS_DECOMP_CSV}\n")

    records = load_decomp_records()
    pbar = tqdm(total=len(all_datasets), desc="Multi-k decomposition", unit="dataset", ncols=70)

    embedder = None

    for ds in all_datasets:
        name = ds["name"]

        need_ridge = not has_ridge_multi(name, K_VALUES, args.model, records)
        need_proj = not has_proj_multi(name, K_VALUES, args.model, records)
        need_resid = not has_resid_multi(name, K_VALUES, args.model, records)

        if not (need_ridge or need_proj or need_resid):
            pbar.set_description(f"[skip] {name}")
            pbar.update(1)
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        try:
            Y_train = load_cached_embeddings(args.model, name, "train", pooling=args.pooling)
            Y_test = load_cached_embeddings(args.model, name, "test", pooling=args.pooling)
        except Exception as e:
            print(f"  [warn] corrupted cache for {name} ({e}); will re-embed", flush=True)
            Y_train = Y_test = None
        generated_embeddings = False

        if Y_train is None or Y_test is None:
            pbar.set_description(f"[embedding] {name}")
            try:
                if embedder is None:
                    embedder = _make_embedder(args.model, args.fm_batch_size, args.pooling)
                Y_train = embedder.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
                Y_test = embedder.embed_sequences(test_seqs, name, "test", args.fm_batch_size)
                generated_embeddings = True
            except Exception as e:
                print(f"  [skip] {name}: cache unusable and re-embed failed ({e})", flush=True)
                pbar.update(1)
                continue

        pbar.set_description(f"[{name}] multi-k")

        max_k = max(K_VALUES)
        grid_size = max(128, 2 ** max_k)
        X_train = extract_multiscale_kmer_features(
            train_seqs, k_values=K_VALUES, grid_size=grid_size, n_workers=args.n_workers
        )
        X_test = extract_multiscale_kmer_features(
            test_seqs, k_values=K_VALUES, grid_size=grid_size, n_workers=args.n_workers
        )

        ridge_result = fit_and_evaluate(
            X_train, Y_train.astype(np.float32),
            X_test, Y_test.astype(np.float32),
        )

        if need_ridge:
            write_ridge_multi(
                name, K_VALUES, args.model,
                {"R2": ridge_result["r2_global"], "MSE": ridge_result["mse_global"]},
            )

        Y_proj_train = ridge_result["Y_pred_train"].astype(np.float32)
        Y_proj_test = ridge_result["Y_pred_test"].astype(np.float32)
        Y_resid_train = Y_train.astype(np.float32) - Y_proj_train
        Y_resid_test = Y_test.astype(np.float32) - Y_proj_test

        if need_proj:
            rf_proj = train_rf_fast(Y_proj_train, train_labels, n_classes)
            metrics_proj = eval_rf(rf_proj, Y_proj_test, test_labels, n_classes)
            write_proj_multi(name, K_VALUES, args.model, metrics_proj)

        if need_resid:
            rf_resid = train_rf_fast(Y_resid_train, train_labels, n_classes)
            metrics_resid = eval_rf(rf_resid, Y_resid_test, test_labels, n_classes)
            write_resid_multi(name, K_VALUES, args.model, metrics_resid)

        records = load_decomp_records()
        pbar.update(1)

        if not args.no_cleanup and generated_embeddings:
            _delete_dataset_cache(args.model, name, pooling=args.pooling)

    pbar.close()
    print(f"\nDone! Results in {RECORDS_DECOMP_CSV}")


if __name__ == "__main__":
    main()
