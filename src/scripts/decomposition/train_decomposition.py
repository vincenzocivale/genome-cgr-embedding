"""
Orthogonal decomposition of FM embeddings into k-mer-explainable and residual
components, with RF classification on each.

For each (dataset, k, model):
  1. Embed sequences with the FM (from cache or on-the-fly)
  2. Fit Ridge: k-mer frequencies -> FM embedding
  3. Decompose: proj = Ridge.predict(kmer), resid = FM - proj
  4. Train RF on proj, train RF on resid, record metrics

Processes one FM model at a time. After each dataset, deletes that dataset's
cached embeddings to free disk (unless --no-cleanup).

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 src/scripts/decomposition/train_decomposition.py --model InstaDeepAI/NTv3_650M_pre
    CUDA_VISIBLE_DEVICES=3 python3 src/scripts/decomposition/train_decomposition.py --model zhihan1996/DNABERT-2-117M --k-values 6
    CUDA_VISIBLE_DEVICES=3 python3 src/scripts/decomposition/train_decomposition.py --model LongSafari/hyenadna-medium-160k-seqlen-hf --no-cleanup
"""

import argparse
import os
import shutil
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.embedding_cache import load_cached_embeddings
from src.embedders.fm_embedder import validate_supported_model
from src.features.kmer_features import extract_kmer_features
from src.records.records import (
    load_decomp_records,
    has_decomp_ridge,
    has_proj,
    has_resid,
    write_decomp_ridge,
    write_proj,
    write_resid,
    RECORDS_DECOMP_CSV,
)
from src.training.ridge_mapping import fit_and_evaluate
from src.training.rf_pipeline import train_rf_fast, eval_rf


_EVO2_MODELS = {"evo2_7b", "evo2_7b_base", "evo2_1b_base", "evo2_40b", "evo2_40b_base", "evo2_20b"}


def _is_evo2(model_name: str) -> bool:
    return model_name in _EVO2_MODELS or model_name.startswith("evo2_")


def _make_embedder(model_name: str, fm_batch_size: int, pooling: str):
    """Lazily create the FM embedder (loads model onto GPU)."""
    model_lc = model_name.lower()
    if _is_evo2(model_name):
        from src.embedders.evo2_embedder import Evo2Embedder
        return Evo2Embedder(model_name=model_name,
                            cache_dir="cache/fm_embeddings")
    elif "hyenadna" in model_lc:
        from src.embedders.hyena_embedder import HyenaEmbedder
        return HyenaEmbedder(model_name=model_name,
                             cache_dir="cache/hyena_embeddings",
                             pooling=pooling)
    else:
        from src.embedders.fm_embedder import FMEmbedder
        return FMEmbedder(model_name=model_name,
                          cache_dir="cache/fm_embeddings",
                          pooling=pooling)


def _cache_paths_for_dataset(model_name: str, dataset_name: str,
                             pooling: str = "mean") -> list[str]:
    """Return all possible cache file paths for a (model, dataset) pair."""
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    paths = []
    for split in ("train", "test"):
        paths.append(os.path.join("cache/fm_embeddings", safe_model, f"pooling_{pooling}", safe_ds,
                                  f"{split}.npz"))
        paths.append(os.path.join("cache/fm_embeddings", safe_model, safe_ds,
                                  f"{split}.npz"))
        if "hyenadna" in model_name.lower():
            paths.append(os.path.join("cache/hyena_embeddings", f"pooling_{pooling}", safe_ds,
                                      f"{split}.npz"))
            paths.append(os.path.join("cache/hyena_embeddings", safe_ds,
                                      f"{split}.npz"))
    return paths


def _embed_dataset_evo2(embedder, ds: dict, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings for a dataset using the Evo2Embedder interface."""
    from src.data.loader import load_dataset
    data = load_dataset(ds)
    X_tr = embedder.embed_sequences(data["X_train"], ds["name"], "train", batch_size=batch_size)
    X_te = embedder.embed_sequences(data["X_test"],  ds["name"], "test",  batch_size=batch_size)
    return X_tr, X_te, data


def _delete_dataset_cache(model_name: str, dataset_name: str,
                          pooling: str = "mean"):
    """Remove cached embeddings for a single dataset."""
    for p in _cache_paths_for_dataset(model_name, dataset_name, pooling=pooling):
        if os.path.exists(p):
            os.remove(p)
    # Clean up empty parent dirs
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    ds_dir = os.path.join("cache/fm_embeddings", safe_model, f"pooling_{pooling}", safe_ds)
    if os.path.isdir(ds_dir) and not os.listdir(ds_dir):
        os.rmdir(ds_dir)
    if "hyenadna" in model_name.lower():
        ds_dir_h = os.path.join("cache/hyena_embeddings", f"pooling_{pooling}", safe_ds)
        if os.path.isdir(ds_dir_h) and not os.listdir(ds_dir_h):
            os.rmdir(ds_dir_h)


def main():
    parser = argparse.ArgumentParser(
        description="Orthogonal decomposition: proj (k-mer) vs residual RF"
    )
    parser.add_argument("--data-root",
                        default="data/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True,
                        help="FM model name (e.g. InstaDeepAI/NTv3_650M_pre)")
    parser.add_argument("--pooling", choices=["mean", "max", "cls"], default="mean")
    parser.add_argument("--mapper", choices=["ridge", "mlp"], default="ridge")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--mlp-batch-size", type=int, default=256)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
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
    print(f"\nModel : {model_tag}")
    print(f"Pooling: {args.pooling}")
    print(f"Mapper : {args.mapper}")
    print(f"k-values: {args.k_values}")
    print(f"Datasets: {len(all_datasets)}")
    print(f"Results : {RECORDS_DECOMP_CSV}\n")

    records = load_decomp_records()
    total = len(all_datasets) * len(args.k_values)
    pbar = tqdm(total=total, desc="Decomposition", unit="task", ncols=70)

    # Lazy-loaded: only instantiated if we actually need to embed
    embedder = None

    for ds in all_datasets:
        name = ds["name"]

        # Check which (k) values still need work
        pending_ks = []
        for k in args.k_values:
            need_ridge = not has_decomp_ridge(
                name, k, args.model, records, pooling=args.pooling, mapper=args.mapper
            )
            need_proj = not has_proj(
                name, k, args.model, records, pooling=args.pooling, mapper=args.mapper
            )
            need_resid = not has_resid(
                name, k, args.model, records, pooling=args.pooling, mapper=args.mapper
            )
            if need_ridge or need_proj or need_resid:
                pending_ks.append((k, need_ridge, need_proj, need_resid))

        if not pending_ks:
            pbar.update(len(args.k_values))
            pbar.set_description(f"[skip] {name}")
            continue

        # Load sequences and labels
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # Try cache first, embed on-the-fly if missing
        Y_train = load_cached_embeddings(args.model, name, "train", pooling=args.pooling)
        Y_test = load_cached_embeddings(args.model, name, "test", pooling=args.pooling)
        generated_embeddings = False

        if Y_train is None or Y_test is None:
            pbar.set_description(f"[embedding] {name}")
            if embedder is None:
                embedder = _make_embedder(args.model, args.fm_batch_size, args.pooling)
            Y_train = embedder.embed_sequences(
                train_seqs, name, "train", args.fm_batch_size
            )
            Y_test = embedder.embed_sequences(
                test_seqs, name, "test", args.fm_batch_size
            )
            generated_embeddings = True

        for k, need_ridge, need_proj, need_resid in pending_ks:
            pbar.set_description(f"[{name}] k={k}")

            grid_size = max(128, 2 ** k)
            X_train = extract_kmer_features(
                train_seqs, k=k, grid_size=grid_size, n_workers=args.n_workers
            )
            X_test = extract_kmer_features(
                test_seqs, k=k, grid_size=grid_size, n_workers=args.n_workers
            )

            # Fit Ridge (always needed for decomposition)
            if args.mapper == "ridge":
                ridge_result = fit_and_evaluate(
                    X_train, Y_train.astype(np.float32),
                    X_test, Y_test.astype(np.float32),
                )
            else:
                from src.training.mlp_mapping import fit_and_evaluate_mlp

                ridge_result = fit_and_evaluate_mlp(
                    X_train, Y_train.astype(np.float32),
                    X_test, Y_test.astype(np.float32),
                    hidden_dim=args.mlp_hidden_dim,
                    epochs=args.mlp_epochs,
                    batch_size=args.mlp_batch_size,
                    lr=args.mlp_lr,
                    weight_decay=args.mlp_weight_decay,
                )

            # Save Ridge metrics
            if need_ridge:
                write_decomp_ridge(
                    name,
                    k,
                    args.model,
                    {
                        "R2": ridge_result["r2_global"],
                        "MSE": ridge_result["mse_global"],
                    },
                    pooling=args.pooling,
                    mapper=args.mapper,
                )

            # Decompose
            Y_proj_train = ridge_result["Y_pred_train"].astype(np.float32)
            Y_proj_test = ridge_result["Y_pred_test"].astype(np.float32)
            Y_resid_train = Y_train.astype(np.float32) - Y_proj_train
            Y_resid_test = Y_test.astype(np.float32) - Y_proj_test

            # RF on projected (k-mer) component
            if need_proj:
                rf_proj = train_rf_fast(Y_proj_train, train_labels, n_classes)
                metrics_proj = eval_rf(rf_proj, Y_proj_test, test_labels, n_classes)
                write_proj(name, k, args.model, metrics_proj,
                           pooling=args.pooling, mapper=args.mapper)

            # RF on residual component
            if need_resid:
                rf_resid = train_rf_fast(Y_resid_train, train_labels, n_classes)
                metrics_resid = eval_rf(rf_resid, Y_resid_test, test_labels, n_classes)
                write_resid(name, k, args.model, metrics_resid,
                            pooling=args.pooling, mapper=args.mapper)

            records = load_decomp_records()
            pbar.update(1)

        # Update remaining k's that were already done
        already_done = len(args.k_values) - len(pending_ks)
        if already_done > 0:
            pbar.update(already_done)

        # Cleanup this dataset's cache to free disk
        if not args.no_cleanup and generated_embeddings:
            _delete_dataset_cache(args.model, name, pooling=args.pooling)

    pbar.close()
    print(f"\nDone! Results in {RECORDS_DECOMP_CSV}")


if __name__ == "__main__":
    main()
