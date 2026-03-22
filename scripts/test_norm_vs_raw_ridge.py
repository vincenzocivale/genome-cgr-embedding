"""
Quick test: Ridge regression performance with normalized vs raw k-mer features.

Compares R² of Ridge(kmer → FM embedding) with:
  - normalize=True  (L1: counts / sum → probability distribution)
  - normalize=False (raw counts)

Tests on a few datasets for k=4 and k=6, using NTv3 embeddings.

Usage:
    python3 scripts/test_norm_vs_raw_ridge.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data.loader import discover_datasets, load_dataset
from src.core.fcgr import batch_fcgr
from src.core.kmer import KmerEmbedder
from src.fm_experiment.ridge_mapping import fit_and_evaluate

DATA_ROOT = "data/dna_foundation_benchmark/"
FM_MODEL = "InstaDeepAI/NTv3_650M_pre"

TEST_DATASETS = [
    "EMP/Yeast_H3",
    "deep4mc/E.coli_4mC",
    "enhancers/enhancer",
    "genomic_benchmark/human_vs_worm",
    "iDNA_ABF/6mA",
    "mouse/mouse_TFBS_1",
]

K_VALUES = [4, 6]


def load_embeddings(model_name, dataset_name, split):
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")
    candidates = [
        os.path.join("cache/fm_embeddings", safe_model, safe_ds, f"{split}.npz"),
        os.path.join("cache/fm_embeddings", safe_ds, f"{split}.npz"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return np.load(path)["embeddings"].astype(np.float32)
    return None


def main():
    all_ds = discover_datasets(DATA_ROOT)
    ds_map = {d["name"]: d for d in all_ds}

    print(f"{'Dataset':<35} {'k':>2}  {'R2 norm':>8} {'R2 raw':>8} {'diff':>8}  {'MSE norm':>10} {'MSE raw':>10}  {'alpha_n':>8} {'alpha_r':>8}")
    print("-" * 120)

    for ds_name in TEST_DATASETS:
        if ds_name not in ds_map:
            print(f"{ds_name:<35}  NOT FOUND")
            continue

        ds = ds_map[ds_name]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )

        # Load FM embeddings
        Y_train = load_embeddings(FM_MODEL, ds_name, "train")
        Y_test = load_embeddings(FM_MODEL, ds_name, "test")
        if Y_train is None or Y_test is None:
            print(f"{ds_name:<35}  NO EMBEDDINGS CACHED")
            continue

        for k in K_VALUES:
            grid_size = max(128, 2 ** k)
            grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=4)
            grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=4)

            # Normalized (L1)
            emb_n = KmerEmbedder(k_size=k, normalize=True)
            X_tr_n = np.array([emb_n.compute_embedding(g) for g in grids_train])
            X_te_n = np.array([emb_n.compute_embedding(g) for g in grids_test])
            res_n = fit_and_evaluate(X_tr_n, Y_train, X_te_n, Y_test)

            # Raw counts
            emb_r = KmerEmbedder(k_size=k, normalize=False)
            X_tr_r = np.array([emb_r.compute_embedding(g) for g in grids_train])
            X_te_r = np.array([emb_r.compute_embedding(g) for g in grids_test])
            res_r = fit_and_evaluate(X_tr_r, Y_train, X_te_r, Y_test)

            diff = res_n["r2_global"] - res_r["r2_global"]
            print(f"{ds_name:<35} {k:>2}  {res_n['r2_global']:>8.4f} {res_r['r2_global']:>8.4f} {diff:>+8.4f}  {res_n['mse_global']:>10.4f} {res_r['mse_global']:>10.4f}  {res_n['best_alpha']:>8.1f} {res_r['best_alpha']:>8.1f}")

    print()


if __name__ == "__main__":
    main()
