"""
Quick test: RF performance with normalized vs raw k-mer frequency vectors.

Runs on a handful of datasets for k=4 and k=6, comparing:
  - normalize=True  (L1: counts / sum → probability distribution)
  - normalize=False (raw counts)

Usage:
    python3 scripts/test_norm_vs_raw.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.data.loader import discover_datasets, load_dataset
from src.core.fcgr import batch_fcgr
from src.core.kmer import KmerEmbedder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_ROOT = "data/dna_foundation_benchmark/"

# Pick a diverse subset: short seqs, medium, long, different categories
TEST_DATASETS = [
    "EMP/Yeast_H3",
    "deep4mc/E.coli_4mC",
    "enhancers/enhancer",
    "genomic_benchmark/human_vs_worm",
    "iDNA_ABF/6mA",
    "mouse/mouse_TFBS_1",
]

K_VALUES = [4, 6]


def train_eval_rf(X_train, y_train, X_test, y_test, n_classes):
    rf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt", max_depth=20,
        n_jobs=2, random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    if n_classes == 2:
        auroc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    else:
        auroc = roc_auc_score(y_test, rf.predict_proba(X_test),
                              multi_class="ovr", average="macro")
    return auroc, acc


def main():
    all_ds = discover_datasets(DATA_ROOT)
    ds_map = {d["name"]: d for d in all_ds}

    print(f"{'Dataset':<35} {'k':>2}  {'AUROC norm':>10} {'AUROC raw':>10} {'diff':>8}  {'Acc norm':>8} {'Acc raw':>8}")
    print("-" * 110)

    for ds_name in TEST_DATASETS:
        if ds_name not in ds_map:
            print(f"{ds_name:<35}  NOT FOUND")
            continue

        ds = ds_map[ds_name]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        for k in K_VALUES:
            grid_size = max(128, 2 ** k)
            grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=4)
            grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=4)

            # Normalized (L1)
            emb_norm = KmerEmbedder(k_size=k, normalize=True)
            X_tr_n = np.array([emb_norm.compute_embedding(g) for g in grids_train])
            X_te_n = np.array([emb_norm.compute_embedding(g) for g in grids_test])
            auroc_n, acc_n = train_eval_rf(X_tr_n, train_labels, X_te_n, test_labels, n_classes)

            # Raw counts
            emb_raw = KmerEmbedder(k_size=k, normalize=False)
            X_tr_r = np.array([emb_raw.compute_embedding(g) for g in grids_train])
            X_te_r = np.array([emb_raw.compute_embedding(g) for g in grids_test])
            auroc_r, acc_r = train_eval_rf(X_tr_r, train_labels, X_te_r, test_labels, n_classes)

            diff = auroc_n - auroc_r
            print(f"{ds_name:<35} {k:>2}  {auroc_n:>10.4f} {auroc_r:>10.4f} {diff:>+8.4f}  {acc_n:>8.4f} {acc_r:>8.4f}")

    print()


if __name__ == "__main__":
    main()
