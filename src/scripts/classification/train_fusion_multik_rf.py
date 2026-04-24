"""
Pilot pipeline: fusion of multi-k k-mer vectors with different strategies.

Creates a separate CSV (results/fusion_records.csv) with wide columns.

Strategies:
- concat: simple concatenation of L2-normalized k-mer vectors
- weighted: per-dataset weight search (LogisticRegression CV) then concat

Usage:
    python3 src/fm_experiment/train_fusion_multik_rf.py --k-values 4 5 6 7 --pilot-ntv3-top 8
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from itertools import product

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.core.fcgr import batch_fcgr
from src.features.kmer_features import kmer_from_grids
from src.training.rf_pipeline import train_rf, eval_rf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

RECORDS_CSV = "results/classification/records.csv"
OUTPUT_CSV = "results/exploratory/fusion_records.csv"

RF_METRICS = ["MCC", "AUROC", "F1", "Accuracy"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (x / norms).astype(np.float32)


def _concat_weighted(feats_list, weights):
    parts = [f * float(w) for f, w in zip(feats_list, weights)]
    return np.concatenate(parts, axis=1)


def _parse_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _load_records_rows(path: str):
    if not os.path.exists(path):
        return [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return rows, r.fieldnames or []


def _save_rows(path: str, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _upsert_row(rows, fieldnames, dataset, updates: dict):
    # ensure fieldnames
    for k in updates.keys():
        if k not in fieldnames:
            fieldnames.append(k)

    # find row
    for row in rows:
        if row.get("dataset") == dataset:
            row.update({k: v for k, v in updates.items()})
            return rows, fieldnames

    # add new row
    new_row = {k: "" for k in fieldnames}
    new_row["dataset"] = dataset
    new_row.update({k: v for k, v in updates.items()})
    rows.append(new_row)
    return rows, fieldnames


def _has_method(rows, dataset, col):
    for row in rows:
        if row.get("dataset") == dataset:
            v = row.get(col)
            return v is not None and str(v).strip() != ""
    return False


def _best_kmer_k4_6(row):
    best = None
    for k in [4, 5, 6]:
        v = _parse_float(row.get(f"kmer_k{k}_MCC"))
        if v is None:
            continue
        if best is None or v > best:
            best = v
    return best


def select_ntv3_better_top_n(n: int) -> list[str]:
    if not os.path.exists(RECORDS_CSV):
        return []
    with open(RECORDS_CSV, newline="") as f:
        r = csv.DictReader(f)
        items = []
        for row in r:
            ntv3 = _parse_float(row.get("fm_NTv3_650M_pre_MCC"))
            best_k = _best_kmer_k4_6(row)
            if ntv3 is None or best_k is None:
                continue
            delta = ntv3 - best_k
            items.append((delta, row.get("dataset")))
    items.sort(key=lambda x: x[0], reverse=True)
    return [ds for _, ds in items[:n]]


def main():
    parser = argparse.ArgumentParser(
        description="Fusion multi-k k-mer RF (pilot pipeline)"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--pilot-ntv3-top", type=int, default=8,
                        help="Use top-N datasets where NTv3 beats best k-mer (k4-6)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Explicit dataset list (overrides pilot selection)")
    parser.add_argument("--weight-grid", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    k_values = tuple(sorted(args.k_values))
    k_tag = "_".join(str(k) for k in k_values)

    # select datasets
    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        selected = [d for d in all_datasets if d["name"] in set(args.datasets)]
    else:
        top_ds = select_ntv3_better_top_n(args.pilot_ntv3_top) if args.pilot_ntv3_top > 0 else []
        selected = [d for d in all_datasets if d["name"] in set(top_ds)] if top_ds else all_datasets

    print("\nFusion multi-k k-mer RF")
    print(f"   k-values : {list(k_values)}")
    print(f"   pilot    : {len(selected)} dataset")
    print(f"   output   : {args.output}\n")

    rows_out, fieldnames = _load_records_rows(args.output)
    if not fieldnames:
        fieldnames = ["dataset"]

    pbar = tqdm(selected, unit="ds", ncols=72)

    for ds in pbar:
        name = ds["name"]
        concat_col = f"fusion_k{k_tag}_concat_MCC"
        weighted_col = f"fusion_k{k_tag}_weighted_MCC"

        if _has_method(rows_out, name, concat_col) and _has_method(rows_out, name, weighted_col):
            pbar.set_description(f"⏭ {name}")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # compute FCGR grids once
        max_k = max(k_values)
        grid_size = max(128, 2 ** max_k)
        t0 = time.perf_counter()
        grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
        grids_test = batch_fcgr(test_seqs, grid_size=grid_size, n_workers=args.n_workers)
        t1 = time.perf_counter()

        # per-k features (L1 then L2 per scale)
        feats_train = []
        feats_test = []
        for k in k_values:
            X_tr = kmer_from_grids(grids_train, k, normalize="l1")
            X_te = kmer_from_grids(grids_test, k, normalize="l1")
            feats_train.append(_l2_normalize(X_tr))
            feats_test.append(_l2_normalize(X_te))

        updates = {}

        # concat baseline
        if not _has_method(rows_out, name, concat_col):
            X_train = np.concatenate(feats_train, axis=1).astype(np.float32)
            X_test = np.concatenate(feats_test, axis=1).astype(np.float32)
            rf = train_rf(X_train, train_labels, n_classes)
            metrics = eval_rf(rf, X_test, test_labels, n_classes)
            for m in RF_METRICS:
                updates[f"fusion_k{k_tag}_concat_{m}"] = metrics[m]

        # weighted concat
        if not _has_method(rows_out, name, weighted_col):
            weight_grid = list(product(args.weight_grid, repeat=len(k_values)))
            scoring = "roc_auc" if n_classes == 2 else "accuracy"
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            clf = LogisticRegression(max_iter=1000, random_state=42)
            best_score = -1.0
            best_weights = list(weight_grid[0])
            for weights in weight_grid:
                X_cv = _concat_weighted(feats_train, weights)
                scores = cross_val_score(clf, X_cv, train_labels, cv=cv, scoring=scoring, n_jobs=1)
                score = float(np.mean(scores))
                if score > best_score:
                    best_score = score
                    best_weights = list(weights)

            X_train = _concat_weighted(feats_train, best_weights).astype(np.float32)
            X_test = _concat_weighted(feats_test, best_weights).astype(np.float32)
            rf = train_rf(X_train, train_labels, n_classes)
            metrics = eval_rf(rf, X_test, test_labels, n_classes)
            for m in RF_METRICS:
                updates[f"fusion_k{k_tag}_weighted_{m}"] = metrics[m]
            updates[f"fusion_k{k_tag}_weighted_weights"] = ",".join(str(w) for w in best_weights)

        if updates:
            rows_out, fieldnames = _upsert_row(rows_out, fieldnames, name, updates)
            _save_rows(args.output, rows_out, fieldnames)

        pbar.set_description(f"✓ {name}")

    print(f"\n✅ Done! Risultati in: {args.output}")


if __name__ == "__main__":
    main()
