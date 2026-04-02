"""
Dataset discovery and loading for the DNA Foundation Benchmark (Feng et al., 2025).

Expected dataset layout:
  <data_root>/
    <category>/<task>/train.csv
    <category>/<task>/test.csv

CSV format: first column = DNA sequence, second column = label.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def discover_datasets(data_root: str) -> list:
    """
    Recursively find all directories that contain both train.csv and test.csv.

    Returns a sorted list of dicts: {name, train_path, test_path}.
    """
    data_root = os.path.abspath(data_root)
    datasets = []
    # Follow symlinks to support dataset subsets built via symlink or bind mounts.
    for dirpath, _dirnames, filenames in os.walk(data_root, followlinks=True):
        if "train.csv" in filenames and "test.csv" in filenames:
            rel = os.path.relpath(dirpath, data_root)
            datasets.append({
                "name": rel,
                "train_path": os.path.join(dirpath, "train.csv"),
                "test_path": os.path.join(dirpath, "test.csv"),
            })
    datasets.sort(key=lambda d: d["name"])
    return datasets


def load_dataset(train_path: str, test_path: str):
    """
    Load a dataset from CSV files.

    Returns:
        (train_seqs, train_labels, test_seqs, test_labels)
        Labels are integer-encoded starting from 0.
    """
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    train_seqs = train_df.iloc[:, 0].str.upper().values
    test_seqs = test_df.iloc[:, 0].str.upper().values

    train_labels_raw = train_df.iloc[:, 1].values
    test_labels_raw = test_df.iloc[:, 1].values

    le = LabelEncoder()
    le.fit(np.concatenate([train_labels_raw, test_labels_raw]))
    train_labels = le.transform(train_labels_raw)
    test_labels = le.transform(test_labels_raw)

    return train_seqs, train_labels, test_seqs, test_labels
