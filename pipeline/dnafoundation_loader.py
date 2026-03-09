"""Loader per i dataset CSV del DNA Foundation Benchmark (Feng et al., 2025)."""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "data", "dna_foundation_benchmark",
)


def discover_datasets(data_root: str = DATA_ROOT) -> list:
    """Trova tutte le directory con train.csv e test.csv.

    Returns:
        Lista ordinata di dict con chiavi: name, train_path, test_path.
    """
    data_root = os.path.abspath(data_root)
    datasets = []

    for dirpath, _dirnames, filenames in os.walk(data_root):
        if "train.csv" in filenames and "test.csv" in filenames:
            rel = os.path.relpath(dirpath, data_root)
            datasets.append({
                "name": rel,
                "train_path": os.path.join(dirpath, "train.csv"),
                "test_path": os.path.join(dirpath, "test.csv"),
            })

    datasets.sort(key=lambda d: d["name"])
    return datasets


def load_dataset_csv(train_path: str, test_path: str):
    """Carica un dataset da file CSV.

    Formato atteso: prima colonna = sequenza DNA, seconda colonna = label.

    Returns:
        (train_seqs, train_labels, test_seqs, test_labels)
    """
    train_df = pd.read_csv(train_path, header=0)
    test_df = pd.read_csv(test_path, header=0)

    train_seqs = train_df.iloc[:, 0].str.upper().values
    test_seqs = test_df.iloc[:, 0].str.upper().values

    train_labels_raw = train_df.iloc[:, 1].values
    test_labels_raw = test_df.iloc[:, 1].values

    le = LabelEncoder()
    le.fit(np.concatenate([train_labels_raw, test_labels_raw]))
    train_labels: np.ndarray = le.transform(train_labels_raw)
    test_labels: np.ndarray = le.transform(test_labels_raw)

    return train_seqs, train_labels, test_seqs, test_labels
