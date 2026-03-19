"""
Popola le colonne descrittive dei dataset in records.csv.

Per ogni dataset calcola:
  - n_train, n_test: numero di sequenze
  - n_classes: numero di classi
  - class_balance: rapporto min_class / max_class (1.0 = perfettamente bilanciato)
  - seq_len_mean, seq_len_std, seq_len_min, seq_len_max
  - gc_content_mean: % media di G+C su tutte le sequenze (train+test)

Usage:
    python3 src/fm_experiment/fill_dataset_info.py
"""

import argparse
import os
import sys

import numpy as np
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.records import (
    load_records, has_dataset_info, write_dataset_info, RECORDS_CSV,
)


def compute_dataset_info(train_seqs, train_labels, test_seqs, test_labels) -> dict:
    all_seqs = np.concatenate([train_seqs, test_seqs])
    all_labels = np.concatenate([train_labels, test_labels])

    lengths = np.array([len(s) for s in all_seqs])

    class_counts = Counter(all_labels)
    n_classes = len(class_counts)
    counts = list(class_counts.values())
    class_balance = min(counts) / max(counts)

    gc_ratios = []
    for s in all_seqs:
        s_upper = s.upper()
        gc = s_upper.count("G") + s_upper.count("C")
        gc_ratios.append(gc / len(s_upper) if len(s_upper) > 0 else 0.0)

    return {
        "n_train": len(train_seqs),
        "n_test": len(test_seqs),
        "n_classes": n_classes,
        "class_balance": round(class_balance, 4),
        "seq_len_mean": round(float(lengths.mean()), 1),
        "seq_len_std": round(float(lengths.std()), 1),
        "seq_len_min": int(lengths.min()),
        "seq_len_max": int(lengths.max()),
        "gc_content_mean": round(float(np.mean(gc_ratios)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Fill dataset info columns in records.csv")
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    records = load_records(RECORDS_CSV)

    pending = [ds for ds in all_datasets if not has_dataset_info(ds["name"], records)]
    print(f"\n📊 Dataset info: {len(pending)} da calcolare, "
          f"{len(all_datasets) - len(pending)} già presenti\n")

    for ds in tqdm(pending, unit="ds", ncols=70):
        name = ds["name"]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        info = compute_dataset_info(train_seqs, train_labels, test_seqs, test_labels)
        write_dataset_info(name, info)
        records = load_records(RECORDS_CSV)

    print(f"\n✅ Completato! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
