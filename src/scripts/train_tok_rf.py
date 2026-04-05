"""
Script: genera embeddings dall'embedding layer (pre-transformer) e addestra RF.

Confronta la rappresentazione statica dei token (solo lookup table)
con quella contestualizzata (output completo del FM).

Usage:
    python3 src/fm_experiment/train_tok_rf.py --model InstaDeepAI/NTv3_650M_pre
    python3 src/fm_experiment/train_tok_rf.py --model LongSafari/hyenadna-medium-160k-seqlen-hf
"""

import argparse
import sys
import os

import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.tokenizer_embedder import TokenizerEmbedder
from src.records.records import (
    load_records, has_tok, write_tok, RECORDS_CSV,
)
from src.training.efficiency import log_efficiency
from src.training.rf_pipeline import train_rf, eval_rf


def main():
    parser = argparse.ArgumentParser(
        description="Train RF su embedding-layer (pre-transformer) per tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True,
                       help="FM model HF name")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets)

    print(f"\n🔤 Tokenizer Embedding RF: {args.model.split('/')[-1]}")
    print(f"📊 Dataset totali: {n_total}")
    print(f"📁 Results: {RECORDS_CSV}\n")

    embedder = TokenizerEmbedder(model_name=args.model)
    records = load_records(RECORDS_CSV)

    pbar = tqdm(all_datasets, desc="Progress", unit="ds", ncols=70)

    for ds in pbar:
        name = ds["name"]

        if has_tok(name, args.model, records):
            pbar.set_description(f"⏭ {name} (skip)")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        t0 = time.perf_counter()
        X_train = embedder.embed_sequences(
            train_seqs, name, "train", args.batch_size
        ).astype(np.float32)
        X_test = embedder.embed_sequences(
            test_seqs, name, "test", args.batch_size
        ).astype(np.float32)
        t1 = time.perf_counter()
        log_efficiency(name, f"tok_{args.model.split('/')[-1]}", X_train.shape[1], t1 - t0)

        rf = train_rf(X_train, train_labels, n_classes)
        metrics = eval_rf(rf, X_test, test_labels, n_classes)
        write_tok(name, args.model, metrics)
        records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name} MCC={metrics['MCC']:.3f}")

    print(f"\n✅ Done! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
