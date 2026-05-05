"""
Extract FM embeddings and train a Random Forest on all classification datasets.

Usage:
    python3 src/scripts/classification/train_fm_rf.py --model InstaDeepAI/NTv3_650M_pre
    python3 src/scripts/classification/train_fm_rf.py --model LongSafari/hyenadna-medium-160k-seqlen-hf
    python3 src/scripts/classification/train_fm_rf.py --model evo2_7b_base --evo2-layer norm
"""

import argparse
import sys
import os

import time
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder, validate_supported_model
from src.embedders.hyena_embedder import HyenaEmbedder
from src.records.records import load_records, has_fm, write_fm, RECORDS_CSV
from src.training.efficiency import log_efficiency
from src.training.rf_pipeline import train_rf, eval_rf

_EVO2_MODELS = {"evo2_7b", "evo2_7b_base", "evo2_1b_base", "evo2_40b", "evo2_40b_base", "evo2_20b"}


def main():
    parser = argparse.ArgumentParser(
        description="Train RF on FM embeddings for all datasets"
    )
    parser.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True,
                       help="FM model: NTv3_650M_pre, hyenadna-medium-160k-seqlen-hf, evo2_7b_base, etc.")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--evo2-layer", type=str, default=None,
                        help="Evo2 layer name for embedding extraction (default: 'norm').")
    parser.add_argument("--evo2-max-length", type=int, default=None,
                        help="Max sequence length (bp) for Evo2 tokenization.")
    args = parser.parse_args()

    is_evo2 = args.model in _EVO2_MODELS
    if not is_evo2:
        try:
            validate_supported_model(args.model)
        except ValueError as exc:
            parser.error(str(exc))

    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets)

    print(f"\nFM: {args.model.split('/')[-1]}")
    print(f"Datasets : {n_total}")
    print(f"Results  : {RECORDS_CSV}\n")

    # Load FM model once; choose embedder based on model name
    model_lc = args.model.lower()
    if is_evo2:
        from src.embedders.evo2_embedder import Evo2Embedder
        fm = Evo2Embedder(
            model_name=args.model,
            cache_dir="cache/fm_embeddings",
            layer_name=args.evo2_layer,
            max_length=args.evo2_max_length,
        )
    elif "hyenadna" in model_lc:
        fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings")
    elif model_lc.startswith("google/enformer"):
        from src.embedders.enformer_embedder import EnformerEmbedder
        fm = EnformerEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")
    else:
        fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")

    records = load_records(RECORDS_CSV)
    completed = 0
    pbar = tqdm(all_datasets, desc="Progress", unit="ds", ncols=70)

    for ds in pbar:
        name = ds["name"]
        n_completed = completed + 1
        n_remaining = n_total - n_completed

        if has_fm(name, args.model, records):
            pbar.set_description(f"[{n_completed}/{n_total}] skip {name}")
            completed = n_completed
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        t0 = time.perf_counter()
        Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
        Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)
        t1 = time.perf_counter()
        log_efficiency(name, f"fm_{args.model.split('/')[-1]}", Y_train.shape[1], t1 - t0)

        rf = train_rf(Y_train, train_labels, n_classes)
        metrics = eval_rf(rf, Y_test, test_labels, n_classes)
        write_fm(name, args.model, metrics)
        records = load_records(RECORDS_CSV)

        completed = n_completed
        pbar.set_description(f"[{n_completed}/{n_total}] done | {n_remaining} remaining")

    print(f"\nDone! {n_total}/{n_total} datasets")


if __name__ == "__main__":
    main()
