"""
Benchmark RF on concatenated features: best FM embedding + best k-mer features.

Selection of "best" uses existing records.csv (per-dataset max on a metric, default MCC).
WARNING: this selection uses test-set performance stored in records.csv;
use with caution for unbiased comparisons.

Usage:
  python3 src/fm_experiment/benchmark_concat_best.py \
    --data-root /data/genomic_bench/dna_foundation_benchmark/ \
    --metric MCC --kmer-include-multi \
    --fm-batch-size 16 --n-workers 8
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder
from src.embedders.hyena_embedder import HyenaEmbedder
from src.features.kmer_features import (
    extract_kmer_features,
    extract_multiscale_kmer_features,
)
from src.records.records import load_records, RF_METRICS
from src.training.rf_pipeline import train_rf, eval_rf

FM_TAG_TO_MODEL = {
    "NTv3_650M_pre": "InstaDeepAI/NTv3_650M_pre",
    "hyenadna-medium-160k-seqlen-hf": "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "DNABERT-2-117M": "zhihan1996/DNABERT-2-117M",
}


def best_kmer_for_dataset(row, metric, include_multi=False):
    # kmer_kX
    k_cols = [c for c in row.index if c.startswith("kmer_k") and c.endswith("_" + metric)]
    # multiscale
    if include_multi:
        k_cols += [c for c in row.index if c.startswith("kmer_multi") and c.endswith("_" + metric)]
    if not k_cols:
        return None

    vals = row[k_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    if vals.dropna().empty:
        return None

    best_col = vals.astype(float).idxmax()

    if best_col.startswith("kmer_k"):
        # parse kmer_k6_MCC
        k = int(best_col.split("_")[1].replace("k", ""))
        return ("k", k, best_col)

    # multiscale: kmer_multi_k4_5_6_MCC
    if best_col.startswith("kmer_multi_k"):
        tag = best_col.split("_")[2]  # "k4"? actually "k4"??
        # safer: split by "kmer_multi_k" and take until _metric
        tag = best_col.replace("kmer_multi_k", "").rsplit("_", 1)[0]
        k_values = tuple(int(x) for x in tag.split("_"))
        return ("multi", k_values, best_col)

    return None


def best_fm_for_dataset(row, metric):
    fm_cols = [c for c in row.index if c.startswith("fm_") and c.endswith("_" + metric)]
    if not fm_cols:
        return None

    vals = row[fm_cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    if vals.dropna().empty:
        return None

    best_col = vals.astype(float).idxmax()
    tag = best_col.replace("fm_", "").replace("_" + metric, "")
    model_name = FM_TAG_TO_MODEL.get(tag)
    return (tag, model_name, best_col)


def main():
    parser = argparse.ArgumentParser(
        description="RF on concatenated features: best FM + best k-mer"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--metric", default="MCC", choices=RF_METRICS)
    parser.add_argument("--kmer-include-multi", action="store_true",
                        help="Allow multiscale k-mer as candidate best")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Filter datasets (exact match)")
    parser.add_argument("--out-csv", default="results/concat/records_concat_best.csv")
    args = parser.parse_args()

    records = load_records(os.path.join("results", "classification", "records.csv"))
    if records.empty:
        raise RuntimeError("records.csv not found or empty")

    all_datasets = discover_datasets(args.data_root)
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]

    print(f"📊 Concat benchmark: {len(all_datasets)} dataset/i")
    print(f"   Metric for best selection: {args.metric}")
    print(f"   k-mer include multi: {args.kmer_include_multi}")
    print(f"   Output: {args.out_csv}\n")

    # cache embedders per model to avoid reload
    embedders = {}

    rows = []
    pbar = tqdm(all_datasets, desc="Concat", unit="ds", ncols=70)

    for ds in pbar:
        name = ds["name"]

        if name not in records.index:
            pbar.set_description(f"[skip] {name} (not in records)")
            continue

        row = records.loc[name]

        # choose best k-mer and FM
        best_k = best_kmer_for_dataset(row, args.metric, include_multi=args.kmer_include_multi)
        best_f = best_fm_for_dataset(row, args.metric)

        if best_k is None or best_f is None:
            pbar.set_description(f"[skip] {name} (missing best)" )
            continue

        k_type, k_val, k_col = best_k
        fm_tag, fm_model, fm_col = best_f

        if fm_model is None:
            pbar.set_description(f"[skip] {name} (unknown FM tag {fm_tag})")
            continue

        # load data
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # k-mer features
        if k_type == "k":
            grid_size = max(128, 2 ** k_val)
            X_train_k = extract_kmer_features(train_seqs, k=k_val, grid_size=grid_size,
                                              n_workers=args.n_workers)
            X_test_k = extract_kmer_features(test_seqs, k=k_val, grid_size=grid_size,
                                             n_workers=args.n_workers)
            k_desc = f"k={k_val}"
        else:
            # multiscale
            k_values = list(k_val)
            X_train_k = extract_multiscale_kmer_features(train_seqs, k_values=k_values,
                                                         grid_size=128, n_workers=args.n_workers)
            X_test_k = extract_multiscale_kmer_features(test_seqs, k_values=k_values,
                                                        grid_size=128, n_workers=args.n_workers)
            k_desc = f"multi={k_values}"

        # FM embeddings
        if fm_model not in embedders:
            if "hyenadna" in fm_model.lower():
                embedders[fm_model] = HyenaEmbedder(model_name=fm_model,
                                                    cache_dir="cache/hyena_embeddings")
            else:
                embedders[fm_model] = FMEmbedder(model_name=fm_model,
                                                cache_dir="cache/fm_embeddings")

        fm = embedders[fm_model]
        Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
        Y_test  = fm.embed_sequences(test_seqs, name, "test",  args.fm_batch_size)

        # concat
        X_train = np.concatenate([Y_train.astype(np.float32), X_train_k.astype(np.float32)], axis=1)
        X_test  = np.concatenate([Y_test.astype(np.float32),  X_test_k.astype(np.float32)], axis=1)

        # train/eval RF
        rf = train_rf(X_train, train_labels, n_classes)
        res = eval_rf(rf, X_test, test_labels, n_classes)

        rows.append({
            "dataset": name,
            "metric_select": args.metric,
            "kmer_choice": k_desc,
            "kmer_col": k_col,
            "fm_model": fm_model,
            "fm_col": fm_col,
            "feat_dim_fm": int(Y_train.shape[1]),
            "feat_dim_kmer": int(X_train_k.shape[1]),
            **res,
        })

    if not rows:
        print("No results produced. Check records and dataset paths.")
        return

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    import pandas as pd
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False)

    print(f"\n✅ Wrote {args.out_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
