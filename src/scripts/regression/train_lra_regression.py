"""
RF benchmark on the two regression tasks of the Genomics Long-Range Benchmark.

Tasks:
  - cage_prediction      (hg38, ~38k train / ~1.9k test)
  - bulk_rna_expression  (hg38, ~20k train / test from gene_coordinates)

Metrics: R², MSE, Spearman.
Results written to results/regression_records.csv (same file used by the notebook).

Usage:
    # k-mer baseline only
    conda run -n cgr_bench python3 src/scripts/regression/train_lra_regression.py

    # add an FM model
    CUDA_VISIBLE_DEVICES=0 conda run -n cgr_bench python3 \\
        src/scripts/regression/train_lra_regression.py \\
        --model kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16

    CUDA_VISIBLE_DEVICES=0 conda run -n cgr_bench python3 \\
        src/scripts/regression/train_lra_regression.py \\
        --model evo2_1b_base
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.features.kmer_features import extract_kmer_features

LRA_DIR     = "/data/genomics-long-range-benchmark"
HG38_FA     = "/home/oem/Scrivania/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta"
RECORDS_CSV = "results/regression_records.csv"

_RF_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_features": ["sqrt"],
    "max_depth":    [20],
    "min_samples_split": [2],
}

_EVO2_MODELS = {"evo2_7b", "evo2_7b_base", "evo2_1b_base", "evo2_40b", "evo2_40b_base", "evo2_20b"}


# ── RF helpers ────────────────────────────────────────────────────────────────

def train_rf(X, y):
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestRegressor(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring="r2", cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X, y)
    return gs


def eval_rf(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "R2":       float(r2_score(y_test, y_pred)),
        "MSE":      float(np.mean((y_test - y_pred) ** 2)),
        "Spearman": float(spearmanr(y_test, y_pred).statistic),
    }


# ── records helpers ───────────────────────────────────────────────────────────

def _load(path=RECORDS_CSV):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["dataset"])


def _save(df, path=RECORDS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _has(df, dataset, col_prefix):
    if df.empty or dataset not in df["dataset"].values:
        return False
    row = df[df["dataset"] == dataset].iloc[0]
    col = f"{col_prefix}_R2"
    return col in df.columns and pd.notna(row.get(col, np.nan))


def _write(dataset, col_prefix, metrics, path=RECORDS_CSV):
    df = _load(path)
    if dataset not in df["dataset"].values:
        df = pd.concat([df, pd.DataFrame([{"dataset": dataset}])], ignore_index=True)
    idx = df[df["dataset"] == dataset].index[0]
    for m, v in metrics.items():
        df.loc[idx, f"{col_prefix}_{m}"] = v
    _save(df, path)


# ── genome helpers ────────────────────────────────────────────────────────────

def _build_chr_index(genome):
    idx = {}
    for name in genome.keys():
        bare = name.split()[0]
        idx[bare] = bare
        if bare.startswith("chr"):
            idx[bare[3:]] = bare
        else:
            idx["chr" + bare] = bare
    return idx


def _extract_seq(genome, chr_idx, chrom, start, end, seq_len=None):
    key = chr_idx.get(chrom) or chr_idx.get(chrom.replace("chr", ""))
    if key is None:
        return None
    chrom_len = len(genome[key])
    if seq_len is not None:
        mid = (start + end) // 2
        start = max(0, mid - seq_len // 2)
        end   = min(chrom_len, start + seq_len)
        start = max(0, end - seq_len)
    start = max(0, start)
    end   = min(chrom_len, end)
    seq = str(genome[key][start:end]).upper()
    return seq.replace("N", "A") if seq else None


# ── data loaders ──────────────────────────────────────────────────────────────

def load_cage_prediction(genome, seq_len=114_688, max_train=None):
    coords = pd.read_csv(os.path.join(LRA_DIR, "cage_prediction", "sequences_coordinates.csv"),
                         index_col=0)
    chr_idx = _build_chr_index(genome)

    def _load_targets(split, idx):
        lo = (idx // 1000) * 1000
        hi = lo + 999
        npz_path = os.path.join(LRA_DIR, "cage_prediction", "targets_subset",
                                f"targets-{split}-{lo}-{hi}.npz")
        if not os.path.exists(npz_path):
            return None
        data = np.load(npz_path)
        # Keys are like "target-train-0", "target-train-42", etc.
        key = f"target-{split}-{idx}"
        if key not in data.files:
            return None
        return data[key]

    def _extract(split_df, split_name):
        seqs, labels = [], []
        for _, row in split_df.iterrows():
            seq = _extract_seq(genome, chr_idx, row["chrom"],
                               int(row["start"]), int(row["stop"]), seq_len)
            if seq is None:
                continue
            tgt = _load_targets(split_name, int(row["npy_idx"]))
            if tgt is None:
                continue
            seqs.append(seq)
            labels.append(float(np.mean(tgt)))
        return np.array(seqs), np.array(labels, dtype=np.float32)

    train_df = coords[coords["split"] == "train"]
    test_df  = coords[coords["split"] == "test"]
    if max_train:
        train_df = train_df.iloc[:max_train]

    print("  Loading CAGE train sequences...")
    X_tr, y_tr = _extract(train_df, "train")
    print("  Loading CAGE test sequences...")
    X_te, y_te = _extract(test_df, "test")
    return X_tr, y_tr, X_te, y_te


def load_bulk_rna_expression(genome, seq_len=100_000, max_train=None):
    lra_bulk = os.path.join(LRA_DIR, "bulk_rna_expression")
    coords   = pd.read_csv(os.path.join(lra_bulk, "gene_coordinates.csv"))
    expr_df  = pd.read_csv(os.path.join(lra_bulk, "rna_expression_values.csv"))
    chr_idx  = _build_chr_index(genome)

    def _extract(split_df):
        seqs, labels = [], []
        for i, row in split_df.iterrows():
            chrom = row["chrom"]
            tss   = int(row["TSS"])
            seq = _extract_seq(genome, chr_idx, chrom,
                               tss - seq_len // 2, tss + seq_len // 2, seq_len)
            if seq is None:
                continue
            if i >= len(expr_df):
                continue
            label = float(np.mean(expr_df.iloc[i].values.astype(float)))
            seqs.append(seq)
            labels.append(label)
        return np.array(seqs), np.array(labels, dtype=np.float32)

    train_df = coords[coords["split"] == "train"]
    test_df  = coords[coords["split"] == "test"]
    if max_train:
        train_df = train_df.iloc[:max_train]

    print("  Loading bulk RNA train sequences...")
    X_tr, y_tr = _extract(train_df)
    print("  Loading bulk RNA test sequences...")
    X_te, y_te = _extract(test_df)
    return X_tr, y_tr, X_te, y_te


# ── embedder factory ──────────────────────────────────────────────────────────

def make_embedder(model_name, pooling="mean"):
    if model_name in _EVO2_MODELS or model_name.startswith("evo2_"):
        from src.embedders.evo2_embedder import Evo2Embedder
        return Evo2Embedder(model_name=model_name,
                            cache_dir="cache/lra_fm_embeddings")
    if model_name.lower().startswith("google/enformer"):
        from src.embedders.enformer_embedder import EnformerEmbedder
        return EnformerEmbedder(model_name=model_name, cache_dir="cache/lra_fm_embeddings")
    if "hyenadna" in model_name.lower():
        from src.embedders.hyena_embedder import HyenaEmbedder
        return HyenaEmbedder(model_name=model_name,
                             cache_dir="cache/lra_hyena_embeddings",
                             pooling=pooling)
    from src.embedders.fm_embedder import FMEmbedder
    return FMEmbedder(model_name=model_name,
                      cache_dir="cache/lra_fm_embeddings",
                      pooling=pooling)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Regression benchmark on LRA tasks (cage_prediction, bulk_rna_expression)"
    )
    parser.add_argument("--tasks", nargs="+",
                        default=["cage_prediction", "bulk_rna_expression"],
                        choices=["cage_prediction", "bulk_rna_expression"])
    parser.add_argument("--model", default=None,
                        help="FM model key, e.g. evo2_1b_base or "
                             "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--fm-batch-size", type=int, default=8)
    parser.add_argument("--pooling", choices=["mean", "max"], default="mean")
    parser.add_argument("--hg38", default=HG38_FA)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap training set size (for quick tests)")
    args = parser.parse_args()

    print(f"\nLRA Regression Benchmark")
    print(f"Tasks    : {args.tasks}")
    print(f"k-values : {args.k_values}")
    print(f"FM model : {args.model or 'none (k-mer only)'}")
    print(f"Results  : {RECORDS_CSV}\n")

    print(f"Loading hg38 from {args.hg38} ...")
    genome  = Fasta(args.hg38, one_based_attributes=False)
    chr_idx = _build_chr_index(genome)
    print("Genome loaded.\n")

    embedder   = None
    model_tag  = None
    if args.model:
        print(f"Loading FM model {args.model} ...")
        embedder  = make_embedder(args.model, pooling=args.pooling)
        model_tag = args.model.split("/")[-1]
        print("Model loaded.\n")

    loaders = {
        "cage_prediction":    lambda: load_cage_prediction(genome, max_train=args.max_train),
        "bulk_rna_expression": lambda: load_bulk_rna_expression(genome, max_train=args.max_train),
    }

    for task in args.tasks:
        print(f"\n{'='*60}\nTask: {task}\n{'='*60}")

        df = _load()
        pending_ks = [k for k in args.k_values if not _has(df, task, f"kmer_k{k}")]
        need_fm    = args.model and not _has(df, task, f"fm_{model_tag}")

        if not pending_ks and not need_fm:
            print("  [skip] all results already present")
            continue

        X_tr, y_tr, X_te, y_te = loaders[task]()
        print(f"  Train: {len(X_tr)} | Test: {len(X_te)}")

        if len(X_tr) == 0 or len(X_te) == 0:
            print("  SKIP: no sequences extracted — check FASTA path")
            continue

        # ── k-mer ──
        for k in pending_ks:
            col = f"kmer_k{k}"
            print(f"  k-mer k={k} ...")
            t0 = time.perf_counter()
            Xtr = extract_kmer_features(X_tr, k=k, n_workers=args.n_workers)
            Xte = extract_kmer_features(X_te, k=k, n_workers=args.n_workers)
            rf  = train_rf(Xtr, y_tr)
            m   = eval_rf(rf, Xte, y_te)
            print(f"  kmer k={k}: R2={m['R2']:.4f}  Spearman={m['Spearman']:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            _write(task, col, m)

        # ── FM ──
        if need_fm:
            col = f"fm_{model_tag}"
            print(f"  FM {model_tag} ...")
            t0 = time.perf_counter()
            Y_tr = embedder.embed_sequences(X_tr, task, "train", args.fm_batch_size)
            Y_te = embedder.embed_sequences(X_te, task, "test",  args.fm_batch_size)
            rf   = train_rf(Y_tr.astype(np.float32), y_tr)
            m    = eval_rf(rf, Y_te.astype(np.float32), y_te)
            print(f"  FM {model_tag}: R2={m['R2']:.4f}  Spearman={m['Spearman']:.4f}  "
                  f"({time.perf_counter()-t0:.0f}s)")
            _write(task, col, m)

    print(f"\nDone. Results in {RECORDS_CSV}")


if __name__ == "__main__":
    main()
