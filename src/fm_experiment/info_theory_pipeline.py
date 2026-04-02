"""
Information-theory analysis pipeline for k-mer and FM embeddings.

Computes per-dataset measures such as entropy/perplexity of k-mer distributions,
mutual information with labels, and class-separation divergence (binary only).
Outputs a wide CSV: results/info_theory.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable

import numpy as np
from sklearn.feature_selection import mutual_info_classif

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.core.fcgr import batch_fcgr
from src.fm_experiment.kmer_features import kmer_from_grids

DEFAULT_FM_MODELS = [
    "InstaDeepAI/NTv3_650M_pre",
    "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "zhihan1996/DNABERT-2-117M",
]


# ----------------------------- helpers ------------------------------------

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


def _safe_mean(x: Iterable[float]) -> float | None:
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def _safe_median(x: Iterable[float]) -> float | None:
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def _entropy_rowwise(P: np.ndarray) -> np.ndarray:
    # P is (N, D) with L1 normalization
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    return -np.sum(P * np.log2(P), axis=1)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence (base 2)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return 0.5 * (kl_pm + kl_qm)


def _subsample(seqs, labels, max_samples: int):
    if max_samples <= 0 or len(seqs) <= max_samples:
        return seqs, labels
    # stratified subsample
    rng = np.random.RandomState(42)
    labels = np.asarray(labels)
    idx_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        n_take = max(1, int(round(max_samples * (len(idx) / len(labels)))))
        n_take = min(n_take, len(idx))
        idx_all.extend(rng.choice(idx, size=n_take, replace=False).tolist())
    idx_all = np.array(sorted(set(idx_all)))
    return seqs[idx_all], labels[idx_all]


def _subsample_idx(labels: np.ndarray, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or len(labels) <= max_samples:
        return np.arange(len(labels))
    rng = np.random.RandomState(42)
    idx_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        n_take = max(1, int(round(max_samples * (len(idx) / len(labels)))))
        n_take = min(n_take, len(idx))
        idx_all.extend(rng.choice(idx, size=n_take, replace=False).tolist())
    return np.array(sorted(set(idx_all)))


def _load_embeddings(model_name: str, dataset_name: str, split: str, kind: str) -> np.ndarray | None:
    """
    kind: "fm" or "tok"
    """
    safe_model = model_name.replace("/", "__")
    safe_ds = dataset_name.replace("/", "__").replace("\\", "__")

    if kind == "fm":
        candidates = [
            os.path.join("cache/fm_embeddings", safe_model, safe_ds, f"{split}.npz"),
            os.path.join("cache/fm_embeddings", safe_ds, f"{split}.npz"),
        ]
        if "hyenadna" in model_name.lower():
            candidates.append(os.path.join("cache/hyena_embeddings", safe_ds, f"{split}.npz"))
    else:
        candidates = [
            os.path.join("cache/tok_embeddings", safe_model, safe_ds, f"{split}.npz"),
        ]

    for path in candidates:
        if os.path.exists(path):
            return np.load(path)["embeddings"].astype(np.float32)
    return None


def _embedding_metrics(Y: np.ndarray, labels: np.ndarray, max_samples: int) -> dict:
    """
    Compute descriptive + info-theory-inspired metrics for embeddings.
    Uses a stratified subsample for stability/perf.
    """
    idx = _subsample_idx(labels, max_samples)
    Y = Y[idx]
    y = labels[idx]

    out = {}
    if Y.size == 0:
        return out

    # basic stats
    norms = np.linalg.norm(Y, axis=1)
    out["norm_mean"] = float(np.mean(norms))
    out["norm_std"] = float(np.std(norms))

    var = np.var(Y, axis=0, ddof=1)
    out["var_mean"] = float(np.mean(var))
    out["var_median"] = float(np.median(var))

    # Gaussian entropy per dimension (base 2)
    eps = 1e-12
    var_safe = np.clip(var, eps, None)
    ent = 0.5 * np.log2(2 * np.pi * np.e * var_safe)
    out["entropy_gauss_mean"] = float(np.mean(ent))

    # effective rank of covariance
    try:
        Yc = Y - np.mean(Y, axis=0, keepdims=True)
        cov = np.cov(Yc, rowvar=False)
        evals = np.linalg.eigvalsh(cov)
        evals = np.clip(evals, eps, None)
        p = evals / np.sum(evals)
        ent_spec = -np.sum(p * np.log(p))
        eff_rank = float(np.exp(ent_spec))
        out["effective_rank"] = eff_rank / Y.shape[1]
    except Exception:
        out["effective_rank"] = ""

    # anisotropy: mean cosine similarity of random pairs
    try:
        rng = np.random.RandomState(42)
        n = Y.shape[0]
        n_pairs = min(5000, n * (n - 1) // 2)
        if n_pairs > 0:
            i = rng.randint(0, n, size=n_pairs)
            j = rng.randint(0, n, size=n_pairs)
            Yi = Y[i]
            Yj = Y[j]
            Yi = Yi / np.linalg.norm(Yi, axis=1, keepdims=True)
            Yj = Yj / np.linalg.norm(Yj, axis=1, keepdims=True)
            cos = np.sum(Yi * Yj, axis=1)
            out["mean_cosine"] = float(np.mean(cos))
        else:
            out["mean_cosine"] = ""
    except Exception:
        out["mean_cosine"] = ""

    # class separability (Fisher ratio)
    try:
        classes = np.unique(y)
        overall = np.mean(Y, axis=0)
        sb = 0.0
        sw = 0.0
        for cls in classes:
            Xc = Y[y == cls]
            if Xc.size == 0:
                continue
            mu = np.mean(Xc, axis=0)
            sb += Xc.shape[0] * np.sum((mu - overall) ** 2)
            sw += np.sum((Xc - mu) ** 2)
        out["fisher_ratio"] = float(sb / sw) if sw > 0 else ""
    except Exception:
        out["fisher_ratio"] = ""

    return out


# ----------------------------- main ---------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Information-theory analysis for k-mer and FM embeddings"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per dataset for MI (0 = no limit)")
    parser.add_argument("--fm-models", nargs="*", default=DEFAULT_FM_MODELS)
    parser.add_argument("--include-tok", action="store_true",
                        help="Also compute MI for tokenizer embeddings if cache exists")
    parser.add_argument("--output", default="results/info_theory.csv")
    args = parser.parse_args()

    k_values = sorted(set(args.k_values))
    max_k = max(k_values)

    datasets = discover_datasets(args.data_root)

    # prepare CSV columns
    fieldnames = [
        "dataset",
        "n_train",
        "n_test",
        "n_classes",
    ]
    for k in k_values:
        fieldnames += [
            f"entropy_k{k}",
            f"perplexity_k{k}",
            f"mi_k{k}_mean",
            f"mi_k{k}_median",
            f"jsd_k{k}",
        ]
    for model in args.fm_models:
        tag = model.split("/")[-1]
        fieldnames += [
            f"mi_fm_{tag}_mean",
            f"mi_fm_{tag}_median",
            f"emb_fm_{tag}_dim",
            f"emb_fm_{tag}_norm_mean",
            f"emb_fm_{tag}_norm_std",
            f"emb_fm_{tag}_var_mean",
            f"emb_fm_{tag}_var_median",
            f"emb_fm_{tag}_entropy_gauss_mean",
            f"emb_fm_{tag}_effective_rank",
            f"emb_fm_{tag}_mean_cosine",
            f"emb_fm_{tag}_fisher_ratio",
        ]
    if args.include_tok:
        for model in args.fm_models:
            tag = model.split("/")[-1]
            fieldnames += [
                f"mi_tok_{tag}_mean",
                f"mi_tok_{tag}_median",
                f"emb_tok_{tag}_dim",
                f"emb_tok_{tag}_norm_mean",
                f"emb_tok_{tag}_norm_std",
                f"emb_tok_{tag}_var_mean",
                f"emb_tok_{tag}_var_median",
                f"emb_tok_{tag}_entropy_gauss_mean",
                f"emb_tok_{tag}_effective_rank",
                f"emb_tok_{tag}_mean_cosine",
                f"emb_tok_{tag}_fisher_ratio",
            ]

    rows_out = []

    for ds in datasets:
        name = ds["name"]
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # build row
        row = {
            "dataset": name,
            "n_train": len(train_seqs),
            "n_test": len(test_seqs),
            "n_classes": n_classes,
        }

        # subsample for MI
        mi_seqs, mi_labels = _subsample(np.array(train_seqs), np.array(train_labels), args.max_samples)

        # compute FCGR grids once for k-mer features
        grid_size = max(128, 2 ** max_k)
        grids = batch_fcgr(mi_seqs, grid_size=grid_size, n_workers=args.n_workers)

        for k in k_values:
            X = kmer_from_grids(grids, k, normalize="l1")  # (N, 4^k)

            # entropy / perplexity (mean over sequences)
            ent = _entropy_rowwise(X)
            row[f"entropy_k{k}"] = float(np.mean(ent))
            row[f"perplexity_k{k}"] = float(2 ** np.mean(ent))

            # mutual information with labels
            try:
                mi = mutual_info_classif(X, mi_labels, discrete_features=False, random_state=42)
                row[f"mi_k{k}_mean"] = float(np.mean(mi))
                row[f"mi_k{k}_median"] = float(np.median(mi))
            except Exception:
                row[f"mi_k{k}_mean"] = ""
                row[f"mi_k{k}_median"] = ""

            # class separation (binary only)
            if n_classes == 2:
                cls0 = X[mi_labels == 0]
                cls1 = X[mi_labels == 1]
                p0 = np.mean(cls0, axis=0)
                p1 = np.mean(cls1, axis=0)
                row[f"jsd_k{k}"] = float(_js_divergence(p0, p1))
            else:
                row[f"jsd_k{k}"] = ""

        # FM embeddings MI (train only, from cache)
        for model in args.fm_models:
            tag = model.split("/")[-1]
            Y = _load_embeddings(model, name, "train", kind="fm")
            if Y is None:
                row[f"mi_fm_{tag}_mean"] = ""
                row[f"mi_fm_{tag}_median"] = ""
                row[f"emb_fm_{tag}_dim"] = ""
                row[f"emb_fm_{tag}_norm_mean"] = ""
                row[f"emb_fm_{tag}_norm_std"] = ""
                row[f"emb_fm_{tag}_var_mean"] = ""
                row[f"emb_fm_{tag}_var_median"] = ""
                row[f"emb_fm_{tag}_entropy_gauss_mean"] = ""
                row[f"emb_fm_{tag}_effective_rank"] = ""
                row[f"emb_fm_{tag}_mean_cosine"] = ""
                row[f"emb_fm_{tag}_fisher_ratio"] = ""
                continue
            try:
                mi_fm = mutual_info_classif(Y, train_labels, discrete_features=False, random_state=42)
                row[f"mi_fm_{tag}_mean"] = float(np.mean(mi_fm))
                row[f"mi_fm_{tag}_median"] = float(np.median(mi_fm))
            except Exception:
                row[f"mi_fm_{tag}_mean"] = ""
                row[f"mi_fm_{tag}_median"] = ""

            # embedding metrics
            row[f"emb_fm_{tag}_dim"] = int(Y.shape[1])
            emb = _embedding_metrics(Y, np.array(train_labels), args.max_samples)
            row[f"emb_fm_{tag}_norm_mean"] = emb.get("norm_mean", "")
            row[f"emb_fm_{tag}_norm_std"] = emb.get("norm_std", "")
            row[f"emb_fm_{tag}_var_mean"] = emb.get("var_mean", "")
            row[f"emb_fm_{tag}_var_median"] = emb.get("var_median", "")
            row[f"emb_fm_{tag}_entropy_gauss_mean"] = emb.get("entropy_gauss_mean", "")
            row[f"emb_fm_{tag}_effective_rank"] = emb.get("effective_rank", "")
            row[f"emb_fm_{tag}_mean_cosine"] = emb.get("mean_cosine", "")
            row[f"emb_fm_{tag}_fisher_ratio"] = emb.get("fisher_ratio", "")

        # tokenizer embeddings MI (optional)
        if args.include_tok:
            for model in args.fm_models:
                tag = model.split("/")[-1]
                Y = _load_embeddings(model, name, "train", kind="tok")
                if Y is None:
                    row[f"mi_tok_{tag}_mean"] = ""
                    row[f"mi_tok_{tag}_median"] = ""
                    row[f"emb_tok_{tag}_dim"] = ""
                    row[f"emb_tok_{tag}_norm_mean"] = ""
                    row[f"emb_tok_{tag}_norm_std"] = ""
                    row[f"emb_tok_{tag}_var_mean"] = ""
                    row[f"emb_tok_{tag}_var_median"] = ""
                    row[f"emb_tok_{tag}_entropy_gauss_mean"] = ""
                    row[f"emb_tok_{tag}_effective_rank"] = ""
                    row[f"emb_tok_{tag}_mean_cosine"] = ""
                    row[f"emb_tok_{tag}_fisher_ratio"] = ""
                    continue
                try:
                    mi_tok = mutual_info_classif(Y, train_labels, discrete_features=False, random_state=42)
                    row[f"mi_tok_{tag}_mean"] = float(np.mean(mi_tok))
                    row[f"mi_tok_{tag}_median"] = float(np.median(mi_tok))
                except Exception:
                    row[f"mi_tok_{tag}_mean"] = ""
                    row[f"mi_tok_{tag}_median"] = ""

                # embedding metrics
                row[f"emb_tok_{tag}_dim"] = int(Y.shape[1])
                emb = _embedding_metrics(Y, np.array(train_labels), args.max_samples)
                row[f"emb_tok_{tag}_norm_mean"] = emb.get("norm_mean", "")
                row[f"emb_tok_{tag}_norm_std"] = emb.get("norm_std", "")
                row[f"emb_tok_{tag}_var_mean"] = emb.get("var_mean", "")
                row[f"emb_tok_{tag}_var_median"] = emb.get("var_median", "")
                row[f"emb_tok_{tag}_entropy_gauss_mean"] = emb.get("entropy_gauss_mean", "")
                row[f"emb_tok_{tag}_effective_rank"] = emb.get("effective_rank", "")
                row[f"emb_tok_{tag}_mean_cosine"] = emb.get("mean_cosine", "")
                row[f"emb_tok_{tag}_fisher_ratio"] = emb.get("fisher_ratio", "")

        rows_out.append(row)

    # write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)

    print(f"Wrote {args.output} ({len(rows_out)} datasets)")


if __name__ == "__main__":
    main()
