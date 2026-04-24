"""
Motif-level analysis on NTv3 residual component for splice datasets.

Workflow:
1) Fit ridge mapping k-mer -> NTv3 embedding on train split.
2) Compute residuals on test split.
3) For known splice motifs, perturb matched motif spans and recompute residuals.
4) Report average residual-norm drop and motif overlap statistics.

This script is intentionally conservative on shared CPU resources.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 src/scripts/analysis/analyze_splice_residual_motifs.py \\
        --model InstaDeepAI/NTv3_650M_pre --k 6 --n-workers 1 --fm-batch-size 8
    Output: results/exploratory/splice_residual_motif_overlap.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder
from src.features.kmer_features import extract_kmer_features
from src.training.ridge_mapping import fit_and_evaluate


@dataclass
class MotifSpec:
    name: str
    pattern: str


MOTIFS = [
    MotifSpec("GT_AG_donor_acceptor", r"GT[ACGT]{8,80}AG"),
    MotifSpec("branch_point_TACTAAC", r"TACTAAC"),
    MotifSpec("polypyrimidine_tract", r"[CT]{6,}"),
]


def _find_first_match(seq: str, pattern: str) -> tuple[int, int] | None:
    m = re.search(pattern, seq)
    if m is None:
        return None
    return m.start(), m.end()


def _perturb_span(seq: str, span: tuple[int, int]) -> str:
    i, j = span
    return seq[:i] + ("A" * max(1, j - i)) + seq[j:]


def _embed_no_cache(fm: FMEmbedder, seqs: list[str], batch_size: int) -> np.ndarray:
    out: list[np.ndarray] = []
    pad_id = fm.tokenizer.pad_token_id
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start : start + batch_size]
        tokens = fm._tokenize(batch)
        input_ids = tokens["input_ids"]
        attn_mask = tokens.get("attention_mask")
        hidden = fm._forward(input_ids, attention_mask=attn_mask)

        input_ids_dev = input_ids.to(fm.device)
        if attn_mask is not None:
            mask = attn_mask.to(fm.device).unsqueeze(-1).to(hidden.dtype)
        elif pad_id is not None:
            mask = (input_ids_dev != pad_id).unsqueeze(-1).to(hidden.dtype)
        else:
            mask = torch.ones_like(hidden[:, :, :1], dtype=hidden.dtype)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out.append(pooled.cpu().float().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Splice motif analysis on NTv3 residuals")
    p.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    p.add_argument("--model", default="InstaDeepAI/NTv3_650M_pre")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--fm-batch-size", type=int, default=16)
    p.add_argument("--max-seqs-per-motif", type=int, default=128)
    p.add_argument("--output-csv", default="results/exploratory/splice_residual_motif_overlap.csv")
    p.add_argument("--output-fig-dir", default="results/figures_pdf")
    args = p.parse_args()

    if args.model.lower().startswith("evo2"):
        raise ValueError("Evo2 is excluded from this plan")

    all_ds = discover_datasets(args.data_root)
    splice_ds = [d for d in all_ds if d["name"].startswith("splice/")]

    if len(splice_ds) != 4:
        print(f"Warning: expected 4 splice datasets, found {len(splice_ds)}")

    fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings", pooling="mean")

    rows: list[dict] = []
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.output_fig_dir, exist_ok=True)

    for ds in splice_ds:
        name = ds["name"]
        train_seqs, train_labels, test_seqs, _ = load_dataset(ds["train_path"], ds["test_path"])
        train_seqs = np.asarray(train_seqs, dtype=object)
        test_seqs = np.asarray(test_seqs, dtype=object)

        y_train = fm.embed_sequences(train_seqs, name, "train", batch_size=args.fm_batch_size).astype(np.float32)
        x_train = extract_kmer_features(
            train_seqs,
            k=args.k,
            grid_size=max(128, 2 ** args.k),
            n_workers=args.n_workers,
        ).astype(np.float32)

        # Fit linear mapper once per dataset, then analyze perturbation in residual space.
        mapper = fit_and_evaluate(x_train, y_train, x_train, y_train)
        ridge = mapper["model"]

        test_list = [str(x) for x in test_seqs.tolist()]

        for motif in MOTIFS:
            matched_idx = []
            spans: list[tuple[int, int] | None] = []
            for i, seq in enumerate(test_list):
                span = _find_first_match(seq, motif.pattern)
                spans.append(span)
                if span is not None:
                    matched_idx.append(i)

            overlap = len(matched_idx) / max(1, len(test_list))
            if not matched_idx:
                rows.append(
                    {
                        "dataset": name,
                        "motif": motif.name,
                        "n_matched": 0,
                        "n_total_test": len(test_list),
                        "overlap_rate": overlap,
                        "mean_delta_resid_l2": np.nan,
                        "std_delta_resid_l2": np.nan,
                    }
                )
                continue

            sel = matched_idx[: args.max_seqs_per_motif]
            orig = [test_list[i] for i in sel]
            pert = [_perturb_span(test_list[i], spans[i]) for i in sel if spans[i] is not None]

            y_orig = _embed_no_cache(fm, orig, batch_size=args.fm_batch_size)
            y_pert = _embed_no_cache(fm, pert, batch_size=args.fm_batch_size)

            x_orig = extract_kmer_features(
                np.asarray(orig, dtype=object),
                k=args.k,
                grid_size=max(128, 2 ** args.k),
                n_workers=args.n_workers,
            ).astype(np.float32)
            x_pert = extract_kmer_features(
                np.asarray(pert, dtype=object),
                k=args.k,
                grid_size=max(128, 2 ** args.k),
                n_workers=args.n_workers,
            ).astype(np.float32)

            resid_orig = y_orig - ridge.predict(x_orig).astype(np.float32)
            resid_pert = y_pert - ridge.predict(x_pert).astype(np.float32)

            l2_orig = np.linalg.norm(resid_orig, axis=1)
            l2_pert = np.linalg.norm(resid_pert, axis=1)
            delta = l2_orig - l2_pert

            rows.append(
                {
                    "dataset": name,
                    "motif": motif.name,
                    "n_matched": len(sel),
                    "n_total_test": len(test_list),
                    "overlap_rate": overlap,
                    "mean_delta_resid_l2": float(np.mean(delta)),
                    "std_delta_resid_l2": float(np.std(delta)),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)

    for ds_name_raw, sub in out.groupby("dataset"):
        ds_name = str(ds_name_raw)
        plt.figure(figsize=(7.5, 3.8))
        x = np.arange(len(sub))
        means = sub["mean_delta_resid_l2"].to_numpy(dtype=float)
        errs = sub["std_delta_resid_l2"].to_numpy(dtype=float)
        labels = sub["motif"].tolist()

        plt.bar(x, means, yerr=errs, capsize=4)
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("Delta residual L2 (orig - perturbed)")
        plt.title(ds_name)
        plt.tight_layout()

        safe = ds_name.replace("/", "__")
        plt.savefig(os.path.join(args.output_fig_dir, f"splice_motif_{safe}.pdf"))
        plt.close()

    print(f"Saved motif table: {args.output_csv}")
    print(f"Saved motif figures in: {args.output_fig_dir}")


if __name__ == "__main__":
    main()
