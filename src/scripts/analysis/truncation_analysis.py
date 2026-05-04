"""
Truncation analysis for DNA Foundation Models across all classification datasets.

For each (model, dataset, split) triple, computes:
  - n_truncated:           sequences where token_count > max_length
  - truncation_rate:       n_truncated / n_sequences
  - mean_tokens:           mean token count (untruncated tokenization)
  - max_tokens:            max token count
  - fraction_retained_mean: mean(max_length / token_count) for truncated seqs; 1.0 if none

Tokenizers are loaded WITHOUT model weights (CPU only, no GPU required).

Usage:
    python3 src/scripts/truncation_analysis.py \\
        --data-root /raid/DATASETS/dna_foundation_benchmark/ \\
        --output results/classification/truncation_analysis.csv \\
        --plot   results/classification/truncation_analysis.png
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# max_length is in tokens.  For char-level tokenizers 1 token ≈ 1 bp.
# For DNABERT-2 BPE, empirically ~5 bp/token → 4096 tokens ≈ ~20,000 bp.
MODELS: dict[str, dict] = {
    "InstaDeepAI/NTv3_650M_pre": {
        "max_length": 32768,
        "trust_remote_code": True,
        "tokenizer_type": "hf",
    },
    "LongSafari/hyenadna-medium-160k-seqlen-hf": {
        "max_length": 32768,
        "trust_remote_code": True,
        "tokenizer_type": "hf",
    },
    "zhihan1996/DNABERT-2-117M": {
        "max_length": 4096,
        "trust_remote_code": True,
        "tokenizer_type": "hf",
    },
}

OUTPUT_COLS = [
    "model",
    "dataset",
    "split",
    "n_sequences",
    "n_truncated",
    "truncation_rate",
    "mean_tokens",
    "max_tokens",
    "fraction_retained_mean",
    "max_length",
]

# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def _load_hf_tokenizer(model_name: str, trust_remote_code: bool):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

def load_tokenizer(model_name: str, cfg: dict):
    return _load_hf_tokenizer(model_name, cfg["trust_remote_code"])


# ---------------------------------------------------------------------------
# Token counting  (truncation=False to get true pre-truncation counts)
# ---------------------------------------------------------------------------

def _count_tokens_hf(tokenizer, sequences: np.ndarray, batch_size: int = 256) -> np.ndarray:
    counts: list[int] = []
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size].tolist()
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,           # CRITICAL: measure true length
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        counts.extend(len(ids) for ids in enc["input_ids"])
    return np.array(counts, dtype=np.int64)


def count_tokens(tokenizer, sequences: np.ndarray, cfg: dict, batch_size: int) -> np.ndarray:
    return _count_tokens_hf(tokenizer, sequences, batch_size)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(token_counts: np.ndarray, max_length: int | None) -> dict:
    n = len(token_counts)
    if max_length is None:
        return {
            "n_sequences": n,
            "n_truncated": 0,
            "truncation_rate": 0.0,
            "mean_tokens": float(np.mean(token_counts)),
            "max_tokens": int(np.max(token_counts)),
            "fraction_retained_mean": 1.0,
        }

    truncated_mask = token_counts > max_length
    n_truncated = int(np.sum(truncated_mask))

    if n_truncated > 0:
        fraction_retained_mean = float(np.mean(max_length / token_counts[truncated_mask]))
    else:
        fraction_retained_mean = 1.0

    return {
        "n_sequences": n,
        "n_truncated": n_truncated,
        "truncation_rate": n_truncated / n,
        "mean_tokens": float(np.mean(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "fraction_retained_mean": fraction_retained_mean,
    }


# ---------------------------------------------------------------------------
# Incremental CSV helpers
# ---------------------------------------------------------------------------

def _load_existing(output_path: str) -> pd.DataFrame:
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    return pd.DataFrame(columns=OUTPUT_COLS)


def _already_done(existing: pd.DataFrame, model: str, dataset: str, split: str) -> bool:
    if existing.empty:
        return False
    return bool(
        ((existing["model"] == model) &
         (existing["dataset"] == dataset) &
         (existing["split"] == split)).any()
    )


def _append_row(row: dict, output_path: str, write_header: bool) -> None:
    pd.DataFrame([row]).to_csv(
        output_path, mode="a", header=write_header, index=False
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_plots(df: pd.DataFrame, records_csv: str, output_path: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    # ------------------------------------------------------------------
    # Load seq_len_max for dataset ordering
    # ------------------------------------------------------------------
    try:
        records = pd.read_csv(records_csv)
        len_map = dict(zip(records["dataset"], records["seq_len_max"]))
    except Exception:
        len_map = {}

    # Aggregate over train+test per (model, dataset)
    agg = (
        df.groupby(["model", "dataset"])
        .agg(
            truncation_rate=("truncation_rate", "max"),
            max_tokens=("max_tokens", "max"),
        )
        .reset_index()
    )

    model_short = {
        "InstaDeepAI/NTv3_650M_pre": "NTv3",
        "LongSafari/hyenadna-medium-160k-seqlen-hf": "HyenaDNA",
        "zhihan1996/DNABERT-2-117M": "DNABERT-2",
    }
    agg["model_short"] = agg["model"].map(model_short).fillna(agg["model"])

    datasets_ordered = (
        agg[["dataset"]]
        .drop_duplicates()
        .assign(seq_len_max=lambda d: d["dataset"].map(len_map).fillna(0))
        .sort_values("seq_len_max")["dataset"]
        .tolist()
    )

    model_limits = {
        model_short.get(m, m): cfg["max_length"]
        for m, cfg in MODELS.items()
        if cfg["max_length"] is not None
    }

    fig, axes = plt.subplots(
        1, 2, figsize=(18, max(8, len(datasets_ordered) * 0.3 + 2))
    )

    # ------------------------------------------------------------------
    # Panel 1: Heatmap — truncation_rate per (model × dataset)
    # ------------------------------------------------------------------
    pivot = agg.pivot(index="dataset", columns="model_short", values="truncation_rate")
    pivot = pivot.reindex(datasets_ordered)

    sns.heatmap(
        pivot,
        ax=axes[0],
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.3,
        linecolor="lightgrey",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "Truncation rate"},
    )
    axes[0].set_title("Truncation rate per (model, dataset)", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Dataset (sorted by seq_len_max)")
    axes[0].tick_params(axis="y", labelsize=7)

    # ------------------------------------------------------------------
    # Panel 2: Scatter — max_tokens vs model threshold
    # ------------------------------------------------------------------
    palette = sns.color_palette("tab10", n_colors=agg["model_short"].nunique())
    model_colors = dict(zip(agg["model_short"].unique(), palette))

    y_positions = {ds: i for i, ds in enumerate(datasets_ordered)}

    for model_s, grp in agg.groupby("model_short"):
        y = [y_positions[ds] for ds in grp["dataset"] if ds in y_positions]
        x = [grp.loc[grp["dataset"] == ds, "max_tokens"].values[0]
             for ds in grp["dataset"] if ds in y_positions]
        axes[1].scatter(x, y, label=model_s, color=model_colors[model_s],
                        alpha=0.7, s=20, zorder=3)

    for model_s, lim in model_limits.items():
        color = model_colors.get(model_s, "grey")
        axes[1].axvline(lim, color=color, linestyle="--", linewidth=1,
                        label=f"{model_s} limit ({lim:,})")

    axes[1].set_yticks(range(len(datasets_ordered)))
    axes[1].set_yticklabels(datasets_ordered, fontsize=7)
    axes[1].set_xlabel("Max token count (untruncated)")
    axes[1].set_title("Max tokens per dataset vs. model limits", fontsize=12)
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Truncation analysis for DNA foundation models (tokenizer-only, CPU)"
    )
    parser.add_argument(
        "--data-root",
        default="data/dna_foundation_benchmark/",
        help="Root directory containing dataset subdirectories",
    )
    parser.add_argument(
        "--output",
        default="results/classification/truncation_analysis.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plot",
        default="results/classification/truncation_analysis.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--records-csv",
        default="results/classification/records_rf.csv",
        help="Canonical RF results file used to sort datasets by seq_len_max in the plot",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Tokenizer batch size (HF models only)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip tokenization, only regenerate the plot from existing CSV",
    )
    args = parser.parse_args()

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # -----------------------------------------------------------------------
    # Plot-only shortcut
    # -----------------------------------------------------------------------
    if args.plot_only:
        df = pd.read_csv(output_path)
        make_plots(df, args.records_csv, args.plot)
        return

    # -----------------------------------------------------------------------
    # Discover datasets
    # -----------------------------------------------------------------------
    all_datasets = discover_datasets(args.data_root)
    print(f"Found {len(all_datasets)} datasets under {args.data_root}\n")

    # -----------------------------------------------------------------------
    # Load existing results (for resumability)
    # -----------------------------------------------------------------------
    existing = _load_existing(output_path)
    write_header = existing.empty
    n_done_start = len(existing)
    if n_done_start:
        print(f"Resuming: {n_done_start} rows already in {output_path}\n")

    # -----------------------------------------------------------------------
    # Determine which models to run
    # -----------------------------------------------------------------------
    active_models = MODELS

    failures: list[str] = []
    rows_written = 0

    for model_name, cfg in active_models.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (max_length={cfg['max_length']})")
        print(f"{'='*60}")

        tokenizer = load_tokenizer(model_name, cfg)
        if tokenizer is None:
            print(f"  [SKIP] Could not load tokenizer for {model_name}")
            continue

        for ds in tqdm(all_datasets, desc=model_name.split("/")[-1], unit="dataset"):
            ds_name = ds["name"]

            # Load sequences once per dataset (shared across splits)
            try:
                train_seqs, _, test_seqs, _ = load_dataset(ds["train_path"], ds["test_path"])
            except Exception as exc:
                msg = f"{model_name} | {ds_name} | load: {exc}"
                warnings.warn(msg)
                failures.append(msg)
                continue

            for split_name, seqs in [("train", train_seqs), ("test", test_seqs)]:
                if _already_done(existing, model_name, ds_name, split_name):
                    continue

                try:
                    token_counts = count_tokens(tokenizer, seqs, cfg, args.batch_size)
                    stats = compute_stats(token_counts, cfg["max_length"])
                    row = {
                        "model": model_name,
                        "dataset": ds_name,
                        "split": split_name,
                        "max_length": cfg["max_length"],
                        **stats,
                    }
                except Exception as exc:
                    msg = f"{model_name} | {ds_name} | {split_name}: {exc}"
                    warnings.warn(msg)
                    failures.append(msg)
                    # Write NaN row so failure is visible in CSV
                    row = {
                        "model": model_name,
                        "dataset": ds_name,
                        "split": split_name,
                        "max_length": cfg["max_length"],
                        "n_sequences": len(seqs),
                        "n_truncated": float("nan"),
                        "truncation_rate": float("nan"),
                        "mean_tokens": float("nan"),
                        "max_tokens": float("nan"),
                        "fraction_retained_mean": float("nan"),
                    }

                _append_row(row, output_path, write_header=write_header)
                write_header = False
                rows_written += 1

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Rows written : {rows_written}")
    print(f"Failures     : {len(failures)}")
    if failures:
        print("Failed entries:")
        for f in failures:
            print(f"  {f}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    df = pd.read_csv(output_path)
    make_plots(df, args.records_csv, args.plot)


if __name__ == "__main__":
    main()
