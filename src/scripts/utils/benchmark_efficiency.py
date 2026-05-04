"""
Benchmark computational efficiency of FM vs k-mer methods on synthetic sequences.

GFLOPs depend only on sequence length, not on the specific dataset or task.
This script measures on random synthetic DNA sequences at fixed lengths,
producing a clean length-vs-cost profile suitable for plotting.

Output: results/efficiency/efficiency_gpu_parallel.csv
        One row per (method, seq_len).

Usage:
    CUDA_VISIBLE_DEVICES=3 python3 src/scripts/utils/benchmark_efficiency.py
    CUDA_VISIBLE_DEVICES=3 python3 src/scripts/utils/benchmark_efficiency.py --seq-lens 100 500 1000 --n-seqs 64
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_CSV = "results/efficiency/efficiency_gpu_parallel.csv"
OUTPUT_COLS = [
    "method", "seq_len", "n_seqs", "params_M",
    "duration_sec", "GFLOPS_per_seq", "GFLOPS_total", "GFLOPS_theory_per_seq",
]

DNA_ALPHABET = list("ACGT")

FM_METHODS: dict[str, str] = {
    "fm_NTv3_650M_pre":                   "InstaDeepAI/NTv3_650M_pre",
    "fm_hyenadna-medium-160k-seqlen-hf":  "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "fm_DNABERT-2-117M":                  "zhihan1996/DNABERT-2-117M",
}

KMER_METHODS = ["kmer_k6"]
ALL_METHODS = list(FM_METHODS.keys()) + KMER_METHODS
DEFAULT_SEQ_LENS = [100, 250, 500, 1000, 2000]

# ---------------------------------------------------------------------------
# Synthetic sequence generation
# ---------------------------------------------------------------------------

def make_synthetic_seqs(seq_len: int, n: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(DNA_ALPHABET, size=seq_len)) for _ in range(n)]

# ---------------------------------------------------------------------------
# GFLOP estimation
# ---------------------------------------------------------------------------

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def gflops_profiler(model: torch.nn.Module, input_ids: torch.Tensor,
                    model_name: str, n_params: int) -> tuple[float, float]:
    """Return (empirical_gflops_per_seq, theory_gflops_per_seq)."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    gflops_theory = 2.0 * n_params * seq_len / 1e9

    activities = [torch.profiler.ProfilerActivity.CPU]
    if input_ids.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            with torch.no_grad():
                model(input_ids)
        total_flops = sum(
            e.flops for e in prof.key_averages() if hasattr(e, "flops") and e.flops
        )
    except Exception as exc:
        warnings.warn(f"Profiler failed for {model_name}: {exc}")
        total_flops = 0

    if total_flops == 0:
        return gflops_theory, gflops_theory

    return total_flops / batch_size / 1e9, gflops_theory


def gflops_kmer(seq_len: int, k: int) -> float:
    """Theoretical GFLOPs for FCGR + k-mer pooling on a single sequence."""
    grid_size = max(128, 2 ** k)
    return (5.0 * seq_len + grid_size * grid_size) / 1e9

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_efficiency(path: str = OUTPUT_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=OUTPUT_COLS)


def has_record(df: pd.DataFrame, method: str, seq_len: int) -> bool:
    if df.empty:
        return False
    return bool(((df["method"] == method) & (df["seq_len"] == seq_len)).any())


def append_row(row: dict, path: str = OUTPUT_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    pd.DataFrame([row])[OUTPUT_COLS].to_csv(
        path, mode="a", header=write_header, index=False
    )

# ---------------------------------------------------------------------------
# FM benchmark
# ---------------------------------------------------------------------------

def benchmark_fm(model_name: str, method_key: str, seq_lens: list[int],
                 n_seqs: int, eff_df: pd.DataFrame,
                 output_path: str) -> pd.DataFrame:
    from src.embedders.fm_embedder import FMEmbedder
    from src.embedders.hyena_embedder import HyenaEmbedder

    model_lc = model_name.lower()
    if "hyenadna" in model_lc:
        fm = HyenaEmbedder(model_name=model_name, cache_dir="cache/hyena_embeddings")
    else:
        fm = FMEmbedder(model_name=model_name, cache_dir="cache/fm_embeddings")

    n_params = _count_params(fm.model)
    params_M = round(n_params / 1e6, 2)

    for seq_len in tqdm(seq_lens, desc=f"[{method_key}]", unit="len"):
        if has_record(eff_df, method_key, seq_len):
            continue

        seqs = make_synthetic_seqs(seq_len, n_seqs)

        try:
            if "hyenadna" in model_lc:
                tokens = fm.tokenizer(
                    seqs,
                    add_special_tokens=False,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
            else:
                tokens = fm._tokenize(seqs)

            # Support both BatchEncoding and raw Tensor tokenization outputs.
            if isinstance(tokens, dict) or hasattr(tokens, "keys"):
                input_ids = tokens["input_ids"].to(fm.device)
                attention_mask = tokens.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(fm.device)
            else:
                input_ids = tokens.to(fm.device)
                attention_mask = None
        except Exception as exc:
            warnings.warn(f"Tokenization failed for len={seq_len}: {exc}")
            continue

        # Wall-clock measured on the same synthetic batch.
        start = time.perf_counter()
        with torch.no_grad():
            fm.model(input_ids, attention_mask=attention_mask)
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize()
        duration_sec = time.perf_counter() - start

        gflops_emp, gflops_theory = gflops_profiler(fm.model, input_ids, model_name, n_params)
        gflops_total = gflops_emp * n_seqs

        append_row({"method": method_key, "seq_len": seq_len, "n_seqs": n_seqs,
                    "params_M": params_M,
                    "duration_sec": round(duration_sec, 6),
                    "GFLOPS_per_seq": round(gflops_emp, 6),
                    "GFLOPS_total": round(gflops_total, 6),
                    "GFLOPS_theory_per_seq": round(gflops_theory, 6)},
                   output_path)
        eff_df = load_efficiency(output_path)

    del fm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return eff_df

# ---------------------------------------------------------------------------
# K-mer benchmark
# ---------------------------------------------------------------------------

def benchmark_kmer(k: int, seq_lens: list[int], n_seqs: int,
                   eff_df: pd.DataFrame, output_path: str,
                   n_workers: int) -> pd.DataFrame:
    method_key = f"kmer_k{k}"

    from src.features.kmer_features import extract_kmer_features

    for seq_len in tqdm(seq_lens, desc=f"[{method_key}]", unit="len"):
        if has_record(eff_df, method_key, seq_len):
            continue

        seqs = make_synthetic_seqs(seq_len, n_seqs)
        start = time.perf_counter()
        _ = extract_kmer_features(
            np.asarray(seqs, dtype=object),
            k=k,
            grid_size=max(128, 2 ** k),
            n_workers=n_workers,
        )
        duration_sec = time.perf_counter() - start

        gflops_theory = gflops_kmer(seq_len, k)
        gflops_total = gflops_theory * n_seqs
        append_row({"method": method_key, "seq_len": seq_len, "n_seqs": n_seqs,
                    "params_M": "", "duration_sec": round(duration_sec, 6),
                    "GFLOPS_per_seq": round(gflops_theory, 9),
                    "GFLOPS_total": round(gflops_total, 9),
                    "GFLOPS_theory_per_seq": round(gflops_theory, 9)},
                   output_path)
        eff_df = load_efficiency(output_path)

    return eff_df

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GFLOPs on synthetic sequences at fixed lengths"
    )
    parser.add_argument("--seq-lens", nargs="+", type=int, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--n-seqs", type=int, default=32,
                        help="Batch size for profiling (default: 32)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Workers for k-mer timing (conservative default: 1)")
    parser.add_argument("--methods", default="all",
                        help=f"Comma-separated or 'all'. Available: {', '.join(ALL_METHODS)}")
    parser.add_argument("--output", default=OUTPUT_CSV)
    args = parser.parse_args()

    methods = ALL_METHODS if args.methods == "all" else [m.strip() for m in args.methods.split(",")]

    print(f"\nSeq lengths : {args.seq_lens}")
    print(f"N sequences : {args.n_seqs}")
    print(f"Methods     : {methods}")
    print(f"Output      : {args.output}\n")

    eff_df = load_efficiency(args.output)

    for method_key, model_name in FM_METHODS.items():
        if method_key not in methods:
            continue
        print(f"\n{'='*55}\n  FM: {method_key}\n{'='*55}")
        eff_df = benchmark_fm(model_name, method_key, args.seq_lens,
                              args.n_seqs, eff_df, args.output)

    for method_key in KMER_METHODS:
        if method_key not in methods:
            continue
        k = int(method_key.split("_k")[1])
        print(f"\n{'='*55}\n  K-mer: {method_key}\n{'='*55}")
        eff_df = benchmark_kmer(
            k, args.seq_lens, args.n_seqs, eff_df, args.output, args.n_workers
        )

    print(f"\nDone! Results saved to: {args.output}")
    print(load_efficiency(args.output).to_string(index=False))


if __name__ == "__main__":
    main()
