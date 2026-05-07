"""
Benchmark k-mer feature extraction throughput vs FM inference cost.

Measures wall-clock time for k-mer extraction (CPU) across k values and
sequence lengths. Outputs results/efficiency/efficiency_kmer.csv with
columns compatible with the existing efficiency.csv FM data.

Usage:
    conda run -n cgr_bench python3 src/scripts/efficiency/benchmark_kmer_timing.py
    conda run -n cgr_bench python3 src/scripts/efficiency/benchmark_kmer_timing.py \
        --n-seqs 1000 --n-reps 5 --n-workers 1
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.features.kmer_features import (
    extract_kmer_features,
    extract_multiscale_kmer_features,
)

OUTPUT_CSV = "results/efficiency/efficiency_kmer.csv"

SEQ_LENGTHS = [100, 250, 500, 1000, 2000]
N_SEQS = 1000
N_REPS = 3

METHODS = [
    ("kmer_k4", lambda seqs, nw: extract_kmer_features(seqs, k=4, n_workers=nw)),
    ("kmer_k5", lambda seqs, nw: extract_kmer_features(seqs, k=5, n_workers=nw)),
    ("kmer_k6", lambda seqs, nw: extract_kmer_features(seqs, k=6, n_workers=nw)),
    ("kmer_multiscale_456", lambda seqs, nw: extract_multiscale_kmer_features(seqs, k_values=[4, 5, 6], n_workers=nw)),
]

ALPHABET = list("ACGT")


def _random_seqs(n: int, length: int) -> np.ndarray:
    return np.array(
        ["".join(np.random.choice(ALPHABET, length)) for _ in range(n)]
    )


def _median_duration(fn, seqs, n_workers, n_reps):
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn(seqs, n_workers)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def run_benchmark(n_seqs: int, n_reps: int, n_workers: int) -> list[dict]:
    rows = []
    for seq_len in SEQ_LENGTHS:
        print(f"  seq_len={seq_len} bp ...", flush=True)
        seqs = _random_seqs(n_seqs, seq_len)
        for method_name, fn in METHODS:
            duration = _median_duration(fn, seqs, n_workers, n_reps)
            throughput = n_seqs / duration
            rows.append({
                "method": method_name,
                "seq_len": seq_len,
                "n_seqs": n_seqs,
                "n_reps": n_reps,
                "n_workers": n_workers,
                "duration_sec": round(duration, 6),
                "throughput_seqs_per_sec": round(throughput, 2),
            })
            print(
                f"    {method_name}: {duration:.3f}s median ({throughput:.0f} seqs/s)",
                flush=True,
            )
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "method", "seq_len", "n_seqs", "n_reps", "n_workers",
        "duration_sec", "throughput_seqs_per_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark k-mer feature extraction timing")
    parser.add_argument("--n-seqs", type=int, default=N_SEQS,
                        help=f"Synthetic sequences per run (default: {N_SEQS})")
    parser.add_argument("--n-reps", type=int, default=N_REPS,
                        help=f"Repetitions per (method, seq_len); median reported (default: {N_REPS})")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers for FCGR computation (default: 1)")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    np.random.seed(42)
    print(
        f"Benchmarking k-mer extraction: {args.n_seqs} seqs x {args.n_reps} reps, "
        f"seq_lens={SEQ_LENGTHS}, n_workers={args.n_workers}"
    )

    rows = run_benchmark(args.n_seqs, args.n_reps, args.n_workers)
    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
