"""
Unified benchmark runner for all representations.

This script dispatches to the specific training scripts with consistent defaults.
"""

import argparse
import os
import sys
import subprocess


def _script_path(name: str) -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, name)


def _run(cmd):
    print(f"\n▶ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmark scripts for CGR multi-resolution vs FM"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--methods", default="all",
                       help="Comma-separated: kmer,multiscale,wms,quadtree,wavelet,fm,tok")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6, 7, 8])
    parser.add_argument("--multiscale-k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--wms-k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--wms-weighting", default="grid")
    parser.add_argument("--wavelet-levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--qt-max-depths", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--qt-p-thresholds", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--qt-min-counts", nargs="+", type=int, default=[0, 4, 8])
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--model", default="InstaDeepAI/NTv3_650M_pre")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--tok-batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.methods == "all":
        methods = {"kmer", "multiscale", "wms", "quadtree", "wavelet", "fm", "tok"}
    else:
        methods = {m.strip() for m in args.methods.split(",") if m.strip()}

    py = sys.executable

    if "kmer" in methods:
        _run([
            py, _script_path("train_kmer_rf.py"),
            "--data-root", args.data_root,
            "--k-values", *[str(k) for k in args.k_values],
            "--n-workers", str(args.n_workers),
        ])

    if "multiscale" in methods:
        _run([
            py, _script_path("train_multiscale_kmer_rf.py"),
            "--data-root", args.data_root,
            "--k-values", *[str(k) for k in args.multiscale_k_values],
            "--n-workers", str(args.n_workers),
        ])

    if "wms" in methods:
        _run([
            py, _script_path("train_weighted_multiscale_rf.py"),
            "--data-root", args.data_root,
            "--k-values", *[str(k) for k in args.wms_k_values],
            "--weighting", args.wms_weighting,
            "--n-workers", str(args.n_workers),
        ])

    if "quadtree" in methods:
        _run([
            py, _script_path("train_quadtree_rf.py"),
            "--data-root", args.data_root,
            "--max-depths", *[str(d) for d in args.qt_max_depths],
            "--p-thresholds", *[str(p) for p in args.qt_p_thresholds],
            "--min-counts", *[str(m) for m in args.qt_min_counts],
            "--n-workers", str(args.n_workers),
        ])

    if "wavelet" in methods:
        _run([
            py, _script_path("train_wavelet_rf.py"),
            "--data-root", args.data_root,
            "--levels", *[str(l) for l in args.wavelet_levels],
            "--n-workers", str(args.n_workers),
        ])

    if "fm" in methods:
        _run([
            py, _script_path("train_fm_rf.py"),
            "--data-root", args.data_root,
            "--model", args.model,
            "--fm-batch-size", str(args.fm_batch_size),
        ])

    if "tok" in methods:
        _run([
            py, _script_path("train_tok_rf.py"),
            "--data-root", args.data_root,
            "--model", args.model,
            "--batch-size", str(args.tok_batch_size),
        ])


if __name__ == "__main__":
    main()
