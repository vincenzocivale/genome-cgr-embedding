"""
Rebuttal E0 — Step 1: sequence-level duplication audit.

Reviewer e1HN asks whether the benchmark splits contain overlapping / duplicated /
near-duplicate sequences that would leak information from train into test.

The classification benchmark (`data/dna_foundation_benchmark/`) stores only
`sequence` / `label` (no genomic coordinates), so leakage here is detectable at the
*sequence* level. For every dataset we quantify:

  * exact duplicates within train and within test,
  * exact train/test cross-contamination  (a test sequence present verbatim in train),
  * reverse-complement cross-contamination (revcomp of a test sequence present in train),
  * near-duplicate cross-contamination     (a test sequence with k-mer Jaccard >= 0.8
    to some train sequence), detected with a numpy MinHash-LSH candidate generator
    followed by exact Jaccard confirmation.

Output: results/rebuttal/E0_data_leakage/sequence_duplication_per_dataset.csv

Usage:
    python3 src/rebuttal/e0_leakage_audit/audit_sequence_duplication.py \
        --data-root data/dna_foundation_benchmark/
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.data.loader import discover_datasets, load_dataset

OUT_CSV = "results/rebuttal/E0_data_leakage/sequence_duplication_per_dataset.csv"

# N-safe reverse complement (repo's kmer_features._reverse_complement lacks N/IUPAC).
# Sequences are uppercased by load_dataset; map A/C/G/T to complements and every
# other character (N and any IUPAC ambiguity code) to N.
_COMP_TABLE = {i: ord("N") for i in range(256)}
for _a, _b in zip("ACGT", "TGCA"):
    _COMP_TABLE[ord(_a)] = ord(_b)
    _COMP_TABLE[ord(_a.lower())] = ord(_b)


def _reverse_complement(seq: str) -> str:
    """Reverse complement; non-ACGT bases (N, IUPAC codes) map to N."""
    return seq.translate(_COMP_TABLE)[::-1]

# Near-duplicate detection parameters.
SHINGLE_K = 8            # k-mer size for shingling
NUM_PERM = 64            # MinHash permutations
LSH_BANDS = 16           # bands  (b)
LSH_ROWS = NUM_PERM // LSH_BANDS   # rows per band (r)  -> b*r == NUM_PERM
JACCARD_THRESHOLD = 0.8  # confirmed near-duplicate cutoff
# Mersenne prime 2**31 - 1. Shingle hashes and permutation coefficients are all
# < 2**31, so products stay < 2**62 and fit in uint64 (no big-int / object math).
_MPRIME = np.uint64((1 << 31) - 1)


# --------------------------------------------------------------------------- #
# Shingling + MinHash  (fast uint64 arithmetic)
# --------------------------------------------------------------------------- #
_HASH_BASE = 131  # base for polynomial k-mer hashing
# Precompute B**j mod prime for j = 0..SHINGLE_K-1 (values < 2**31).
_POWERS = (np.array([pow(_HASH_BASE, j, int(_MPRIME)) for j in range(SHINGLE_K)],
                    dtype=np.uint64))


def _shingle_hashes(seq: str, k: int = SHINGLE_K) -> np.ndarray:
    """
    Return the unique set of 31-bit polynomial-hashed k-mer shingles of `seq`.

    Uses a fully vectorized sliding-window polynomial hash: each window of length k
    is hashed as sum_j code[j] * B**j (mod prime). Products (<=255 * 2**31 < 2**39)
    and their length-k sum (< 2**42) fit in uint64, so no big-int math is needed.
    """
    codes = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8).astype(np.uint64)
    if codes.size <= k:
        # short sequence: single shingle over the whole (padded) sequence
        h = (codes * _POWERS[:codes.size]).sum() % _MPRIME
        return np.asarray([h], dtype=np.uint64)
    windows = np.lib.stride_tricks.sliding_window_view(codes, k)   # (nw, k) uint64
    h = (windows * _POWERS[None, :]).sum(axis=1) % _MPRIME
    return np.unique(h)


def _make_permutations(num_perm: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, (1 << 31) - 1, size=num_perm, dtype=np.uint64)
    return a, b


def _minhash_signature(shingle_hashes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the (num_perm,) MinHash signature for one sequence's shingle set."""
    if shingle_hashes.size == 0:
        return np.zeros(a.shape[0], dtype=np.uint64)
    # (num_perm, n_shingles): (a[:,None]*x[None,:] + b[:,None]) mod prime, all uint64.
    hashed = (a[:, None] * shingle_hashes[None, :] + b[:, None]) % _MPRIME
    return hashed.min(axis=1).astype(np.uint64)


def _signatures_for(seqs, a, b):
    """Compute MinHash signatures and keep shingle sets (for exact Jaccard)."""
    sigs = np.empty((len(seqs), a.shape[0]), dtype=np.uint64)
    shingle_sets = []
    for i, s in enumerate(seqs):
        sh = _shingle_hashes(s)
        shingle_sets.append(set(sh.tolist()))
        sigs[i] = _minhash_signature(sh, a, b)
    return sigs, shingle_sets


def _lsh_buckets(sigs: np.ndarray):
    """Map band-bucket key -> list of row indices (banded LSH over signatures)."""
    from collections import defaultdict
    buckets = defaultdict(list)
    n, p = sigs.shape
    for band in range(LSH_BANDS):
        cols = sigs[:, band * LSH_ROWS:(band + 1) * LSH_ROWS]
        for i in range(n):
            key = (band,) + tuple(cols[i].tolist())
            buckets[key].append(i)
    return buckets


def _exact_jaccard(s1: set, s2: set) -> float:
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    return inter / (len(s1) + len(s2) - inter)


def _neardup_cross_count(train_seqs, test_seqs, max_seqs=60000, seed=0):
    """
    Count test sequences that have a near-duplicate (Jaccard >= threshold) in train.

    If either split exceeds `max_seqs`, a deterministic random subsample of that size
    is used and the count is reported as a rate scaled to the full split; the sampled
    sizes are returned so the caller can annotate the approximation.
    """
    rng = np.random.default_rng(seed)
    tr_idx = np.arange(len(train_seqs))
    te_idx = np.arange(len(test_seqs))
    tr_sampled = te_sampled = False
    if len(train_seqs) > max_seqs:
        tr_idx = rng.choice(len(train_seqs), max_seqs, replace=False); tr_sampled = True
    if len(test_seqs) > max_seqs:
        te_idx = rng.choice(len(test_seqs), max_seqs, replace=False); te_sampled = True
    tr = [train_seqs[i] for i in tr_idx]
    te = [test_seqs[i] for i in te_idx]

    a, b = _make_permutations(NUM_PERM, seed=seed)
    tr_sig, tr_shg = _signatures_for(tr, a, b)
    te_sig, te_shg = _signatures_for(te, a, b)
    tr_buckets = _lsh_buckets(tr_sig)

    near_hits = 0
    for j in range(len(te)):
        cand = set()
        for band in range(LSH_BANDS):
            cols = te_sig[j, band * LSH_ROWS:(band + 1) * LSH_ROWS]
            key = (band,) + tuple(cols.tolist())
            if key in tr_buckets:
                cand.update(tr_buckets[key])
        hit = False
        for ci in cand:
            if _exact_jaccard(te_shg[j], tr_shg[ci]) >= JACCARD_THRESHOLD:
                hit = True
                break
        if hit:
            near_hits += 1

    n_te_eval = len(te)
    rate = near_hits / max(1, n_te_eval)
    return {
        "neardup_cross_count_sampled": int(near_hits),
        "neardup_cross_pct": 100.0 * rate,
        "neardup_n_test_eval": int(n_te_eval),
        "neardup_n_train_eval": int(len(tr)),
        "neardup_subsampled": bool(tr_sampled or te_sampled),
    }


def neardup_cross_mask(train_seqs, test_seqs, seed=0):
    """
    Return a boolean array over *all* test rows (original order), True where the test
    sequence has a near-duplicate (k-mer Jaccard >= threshold) in train.

    Unlike `_neardup_cross_count`, this does NOT subsample — it is meant for the small
    set of high-risk datasets when constructing leakage-resistant test splits.
    """
    a, b = _make_permutations(NUM_PERM, seed=seed)
    tr_sig, tr_shg = _signatures_for(list(map(str, train_seqs)), a, b)
    te_sig, te_shg = _signatures_for(list(map(str, test_seqs)), a, b)
    tr_buckets = _lsh_buckets(tr_sig)

    mask = np.zeros(len(test_seqs), dtype=bool)
    for j in range(len(test_seqs)):
        cand = set()
        for band in range(LSH_BANDS):
            cols = te_sig[j, band * LSH_ROWS:(band + 1) * LSH_ROWS]
            key = (band,) + tuple(cols.tolist())
            if key in tr_buckets:
                cand.update(tr_buckets[key])
        for ci in cand:
            if _exact_jaccard(te_shg[j], tr_shg[ci]) >= JACCARD_THRESHOLD:
                mask[j] = True
                break
    return mask


# --------------------------------------------------------------------------- #
# Exact / revcomp
# --------------------------------------------------------------------------- #
def _exact_stats(train_seqs, test_seqs):
    tr = pd.Series(train_seqs)
    te = pd.Series(test_seqs)
    tr_set = set(tr)
    te_set = set(te)
    dup_train = int(tr.duplicated().sum())
    dup_test = int(te.duplicated().sum())
    exact_cross = len(tr_set & te_set)  # unique test seqs also present in train

    # reverse-complement cross: revcomp(test) present in train (exclude those already
    # counted as exact palindromic matches for a clean "additional" count).
    revcomp_cross = 0
    for s in te_set:
        rc = _reverse_complement(s)
        if rc in tr_set and rc != s:
            revcomp_cross += 1

    return {
        "exact_dup_train": dup_train,
        "exact_dup_test": dup_test,
        "exact_cross_unique": exact_cross,
        "revcomp_cross_unique": revcomp_cross,
        "n_test_unique": len(te_set),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="E0 sequence-level duplication audit")
    ap.add_argument("--data-root", default="data/dna_foundation_benchmark/")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--max-neardup", type=int, default=60000,
                    help="Subsample cap per split for near-dup MinHash (0 to skip near-dup)")
    args = ap.parse_args()

    datasets = discover_datasets(args.data_root)
    print(f"E0 sequence-duplication audit — {len(datasets)} datasets\n")

    rows = []
    for ds in datasets:
        name = ds["name"]
        t0 = time.perf_counter()
        train_seqs, _y_tr, test_seqs, _y_te = load_dataset(ds["train_path"], ds["test_path"])
        train_seqs = list(map(str, train_seqs))
        test_seqs = list(map(str, test_seqs))

        ex = _exact_stats(train_seqs, test_seqs)
        n_test = len(test_seqs)
        exact_cross_pct = 100.0 * ex["exact_cross_unique"] / max(1, ex["n_test_unique"])
        revcomp_cross_pct = 100.0 * ex["revcomp_cross_unique"] / max(1, ex["n_test_unique"])

        if args.max_neardup > 0:
            nd = _neardup_cross_count(train_seqs, test_seqs, max_seqs=args.max_neardup)
        else:
            nd = {"neardup_cross_pct": np.nan, "neardup_subsampled": False,
                  "neardup_n_test_eval": 0, "neardup_n_train_eval": 0}

        # Union contamination: a test sequence is "contaminated" if it is an exact,
        # revcomp, or near-duplicate of some train sequence. Exact+revcomp are exact
        # over unique test seqs; near-dup pct is over (possibly subsampled) test rows.
        # We take the max of the exact/revcomp union and the near-dup rate as a
        # conservative single contamination figure.
        exact_union_pct = 100.0 * (ex["exact_cross_unique"] + ex["revcomp_cross_unique"]) \
            / max(1, ex["n_test_unique"])
        contaminated_pct = float(max(exact_union_pct, nd.get("neardup_cross_pct", 0.0) or 0.0))

        if contaminated_pct >= 5:
            risk = "high"
        elif contaminated_pct >= 1:
            risk = "medium"
        elif contaminated_pct > 0:
            risk = "low"
        else:
            risk = "none"

        row = {
            "dataset": name,
            "n_train": len(train_seqs),
            "n_test": n_test,
            "n_test_unique": ex["n_test_unique"],
            "exact_dup_train": ex["exact_dup_train"],
            "exact_dup_test": ex["exact_dup_test"],
            "exact_cross_unique": ex["exact_cross_unique"],
            "exact_cross_pct": round(exact_cross_pct, 3),
            "revcomp_cross_unique": ex["revcomp_cross_unique"],
            "revcomp_cross_pct": round(revcomp_cross_pct, 3),
            "neardup_cross_pct": round(float(nd.get("neardup_cross_pct", np.nan)), 3)
                if not np.isnan(nd.get("neardup_cross_pct", np.nan)) else np.nan,
            "neardup_subsampled": nd.get("neardup_subsampled", False),
            "contaminated_test_pct": round(contaminated_pct, 3),
            "risk_level": risk,
        }
        rows.append(row)
        dt = time.perf_counter() - t0
        print(f"  {name:44s} exact={exact_cross_pct:5.1f}% revcomp={revcomp_cross_pct:4.1f}% "
              f"neardup={row['neardup_cross_pct'] if row['neardup_cross_pct']==row['neardup_cross_pct'] else float('nan'):5} "
              f"-> {risk:6s} ({dt:.1f}s)")

    df = pd.DataFrame(rows).sort_values("contaminated_test_pct", ascending=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print("\nTop contaminated datasets:")
    print(df.head(12)[["dataset", "exact_cross_pct", "revcomp_cross_pct",
                       "neardup_cross_pct", "contaminated_test_pct", "risk_level"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
