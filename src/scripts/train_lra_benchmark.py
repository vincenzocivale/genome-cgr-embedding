"""
Benchmark RF su i dataset del Genomics Long-Range Benchmark (Nguyen et al. 2024).

Supporta:
  - k-mer features (k=4,5,6)
  - FM embeddings (opzionale, via --model)

Dataset supportati:
  hg38:
    - variant_effect_pathogenic_clinvar   (38k train / 1k test, classificazione binaria)
    - variant_effect_causal_eqtl          (89k train / 8.8k test, classificazione binaria)
    - cage_prediction                     (33k train / 1.9k test, regressione multi-output)
    - regulatory_element_promoter         (subset: 100k train / 96k test, binaria)
    - regulatory_element_enhancer         (subset: 100k train / 192k test, binaria)
  hg19 (richiede --hg19):
    - bulk_rna_expression                 (regressione multi-output)

Task di regressione (cage_prediction, bulk_rna_expression):
  Le label multi-output vengono aggregate via mean per ottenere un target scalare.
  Metriche: R2, MSE, Spearman.

Risultati salvati in results/lra_records.csv.

Usage:
    python3 src/fm_experiment/train_lra_benchmark.py --k-values 4 5 6
    python3 src/fm_experiment/train_lra_benchmark.py --k-values 4 5 6 --model InstaDeepAI/NTv3_650M_pre
    python3 src/fm_experiment/train_lra_benchmark.py --k-values 4 5 6 --tasks cage_prediction --hg38 /path/hg38.fa
    python3 src/fm_experiment/train_lra_benchmark.py --k-values 4 5 6 --tasks bulk_rna_expression --hg19 /path/hg19.fa
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.fcgr import batch_fcgr
from src.features.kmer_features import kmer_from_grids
from src.training.rf_pipeline import (
    train_rf_classifier,
    eval_rf_classifier,
    train_rf_regressor,
    eval_rf_regressor,
)

# ── one-hot encoding ───────────────────────────────────────────────────────────

_NUC_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def onehot_encode(sequences: np.ndarray, window: int) -> np.ndarray:
    """
    One-hot encode the central `window` bp of each sequence.
    Returns (N, 4*window) float32 array.
    Unknown nucleotides (N, etc.) are encoded as all-zeros.
    """
    N = len(sequences)
    X = np.zeros((N, 4 * window), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        mid = len(seq) // 2
        half = window // 2
        fragment = seq[mid - half: mid - half + window]
        for j, nuc in enumerate(fragment):
            idx = _NUC_IDX.get(nuc)
            if idx is not None:
                X[i, j * 4 + idx] = 1.0
    return X

# ── paths ──────────────────────────────────────────────────────────────────────

LRA_DIR = "/raid/DATASETS/genomics-long-range-benchmark"
HG38_FA = "/raid/DATASETS/genomics-long-range-benchmark/Homo_sapiens_assembly38.fasta"
LRA_RECORDS_CSV = "results/regression/lra_records.csv"



# ── records helpers ────────────────────────────────────────────────────────────

def load_lra_records(path: str = LRA_RECORDS_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def save_lra_records(df: pd.DataFrame, path: str = LRA_RECORDS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


def _ensure_row(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    if dataset not in df.index:
        new_row = pd.DataFrame([[pd.NA]], index=[dataset], columns=["_init"])
        df = pd.concat([df, new_row]) if not df.empty else new_row
        if "_init" in df.columns:
            df = df.drop(columns=["_init"])
    return df


def write_result(dataset: str, col_prefix: str, metrics: dict,
                 path: str = LRA_RECORDS_CSV) -> pd.DataFrame:
    df = _ensure_row(load_lra_records(path), dataset)
    for m, v in metrics.items():
        df.loc[dataset, f"{col_prefix}_{m}"] = v
    save_lra_records(df, path)
    return df


def has_result(dataset: str, col_prefix: str, is_regression: bool = False,
               path: str = LRA_RECORDS_CSV) -> bool:
    df = load_lra_records(path)
    if df.empty or dataset not in df.index:
        return False
    sentinel = f"{col_prefix}_R2" if is_regression else f"{col_prefix}_MCC"
    return sentinel in df.columns and pd.notna(df.loc[dataset, sentinel])


# ── chromosome name normalization ─────────────────────────────────────────────

# NCBI GRCh37/GRCh38 accession → UCSC chr name
_NCBI_TO_CHR = {
    # GRCh37 (hg19) autosomes + sex + MT
    "NC_000001.10": "chr1",  "NC_000002.11": "chr2",  "NC_000003.11": "chr3",
    "NC_000004.11": "chr4",  "NC_000005.9":  "chr5",  "NC_000006.11": "chr6",
    "NC_000007.13": "chr7",  "NC_000008.10": "chr8",  "NC_000009.11": "chr9",
    "NC_000010.10": "chr10", "NC_000011.9":  "chr11", "NC_000012.11": "chr12",
    "NC_000013.10": "chr13", "NC_000014.8":  "chr14", "NC_000015.9":  "chr15",
    "NC_000016.9":  "chr16", "NC_000017.10": "chr17", "NC_000018.9":  "chr18",
    "NC_000019.9":  "chr19", "NC_000020.10": "chr20", "NC_000021.8":  "chr21",
    "NC_000022.10": "chr22", "NC_000023.10": "chrX",  "NC_000024.9":  "chrY",
    "NC_012920.1":  "chrM",
    # GRCh38 (hg38) autosomes + sex + MT
    "NC_000001.11": "chr1",  "NC_000002.12": "chr2",  "NC_000003.12": "chr3",
    "NC_000004.12": "chr4",  "NC_000005.10": "chr5",  "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",  "NC_000008.11": "chr8",  "NC_000009.12": "chr9",
    "NC_000010.11": "chr10", "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9":  "chr14", "NC_000015.10": "chr15",
    "NC_000016.10": "chr16", "NC_000017.11": "chr17", "NC_000018.10": "chr18",
    "NC_000019.10": "chr19", "NC_000020.11": "chr20", "NC_000021.9":  "chr21",
    "NC_000022.11": "chr22", "NC_000023.11": "chrX",  "NC_000024.10": "chrY",
    "NC_012920.1":  "chrM",
}


def _build_chr_index(genome: Fasta) -> dict[str, str]:
    """
    Build a mapping from UCSC chr names (chr1, chr2, ...) to the actual
    keys used in the Fasta object, handling both UCSC and NCBI naming.
    Returns {ucsc_name: fasta_key}.
    """
    index: dict[str, str] = {}
    for key in genome.keys():
        if key.startswith("chr"):
            index[key] = key
        elif key in _NCBI_TO_CHR:
            index[_NCBI_TO_CHR[key]] = key
    return index


def _lookup_chrom(chr_index: dict[str, str], chrom: str) -> str | None:
    """Return the fasta key for a UCSC chrom name, or None if not found."""
    return chr_index.get(chrom)


# ── sequence extraction ────────────────────────────────────────────────────────

def _standardize(seq: str) -> str:
    return re.sub("[^ATCG]", "N", seq.upper())


def _pad_sequence(chrom, start: int, seq_len: int, end: int | None = None) -> str | None:
    if end is not None:
        pad = (seq_len - (end - start)) // 2
        s = start - pad
        e = end + pad + (seq_len % 2)
    else:
        pad = seq_len // 2
        e = start + pad + (seq_len % 2)
        s = start - pad
    if s < 0 or e >= len(chrom):
        return None
    return chrom[s:e].seq


# ── task loaders ───────────────────────────────────────────────────────────────

def load_regulatory_elements(task: str, genome: Fasta, subset: bool = True,
                             seq_len: int = 100_000) -> tuple:
    if "promoter" in task:
        fname = "promoter_dataset_subset.csv" if subset else "promoter_dataset.csv"
    else:
        fname = "enhancer_dataset_subset.csv" if subset else "enhancer_dataset.csv"
    path = os.path.join(LRA_DIR, "regulatory_elements", fname)
    df = pd.read_csv(path)
    chr_index = _build_chr_index(genome)

    def _extract(split_df):
        seqs, labels = [], []
        for _, row in split_df.iterrows():
            fkey = _lookup_chrom(chr_index, row["CHROM"])
            if fkey is None:
                continue
            seq = _pad_sequence(genome[fkey], row["START"] - 1, seq_len, end=row["STOP"] - 1)
            if seq:
                seqs.append(_standardize(seq))
                labels.append(row["label"])
        return np.array(seqs), np.array(labels)

    print(f"  Extracting train sequences for {task}...")
    train_seqs, train_labels = _extract(df[df["split"] == "train"])
    print(f"  Extracting test sequences for {task}...")
    test_seqs, test_labels = _extract(df[df["split"] == "test"])
    return train_seqs, train_labels, test_seqs, test_labels


def load_vep_coding(genome: Fasta, seq_len: int = 100_000) -> tuple:
    path = "/raid/DATASETS/dna_foundation_benchmark/pathogenic/vep_pathogenic_coding.csv"
    df = pd.read_csv(path, low_memory=False)
    chr_index = _build_chr_index(genome)

    def _extract(split_df):
        seqs, labels = [], []
        for _, row in split_df.iterrows():
            fkey = _lookup_chrom(chr_index, row["CHROM"])
            if fkey is None:
                continue
            seq = _pad_sequence(genome[fkey], row["POS"] - 1, seq_len)
            if seq:
                seqs.append(_standardize(seq))
                labels.append(int(row["INT_LABEL"]))
        return np.array(seqs), np.array(labels)

    print("  Extracting train sequences for vep_pathogenic_clinvar...")
    train_seqs, train_labels = _extract(df[df["split"] == "train"])
    print("  Extracting test sequences for vep_pathogenic_clinvar...")
    test_seqs, test_labels = _extract(df[df["split"] == "test"])
    return train_seqs, train_labels, test_seqs, test_labels


def load_eqtl(genome: Fasta, seq_len: int = 100_000) -> tuple:
    path = os.path.join(LRA_DIR, "variant_effect_causal_eqtl", "All_Tissues.csv")
    df = pd.read_csv(path)
    chr_index = _build_chr_index(genome)

    def _extract(split_df):
        seqs, labels = [], []
        for _, row in split_df.iterrows():
            fkey = _lookup_chrom(chr_index, row["CHROM"])
            if fkey is None:
                continue
            seq = _pad_sequence(genome[fkey], row["POS"] - 1, seq_len)
            if seq:
                seqs.append(_standardize(seq))
                labels.append(int(row["label"]))
        return np.array(seqs), np.array(labels)

    print("  Extracting train sequences for variant_effect_causal_eqtl...")
    train_seqs, train_labels = _extract(df[df["split"] == "train"])
    print("  Extracting test sequences for variant_effect_causal_eqtl...")
    test_seqs, test_labels = _extract(df[df["split"] == "test"])
    return train_seqs, train_labels, test_seqs, test_labels


def load_cage_prediction(genome: Fasta, seq_len: int = 114_688) -> tuple:
    """
    Load CAGE prediction task (hg38, regressione).

    Label: media su tutti i 896 bin × 50 CAGE tracks → scalare per sequenza.
    """
    coord_path = os.path.join(LRA_DIR, "cage_prediction", "sequences_coordinates.csv")
    npz_dir = os.path.join(LRA_DIR, "cage_prediction", "targets_subset")
    coords = pd.read_csv(coord_path)
    chr_index = _build_chr_index(genome)
    NPZ_SPLIT = 1000

    def _load_npz(split_name: str, idx: int) -> np.ndarray | None:
        for fname in os.listdir(npz_dir):
            if not fname.endswith(".npz"):
                continue
            parts = fname.replace(".npz", "").split("-")
            if len(parts) < 4 or parts[1] != split_name:
                continue
            if int(parts[2]) <= idx <= int(parts[3]):
                npz = np.load(os.path.join(npz_dir, fname))
                key = f"target-{split_name}-{idx}"
                if key in npz:
                    return npz[key][0]  # shape (896, 50)
        return None

    def _extract(split_df, hf_split: str):
        seqs, labels = [], []
        npz_split_name = "valid" if hf_split == "validation" else hf_split
        for _, row in split_df.iterrows():
            fkey = _lookup_chrom(chr_index, row["chrom"])
            if fkey is None:
                continue
            start = int(row["start"]) - 1
            end = int(row["stop"]) - 1
            seq = _pad_sequence(genome[fkey], start, seq_len, end=end)
            if seq is None:
                continue
            targets = _load_npz(npz_split_name, int(row["npy_idx"]))
            if targets is None:
                continue
            seqs.append(_standardize(seq))
            labels.append(float(targets.mean()))
        return np.array(seqs), np.array(labels, dtype=np.float32)

    print("  Extracting train sequences for cage_prediction...")
    train_seqs, train_labels = _extract(coords[coords["split"] == "train"], "train")
    print("  Extracting test sequences for cage_prediction...")
    test_seqs, test_labels = _extract(coords[coords["split"] == "test"], "test")
    return train_seqs, train_labels, test_seqs, test_labels


def load_bulk_rna_expression(genome: Fasta, seq_len: int = 100_000) -> tuple:
    """
    Load Bulk RNA Expression task (hg19, regressione).

    Label: vettore di espressione per N tessuti → aggregato via mean.
    """
    coord_path = os.path.join(LRA_DIR, "bulk_rna_expression", "gene_coordinates.csv")
    labels_path = os.path.join(LRA_DIR, "bulk_rna_expression", "rna_expression_values.csv")
    coords = pd.read_csv(coord_path)
    labels_df = pd.read_csv(labels_path)
    chr_index = _build_chr_index(genome)

    def _extract(split_df):
        seqs, labels = [], []
        for idx, row in split_df.iterrows():
            fkey = _lookup_chrom(chr_index, row["chrom"])
            if fkey is None:
                continue
            start = row["CAGE_representative_TSS"] - 1
            negative = row["strand"] == "-"
            pad = seq_len // 2
            s = start - pad
            e = start + pad + (seq_len % 2)
            if s < 0 or e >= len(genome[fkey]):
                continue
            raw = genome[fkey][s:e]
            seq = raw.reverse.complement.seq if negative else raw.seq
            label_row = labels_df.loc[idx].values.astype(np.float32)
            seqs.append(_standardize(seq))
            labels.append(float(label_row.mean()))
        return np.array(seqs), np.array(labels, dtype=np.float32)

    print("  Extracting train sequences for bulk_rna_expression...")
    train_seqs, train_labels = _extract(coords[coords["split"] == "train"])
    print("  Extracting test sequences for bulk_rna_expression...")
    test_seqs, test_labels = _extract(coords[coords["split"] == "test"])
    return train_seqs, train_labels, test_seqs, test_labels


# ── task metadata ──────────────────────────────────────────────────────────────

# task_name -> (is_regression, genome_version)
TASK_META: dict[str, tuple[bool, str]] = {
    "variant_effect_pathogenic_clinvar": (False, "hg38"),
    "variant_effect_causal_eqtl":        (False, "hg38"),
    "cage_prediction":                   (True,  "hg38"),
    "bulk_rna_expression":               (True,  "hg19"),
    "regulatory_element_promoter":       (False, "hg38"),
    "regulatory_element_enhancer":       (False, "hg38"),
}

# Default tasks (no subset required, genome available without extra arg)
ALL_TASKS = [
    "variant_effect_pathogenic_clinvar",
    "variant_effect_causal_eqtl",
    "cage_prediction",
]

_SUBSET_REQUIRED_TASKS = [
    "regulatory_element_promoter",
    "regulatory_element_enhancer",
]

_HG19_TASKS = ["bulk_rna_expression"]


def load_task(task: str, hg38: Fasta | None, hg19: Fasta | None,
              subset: bool = True) -> tuple:
    is_regression, genome_ver = TASK_META[task]
    genome = hg38 if genome_ver == "hg38" else hg19
    if genome is None:
        raise RuntimeError(
            f"Task '{task}' richiede {genome_ver} ma non è stato fornito. "
            f"Usa --hg19 <path>." if genome_ver == "hg19" else
            f"Usa --hg38 <path>."
        )
    if task == "regulatory_element_promoter":
        return load_regulatory_elements(task, genome, subset=subset)
    if task == "regulatory_element_enhancer":
        return load_regulatory_elements(task, genome, subset=subset)
    if task == "variant_effect_pathogenic_clinvar":
        return load_vep_coding(genome)
    if task == "variant_effect_causal_eqtl":
        return load_eqtl(genome)
    if task == "cage_prediction":
        return load_cage_prediction(genome)
    if task == "bulk_rna_expression":
        return load_bulk_rna_expression(genome)
    raise ValueError(f"Unknown task: {task}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    all_supported = ALL_TASKS + _SUBSET_REQUIRED_TASKS + _HG19_TASKS
    parser = argparse.ArgumentParser(
        description="Benchmark RF su Genomics Long-Range datasets"
    )
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                        choices=all_supported,
                        help=f"Tasks da eseguire (default: {ALL_TASKS})")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 5, 6])
    parser.add_argument("--model", default=None,
                        help="FM model (opzionale). Es: InstaDeepAI/NTv3_650M_pre")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    parser.add_argument("--evo2-layer", type=str, default=None,
                        help="Evo2 layer name for embeddings (optional).")
    parser.add_argument("--evo2-max-length", type=int, default=None,
                        help="Optional max length (bp) for Evo2 tokenization.")
    parser.add_argument("--onehot-windows", nargs="+", type=int, default=[],
                        metavar="W",
                        help="One-hot encode central W bp (es: --onehot-windows 512 1024)")
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--hg38", default=HG38_FA,
                        help=f"Path a hg38 FASTA (default: {HG38_FA})")
    parser.add_argument("--hg19", default=None,
                        help="Path a hg19 FASTA (richiesto per bulk_rna_expression)")
    parser.add_argument("--with-subset-tasks", action="store_true",
                        help="Includi anche promoter/enhancer (subset, ~100k seqs)")
    parser.add_argument("--no-subset", action="store_true",
                        help="Per promoter/enhancer: usa il dataset completo")
    args = parser.parse_args()

    use_subset = not args.no_subset

    tasks_to_run = list(args.tasks)
    if args.with_subset_tasks:
        for t in _SUBSET_REQUIRED_TASKS:
            if t not in tasks_to_run:
                tasks_to_run.append(t)

    print(f"\nGenomics Long-Range Benchmark")
    print(f"Tasks: {tasks_to_run}")
    print(f"K-values: {args.k_values}")
    print(f"FM model: {args.model or 'nessuno'}")
    print(f"Results: {LRA_RECORDS_CSV}\n")

    # Load FM se richiesto
    fm = None
    if args.model:
        from src.embedders.fm_embedder import FMEmbedder
        from src.embedders.hyena_embedder import HyenaEmbedder
        from src.embedders.evo2_embedder import Evo2Embedder
        model_lc = args.model.lower()
        if model_lc.startswith("evo2"):
            fm = Evo2Embedder(
                model_name=args.model,
                cache_dir="cache/lra_fm_embeddings",
                layer_name=args.evo2_layer,
                max_length=args.evo2_max_length,
            )
        elif "hyenadna" in model_lc:
            fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/lra_hyena_embeddings")
        else:
            fm = FMEmbedder(model_name=args.model, cache_dir="cache/lra_fm_embeddings")

    # Load genome FASTA files
    print(f"Caricamento hg38 da {args.hg38} ...")
    hg38 = Fasta(args.hg38, one_based_attributes=False)
    hg19 = None
    if args.hg19:
        print(f"Caricamento hg19 da {args.hg19} ...")
        hg19 = Fasta(args.hg19, one_based_attributes=False)
    print()

    for task in tasks_to_run:
        is_regression = TASK_META[task][0]

        print(f"\n{'='*60}")
        print(f"Task: {task}  ({'regressione' if is_regression else 'classificazione'})")
        print(f"{'='*60}")

        train_seqs, train_labels, test_seqs, test_labels = load_task(
            task, hg38, hg19, subset=use_subset
        )
        print(f"  Train: {len(train_seqs)} | Test: {len(test_seqs)}")

        if len(train_seqs) == 0 or len(test_seqs) == 0:
            print(f"  SKIP: nessuna sequenza estratta")
            continue

        n_classes = len(set(train_labels)) if not is_regression else None

        # ── K-mer ──
        pending_ks = [k for k in args.k_values
                      if not has_result(task, f"kmer_k{k}", is_regression, LRA_RECORDS_CSV)]

        if pending_ks:
            max_k = max(pending_ks)
            grid_size = max(128, 2 ** max_k)
            print(f"  Calcolo FCGR grid={grid_size} ...")
            t0 = time.perf_counter()
            grids_train = batch_fcgr(train_seqs, grid_size=grid_size, n_workers=args.n_workers)
            grids_test  = batch_fcgr(test_seqs,  grid_size=grid_size, n_workers=args.n_workers)
            print(f"  FCGR: {time.perf_counter()-t0:.1f}s")

            for k in args.k_values:
                col_prefix = f"kmer_k{k}"
                if has_result(task, col_prefix, is_regression, LRA_RECORDS_CSV):
                    print(f"  [skip] kmer k={k}")
                    continue
                X_train = kmer_from_grids(grids_train, k, normalize="l1")
                X_test  = kmer_from_grids(grids_test,  k, normalize="l1")
                print(f"  Training RF kmer k={k} ...")
                if is_regression:
                    rf = train_rf_regressor(X_train, train_labels)
                    metrics = eval_rf_regressor(rf, X_test, test_labels)
                    print(f"  kmer k={k}: R2={metrics['R2']:.4f} Spearman={metrics['Spearman']:.4f}")
                else:
                    rf = train_rf_classifier(X_train, train_labels, n_classes)
                    metrics = eval_rf_classifier(rf, X_test, test_labels, n_classes)
                    print(f"  kmer k={k}: MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f}")
                write_result(task, col_prefix, metrics, LRA_RECORDS_CSV)
        else:
            print(f"  [skip] tutti i k-mer già presenti")

        # ── One-hot ──
        for W in args.onehot_windows:
            col_prefix = f"onehot_{W}bp"
            if has_result(task, col_prefix, is_regression, LRA_RECORDS_CSV):
                print(f"  [skip] {col_prefix}")
                continue
            print(f"  One-hot encoding finestra centrale {W} bp ...")
            X_train = onehot_encode(train_seqs, W)
            X_test  = onehot_encode(test_seqs,  W)
            print(f"  Training RF {col_prefix} ...")
            if is_regression:
                rf = train_rf_regressor(X_train, train_labels)
                metrics = eval_rf_regressor(rf, X_test, test_labels)
                print(f"  {col_prefix}: R2={metrics['R2']:.4f} Spearman={metrics['Spearman']:.4f}")
            else:
                rf = train_rf_classifier(X_train, train_labels, n_classes)
                metrics = eval_rf_classifier(rf, X_test, test_labels, n_classes)
                print(f"  {col_prefix}: MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f}")
            write_result(task, col_prefix, metrics, LRA_RECORDS_CSV)

        # ── FM ──
        if fm is not None:
            model_tag = args.model.split("/")[-1]
            col_prefix = f"fm_{model_tag}"
            if has_result(task, col_prefix, is_regression, LRA_RECORDS_CSV):
                print(f"  [skip] FM {model_tag}")
            else:
                print(f"  Calcolo FM embeddings ({model_tag}) ...")
                t0 = time.perf_counter()
                Y_train = fm.embed_sequences(train_seqs, task, "train", args.fm_batch_size)
                Y_test  = fm.embed_sequences(test_seqs,  task, "test",  args.fm_batch_size)
                print(f"  FM embedding: {time.perf_counter()-t0:.1f}s")
                print(f"  Training RF FM ...")
                if is_regression:
                    rf = train_rf_regressor(Y_train.astype(np.float32), train_labels)
                    metrics = eval_rf_regressor(rf, Y_test.astype(np.float32), test_labels)
                    print(f"  FM {model_tag}: R2={metrics['R2']:.4f} Spearman={metrics['Spearman']:.4f}")
                else:
                    rf = train_rf_classifier(Y_train.astype(np.float32), train_labels, n_classes)
                    metrics = eval_rf_classifier(rf, Y_test.astype(np.float32), test_labels, n_classes)
                    print(f"  FM {model_tag}: MCC={metrics['MCC']:.4f} AUROC={metrics['AUROC']:.4f}")
                write_result(task, col_prefix, metrics, LRA_RECORDS_CSV)

    print(f"\n✓ Done! Risultati in {LRA_RECORDS_CSV}")


if __name__ == "__main__":
    main()
