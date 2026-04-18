"""
Confusion matrix analysis for polarised datasets.

Re-trains RF on k-mer (k=6) and FM (NTv3) features for:
  - splice/splice_site_type_DNABERT  (FM >> k-mer, delta MCC = +0.34)
  - genomic_benchmark/regulatory_region_type  (k-mer >> FM, delta MCC = -0.10)

Saves per-class metrics and confusion matrices to
  results/classification/confusion_matrix_analysis.csv
  results/classification/confusion_matrices.npz
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# make sure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import load_dataset
from src.features.kmer_features import extract_kmer_features
from src.training.rf_pipeline import train_rf_fast, eval_rf

DATA_ROOT = "/raid/DATASETS/dna_foundation_benchmark"
FM_MODEL  = "InstaDeepAI/NTv3_650M_pre"
K         = 6
BATCH_SIZE = 32

DATASETS = {
    "splice/splice_site_type_DNABERT": {
        "class_names": {0: "acceptor", 1: "donor", 2: "neither"},
    },
    "genomic_benchmark/regulatory_region_type": {
        "class_names": {0: "enhancer", 1: "open_chromatin", 2: "promoter"},
    },
}

OUT_CSV = "results/classification/confusion_matrix_analysis.csv"
OUT_NPZ = "results/classification/confusion_matrices.npz"


def load_data(ds_name: str, cfg: dict):
    cat, name = ds_name.split("/")
    base = os.path.join(DATA_ROOT, cat, name)
    train_seqs, train_labels, test_seqs, test_labels = load_dataset(
        os.path.join(base, "train.csv"),
        os.path.join(base, "test.csv"),
    )
    return train_seqs, test_seqs, train_labels, test_labels


def get_fm_embeddings(train_seqs, test_seqs, ds_name: str):
    from src.embedders.fm_embedder import FMEmbedder
    embedder = FMEmbedder(model_name=FM_MODEL, cache_dir="cache/fm_embeddings")
    cat, name = ds_name.split("/")
    Y_train = embedder.embed_sequences(train_seqs, ds_name, "train", BATCH_SIZE)
    Y_test  = embedder.embed_sequences(test_seqs,  ds_name, "test",  BATCH_SIZE)
    return Y_train, Y_test


def per_class_metrics(cm: np.ndarray, class_names: dict) -> pd.DataFrame:
    rows = []
    n_total = cm.sum()
    for i, cname in class_names.items():
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = n_total - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append({"class": cname, "precision": prec, "recall": rec, "f1": f1, "support": int(cm[i].sum())})
    return pd.DataFrame(rows)


def run_one(ds_name: str, cfg: dict, results: list, cms: dict):
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print('='*60)

    train_seqs, test_seqs, train_labels, test_labels = load_data(ds_name, cfg)
    n_classes = len(np.unique(train_labels))
    class_names = cfg["class_names"]

    # --- k-mer ---
    print(f"Computing k={K} features...")
    X_train_kmer = extract_kmer_features(train_seqs, k=K)
    X_test_kmer  = extract_kmer_features(test_seqs,  k=K)

    print("Training RF on k-mer...")
    rf_kmer = train_rf_fast(X_train_kmer, train_labels, n_classes)
    y_pred_kmer = rf_kmer.predict(X_test_kmer)
    cm_kmer = confusion_matrix(test_labels, y_pred_kmer)
    print("k-mer confusion matrix:")
    print(cm_kmer)
    pcm_kmer = per_class_metrics(cm_kmer, class_names)
    pcm_kmer.insert(0, "model", "kmer_k6")
    pcm_kmer.insert(0, "dataset", ds_name)

    # --- FM ---
    print(f"Getting FM embeddings ({FM_MODEL})...")
    X_train_fm, X_test_fm = get_fm_embeddings(train_seqs, test_seqs, ds_name)

    print("Training RF on FM embeddings...")
    rf_fm = train_rf_fast(X_train_fm, train_labels, n_classes)
    y_pred_fm = rf_fm.predict(X_test_fm)
    cm_fm = confusion_matrix(test_labels, y_pred_fm)
    print("FM confusion matrix:")
    print(cm_fm)
    pcm_fm = per_class_metrics(cm_fm, class_names)
    pcm_fm.insert(0, "model", f"fm_{FM_MODEL.split('/')[-1]}")
    pcm_fm.insert(0, "dataset", ds_name)

    # Delta per class (FM - kmer recall)
    print("\nDelta recall (FM - k-mer) per class:")
    for i, cname in class_names.items():
        d = pcm_fm.loc[pcm_fm["class"]==cname, "recall"].values[0] - \
            pcm_kmer.loc[pcm_kmer["class"]==cname, "recall"].values[0]
        print(f"  {cname:<20}: {d:+.3f}")

    results.extend([pcm_kmer, pcm_fm])
    tag = ds_name.replace("/", "__")
    cms[f"{tag}_kmer"] = cm_kmer
    cms[f"{tag}_fm"]   = cm_fm


def main():
    os.makedirs("results/classification", exist_ok=True)
    results, cms = [], {}

    for ds_name, cfg in DATASETS.items():
        run_one(ds_name, cfg, results, cms)

    df_out = pd.concat(results, ignore_index=True)
    df_out.to_csv(OUT_CSV, index=False)
    np.savez_compressed(OUT_NPZ, **cms)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Saved: {OUT_NPZ}")
    print("\n=== FULL RESULTS ===")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
