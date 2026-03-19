"""
Script: genera embeddings dall'embedding layer (pre-transformer) e addestra RF.

Confronta la rappresentazione statica dei token (solo lookup table)
con quella contestualizzata (output completo del FM).

Usage:
    python3 src/fm_experiment/train_tok_rf.py --model InstaDeepAI/NTv3_650M_pre
    python3 src/fm_experiment/train_tok_rf.py --model LongSafari/hyenadna-medium-160k-seqlen-hf
"""

import argparse
import sys
import os

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.tokenizer_embedder import TokenizerEmbedder
from src.fm_experiment.records import (
    load_records, has_tok, write_tok, RECORDS_CSV,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score


_RF_PARAM_GRID = {
    "n_estimators":      [200, 500],
    "max_features":      ["sqrt"],
    "max_depth":         [20],
    "min_samples_split": [2],
}


def train_rf(X_train, y_train, n_classes):
    scoring = "roc_auc" if n_classes == 2 else "accuracy"
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring=scoring, cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def eval_rf(model, X_test, y_test, n_classes):
    y_pred = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    if n_classes == 2:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test),
                              multi_class="ovr", average="macro")
    return {"MCC": mcc, "AUROC": auroc, "F1": f1, "Accuracy": acc}


def main():
    parser = argparse.ArgumentParser(
        description="Train RF su embedding-layer (pre-transformer) per tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True,
                       help="FM model HF name")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets)

    print(f"\n🔤 Tokenizer Embedding RF: {args.model.split('/')[-1]}")
    print(f"📊 Dataset totali: {n_total}")
    print(f"📁 Risultati: {RECORDS_CSV}\n")

    embedder = TokenizerEmbedder(model_name=args.model)
    records = load_records(RECORDS_CSV)

    pbar = tqdm(all_datasets, desc="Progresso", unit="ds", ncols=70)

    for ds in pbar:
        name = ds["name"]

        if has_tok(name, args.model, records):
            pbar.set_description(f"⏭ {name} (skip)")
            continue

        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        X_train = embedder.embed_sequences(
            train_seqs, name, "train", args.batch_size
        ).astype(np.float32)
        X_test = embedder.embed_sequences(
            test_seqs, name, "test", args.batch_size
        ).astype(np.float32)

        rf = train_rf(X_train, train_labels, n_classes)
        metrics = eval_rf(rf, X_test, test_labels, n_classes)
        write_tok(name, args.model, metrics)
        records = load_records(RECORDS_CSV)

        pbar.set_description(f"✓ {name} MCC={metrics['MCC']:.3f}")

    print(f"\n✅ Completato! Risultati in: {RECORDS_CSV}")


if __name__ == "__main__":
    main()
