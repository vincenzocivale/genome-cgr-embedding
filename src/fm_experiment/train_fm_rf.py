"""
Script semplificato: genera embeddings FM e addestra RF su tutti i dataset.

Mostra barra di progresso con dataset completati/mancanti.

Usage:
    python3 src/fm_experiment/train_fm_rf.py --model InstaDeepAI/NTv3_650M_pre
    python3 src/fm_experiment/train_fm_rf.py --model LongSafari/hyenadna-medium-160k-seqlen-hf
"""

import argparse
import sys
import os

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.loader import discover_datasets, load_dataset
from src.fm_experiment.fm_embedder import FMEmbedder
from src.fm_experiment.hyena_embedder import HyenaEmbedder
from src.fm_experiment.records import load_records, has_fm, write_fm, RECORDS_CSV
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
    """Train RF con GridSearchCV."""
    scoring = "roc_auc" if n_classes == 2 else "accuracy"
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring=scoring, cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def eval_rf(model, X_test, y_test, n_classes):
    """Valuta RF su test set."""
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
        description="Train RF su FM embeddings per tutti i dataset"
    )
    parser.add_argument("--data-root", default="/data/genomic_bench/dna_foundation_benchmark/")
    parser.add_argument("--model", required=True,
                       help="FM model: NTv3_650M_pre, hyenadna-medium-160k-seqlen-hf, etc.")
    parser.add_argument("--fm-batch-size", type=int, default=32)
    args = parser.parse_args()

    # Carica lista dataset
    all_datasets = discover_datasets(args.data_root)
    n_total = len(all_datasets)

    print(f"\n🧬 FM: {args.model.split('/')[-1]}")
    print(f"📊 Dataset totali: {n_total}")
    print(f"📁 Risultati: results/records.csv\n")

    # Carica FM una sola volta (scegli embedder basato su model name)
    if "hyenadna" in args.model.lower():
        fm = HyenaEmbedder(model_name=args.model, cache_dir="cache/hyena_embeddings")
    else:
        fm = FMEmbedder(model_name=args.model, cache_dir="cache/fm_embeddings")

    records = load_records(RECORDS_CSV)

    # Barra di progresso: mostra quanti completati e quanti mancano
    completed = 0
    pbar = tqdm(all_datasets, desc="Progresso", unit="ds", ncols=70)

    for ds in pbar:
        name = ds["name"]
        n_completed = completed + 1
        n_remaining = n_total - n_completed

        # ── Skip se già completato ──
        if has_fm(name, args.model, records):
            pbar.set_description(f"[{n_completed}/{n_total}] ⏭ {name} (skip)")
            completed = n_completed
            continue

        # Carica sequenze
        train_seqs, train_labels, test_seqs, test_labels = load_dataset(
            ds["train_path"], ds["test_path"]
        )
        n_classes = len(set(train_labels))

        # FM embeddings (cached)
        Y_train = fm.embed_sequences(train_seqs, name, "train", args.fm_batch_size)
        Y_test = fm.embed_sequences(test_seqs, name, "test", args.fm_batch_size)

        # ── Train RF ──
        rf = train_rf(Y_train.astype(np.float32), train_labels, n_classes)
        metrics = eval_rf(rf, Y_test.astype(np.float32), test_labels, n_classes)
        write_fm(name, args.model, metrics)
        records = load_records(RECORDS_CSV)

        # Update progress bar
        completed = n_completed
        pbar.set_description(f"[{n_completed}/{n_total}] ✓ | Mancano {n_remaining}")

    print(f"\n✅ Completato! {n_total}/{n_total} dataset")


if __name__ == "__main__":
    main()
