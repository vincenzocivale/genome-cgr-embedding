"""Evaluate pooling choices with the exact E4 classification protocol."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.data.loader import discover_datasets, load_dataset
from src.embedders.embedding_cache import load_cached_embeddings
from src.embedders.evo2_embedder import Evo2Embedder
from src.embedders.fm_embedder import FMEmbedder
from src.features.kmer_features import extract_kmer_features
from src.rebuttal.common import N_CANDIDATES, N_CV_SPLITS, SEED, evaluate_classifier, fit_probe, selection_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/dna_foundation_benchmark"); p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--poolings", nargs="+", default=["mean", "max", "mean_max", "cls", "attention"])
    p.add_argument("--probe", choices=["rf", "linear", "mlp"], default="rf"); p.add_argument("--k", type=int, default=6)
    p.add_argument("--datasets", nargs="*"); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--n-workers", type=int, default=2); p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--output", default="results/rebuttal/E5_pooling/pooling_results.csv")
    a = p.parse_args()
    existing = pd.read_csv(a.output) if os.path.exists(a.output) else pd.DataFrame()
    rows = existing.to_dict("records")
    completed = set(zip(existing.get("dataset", []), existing.get("model", []),
                        existing.get("pooling", []), existing.get("probe", [])))
    datasets = discover_datasets(a.data_root)
    if a.datasets: datasets = [d for d in datasets if d["name"] in a.datasets]
    for model_name in a.models:
        for pooling in a.poolings:
            # These tokenizers never prepend a CLS token at position 0 (verified
            # empirically), so CLS pooling is not applicable.  DNABERT-2 is NOT
            # in this list: it is a BERT/BPE model that genuinely prepends [CLS].
            if pooling == "cls" and any(x in model_name.lower() for x in ("hyenadna", "caduceus", "evo2")):
                print(f"[not applicable] {model_name} / cls"); continue
            embedder = None
            for ds in datasets:
                trseq, ytr, teseq, yte = load_dataset(ds["train_path"], ds["test_path"])
                xtr = load_cached_embeddings(model_name, ds["name"], "train", pooling)
                xte = load_cached_embeddings(model_name, ds["name"], "test", pooling)
                key = (ds["name"], model_name, pooling, a.probe)
                if key in completed:
                    print(f"[skip/completed] {ds['name']} {pooling}"); continue
                if xtr is None or xte is None:
                    if embedder is None:
                        embedder = (Evo2Embedder(model_name=model_name, pooling=pooling)
                                    if model_name.lower().startswith("evo2_")
                                    else FMEmbedder(model_name=model_name, pooling=pooling))
                    xtr = embedder.embed_sequences(trseq, ds["name"], "train", a.batch_size)
                    xte = embedder.embed_sequences(teseq, ds["name"], "test", a.batch_size)
                clf = fit_probe(xtr, ytr, len(set(ytr)), a.probe, a.n_jobs)
                row = dict(dataset=ds["name"], model=model_name, pooling=pooling, probe=a.probe,
                           feature_dim=xtr.shape[1], n_train=len(ytr), n_test=len(yte), n_classes=len(set(ytr)),
                           seed=SEED, cv_folds=N_CV_SPLITS, tuning_candidates=N_CANDIDATES,
                           selection_metric=selection_metric(len(set(ytr))), cv_score=clf.best_score_, best_params=str(clf.best_params_))
                row.update(evaluate_classifier(clf, xte, yte, len(set(ytr)))); rows.append(row)
                print(ds["name"], pooling, f"MCC={row['MCC']:.3f}")
                Path(a.output).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(a.output, index=False); completed.add(key)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(a.output, index=False)

if __name__ == "__main__": main()
