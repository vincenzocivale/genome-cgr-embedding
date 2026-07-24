"""Run E4 on cached FM embeddings, saving one tidy row per matrix cell.

Example: python src/rebuttal/e4_probe_fairness/run_probe_matrix.py \
  --model InstaDeepAI/NTv3_650M_pre --k 6 --datasets splice/splice_site_type_NT
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder
from src.features.kmer_features import extract_kmer_features
from src.rebuttal.common import N_CANDIDATES, N_CV_SPLITS, SEED, evaluate_classifier, fit_probe, load_cached_embedding, project_kmers, selection_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/dna_foundation_benchmark")
    p.add_argument("--model", required=True); p.add_argument("--k", type=int, default=6)
    p.add_argument("--pooling", default="mean"); p.add_argument("--datasets", nargs="*")
    p.add_argument("--n-workers", type=int, default=2); p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--completed-from", nargs="*", default=[],
                   help="read-only result CSVs whose completed keys should be skipped")
    p.add_argument("--embed-missing", action="store_true",
                   help="compute/cache missing FM embeddings before the probe matrix")
    p.add_argument("--output", default="results/rebuttal/E4_probe_fairness/probe_matrix.csv")
    args = p.parse_args()
    existing = pd.read_csv(args.output) if os.path.exists(args.output) else pd.DataFrame()
    rows = existing.to_dict("records")
    completion_frames = [existing]
    for path in args.completed_from:
        if os.path.exists(path):
            completion_frames.append(pd.read_csv(path))
        else:
            print(f"[warning/missing-completion-file] {path}")
    completion = pd.concat(completion_frames, ignore_index=True) if completion_frames else pd.DataFrame()
    completed = set(zip(completion.get("dataset", []), completion.get("model", []),
                        completion.get("pooling", []), completion.get("representation", []),
                        completion.get("probe", [])))
    datasets = discover_datasets(args.data_root)
    if args.datasets: datasets = [d for d in datasets if d["name"] in args.datasets]
    embedder = None
    for ds in datasets:
        ytr_seqs, ytr, yte_seqs, yte = load_dataset(ds["train_path"], ds["test_path"])
        fmtr = load_cached_embedding(args.model, ds["name"], "train", args.pooling)
        fmte = load_cached_embedding(args.model, ds["name"], "test", args.pooling)
        if fmtr is None or fmte is None:
            if not args.embed_missing:
                print(f"[skip/no-cache] {ds['name']}"); continue
            if embedder is None:
                embedder = FMEmbedder(model_name=args.model, pooling=args.pooling)
            fmtr = embedder.embed_sequences(ytr_seqs, ds["name"], "train", args.batch_size)
            fmte = embedder.embed_sequences(yte_seqs, ds["name"], "test", args.batch_size)
        # Exact k-mer frequencies need only a 2**k grid; retaining the historic
        # 128 grid would allocate four times more memory at k=6 without changing X.
        xtr = extract_kmer_features(ytr_seqs, k=args.k, grid_size=2**args.k, n_workers=args.n_workers)
        xte = extract_kmer_features(yte_seqs, k=args.k, grid_size=2**args.k, n_workers=args.n_workers)
        ptr, pte, rtr, rte, mapping = project_kmers(xtr, fmtr, xte, fmte)
        for representation, train, test in (("kmer", xtr, xte), ("fm_embedding", fmtr, fmte),
                                            ("projected_component", ptr, pte), ("residual", rtr, rte)):
            for probe in ("rf", "linear", "mlp"):
                key = (ds["name"], args.model, args.pooling, representation, probe)
                if key in completed:
                    print(f"[skip/completed] {ds['name']} {representation} {probe}")
                    continue
                model = fit_probe(train, ytr, len(np.unique(ytr)), probe, args.n_jobs)
                row = dict(dataset=ds["name"], model=args.model, pooling=args.pooling, k=args.k,
                           representation=representation, probe=probe, feature_dim=train.shape[1],
                           n_train=len(ytr), n_test=len(yte), n_classes=len(np.unique(ytr)), seed=SEED,
                           cv_folds=N_CV_SPLITS, tuning_candidates=N_CANDIDATES,
                           selection_metric=selection_metric(len(np.unique(ytr))), cv_score=model.best_score_,
                           best_params=str(model.best_params_), mapping_r2=mapping["r2"])
                row.update(evaluate_classifier(model, test, yte, len(np.unique(ytr))))
                rows.append(row); print(ds["name"], representation, probe, f"MCC={row['MCC']:.3f}")
                # Preserve completed cells if an expensive neural probe is interrupted.
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(args.output, index=False)
                completed.add(key)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

if __name__ == "__main__": main()
