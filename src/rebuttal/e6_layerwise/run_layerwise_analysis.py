"""Sample FM depth and measure downstream MCC, k-mer R² and residual MCC."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from src.data.loader import discover_datasets, load_dataset
from src.embedders.fm_embedder import FMEmbedder
from src.features.kmer_features import extract_kmer_features
from src.rebuttal.common import N_CANDIDATES, N_CV_SPLITS, SEED, evaluate_classifier, fit_probe, project_kmers, selection_metric


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", default="data/dna_foundation_benchmark")
    p.add_argument("--model", required=True); p.add_argument("--pooling", default="mean"); p.add_argument("--probe", choices=["rf", "linear", "mlp"], default="rf")
    p.add_argument("--k", type=int, default=6); p.add_argument("--fractions", type=float, nargs="+", default=[.25, .5, .75, 1.])
    p.add_argument("--datasets", nargs="*"); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--n-workers", type=int, default=2); p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--output", default="results/rebuttal/E6_layerwise/layerwise_results.csv"); a = p.parse_args()
    existing=pd.read_csv(a.output) if os.path.exists(a.output) else pd.DataFrame()
    rows=existing.to_dict("records")
    completed=set(zip(existing.get("dataset", []), existing.get("model", []), existing.get("pooling", []),
                      existing.get("probe", []), existing.get("layer", []), existing.get("representation", [])))
    datasets=discover_datasets(a.data_root)
    if a.datasets: datasets=[d for d in datasets if d["name"] in a.datasets]
    embedder=FMEmbedder(model_name=a.model, pooling=a.pooling)
    for ds in datasets:
        trseq,ytr,teseq,yte=load_dataset(ds["train_path"], ds["test_path"])
        xtr=extract_kmer_features(trseq,k=a.k,grid_size=2**a.k,n_workers=a.n_workers)
        xte=extract_kmer_features(teseq,k=a.k,grid_size=2**a.k,n_workers=a.n_workers)
        layers_tr=embedder.embed_sequences_layers(trseq,tuple(a.fractions),a.batch_size,a.pooling)
        layers_te=embedder.embed_sequences_layers(teseq,tuple(a.fractions),a.batch_size,a.pooling)
        final=max(layers_tr)
        for layer, emb_tr in layers_tr.items():
            emb_te=layers_te[layer]; ptr,pte,rtr,rte,mapping=project_kmers(xtr,emb_tr,xte,emb_te)
            for representation, train, test in (("fm_embedding",emb_tr,emb_te),("residual",rtr,rte)):
                key=(ds["name"],a.model,a.pooling,a.probe,layer,representation)
                if key in completed:
                    print(f"[skip/completed] {ds['name']} layer={layer} {representation}"); continue
                clf=fit_probe(train,ytr,len(set(ytr)),a.probe,a.n_jobs)
                row=dict(dataset=ds["name"],model=a.model,pooling=a.pooling,probe=a.probe,k=a.k,layer=layer,
                         depth_fraction=layer/final if final else 0.,representation=representation,feature_dim=train.shape[1],
                         effective_rank=int(__import__('numpy').linalg.matrix_rank(train)),mapping_r2=mapping["r2"],ridge_alpha=mapping["ridge_alpha"],
                         seed=SEED,cv_folds=N_CV_SPLITS,tuning_candidates=N_CANDIDATES,selection_metric=selection_metric(len(set(ytr))),cv_score=clf.best_score_)
                row.update(evaluate_classifier(clf,test,yte,len(set(ytr)))); rows.append(row)
                print(ds["name"],layer,representation,f"MCC={row['MCC']:.3f}")
                Path(a.output).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False);completed.add(key)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output,index=False)

if __name__ == "__main__": main()
