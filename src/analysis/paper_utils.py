from __future__ import annotations

import pandas as pd

from src.records.records import best_fixed_k, best_kmer_metric

CATEGORY_LABELS = {
    "histone": "Histone (Yeast)",
    "methylation": "Methylation",
    "promoter": "Promoter",
    "splice": "Splice",
    "tfbs": "TFBS",
    "enhancer_generic": "Enhancer/Generic",
}


def infer_biocat(dataset: str) -> str:
    if dataset.startswith("EMP/"):
        return "histone"
    if dataset.startswith(("deep4mc/", "iDNA_ABF/")):
        return "methylation"
    if dataset.startswith("splice/"):
        return "splice"
    if dataset.startswith(("tf/", "mouse/")):
        return "tfbs"
    if "promoter" in dataset.lower() or dataset.startswith("iPro-WAEL/Promoter_") or dataset.startswith("prom/"):
        return "promoter"
    return "enhancer_generic"


def add_dataset_annotations(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    annotated["biocat"] = annotated.index.to_series().map(infer_biocat)
    annotated["biocat_label"] = annotated["biocat"].map(CATEGORY_LABELS)
    annotated["best_k"] = best_fixed_k(annotated)
    annotated["best_kmer_MCC"] = best_kmer_metric(annotated, metric="MCC")
    annotated["best_kmer_AUROC"] = best_kmer_metric(annotated, metric="AUROC")
    return annotated
