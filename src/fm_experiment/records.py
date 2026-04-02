"""
Records store: CSV "wide" con una riga per dataset e colonne dinamiche.

Schema colonne:
  - dataset                         (chiave)
  - kmer_k{K}_{metric}              per ogni K e metrica RF k-mer
  - fm_{model_tag}_{metric}         per ogni FM e metrica RF embedding
  - ridge_k{K}_fm_{model_tag}_{m}   per le metriche Ridge

Metriche RF: MCC, AUROC, F1, Accuracy
Metriche Ridge: R2, MSE

model_tag = ultima parte del model ID (es. NTv3_650M_pre)
"""
from __future__ import annotations

import os
import pandas as pd

RECORDS_CSV = "results/records.csv"
RECORDS_MI_CSV = "results/records_mi.csv"
RF_METRICS  = ["MCC", "AUROC", "F1", "Accuracy"]
RIDGE_METRICS = ["R2", "MSE"]
DATASET_INFO_COLS = [
    "n_train", "n_test", "n_classes", "class_balance",
    "seq_len_mean", "seq_len_std", "seq_len_min", "seq_len_max",
    "gc_content_mean",
]


# ── helpers ────────────────────────────────────────────────────────────────

def _model_tag(model_name: str) -> str:
    return model_name.split("/")[-1]


def kmer_col(k: int, metric: str) -> str:
    return f"kmer_k{k}_{metric}"


def fm_col(model_name: str, metric: str) -> str:
    return f"fm_{_model_tag(model_name)}_{metric}"


def ridge_col(k: int, model_name: str, metric: str) -> str:
    return f"ridge_k{k}_{_model_tag(model_name)}_{metric}"


def tok_col(model_name: str, metric: str) -> str:
    return f"tok_{_model_tag(model_name)}_{metric}"


def multiscale_col(k_values: tuple[int, ...], metric: str) -> str:
    tag = "_".join(str(k) for k in sorted(k_values))
    return f"kmer_multi_k{tag}_{metric}"


def ridge_multi_col(k_values: tuple[int, ...], model_name: str, metric: str) -> str:
    tag = "_".join(str(k) for k in sorted(k_values))
    return f"ridge_multi_k{tag}_{_model_tag(model_name)}_{metric}"


def mi_kmer_col(k: int) -> str:
    return f"mi_kmer_k{k}"


def mi_fm_col(model_name: str) -> str:
    return f"mi_fm_{_model_tag(model_name)}"


def mi_tok_col(model_name: str) -> str:
    return f"mi_tok_{_model_tag(model_name)}"


def _float_tag(x: float) -> str:
    s = f"{x:.4f}"
    s = s.rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace(".", "p")


def quadtree_col(max_depth: int, p_threshold: float,
                 min_count: int, metric: str) -> str:
    tag = f"d{max_depth}_p{_float_tag(p_threshold)}_m{min_count}"
    return f"qt_{tag}_{metric}"


def wavelet_col(levels: int, metric: str) -> str:
    return f"wavelet_l{levels}_{metric}"


def wms_col(k_values: tuple[int, ...], weighting: str, metric: str) -> str:
    tag = "_".join(str(k) for k in sorted(k_values))
    return f"wms_k{tag}_{weighting}_{metric}"


def wms_weights_col(k_values: tuple[int, ...], weighting: str) -> str:
    tag = "_".join(str(k) for k in sorted(k_values))
    return f"wms_k{tag}_{weighting}_weights"


# ── load / save ─────────────────────────────────────────────────────────────

def load_records(path: str = RECORDS_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def _save(df: pd.DataFrame, path: str = RECORDS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=True)


# ── check existence ──────────────────────────────────────────────────────────

def has_dataset_info(dataset: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    return "n_train" in df.columns and pd.notna(df.loc[dataset, "n_train"])


def has_kmer(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = kmer_col(k, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_fm(dataset: str, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = fm_col(model_name, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_ridge(dataset: str, k: int, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = ridge_col(k, model_name, RIDGE_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_ridge_multi(dataset: str, k_values: tuple[int, ...],
                    model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = ridge_multi_col(k_values, model_name, RIDGE_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_multiscale(dataset: str, k_values: tuple[int, ...], df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = multiscale_col(k_values, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_tok(dataset: str, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = tok_col(model_name, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_quadtree(dataset: str, max_depth: int, p_threshold: float,
                 min_count: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = quadtree_col(max_depth, p_threshold, min_count, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_wavelet(dataset: str, levels: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = wavelet_col(levels, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_wms(dataset: str, k_values: tuple[int, ...], weighting: str,
            df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = wms_col(k_values, weighting, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def load_mi_records(path: str = RECORDS_MI_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def has_mi_tok(dataset: str, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = mi_tok_col(model_name)
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_mi_kmer(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = mi_kmer_col(k)
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_mi_fm(dataset: str, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = mi_fm_col(model_name)
    return col in df.columns and pd.notna(df.loc[dataset, col])


# ── write results ─────────────────────────────────────────────────────────────

def _ensure_row(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Add a row for dataset if missing, compatible with empty DataFrames."""
    if dataset not in df.index:
        new_row = pd.DataFrame([[pd.NA]], index=[dataset], columns=["_init"])
        df = pd.concat([df, new_row]) if not df.empty else new_row
        if "_init" in df.columns:
            df = df.drop(columns=["_init"])
    return df


def write_dataset_info(dataset: str, info: dict,
                       path: str = RECORDS_CSV) -> pd.DataFrame:
    """info keys: see DATASET_INFO_COLS"""
    df = _ensure_row(load_records(path), dataset)
    for col in DATASET_INFO_COLS:
        if col in info:
            df.loc[dataset, col] = info[col]
    _save(df, path)
    return df


def write_multiscale(dataset: str, k_values: tuple[int, ...], rf_metrics: dict,
                     path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, multiscale_col(k_values, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_tok(dataset: str, model_name: str, rf_metrics: dict,
              path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, tok_col(model_name, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_mi_tok(dataset: str, model_name: str, mi_value: float,
                 path: str = RECORDS_MI_CSV) -> pd.DataFrame:
    df = _ensure_row(load_mi_records(path), dataset)
    df.loc[dataset, mi_tok_col(model_name)] = mi_value
    _save(df, path)
    return df


def write_mi_kmer(dataset: str, k: int, mi_value: float,
                  path: str = RECORDS_MI_CSV) -> pd.DataFrame:
    df = _ensure_row(load_mi_records(path), dataset)
    df.loc[dataset, mi_kmer_col(k)] = mi_value
    _save(df, path)
    return df


def write_mi_fm(dataset: str, model_name: str, mi_value: float,
                path: str = RECORDS_MI_CSV) -> pd.DataFrame:
    df = _ensure_row(load_mi_records(path), dataset)
    df.loc[dataset, mi_fm_col(model_name)] = mi_value
    _save(df, path)
    return df


def write_kmer(dataset: str, k: int, rf_metrics: dict,
               path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, kmer_col(k, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_fm(dataset: str, model_name: str, rf_metrics: dict,
             path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, fm_col(model_name, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_quadtree(dataset: str, max_depth: int, p_threshold: float,
                   min_count: int, rf_metrics: dict,
                   path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, quadtree_col(max_depth, p_threshold, min_count, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_wavelet(dataset: str, levels: int, rf_metrics: dict,
                  path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, wavelet_col(levels, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_wms(dataset: str, k_values: tuple[int, ...], weighting: str,
              rf_metrics: dict, weights: list[float] | None = None,
              path: str = RECORDS_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, wms_col(k_values, weighting, m)] = rf_metrics[m]
    if weights is not None:
        df.loc[dataset, wms_weights_col(k_values, weighting)] = ",".join(
            f"{w:.4f}" for w in weights
        )
    _save(df, path)
    return df


def write_ridge_multi(dataset: str, k_values: tuple[int, ...], model_name: str,
                      ridge_metrics: dict, path: str = RECORDS_CSV) -> pd.DataFrame:
    """ridge_metrics keys: R2, MSE"""
    df = _ensure_row(load_records(path), dataset)
    for m in RIDGE_METRICS:
        df.loc[dataset, ridge_multi_col(k_values, model_name, m)] = ridge_metrics[m]
    _save(df, path)
    return df


def write_ridge(dataset: str, k: int, model_name: str, ridge_metrics: dict,
                path: str = RECORDS_CSV) -> pd.DataFrame:
    """ridge_metrics keys: R2, MSE"""
    df = _ensure_row(load_records(path), dataset)
    for m in RIDGE_METRICS:
        df.loc[dataset, ridge_col(k, model_name, m)] = ridge_metrics[m]
    _save(df, path)
    return df


# ── read back ──────────────────────────────────────────────────────────────

def get_kmer(dataset: str, k: int,
             df: pd.DataFrame) -> dict | None:
    if not has_kmer(dataset, k, df):
        return None
    return {m: df.loc[dataset, kmer_col(k, m)] for m in RF_METRICS}


def get_fm(dataset: str, model_name: str,
           df: pd.DataFrame) -> dict | None:
    if not has_fm(dataset, model_name, df):
        return None
    return {m: df.loc[dataset, fm_col(model_name, m)] for m in RF_METRICS}


def get_quadtree(dataset: str, max_depth: int, p_threshold: float,
                 min_count: int, df: pd.DataFrame) -> dict | None:
    if not has_quadtree(dataset, max_depth, p_threshold, min_count, df):
        return None
    return {m: df.loc[dataset, quadtree_col(max_depth, p_threshold, min_count, m)]
            for m in RF_METRICS}


def get_wavelet(dataset: str, levels: int, df: pd.DataFrame) -> dict | None:
    if not has_wavelet(dataset, levels, df):
        return None
    return {m: df.loc[dataset, wavelet_col(levels, m)] for m in RF_METRICS}


def get_wms(dataset: str, k_values: tuple[int, ...], weighting: str,
            df: pd.DataFrame) -> dict | None:
    if not has_wms(dataset, k_values, weighting, df):
        return None
    return {m: df.loc[dataset, wms_col(k_values, weighting, m)] for m in RF_METRICS}


def get_ridge(dataset: str, k: int, model_name: str,
              df: pd.DataFrame) -> dict | None:
    if not has_ridge(dataset, k, model_name, df):
        return None
    return {m: df.loc[dataset, ridge_col(k, model_name, m)] for m in RIDGE_METRICS}
