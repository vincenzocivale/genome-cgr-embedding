"""
Helpers for the repository's canonical wide-format result CSVs.

Each table uses one row per dataset and dynamic metric columns.
"""
from __future__ import annotations

import fcntl
import os
import pandas as pd

RECORDS_CSV = "results/classification/records_rf.csv"
RECORDS_PCA_CSV = "results/classification/records_pca.csv"
RECORDS_DECOMP_CSV = "results/decomposition/records_decomposition.csv"
RECORDS_MI_CSV = "results/exploratory/records_mi.csv"
RECORDS_LINEAR_CSV = "results/classification/records_linear_probe.csv"
RECORDS_CANONICAL_CSV = "results/classification/records_canonical_kmer.csv"
EFFICIENCY_CSV = "results/efficiency/efficiency.csv"
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


def proj_col(k: int, model_name: str, metric: str) -> str:
    return f"proj_k{k}_{_model_tag(model_name)}_{metric}"


def resid_col(k: int, model_name: str, metric: str) -> str:
    return f"resid_k{k}_{_model_tag(model_name)}_{metric}"


def _decomp_suffix(pooling: str = "mean", mapper: str = "ridge") -> str:
    if pooling == "mean" and mapper == "ridge":
        return ""
    return f"__pool_{pooling}__map_{mapper}"


def decomp_ridge_col(k: int, model_name: str, metric: str,
                     pooling: str = "mean", mapper: str = "ridge") -> str:
    return f"{ridge_col(k, model_name, metric)}{_decomp_suffix(pooling, mapper)}"


def decomp_proj_col(k: int, model_name: str, metric: str,
                    pooling: str = "mean", mapper: str = "ridge") -> str:
    return f"{proj_col(k, model_name, metric)}{_decomp_suffix(pooling, mapper)}"


def decomp_resid_col(k: int, model_name: str, metric: str,
                     pooling: str = "mean", mapper: str = "ridge") -> str:
    return f"{resid_col(k, model_name, metric)}{_decomp_suffix(pooling, mapper)}"


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


def best_kmer_cols(
    df: pd.DataFrame,
    metric: str = "MCC",
    prefix: str = "kmer",
    include_multiscale: bool = False,
) -> list[str]:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_k") and c.endswith(f"_{metric}")]
    if include_multiscale:
        cols.extend(
            c for c in df.columns if c.startswith(f"{prefix}_multi_k") and c.endswith(f"_{metric}")
        )
    return cols


def best_kmer_metric(
    df: pd.DataFrame,
    metric: str = "MCC",
    prefix: str = "kmer",
    include_multiscale: bool = False,
) -> pd.Series:
    cols = best_kmer_cols(df, metric=metric, prefix=prefix, include_multiscale=include_multiscale)
    if not cols:
        return pd.Series(dtype=float)
    return df[cols].apply(pd.to_numeric, errors="coerce").max(axis=1)


def best_fixed_k(df: pd.DataFrame, metric: str = "MCC", prefix: str = "kmer") -> pd.Series:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_k") and c.endswith(f"_{metric}")]
    if not cols:
        return pd.Series(dtype="Int64")
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    best_cols = numeric.idxmax(axis=1)
    return best_cols.str.extract(rf"{prefix}_k(\d+)_{metric}")[0].astype("Int64")


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


def write_kmer_rf_std(dataset: str, k: int, metrics_dict: dict,
                      path: str = RECORDS_CSV) -> pd.DataFrame:
    """metrics_dict keys: MCC_mean/std, AUROC_mean/std, F1_mean/std, Accuracy_mean/std"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        mean_key = f"{m}_mean"
        std_key = f"{m}_std"
        if mean_key in metrics_dict:
            df.loc[dataset, f"{kmer_col(k, m)}_mean"] = metrics_dict[mean_key]
        if std_key in metrics_dict:
            df.loc[dataset, f"{kmer_col(k, m)}_std"] = metrics_dict[std_key]
    _save(df, path)
    return df


def write_fm_rf_std(dataset: str, model_name: str, metrics_dict: dict,
                    path: str = RECORDS_CSV) -> pd.DataFrame:
    """metrics_dict keys: MCC_mean/std, AUROC_mean/std, F1_mean/std, Accuracy_mean/std"""
    df = _ensure_row(load_records(path), dataset)
    for m in RF_METRICS:
        mean_key = f"{m}_mean"
        std_key = f"{m}_std"
        if mean_key in metrics_dict:
            df.loc[dataset, f"{fm_col(model_name, m)}_mean"] = metrics_dict[mean_key]
        if std_key in metrics_dict:
            df.loc[dataset, f"{fm_col(model_name, m)}_std"] = metrics_dict[std_key]
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


# ── decomposition columns ─────────────────────────────────────────────────────

def load_decomp_records(path: str = RECORDS_DECOMP_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def has_decomp_ridge(dataset: str, k: int, model_name: str,
                     df: pd.DataFrame, pooling: str = "mean",
                     mapper: str = "ridge") -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = decomp_ridge_col(k, model_name, RIDGE_METRICS[0], pooling, mapper)
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_proj(dataset: str, k: int, model_name: str,
             df: pd.DataFrame, pooling: str = "mean",
             mapper: str = "ridge") -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = decomp_proj_col(k, model_name, RF_METRICS[0], pooling, mapper)
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_resid(dataset: str, k: int, model_name: str,
              df: pd.DataFrame, pooling: str = "mean",
              mapper: str = "ridge") -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = decomp_resid_col(k, model_name, RF_METRICS[0], pooling, mapper)
    return col in df.columns and pd.notna(df.loc[dataset, col])


DECOMP_RIDGE_METRICS = ["R2"]
DECOMP_RF_METRICS = ["MCC", "AUROC"]


def write_decomp_ridge(dataset: str, k: int, model_name: str,
                       ridge_metrics: dict,
                       pooling: str = "mean",
                       mapper: str = "ridge",
                       path: str = RECORDS_DECOMP_CSV) -> pd.DataFrame:
    """ridge_metrics keys: R2"""
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            df = _ensure_row(load_decomp_records(path), dataset)
            for m in DECOMP_RIDGE_METRICS:
                df.loc[dataset, decomp_ridge_col(k, model_name, m, pooling, mapper)] = ridge_metrics[m]
            _save(df, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return df


def write_proj(dataset: str, k: int, model_name: str, rf_metrics: dict,
               pooling: str = "mean",
               mapper: str = "ridge",
               path: str = RECORDS_DECOMP_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC"""
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            df = _ensure_row(load_decomp_records(path), dataset)
            for m in DECOMP_RF_METRICS:
                df.loc[dataset, decomp_proj_col(k, model_name, m, pooling, mapper)] = rf_metrics[m]
            _save(df, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return df


def write_resid(dataset: str, k: int, model_name: str, rf_metrics: dict,
                pooling: str = "mean",
                mapper: str = "ridge",
                path: str = RECORDS_DECOMP_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC"""
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            df = _ensure_row(load_decomp_records(path), dataset)
            for m in DECOMP_RF_METRICS:
                df.loc[dataset, decomp_resid_col(k, model_name, m, pooling, mapper)] = rf_metrics[m]
            _save(df, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return df


# ── PCA columns ───────────────────────────────────────────────────────────────

def pca_kmer_col(k: int, metric: str) -> str:
    return f"pca_kmer_k{k}_{metric}"


def pca_fm_col(model_name: str, metric: str) -> str:
    return f"pca_fm_{_model_tag(model_name)}_{metric}"


def has_pca_kmer(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = pca_kmer_col(k, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_pca_fm(dataset: str, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = pca_fm_col(model_name, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def load_pca_records(path: str = RECORDS_PCA_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def write_pca_kmer(dataset: str, k: int, rf_metrics: dict,
                   path: str = RECORDS_PCA_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_pca_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, pca_kmer_col(k, m)] = rf_metrics[m]
    _save(df, path)
    return df


def write_pca_fm(dataset: str, model_name: str, rf_metrics: dict,
                 path: str = RECORDS_PCA_CSV) -> pd.DataFrame:
    """rf_metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_pca_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, pca_fm_col(model_name, m)] = rf_metrics[m]
    _save(df, path)
    return df


# ── Linear Probe columns ──────────────────────────────────────────────────────

def lp_kmer_col(k: int, metric: str) -> str:
    return f"lp_kmer_k{k}_{metric}"


def lp_fm_col(model_name: str, metric: str) -> str:
    return f"lp_fm_{_model_tag(model_name)}_{metric}"


def lp_fm_pool_col(model_name: str, metric: str, pooling: str = "mean") -> str:
    base = lp_fm_col(model_name, metric)
    if pooling == "mean":
        return base
    return f"{base}__pool_{pooling}"


def load_lp_records(path: str = RECORDS_LINEAR_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def has_lp_kmer(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = lp_kmer_col(k, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_lp_fm(dataset: str, model_name: str, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = lp_fm_col(model_name, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_lp_fm_pool(dataset: str, model_name: str, df: pd.DataFrame,
                   pooling: str = "mean") -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = lp_fm_pool_col(model_name, RF_METRICS[0], pooling)
    return col in df.columns and pd.notna(df.loc[dataset, col])


def write_lp_kmer(dataset: str, k: int, metrics: dict,
                  path: str = RECORDS_LINEAR_CSV) -> pd.DataFrame:
    """metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_lp_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, lp_kmer_col(k, m)] = metrics[m]
    _save(df, path)
    return df


def write_lp_fm(dataset: str, model_name: str, metrics: dict,
                path: str = RECORDS_LINEAR_CSV) -> pd.DataFrame:
    """metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_lp_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, lp_fm_col(model_name, m)] = metrics[m]
    _save(df, path)
    return df


def write_lp_fm_pool(dataset: str, model_name: str, metrics: dict,
                     pooling: str = "mean",
                     path: str = RECORDS_LINEAR_CSV) -> pd.DataFrame:
    """metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_lp_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, lp_fm_pool_col(model_name, m, pooling)] = metrics[m]
    _save(df, path)
    return df


# ── Canonical k-mer columns ───────────────────────────────────────────────────

def canon_kmer_col(k: int, metric: str) -> str:
    return f"ckmer_k{k}_{metric}"


def canon_lp_kmer_col(k: int, metric: str) -> str:
    return f"ckmer_lp_k{k}_{metric}"


def load_canon_records(path: str = RECORDS_CANONICAL_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0)
        df.index.name = "dataset"
        return df
    return pd.DataFrame()


def has_canon_kmer(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = canon_kmer_col(k, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def has_canon_lp_kmer(dataset: str, k: int, df: pd.DataFrame) -> bool:
    if df.empty or dataset not in df.index:
        return False
    col = canon_lp_kmer_col(k, RF_METRICS[0])
    return col in df.columns and pd.notna(df.loc[dataset, col])


def write_canon_kmer(dataset: str, k: int, metrics: dict,
                     path: str = RECORDS_CANONICAL_CSV) -> pd.DataFrame:
    """metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_canon_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, canon_kmer_col(k, m)] = metrics[m]
    _save(df, path)
    return df


def write_canon_lp_kmer(dataset: str, k: int, metrics: dict,
                        path: str = RECORDS_CANONICAL_CSV) -> pd.DataFrame:
    """metrics keys: MCC, AUROC, F1, Accuracy"""
    df = _ensure_row(load_canon_records(path), dataset)
    for m in RF_METRICS:
        df.loc[dataset, canon_lp_kmer_col(k, m)] = metrics[m]
    _save(df, path)
    return df
