"""
Ridge regression mapping from k-mer features to FM embeddings.

Uses RidgeCV with efficient GCV (Generalized Cross-Validation) which
avoids materialising large LOO matrices — O(N * d * n_alphas) memory
instead of O(N² * d).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error


ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def fit_and_evaluate(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    alphas: list | None = None,
) -> dict:
    """
    Fit RidgeCV (multi-output) on train, evaluate on test.

    Parameters
    ----------
    X_train, X_test : (N, d_kmer) k-mer feature arrays
    Y_train, Y_test : (N, d_embed) FM embedding arrays
    alphas : regularisation candidates (default ALPHAS)

    Returns
    -------
    dict with r2_global, mse_global, r2_per_dim, mse_per_dim, best_alpha, etc.
    """
    if alphas is None:
        alphas = ALPHAS

    ridge = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="auto")
    ridge.fit(X_train, Y_train)

    Y_pred_test = ridge.predict(X_test)
    Y_pred_train = ridge.predict(X_train)

    r2_global = r2_score(Y_test, Y_pred_test, multioutput="uniform_average")
    mse_global = mean_squared_error(Y_test, Y_pred_test)

    n_dims = Y_test.shape[1]
    # Vectorised per-dim metrics (avoids Python loop over 1024 dims)
    ss_res = np.sum((Y_test - Y_pred_test) ** 2, axis=0)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    mse_per_dim = ss_res / Y_test.shape[0]

    return {
        "r2_global": r2_global,
        "mse_global": mse_global,
        "r2_per_dim": r2_per_dim,
        "mse_per_dim": mse_per_dim,
        "r2_median_dim": float(np.median(r2_per_dim)),
        "best_alpha": float(ridge.alpha_),
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_kmer_features": X_train.shape[1],
        "n_embed_dims": n_dims,
        "model": ridge,
        "Y_pred_train": Y_pred_train,
        "Y_pred_test": Y_pred_test,
    }
