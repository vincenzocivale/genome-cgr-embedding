"""
Ridge regression mapping from k-mer features to FM embeddings.

Uses RidgeCV with efficient GCV (Generalized Cross-Validation) which
avoids materialising large LOO matrices — O(N * d * n_alphas) memory
instead of O(N² * d).
"""

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

    ridge = RidgeCV(alphas=alphas, fit_intercept=True, gcv_mode="svd")
    ridge.fit(X_train, Y_train)

    Y_pred = ridge.predict(X_test)

    r2_global = r2_score(Y_test, Y_pred, multioutput="uniform_average")
    mse_global = mean_squared_error(Y_test, Y_pred)

    n_dims = Y_test.shape[1]
    r2_per_dim = np.array([r2_score(Y_test[:, d], Y_pred[:, d]) for d in range(n_dims)])
    mse_per_dim = np.array(
        [mean_squared_error(Y_test[:, d], Y_pred[:, d]) for d in range(n_dims)]
    )

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
    }
