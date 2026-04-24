"""
Non-linear mapping from k-mer features to FM embeddings.

This module provides a compact 2-layer MLP used as an upper-bound mapper for
k-mer -> FM decomposition experiments.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score


class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_and_evaluate_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    hidden_dim: int = 512,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train a 2-layer MLP and return decomposition-compatible metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    Y_train_t = torch.from_numpy(Y_train.astype(np.float32))
    X_test_t = torch.from_numpy(X_test.astype(np.float32))

    model = _MLP(X_train_t.shape[1], Y_train_t.shape[1], hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t.to(device)).cpu().numpy().astype(np.float32)
        y_pred_test = model(X_test_t.to(device)).cpu().numpy().astype(np.float32)

    r2_global = r2_score(Y_test, y_pred_test, multioutput="uniform_average")
    mse_global = mean_squared_error(Y_test, y_pred_test)

    ss_res = np.sum((Y_test - y_pred_test) ** 2, axis=0)
    ss_tot = np.sum((Y_test - Y_test.mean(axis=0)) ** 2, axis=0)
    r2_per_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    mse_per_dim = ss_res / Y_test.shape[0]

    return {
        "r2_global": float(r2_global),
        "mse_global": float(mse_global),
        "r2_per_dim": r2_per_dim,
        "mse_per_dim": mse_per_dim,
        "r2_median_dim": float(np.median(r2_per_dim)),
        "best_alpha": np.nan,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_kmer_features": X_train.shape[1],
        "n_embed_dims": Y_test.shape[1],
        "model": model,
        "Y_pred_train": y_pred_train,
        "Y_pred_test": y_pred_test,
    }
