#!/usr/bin/env python3
"""Demonstrate the R² calculation bug."""

import numpy as np
from sklearn.metrics import r2_score

# Create synthetic data to illustrate the bug
np.random.seed(42)
n_samples = 100
n_dims = 1920

# Y_true: random embeddings
Y_true = np.random.randn(n_samples, n_dims)

# Y_pred: completely random predictions (should have R² ≈ 0 per-dim, negative overall)
Y_pred = np.random.randn(n_samples, n_dims)

# Method 1 (WRONG - as in parallel script): global SS
ss_res_global = np.sum((Y_true - Y_pred) ** 2)
ss_tot_global = np.sum((Y_true - np.mean(Y_true)) ** 2)
r2_wrong = 1.0 - (ss_res_global / ss_tot_global)

# Method 2 (CORRECT - sklearn): per-dimension average
r2_correct = r2_score(Y_true, Y_pred, multioutput='uniform_average')

print(f"Wrong R² (global SS): {r2_wrong:.6f}")
print(f"Correct R² (per-dim average): {r2_correct:.6f}")

# Check per-dimension R² directly
per_dim_r2 = []
for i in range(n_dims):
    ss_res_i = np.sum((Y_true[:, i] - Y_pred[:, i]) ** 2)
    ss_tot_i = np.sum((Y_true[:, i] - Y_true[:, i].mean()) ** 2)
    r2_i = 1.0 - (ss_res_i / ss_tot_i)
    per_dim_r2.append(r2_i)

print(f"Per-dim R² (manual): mean={np.mean(per_dim_r2):.6f}, median={np.median(per_dim_r2):.6f}")
print()
print("=== THE BUG ===")
print(f"The parallel script uses global SS which gives R² = {r2_wrong:.4f}")
print(f"But the correct multioutput R² is {r2_correct:.4f}")
print(f"These are VERY different!")
