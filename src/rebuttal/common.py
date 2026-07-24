"""Shared, deliberately fixed protocol for the rebuttal experiments.

The functions here keep the split, scaling, selection metric and number of
candidate configurations identical *within each representation/probe cell*.
They are intentionally independent of the paper's historical result tables.
"""
from __future__ import annotations

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
N_CV_SPLITS = 4
N_CANDIDATES = 4


def load_cached_embedding(model_name: str, dataset_name: str, split: str,
                          pooling: str = "mean") -> np.ndarray | None:
    """Torch-free cache reader, useful for E4 reruns on existing embeddings."""
    model = model_name.replace("/", "__")
    dataset = dataset_name.replace("/", "__").replace("\\", "__")
    paths = [
        f"cache/fm_embeddings/{model}/pooling_{pooling}/{dataset}/{split}.npz",
    ]
    # These legacy caches predate the pooling-aware subdirectory and only
    # ever contain mean-pooled embeddings; only use them as a fallback when
    # mean pooling was actually requested, otherwise a non-mean pooling that
    # hasn't been computed yet would silently resolve to mean embeddings.
    if pooling == "mean":
        paths.append(f"cache/fm_embeddings/{model}/{dataset}/{split}.npz")
        # Historical HyenaDNA and Evo2 runners used model-specific cache
        # roots rather than FMEmbedder's common directory.  Rebuttal
        # comparisons must consume those already-computed embeddings instead
        # of silently omitting the corresponding paper models.
        if "hyenadna" in model_name.lower():
            paths.append(f"cache/hyena_embeddings/{dataset}/{split}.npz")
        if model_name.lower().startswith("evo2_"):
            paths.append(f"cache/{model}/{dataset}/{split}.npz")
        # This legacy flat cache predates per-model cache names and contains
        # the NTv3 embeddings.  It must never be used as a fallback for
        # another model.
        if model_name == "InstaDeepAI/NTv3_650M_pre":
            paths.append(f"cache/fm_embeddings/{dataset}/{split}.npz")
    elif "hyenadna" in model_name.lower():
        paths.append(f"cache/hyena_embeddings/pooling_{pooling}/{dataset}/{split}.npz")
    for path in paths:
        if os.path.exists(path):
            return np.load(path)["embeddings"].astype(np.float32)
    return None


def selection_metric(n_classes: int) -> str:
    return "roc_auc" if n_classes == 2 else "accuracy"


def _estimator_and_grid(probe: str):
    # Exactly four configurations per probe.  Scaling is applied before every
    # estimator (including RF) so preprocessing cannot explain a comparison.
    if probe == "rf":
        return RandomForestClassifier(random_state=SEED, n_jobs=1), {
            "clf__n_estimators": [200, 500], "clf__max_depth": [12, 20],
            "clf__max_features": ["sqrt"],
        }
    if probe == "linear":
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED), {
            "clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"],
        }
    if probe == "mlp":
        return MLPClassifier(max_iter=50, early_stopping=True, validation_fraction=0.15,
                             n_iter_no_change=10, batch_size=256, random_state=SEED), {
            "clf__hidden_layer_sizes": [(16,), (32,)],
            "clf__alpha": [1e-4, 1e-3],
        }
    raise ValueError(f"Unknown probe: {probe}")


def fit_probe(X_train, y_train, n_classes: int, probe: str, n_jobs: int = 1):
    estimator, grid = _estimator_and_grid(probe)
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=SEED)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    model = GridSearchCV(pipe, grid, scoring=selection_metric(n_classes), cv=cv,
                         n_jobs=n_jobs, refit=True, error_score="raise")
    model.fit(np.asarray(X_train, dtype=np.float32), y_train)
    return model


def evaluate_classifier(model, X_test, y_test, n_classes: int) -> dict[str, float]:
    pred = model.predict(np.asarray(X_test, dtype=np.float32))
    prob = model.predict_proba(np.asarray(X_test, dtype=np.float32))
    auroc = (roc_auc_score(y_test, prob[:, 1]) if n_classes == 2 else
             roc_auc_score(y_test, prob, multi_class="ovr", average="macro"))
    return {"MCC": float(matthews_corrcoef(y_test, pred)), "AUROC": float(auroc),
            "F1": float(f1_score(y_test, pred, average="macro")),
            "Accuracy": float(accuracy_score(y_test, pred))}


def project_kmers(X_train, Y_train, X_test, Y_test):
    """Fit the k-mer-to-FM map only on train data and return proj/residual."""
    # Scaling X is crucial for a regularised mapping and is fitted only on train.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(np.asarray(X_train, dtype=np.float32))
    Xte = scaler.transform(np.asarray(X_test, dtype=np.float32))
    ridge = RidgeCV(alphas=[0.01, 0.1, 1., 10., 100., 1000.]).fit(Xtr, Y_train)
    proj_train, proj_test = ridge.predict(Xtr), ridge.predict(Xte)
    return proj_train, proj_test, Y_train - proj_train, Y_test - proj_test, {
        "r2": float(r2_score(Y_test, proj_test, multioutput="uniform_average")),
        "ridge_alpha": float(ridge.alpha_),
    }
