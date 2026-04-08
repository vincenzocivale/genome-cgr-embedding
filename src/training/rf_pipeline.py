"""
Shared Random Forest training and evaluation utilities.

All classification training scripts use the same RF configuration and
evaluation protocol. This module centralises those definitions.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

_RF_PARAM_GRID = {
    "n_estimators":      [200, 500],
    "max_features":      ["sqrt"],
    "max_depth":         [20],
    "min_samples_split": [2],
}


def train_rf(X_train, y_train, n_classes):
    """Train a Random Forest classifier with GridSearchCV."""
    scoring = "roc_auc" if n_classes == 2 else "accuracy"
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring=scoring, cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def train_rf_fast(X_train, y_train, n_classes, n_jobs=4):
    """Train RF without grid search (500 trees, sqrt features, depth 20)."""
    rf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", max_depth=20,
        min_samples_split=2, n_jobs=n_jobs, random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf


def eval_rf(model, X_test, y_test, n_classes):
    """Evaluate a classifier on the test set. Returns MCC, AUROC, F1, Accuracy."""
    y_pred = model.predict(X_test)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    if n_classes == 2:
        auroc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auroc = roc_auc_score(
            y_test, model.predict_proba(X_test), multi_class="ovr", average="macro"
        )
    return {"MCC": mcc, "AUROC": auroc, "F1": f1, "Accuracy": acc}


def train_rf_regressor(X_train, y_train):
    """Train a Random Forest regressor with GridSearchCV."""
    cv = KFold(n_splits=4, shuffle=True, random_state=42)
    gs = GridSearchCV(
        RandomForestRegressor(n_jobs=4, random_state=42),
        _RF_PARAM_GRID, scoring="r2", cv=cv, n_jobs=1, refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def eval_rf_regressor(model, X_test, y_test):
    """Evaluate a regressor on the test set. Returns R2, MSE, Spearman."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = float(np.mean((y_test - y_pred) ** 2))
    spearman = float(spearmanr(y_test, y_pred).statistic)
    return {"R2": r2, "MSE": mse, "Spearman": spearman}


# Aliases used by train_lra_benchmark.py
train_rf_classifier = train_rf
eval_rf_classifier = eval_rf
