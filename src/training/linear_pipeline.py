"""
Shared linear model training and evaluation utilities.

Analogue to rf_pipeline.py but uses regularised linear models.
StandardScaler is mandatory — linear models are scale-sensitive.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_LP_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def train_linear_classifier(X_train, y_train, n_classes, n_jobs=4):
    """
    Pipeline: StandardScaler -> LogisticRegressionCV(L2).
    Returns a fitted sklearn Pipeline.
    n_jobs: parallel CV folds (use 4-8 to balance per-job BLAS threads).
    """
    scoring = "roc_auc" if n_classes == 2 else "accuracy"
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    clf = LogisticRegressionCV(
        Cs=_LP_CS,
        cv=cv,
        scoring=scoring,
        solver="lbfgs",
        max_iter=500,
        tol=1e-3,
        n_jobs=n_jobs,
        random_state=42,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def eval_linear_classifier(model, X_test, y_test, n_classes):
    """Evaluate a classifier. Returns {MCC, AUROC, F1, Accuracy}."""
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


def train_linear_regressor(X_train, y_train):
    """
    Pipeline: StandardScaler -> RidgeCV.
    Returns a fitted sklearn Pipeline.
    """
    reg = RidgeCV(alphas=_RIDGE_ALPHAS, cv=5)
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
    pipe.fit(X_train, y_train)
    return pipe


def eval_linear_regressor(model, X_test, y_test):
    """Evaluate a regressor. Returns {R2, MSE, Spearman}."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = float(np.mean((y_test - y_pred) ** 2))
    spearman = float(spearmanr(y_test, y_pred).statistic)
    return {"R2": r2, "MSE": mse, "Spearman": spearman}
