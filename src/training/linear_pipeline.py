"""
Shared linear classification utilities.

StandardScaler is mandatory because the retained public probes are
scale-sensitive linear models.
"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_LP_CS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


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
