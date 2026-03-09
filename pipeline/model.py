import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    matthews_corrcoef,
    roc_auc_score,
    f1_score,
    accuracy_score,
)

# Griglia identica a classify_cadph.py del paper (Feng et al., 2025)
_RF_PARAM_GRID = {
    "n_estimators": [1000, 500, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [20, None],
    "min_samples_split": [2, 5],
}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_jobs: int = -1,
) -> dict:
    """
    Addestra un Random Forest con GridSearchCV (4-fold CV sul training set)
    e valuta sul test set.

    Strategia identica a quella del paper (Feng et al., 2025):
    - Binario: scoring='roc_auc', AUC calcolato su probabilità classe 1
    - Multiclasse: scoring='accuracy', AUC con multi_class='ovr'

    Returns:
        Dict con mcc, auroc, f1_macro, accuracy, n_classes, train_size, test_size.
    """
    n_classes = len(np.unique(y_train))
    is_binary = n_classes == 2

    scoring = "roc_auc" if is_binary else "accuracy"

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=n_jobs),
        param_grid=_RF_PARAM_GRID,
        cv=4,
        scoring=scoring,
        n_jobs=1,  # parallelismo già in RF via n_jobs
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)

    metrics = {
        "mcc": matthews_corrcoef(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "accuracy": accuracy_score(y_test, y_pred),
        "n_classes": n_classes,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "best_params": str(gs.best_params_),
    }

    try:
        if is_binary:
            metrics["auroc"] = roc_auc_score(y_test, y_prob[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="macro"
            )
    except ValueError:
        metrics["auroc"] = float("nan")

    return metrics
