import csv
import os
from datetime import datetime


COLUMNS = [
    "task",
    "method",
    "k",
    "grid_size",
    "feature_dim",
    "train_size",
    "test_size",
    "n_classes",
    "mcc",
    "auroc",
    "f1_macro",
    "accuracy",
    "best_params",
    "timestamp",
]


def save_result(filepath: str, result: dict):
    """Salva un risultato in append al CSV. Crea il file con header se non esiste."""
    result["timestamp"] = datetime.now().isoformat()

    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: result.get(k, "") for k in COLUMNS})
