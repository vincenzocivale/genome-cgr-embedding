"""
Utility to log feature extraction efficiency.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime


EFFICIENCY_CSV = "results/efficiency.csv"


def log_efficiency(dataset: str, method: str, feat_dim: int,
                   time_sec: float, path: str = EFFICIENCY_CSV) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "dataset", "method", "feat_dim", "time_sec"])
        writer.writerow([datetime.utcnow().isoformat(), dataset, method,
                         int(feat_dim), float(time_sec)])
