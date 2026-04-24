import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


RF_PATH = Path("results/classification/records_rf.csv")
BACKUP_DIR = Path("results/exploratory/plan_runs")


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"records_rf.before_exp3_restart.{timestamp}.csv"
    shutil.copy2(RF_PATH, backup_path)

    df = pd.read_csv(RF_PATH, index_col=0)

    drop_cols = []
    prefixes = [
        "kmer_k4_",
        "kmer_k5_",
        "kmer_k6_",
        "fm_DNABERT-2-117M_",
    ]
    for col in df.columns:
        if not (col.endswith("_std") or col.endswith("_mean")):
            continue
        if any(col.startswith(prefix) for prefix in prefixes):
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)
        df.to_csv(RF_PATH, index=True)

    print(f"backup={backup_path}")
    print(f"dropped={len(drop_cols)}")


if __name__ == "__main__":
    main()
