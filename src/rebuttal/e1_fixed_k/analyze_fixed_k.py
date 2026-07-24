"""
Rebuttal E1 — Fixed k for all tasks.

Reviewers HnDT and eZYW ask whether the k-mer baseline's advantage over FM
embeddings depends on picking the best k per task ("best-k": the repo's default,
a post-hoc max over the k=4,5,6 test-set metrics — see `best_fixed_k` /
`best_kmer_metric` in src/records/records.py). This reruns the paired FM-vs-k-mer
comparison (win/loss counts, mean/median delta, BH-corrected Wilcoxon) separately
for a single global k in {4, 5, 6}, and compares against the existing best-k result.

No new training: reuses `records_rf.csv` / `records_linear_probe.csv`, which already
contain kmer_k4/k5/k6 columns for all 57 datasets.

Output:
    results/rebuttal/E1_fixed_k/fdr_results_{mode}_{probe_type}.csv  (mode in k4,k5,k6,best)
    results/rebuttal/E1_fixed_k/best_k_per_task.csv
    results/rebuttal/E1_fixed_k/E1_summary.md

Usage:
    python3 src/rebuttal/e1_fixed_k/analyze_fixed_k.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.analysis.fdr_analysis import run_fdr_analysis
from src.analysis.paper_utils import infer_biocat, CATEGORY_LABELS
from src.records.records import RECORDS_CSV, RECORDS_LINEAR_CSV, best_fixed_k, best_kmer_metric

OUT_DIR = "results/rebuttal/E1_fixed_k"
MODES = ["k4", "k5", "k6", "best"]
METRICS = ["MCC", "AUROC"]
RECORDS = [
    (RECORDS_CSV, "rf", "kmer"),
    (RECORDS_LINEAR_CSV, "linear_probe", "lp_kmer"),
]


def run_all_modes():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    for mode in MODES:
        for records_path, probe_type, kmer_prefix in RECORDS:
            for metric in METRICS:
                result = run_fdr_analysis(
                    records_path, metric=metric, kmer_prefix=kmer_prefix, kmer_mode=mode,
                )
                if result.empty:
                    continue
                result.insert(0, "metric", metric)
                result.insert(0, "probe_type", probe_type)
                result.insert(0, "kmer_mode", mode)
                all_results.append(result)

    if not all_results:
        print("No results generated.")
        return pd.DataFrame()

    final = pd.concat(all_results, ignore_index=True)
    for mode in MODES:
        sub = final[final["kmer_mode"] == mode]
        out_path = os.path.join(OUT_DIR, f"fdr_results_{mode}.csv")
        sub.drop(columns="kmer_mode").to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    combined_path = os.path.join(OUT_DIR, "fdr_results_all_modes.csv")
    final.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}")
    return final


def write_best_k_table():
    df = pd.read_csv(RECORDS_CSV, index_col=0)
    df.index.name = "dataset"
    out = pd.DataFrame(index=df.index)
    out["biocat"] = out.index.to_series().map(infer_biocat)
    out["biocat_label"] = out["biocat"].map(CATEGORY_LABELS)
    for metric in METRICS:
        out[f"best_k_{metric}"] = best_fixed_k(df, metric=metric)
        out[f"best_kmer_{metric}"] = best_kmer_metric(df, metric=metric)
        for mode in ("k4", "k5", "k6"):
            out[f"kmer_{mode}_{metric}"] = df.get(f"kmer_{mode}_{metric}")
    out_path = os.path.join(OUT_DIR, "best_k_per_task.csv")
    out.to_csv(out_path)
    print(f"Wrote {out_path}")
    return out


def write_summary(final: pd.DataFrame, best_k_table: pd.DataFrame):
    lines = ["# E1 — Fixed k for all tasks: summary\n"]
    lines.append(
        "Success criterion: main conclusions (which method wins more datasets, "
        "sign/magnitude of the median delta, statistical significance) hold using "
        "at least one single global k, not only the post-hoc best-k selection.\n"
    )

    lines.append("## Win/loss counts and effect size by k-mer selection mode (RF, MCC)\n")
    sub = final[(final["probe_type"] == "rf") & (final["metric"] == "MCC")]
    if not sub.empty:
        cols = ["kmer_mode", "comparison", "n_datasets", "kmer_wins", "fm_wins",
                "mean_delta", "median_delta", "adjusted_p", "significant"]
        lines.append("```\n" + sub[cols].to_string(index=False) + "\n```")
        lines.append("")

    lines.append("\n## Best-k distribution across the 57 tasks (MCC)\n")
    if "best_k_MCC" in best_k_table.columns:
        counts = best_k_table["best_k_MCC"].value_counts().sort_index()
        lines.append("```\n" + counts.to_string() + "\n```")
        lines.append("")

    out_path = os.path.join(OUT_DIR, "E1_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    final = run_all_modes()
    best_k_table = write_best_k_table()
    if not final.empty:
        write_summary(final, best_k_table)


if __name__ == "__main__":
    main()
