"""Create the requested E4--E6 tables and figures from tidy result files."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon


def task_group(name: str) -> str:
    n = name.lower()
    if "prom" in n: return "promoter"
    if "splice" in n or "acceptor" in n or "donor" in n: return "splice-site"
    if any(x in n for x in ("regulatory", "enhancer", "tfbs", "chromatin", "dnase", "dhs")): return "regulatory"
    return "other"


def e4(path: Path, out: Path):
    df = pd.read_csv(path); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["model", "probe", "representation"], as_index=False)["MCC"].agg(["mean", "median", "count"]).reset_index()
    summary.to_csv(out / "ranking_by_probe.csv", index=False)
    paired = df.pivot_table(index=["dataset", "model", "probe"], columns="representation", values="MCC").reset_index()
    if {"kmer", "fm_embedding"}.issubset(paired):
        paired["kmer_minus_fm"] = paired["kmer"] - paired["fm_embedding"]
        interaction = paired.groupby(["model", "probe"], as_index=False)["kmer_minus_fm"].agg(["mean", "median", "count", "std"]).reset_index()
        interaction.to_csv(out / "representation_classifier_interaction.csv", index=False)
        plt.figure(figsize=(8, 4))
        probes = list(paired.probe.unique())
        for i, model in enumerate(paired.model.unique()):
            vals = [paired[(paired.model == model) & (paired.probe == probe)].kmer_minus_fm.dropna() for probe in probes]
            plt.boxplot(vals, positions=[j + 1 + i * .18 for j in range(len(probes))], widths=.15)
        plt.xticks(range(1, len(probes) + 1), probes)
        plt.axhline(0, color="black", lw=.8); plt.ylabel("MCC(k-mer) − MCC(FM)"); plt.tight_layout(); plt.savefig(fig / "fm_vs_kmer_by_probe.pdf"); plt.close()


def _pooling_significance(df: pd.DataFrame, group_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank (mean pooling vs each alternative, per
    dataset) with BH correction across all rows produced in one call.

    Mirrors the convention in src/analysis/fdr_analysis.py (two-sided,
    zero_method="wilcox", >=5 paired datasets, significant at adjusted_p<.05)
    so the claim that a pooling differs from mean is a statistical one, not
    just a difference in means.
    """
    keys = ["model", "probe", *group_cols]
    rows = []
    for group_vals, g in df.groupby(keys):
        piv = g.pivot_table(index="dataset", columns="pooling", values="MCC")
        if "mean" not in piv.columns:
            continue
        for pooling in [c for c in piv.columns if c != "mean"]:
            paired = piv[["mean", pooling]].dropna()
            if len(paired) < 5 or (paired[pooling] == paired["mean"]).all():
                continue
            _, p = wilcoxon(paired[pooling], paired["mean"], alternative="two-sided", zero_method="wilcox")
            delta = paired[pooling] - paired["mean"]
            row = dict(zip(keys, group_vals))
            row.update(pooling=pooling, n_datasets=len(paired), mean_delta=float(delta.mean()),
                       median_delta=float(delta.median()), pooling_wins=int((delta > 0).sum()),
                       mean_wins=int((delta < 0).sum()), raw_p=float(p))
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["adjusted_p"] = false_discovery_control(out["raw_p"], method="bh")
        out["significant"] = out["adjusted_p"] < 0.05
    return out


def e5(path: Path, out: Path, e4_path: Path | None):
    df = pd.read_csv(path); df["task_group"] = df.dataset.map(task_group); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    table = df.groupby(["model", "pooling", "probe"], as_index=False)["MCC"].agg(["mean", "median", "count"]).reset_index()
    table.to_csv(out / "model_pooling_performance.csv", index=False)
    best = table.loc[table.groupby(["model", "probe"])["mean"].idxmax()].copy(); best.to_csv(out / "best_pooling_by_model.csv", index=False)
    df.groupby(["model", "pooling", "probe", "task_group"], as_index=False)["MCC"].mean().to_csv(out / "pooling_by_task_group.csv", index=False)
    # Best pooling per individual task (dataset), not only averaged model-wide.
    best_by_task = df.loc[df.groupby(["dataset", "model", "probe"])["MCC"].idxmax()].copy()
    best_by_task.to_csv(out / "best_pooling_by_task.csv", index=False)
    # Whether an alternative pooling beats mean is a claim that needs a paired
    # significance test, not just a higher average MCC.
    _pooling_significance(df).to_csv(out / "pooling_significance.csv", index=False)
    _pooling_significance(df, group_cols=("task_group",)).to_csv(out / "pooling_significance_by_task_group.csv", index=False)
    plt.figure(figsize=(max(7, len(table.model.unique()) * 2.5), 4))
    pivot = table.pivot(index="pooling", columns="model", values="mean")
    pivot.plot.bar(ax=plt.gca())
    plt.ylabel("Mean test MCC"); plt.tight_layout(); plt.savefig(fig / "pooling_comparison.pdf"); plt.close()
    if e4_path and e4_path.exists():
        e4 = pd.read_csv(e4_path); km = e4[e4.representation.eq("kmer")][["dataset", "model", "probe", "MCC"]].rename(columns={"MCC":"kmer_MCC"})
        selected = df.merge(best[["model", "probe", "pooling"]], on=["model", "probe", "pooling"])
        selected.merge(km, on=["dataset", "model", "probe"], how="inner").to_csv(out / "best_pooling_vs_kmer.csv", index=False)


def e6(path: Path, out: Path):
    df = pd.read_csv(path); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "layerwise_tidy_copy.csv", index=False)
    for value, filename, title in (("MCC", "layerwise_mcc.pdf", "Downstream MCC"), ("mapping_r2", "layerwise_r2.pdf", "k-mer → embedding R²")):
        values = df[df.representation.eq("fm_embedding")].groupby(["model", "layer"], as_index=False)[value].mean()
        matrix = values.pivot(index="model", columns="layer", values=value)
        plt.figure(figsize=(max(5, matrix.shape[1]*1.2), max(2.5, matrix.shape[0]*.7)))
        ax = plt.gca(); image = ax.imshow(matrix.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(matrix.shape[1]), matrix.columns); ax.set_yticks(range(matrix.shape[0]), matrix.index)
        for (i, j), value in __import__('numpy').ndenumerate(matrix.values): ax.text(j, i, f"{value:.2f}", ha="center", va="center")
        plt.colorbar(image); plt.title(title); plt.tight_layout(); plt.savefig(fig / filename); plt.close()
    curve = df[df.representation.eq("fm_embedding")].groupby(["model", "depth_fraction"], as_index=False)[["MCC", "mapping_r2"]].mean()
    curve.to_csv(out / "layerwise_curves.csv", index=False)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--e4", type=Path); p.add_argument("--e5", type=Path); p.add_argument("--e6", type=Path); p.add_argument("--out-root", type=Path, default=Path("results/rebuttal")); a=p.parse_args()
    if a.e4: e4(a.e4, a.out_root / "E4_probe_fairness")
    if a.e5: e5(a.e5, a.out_root / "E5_pooling", a.e4)
    if a.e6: e6(a.e6, a.out_root / "E6_layerwise")

if __name__ == "__main__": main()
