from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from src.records.records import best_fixed_k, best_kmer_metric, load_decomp_records, load_records


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_canonical_result_files_exist_and_load():
    paths = [
        ROOT / "results/classification/records_rf.csv",
        ROOT / "results/classification/records_linear_probe.csv",
        ROOT / "results/classification/records_canonical_kmer.csv",
        ROOT / "results/classification/records_pca.csv",
        ROOT / "results/decomposition/records_decomposition.csv",
        ROOT / "results/analysis/fdr_results.csv",
        ROOT / "results/analysis/auroc_consistency.csv",
        ROOT / "results/concat/records_concat_best.csv",
        ROOT / "results/classification/truncation_analysis.csv",
        ROOT / "results/exploratory/splice_residual_motif_overlap.csv",
        ROOT / "results/efficiency/efficiency.csv",
        ROOT / "results/efficiency/efficiency_gpu_parallel.csv",
    ]
    for path in paths:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_best_k_is_recomputed_from_records_rf():
    records = load_records(ROOT / "results/classification/records_rf.csv")
    best_k = best_fixed_k(records)
    best_mcc = best_kmer_metric(records, metric="MCC")

    assert not best_k.dropna().empty
    assert set(best_k.dropna().astype(int).unique()).issubset({4, 5, 6})
    assert not best_mcc.dropna().empty
    assert best_mcc.notna().sum() == len(records)


def test_decomposition_records_load_from_canonical_path():
    records = load_decomp_records(ROOT / "results/decomposition/records_decomposition.csv")
    assert not records.empty
    assert "ridge_k6_NTv3_650M_pre_R2" in records.columns


def test_paper_stats_audit_smoke(capsys):
    module = _load_module(ROOT / "scripts/paper_stats_audit.py", "paper_stats_audit")
    module.main()
    output = capsys.readouterr().out
    assert "Paper Statistics Audit" in output
    assert "Classification" in output


def test_export_paper_figures_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    module = _load_module(ROOT / "scripts/export_paper_figures.py", "export_paper_figures")
    module.OUT_DIR = tmp_path
    module.main()
    assert (tmp_path / "01-fm-vs-kmer-by-model-multipanel.pdf").exists()
    assert (tmp_path / "10_efficiency_gflops_per_seq.pdf").exists()
