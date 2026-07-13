from pathlib import Path

from benchmark_v2.datasets import adapters
from benchmark_v2.datasets.confirmatory_adapters import build_nyc_311


def test_real_file_selection_is_deterministic_and_not_adapter_limited():
    lending = adapters._select_lending_file(Path("data/raw/lending_club"))
    bank = adapters._select_bank_file(Path("data/raw/bank_marketing"))
    assert lending is not None and lending.name.lower().startswith("accepted")
    assert bank is not None and bank.name == "bank-additional-full.csv"


def test_nyc_identity_and_lineage_are_real_311():
    task = build_nyc_311()
    assert task.name == "NYC311"
    assert task.lineage["dataset"] == "NYC311"
    assert task.lineage["is_synthetic"] is False


def test_natural_runner_prioritizes_rank_and_paired_harm_metrics():
    source = Path("experiments/leakbench/run_natural_case_studies.py").read_text()
    for metric in ("mrr", "top5_recall", "diagnostic_normalized_ap", "paired_harm"):
        assert metric in source
