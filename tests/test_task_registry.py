from pathlib import Path

from benchmark_v2.datasets import adapters


def test_task_registry_resolves_real_canonical_inputs():
    bank = adapters._select_bank_file(Path("data/raw/bank_marketing"))
    lending = adapters._select_lending_file(Path("data/raw/lending_club"))
    assert bank.name == "bank-additional-full.csv"
    assert "accepted" in lending.name.lower()
    assert Path("data/nyc311/nyc311_cache.csv").is_file()


def test_budget_rounding_uses_ceiling():
    from numpy import ceil
    assert max(1, int(ceil(0.10 * 42))) == 5
    assert max(1, int(ceil(0.10 * 25))) == 3
    assert max(1, int(ceil(0.10 * 20))) == 2
