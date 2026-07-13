import pytest

from paper.aaai27.source_data import generate_result_tables as tables


def test_frozen_task_registry_table_has_twenty_tasks_and_bound_sources():
    rendered, sources = tables.task_registry()
    assert rendered.count("panel\\_") == 20
    assert len(sources) == 22  # manifest, v2 freeze, and twenty task bundles
    assert "pilot" not in rendered.lower()
    assert "tab:complete-task-registry" in rendered


def test_public_natural_table_has_five_fixed_cases_and_no_private_paths():
    rendered, sources = tables.natural_cases()
    for task in ("BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"):
        assert rendered.count(task) == 1
    assert "/Users/" not in rendered
    assert ":\\" not in rendered
    assert len(sources) == 3


def test_interval_formatter_rejects_point_outside_interval():
    with pytest.raises(tables.TableError, match="Unordered interval"):
        tables.interval(0.6, 0.1, 0.5)


def test_final_table_gate_is_closed_until_claim_release_exists():
    if tables.CLAIMS.is_file():
        pytest.skip("Final claim release is present in this checkout")
    with pytest.raises(tables.TableError, match="Required file is missing"):
        tables.validate_claims()
