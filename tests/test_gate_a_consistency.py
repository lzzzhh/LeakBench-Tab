import pandas as pd


def test_m03_pilot_lineage_is_unique_and_condition_specific():
    cells = pd.read_csv("results/corrected_v2/pilot_protocol_v2_cells.csv")
    m03 = cells[cells["mechanism"] == "M03"]
    assert len(m03) == 3 * 5 * 4 * 3
    assert not m03.duplicated(["dataset_id", "mechanism", "strength", "model", "seed"]).any()
    assert m03["code_hash"].nunique() == 1
    assert m03["config_hash"].nunique() == 1


def test_normalized_average_precision_bounds():
    for raw, prevalence in ((1.0, 0.09), (0.5, 0.09), (0.1, 0.09), (0.05, 0.09)):
        normalized = (raw - prevalence) / (1 - prevalence)
        assert -1.0 <= normalized <= 1.0


def test_pilot_correlation_analysis_is_rebuilt_from_cells():
    summary = pd.read_csv("results/corrected_v2/pilot_statistics/mechanism_summary.csv")
    assert set(summary["mechanism"]) == {f"M{i:02d}" for i in range(1, 12)}
    assert summary["paired_harm"].notna().all()
    assert summary["diagnostic_normalized_ap"].between(-1, 1).all()
