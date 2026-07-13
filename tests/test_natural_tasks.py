from pathlib import Path

from benchmark_v2.datasets.adapters import build_bank_marketing
from benchmark_v2.datasets.confirmatory_adapters import build_bts_flights
from benchmark_v2.core.models import LeakageLabel


def test_bank_prediction_boundary_marks_duration_post_outcome():
    task = build_bank_marketing()
    index = [spec.feature_id for spec in task.feature_specs].index("bm_duration")
    assert task.ground_truth[index].label == LeakageLabel.POST_OUTCOME
    assert task.availability[index].available_at_prediction is False
    assert task.lineage["is_synthetic"] is False


def test_natural_case_study_compares_strict_and_permissive_protocols():
    source = Path("experiments/leakbench/run_natural_case_studies.py").read_text()
    assert "strict_auc" in source
    assert "permissive_auc" in source
    assert "paired_harm" in source


def test_bts_schedule_and_actual_arrival_respect_prediction_boundary():
    task = build_bts_flights()
    by_name = {
        spec.name_pool.get("natural"): availability.available_at_prediction
        for spec, availability in zip(task.feature_specs, task.availability)
    }
    assert by_name["CRSArrTime"] is True
    assert by_name["ArrTime"] is False
    assert by_name["DepDelay"] is False
    assert by_name["Div1Airport"] is False
    assert task.lineage["prediction_boundary"] == "immediately_before_scheduled_departure"
