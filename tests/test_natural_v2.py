import zipfile

import numpy as np
import pandas as pd
import pytest

from benchmark_v2.core.models import LeakageLabel
from benchmark_v2.datasets import adapters, confirmatory_adapters


def _feature_column(task, feature_id):
    index = [spec.feature_id for spec in task.feature_specs].index(feature_id)
    return task.X[:, index]


def test_lending_requires_real_accepted_input_unless_explicitly_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "DATA_DIR", tmp_path)
    rejected = tmp_path / "raw" / "lending_club" / "rejected.csv"
    rejected.parent.mkdir(parents=True)
    rejected.write_text("id,loan_status\n1,Fully Paid\n")

    with pytest.raises(FileNotFoundError, match="accepted Lending Club"):
        adapters.build_lending_club()

    task = adapters.build_lending_club(allow_synthetic=True)
    assert task.name == "LendingClub_SYNTHETIC"
    assert task.lineage["is_synthetic"] is True


def test_lending_reads_accepted_gzip_filters_unresolved_and_sorts_by_issue_date(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "DATA_DIR", tmp_path)
    raw = tmp_path / "raw" / "lending_club"
    raw.mkdir(parents=True)
    pd.DataFrame({"id": [999], "loan_status": ["Fully Paid"]}).to_csv(raw / "rejected_oversized.csv", index=False)
    accepted = raw / "accepted_fixture.csv.gz"
    pd.DataFrame({
        "id": [1, 2, 3, 4],
        "issue_d": ["Mar-2018", "Feb-2018", "Jan-2018", "Feb-2018"],
        "loan_status": ["Fully Paid", "Current", "Charged Off", "Fully Paid"],
        "annual_inc": [10.0, 20.0, 30.0, 40.0],
        "recoveries": [0.0, 0.0, 25.0, 0.0],
        "last_credit_pull_d": ["Apr-2018"] * 4,
        "payment_plan_start_date": ["May-2018"] * 4,
    }).to_csv(accepted, index=False, compression="gzip")

    task = adapters.build_lending_club(max_rows=20)

    np.testing.assert_array_equal(_feature_column(task, "lc_id"), [3.0, 4.0, 1.0])
    np.testing.assert_array_equal(task.y, [1.0, 0.0, 0.0])
    assert task.source == str(accepted.resolve())
    assert task.lineage["rows_scanned"] == 4
    assert task.lineage["resolved_outcome_rows"] == 3
    assert task.lineage["sort_column"] == "issue_d"
    assert task.lineage["time_min"].startswith("2018-01")
    assert len(task.lineage["source_sha256"]) == 64
    recovery_index = [spec.feature_id for spec in task.feature_specs].index("lc_recoveries")
    assert task.ground_truth[recovery_index].label == LeakageLabel.POST_OUTCOME
    assert task.availability[recovery_index].available_at_prediction is False
    for feature_id in ("lc_last_credit_pull_d", "lc_payment_plan_start_date"):
        index = [spec.feature_id for spec in task.feature_specs].index(feature_id)
        assert task.ground_truth[index].label == LeakageLabel.POST_OUTCOME
        assert task.availability[index].available_at_prediction is False


def test_lending_temporal_systematic_sample_spans_observed_period(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "DATA_DIR", tmp_path)
    raw = tmp_path / "raw" / "lending_club"
    raw.mkdir(parents=True)
    accepted = raw / "accepted.csv"
    pd.DataFrame({
        "id": np.arange(10),
        "issue_d": pd.date_range("2010-01-01", periods=10, freq="YS").strftime("%b-%Y"),
        "loan_status": ["Fully Paid", "Charged Off"] * 5,
    }).sample(frac=1, random_state=7).to_csv(accepted, index=False)

    task = adapters.build_lending_club(max_rows=3)

    np.testing.assert_array_equal(_feature_column(task, "lc_id"), [0.0, 4.0, 9.0])
    assert task.lineage["sampling_rule"] == "systematic_over_chronologically_sorted_resolved_outcomes"


def test_bank_selects_canonical_full_csv_and_marks_duration_post_outcome(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "DATA_DIR", tmp_path)
    raw = tmp_path / "raw" / "bank_marketing" / "nested"
    raw.mkdir(parents=True)
    pd.DataFrame({"age": [99], "duration": [1], "y": ["no"]}).to_csv(
        raw / "bank-additional.csv", sep=";", index=False
    )
    full = raw / "bank-additional-full.csv"
    pd.DataFrame({
        "age": [20, 30, 40], "job": ["a", "b", "c"],
        "duration": [5, 10, 15], "y": ["no", "yes", "no"],
    }).to_csv(full, sep=";", index=False)

    task = adapters.build_bank_marketing()

    assert task.X.shape == (3, 3)
    np.testing.assert_array_equal(_feature_column(task, "bm_age"), [20.0, 30.0, 40.0])
    duration_index = [spec.feature_id for spec in task.feature_specs].index("bm_duration")
    assert task.ground_truth[duration_index].label == LeakageLabel.POST_OUTCOME
    assert task.availability[duration_index].available_at_prediction is False
    assert task.source == str(full.resolve())
    assert task.lineage["chronological_sort"] is False
    assert task.lineage["source_order_preserved"] is True


def test_bank_does_not_treat_reduced_csv_as_full_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "DATA_DIR", tmp_path)
    raw = tmp_path / "raw" / "bank_marketing"
    raw.mkdir(parents=True)
    pd.DataFrame({"age": [20], "duration": [5], "y": ["no"]}).to_csv(
        raw / "bank-additional.csv", sep=";", index=False
    )

    with pytest.raises(FileNotFoundError, match="full Bank Marketing"):
        adapters.build_bank_marketing()

    assert adapters.build_bank_marketing(allow_synthetic=True).lineage["is_synthetic"] is True


def test_nyc_311_sorts_by_created_timestamp_and_records_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmatory_adapters, "DATA_DIR", tmp_path)
    cache = tmp_path / "nyc311" / "nyc311_cache.csv"
    cache.parent.mkdir(parents=True)
    pd.DataFrame({
        "unique_key": [30, 10, 20],
        "created_date": ["2024-03-01", "2024-01-01", "2024-02-01"],
        "closed_date": ["2024-03-03", "2024-01-20", "2024-02-02"],
        "agency": ["A", "B", "C"],
        "status": ["Closed", "Closed", "Closed"],
        "resolution_description": ["x", "y", "z"],
        "resolution_action_updated_date": ["2024-03-03", "2024-01-20", "2024-02-02"],
    }).to_csv(cache, index=False)

    task = confirmatory_adapters.build_nyc_311()

    np.testing.assert_array_equal(_feature_column(task, "nyc_unique_key"), [10.0, 20.0, 30.0])
    assert task.lineage["sort_column"] == "created_date"
    assert task.lineage["time_min"].startswith("2024-01")
    status_index = [spec.feature_id for spec in task.feature_specs].index("nyc_status")
    assert task.ground_truth[status_index].label == LeakageLabel.POST_OUTCOME
    assert task.availability[status_index].available_at_prediction is False


def test_chicago_food_sorts_by_inspection_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmatory_adapters, "DATA_DIR", tmp_path)
    cache = tmp_path / "chicago_food" / "chicago_food_cache.csv"
    cache.parent.mkdir(parents=True)
    pd.DataFrame({
        "inspection_id": [3, 1, 2],
        "inspection_date": ["2024-03-01", "2024-01-01", "2024-02-01"],
        "results": ["Pass", "Fail", "Pass"],
        "violations": ["none", "bad", "none"],
        "risk": [1.0, 2.0, 3.0],
    }).to_csv(cache, index=False)

    task = confirmatory_adapters.build_chicago_food()

    np.testing.assert_array_equal(_feature_column(task, "chi_inspection_id"), [1.0, 2.0, 3.0])
    assert task.lineage["sort_column"] == "inspection_date"
    result_index = [spec.feature_id for spec in task.feature_specs].index("chi_chi_results")
    assert task.ground_truth[result_index].label == LeakageLabel.POST_OUTCOME
    assert task.availability[result_index].available_at_prediction is False


def test_bts_sorts_by_flight_date_before_time_split(tmp_path, monkeypatch):
    monkeypatch.setattr(confirmatory_adapters, "DATA_DIR", tmp_path)
    archive = tmp_path / "bts" / "bts_2023_1.csv.zip"
    archive.parent.mkdir(parents=True)
    frame = pd.DataFrame({
        "FlightDate": ["2023-03-01", "2023-01-01", "2023-02-01"],
        "ArrDel15": [0, 1, 0], "Cancelled": [0, 0, 0], "Diverted": [0, 0, 0],
    })
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("fixture.csv", frame.to_csv(index=False))

    task = confirmatory_adapters.build_bts_flights()

    np.testing.assert_array_equal(_feature_column(task, "bts_FlightDate"), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(task.y, [1.0, 0.0, 0.0])
    assert task.lineage["sort_column"] == "FlightDate"
    assert task.lineage["archive_member"] == "fixture.csv"
