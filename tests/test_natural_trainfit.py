from dataclasses import replace

import numpy as np

from benchmark_v2.core.models import FeatureAvailability, FeatureRole, FeatureSpec, LeakageGroundTruth, LeakageLabel
from benchmark_v2.datasets.confirmatory_adapters import NaturalTask
from experiments.leakbench import run_natural_case_studies_trainfit as amended


def test_train_fitted_category_remap_hides_test_only_vocabulary(monkeypatch):
    # Global codes 0 and 2 occur in training; code 1 is test-only. Compaction
    # must map train codes to 0/1 and the unseen code to the unknown sentinel.
    task = NaturalTask(
        name="BankMarketing",
        X=np.array([[0.0], [2.0], [0.0], [1.0], [2.0]], dtype=np.float32),
        y=np.array([0, 1, 0, 1, 0], dtype=np.float32),
        feature_specs=[FeatureSpec(feature_id="bm_job", role=FeatureRole.PREDICTOR, name_pool={"natural": "job"})],
        availability=[FeatureAvailability(feature_id="bm_job", available_at_prediction=False)],
        ground_truth=[LeakageGroundTruth(feature_id="bm_job", label=LeakageLabel.POST_OUTCOME)],
        train_idx=np.array([0, 1, 2]), val_idx=np.array([3]), test_idx=np.array([4]),
        lineage={"is_synthetic": False},
    )
    monkeypatch.setitem(amended.CATEGORICAL_FEATURES, "BankMarketing", {"job"})
    monkeypatch.setitem(amended.DROP_FEATURES, "BankMarketing", set())
    X, truth, names, audit = amended.apply_train_fitted_category_protocol(task)
    np.testing.assert_array_equal(X[:, 0], [0.0, 1.0, 0.0, -2.0, 1.0])
    assert truth.tolist() == [True]
    assert names == ["job"]
    assert audit["fit_rows"] == "train_idx_only"
    assert audit["unseen_category_counts"]["job"]["validation"] == 1


def test_date_string_features_are_dropped_before_modeling(monkeypatch):
    task = NaturalTask(
        name="BTSFlights",
        X=np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        y=np.array([0, 1, 0], dtype=np.float32),
        feature_specs=[
            FeatureSpec(feature_id="bts_FlightDate", role=FeatureRole.PREDICTOR, name_pool={"natural": "FlightDate"}),
            FeatureSpec(feature_id="bts_ArrDelay", role=FeatureRole.PREDICTOR, name_pool={"natural": "ArrDelay"}),
        ],
        availability=[FeatureAvailability(feature_id="bts_FlightDate"), FeatureAvailability(feature_id="bts_ArrDelay", available_at_prediction=False)],
        ground_truth=[LeakageGroundTruth(feature_id="bts_FlightDate", label=LeakageLabel.LEGITIMATE), LeakageGroundTruth(feature_id="bts_ArrDelay", label=LeakageLabel.POST_OUTCOME)],
        train_idx=np.array([0]), val_idx=np.array([1]), test_idx=np.array([2]), lineage={"is_synthetic": False},
    )
    monkeypatch.setitem(amended.CATEGORICAL_FEATURES, "BTSFlights", set())
    monkeypatch.setitem(amended.DROP_FEATURES, "BTSFlights", {"FlightDate"})
    X, truth, names, audit = amended.apply_train_fitted_category_protocol(task)
    assert X.shape == (3, 1)
    assert names == ["ArrDelay"]
    assert truth.tolist() == [True]
    assert audit["dropped_date_string_features"] == ["FlightDate"]
