from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.analyze_full_b1_r10e import (
    add_score_metrics,
    estimate,
    semantic_metrics_for_selection,
)


def test_score_decomposition_separates_repair_and_overcorrection():
    frame = pd.DataFrame([
        {"strict_auc": 0.70, "full_auc": 0.80, "governed_auc": 0.75, "legacy_sdr": 0.05},
        {"strict_auc": 0.70, "full_auc": 0.80, "governed_auc": 0.65, "legacy_sdr": 0.05},
    ])
    out = add_score_metrics(frame)
    assert np.isclose(out.loc[0, "directional_repair"], 0.05)
    assert np.isclose(out.loc[0, "overcorrection"], 0.0)
    assert np.isclose(out.loc[1, "directional_repair"], 0.10)
    assert np.isclose(out.loc[1, "overcorrection"], 0.05)


def test_semantic_metrics_require_full_group_removal():
    key = (0, "M09", "S1", 13)
    registry = {
        key: {
            "groups": {"leak": frozenset({2, 3}), "legit": frozenset({0, 1})},
            "leak_groups": ("leak",),
            "leak_indices": frozenset({2, 3}),
            "n_leak": 2,
            "n_legit": 2,
        }
    }
    partial = pd.Series({
        "dataset_index": 0, "mechanism": "M09", "strength": "S1",
        "training_seed": 13, "removed_encoded_indices": json.dumps([2]),
    })
    full = partial.copy()
    full["removed_encoded_indices"] = json.dumps([2, 3])
    partial_metrics = semantic_metrics_for_selection(partial, registry)
    full_metrics = semantic_metrics_for_selection(full, registry)
    assert partial_metrics["semantic_group_recall_any"] == 1.0
    assert partial_metrics["semantic_group_recall_full"] == 0.0
    assert partial_metrics["partial_group_violation_rate"] == 1.0
    assert full_metrics["semantic_group_recall_full"] == 1.0
    assert full_metrics["partial_group_violation_rate"] == 0.0


def test_task_cluster_estimate_is_deterministic_and_task_weighted():
    frame = pd.DataFrame({
        "dataset_index": [0, 0, 0, 1],
        "delta_metric": [1.0, 1.0, 1.0, -1.0],
    })
    first = estimate(frame, "delta_metric", "fixed")
    second = estimate(frame, "delta_metric", "fixed")
    assert first == second
    assert first["mean"] == 0.0
    assert first["n_keys"] == 4
    assert first["n_tasks"] == 2
