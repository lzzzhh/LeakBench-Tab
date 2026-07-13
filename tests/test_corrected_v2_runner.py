"""Small, disjoint-pilot checks for the corrected_v2 core runner."""
from __future__ import annotations

import numpy as np

from experiments.leakbench.run_corrected_core import build_mechanism_config, diagnostic_metrics
from src.leakbench.datasets import build_panel_task
from src.leakbench.mechanisms import LeakBenchInjector
from src.leakbench.models.core_models import fit_predict_core_model


def test_verified_core_model_smoke_and_probability_shape():
    task = build_panel_task(0, namespace="pilot")
    output = fit_predict_core_model(
        "lr",
        task.X[task.train_idx], task.y[task.train_idx],
        task.X[task.val_idx], task.y[task.val_idx],
        task.X[task.test_idx], 13,
    )
    assert output.implementation == "sklearn.StandardScaler+LogisticRegression"
    assert output.probabilities.shape == (len(task.test_idx),)
    assert np.all((0.0 <= output.probabilities) & (output.probabilities <= 1.0))


def test_diagnostic_uses_returned_ground_truth_without_name_assumptions():
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path("configs/paper/corrected_v2.yaml").read_text())
    base = build_panel_task(1, namespace="pilot")
    mechanism = build_mechanism_config("M09", "S3", config, 42)
    task = LeakBenchInjector(seed=42).inject(
        base.X, base.y, [mechanism], feature_names=[f"anonymous_{i}" for i in range(base.X.shape[1])],
        timestamps=base.timestamps, entity_ids=base.entity_ids, split_type="time",
    )
    ap, normalized_ap, recall = diagnostic_metrics(task)
    assert 0.0 <= ap <= 1.0
    assert -1.0 <= normalized_ap <= 1.0
    assert 0.0 <= recall <= 1.0
