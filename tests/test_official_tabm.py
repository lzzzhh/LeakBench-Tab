"""Contract tests for the identity-safe official TabM path."""
from __future__ import annotations

import importlib.metadata
import sys
import types

import numpy as np
import pytest

from benchmark_v2.models.tabm import TabMEnsemble
from experiments.leakbench.run_corrected_tabm import empty_result_row, main
from src.leakbench.models import official_tabm


def _binary_arrays(seed=7):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(160, 6)).astype(np.float32)
    y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(np.float32)
    return X[:96], y[:96], X[96:128], y[96:128], X[128:]


def test_removed_proxy_identity_guard_fails_explicitly():
    with pytest.raises(RuntimeError, match="identity-invalid"):
        TabMEnsemble(6)


def test_missing_official_package_has_no_score_fallback(monkeypatch):
    def missing(_):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(official_tabm.importlib.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="no fallback is permitted"):
        official_tabm.fit_predict_official_tabm(
            *_binary_arrays(), seed=13, device="cpu", max_epochs=1, patience=1
        )


def test_adapter_is_deterministic_and_records_official_contract(monkeypatch):
    import torch

    class TabM(torch.nn.Module):
        __module__ = "tabm"

        def __init__(self, n_num_features, k):
            super().__init__()
            self.k = k
            self.linear = torch.nn.Linear(n_num_features, 1)
            self.member_offset = torch.nn.Parameter(torch.zeros(k))

        @classmethod
        def make(cls, **kwargs):
            return cls(kwargs["n_num_features"], kwargs["k"])

        def forward(self, values):
            base = self.linear(values)[:, None, :]
            return base + self.member_offset[None, :, None]

    fake_package = types.ModuleType("tabm")
    fake_package.TabM = TabM
    monkeypatch.setitem(sys.modules, "tabm", fake_package)
    monkeypatch.setattr(
        official_tabm.importlib.metadata,
        "version",
        lambda package: official_tabm.PINNED_TABM_VERSION,
    )
    arrays = _binary_arrays()
    kwargs = dict(
        seed=42,
        device="cpu",
        k=4,
        max_epochs=4,
        patience=2,
        batch_size=32,
    )
    first = official_tabm.fit_predict_official_tabm(*arrays, **kwargs)
    second = official_tabm.fit_predict_official_tabm(*arrays, **kwargs)

    assert np.array_equal(first.probabilities, second.probabilities)
    assert first.probabilities.shape == (len(arrays[-1]),)
    assert np.all((0.0 <= first.probabilities) & (first.probabilities <= 1.0))
    assert first.implementation == "tabm.TabM@0.0.3"
    assert first.manifest["model_class"] == "tabm.TabM"
    assert first.manifest["preprocessing"] == "sklearn.StandardScaler fit on train only"
    assert first.manifest["training_loss"] == "mean of per-k BCEWithLogits losses"
    assert first.manifest["inference_aggregation"] == "mean of per-k sigmoid probabilities"


def test_runner_default_failure_schema_uses_nan_not_neutral_scores():
    row = empty_result_row()
    assert np.isnan(row["clean_auc"])
    assert np.isnan(row["full_auc"])
    assert np.isnan(row["paired_harm"])


def test_confirmatory_runner_requires_explicit_gate(tmp_path):
    with pytest.raises(RuntimeError, match="Confirmatory TabM execution is locked"):
        main([
            "--namespace", "confirmatory",
            "--output", str(tmp_path / "must_not_exist.csv"),
        ])
    assert not (tmp_path / "must_not_exist.csv").exists()
