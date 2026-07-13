"""Construction tests for the opt-in constant-prior structured amendment."""
from __future__ import annotations

import numpy as np
import pytest

from src.leakbench.mechanisms import MechanismConfig, MechanismID
from src.leakbench.mechanisms.structured_prior_v1 import (
    AMENDMENT_VERSION,
    CONSTANT_PRIOR,
    StructuredPriorV1Injector,
)


STRUCTURED = (
    MechanismID.POST_OUTCOME,
    MechanismID.TEMPORAL_LEAK,
    MechanismID.ENTITY_LEAK,
)


def _inject(mechanism, X, y, timestamps, entity_ids, *, seed=17, noise_std=0.0):
    config = MechanismConfig(
        mechanism,
        strength=1.0,
        noise_std=noise_std,
        time_offset=0.0,
        prior_strength=2.0,
        seed=seed,
    )
    return StructuredPriorV1Injector(seed=seed).inject(
        X,
        y,
        [config],
        timestamps=timestamps,
        entity_ids=entity_ids,
        split_type="time",
    )


@pytest.mark.parametrize("mechanism", STRUCTURED)
def test_no_eligible_future_has_noiseless_constant_value(mechanism):
    X = np.zeros((4, 2), dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    entity_ids = np.zeros(4, dtype=int)

    task = _inject(mechanism, X, y, timestamps, entity_ids)

    assert task.X[-1, -1] == CONSTANT_PRIOR
    count_key = {
        MechanismID.POST_OUTCOME: "m04_future_count",
        MechanismID.TEMPORAL_LEAK: "m05_future_count",
        MechanismID.ENTITY_LEAK: "m08_future_count",
    }[mechanism]
    assert task.sample_metadata[count_key][-1] == 0
    assert task.mechanism_params[-1]["prior_value"] == CONSTANT_PRIOR
    assert task.mechanism_params[-1]["amendment_version"] == AMENDMENT_VERSION


@pytest.mark.parametrize("mechanism", STRUCTURED)
def test_flipping_current_label_cannot_change_that_rows_feature(mechanism):
    rng = np.random.RandomState(3)
    X = rng.normal(size=(12, 3)).astype(np.float32)
    y = rng.randint(0, 2, size=12).astype(np.float32)
    timestamps = np.arange(12, dtype=float)
    entity_ids = np.repeat(np.arange(3), 4)
    row = 5

    original = _inject(mechanism, X, y, timestamps, entity_ids, noise_std=0.07)
    changed_y = y.copy()
    changed_y[row] = 1.0 - changed_y[row]
    changed = _inject(
        mechanism, X, changed_y, timestamps, entity_ids, noise_std=0.07
    )

    assert original.X[row, -1] == changed.X[row, -1]


@pytest.mark.parametrize("mechanism", STRUCTURED)
def test_same_timestamp_labels_are_excluded(mechanism):
    X = np.zeros((8, 2), dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    timestamps = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    entity_ids = np.zeros(8, dtype=int)

    original = _inject(mechanism, X, y, timestamps, entity_ids)
    changed_y = y.copy()
    changed_y[1] = 1.0
    changed = _inject(mechanism, X, changed_y, timestamps, entity_ids)

    assert original.X[0, -1] == changed.X[0, -1]
    assert original.mechanism_params[-1]["excludes_same_timestamp"] is True


def test_m08_uses_only_future_same_entity_labels_and_constant_shrinkage():
    X = np.zeros((6, 2), dtype=np.float32)
    y = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    timestamps = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    entity_ids = np.array([7, 7, 7, 8, 8, 8])

    task = _inject(MechanismID.ENTITY_LEAK, X, y, timestamps, entity_ids)
    rate = task.sample_metadata["m08_future_rate"]

    # Entity 7, t=0 sees only labels [1, 1], shrunk by two pseudo-observations
    # at the fixed 0.5 prior: (2 + 2*0.5) / (2 + 2) = 0.75.
    assert rate[0] == pytest.approx(0.75)
    assert task.X[0, -1] == pytest.approx(0.75)
    assert task.sample_metadata["m08_future_count"][0] == 2
    assert rate[2] == CONSTANT_PRIOR

    changed_y = y.copy()
    changed_y[3:] = 1.0 - changed_y[3:]
    changed_other_entity = _inject(
        MechanismID.ENTITY_LEAK, X, changed_y, timestamps, entity_ids
    )
    assert changed_other_entity.X[0, -1] == task.X[0, -1]
    assert (
        task.mechanism_params[-1]["aggregation"]
        == "strict_future_same_entity_constant_shrinkage_mean"
    )


@pytest.mark.parametrize("mechanism", STRUCTURED)
def test_structured_amendment_is_deterministic(mechanism):
    rng = np.random.RandomState(9)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = rng.randint(0, 2, size=30).astype(np.float32)
    timestamps = np.tile(np.arange(10), 3).astype(float)
    entity_ids = np.repeat(np.arange(3), 10)

    first = _inject(mechanism, X, y, timestamps, entity_ids, seed=2026, noise_std=0.1)
    second = _inject(mechanism, X, y, timestamps, entity_ids, seed=2026, noise_std=0.1)

    assert np.array_equal(first.X, second.X)
    assert first.feature_names == second.feature_names
    assert first.mechanism_params == second.mechanism_params
    assert first.sample_metadata.keys() == second.sample_metadata.keys()
    for key in first.sample_metadata:
        assert np.array_equal(first.sample_metadata[key], second.sample_metadata[key])
