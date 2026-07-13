"""Construction-validity tests for the corrected_v2 contamination mechanisms."""
from __future__ import annotations

import numpy as np

from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID


def _panel(seed: int = 7, n_entities: int = 20, periods: int = 40):
    rng = np.random.RandomState(seed)
    entity_ids = np.repeat(np.arange(n_entities), periods)
    timestamps = np.tile(np.arange(periods), n_entities).astype(float)
    X = rng.normal(size=(len(entity_ids), 6)).astype(np.float32)
    entity_logit = np.linspace(-2.2, 2.2, n_entities)
    probability = 1.0 / (1.0 + np.exp(-(entity_logit[entity_ids] + 0.25 * X[:, 0])))
    y = (rng.rand(len(entity_ids)) < probability).astype(np.float32)
    return X, y, timestamps, entity_ids, probability


def test_m08_uses_strict_future_entity_outcomes_and_returns_metadata():
    X, y, timestamps, entity_ids, _ = _panel()
    task = LeakBenchInjector(seed=13).inject(
        X,
        y,
        [MechanismConfig(MechanismID.ENTITY_LEAK, strength=0.8, n_entities=20, seed=13)],
        timestamps=timestamps,
        entity_ids=entity_ids,
        split_type="time",
    )

    assert np.array_equal(task.entity_ids, entity_ids)
    assert np.array_equal(task.timestamps, timestamps)
    assert "m08_future_count" in task.sample_metadata
    counts = task.sample_metadata["m08_future_count"]
    assert counts.shape == (len(y),)
    assert counts[0] == 39
    assert counts[39] == 0
    assert task.mechanism_params[-1]["excludes_current"] is True
    assert task.feature_availability[task.feature_names[-1]] is False


def test_m08_current_label_cannot_change_its_own_feature():
    X, y, timestamps, entity_ids, _ = _panel()
    row = 17
    config = [MechanismConfig(MechanismID.ENTITY_LEAK, strength=0.8, n_entities=20, seed=19)]
    original = LeakBenchInjector(seed=19).inject(
        X, y, config, timestamps=timestamps, entity_ids=entity_ids, split_type="time"
    )
    changed_y = y.copy()
    changed_y[row] = 1.0 - changed_y[row]
    changed = LeakBenchInjector(seed=19).inject(
        X, changed_y, config, timestamps=timestamps, entity_ids=entity_ids, split_type="time"
    )
    assert original.X[row, -1] == changed.X[row, -1]


def test_m08_future_outcome_shuffle_destroys_entity_signal():
    X, y, timestamps, entity_ids, probability = _panel(n_entities=24, periods=50)
    config = [MechanismConfig(MechanismID.ENTITY_LEAK, strength=1.0, noise_std=0.01, n_entities=24)]
    observed = LeakBenchInjector(seed=23).inject(
        X, y, config, timestamps=timestamps, entity_ids=entity_ids, split_type="time"
    ).X[:, -1]
    shuffled_y = np.random.RandomState(99).permutation(y)
    negative = LeakBenchInjector(seed=23).inject(
        X, shuffled_y, config, timestamps=timestamps, entity_ids=entity_ids, split_type="time"
    ).X[:, -1]
    observed_corr = abs(np.corrcoef(observed, probability)[0, 1])
    negative_corr = abs(np.corrcoef(negative, probability)[0, 1])
    assert observed_corr > 0.60
    assert negative_corr < 0.20


def test_m09_is_complete_one_hot_outcome_dependent_source():
    rng = np.random.RandomState(31)
    n = 1600
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.tile([0.0, 1.0], n // 2).astype(np.float32)
    task = LeakBenchInjector(seed=31).inject(
        X,
        y,
        [MechanismConfig(MechanismID.SOURCE_LEAK, strength=0.8, n_sources=8, min_group_count=5)],
    )
    block = task.X[:, task.n_original :]

    assert block.shape == (n, 8)
    assert np.allclose(block.sum(axis=1), 1.0)
    assert np.all((block == 0.0) | (block == 1.0))
    assert len(np.unique(task.source_ids)) == 8
    assert task.mechanism_params[-1]["encoding"] == "one_hot"
    assert task.mechanism_params[-1]["js_divergence"] > 0.0
    for source in range(8):
        for label in (0.0, 1.0):
            assert np.sum((task.source_ids == source) & (y == label)) >= 5


def test_m09_source_label_permutation_cannot_change_model_metric():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(37)
    n = 1200
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.tile([0.0, 1.0], n // 2).astype(np.float32)
    task = LeakBenchInjector(seed=37).inject(
        X, y, [MechanismConfig(MechanismID.SOURCE_LEAK, strength=0.7, n_sources=8)]
    )
    one_hot = task.X[:, task.n_original :]
    permutation = np.random.RandomState(101).permutation(one_hot.shape[1])
    train, test = task.train_idx, task.test_idx

    first = LogisticRegression(max_iter=1000).fit(one_hot[train], y[train])
    second = LogisticRegression(max_iter=1000).fit(one_hot[train][:, permutation], y[train])
    auc_first = roc_auc_score(y[test], first.predict_proba(one_hot[test])[:, 1])
    auc_second = roc_auc_score(y[test], second.predict_proba(one_hot[test][:, permutation])[:, 1])
    assert abs(auc_first - auc_second) < 1e-12


def test_m04_and_m05_exclude_current_label():
    rng = np.random.RandomState(41)
    n = 300
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = rng.randint(0, 2, size=n).astype(np.float32)
    timestamps = np.arange(n, dtype=float)
    row = 120
    for mechanism in (MechanismID.POST_OUTCOME, MechanismID.TEMPORAL_LEAK):
        config = [MechanismConfig(mechanism, strength=0.8, time_offset=0.05)]
        original = LeakBenchInjector(seed=41).inject(
            X, y, config, timestamps=timestamps, split_type="time"
        )
        changed_y = y.copy()
        changed_y[row] = 1.0 - changed_y[row]
        changed = LeakBenchInjector(seed=41).inject(
            X, changed_y, config, timestamps=timestamps, split_type="time"
        )
        assert original.X[row, -1] == changed.X[row, -1]
        assert np.all(np.diff(original.timestamps[original.train_idx]) >= 0)
        assert original.train_idx.max() < original.val_idx.min() < original.test_idx.min()


def test_m10_legitimate_component_is_copied_from_clean_input_and_not_quarantined():
    rng = np.random.RandomState(43)
    X = rng.normal(size=(500, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.5, size=500) > 0).astype(np.float32)
    task = LeakBenchInjector(seed=43).inject(
        X, y, [MechanismConfig(MechanismID.MIXED, strength=0.7)]
    )
    legit_column = task.n_original
    leak_column = task.n_original + 1

    assert np.array_equal(task.X[:, legit_column], X[:, 0])
    assert task.mechanism_labels[legit_column] == "legitimate"
    assert not task.leakage_mask[legit_column]
    assert task.legitimate_mask[legit_column]
    assert task.mechanism_labels[leak_column] == "M10"
    assert task.leakage_mask[leak_column]
