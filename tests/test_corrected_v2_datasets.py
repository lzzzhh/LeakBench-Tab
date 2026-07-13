"""Protocol invariants for the corrected_v2 confirmatory base tasks."""
from __future__ import annotations

import numpy as np

from src.leakbench.datasets import build_panel_task, build_panel_suite
from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID


def test_confirmatory_suite_has_twenty_deterministic_heterogeneous_tasks():
    first = build_panel_suite(20)
    second = build_panel_suite(20)
    assert len(first) == 20
    assert len({task.dataset_id for task in first}) == 20
    assert len({task.archetype for task in first}) == 5
    for left, right in zip(first, second):
        assert np.array_equal(left.X, right.X)
        assert np.array_equal(left.y, right.y)
        assert len(np.unique(left.entity_ids)) >= 50
        assert min(np.bincount(left.entity_ids)) >= 20
        assert 0.15 < left.y.mean() < 0.75


def test_clean_data_and_split_are_independent_of_injection_seed_and_mechanism():
    base = build_panel_task(0)
    tasks = []
    for seed, mechanism in ((13, MechanismID.DIRECT_COPY), (42, MechanismID.ENTITY_LEAK), (2026, MechanismID.SOURCE_LEAK)):
        task = LeakBenchInjector(seed=seed).inject(
            base.X,
            base.y,
            [MechanismConfig(mechanism, strength=0.6, seed=seed)],
            feature_names=list(base.feature_names),
            timestamps=base.timestamps,
            entity_ids=base.entity_ids,
            split_type="time",
        )
        tasks.append(task)
    for task in tasks:
        assert np.array_equal(task.X[:, : task.n_original], base.X)
        assert np.array_equal(task.train_idx, base.train_idx)
        assert np.array_equal(task.val_idx, base.val_idx)
        assert np.array_equal(task.test_idx, base.test_idx)


def test_pilot_namespace_is_disjoint_from_confirmatory_data():
    confirmatory = build_panel_task(0, namespace="confirmatory")
    pilot = build_panel_task(0, namespace="pilot")
    assert confirmatory.generator_seed != pilot.generator_seed
    assert not np.array_equal(confirmatory.X, pilot.X)
