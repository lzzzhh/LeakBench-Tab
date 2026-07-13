"""Deterministic metadata-complete base tasks for corrected_v2.

These controlled panel tasks are the confirmatory construction benchmark.  The
task seed is fixed by dataset ID; injection/model seeds never change the clean
data or the chronological split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PanelTask:
    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    timestamps: np.ndarray
    entity_ids: np.ndarray
    source_ids: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    generator_seed: int
    archetype: str


ARCHETYPES = ("linear", "interaction", "nonlinear", "sparse", "drifting")


def _sigmoid(values):
    values = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-values))


def build_panel_task(dataset_index: int, namespace: str = "confirmatory") -> PanelTask:
    """Generate one heterogeneous panel task without using injection seeds.

    Confirmatory IDs are 0..19.  Pilot tuning must use a different namespace or
    indices >=100 so no confirmatory outcome is observed while fixing protocol.
    """
    if dataset_index < 0:
        raise ValueError("dataset_index must be non-negative")
    namespace_offset = 0 if namespace == "confirmatory" else 10_000_019
    seed = 202_607_13 + 7_919 * dataset_index + namespace_offset
    rng = np.random.RandomState(seed % (2**32 - 1))
    archetype = ARCHETYPES[dataset_index % len(ARCHETYPES)]

    n_entities = 50 + 5 * (dataset_index % 5)
    periods = 20 + 2 * ((dataset_index // 5) % 4)
    n_features = 12 + 4 * (dataset_index % 4)
    n = n_entities * periods
    entity_ids = np.repeat(np.arange(n_entities), periods)
    timestamps = np.tile(np.arange(periods, dtype=float), n_entities)

    entity_covariates = rng.normal(size=(n_entities, n_features))
    time_covariates = rng.normal(size=(periods, n_features))
    X = (
        0.35 * entity_covariates[entity_ids]
        + 0.15 * time_covariates[timestamps.astype(int)]
        + rng.normal(scale=0.9, size=(n, n_features))
    )
    coefficient = rng.normal(size=n_features)
    coefficient /= np.linalg.norm(coefficient) + 1e-12
    clean_signal = X @ coefficient
    if archetype == "interaction":
        clean_signal = 0.7 * clean_signal + 0.8 * X[:, 0] * X[:, 1]
    elif archetype == "nonlinear":
        clean_signal = 0.6 * clean_signal + np.sin(X[:, 0]) + 0.4 * X[:, 1] ** 2
    elif archetype == "sparse":
        clean_signal = 1.1 * X[:, 0] - 0.8 * X[:, 3] + 0.5 * (X[:, 5] > 0)
    elif archetype == "drifting":
        phase = 2.0 * np.pi * timestamps / max(1.0, periods - 1)
        clean_signal = 0.65 * clean_signal + 0.6 * np.sin(phase) * X[:, 0]

    entity_effect = rng.normal(scale=0.65 + 0.05 * (dataset_index % 4), size=n_entities)
    phase = 2.0 * np.pi * timestamps / max(1.0, periods - 1)
    time_effect = (0.25 + 0.05 * (dataset_index % 3)) * np.sin(phase + 0.2 * dataset_index)
    target_prevalence = 0.30 + 0.05 * (dataset_index % 5)
    raw_logit = clean_signal + entity_effect[entity_ids] + time_effect
    intercept = -float(np.quantile(raw_logit, 1.0 - target_prevalence))
    probability = _sigmoid(raw_logit + intercept)
    y = (rng.rand(n) < probability).astype(np.float32)

    # A baseline source exists for metadata completeness but is outcome-independent.
    source_ids = rng.randint(0, 8, size=n)
    order = np.argsort(timestamps, kind="stable")
    train_end, val_end = int(0.6 * n), int(0.8 * n)
    feature_names = tuple(f"x_{index:03d}" for index in range(n_features))
    return PanelTask(
        dataset_id=f"panel_{dataset_index:02d}",
        X=X.astype(np.float32),
        y=y,
        feature_names=feature_names,
        timestamps=timestamps,
        entity_ids=entity_ids,
        source_ids=source_ids,
        train_idx=order[:train_end],
        val_idx=order[train_end:val_end],
        test_idx=order[val_end:],
        generator_seed=seed,
        archetype=archetype,
    )


def build_panel_suite(count: int = 20, namespace: str = "confirmatory") -> list[PanelTask]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [build_panel_task(index, namespace=namespace) for index in range(count)]
