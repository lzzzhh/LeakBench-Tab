"""Regression tests for the post-audit statistical amendment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_cluster_sensitivity_v2 import (
    analyze,
    auc_columns,
    synchronized_inner_effects,
)
from scripts.analyze_corrected_v2_amendment import (
    build_category_contrasts,
    exact_sign_flip_p,
    joint_dx_bootstrap,
)


def test_exact_sign_flip_enumerates_the_task_randomization_distribution():
    assert exact_sign_flip_p(np.array([1.0, 1.0])) == 0.5
    assert exact_sign_flip_p(np.array([1.0, -1.0])) == 1.0
    assert exact_sign_flip_p(np.ones(20)) == 2 / (2**20)


def test_category_contrasts_use_twenty_task_effects_and_holm():
    rows = []
    for dataset in range(20):
        for seed in (13, 42):
            for mechanism, category, harm in (
                ("M01", "simple", 0.30),
                ("M03", "boundary", 0.20),
                ("M04", "structured", 0.10),
            ):
                rows.append({
                    "dataset_id": f"d{dataset:02d}",
                    "seed": seed,
                    "mechanism": mechanism,
                    "category": category,
                    "paired_harm": harm,
                })
    result = build_category_contrasts(pd.DataFrame(rows), repetitions=50, seed=7)
    assert set(result["test_method"]) == {"exact_two_sided_task_level_sign_flip"}
    assert set(result["n_tasks"]) == {20}
    assert np.allclose(result["sign_flip_p"], 2 / (2**20))
    assert np.allclose(result["holm_p"], 3 * 2 / (2**20))


def test_joint_dx_bootstrap_resamples_detectability_and_harm_together():
    # D varies by dataset while X has its own aligned task variation.  A fixed-D
    # bootstrap would be unable to reproduce this distribution.
    d = np.array([
        [[0.1, 0.3, 0.7, 0.9], [0.2, 0.4, 0.6, 0.8]],
        [[0.9, 0.1, 0.8, 0.2], [0.8, 0.2, 0.7, 0.3]],
        [[0.4, 0.9, 0.1, 0.6], [0.3, 0.8, 0.2, 0.7]],
    ])
    x = np.array([
        [[0.0, 0.2, 0.6, 1.0], [0.1, 0.3, 0.7, 0.9]],
        [[0.8, 0.0, 0.9, 0.1], [0.9, 0.1, 0.8, 0.2]],
        [[0.2, 1.0, 0.0, 0.7], [0.3, 0.9, 0.1, 0.6]],
    ])
    left = joint_dx_bootstrap(d, x, repetitions=100, seed=11)
    right = joint_dx_bootstrap(d, x, repetitions=100, seed=11)
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[1], right[1])
    assert len(np.unique(left[0])) > 1


def test_synchronized_draw_cancels_opposite_cell_effects():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    b = a[::-1]
    clean = np.column_stack([a, b])
    full = np.column_stack([b, a])
    clusters = np.repeat(np.arange(4), 2)
    effects = synchronized_inner_effects(
        y, clean, full, clusters, repetitions=30, seed=19
    )
    assert np.allclose(effects, 0.0, atol=1e-15)


def _write_prediction(
    path: Path,
    y: np.ndarray,
    clean: np.ndarray,
    full: np.ndarray,
    entities: np.ndarray,
    sources: np.ndarray,
) -> float:
    np.savez_compressed(
        path,
        row_id=np.arange(len(y)),
        y=y,
        clean_probability=clean,
        full_probability=full,
        entity_id=entities,
        source_id=sources,
    )
    return float(auc_columns(y, full)[0] - auc_columns(y, clean)[0])


def test_cluster_analysis_labels_m09_as_descriptive_reweighting(tmp_path):
    config = {
        "protocol": {
            "dataset_count": 1,
            "strengths": ["S1", "S2"],
            "core_models": ["lr", "rf"],
            "seeds": [13],
        },
        "mechanism_parameters": {"M09": {"n_sources": 2}},
    }
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    entities = np.repeat(np.arange(4), 2)
    rows = []
    paths = {}
    for mechanism in ("M08", "M09"):
        for strength_index, strength in enumerate(("S1", "S2")):
            sources = np.tile(np.array([0, 1]), 4) if mechanism == "M09" else np.zeros(8, dtype=int)
            clean = np.linspace(0.2, 0.8, 8)
            full = np.clip(clean + (2 * y - 1) * (0.02 + 0.01 * strength_index), 0, 1)
            for model in ("lr", "rf"):
                run_id = f"{mechanism}_{strength}_{model}"
                path = tmp_path / f"{run_id}.npz"
                harm = _write_prediction(path, y, clean, full, entities, sources)
                paths[run_id] = path
                rows.append({
                    "run_id": run_id,
                    "dataset_id": "d0",
                    "mechanism": mechanism,
                    "strength": strength,
                    "model": model,
                    "seed": 13,
                    "paired_harm": harm,
                })
    result = analyze(
        pd.DataFrame(rows), paths, config,
        inner_repetitions=5, outer_repetitions=10, seed=23,
    )["analyses"]
    assert result["M08"]["shared_draw_scope"] == ["model", "strength"]
    assert result["M08"]["status"] == "INFERENTIAL_SENSITIVITY"
    assert result["M09"]["shared_draw_scope"] == ["model"]
    assert result["M09"]["status"] == "DESCRIPTIVE_DESIGNED_CATEGORY_REWEIGHTING"
    assert result["M09"]["inferential_source_population_claim_allowed"] is False
