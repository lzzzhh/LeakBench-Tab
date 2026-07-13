"""Regression tests for the second post-audit cluster amendment."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.analyze_cluster_sensitivity_amendment_v2 as amendment
from scripts.analyze_cluster_sensitivity_amendment_v2 import (
    FrozenTaskReference,
    frozen_task_sha256,
    load_frozen_task_references,
    synchronized_inner_effects_by_seed,
    validate_prediction_against_reference,
)


def test_m08_entity_draw_is_shared_across_injection_seeds():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    forward = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    reverse = forward[::-1]
    clean = np.column_stack([forward, reverse])
    full = np.column_stack([reverse, forward])
    effects = synchronized_inner_effects_by_seed(
        y=y,
        clean=clean,
        full=full,
        clusters=np.repeat(np.arange(4), 2),
        column_seeds=np.array([13, 42]),
        seeds=[13, 42],
        repetitions=30,
        seed=19,
    )
    assert effects.shape == (30, 2)
    assert np.allclose(effects[:, 0], -effects[:, 1], atol=1e-15)


def test_prediction_reference_check_fails_closed_for_every_bound_array():
    reference = FrozenTaskReference(
        row_id=np.arange(4),
        y=np.array([0, 1, 0, 1]),
        entity_id=np.array([0, 0, 1, 1]),
        source_id=np.array([0, 1, 0, 1]),
        task_hash="a" * 64,
        split_hash="b" * 64,
        bundle_path="bundle.npz",
        bundle_sha256="c" * 64,
    )
    prediction = {
        "row_id": reference.row_id.copy(),
        "y": reference.y.copy(),
        "entity_id": reference.entity_id.copy(),
        "source_id": reference.source_id.copy(),
    }
    validate_prediction_against_reference(prediction, reference, "run")
    for field in prediction:
        corrupted = {name: values.copy() for name, values in prediction.items()}
        corrupted[field][0] += 1
        with pytest.raises(ValueError, match=field):
            validate_prediction_against_reference(corrupted, reference, "run")


def _write_minimal_frozen_contract(root: Path, bad_task_hash: bool = False) -> Path:
    base_x = np.arange(24, dtype=float).reshape(8, 3)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    train_idx = np.array([0, 1, 2, 3])
    val_idx = np.array([4, 5])
    test_idx = np.array([6, 7])
    arrays: dict[str, np.ndarray] = {
        "base_X": base_x,
        "y": y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }
    task_rows = []
    for mechanism in ("M08", "M09"):
        key = f"{mechanism}_S1_13"
        block = np.arange(8, dtype=float).reshape(8, 1)
        leakage_mask = np.array([False, False, False, True])
        entity = np.repeat(np.arange(4), 2)
        source = np.tile(np.array([0, 1]), 4)
        arrays[f"block__{key}"] = block
        arrays[f"leak_mask__{key}"] = leakage_mask
        arrays[f"entity_ids__{key}"] = entity
        arrays[f"source_ids__{key}"] = source
        task_hash = frozen_task_sha256(
            base_x, block, y, train_idx, val_idx, test_idx,
            leakage_mask, entity, source,
        )
        task_rows.append({
            "dataset_id": "panel_00",
            "dataset_namespace": "confirmatory",
            "mechanism": mechanism,
            "strength": "S1",
            "seed": 13,
            "bundle_key": key,
            "task_hash": ("0" * 64) if bad_task_hash and mechanism == "M08" else task_hash,
            "split_hash": hashlib.sha256(test_idx.tobytes()).hexdigest(),
            "bundle_path": "bundle.npz",
        })
    bundle_path = root / "bundle.npz"
    np.savez_compressed(bundle_path, **arrays)
    bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    for row in task_rows:
        row["bundle_sha256"] = bundle_hash
    manifest_path = root / "task_manifest.csv"
    pd.DataFrame(task_rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_frozen_bundle_reconstructs_and_verifies_task_hash(monkeypatch, tmp_path):
    monkeypatch.setattr(amendment, "ROOT", tmp_path)
    monkeypatch.setattr(
        amendment,
        "resolve",
        lambda path: Path(path) if Path(path).is_absolute() else tmp_path / Path(path),
    )
    monkeypatch.setattr(
        amendment,
        "relative",
        lambda path: str(Path(path).resolve().relative_to(tmp_path.resolve())),
    )
    config = {
        "protocol": {
            "dataset_count": 1,
            "strengths": ["S1"],
            "seeds": [13],
        }
    }
    manifest_path = _write_minimal_frozen_contract(tmp_path)
    references, bundles = load_frozen_task_references(manifest_path, config)
    assert len(references) == 2
    assert len(bundles) == 1

    bad_manifest = _write_minimal_frozen_contract(tmp_path, bad_task_hash=True)
    with pytest.raises(ValueError, match="task_hash"):
        load_frozen_task_references(bad_manifest, config)
