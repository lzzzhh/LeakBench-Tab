"""Regression tests for the post-unblinding diagnostic RNG amendment."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_diagnostic_rng_amendment import (
    IDENTITY_COLUMNS,
    METRIC_COLUMNS,
    build_canonical_frame,
)
from scripts.build_corrected_v2_claim_state import (
    default_paths,
    validate_diagnostic_rng_amendment,
)


def _raw_rows() -> pd.DataFrame:
    common = {
        "dataset_id": "panel_00",
        "dataset_index": 0,
        "dataset_namespace": "confirmatory",
        "mechanism": "M01",
        "strength": "S1",
        "seed": 13,
        "status": "SUCCESS",
        "failure_reason": "",
        "n_leak": 1,
        "n_features": 13,
        "runtime_sec": 0.5,
        "task_hash": "a" * 64,
        "split_hash": "b" * 64,
        "bundle_sha256": "c" * 64,
        "task_manifest_sha256": "d" * 64,
        "integrity_verified": True,
        "config_hash": "e" * 64,
        "code_hash": "f" * 64,
    }
    rows = []
    for index, method in enumerate((
        "mutual_information", "absolute_correlation", "lr_coefficient", "rf_permutation"
    )):
        rows.append({
            **common,
            "diagnostic_run_id": f"run-{index}",
            "method": method,
            "localization_ap": 0.1 + index,
            "localization_normalized_ap": 0.2 + index,
            "top5_recall": 0.3 + index,
        })
    return pd.DataFrame(rows)


def _task_rows() -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset_id": "panel_00", "dataset_index": 0,
        "dataset_namespace": "confirmatory", "mechanism": "M01",
        "strength": "S1", "seed": 13, "task_hash": "a" * 64,
        "split_hash": "b" * 64, "bundle_sha256": "c" * 64,
        "n_leak": 1, "diagnostic_ap": 0.91,
        "diagnostic_normalized_ap": 0.90, "top5_recall": 1.0,
    }])


def test_amendment_replaces_only_mi_metrics_and_preserves_identity():
    raw = _raw_rows()
    canonical, audit = build_canonical_frame(raw, _task_rows())
    mi = canonical.loc[canonical["method"] == "mutual_information"].iloc[0]
    assert tuple(mi[column] for column in METRIC_COLUMNS) == pytest.approx((0.91, 0.90, 1.0))
    assert mi["diagnostic_rng_policy"] == "fixed_seed_42"
    assert bool(mi["rng_amendment_applied"])

    other = canonical.loc[canonical["method"] != "mutual_information"]
    expected = raw.loc[raw["method"] != "mutual_information"]
    pd.testing.assert_frame_equal(
        other[raw.columns].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=True,
    )
    assert audit["rows_replaced"] == 1
    assert audit["rows_preserved"] == 3
    assert audit["scientific_identity_exact"] is True
    assert set(canonical.columns).issuperset(IDENTITY_COLUMNS)


def test_amendment_fails_closed_on_task_identity_mismatch():
    tasks = _task_rows()
    tasks.loc[0, "task_hash"] = "0" * 64
    with pytest.raises(ValueError, match="task_hash"):
        build_canonical_frame(_raw_rows(), tasks)


def test_amendment_fails_closed_on_duplicate_scientific_identity():
    raw = _raw_rows()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_canonical_frame(raw, _task_rows())


def test_checked_in_amendment_is_complete_and_hash_bound():
    root = Path(__file__).resolve().parents[1]
    freeze_path = root / "results/corrected_v2/diagnostic_rng_amendment_freeze.json"
    canonical_path = root / "results/corrected_v2/diagnostic_canonical_cells.csv"
    manifest_path = root / "results/corrected_v2/diagnostic_canonical_cells.manifest.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert freeze["no_tuning"] is True
    assert freeze["thresholds_changed"] is False
    assert freeze["expected_replaced_rows"] == 5500
    assert freeze["expected_preserved_rows"] == 16500
    assert manifest["rows_replaced"] == 5500
    assert manifest["rows_preserved"] == 16500
    assert manifest["canonical"]["sha256"] == hashlib.sha256(
        canonical_path.read_bytes()
    ).hexdigest()
    assert manifest["identity_checks"]["scientific_identity_exact"] is True
    assert manifest["identity_checks"]["non_mi_existing_columns_exact"] is True
    assert manifest["identity_checks"]["mi_replacement_metrics_exact"] is True
    assert manifest["identity_checks"]["task_split_bundle_lineage_exact"] is True
    assert manifest["identity_checks"]["raw_duplicate_identities"] is False
    assert manifest["identity_checks"]["replacement_duplicate_identities"] is False
    assert manifest["identity_checks"]["canonical_duplicate_identities"] is False


def test_claim_chain_accepts_only_amended_canonical_not_raw_suite():
    root = Path(__file__).resolve().parents[1]
    tasks = pd.read_csv(root / "results/corrected_v2/task_bundles/task_manifest.csv")
    validate_diagnostic_rng_amendment(default_paths(), tasks)
    raw_paths = replace(
        default_paths(),
        diagnostic_cells=root / "results/corrected_v2/diagnostic_confirmatory_cells.csv",
    )
    with pytest.raises(ValueError, match="Raw task-seeded diagnostic output is forbidden"):
        validate_diagnostic_rng_amendment(raw_paths, tasks)
