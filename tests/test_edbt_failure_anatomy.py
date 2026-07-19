"""Regression tests for the post-hoc EDBT failure-anatomy evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ANATOMY = ROOT / "results/edbt_eab_revision/failure_anatomy"
PAPER_SOURCE = ROOT / "paper/edbt_eab/source_data/generated"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failure_anatomy_manifest_binds_inputs_and_outputs():
    manifest = json.loads((ANATOMY / "failure_anatomy_manifest.json").read_text())
    assert manifest["status"] == "POST_HOC_DESCRIPTIVE_DIAGNOSTIC_COMPLETE"
    assert manifest["parent_revision_status"] == "COMPLETE_WITH_DISCLOSED_LIMITATIONS"
    assert manifest["selection_hash_validation"] == {
        "all_matched": True,
        "nyc311_rows": 3,
        "sparse_keys": 1100,
    }
    for section in ("input_sha256", "output_sha256"):
        for relative, expected in manifest[section].items():
            path = ROOT / relative
            assert path.is_file()
            assert sha256(path) == expected


def test_sparse_failure_anatomy_is_descriptive_and_complete():
    summary = json.loads((ANATOMY / "failure_anatomy_summary.json").read_text())
    sparse = summary["sparse"]
    assert summary["downstream_model_fits"] == 0
    assert sparse["status"] == "POST_HOC_DESCRIPTIVE_DIAGNOSTIC"
    assert sparse["n_keys"] == 1100
    assert sparse["n_tasks"] == 4
    assert sparse["n_mechanisms"] == 11
    assert sparse["negative_tasks"] == 4
    assert sparse["negative_mechanisms"] == 9
    assert sparse["repair_advantage"] == pytest.approx(-0.1183321483)
    assert sparse["p3_leak_recall"] == pytest.approx(0.5699621212)
    assert sparse["p3_legitimate_retention"] == pytest.approx(0.8481044397)
    assert sparse["sparse_signal_removal_rates"]["x_000"] == pytest.approx(0.8927272727)
    assert sparse["sparse_signal_removal_rates"]["x_003"] == pytest.approx(0.8490909091)

    table = pd.read_csv(ANATOMY / "sparse_failure_anatomy.csv")
    assert (table["row_type"] == "task").sum() == 4
    assert (table["row_type"] == "mechanism").sum() == 11
    assert not table.duplicated(["row_type", "scope"]).any()


def test_nyc311_selection_explains_the_low_opportunity_failure():
    summary = json.loads((ANATOMY / "failure_anatomy_summary.json").read_text())["nyc311"]
    assert summary["initial_gap"] == pytest.approx(0.0190457312)
    assert summary["repair_advantage"] == pytest.approx(-0.1081896998)
    assert summary["budget_k"] == 8
    assert summary["n_features"] == 40
    assert summary["p3_leak_recall"] == 0.5
    assert summary["p3_legitimate_retention"] == pytest.approx(0.8157894737)
    assert summary["selected_invalid_fields"] == ["resolution_description"]
    assert summary["missed_invalid_fields"] == ["status"]
    assert len(summary["selected_valid_fields"]) == 7

    selected = pd.read_csv(ANATOMY / "nyc311_selection_diagnostic.csv")
    assert len(selected) == 8
    assert (selected["contract_label"] == "invalid").sum() == 1
    assert selected["selection_mask_hash"].nunique() == 1


def test_paper_assets_bind_failure_anatomy_without_adding_a_core_table():
    manifest = json.loads((PAPER_SOURCE / "paper_asset_manifest.json").read_text())
    assert manifest["paper_table_count"] == 3
    assert manifest["row_counts"]["governance_results.csv"] == 69
    assert "results/edbt_eab_revision/failure_anatomy/failure_anatomy_manifest.json" in manifest["source_sha256"]

    governance = pd.read_csv(PAPER_SOURCE / "governance_results.csv")
    sparse = governance[governance["row_type"] == "failure_anatomy"]
    assert len(sparse) == 1
    assert sparse.iloc[0]["interpretation"] == "POST_HOC_CONSTRUCTION_DIAGNOSIS"
    assert sparse.iloc[0]["negative_task_count"] == 4
    assert sparse.iloc[0]["negative_mechanism_count"] == 9

    natural = pd.read_csv(PAPER_SOURCE / "natural_cases.csv").set_index("task")
    assert natural.loc["NYC311", "p3_removed_invalid_count"] == 1
    assert natural.loc["NYC311", "p3_removed_valid_count"] == 7
    assert natural.loc["NYC311", "selected_invalid_features"] == "resolution_description"
    assert natural.loc["NYC311", "missed_invalid_features"] == "status"
