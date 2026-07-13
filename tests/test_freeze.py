import hashlib
import json
from pathlib import Path

import pytest


def test_frozen_protocol_hashes_still_match():
    freeze = json.loads(Path("results/corrected_v2/protocol_freeze.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_CONFIRMATORY_RUN"
    for entry in freeze["files"]:
        path = Path(entry["path"])
        assert path.exists(), entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]


def test_confirmatory_and_pilot_namespaces_are_separate():
    freeze = json.loads(Path("results/corrected_v2/protocol_freeze.json").read_text())
    assert freeze["pilot_namespace"] == "pilot"
    assert freeze["confirmatory_namespace"] == "confirmatory"
    assert freeze["pilot_namespace"] != freeze["confirmatory_namespace"]


def test_official_tabm_frozen_hashes_still_match():
    freeze = json.loads(Path("results/corrected_v2/tabm_protocol_freeze.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_TABM_CONFIRMATORY_RUN"
    assert freeze["model_identity"] == "tabm.TabM"
    for entry in freeze["files"]:
        path = Path(entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]


def test_bundle_tabm_protocol_supersedes_generator_protocol_and_matches_hashes():
    freeze = json.loads(Path("results/corrected_v2/tabm_bundle_protocol_freeze.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_BUNDLE_CONFIRMATORY_RUN"
    assert freeze["confirmatory_tasks"] == 5500
    assert freeze["supersedes"] == "results/corrected_v2/tabm_protocol_freeze.json"
    for entry in freeze["files"]:
        path = Path(entry["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]


def test_m10_amendment_protocol_is_frozen_before_replacement_runs():
    freeze = json.loads(
        Path("results/corrected_v2/m10_amendment_protocol_freeze.json").read_text()
    )
    assert freeze["status"] == "FROZEN_BEFORE_M10_AMENDMENT_CONFIRMATORY_RUN"
    assert freeze["amendment_version"] == "m10_strict_mask_v1"
    assert freeze["strict_policy"] == "task.X[:, ~task.leakage_mask]"
    assert freeze["expected_replacement_cells"] == 2500
    assert freeze["expected_cpu_cells"] == 2000
    assert freeze["expected_tabm_cells"] == 500
    assert freeze["verified_confirmatory_tasks"] == 500
    for relative_path, entry in freeze["frozen_files"].items():
        path = Path(relative_path)
        assert path.stat().st_size == entry["size_bytes"], relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], relative_path


def test_corrected_v2_required_artifacts_exist():
    for path in (
        "configs/paper/corrected_v2.yaml",
        "results/corrected_v2/inventory/legacy_sha256_manifest.json",
        "results/corrected_v2/superseded_evidence.json",
        "scripts/analyze_corrected_v2.py",
    ):
        assert Path(path).is_file(), path


def test_diagnostic_protocol_is_frozen_and_hashes_match():
    freeze = json.loads(Path("results/corrected_v2/diagnostic_protocol_freeze.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_DIAGNOSTIC_CONFIRMATORY_RUN"
    assert freeze["expected_diagnostic_cells"] == 22000
    assert len(freeze["diagnostic_methods"]) == 4
    for relative_path, entry in freeze["frozen_files"].items():
        path = Path(relative_path)
        assert path.exists(), relative_path
        assert path.stat().st_size == entry["size_bytes"], relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], relative_path


def test_secondary_analysis_protocol_is_frozen_and_hashes_match():
    freeze = json.loads(Path("results/corrected_v2/secondary_analysis_protocol_freeze.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_CONFIRMATORY_COMPLETION"
    for relative_path, entry in freeze["frozen_files"].items():
        path = Path(relative_path)
        assert path.exists(), relative_path
        assert path.stat().st_size == entry["size_bytes"], relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], relative_path


def test_statistical_amendment_v2_is_frozen_with_dataset_synchronized_draws():
    freeze = json.loads(
        Path("results/corrected_v2/statistical_amendment_protocol_v2_freeze.json").read_text()
    )
    assert freeze["status"] == "FROZEN_BEFORE_FINAL_STATISTICAL_AMENDMENT_V2_ANALYSIS"
    assert freeze["discovery_phase"] == "second_post_unblinding_methodological_audit"
    assert freeze["decision_thresholds"] is None
    assert freeze["threshold_based_profile_claims_allowed"] is False
    assert freeze["retained_from_v1"]["category_contrasts"] == "exact_two_sided_task_level_sign_flip_with_holm"
    assert freeze["cluster_sensitivity"]["M08"]["grouping_key"] == ["dataset_id"]
    assert freeze["cluster_sensitivity"]["M08"]["shared_draw_scope"] == ["seed", "model", "strength"]
    assert freeze["cluster_sensitivity"]["M08"]["shared_cells_per_inner_draw"] == 125
    assert freeze["cluster_sensitivity"]["M08"]["inferential_practical_null_claim_allowed"] is False
    assert freeze["cluster_sensitivity"]["M09"]["inferential_source_population_claim_allowed"] is False
    for relative_path, entry in freeze["frozen_files"].items():
        path = Path(relative_path)
        assert path.stat().st_size == entry["size_bytes"], relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], relative_path


def test_natural_case_study_protocol_is_frozen_and_hashes_match():
    path = Path("results/corrected_v2/natural_protocol_freeze.json")
    if not path.is_file():
        pytest.skip("private natural source freeze is intentionally excluded publicly")
    freeze = json.loads(path.read_text())
    assert freeze["status"] == "FROZEN_BEFORE_BOUNDARY_CORRECTED_NATURAL_RERUN"
    assert freeze["expected_cells"] == 60
    assert len(freeze["tasks"]) == 5
    for relative_path, entry in freeze["code_files"].items():
        path = Path(relative_path)
        assert path.stat().st_size == entry["size_bytes"], relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], relative_path
    for entry in freeze["source_files"].values():
        path = Path(entry["path"])
        if not path.is_file():
            pytest.skip("raw natural data are intentionally unavailable in this checkout")
        assert path.stat().st_size == entry["size_bytes"], entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]
