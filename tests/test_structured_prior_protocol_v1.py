"""Frozen-design checks for the structured replacement and replication v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.leakbench.run_structured_prior_v1_bundle import main as run_bundle


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocols/structured_prior_v1"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_task_counts_and_model_cell_arithmetic():
    replacement = pd.read_csv(PROTOCOL_DIR / "structured_prior_replacement_v1_tasks.csv")
    replication = pd.read_csv(PROTOCOL_DIR / "independent_replication_v1_tasks.csv")

    assert len(replacement) == 1500
    assert int(replacement["expected_model_cells"].sum()) == 7500
    assert replacement["dataset_index"].nunique() == 20
    assert set(replacement["mechanism"]) == {"M04", "M05", "M08"}

    assert len(replication) == 3000
    assert int(replication["expected_model_cells"].sum()) == 15000
    assert replication["dataset_index"].nunique() == 25
    assert set(replication["dataset_namespace"]) == {"independent_replication_v1"}
    assert set(replication["seed"].astype(int)) == {13, 2026, 7777}
    assert set(replication["mechanism"]) == {
        "M01", "M02", "M06", "M10", "M04", "M05", "M08", "M09"
    }


def test_replication_is_five_preselected_tasks_per_archetype_and_disjoint():
    replacement = pd.read_csv(PROTOCOL_DIR / "structured_prior_replacement_v1_tasks.csv")
    replication = pd.read_csv(PROTOCOL_DIR / "independent_replication_v1_tasks.csv")
    generators = replication.drop_duplicates("dataset_index")

    assert generators["archetype"].value_counts().to_dict() == {
        "linear": 5,
        "interaction": 5,
        "nonlinear": 5,
        "sparse": 5,
        "drifting": 5,
    }
    assert set(replacement["dataset_index"]).isdisjoint(set(replication["dataset_index"]))
    assert sorted(generators["dataset_index"].astype(int)) == list(range(1000, 1025))


def test_freeze_hashes_bind_all_generated_protocol_outputs():
    freeze = json.loads((PROTOCOL_DIR / "freeze_manifest_v1.json").read_text())
    assert freeze["status"] == "FROZEN_BEFORE_ANY_MODEL_RUN"
    assert freeze["model_results_observed"] is False
    assert freeze["models_executed"] == 0
    for relative, entry in freeze["files"].items():
        path = ROOT / relative
        assert path.is_file()
        assert _sha256(path) == entry["sha256"]
        assert path.stat().st_size == entry["size_bytes"]


def test_inference_protocol_is_task_first_complete_case_and_single_claim():
    inference = json.loads((PROTOCOL_DIR / "inference_protocol_v1.json").read_text())
    decision = inference["decision"]
    assert inference["independent_unit"] == "generator_task"
    assert inference["within_task_aggregation"]["cells_per_task"] == 600
    assert inference["confidence_interval"]["strata"] == "archetype"
    assert inference["sign_test"]["trials"] == 25
    assert decision["multiplicity_family_size"] == 1
    assert decision["supported_iff"] == [
        "all 15000 prespecified model cells are SUCCESS and integrity_verified",
        "lower endpoint of the frozen 95 percent bootstrap CI is greater than 0",
        "exact two-sided binomial sign-test p-value is less than or equal to 0.05",
    ]
    assert inference["exclusions_and_missingness"]["missing_cell"] == (
        "invalidates the primary inference"
    )
    assert inference["exclusions_and_missingness"]["task_substitution"] == "forbidden"


def test_configs_freeze_required_namespaces_seeds_and_execution_gate():
    replacement = yaml.safe_load(
        (ROOT / "configs/paper/structured_prior_replacement_v1.yaml").read_text()
    )["protocol"]
    replication = yaml.safe_load(
        (ROOT / "configs/paper/independent_replication_v1.yaml").read_text()
    )["protocol"]
    assert replacement["dataset_namespace"] == "confirmatory"
    assert replacement["seeds"] == [13, 42, 2026, 3407, 7777]
    assert replication["dataset_namespace"] == "independent_replication_v1"
    assert replication["seeds"] == [13, 2026, 7777]
    assert replacement["execution_gate"] == replication["execution_gate"] == "explicit_allow_run"


def test_model_runner_fails_closed_without_explicit_authorization():
    with pytest.raises(RuntimeError, match="model execution is locked"):
        run_bundle([
            "--config", "configs/paper/independent_replication_v1.yaml",
            "--task-manifest", "does-not-exist.csv",
        ])
