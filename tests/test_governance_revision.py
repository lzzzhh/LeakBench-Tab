"""Regression tests for the EDBT governance revision evidence chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_governance_revision import (
    SEED,
    get_paired,
    learner_interaction_boot,
    task_boot,
)
from scripts.build_governance_revision_claim_state import derive


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "results/edbt_eab_revision"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_fixture():
    rows = []
    for dataset_index in (0, 1):
        key = {
            "dataset_index": dataset_index,
            "mechanism": "M01",
            "strength": "S1",
            "training_seed": 13,
            "status": "SUCCESS",
            "initial_gap": 0.2,
        }
        rows.append({**key, "policy": "P0_keep", "governance_seed": -1,
                     "budget_fraction": 0.0, "strict_distance_reduction": 0.0})
        for budget, p3_value in ((0.10, 9.0), (0.20, 0.4 + 0.2 * dataset_index)):
            rows.append({**key, "policy": "P3_blind_mi", "governance_seed": -1,
                         "budget_fraction": budget, "strict_distance_reduction": p3_value})
            for governance_seed, p2_value in ((101, 0.1), (102, 0.3)):
                rows.append({**key, "policy": "P2_random", "governance_seed": governance_seed,
                             "budget_fraction": budget, "strict_distance_reduction": p2_value})
    return pd.DataFrame(rows)


def test_get_paired_filters_one_explicit_budget():
    paired = get_paired(paired_fixture(), budget=0.20)
    assert len(paired) == 2
    assert set(paired["budget_fraction"]) == {0.20}
    assert paired.sort_values("dataset_index")["paired"].tolist() == pytest.approx([0.2, 0.4])


def test_task_boot_uses_paired_task_draws_for_probability():
    p3 = {0: 0.9, 1: 0.4, 2: 0.2, 3: -0.2}
    p2 = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.0}
    repetitions = 200
    observed, bootstrap_mean, low, high, probability = task_boot(p3, p2, nb=repetitions)
    differences = np.asarray([0.8, 0.1, -0.3, -0.2])
    rng = np.random.RandomState(SEED)
    draws = differences[rng.randint(0, 4, size=(repetitions, 4))].mean(axis=1)
    assert observed == pytest.approx(differences.mean())
    assert bootstrap_mean == pytest.approx(draws.mean())
    assert (low, high) == pytest.approx((np.percentile(draws, 2.5), np.percentile(draws, 97.5)))
    assert probability == pytest.approx(float(np.mean(draws > 0)))


def test_learner_interaction_separates_observed_and_bootstrap_mean():
    left = get_paired(paired_fixture(), budget=0.20)
    right = left.copy()
    right["paired"] = right["paired"] - 0.05
    observed, bootstrap_mean, low, high, probability = learner_interaction_boot(left, right, nb=200)
    assert observed == pytest.approx(0.05)
    assert bootstrap_mean == pytest.approx(0.05)
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.05)
    assert probability == 1.0


def test_formal_summary_is_matched_at_twenty_percent():
    summary = json.loads((REVISION / "analysis_summary.json").read_text())
    assert summary["primary_budget"] == 0.20
    assert summary["expected_keys_per_model"] == 5500
    for model in ("LR", "RF", "LightGBM"):
        assert summary[f"{model}_overall"]["n_keys"] == 5500
    assert summary["gap_quartile_Q1_low"]["n_keys"] == 1375
    assert summary["gap_quartile_Q4_high"]["n_keys"] == 1375
    assert summary["LR_overall"]["paired"] == pytest.approx(0.043413, abs=1e-6)
    assert summary["LR_M09"]["paired"] == pytest.approx(0.148903, abs=1e-6)


def test_claim_state_is_builder_derivable():
    summary = json.loads((REVISION / "analysis_summary.json").read_text())
    expected = derive(summary)
    actual = json.loads((REVISION / "claim_state.json").read_text())
    assert actual["claims"] == expected["claims"]
    assert actual["analysis_summary_sha256"] == sha256(REVISION / "analysis_summary.json")


def test_selection_hashes_are_complete_and_cross_model_matched():
    columns = [
        "dataset_index", "mechanism", "strength", "training_seed", "governance_seed",
        "policy", "budget_k", "budget_fraction", "selection_mask_hash",
    ]
    lr = pd.read_csv(REVISION / "b1_multiseed_p2.csv", usecols=columns)
    lr = lr[np.isclose(lr["budget_fraction"], 0.20)]
    assert lr["selection_mask_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    join = ["dataset_index", "mechanism", "strength", "training_seed", "governance_seed", "policy", "budget_k"]
    reference = lr[join + ["selection_mask_hash"]]
    for filename in ("b2_rf.csv", "b2_lgbm.csv"):
        other = pd.read_csv(REVISION / filename, usecols=columns)
        merged = reference.merge(other, on=join, suffixes=("_lr", "_other"), validate="one_to_one")
        assert len(merged) == len(reference)
        assert (merged["selection_mask_hash_lr"] == merged["selection_mask_hash_other"]).all()


def test_revision_manifest_binds_every_declared_artifact():
    manifest = json.loads((REVISION / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE_WITH_DISCLOSED_LIMITATIONS"
    for entry in manifest["artifacts"]:
        path = ROOT / entry["path"]
        assert path.exists()
        assert entry["sha256"] == sha256(path)
