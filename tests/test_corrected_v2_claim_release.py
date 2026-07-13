"""Claim-scope and schema tests for the fail-closed corrected_v2 release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_corrected_v2_claim_state import (
    CATEGORIES,
    DIAGNOSTIC_METHODS,
    M10_FREEZE_SHA256,
    build_diagnostic_sensitivity,
    derive_claims,
    load_superseded,
    reject_superseded,
    validate_diagnostic_statistics_schema,
    validate_statistics_schema,
)


def _evidence_fixture():
    mechanism_rows = []
    detectability_rows = []
    for name, category in CATEGORIES.items():
        harm, harm_low, harm_high, holm = 0.04, 0.01, 0.07, 0.04
        detectability, d_low, d_high = 0.4, 0.35, 0.45
        if name == "M03":
            harm, harm_low, harm_high, holm = 0.20, 0.10, 0.30, 0.001
            detectability, d_low, d_high = 0.20, 0.15, 0.25
        elif name == "M08":
            harm, harm_low, harm_high, holm = 0.00, -0.01, 0.01, 1.0
            detectability, d_low, d_high = 0.60, 0.55, 0.65
        elif name == "M09":
            harm, harm_low, harm_high, holm = 0.20, 0.10, 0.30, 0.001
            detectability, d_low, d_high = 0.80, 0.70, 0.90
        mechanism_rows.append({
            "mechanism": name, "category": category, "paired_harm": harm,
            "paired_harm_ci_low": harm_low, "paired_harm_ci_high": harm_high,
            "diagnostic_normalized_ap": detectability, "sign_flip_p": holm, "holm_p": holm,
        })
        detectability_rows.append({
            "mechanism": name, "category": category,
            "diagnostic_normalized_ap": detectability,
            "diagnostic_normalized_ap_ci_low": d_low,
            "diagnostic_normalized_ap_ci_high": d_high,
            "top5_recall": 0.5, "top5_recall_ci_low": 0.4, "top5_recall_ci_high": 0.6,
        })
    by_model = pd.DataFrame([
        {
            "model": model, "simple_minus_structured": 0.1,
            "ci_low": 0.02 if index < 4 else -0.01, "ci_high": 0.2,
            "sign_flip_p": 0.01, "holm_p": 0.04,
        }
        for index, model in enumerate(["catboost", "lightgbm", "lr", "rf", "tabm"])
    ])
    diagnostic_rows = []
    method_values = {
        "mutual_information": 0.20,
        "absolute_correlation": 0.40,
        "lr_coefficient": 0.10,
        "rf_permutation": 0.25,
    }
    for mechanism in CATEGORIES:
        for method, value in method_values.items():
            diagnostic_rows.append({
                "method": method, "mechanism": mechanism,
                "diagnostic_normalized_ap": value,
                "ci_low": value - 0.05, "ci_high": value + 0.05,
            })
    profiles = pd.DataFrame([
        {
            "mechanism": mechanism,
            "between_diagnostic_range": 0.30,
            "best_evaluated_diagnostic": 0.40,
            "worst_evaluated_diagnostic": 0.10,
            "low_across_all_evaluated_diagnostics": mechanism in {"M04", "M05"},
        }
        for mechanism in CATEGORIES
    ])
    return {
        "statistics": {
            "mechanism": pd.DataFrame(mechanism_rows),
            "detectability": pd.DataFrame(detectability_rows),
            "category_contrasts": pd.DataFrame([{
                "contrast": "simple_minus_structured", "difference": 0.15,
                "ci_low": 0.05, "ci_high": 0.25,
                "sign_flip_p": 0.001, "holm_p": 0.003,
            }]),
            "by_model": by_model,
            "correlation": {
                "global_spearman": 0.7, "global_spearman_ci": [0.4, 0.9],
                "category_r2": 0.5, "category_plus_detectability_r2": 0.6,
                "incremental_r2": 0.1, "incremental_permutation_p": 0.2,
                "category_lomo_r2": 0.1, "category_plus_detectability_lomo_r2": -0.2,
                "incremental_lomo_r2": -0.3,
            },
            "cluster": {
                "M08": {"synchronized_cluster_ci": [-0.02, 0.02]},
                "M09": {"descriptive_reweighting_interval": [0.08, 0.25]},
            },
        },
        "diagnostic": {
            "by_mechanism": pd.DataFrame(diagnostic_rows),
            "profiles": profiles,
        },
    }


def test_superseded_pilot_statistics_cannot_satisfy_amended_claim_schema():
    with pytest.raises(FileNotFoundError):
        validate_statistics_schema(Path("results/corrected_v2/pilot_statistics"))
    validate_diagnostic_statistics_schema(Path("results/corrected_v2/diagnostic_pilot_statistics"))


def test_only_directional_exact_holm_contrast_can_be_supported():
    evidence = _evidence_fixture()
    claims = derive_claims(evidence)
    assert claims["simple_vs_structured"]["status"] == "SUPPORTED"
    assert claims["simple_vs_structured"]["metrics"]["model_positive_direction_count"] == 5
    assert claims["simple_vs_structured"]["metrics"]["model_ci_excludes_zero_count"] == 4
    assert claims["m03_profile"]["status"] == "DESCRIPTIVE_ONLY"
    assert claims["m08_profile"]["status"] == "DESCRIPTIVE_ONLY"
    assert claims["m09_counterexample"]["status"] == "DESCRIPTIVE_ONLY"
    assert claims["m09_counterexample"]["metrics"]["designed_category_reweighting_is_inferential"] is False
    assert claims["m09_counterexample"]["metrics"]["detectability_unit"] == "encoded_column"
    assert claims["m09_counterexample"]["metrics"]["complete_one_hot"] is True
    assert claims["m09_counterexample"]["metrics"]["encoded_column_count"] == 8
    assert claims["m09_counterexample"]["metrics"]["semantic_field_count"] == 1
    assert claims["m09_counterexample"]["metrics"]["representation_conditional"] is True
    assert claims["detectability_exploitability_relation"]["status"] == "DESCRIPTIVE_ONLY"
    assert claims["D_METHOD_CONDITIONAL"]["status"] == "DESCRIPTIVE_ONLY"
    assert "m04_m05_low_across_all" not in claims["D_METHOD_CONDITIONAL"]["metrics"]
    assert "non-deployable" in claims["D_METHOD_CONDITIONAL"]["allowed_wording"]


def test_model_and_profile_values_do_not_drive_support_decisions():
    evidence = _evidence_fixture()
    evidence["statistics"]["by_model"].loc[0, "simple_minus_structured"] = -0.01
    evidence["statistics"]["cluster"]["M08"]["synchronized_cluster_ci"] = [-0.04, 0.02]
    evidence["statistics"]["detectability"].loc[
        evidence["statistics"]["detectability"]["mechanism"] == "M09",
        "diagnostic_normalized_ap_ci_low",
    ] = 0.40
    evidence["diagnostic"]["by_mechanism"].loc[
        (evidence["diagnostic"]["by_mechanism"]["mechanism"] == "M03")
        & (evidence["diagnostic"]["by_mechanism"]["method"] == "absolute_correlation"),
        "ci_low",
    ] = 0.12
    claims = derive_claims(evidence)
    assert claims["simple_vs_structured"]["status"] == "SUPPORTED"
    assert claims["m08_profile"]["status"] == "DESCRIPTIVE_ONLY"
    assert claims["m09_counterexample"]["status"] == "DESCRIPTIVE_ONLY"
    assert claims["D_METHOD_CONDITIONAL"]["status"] == "DESCRIPTIVE_ONLY"


@pytest.mark.parametrize(("field", "value"), (("holm_p", 0.051), ("ci_low", -0.001)))
def test_exact_holm_or_ci_failure_downgrades_the_directional_claim(field, value):
    evidence = _evidence_fixture()
    evidence["statistics"]["category_contrasts"].loc[0, field] = value
    assert derive_claims(evidence)["simple_vs_structured"]["status"] == "NOT_SUPPORTED"


def test_diagnostic_paper_schema_is_exactly_three_by_four():
    sensitivity = build_diagnostic_sensitivity(_evidence_fixture())
    assert sensitivity["status"] == "DESCRIPTIVE_ONLY"
    assert sensitivity["methods"] == DIAGNOSTIC_METHODS
    assert set(sensitivity["mechanisms"]) == {"M03", "M04", "M05"}
    assert all(set(methods) == set(DIAGNOSTIC_METHODS) for methods in sensitivity["mechanisms"].values())


def test_superseded_evidence_is_rejected_even_if_it_exists():
    with pytest.raises(ValueError, match="Superseded evidence is forbidden"):
        reject_superseded(
            [Path("results/ce2r_neural.csv")],
            {"results/ce2r_neural.csv"},
        )


def test_selector_scoped_raw_provenance_is_not_treated_as_whole_file_superseded():
    whole_file = load_superseded(
        Path("results/corrected_v2/superseded_evidence.json")
    )
    assert "results/ce2r_neural.csv" in whole_file
    assert "results/corrected_v2/core_cpu_cells.csv" not in whole_file
    assert (
        "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv"
        not in whole_file
    )
    assert "results/corrected_v2/diagnostic_confirmatory_cells.csv" not in whole_file


def test_claim_release_binds_the_reviewed_m10_freeze_and_replacement_paths():
    path = Path("results/corrected_v2/m10_amendment_protocol_freeze.json")
    freeze = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == M10_FREEZE_SHA256
    assert freeze["status"] == "FROZEN_BEFORE_M10_AMENDMENT_CONFIRMATORY_RUN"
    assert freeze["verified_confirmatory_tasks"] == 500
    assert freeze["outputs"] == {
        "cpu": "results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv",
        "tabm": "results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv",
    }
    assert freeze["output_manifests"] == {
        "cpu": "results/corrected_v2/m10_amendment_confirmatory/cpu_cells_manifest.json",
        "tabm": "results/corrected_v2/m10_amendment_confirmatory/tabm_cells_manifest.json",
    }
