import copy
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path("paper/aaai27/source_data/generate_result_macros.py")
SPEC = importlib.util.spec_from_file_location("paper_macro_generator", MODULE_PATH)
generator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(generator)


def _profile(detectability, harm, status="DESCRIPTIVE_ONLY"):
    return {
        "status": status,
        "metrics": {
            "detectability": detectability,
            "detectability_ci_low": detectability - 0.05,
            "detectability_ci_high": detectability + 0.05,
            "paired_harm": harm,
            "paired_harm_ci_low": harm - 0.05,
            "paired_harm_ci_high": harm + 0.05,
        },
    }


def _diagnostic_mechanism(mi):
    values = {
        "mutual_information": mi,
        "absolute_correlation": mi + 0.1,
        "lr_coefficient": mi + 0.2,
        "rf_permutation": mi + 0.3,
    }
    return {
        method: {
            "detectability": value,
            "ci_low": value - 0.05,
            "ci_high": value + 0.05,
        }
        for method, value in values.items()
    }


def _valid_release():
    m08 = _profile(0.6, 0.0)
    m08["metrics"].update({"cluster_ci_low": -0.04, "cluster_ci_high": 0.04})
    m09 = _profile(0.7, 0.2)
    m09["metrics"].update({
        "designed_category_reweighting_low": 0.10,
        "designed_category_reweighting_high": 0.30,
        "designed_category_reweighting_is_inferential": False,
        "detectability_unit": "encoded_column",
        "complete_one_hot": True,
        "encoded_column_count": 8,
        "semantic_field_count": 1,
        "representation_conditional": True,
    })
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-07-13T00:00:00Z",
        "evidence_tier": "confirmatory",
        "protocol_integrity": {
            "dataset_count": 20,
            "mechanism_count": 11,
            "strength_count": 5,
            "model_count": 5,
            "seed_count": 5,
            "expected_cells": 27_500,
            "successful_cells": 27_500,
            "completion_rate": 1.0,
            "models": ["lr", "rf", "catboost", "lightgbm", "tabm"],
        },
        "protocol_amendments": {
            "statistical_inference": {
                "version": "statistical_inference_amendment_v2",
                "discovery_phase": "second_post_unblinding_methodological_audit",
                "category_test": "exact_two_sided_task_level_sign_flip",
                "category_multiplicity": "holm_over_three_declared_contrasts",
                "correlation_interval": "joint_paired_dataset_then_seed_resampling_of_D_and_X",
                "m08_cluster_draw": "shared_across_seeds_models_and_strengths_within_dataset",
                "m08_seed_effects_preserved": True,
                "prediction_arrays_bound_directly_to_frozen_task_bundles": True,
                "m09_reweighting": "descriptive_designed_category_reweighting_only",
                "old_bootstrap_tail_p_accepted": False,
                "old_independent_cell_cluster_interval_accepted": False,
                "v1_seed_independent_entity_interval_accepted": False,
                "threshold_based_profile_claims_allowed": False,
                "protocol_freeze_sha256": "a" * 64,
            },
            "natural_preprocessing": {
                "version": "natural_trainfit_categories_v2",
                "fit_scope": "training rows only",
                "unseen_category_policy": "map_to_reserved_unknown_value",
                "globally_encoded_date_strings_accepted": False,
                "superseded_full_table_category_vocabulary_accepted": False,
                "replacement_cells": 60,
                "protocol_freeze_sha256": "b" * 64,
            },
        },
        "claims": {
            "simple_vs_structured": {
                "status": "SUPPORTED",
                "metrics": {
                    "difference": 0.1,
                    "ci_low": 0.05,
                    "ci_high": 0.15,
                    "holm_p": 0.01,
                    "model_positive_direction_count": 5,
                    "model_ci_excludes_zero_count": 4,
                },
            },
            "m03_profile": _profile(0.2, 0.2),
            "m08_profile": m08,
            "m09_counterexample": m09,
            "detectability_exploitability_relation": {
                "status": "DESCRIPTIVE_ONLY",
                "metrics": {
                    "global_spearman": 0.5,
                    "global_spearman_ci_low": 0.2,
                    "global_spearman_ci_high": 0.8,
                    "category_r2": 0.5,
                    "category_plus_detectability_r2": 0.6,
                    "incremental_r2": 0.1,
                    "incremental_permutation_p": 0.2,
                    "category_lomo_r2": -0.1,
                    "category_plus_detectability_lomo_r2": -0.2,
                    "incremental_lomo_r2": -0.1,
                },
            },
            "D_METHOD_CONDITIONAL": {
                "status": "DESCRIPTIVE_ONLY",
                "metrics": {
                    "m03_method_range": 0.3,
                    "conservative_ci_separation": True,
                    "m03_best_evaluated_method": "rf_permutation",
                    "m03_worst_evaluated_method": "mutual_information",
                },
            },
        },
        "diagnostic_sensitivity": {
            "status": "DESCRIPTIVE_ONLY",
            "method_count": 4,
            "expected_cells": 22_000,
            "successful_cells": 22_000,
            "completion_rate": 1.0,
            "primary_method": "mutual_information",
            "methods": [
                "mutual_information",
                "absolute_correlation",
                "lr_coefficient",
                "rf_permutation",
            ],
            "mechanisms": {
                "M03": _diagnostic_mechanism(0.2),
                "M04": _diagnostic_mechanism(0.1),
                "M05": _diagnostic_mechanism(0.15),
            },
        },
        "natural": {
            "status": "CASE_STUDY_ONLY",
            "task_count": 5,
            "model_count": 4,
            "all_task_effects_positive": True,
            "mean_paired_harm": 0.2,
            "bootstrap_ci_low": 0.1,
            "bootstrap_ci_high": 0.3,
            "exact_sign_flip_p": 0.0625,
        },
        "pending": {"metadata": "PENDING", "governance": "PENDING"},
        "provenance": {"canonical_manifest_sha256": "a" * 64},
    }


def test_complete_confirmatory_release_renders_both_matrix_counts():
    rendered = generator.render_macros(_valid_release(), "b" * 64)
    assert "\\LBResultsReadytrue" in rendered
    assert "\\renewcommand{\\LBExpectedCells}{27,500}" in rendered
    assert "\\renewcommand{\\LBExpectedDiagnosticCells}{22,000}" in rendered
    assert "\\renewcommand{\\LBMThreeDiagnosticMin}{0.200}" in rendered
    assert "\\renewcommand{\\LBMThreeDiagnosticMax}{0.500}" in rendered


def test_cli_writes_macros_only_after_full_validation(tmp_path):
    source = tmp_path / "paper_claims.json"
    output = tmp_path / "result_macros.tex"
    source.write_text(json.dumps(_valid_release()), encoding="utf-8")
    assert generator.main(["--input", str(source), "--output", str(output)]) == 0
    rendered = output.read_text(encoding="utf-8")
    assert "\\LBResultsReadytrue" in rendered
    assert "paper_claims.json sha256:" in rendered


@pytest.mark.parametrize(
    ("field", "value"),
    (("successful_cells", 27_499), ("completion_rate", 0.999)),
)
def test_incomplete_core_release_fails_closed(field, value):
    document = _valid_release()
    document["protocol_integrity"][field] = value
    with pytest.raises(generator.ClaimReleaseError):
        generator.validate_release(document)


def test_nonconfirmatory_or_pilot_release_fails_closed():
    document = _valid_release()
    document["evidence_tier"] = "pilot"
    with pytest.raises(generator.ClaimReleaseError, match="confirmatory"):
        generator.validate_release(document)


def test_macro_generator_rejects_missing_statistical_amendment():
    document = _valid_release()
    del document["protocol_amendments"]["statistical_inference"]
    with pytest.raises(generator.ClaimReleaseError, match="statistical_inference"):
        generator.validate_release(document)


def test_blocked_main_claim_fails_even_when_extra():
    document = _valid_release()
    document["claims"]["metadata"] = {"status": "PENDING", "metrics": {}}
    with pytest.raises(generator.ClaimReleaseError, match="blocked"):
        generator.validate_release(document)


def test_incomplete_diagnostic_suite_fails_closed():
    document = _valid_release()
    document["diagnostic_sensitivity"]["successful_cells"] = 21_999
    with pytest.raises(generator.ClaimReleaseError):
        generator.validate_release(document)


def test_primary_mi_value_must_match_profile():
    document = copy.deepcopy(_valid_release())
    document["diagnostic_sensitivity"]["mechanisms"]["M03"]["mutual_information"][
        "detectability"
    ] = 0.25
    with pytest.raises(generator.ClaimReleaseError, match="disagrees"):
        generator.validate_release(document)


def test_diagnostic_best_method_is_not_presented_as_selector():
    main = Path("paper/aaai27/main.tex").read_text(encoding="utf-8")
    supplement = Path("paper/aaai27/supplement.tex").read_text(encoding="utf-8")
    assert "do not interpret the maximum as an operational" in main
    assert "row-wise maximum is not an" in supplement


def test_m09_designed_sources_cannot_be_rendered_as_inferential_clusters():
    document = _valid_release()
    document["claims"]["m09_counterexample"]["metrics"][
        "designed_category_reweighting_is_inferential"
    ] = True
    with pytest.raises(generator.ClaimReleaseError, match="non-inferential"):
        generator.validate_release(document)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("detectability_unit", "semantic_field"),
        ("complete_one_hot", False),
        ("encoded_column_count", 7),
        ("semantic_field_count", 2),
        ("representation_conditional", False),
    ),
)
def test_m09_representation_contract_is_fail_closed(field, value):
    document = _valid_release()
    document["claims"]["m09_counterexample"]["metrics"][field] = value
    with pytest.raises(generator.ClaimReleaseError, match="representation contract"):
        generator.validate_release(document)


def test_natural_case_studies_cannot_carry_a_support_boolean():
    document = _valid_release()
    document["natural"]["evidence_supported"] = True
    with pytest.raises(generator.ClaimReleaseError, match="support boolean"):
        generator.validate_release(document)


def test_m10_protocol_amendment_is_disclosed_without_outcome_claims():
    supplement = Path("paper/aaai27/supplement.tex").read_text(encoding="utf-8")
    assert "Frozen M10 protocol amendment" in supplement
    assert "original 2,000 CPU and 500 TabM" in supplement
    assert r"task.X[:, \string~leakage\_mask]" in supplement
    assert "excludes every original M10 run identifier" in supplement
    assert "exactly 27,500 model-training cells" in supplement


def test_diagnostic_rng_amendment_and_validation_use_are_disclosed():
    main = Path("paper/aaai27/main.tex").read_text(encoding="utf-8")
    supplement = Path("paper/aaai27/supplement.tex").read_text(encoding="utf-8")
    assert "Post-unblinding evidence reconciliation" in main
    assert "all and only the 5,500 MI rows" in main
    assert "Frozen diagnostic RNG amendment" in supplement
    assert "every and only the 5,500 mutual-information rows" in supplement
    assert "other 16,500 rows" in supplement
    assert "model-outcome columns" in supplement
    assert "RF permutation" in main and "frozen validation" in main
    assert "RF permutation" in supplement and "frozen validation" in supplement
    assert "representation-conditional" in main
    assert "preregister" not in (main + supplement).lower()


def test_readme_documents_official_docker_pdflatex_fallback():
    readme = Path("paper/aaai27/README.md").read_text(encoding="utf-8")
    assert "texlive/texlive:latest-small" in readme
    assert "absence of host PDFLaTeX is not" in readme
