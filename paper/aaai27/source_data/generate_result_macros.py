#!/usr/bin/env python3
"""Generate AAAI paper macros from the final corrected_v2 claim release.

This script is intentionally fail-closed.  It accepts only the complete
confirmatory protocol and never reads pilot tables, legacy ledgers, or the
metadata/governance evidence that is outside the main-paper claim scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "results/corrected_v2/paper_claims.json"
DEFAULT_OUTPUT = ROOT / "paper/aaai27/generated/result_macros.tex"

EXPECTED_PROTOCOL = {
    "dataset_count": 20,
    "mechanism_count": 11,
    "strength_count": 5,
    "model_count": 5,
    "seed_count": 5,
    "expected_cells": 27_500,
    "successful_cells": 27_500,
}
EXPECTED_MODELS = {"lr", "rf", "catboost", "lightgbm", "tabm"}
DIAGNOSTIC_METHODS = (
    "mutual_information",
    "absolute_correlation",
    "lr_coefficient",
    "rf_permutation",
)
DIAGNOSTIC_MECHANISMS = ("M03", "M04", "M05")
PROFILE_METRICS = (
    "detectability",
    "detectability_ci_low",
    "detectability_ci_high",
    "paired_harm",
    "paired_harm_ci_low",
    "paired_harm_ci_high",
)
REQUIRED_CLAIMS = {
    "simple_vs_structured",
    "m03_profile",
    "m08_profile",
    "m09_counterexample",
    "detectability_exploitability_relation",
    "D_METHOD_CONDITIONAL",
}
FINAL_STATUSES = {"SUPPORTED", "NOT_SUPPORTED"}
BLOCKED_STATUSES = {"PENDING", "REFUTED", "INTEGRITY_HOLD", "BLOCKED"}


class ClaimReleaseError(ValueError):
    """Raised when a paper claim release is unsafe to render."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaimReleaseError(f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClaimReleaseError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ClaimReleaseError(f"{label} must be finite")
    return value


def _integer(value: Any, label: str) -> int:
    number = _finite_number(value, label)
    if not number.is_integer():
        raise ClaimReleaseError(f"{label} must be an integer")
    return int(number)


def _metric(metrics: Mapping[str, Any], name: str, claim_id: str) -> float:
    if name not in metrics:
        raise ClaimReleaseError(f"claims.{claim_id}.metrics.{name} is required")
    return _finite_number(metrics[name], f"claims.{claim_id}.metrics.{name}")


def _interval(
    metrics: Mapping[str, Any],
    claim_id: str,
    point_name: str,
    low_name: str,
    high_name: str,
) -> None:
    point = _metric(metrics, point_name, claim_id)
    low = _metric(metrics, low_name, claim_id)
    high = _metric(metrics, high_name, claim_id)
    if not low <= point <= high:
        raise ClaimReleaseError(
            f"claims.{claim_id} has unordered {point_name} interval: "
            f"{low} <= {point} <= {high} is false"
        )


def _probability(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ClaimReleaseError(f"{label} must lie in [0, 1]")


def _validate_protocol(protocol: Mapping[str, Any]) -> None:
    for field, expected in EXPECTED_PROTOCOL.items():
        observed = _integer(protocol.get(field), f"protocol_integrity.{field}")
        if observed != expected:
            raise ClaimReleaseError(
                f"protocol_integrity.{field}={observed}, expected {expected}"
            )
    completion_rate = _finite_number(
        protocol.get("completion_rate"), "protocol_integrity.completion_rate"
    )
    if not math.isclose(completion_rate, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ClaimReleaseError("confirmatory completion_rate must equal 1.0")
    models = protocol.get("models")
    if not isinstance(models, list) or set(models) != EXPECTED_MODELS or len(models) != 5:
        raise ClaimReleaseError(
            "protocol_integrity.models must contain exactly lr, rf, catboost, "
            "lightgbm, and tabm"
        )
    product = (
        _integer(protocol["dataset_count"], "dataset_count")
        * _integer(protocol["mechanism_count"], "mechanism_count")
        * _integer(protocol["strength_count"], "strength_count")
        * _integer(protocol["model_count"], "model_count")
        * _integer(protocol["seed_count"], "seed_count")
    )
    if product != protocol["expected_cells"]:
        raise ClaimReleaseError("protocol dimensions do not multiply to expected_cells")


def _validate_required_amendments(amendments: Mapping[str, Any]) -> None:
    statistical = _mapping(
        amendments.get("statistical_inference"),
        "protocol_amendments.statistical_inference",
    )
    required_statistical = {
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
    }
    for field, expected in required_statistical.items():
        if statistical.get(field) != expected:
            raise ClaimReleaseError(
                f"protocol_amendments.statistical_inference.{field} must equal {expected!r}"
            )
    natural = _mapping(
        amendments.get("natural_preprocessing"),
        "protocol_amendments.natural_preprocessing",
    )
    required_natural = {
        "version": "natural_trainfit_categories_v2",
        "fit_scope": "training rows only",
        "unseen_category_policy": "map_to_reserved_unknown_value",
        "globally_encoded_date_strings_accepted": False,
        "superseded_full_table_category_vocabulary_accepted": False,
        "replacement_cells": 60,
    }
    for field, expected in required_natural.items():
        if natural.get(field) != expected:
            raise ClaimReleaseError(
                f"protocol_amendments.natural_preprocessing.{field} must equal {expected!r}"
            )


def _validate_final_claim(claims: Mapping[str, Any], claim_id: str) -> Mapping[str, Any]:
    claim = _mapping(claims.get(claim_id), f"claims.{claim_id}")
    status = claim.get("status")
    if status in BLOCKED_STATUSES or status not in FINAL_STATUSES:
        raise ClaimReleaseError(
            f"claims.{claim_id}.status must be SUPPORTED or NOT_SUPPORTED, got {status!r}"
        )
    metrics = _mapping(claim.get("metrics"), f"claims.{claim_id}.metrics")
    return {"status": status, "metrics": metrics}


def _validate_descriptive_profile(
    claims: Mapping[str, Any], claim_id: str
) -> Mapping[str, Any]:
    claim = _mapping(claims.get(claim_id), f"claims.{claim_id}")
    if claim.get("status") != "DESCRIPTIVE_ONLY":
        raise ClaimReleaseError(
            f"claims.{claim_id}.status must remain DESCRIPTIVE_ONLY"
        )
    return {
        "status": "DESCRIPTIVE_ONLY",
        "metrics": _mapping(claim.get("metrics"), f"claims.{claim_id}.metrics"),
    }


def _validate_diagnostic_sensitivity(
    block: Mapping[str, Any], primary_m03_detectability: float
) -> Mapping[str, Any]:
    if block.get("status") != "DESCRIPTIVE_ONLY":
        raise ClaimReleaseError("diagnostic_sensitivity must remain DESCRIPTIVE_ONLY")
    for field, expected in (
        ("method_count", 4),
        ("expected_cells", 22_000),
        ("successful_cells", 22_000),
    ):
        if _integer(block.get(field), f"diagnostic_sensitivity.{field}") != expected:
            raise ClaimReleaseError(f"diagnostic_sensitivity.{field} must equal {expected}")
    completion_rate = _finite_number(
        block.get("completion_rate"), "diagnostic_sensitivity.completion_rate"
    )
    if not math.isclose(completion_rate, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ClaimReleaseError("diagnostic_sensitivity.completion_rate must equal 1.0")
    if block.get("primary_method") != "mutual_information":
        raise ClaimReleaseError("diagnostic_sensitivity.primary_method must be mutual_information")
    methods = block.get("methods")
    if not isinstance(methods, list) or tuple(methods) != DIAGNOSTIC_METHODS:
        raise ClaimReleaseError(
            "diagnostic_sensitivity.methods must use the frozen four-method order"
        )
    mechanisms = _mapping(
        block.get("mechanisms"), "diagnostic_sensitivity.mechanisms"
    )
    for mechanism in DIAGNOSTIC_MECHANISMS:
        method_rows = _mapping(
            mechanisms.get(mechanism), f"diagnostic_sensitivity.mechanisms.{mechanism}"
        )
        for method in DIAGNOSTIC_METHODS:
            row = _mapping(
                method_rows.get(method),
                f"diagnostic_sensitivity.mechanisms.{mechanism}.{method}",
            )
            point = _finite_number(
                row.get("detectability"),
                f"diagnostic_sensitivity.mechanisms.{mechanism}.{method}.detectability",
            )
            low = _finite_number(
                row.get("ci_low"),
                f"diagnostic_sensitivity.mechanisms.{mechanism}.{method}.ci_low",
            )
            high = _finite_number(
                row.get("ci_high"),
                f"diagnostic_sensitivity.mechanisms.{mechanism}.{method}.ci_high",
            )
            if not low <= point <= high:
                raise ClaimReleaseError(
                    f"diagnostic_sensitivity {mechanism}/{method} interval is unordered"
                )
    m03_mi = mechanisms["M03"]["mutual_information"]["detectability"]
    if not math.isclose(
        float(m03_mi), primary_m03_detectability, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ClaimReleaseError(
            "M03 primary MI detectability disagrees between profile and diagnostic suite"
        )
    return block


def validate_release(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return the normalized sections needed by the renderer."""

    if document.get("schema_version") != 1:
        raise ClaimReleaseError("schema_version must equal 1")
    if document.get("evidence_tier") != "confirmatory":
        raise ClaimReleaseError("evidence_tier must equal 'confirmatory'")
    if not isinstance(document.get("generated_at_utc"), str) or not document["generated_at_utc"]:
        raise ClaimReleaseError("generated_at_utc is required")

    protocol = _mapping(document.get("protocol_integrity"), "protocol_integrity")
    _validate_protocol(protocol)
    _validate_required_amendments(
        _mapping(document.get("protocol_amendments"), "protocol_amendments")
    )
    claims = _mapping(document.get("claims"), "claims")
    missing = REQUIRED_CLAIMS - set(claims)
    if missing:
        raise ClaimReleaseError(f"missing required main-paper claims: {sorted(missing)}")
    for claim_id, raw_claim in claims.items():
        claim_status = _mapping(raw_claim, f"claims.{claim_id}").get("status")
        if claim_status in BLOCKED_STATUSES:
            raise ClaimReleaseError(
                f"claims.{claim_id}.status={claim_status!r} is blocked from paper generation"
            )

    primary = _validate_final_claim(claims, "simple_vs_structured")
    primary_metrics = primary["metrics"]
    _interval(primary_metrics, "simple_vs_structured", "difference", "ci_low", "ci_high")
    _probability(
        _metric(primary_metrics, "holm_p", "simple_vs_structured"),
        "claims.simple_vs_structured.metrics.holm_p",
    )
    expected_supported = bool(
        float(primary_metrics["ci_low"]) > 0.0
        and float(primary_metrics["holm_p"]) <= 0.05
    )
    if (primary["status"] == "SUPPORTED") != expected_supported:
        raise ClaimReleaseError(
            "simple_vs_structured status must apply exact Holm p<=0.05 and CI low>0"
        )
    for field in ("model_positive_direction_count", "model_ci_excludes_zero_count"):
        count = _integer(primary_metrics.get(field), f"claims.simple_vs_structured.metrics.{field}")
        if not 0 <= count <= 5:
            raise ClaimReleaseError(f"{field} must lie in [0, 5]")

    profiles: dict[str, Mapping[str, Any]] = {}
    for claim_id in ("m03_profile", "m08_profile", "m09_counterexample"):
        claim = _validate_descriptive_profile(claims, claim_id)
        metrics = claim["metrics"]
        for field in PROFILE_METRICS:
            _metric(metrics, field, claim_id)
        _interval(
            metrics,
            claim_id,
            "detectability",
            "detectability_ci_low",
            "detectability_ci_high",
        )
        _interval(
            metrics,
            claim_id,
            "paired_harm",
            "paired_harm_ci_low",
            "paired_harm_ci_high",
        )
        if claim_id == "m08_profile":
            low = _metric(metrics, "cluster_ci_low", claim_id)
            high = _metric(metrics, "cluster_ci_high", claim_id)
            if low > high:
                raise ClaimReleaseError("M08 cluster interval is unordered")
        profiles[claim_id] = claim
    m09_metrics = profiles["m09_counterexample"]["metrics"]
    if m09_metrics.get("designed_category_reweighting_is_inferential") is not False:
        raise ClaimReleaseError(
            "M09 designed-category reweighting must be explicitly non-inferential"
        )
    expected_m09_contract = {
        "detectability_unit": "encoded_column",
        "complete_one_hot": True,
        "encoded_column_count": 8,
        "semantic_field_count": 1,
        "representation_conditional": True,
    }
    for field, expected in expected_m09_contract.items():
        if m09_metrics.get(field) != expected:
            raise ClaimReleaseError(
                f"M09 representation contract {field} must equal {expected!r}"
            )
    m09_reweighting_low = _metric(
        m09_metrics, "designed_category_reweighting_low", "m09_counterexample"
    )
    m09_reweighting_high = _metric(
        m09_metrics, "designed_category_reweighting_high", "m09_counterexample"
    )
    if m09_reweighting_low > m09_reweighting_high:
        raise ClaimReleaseError("M09 descriptive reweighting interval is unordered")
    if "cluster_ci_low" in m09_metrics or "cluster_ci_high" in m09_metrics:
        raise ClaimReleaseError("M09 source categories cannot be rendered as an inferential cluster CI")

    relation = _mapping(
        claims.get("detectability_exploitability_relation"),
        "claims.detectability_exploitability_relation",
    )
    if relation.get("status") != "DESCRIPTIVE_ONLY":
        raise ClaimReleaseError(
            "detectability_exploitability_relation must remain DESCRIPTIVE_ONLY"
        )
    relation_metrics = _mapping(
        relation.get("metrics"), "claims.detectability_exploitability_relation.metrics"
    )
    relation_fields = (
        "global_spearman",
        "global_spearman_ci_low",
        "global_spearman_ci_high",
        "category_r2",
        "category_plus_detectability_r2",
        "incremental_r2",
        "incremental_permutation_p",
        "category_lomo_r2",
        "category_plus_detectability_lomo_r2",
        "incremental_lomo_r2",
    )
    for field in relation_fields:
        _metric(relation_metrics, field, "detectability_exploitability_relation")
    _interval(
        relation_metrics,
        "detectability_exploitability_relation",
        "global_spearman",
        "global_spearman_ci_low",
        "global_spearman_ci_high",
    )
    _probability(
        _metric(
            relation_metrics,
            "incremental_permutation_p",
            "detectability_exploitability_relation",
        ),
        "claims.detectability_exploitability_relation.metrics.incremental_permutation_p",
    )

    diagnostic_claim = _mapping(
        claims.get("D_METHOD_CONDITIONAL"), "claims.D_METHOD_CONDITIONAL"
    )
    if diagnostic_claim.get("status") != "DESCRIPTIVE_ONLY":
        raise ClaimReleaseError(
            "D_METHOD_CONDITIONAL must remain DESCRIPTIVE_ONLY without paired simultaneous comparison"
        )
    diagnostic_claim_metrics = _mapping(
        diagnostic_claim.get("metrics"), "claims.D_METHOD_CONDITIONAL.metrics"
    )
    for field in (
        "m03_method_range",
        "conservative_ci_separation",
        "m03_best_evaluated_method",
        "m03_worst_evaluated_method",
    ):
        if field not in diagnostic_claim_metrics:
            raise ClaimReleaseError(f"claims.D_METHOD_CONDITIONAL.metrics.{field} is required")

    natural = _mapping(document.get("natural"), "natural")
    if natural.get("status") != "CASE_STUDY_ONLY":
        raise ClaimReleaseError("natural.status must equal CASE_STUDY_ONLY")
    if "evidence_supported" in natural:
        raise ClaimReleaseError("natural evidence cannot carry a support boolean")
    if _integer(natural.get("task_count"), "natural.task_count") != 5:
        raise ClaimReleaseError("natural.task_count must equal 5")
    if _integer(natural.get("model_count"), "natural.model_count") != 4:
        raise ClaimReleaseError("natural.model_count must equal 4")
    if not isinstance(natural.get("all_task_effects_positive"), bool):
        raise ClaimReleaseError("natural.all_task_effects_positive must be boolean")
    natural_point = _finite_number(natural.get("mean_paired_harm"), "natural.mean_paired_harm")
    natural_low = _finite_number(natural.get("bootstrap_ci_low"), "natural.bootstrap_ci_low")
    natural_high = _finite_number(natural.get("bootstrap_ci_high"), "natural.bootstrap_ci_high")
    if not natural_low <= natural_point <= natural_high:
        raise ClaimReleaseError("natural bootstrap interval is unordered")
    _probability(
        _finite_number(natural.get("exact_sign_flip_p"), "natural.exact_sign_flip_p"),
        "natural.exact_sign_flip_p",
    )

    diagnostic_sensitivity = _validate_diagnostic_sensitivity(
        _mapping(document.get("diagnostic_sensitivity"), "diagnostic_sensitivity"),
        float(profiles["m03_profile"]["metrics"]["detectability"]),
    )

    provenance = _mapping(document.get("provenance"), "provenance")
    if not provenance:
        raise ClaimReleaseError("provenance must not be empty")
    return {
        "protocol": protocol,
        "primary": primary,
        "profiles": profiles,
        "relation": {"status": relation["status"], "metrics": relation_metrics},
        "natural": natural,
        "diagnostic_sensitivity": diagnostic_sensitivity,
        "diagnostic_claim": diagnostic_claim,
    }


def _estimate(value: float) -> str:
    return f"{value:.3f}"


def _p_value(value: float) -> str:
    return r"<0.001" if value < 0.001 else f"{value:.3f}"


def _status_text(status: str) -> str:
    return {
        "SUPPORTED": "supported",
        "NOT_SUPPORTED": "not supported",
        "DESCRIPTIVE_ONLY": "descriptive only",
        "CASE_STUDY_ONLY": "case-study only",
        "SUPPORTED_CONDITIONAL": "conditionally supported",
    }[status]


def render_macros(document: Mapping[str, Any], source_sha256: str) -> str:
    validated = validate_release(document)
    protocol = validated["protocol"]
    primary = validated["primary"]
    profiles = validated["profiles"]
    relation = validated["relation"]
    natural = validated["natural"]
    diagnostic_sensitivity = validated["diagnostic_sensitivity"]
    diagnostic_claim = validated["diagnostic_claim"]

    lines = [
        "% AUTO-GENERATED by source_data/generate_result_macros.py.",
        "% Do not edit by hand.",
        f"% paper_claims.json sha256: {source_sha256}",
        r"\LBResultsReadytrue",
        (
            r"\LBSimpleStructuredSupportedtrue"
            if primary["status"] == "SUPPORTED"
            else r"\LBSimpleStructuredSupportedfalse"
        ),
    ]

    def macro(name: str, value: str) -> None:
        lines.append(rf"\renewcommand{{\{name}}}{{{value}}}")

    macro("LBSourceSHA", source_sha256)
    macro("LBControlledTaskCount", str(protocol["dataset_count"]))
    macro("LBMechanismCount", str(protocol["mechanism_count"]))
    macro("LBStrengthCount", str(protocol["strength_count"]))
    macro("LBCoreModelCount", str(protocol["model_count"]))
    macro("LBSeedCount", str(protocol["seed_count"]))
    macro("LBExpectedCells", f"{int(protocol['expected_cells']):,}")
    macro("LBSuccessfulCells", f"{int(protocol['successful_cells']):,}")
    macro("LBCompletionRate", f"{100.0 * float(protocol['completion_rate']):.1f}")
    macro("LBDiagnosticMethodCount", str(int(diagnostic_sensitivity["method_count"])))
    macro("LBExpectedDiagnosticCells", f"{int(diagnostic_sensitivity['expected_cells']):,}")
    macro("LBSuccessfulDiagnosticCells", f"{int(diagnostic_sensitivity['successful_cells']):,}")
    macro(
        "LBDiagnosticCompletionRate",
        f"{100.0 * float(diagnostic_sensitivity['completion_rate']):.1f}",
    )
    macro("LBDiagnosticConditionalStatus", _status_text(diagnostic_claim["status"]))
    macro("LBSimpleStructuredStatus", _status_text(primary["status"]))
    primary_metrics = primary["metrics"]
    macro("LBSimpleStructuredDifference", _estimate(primary_metrics["difference"]))
    macro("LBSimpleStructuredCILow", _estimate(primary_metrics["ci_low"]))
    macro("LBSimpleStructuredCIHigh", _estimate(primary_metrics["ci_high"]))
    macro("LBSimpleStructuredHolmP", _p_value(primary_metrics["holm_p"]))
    macro("LBModelPositiveDirectionCount", str(int(primary_metrics["model_positive_direction_count"])))
    macro("LBModelCIExcludesZeroCount", str(int(primary_metrics["model_ci_excludes_zero_count"])))

    for claim_id, prefix in (
        ("m03_profile", "LBMThree"),
        ("m08_profile", "LBMEight"),
        ("m09_counterexample", "LBMNine"),
    ):
        claim = profiles[claim_id]
        metrics = claim["metrics"]
        macro(f"{prefix}Status", _status_text(claim["status"]))
        macro(f"{prefix}Detectability", _estimate(metrics["detectability"]))
        macro(f"{prefix}DetectabilityCILow", _estimate(metrics["detectability_ci_low"]))
        macro(f"{prefix}DetectabilityCIHigh", _estimate(metrics["detectability_ci_high"]))
        macro(f"{prefix}Harm", _estimate(metrics["paired_harm"]))
        macro(f"{prefix}HarmCILow", _estimate(metrics["paired_harm_ci_low"]))
        macro(f"{prefix}HarmCIHigh", _estimate(metrics["paired_harm_ci_high"]))
    macro("LBMEightClusterCILow", _estimate(profiles["m08_profile"]["metrics"]["cluster_ci_low"]))
    macro("LBMEightClusterCIHigh", _estimate(profiles["m08_profile"]["metrics"]["cluster_ci_high"]))
    macro(
        "LBMNineReweightingLow",
        _estimate(profiles["m09_counterexample"]["metrics"]["designed_category_reweighting_low"]),
    )
    macro(
        "LBMNineReweightingHigh",
        _estimate(profiles["m09_counterexample"]["metrics"]["designed_category_reweighting_high"]),
    )

    relation_metrics = relation["metrics"]
    macro("LBRelationStatus", _status_text(relation["status"]))
    macro("LBGlobalSpearman", _estimate(relation_metrics["global_spearman"]))
    macro("LBGlobalSpearmanCILow", _estimate(relation_metrics["global_spearman_ci_low"]))
    macro("LBGlobalSpearmanCIHigh", _estimate(relation_metrics["global_spearman_ci_high"]))
    macro("LBCategoryRSquared", _estimate(relation_metrics["category_r2"]))
    macro("LBCategoryPlusDRSquared", _estimate(relation_metrics["category_plus_detectability_r2"]))
    macro("LBIncrementalRSquared", _estimate(relation_metrics["incremental_r2"]))
    macro("LBIncrementalPermutationP", _p_value(relation_metrics["incremental_permutation_p"]))
    macro("LBCategoryLomoRSquared", _estimate(relation_metrics["category_lomo_r2"]))
    macro("LBCategoryPlusDLomoRSquared", _estimate(relation_metrics["category_plus_detectability_lomo_r2"]))
    macro("LBIncrementalLomoRSquared", _estimate(relation_metrics["incremental_lomo_r2"]))

    macro("LBNaturalStatus", _status_text(natural["status"]))
    macro("LBNaturalTaskCount", str(int(natural["task_count"])))
    macro("LBNaturalModelCount", str(int(natural["model_count"])))
    macro("LBNaturalAllPositive", "yes" if natural["all_task_effects_positive"] else "no")
    macro("LBNaturalMeanHarm", _estimate(natural["mean_paired_harm"]))
    macro("LBNaturalCILow", _estimate(natural["bootstrap_ci_low"]))
    macro("LBNaturalCIHigh", _estimate(natural["bootstrap_ci_high"]))
    macro("LBNaturalSignFlipP", _p_value(natural["exact_sign_flip_p"]))

    method_prefixes = {
        "mutual_information": "MI",
        "absolute_correlation": "Correlation",
        "lr_coefficient": "LRCoefficient",
        "rf_permutation": "RFPermutation",
    }
    mechanism_prefixes = {"M03": "LBMThree", "M04": "LBMFour", "M05": "LBMFive"}
    for mechanism in DIAGNOSTIC_MECHANISMS:
        points = []
        for method in DIAGNOSTIC_METHODS:
            row = diagnostic_sensitivity["mechanisms"][mechanism][method]
            prefix = f"{mechanism_prefixes[mechanism]}{method_prefixes[method]}"
            macro(f"{prefix}Detectability", _estimate(row["detectability"]))
            macro(f"{prefix}CILow", _estimate(row["ci_low"]))
            macro(f"{prefix}CIHigh", _estimate(row["ci_high"]))
            points.append(float(row["detectability"]))
        macro(f"{mechanism_prefixes[mechanism]}DiagnosticMin", _estimate(min(points)))
        macro(f"{mechanism_prefixes[mechanism]}DiagnosticMax", _estimate(max(points)))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate the final release without writing LaTeX",
    )
    args = parser.parse_args(argv)
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"final claim release not found: {source}")
    raw = source.read_bytes()
    try:
        document = json.loads(raw)
        rendered = render_macros(_mapping(document, "document"), hashlib.sha256(raw).hexdigest())
    except (json.JSONDecodeError, ClaimReleaseError) as exc:
        parser.error(str(exc))
    if args.check_only:
        print(f"VALID confirmatory claim release: {source}")
        return 0
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
