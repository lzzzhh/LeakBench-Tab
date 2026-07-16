#!/usr/bin/env python3
"""Fail-closed release validator for the corrected_v2 paper evidence package."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_corrected_v2_claim_state import (  # noqa: E402
    MAIN_CLAIM_IDS,
    build_document,
    collect_evidence,
    default_paths,
    load_json,
    relative,
    sha256,
)
from scripts.build_corrected_v2_artifact import (  # noqa: E402
    GPU_INTERIM_INCIDENT,
    PAPER_BUILD_MANIFEST,
    compute_pre_release_inventory,
)


def _status_values(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "status" and isinstance(item, str):
                values.append(item)
            values.extend(_status_values(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_status_values(item))
    return values


def validate_claim_policy(document: dict[str, Any]) -> None:
    if document.get("schema_version") != 1 or document.get("evidence_tier") != "confirmatory":
        raise ValueError("Paper claims do not have the confirmatory schema/evidence tier")
    integrity = document.get("protocol_integrity", {})
    required_integrity = {
        "dataset_count": 20, "mechanism_count": 11, "strength_count": 5,
        "model_count": 5, "seed_count": 5, "expected_cells": 27500,
        "successful_cells": 27500, "completion_rate": 1.0,
        "models": ["catboost", "lightgbm", "lr", "rf", "tabm"],
    }
    if integrity != required_integrity:
        raise ValueError("Paper claims protocol_integrity is not the exact frozen design")
    amendment = document.get("protocol_amendments", {}).get("M10", {})
    if (
        amendment.get("version") != "m10_strict_mask_v1"
        or amendment.get("replacement_cells") != 2500
        or amendment.get("strict_policy") != "task.X[:, ~task.leakage_mask]"
        or amendment.get("original_m10_rows_accepted") is not False
    ):
        raise ValueError("Paper claims do not bind the 2,500-cell M10 amendment")
    diagnostic_amendment = document.get("protocol_amendments", {}).get("diagnostic_rng", {})
    if diagnostic_amendment != {
        "version": "diagnostic_mi_fixed_seed_42_v1",
        "discovery_phase": "post_unblinding",
        "replacement_method": "mutual_information",
        "replacement_random_state": 42,
        "replacement_cells": 5500,
        "preserved_cells": 16500,
        "expected_canonical_cells": 22000,
        "no_tuning": True,
        "thresholds_changed": False,
        "raw_task_seeded_mi_accepted_as_final": False,
        "protocol_freeze_sha256": document.get("provenance", {}).get("input_sha256", {}).get(
            "results/corrected_v2/diagnostic_rng_amendment_freeze.json"
        ),
        "canonical_manifest_sha256": document.get("provenance", {}).get("input_sha256", {}).get(
            "results/corrected_v2/diagnostic_canonical_cells.manifest.json"
        ),
    }:
        raise ValueError("Paper claims do not bind the 5,500-row fixed-seed-42 diagnostic amendment")
    statistical_amendment = document.get("protocol_amendments", {}).get(
        "statistical_inference", {}
    )
    if statistical_amendment != {
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
        "protocol_freeze_sha256": document.get("provenance", {}).get(
            "input_sha256", {}
        ).get("results/corrected_v2/statistical_amendment_protocol_v2_freeze.json"),
    }:
        raise ValueError("Paper claims do not bind the statistical inference amendment")
    natural_amendment = document.get("protocol_amendments", {}).get(
        "natural_preprocessing", {}
    )
    if natural_amendment != {
        "version": "natural_trainfit_categories_v2",
        "fit_scope": "training rows only",
        "unseen_category_policy": "map_to_reserved_unknown_value",
        "globally_encoded_date_strings_accepted": False,
        "superseded_full_table_category_vocabulary_accepted": False,
        "replacement_cells": 60,
        "protocol_freeze_sha256": document.get("provenance", {}).get(
            "input_sha256", {}
        ).get(
            "results/corrected_v2/public_natural/natural_protocol_v2_freeze.json"
        ),
    }:
        raise ValueError("Paper claims do not bind natural train-fitted preprocessing v2")
    claims = document.get("claims", {})
    if set(claims) != MAIN_CLAIM_IDS:
        raise ValueError("Paper claims use an unknown or incomplete stable claim-ID set")
    if claims["simple_vs_structured"].get("status") not in {"SUPPORTED", "NOT_SUPPORTED"}:
        raise ValueError("Invalid status for simple_vs_structured")
    for claim_id in ("m03_profile", "m08_profile", "m09_counterexample"):
        if claims[claim_id].get("status") != "DESCRIPTIVE_ONLY":
            raise ValueError(f"{claim_id} exceeded its descriptive-only cap")
    if claims["detectability_exploitability_relation"].get("status") != "DESCRIPTIVE_ONLY":
        raise ValueError("Global D-X evidence exceeded its descriptive claim cap")
    if claims["D_METHOD_CONDITIONAL"].get("status") != "DESCRIPTIVE_ONLY":
        raise ValueError("Diagnostic-method comparison exceeded its descriptive-only cap")
    diagnostic_claim_metrics = claims["D_METHOD_CONDITIONAL"].get("metrics", {})
    if not {
        "m03_method_range", "conservative_ci_separation",
        "m03_best_evaluated_method", "m03_worst_evaluated_method",
    }.issubset(diagnostic_claim_metrics):
        raise ValueError("Diagnostic-method claim has an incomplete metric set")
    forbidden = {"PENDING", "REFUTED", "INTEGRITY_HOLD", "BLOCKED", "CONFIRMED"}
    observed_forbidden = forbidden.intersection(_status_values(claims))
    if observed_forbidden:
        raise ValueError(f"Main claims contain forbidden statuses: {sorted(observed_forbidden)}")

    simple_metrics = claims["simple_vs_structured"].get("metrics", {})
    for field in (
        "difference", "ci_low", "ci_high", "holm_p",
        "model_positive_direction_count", "model_ci_excludes_zero_count",
    ):
        if field not in simple_metrics:
            raise ValueError(f"simple_vs_structured is missing {field}")
    for field in ("model_positive_direction_count", "model_ci_excludes_zero_count"):
        value = simple_metrics[field]
        if not isinstance(value, int) or not 0 <= value <= 5:
            raise ValueError(f"simple_vs_structured {field} is outside 0..5")
    simple_criteria = claims["simple_vs_structured"].get("criteria", {})
    if simple_criteria != {
        "pooled_ci_low_gt": 0.0,
        "exact_holm_p_lte": 0.05,
        "model_results_used_for_support_decision": False,
    }:
        raise ValueError("simple_vs_structured support rule changed")
    expected_supported = bool(
        float(simple_metrics["ci_low"]) > 0.0
        and float(simple_metrics["holm_p"]) <= 0.05
    )
    if (claims["simple_vs_structured"]["status"] == "SUPPORTED") != expected_supported:
        raise ValueError("simple_vs_structured status does not apply the exact Holm+CI rule")

    required_profile_metrics = {
        "detectability", "detectability_ci_low", "detectability_ci_high",
        "paired_harm", "paired_harm_ci_low", "paired_harm_ci_high",
    }
    for claim_id in ("m03_profile", "m08_profile", "m09_counterexample"):
        missing = required_profile_metrics - set(claims[claim_id].get("metrics", {}))
        if missing:
            raise ValueError(f"{claim_id} is missing metrics {sorted(missing)}")
    m09 = claims["m09_counterexample"]
    m09_metrics = m09.get("metrics", {})
    if (
        m09_metrics.get("designed_category_reweighting_is_inferential") is not False
        or m09_metrics.get("detectability_unit") != "encoded_column"
        or m09_metrics.get("complete_one_hot") is not True
        or m09_metrics.get("encoded_column_count") != 8
        or m09_metrics.get("semantic_field_count") != 1
        or m09_metrics.get("representation_conditional") is not True
        or "cluster_ci_low" in m09_metrics
        or "cluster_ci_high" in m09_metrics
        or "cluster_ci_low_gt" in m09.get("criteria", {})
    ):
        raise ValueError("M09 descriptive designed-category reweighting was used inferentially")
    m08 = claims["m08_profile"]
    if (
        m08.get("criteria", {}).get("equivalence_or_practical_null_claim_allowed") is not False
        or "practically null" in str(m08.get("statement", "")).lower()
    ):
        raise ValueError("M08 descriptive interval was converted into an equivalence claim")
    claim_policy = document.get("claim_policy", {})
    if (
        claim_policy.get("only_thresholded_main_claim") != "simple_vs_structured"
        or claim_policy.get("simple_vs_structured_support_rule")
        != "exact_holm_p_lte_0.05_and_ci_low_gt_0"
        or claim_policy.get("profile_thresholds") is not None
        or claim_policy.get("mechanism_profile_status_cap") != "DESCRIPTIVE_ONLY"
        or claim_policy.get("diagnostic_method_status_cap") != "DESCRIPTIVE_ONLY"
        or claim_policy.get("model_specific_contrasts_status_cap") != "DESCRIPTIVE_ONLY"
    ):
        raise ValueError("Claim policy regained unbound profile/model thresholds")
    relation_fields = {
        "global_spearman", "global_spearman_ci_low", "global_spearman_ci_high",
        "category_r2", "category_plus_detectability_r2", "incremental_r2",
        "incremental_permutation_p", "category_lomo_r2",
        "category_plus_detectability_lomo_r2", "incremental_lomo_r2",
    }
    if relation_fields - set(claims["detectability_exploitability_relation"].get("metrics", {})):
        raise ValueError("D-X descriptive claim has an incomplete metric set")

    diagnostic = document.get("diagnostic_sensitivity", {})
    if (
        diagnostic.get("status") != "DESCRIPTIVE_ONLY"
        or diagnostic.get("method_count") != 4
        or diagnostic.get("expected_cells") != 22000
        or diagnostic.get("successful_cells") != 22000
        or diagnostic.get("completion_rate") != 1.0
        or diagnostic.get("primary_method") != "mutual_information"
        or diagnostic.get("methods") != [
            "mutual_information", "absolute_correlation", "lr_coefficient", "rf_permutation"
        ]
    ):
        raise ValueError("Diagnostic sensitivity summary is incomplete or over-claimed")
    diagnostic_mechanisms = diagnostic.get("mechanisms", {})
    if set(diagnostic_mechanisms) != {"M03", "M04", "M05"}:
        raise ValueError("Diagnostic sensitivity does not contain exactly M03/M04/M05")
    for mechanism, methods in diagnostic_mechanisms.items():
        if set(methods) != set(diagnostic["methods"]):
            raise ValueError(f"Diagnostic sensitivity is not 4-method complete for {mechanism}")
        for method, metrics in methods.items():
            if set(metrics) != {"detectability", "ci_low", "ci_high"}:
                raise ValueError(f"Diagnostic sensitivity fields changed for {mechanism}/{method}")
    warning = str(diagnostic.get("best_method_interpretation", "")).lower()
    if "non-deployable" not in warning or "not a zero-shot selector" not in warning:
        raise ValueError("Best-diagnostic summary lacks the mandatory deployment warning")

    natural = document.get("natural", {})
    if (
        natural.get("status") != "CASE_STUDY_ONLY"
        or "evidence_supported" in natural
        or natural.get("task_count") != 5
        or natural.get("model_count") != 4
        or "population-level" not in str(natural.get("interpretation", ""))
    ):
        raise ValueError("Natural evidence exceeded the fixed-case-study scope")
    pending = document.get("pending", {})
    if set(pending) != {"metadata", "governance"} or any(
        value.get("status") != "PENDING" for value in pending.values()
    ):
        raise ValueError("Metadata/governance pending state changed")


def validate_claim_files(
    evidence: dict[str, Any], paper_claims: Path, claim_state: Path,
) -> dict[str, Any]:
    if not paper_claims.is_file() or not claim_state.is_file():
        raise FileNotFoundError("Both paper_claims.json and claim_state.json are required")
    if paper_claims.read_bytes() != claim_state.read_bytes():
        raise ValueError("paper_claims.json and claim_state.json are not byte-identical")
    document = load_json(paper_claims)
    expected = build_document(evidence, generated_at=document.get("generated_at_utc"))
    if document != expected:
        raise ValueError("Claim document differs from a fresh derivation of current evidence")
    validate_claim_policy(document)
    superseded = evidence["superseded"]
    provenance_paths = set(document.get("provenance", {}).get("input_sha256", {}))
    forbidden_inputs = provenance_paths.intersection(superseded)
    if forbidden_inputs:
        raise ValueError(f"Claim provenance consumes superseded evidence: {sorted(forbidden_inputs)}")
    for name, expected_hash in document["provenance"]["input_sha256"].items():
        source = ROOT / name
        if not source.is_file() or sha256(source) != expected_hash:
            raise ValueError(f"Claim provenance hash mismatch: {name}")
    generator = document["provenance"]["generator"]
    generator_path = ROOT / generator["path"]
    if sha256(generator_path) != generator["sha256"]:
        raise ValueError("Claim generator changed after claim derivation")
    return document


def rebuild_and_validate_result_macros(paper_claims: Path) -> Path:
    output = ROOT / "paper/aaai27/generated/result_macros.tex"
    generator = ROOT / "paper/aaai27/source_data/generate_result_macros.py"
    completed = subprocess.run(
        [
            sys.executable, str(generator),
            "--input", str(paper_claims),
            "--output", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-8:])
        raise RuntimeError(f"Result-macro regeneration failed:\n{tail}")
    claims_hash = sha256(paper_claims)
    expected_marker = f"% paper_claims.json sha256: {claims_hash}"
    if not output.is_file() or expected_marker not in output.read_text(encoding="utf-8"):
        raise ValueError("Generated result macros are not bound to the current paper claims")
    return output


def validate_generated_figures(paper_claims: Path) -> dict[str, Any]:
    output = ROOT / "paper/aaai27/figures/generated"
    manifest_path = output / "figure_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("evidence_tier") != "confirmatory":
        raise ValueError("Figure manifest is not confirmatory")
    if manifest.get("pilot_inputs_forbidden") is not True:
        raise ValueError("Figure manifest does not explicitly forbid pilot inputs")
    generator_name = manifest.get("generator")
    generator = ROOT / str(generator_name)
    if generator_name != "scripts/generate_corrected_v2_figures.py" or not generator.is_file():
        raise ValueError("Figure generator identity changed")
    if sha256(generator) != manifest.get("generator_sha256"):
        raise ValueError("Figure generator changed after rendering")
    expected_figures = {
        "paper/aaai27/figures/generated/cdx_scatter.pdf",
        "paper/aaai27/figures/generated/mechanism_model_heatmap.pdf",
        "paper/aaai27/figures/generated/strength_diagnostic_robustness.pdf",
    }
    figure_hashes = manifest.get("figure_sha256", {})
    if set(figure_hashes) != expected_figures:
        raise ValueError("Figure manifest does not contain exactly the three final figures")
    for name, expected_hash in figure_hashes.items():
        path = ROOT / name
        if not path.is_file() or path.stat().st_size < 1024 or not path.read_bytes().startswith(b"%PDF"):
            raise ValueError(f"Generated figure is missing or invalid: {name}")
        if sha256(path) != expected_hash:
            raise ValueError(f"Generated figure hash mismatch: {name}")
    sources = manifest.get("source_sha256", {})
    required_sources = {
        relative(paper_claims),
        "results/corrected_v2/canonical_manifest.json",
        "results/corrected_v2/statistics/integrity_summary.json",
        "results/corrected_v2/statistics/diagnostic_integrity.json",
        "results/corrected_v2/statistics/secondary_integrity.json",
        "results/corrected_v2/statistics/mechanism_summary.csv",
        "results/corrected_v2/statistics/detectability_mechanism_summary.csv",
        "results/corrected_v2/statistics/mechanism_model_summary.csv",
        "results/corrected_v2/statistics/strength_dose_response.csv",
        "results/corrected_v2/statistics/diagnostic_method_by_mechanism.csv",
    }
    if not required_sources.issubset(sources):
        raise ValueError(f"Figure manifest omits final inputs: {sorted(required_sources - set(sources))}")
    for name, expected_hash in sources.items():
        if "pilot" in name.lower():
            raise ValueError(f"Pilot input leaked into a submission figure: {name}")
        path = ROOT / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"Figure source hash mismatch: {name}")
    return manifest


def _run_recomputation(command: list[str], label: str) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f"{label} recomputation failed:\n{tail}")


def _require_byte_identical(
    released_dir: Path,
    rebuilt_dir: Path,
    names: tuple[str, ...],
    label: str,
) -> None:
    for name in names:
        released = released_dir / name
        rebuilt = rebuilt_dir / name
        if not released.is_file() or not rebuilt.is_file():
            raise FileNotFoundError(f"{label} comparison lacks {name}")
        if released.read_bytes() != rebuilt.read_bytes():
            raise ValueError(
                f"Released {label} is not byte-identical to recomputation: {relative(released)}"
            )


def validate_amended_statistics_recomputation(evidence: dict[str, Any]) -> None:
    """Deep-rerun every paper-facing statistic and require byte identity."""
    paths = evidence["paths"]
    statistics = evidence["statistics"]
    released_dir = paths.statistics_dir
    scratch_root = ROOT / "results/corrected_v2"
    with tempfile.TemporaryDirectory(prefix=".release-stat-recompute-", dir=scratch_root) as temporary:
        temporary_path = Path(temporary)

        primary_dir = temporary_path / "primary"
        _run_recomputation([
            sys.executable,
            str(ROOT / "scripts/analyze_corrected_v2.py"),
            "--core", str(paths.canonical),
            "--config", str(paths.config),
            "--output-dir", str(primary_dir),
            "--namespace", "confirmatory",
            "--bootstrap-reps", "20000",
            "--require-complete",
        ], "primary mechanism statistics")
        _require_byte_identical(
            released_dir, primary_dir,
            (
                "mechanism_summary.csv", "category_summary.csv", "model_summary.csv",
                "integrity_summary.json",
            ),
            "primary mechanism statistic",
        )

        detectability_dir = temporary_path / "detectability"
        _run_recomputation([
            sys.executable,
            str(ROOT / "scripts/analyze_detectability_v2.py"),
            "--core", str(paths.canonical),
            "--namespace", "confirmatory",
            "--repetitions", "20000",
            "--seed", "20260713",
            "--output-dir", str(detectability_dir),
        ], "detectability statistics")
        _require_byte_identical(
            released_dir, detectability_dir,
            ("detectability_mechanism_summary.csv", "detectability_category_summary.csv"),
            "detectability statistic",
        )

        model_dir = temporary_path / "model"
        _run_recomputation([
            sys.executable,
            str(ROOT / "scripts/analyze_model_contrasts_v2.py"),
            "--core", str(paths.canonical),
            "--namespace", "confirmatory",
            "--repetitions", "20000",
            "--seed", "20260713",
            "--output-dir", str(model_dir),
        ], "model contrast statistics")
        _require_byte_identical(
            released_dir, model_dir,
            ("simple_structured_by_model.csv", "model_vs_lr_contrasts.csv"),
            "model contrast statistic",
        )

        secondary_dir = temporary_path / "secondary"
        _run_recomputation([
            sys.executable,
            str(ROOT / "scripts/analyze_secondary_v2.py"),
            "--core", str(paths.canonical),
            "--config", str(paths.config),
            "--namespace", "confirmatory",
            "--output-dir", str(secondary_dir),
            "--repetitions", "20000",
            "--seed", "20260713",
            "--require-complete",
        ], "secondary statistics")
        _require_byte_identical(
            released_dir, secondary_dir,
            (
                "mechanism_model_summary.csv", "mechanism_model_dispersion.csv",
                "strength_dose_response.csv", "secondary_integrity.json",
            ),
            "secondary statistic",
        )

        diagnostic_dir = temporary_path / "diagnostic"
        _run_recomputation([
            sys.executable,
            str(ROOT / "scripts/analyze_diagnostic_suite.py"),
            "--input", str(paths.diagnostic_cells),
            "--namespace", "confirmatory",
            "--output-dir", str(diagnostic_dir),
            "--repetitions", "20000",
            "--seed", "20260713",
            "--require-complete",
        ], "diagnostic statistics")
        _require_byte_identical(
            released_dir, diagnostic_dir,
            (
                "diagnostic_method_by_mechanism.csv", "diagnostic_method_summary.csv",
                "diagnostic_robustness_profiles.csv", "diagnostic_integrity.json",
            ),
            "diagnostic statistic",
        )

        amended_dir = temporary_path / "amended"
        _run_recomputation([
            sys.executable,
            str(ROOT / "scripts/analyze_corrected_v2_amendment.py"),
            "--canonical", str(paths.canonical),
            "--config", str(paths.config),
            "--output-dir", str(amended_dir),
            "--namespace", "confirmatory",
            "--bootstrap-reps", "20000",
            "--permutation-reps", "20000",
            "--seed", "20260713",
        ], "statistical amendment")
        for released, rebuilt in (
            (
                statistics["paths"]["category_contrasts"],
                amended_dir / "category_contrasts_amended.csv",
            ),
            (
                statistics["paths"]["correlation"],
                amended_dir / "correlation_analysis_amended.json",
            ),
        ):
            if released.read_bytes() != rebuilt.read_bytes():
                raise ValueError(
                    f"Released amended statistic is not byte-identical to recomputation: {relative(released)}"
                )

        cluster_manifest = statistics["cluster_manifest"]
        prediction_directories = cluster_manifest.get("prediction_directories", [])
        if not isinstance(prediction_directories, list) or not prediction_directories:
            raise ValueError("Cluster manifest lacks prediction directories for recomputation")
        rebuilt_cluster = temporary_path / "cluster_sensitivity_v3.json"
        rebuilt_cluster_manifest = temporary_path / "cluster_sensitivity_v3_manifest.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/analyze_cluster_sensitivity_amendment_v2.py"),
            "--canonical", str(paths.canonical),
            "--config", str(paths.config),
            "--task-manifest", str(
                ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
            ),
            "--prediction-dirs",
            *[str(ROOT / directory) for directory in prediction_directories],
            "--namespace", "confirmatory",
            "--inner-reps", "200",
            "--outer-reps", "5000",
            "--seed", "20260713",
            "--output", str(rebuilt_cluster),
            "--manifest", str(rebuilt_cluster_manifest),
        ]
        _run_recomputation(command, "synchronized cluster sensitivity")
        released_cluster = statistics["paths"]["cluster"]
        if released_cluster.read_bytes() != rebuilt_cluster.read_bytes():
            raise ValueError(
                "Released synchronized cluster result is not byte-identical to recomputation"
            )


def validate_natural_statistics_recomputation(evidence: dict[str, Any]) -> None:
    """Recompute the two natural-case quantities rendered in the paper."""
    cells = pd.read_csv(evidence["paths"].natural_cells)
    released = evidence["natural"]
    effects = cells.groupby("task")["paired_harm"].mean().to_numpy(dtype=float)
    if len(effects) != 5:
        raise ValueError("Natural recomputation does not contain exactly five fixed tasks")
    rng = np.random.RandomState(20260713)
    bootstrap = np.empty(20_000, dtype=float)
    for repetition in range(len(bootstrap)):
        bootstrap[repetition] = rng.choice(
            effects, size=len(effects), replace=True
        ).mean()
    expected_interval = np.quantile(bootstrap, [0.025, 0.975])
    if not np.allclose(
        released.get("task_bootstrap_ci", []), expected_interval,
        atol=1e-15, rtol=0,
    ):
        raise ValueError("Released natural bootstrap CI differs from fixed-seed recomputation")
    signs = np.array(
        np.meshgrid(*[[-1.0, 1.0]] * len(effects))
    ).T.reshape(-1, len(effects))
    observed = abs(effects.mean())
    expected_p = float(
        np.mean(np.abs((signs * effects).mean(axis=1)) >= observed - 1e-15)
    )
    if not np.isclose(
        float(released.get("exact_two_sided_sign_flip_p", float("nan"))),
        expected_p, atol=1e-15, rtol=0,
    ):
        raise ValueError("Released natural exact sign-flip p differs from 2^5 enumeration")


def run_tests() -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    tail = "\n".join((completed.stdout or completed.stderr).strip().splitlines()[-3:])
    if completed.returncode != 0:
        raise RuntimeError(f"Test suite failed:\n{tail}")
    if not re.search(r"\b[1-9][0-9]* passed\b", completed.stdout):
        raise RuntimeError(f"Test suite did not execute any passing tests:\n{tail}")
    return {"command": "python -m pytest tests -q", "status": "PASS", "tail": tail}


def rebuild_and_validate_result_tables(paper_claims: Path) -> Path:
    generator = ROOT / "paper/aaai27/source_data/generate_result_tables.py"
    completed = subprocess.run(
        [sys.executable, str(generator)], cwd=ROOT, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or completed.stdout).splitlines()[-8:])
        raise RuntimeError(f"Final result-table generation failed:\n{tail}")
    manifest_path = ROOT / "paper/aaai27/generated/result_tables_manifest.json"
    manifest = load_json(manifest_path)
    expected_outputs = {
        f"paper/aaai27/generated/table_{name}.tex"
        for name in (
            "task_registry", "mechanism_profiles", "mechanism_models",
            "diagnostic_methods", "strength_response", "natural_cases", "claim_scope",
        )
    }
    wrapper = "paper/aaai27/generated/result_tables.tex"
    expected_outputs.add(wrapper)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PASS"
        or manifest.get("evidence_tier") != "confirmatory"
        or manifest.get("pilot_inputs_forbidden") is not True
        or manifest.get("table_count") != 7
        or manifest.get("generator") != relative(generator)
        or manifest.get("generator_sha256") != sha256(generator)
        or manifest.get("paper_claims_sha256") != sha256(paper_claims)
        or manifest.get("wrapper") != wrapper
        or set(manifest.get("table_sha256", {})) != expected_outputs
    ):
        raise ValueError("Final result-table manifest identity or coverage changed")
    for mapping_name in ("source_sha256", "table_sha256"):
        mapping = manifest.get(mapping_name, {})
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Result-table manifest lacks {mapping_name}")
        for name, digest in mapping.items():
            path = ROOT / name
            if (
                "pilot" in name.lower()
                or not path.is_file()
                or sha256(path) != digest
            ):
                raise ValueError(f"Result-table hash/source policy failed: {name}")
    return manifest_path


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-claims", default="results/corrected_v2/paper_claims.json")
    parser.add_argument("--claim-state", default="results/corrected_v2/claim_state.json")
    parser.add_argument("--report", default="results/corrected_v2/release_validation.json")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    report_path = ROOT / args.report
    report: dict[str, Any] = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED",
        "checks": [],
    }
    try:
        evidence = collect_evidence(default_paths())
        report["checks"].append({"name": "corrected_evidence_chain", "status": "PASS"})
        document = validate_claim_files(
            evidence, ROOT / args.paper_claims, ROOT / args.claim_state,
        )
        report["checks"].append({"name": "derived_claim_state", "status": "PASS"})
        result_macros = rebuild_and_validate_result_macros(ROOT / args.paper_claims)
        report["checks"].append({
            "name": "paper_claims_bound_result_macros", "status": "PASS"
        })
        result_tables = rebuild_and_validate_result_tables(ROOT / args.paper_claims)
        report["checks"].append({
            "name": "final_claim_scoped_result_tables", "status": "PASS"
        })
        validate_amended_statistics_recomputation(evidence)
        report["checks"].append({
            "name": "all_statistics_deep_recomputation", "status": "PASS"
        })
        validate_natural_statistics_recomputation(evidence)
        report["checks"].append({
            "name": "natural_statistics_deep_recomputation", "status": "PASS"
        })
        validate_generated_figures(ROOT / args.paper_claims)
        report["checks"].append({"name": "final_only_generated_figures", "status": "PASS"})
        if not args.skip_tests:
            report["tests"] = run_tests()
            report["checks"].append({"name": "tests", "status": "PASS"})
        inventory, _, paper_build, _ = compute_pre_release_inventory()
        report["checks"].append({
            "name": "gpu_interim_access_incident_locked", "status": "PASS"
        })
        report["checks"].append({
            "name": "submission_pdf_build_manifest", "status": "PASS"
        })
        report["checks"].append({
            "name": "fresh_full_artifact_input_inventory", "status": "PASS"
        })
        report.update({
            "status": "PASS",
            "tests_skipped": args.skip_tests,
            "release_state": "RELEASE_VALIDATED",
            "paper_claims_sha256": sha256(ROOT / args.paper_claims),
            "result_macros_path": relative(result_macros),
            "result_macros_sha256": sha256(result_macros),
            "result_tables_manifest_path": relative(result_tables),
            "result_tables_manifest_sha256": sha256(result_tables),
            "paper_build_manifest_path": str(PAPER_BUILD_MANIFEST),
            "paper_build_manifest_sha256": sha256(ROOT / str(PAPER_BUILD_MANIFEST)),
            "paper_build_schema_version": paper_build["schema_version"],
            "gpu_interim_access_incident_path": str(GPU_INTERIM_INCIDENT),
            "gpu_interim_access_incident_sha256": sha256(
                ROOT / str(GPU_INTERIM_INCIDENT)
            ),
            "artifact_input_inventory": {
                "policy_version": inventory["policy_version"],
                "sha256": inventory["sha256"],
                "file_count": inventory["file_count"],
                "total_bytes": inventory["total_bytes"],
            },
            "canonical_sha256": document["provenance"]["input_sha256"]["results/corrected_v2/canonical_cells.csv"],
            "evidence_tier": "confirmatory",
        })
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        write_report(report_path, report)
        print(json.dumps(report, indent=2))
        return 1
    write_report(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
