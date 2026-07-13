#!/usr/bin/env python3
"""Build the only paper-facing claim state from corrected_v2 evidence.

The default path is intentionally fail-closed: it accepts only the complete
confirmatory matrix, frozen protocols, complete diagnostic sensitivity suite,
and fixed natural case studies.  Legacy/superseded evidence is read only as a
deny-list and can never supply a claim value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
CORRECTED_ROOT = ROOT / "results/corrected_v2"
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}
DIAGNOSTIC_METHODS = [
    "mutual_information", "absolute_correlation", "lr_coefficient", "rf_permutation",
]
MAIN_CLAIM_IDS = {
    "simple_vs_structured",
    "m03_profile",
    "m08_profile",
    "m09_counterexample",
    "detectability_exploitability_relation",
    "D_METHOD_CONDITIONAL",
}
ALPHA = 0.05
M10_FREEZE_SHA256 = "5442c9aff8f2329c90845d904fe02c9639365aac6f0f30c5e387fe12a4e1c4ce"


@dataclass(frozen=True)
class EvidencePaths:
    canonical: Path
    canonical_manifest: Path
    statistics_dir: Path
    diagnostic_cells: Path
    diagnostic_canonical_manifest: Path
    diagnostic_amendment_freeze: Path
    diagnostic_statistics_dir: Path
    natural_statistics: Path
    natural_cells: Path
    natural_tasks: Path
    natural_freeze: Path
    natural_public_manifest: Path
    config: Path
    protocol_freeze: Path
    tabm_bundle_freeze: Path
    diagnostic_freeze: Path
    statistical_amendment_freeze: Path
    superseded: Path
    local_environment: Path
    tabm_environment: Path
    requirements: Path
    m10_config: Path
    m10_freeze: Path
    m10_cpu: Path
    m10_cpu_manifest: Path
    m10_tabm: Path
    m10_tabm_manifest: Path


def default_paths(
    canonical: str = "results/corrected_v2/canonical_cells.csv",
    canonical_manifest: str = "results/corrected_v2/canonical_manifest.json",
    statistics_dir: str = "results/corrected_v2/statistics",
    diagnostic_cells: str = "results/corrected_v2/diagnostic_canonical_cells.csv",
    diagnostic_canonical_manifest: str = "results/corrected_v2/diagnostic_canonical_cells.manifest.json",
    diagnostic_amendment_freeze: str = "results/corrected_v2/diagnostic_rng_amendment_freeze.json",
    diagnostic_statistics_dir: str = "results/corrected_v2/statistics",
    natural_statistics: str = "results/corrected_v2/public_natural/natural_statistics.json",
    natural_cells: str = "results/corrected_v2/public_natural/natural_cells.csv",
    natural_tasks: str = "results/corrected_v2/public_natural/natural_task_summary.csv",
    natural_freeze: str = "results/corrected_v2/public_natural/natural_protocol_v2_freeze.json",
    natural_public_manifest: str = "results/corrected_v2/public_natural/public_natural_provenance_manifest.json",
    config: str = "configs/paper/corrected_v2.yaml",
    protocol_freeze: str = "results/corrected_v2/protocol_freeze.json",
    tabm_bundle_freeze: str = "results/corrected_v2/tabm_bundle_protocol_freeze.json",
    diagnostic_freeze: str = "results/corrected_v2/diagnostic_protocol_freeze.json",
    statistical_amendment_freeze: str = "results/corrected_v2/statistical_amendment_protocol_v2_freeze.json",
    superseded: str = "results/corrected_v2/superseded_evidence.json",
    local_environment: str = "results/corrected_v2/local_environment_lock.json",
    tabm_environment: str = "results/corrected_v2/tabm_official_environment_lock.json",
    requirements: str = "requirements-corrected-v2.txt",
    m10_config: str = "configs/paper/m10_amendment_v1.yaml",
    m10_freeze: str = "results/corrected_v2/m10_amendment_protocol_freeze.json",
    m10_cpu: str = "results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv",
    m10_cpu_manifest: str = "results/corrected_v2/m10_amendment_confirmatory/cpu_cells_manifest.json",
    m10_tabm: str = "results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv",
    m10_tabm_manifest: str = "results/corrected_v2/m10_amendment_confirmatory/tabm_cells_manifest.json",
) -> EvidencePaths:
    def path(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else ROOT / candidate

    return EvidencePaths(
        canonical=path(canonical),
        canonical_manifest=path(canonical_manifest),
        statistics_dir=path(statistics_dir),
        diagnostic_cells=path(diagnostic_cells),
        diagnostic_canonical_manifest=path(diagnostic_canonical_manifest),
        diagnostic_amendment_freeze=path(diagnostic_amendment_freeze),
        diagnostic_statistics_dir=path(diagnostic_statistics_dir),
        natural_statistics=path(natural_statistics),
        natural_cells=path(natural_cells),
        natural_tasks=path(natural_tasks),
        natural_freeze=path(natural_freeze),
        natural_public_manifest=path(natural_public_manifest),
        config=path(config),
        protocol_freeze=path(protocol_freeze),
        tabm_bundle_freeze=path(tabm_bundle_freeze),
        diagnostic_freeze=path(diagnostic_freeze),
        statistical_amendment_freeze=path(statistical_amendment_freeze),
        superseded=path(superseded),
        local_environment=path(local_environment),
        tabm_environment=path(tabm_environment),
        requirements=path(requirements),
        m10_config=path(m10_config),
        m10_freeze=path(m10_freeze),
        m10_cpu=path(m10_cpu),
        m10_cpu_manifest=path(m10_cpu_manifest),
        m10_tabm=path(m10_tabm),
        m10_tabm_manifest=path(m10_tabm_manifest),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"Evidence path is outside the repository: {path}") from error


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{relative(source)} is missing columns: {missing}")


def require_finite(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{relative(source)} contains non-finite values in {columns}")


def normalize_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Not a strict boolean: {value!r}")


def validate_freeze(path: Path, expected_status: str, files_key: str) -> dict[str, Any]:
    freeze = load_json(path)
    if freeze.get("status") != expected_status:
        raise ValueError(f"Wrong freeze status in {relative(path)}: {freeze.get('status')!r}")
    entries = freeze.get(files_key)
    if files_key == "files":
        iterator = ((entry.get("path"), entry) for entry in entries or [])
    else:
        iterator = (entries or {}).items()
    observed = 0
    for item_path, entry in iterator:
        observed += 1
        frozen_path = ROOT / str(item_path)
        if not frozen_path.is_file() or sha256(frozen_path) != entry.get("sha256"):
            raise ValueError(f"Frozen hash mismatch: {item_path}")
    if observed == 0:
        raise ValueError(f"Freeze has no file entries: {relative(path)}")
    return freeze


def load_superseded(path: Path) -> set[str]:
    payload = load_json(path)
    if payload.get("status") != "INTEGRITY_HOLD" or payload.get("rule") is None:
        raise ValueError("Superseded-evidence deny-list is not active")
    whole_file: set[str] = set()
    selector_scoped: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    selector_entry_count = 0
    for item in payload.get("superseded", []):
        name = str(Path(item["path"]))
        selector = item.get("selector")
        if selector is None:
            whole_file.add(name)
            continue
        selector_entry_count += 1
        if not isinstance(selector, dict) or not selector or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in selector.items()
        ):
            raise ValueError(f"Invalid selector-scoped supersession: {name}")
        selector_scoped.add((name, tuple(sorted(selector.items()))))
    expected_selector_scoped = {
        (
            "results/corrected_v2/core_cpu_cells.csv",
            (("mechanism", "M10"),),
        ),
        (
            "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv",
            (("mechanism", "M10"),),
        ),
        (
            "results/corrected_v2/diagnostic_confirmatory_cells.csv",
            (("method", "mutual_information"),),
        ),
    }
    if selector_entry_count != 3 or selector_scoped != expected_selector_scoped:
        raise ValueError("Selector-scoped supersession policy changed")
    return whole_file


def reject_superseded(paths: list[Path], superseded: set[str]) -> None:
    for path in paths:
        name = relative(path)
        if name in superseded:
            raise ValueError(f"Superseded evidence is forbidden: {name}")
        if path.exists():
            try:
                path.resolve().relative_to(CORRECTED_ROOT.resolve())
            except ValueError:
                if not name.startswith(("configs/", "scripts/", "experiments/")):
                    raise ValueError(f"Evidence resolved outside corrected_v2: {name}")


def statistics_files(directory: Path) -> dict[str, Path]:
    return {
        "mechanism": directory / "mechanism_summary.csv",
        "detectability": directory / "detectability_mechanism_summary.csv",
        "category_contrasts": directory / "category_contrasts_amended.csv",
        "by_model": directory / "simple_structured_by_model.csv",
        "correlation": directory / "correlation_analysis_amended.json",
        "cluster": directory / "cluster_sensitivity_v3.json",
        "statistical_amendment_manifest": directory / "statistical_amendment_manifest.json",
        "cluster_manifest": directory / "cluster_sensitivity_v3_manifest.json",
        "integrity": directory / "integrity_summary.json",
    }


def diagnostic_statistics_files(directory: Path) -> dict[str, Path]:
    return {
        "by_mechanism": directory / "diagnostic_method_by_mechanism.csv",
        "method_summary": directory / "diagnostic_method_summary.csv",
        "profiles": directory / "diagnostic_robustness_profiles.csv",
        "integrity": directory / "diagnostic_integrity.json",
    }


def validate_statistics_schema(directory: Path) -> dict[str, Any]:
    paths = statistics_files(directory)
    mechanism_frame = pd.read_csv(paths["mechanism"])
    require_columns(mechanism_frame, {
        "mechanism", "category", "paired_harm", "paired_harm_ci_low",
        "paired_harm_ci_high", "diagnostic_normalized_ap", "sign_flip_p", "holm_p",
    }, paths["mechanism"])
    detectability = pd.read_csv(paths["detectability"])
    require_columns(detectability, {
        "mechanism", "category", "diagnostic_normalized_ap",
        "diagnostic_normalized_ap_ci_low", "diagnostic_normalized_ap_ci_high",
        "top5_recall", "top5_recall_ci_low", "top5_recall_ci_high",
    }, paths["detectability"])
    contrasts = pd.read_csv(paths["category_contrasts"])
    require_columns(contrasts, {
        "contrast", "difference", "ci_low", "ci_high", "sign_flip_p", "holm_p",
        "n_tasks", "test_method", "multiplicity_family",
    }, paths["category_contrasts"])
    by_model = pd.read_csv(paths["by_model"])
    require_columns(by_model, {
        "model", "simple_minus_structured", "ci_low", "ci_high", "sign_flip_p", "holm_p",
    }, paths["by_model"])
    correlation = load_json(paths["correlation"])
    correlation_fields = {
        "global_spearman", "global_spearman_ci", "global_pearson", "global_pearson_ci",
        "category_r2", "category_plus_detectability_r2", "incremental_r2",
        "incremental_permutation_p", "category_lomo_r2",
        "category_plus_detectability_lomo_r2", "incremental_lomo_r2",
    }
    missing = sorted(correlation_fields - set(correlation))
    if missing:
        raise ValueError(f"Correlation analysis is missing fields: {missing}")
    if (
        correlation.get("schema_version") != 2
        or correlation.get("analysis_version") != "joint_paired_dx_bootstrap_v1"
        or correlation.get("status") != "DESCRIPTIVE_ONLY"
        or correlation.get("bootstrap_method")
        != "joint_paired_dataset_then_seed_resampling_of_D_and_X"
        or correlation.get("bootstrap_includes_detectability_uncertainty") is not True
        or correlation.get("bootstrap_includes_exploitation_uncertainty") is not True
    ):
        raise ValueError("Correlation analysis is not the joint paired D--X amendment")
    cluster_payload = load_json(paths["cluster"])
    if (
        cluster_payload.get("schema_version") != 3
        or cluster_payload.get("analysis_version")
        != "synchronized_cluster_sensitivity_amendment_v2"
        or cluster_payload.get("claim_scope") != "DESCRIPTIVE_ONLY"
        or cluster_payload.get("prediction_bundle_fields_verified") != [
            "test_idx", "y", "entity_ids", "source_ids", "task_hash",
        ]
        or cluster_payload.get("prediction_metrics_verified")
        != ["clean_auc", "full_auc", "paired_harm"]
        or set(cluster_payload.get("analyses", {})) != {"M08", "M09"}
    ):
        raise ValueError("Cluster sensitivity is not the synchronized v2 amendment")
    cluster = cluster_payload["analyses"]
    if (
        cluster["M08"].get("status")
        != "DESCRIPTIVE_SYNCHRONIZED_CLUSTER_INTERVAL"
        or cluster["M08"].get("inferential_practical_null_claim_allowed") is not False
        or cluster["M08"].get("synchronized_cluster_ci") is None
        or cluster["M09"].get("status") != "DESCRIPTIVE_DESIGNED_CATEGORY_REWEIGHTING"
        or cluster["M09"].get("inferential_source_population_claim_allowed") is not False
        or cluster["M09"].get("descriptive_reweighting_interval") is None
    ):
        raise ValueError("Cluster amendment status/interval semantics changed")
    integrity = load_json(paths["integrity"])
    return {
        "paths": paths,
        "mechanism": mechanism_frame,
        "detectability": detectability,
        "category_contrasts": contrasts,
        "by_model": by_model,
        "correlation": correlation,
        "cluster": cluster,
        "cluster_payload": cluster_payload,
        "statistical_amendment_manifest": load_json(paths["statistical_amendment_manifest"]),
        "cluster_manifest": load_json(paths["cluster_manifest"]),
        "integrity": integrity,
    }


def validate_diagnostic_statistics_schema(directory: Path) -> dict[str, Any]:
    paths = diagnostic_statistics_files(directory)
    by_mechanism = pd.read_csv(paths["by_mechanism"])
    require_columns(by_mechanism, {
        "method", "mechanism", "category", "diagnostic_normalized_ap", "ci_low", "ci_high",
    }, paths["by_mechanism"])
    method_summary = pd.read_csv(paths["method_summary"])
    require_columns(method_summary, {"method", "diagnostic_normalized_ap", "ci_low", "ci_high"}, paths["method_summary"])
    profiles = pd.read_csv(paths["profiles"])
    require_columns(profiles, {
        "mechanism", "category", "best_evaluated_diagnostic", "best_ci_low", "best_ci_high",
        "worst_evaluated_diagnostic", "worst_ci_low", "worst_ci_high",
        "low_across_all_evaluated_diagnostics", "between_diagnostic_range",
    }, paths["profiles"])
    return {
        "paths": paths,
        "by_mechanism": by_mechanism,
        "method_summary": method_summary,
        "profiles": profiles,
        "integrity": load_json(paths["integrity"]),
    }


def _validate_exact_mechanisms(frame: pd.DataFrame, source: Path) -> None:
    if set(frame["mechanism"].astype(str)) != set(CATEGORIES) or frame["mechanism"].duplicated().any():
        raise ValueError(f"{relative(source)} does not contain exactly one row per mechanism")
    observed = dict(zip(frame["mechanism"].astype(str), frame["category"].astype(str)))
    if observed != CATEGORIES:
        raise ValueError(f"Mechanism categories changed in {relative(source)}")


def _assert_close(actual: Any, expected: Any, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Point estimate mismatch for {label}: {actual} != {expected}")


def _exact_task_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) != 20 or not np.isfinite(values).all():
        raise ValueError("Category sign-flip validation requires exactly 20 finite task effects")
    observed = abs(float(values.mean()))
    total = 1 << len(values)
    extreme = 0
    bits = np.arange(len(values), dtype=np.uint64)
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, observed)
    for start in range(0, total, 65_536):
        integers = np.arange(start, min(start + 65_536, total), dtype=np.uint64)
        signs = 2.0 * ((integers[:, None] >> bits) & 1).astype(float) - 1.0
        null = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(null >= observed - tolerance))
    return float(extreme / total)


def validate_m10_amendment(paths: EvidencePaths, config: dict[str, Any]) -> dict[str, Any]:
    if sha256(paths.m10_freeze) != M10_FREEZE_SHA256:
        raise ValueError("M10 amendment freeze differs from the reviewed confirmatory freeze")
    freeze = validate_freeze(
        paths.m10_freeze, "FROZEN_BEFORE_M10_AMENDMENT_CONFIRMATORY_RUN", "frozen_files"
    )
    required_freeze = {
        "expected_replacement_cells": 2500,
        "expected_cpu_cells": 2000,
        "expected_tabm_cells": 500,
    }
    for field, expected in required_freeze.items():
        if freeze.get(field) != expected:
            raise ValueError(f"M10 amendment freeze mismatch for {field}")
    if (
        freeze.get("amendment_version") != "m10_strict_mask_v1"
        or freeze.get("strict_policy") != "task.X[:, ~task.leakage_mask]"
        or freeze.get("full_policy") != "task.X"
        or freeze.get("expected_confirmatory_tasks") != 500
        or freeze.get("verified_confirmatory_tasks") != 500
        or freeze.get("base_config_sha256") != sha256(paths.config)
        or freeze.get("amendment_config_sha256") != sha256(paths.m10_config)
    ):
        raise ValueError("M10 amendment freeze identity/task verification is incomplete")
    if freeze.get("outputs") != {
        "cpu": relative(paths.m10_cpu), "tabm": relative(paths.m10_tabm)
    }:
        raise ValueError("M10 amendment freeze output paths changed")
    if freeze.get("output_manifests") != {
        "cpu": relative(paths.m10_cpu_manifest), "tabm": relative(paths.m10_tabm_manifest)
    }:
        raise ValueError("M10 amendment freeze output-manifest paths changed")
    task_manifest_path = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
    source_task = freeze.get("source_task_manifest", {})
    if source_task != {
        "path": relative(task_manifest_path), "sha256": sha256(task_manifest_path)
    }:
        raise ValueError("M10 amendment freeze does not bind the full frozen task manifest")
    bundle_summary_path = task_manifest_path.parent / "bundle_summary.json"
    if freeze.get("source_bundle_summary") != {
        "path": relative(bundle_summary_path), "sha256": sha256(bundle_summary_path)
    }:
        raise ValueError("M10 amendment freeze does not bind the frozen bundle summary")

    amendment_config = yaml.safe_load(paths.m10_config.read_text(encoding="utf-8"))
    amendment = amendment_config.get("amendment", {})
    if (
        amendment.get("version") != "m10_strict_mask_v1"
        or amendment.get("mechanism") != "M10"
        or amendment.get("strict_policy") != "task.X[:, ~task.leakage_mask]"
        or amendment.get("full_policy") != "task.X"
        or amendment.get("base_config_sha256") != sha256(paths.config)
        or amendment.get("confirmatory_task_manifest_sha256") != sha256(task_manifest_path)
        or amendment.get("expected_replacement_cells") != 2500
    ):
        raise ValueError("M10 amendment configuration is not the frozen mask-derived policy")

    frames = []
    expected_by_output = (
        (paths.m10_cpu, paths.m10_cpu_manifest, 2000, {"lr", "rf", "catboost", "lightgbm"}),
        (paths.m10_tabm, paths.m10_tabm_manifest, 500, {"tabm"}),
    )
    required_columns = {
        "run_id", "dataset_id", "dataset_namespace", "mechanism", "strength", "model", "seed",
        "status", "clean_auc", "strict_auc", "full_auc", "paired_harm", "n_original", "n_injected",
        "n_leak", "task_hash", "source_task_hash", "task_manifest_sha256", "bundle_summary_sha256",
        "split_hash", "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall",
        "bundle_path", "bundle_sha256", "integrity_verified", "amendment_version", "strict_policy",
        "full_policy", "strict_feature_count", "legitimate_injected_count",
        "contamination_removed_count", "strict_view_hash", "full_view_hash", "leakage_mask_hash",
        "config_hash", "amendment_config_hash", "runner_sha256", "model_adapter_sha256", "code_hash",
    }
    runner_path = ROOT / "experiments/leakbench/run_m10_amendment.py"
    for output, manifest_path, expected_rows, expected_models in expected_by_output:
        frame = pd.read_csv(output)
        require_columns(frame, required_columns, output)
        if len(frame) != expected_rows or frame["run_id"].astype(str).duplicated().any():
            raise ValueError(f"M10 amendment output coverage mismatch: {relative(output)}")
        key = ["dataset_id", "mechanism", "strength", "model", "seed"]
        if frame.duplicated(key).any():
            raise ValueError(f"Duplicate M10 scientific cells: {relative(output)}")
        if (
            set(frame["dataset_namespace"].astype(str)) != {"confirmatory"}
            or set(frame["mechanism"].astype(str)) != {"M10"}
            or set(frame["model"].astype(str)) != expected_models
            or not (frame["status"].astype(str) == "SUCCESS").all()
            or not all(normalize_bool(value) for value in frame["integrity_verified"])
        ):
            raise ValueError(f"Invalid M10 amendment cells: {relative(output)}")
        require_finite(frame, ["clean_auc", "strict_auc", "full_auc", "paired_harm"], output)
        if not np.allclose(frame["clean_auc"], frame["strict_auc"], atol=1e-12, rtol=0):
            raise ValueError("M10 clean_auc is not the amended strict_auc")
        if not np.allclose(frame["full_auc"] - frame["strict_auc"], frame["paired_harm"], atol=1e-12, rtol=0):
            raise ValueError("M10 paired harm is not full minus amended strict AUROC")
        constant_expectations = {
            "amendment_version": "m10_strict_mask_v1",
            "strict_policy": "task.X[:, ~task.leakage_mask]",
            "full_policy": "task.X",
            "legitimate_injected_count": 1,
            "contamination_removed_count": 1,
            "n_injected": 2,
            "n_leak": 1,
            "task_manifest_sha256": sha256(task_manifest_path),
            "bundle_summary_sha256": sha256(bundle_summary_path),
            "config_hash": sha256(paths.config),
            "amendment_config_hash": sha256(paths.m10_config),
            "runner_sha256": sha256(runner_path),
        }
        for field, expected in constant_expectations.items():
            if set(frame[field].astype(str)) != {str(expected)}:
                raise ValueError(f"M10 amendment field mismatch: {field}")
        expected_adapter = sha256(
            ROOT / (
                "src/leakbench/models/official_tabm.py"
                if expected_models == {"tabm"}
                else "src/leakbench/models/core_models.py"
            )
        )
        expected_code = hashlib.sha256(
            (expected_adapter + sha256(runner_path)).encode()
        ).hexdigest()
        if set(frame["model_adapter_sha256"].astype(str)) != {expected_adapter}:
            raise ValueError(f"M10 amendment adapter hash mismatch: {relative(output)}")
        if set(frame["code_hash"].astype(str)) != {expected_code}:
            raise ValueError(f"M10 amendment combined code hash mismatch: {relative(output)}")
        if not (frame["strict_feature_count"].astype(int) == frame["n_original"].astype(int) + 1).all():
            raise ValueError("M10 strict view did not retain exactly one legitimate injected feature")
        if not (frame["task_hash"].astype(str) == frame["source_task_hash"].astype(str)).all():
            raise ValueError("M10 amendment changed task identity")
        for hash_field in (
            "strict_view_hash", "full_view_hash", "leakage_mask_hash", "task_hash",
            "bundle_sha256", "model_adapter_sha256", "code_hash",
        ):
            if not frame[hash_field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
                raise ValueError(f"M10 amendment has an invalid hash field: {hash_field}")

        task_key = ["dataset_id", "mechanism", "strength", "seed"]
        task_fields = task_key + [
            "task_hash", "split_hash", "diagnostic_ap", "diagnostic_normalized_ap",
            "top5_recall", "n_leak", "bundle_path", "bundle_sha256",
        ]
        frozen_tasks = pd.read_csv(task_manifest_path)
        frozen_m10 = frozen_tasks.loc[
            frozen_tasks["mechanism"].astype(str) == "M10", task_fields
        ]
        unique_tasks = frame.drop_duplicates(task_key)
        checked = unique_tasks.merge(
            frozen_m10, on=task_key, suffixes=("", "_task"), validate="one_to_one"
        )
        if len(checked) != 500:
            raise ValueError(f"M10 amendment does not cover all 500 frozen tasks: {relative(output)}")
        for field in ("task_hash", "split_hash", "bundle_path", "bundle_sha256"):
            if not (checked[field].astype(str) == checked[f"{field}_task"].astype(str)).all():
                raise ValueError(f"M10 amendment {field} differs from frozen tasks")
        if not (checked["source_task_hash"].astype(str) == checked["task_hash_task"].astype(str)).all():
            raise ValueError("M10 source_task_hash differs from frozen tasks")
        for field in ("diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "n_leak"):
            if not np.allclose(checked[field], checked[f"{field}_task"], atol=1e-12, rtol=0):
                raise ValueError(f"M10 amendment {field} differs from frozen tasks")

        manifest = load_json(manifest_path)
        expected_manifest = {
            "status": "CONFIRMATORY", "amendment_version": "m10_strict_mask_v1",
            "strict_policy": "task.X[:, ~task.leakage_mask]", "full_policy": "task.X",
            "dataset_namespace": "confirmatory", "requested_cells": expected_rows,
            "success_cells": expected_rows, "failure_cells": 0,
            "integrity_verified_cells": expected_rows, "mechanism": "M10",
            "result_sha256": sha256(output),
        }
        for field, expected in expected_manifest.items():
            if manifest.get(field) != expected:
                raise ValueError(f"M10 output manifest mismatch for {field}: {relative(manifest_path)}")
        if set(manifest.get("models", [])) != expected_models:
            raise ValueError(f"M10 output manifest model set changed: {relative(manifest_path)}")
        if (
            manifest.get("task_manifest_sha256") != sha256(task_manifest_path)
            or manifest.get("bundle_summary_sha256") != sha256(bundle_summary_path)
            or manifest.get("amendment_config_sha256") != sha256(paths.m10_config)
            or manifest.get("runner_sha256") != sha256(runner_path)
        ):
            raise ValueError(f"M10 output manifest provenance changed: {relative(manifest_path)}")
        frames.append(frame)
    replacement = pd.concat(frames, ignore_index=True, sort=False)
    if len(replacement) != 2500 or replacement.duplicated(
        ["dataset_id", "mechanism", "strength", "model", "seed"]
    ).any():
        raise ValueError("M10 replacement union is not exactly 2,500 unique cells")
    return {"freeze": freeze, "replacement": replacement}


def validate_canonical(
    paths: EvidencePaths, config: dict[str, Any], m10_amendment: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    canonical = pd.read_csv(paths.canonical)
    manifest = load_json(paths.canonical_manifest)
    protocol = config["protocol"]
    required = {
        "run_id", "dataset_id", "dataset_namespace", "mechanism", "strength", "model", "seed",
        "status", "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "n_leak",
        "clean_auc", "full_auc", "paired_harm", "implementation", "split_hash", "config_hash",
        "evidence_tier", "task_source", "amendment_version",
    }
    require_columns(canonical, required, paths.canonical)
    expected_cells = int(protocol["expected_model_training_cells"])
    expected_models = sorted(protocol["core_models"])
    expected_seeds = sorted(int(seed) for seed in protocol["seeds"])
    expected_mechanisms = list(protocol["mechanisms"])
    expected_strengths = list(protocol["strengths"])
    key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    if len(canonical) != expected_cells or canonical.duplicated(key).any():
        raise ValueError("Canonical table is not the exact frozen 27,500-cell matrix")
    if set(canonical["dataset_namespace"].astype(str)) != {"confirmatory"}:
        raise ValueError("Canonical evidence is not confirmatory")
    if not (canonical["status"].astype(str) == "SUCCESS").all():
        raise ValueError("Canonical evidence contains a failed or invalid cell")
    if canonical["run_id"].astype(str).duplicated().any():
        raise ValueError("Canonical run_id values are not unique")
    if canonical["dataset_id"].nunique() != int(protocol["dataset_count"]):
        raise ValueError("Canonical dataset count differs from the frozen protocol")
    if sorted(canonical["model"].astype(str).unique()) != expected_models:
        raise ValueError("Canonical model set differs from the frozen protocol")
    if sorted(int(value) for value in canonical["seed"].unique()) != expected_seeds:
        raise ValueError("Canonical seed set differs from the frozen protocol")
    if set(canonical["mechanism"].astype(str)) != set(expected_mechanisms):
        raise ValueError("Canonical mechanism set differs from the frozen protocol")
    if set(canonical["strength"].astype(str)) != set(expected_strengths):
        raise ValueError("Canonical strength set differs from the frozen protocol")
    counts = canonical.groupby("model").size()
    if not (counts == expected_cells // len(expected_models)).all():
        raise ValueError("Canonical model coverage is unbalanced")
    require_finite(canonical, [
        "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "n_leak",
        "clean_auc", "full_auc", "paired_harm",
    ], paths.canonical)
    if set(canonical["config_hash"].astype(str)) != {sha256(paths.config)}:
        raise ValueError("Canonical cells use a config other than the frozen corrected_v2 config")
    diagnostic_range = canonical.groupby(
        ["dataset_id", "mechanism", "strength", "seed"]
    )["diagnostic_normalized_ap"].agg(lambda values: float(values.max() - values.min()))
    if float(diagnostic_range.max()) > 1e-10:
        raise ValueError("Primary diagnostic differs across downstream models")
    split_count = canonical.groupby(
        ["dataset_id", "mechanism", "strength", "seed"]
    )["split_hash"].nunique()
    if int(split_count.max()) != 1:
        raise ValueError("Split hashes differ across models for the same task")
    tabm = canonical[canonical["model"] == "tabm"]
    if "integrity_verified" not in tabm.columns or not all(normalize_bool(value) for value in tabm["integrity_verified"]):
        raise ValueError("Official TabM cells are not all integrity verified")
    if not tabm["implementation"].astype(str).str.contains("tabm.TabM", regex=False).all():
        raise ValueError("Official TabM model identity is absent")

    if manifest.get("status") != "CANONICAL" or manifest.get("cells") != expected_cells:
        raise ValueError("Canonical manifest is incomplete")
    canonical_builder = ROOT / "scripts/build_canonical_corrected_v2.py"
    if manifest.get("builder") != {
        "path": relative(canonical_builder), "sha256": sha256(canonical_builder),
    }:
        raise ValueError("Canonical manifest is not bound to the current canonical builder")
    if manifest.get("successful_cells") != expected_cells or manifest.get("canonical_sha256") != sha256(paths.canonical):
        raise ValueError("Canonical manifest hash/success count mismatch")
    if manifest.get("config_sha256") != sha256(paths.config):
        raise ValueError("Canonical manifest is bound to the wrong config")
    source_paths = {
        "cpu": ROOT / "results/corrected_v2/core_cpu_cells.csv",
        "tabm": ROOT / "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv",
        "tasks": ROOT / "results/corrected_v2/task_bundles/task_manifest.csv",
        "m10_cpu": paths.m10_cpu,
        "m10_tabm": paths.m10_tabm,
    }
    if set(manifest.get("source_sha256", {})) != set(source_paths):
        raise ValueError("Canonical manifest source set changed")
    for name, source in source_paths.items():
        if not source.is_file() or manifest["source_sha256"][name] != sha256(source):
            raise ValueError(f"Canonical source hash mismatch: {name}")
    source_manifest_paths = {
        "cpu": ROOT / "results/corrected_v2/core_cpu_cells_manifest.json",
        "tabm": ROOT / "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_manifest.json",
        "m10_cpu": paths.m10_cpu_manifest,
        "m10_tabm": paths.m10_tabm_manifest,
    }
    if manifest.get("source_manifest_sha256") != {
        name: sha256(source) for name, source in source_manifest_paths.items()
    }:
        raise ValueError("Canonical raw source-manifest hashes changed")
    amendment_manifest = manifest.get("m10_amendment", {})
    if amendment_manifest != {
        "version": "m10_strict_mask_v1",
        "replacement_cells": 2500,
        "cpu_cells": 2000,
        "tabm_cells": 500,
        "protocol_freeze_sha256": sha256(paths.m10_freeze),
    }:
        raise ValueError("Canonical manifest does not record the exact M10 amendment")
    runner_hash = sha256(ROOT / "experiments/leakbench/run_m10_amendment.py")
    core_adapter_hash = sha256(ROOT / "src/leakbench/models/core_models.py")
    tabm_adapter_hash = sha256(ROOT / "src/leakbench/models/official_tabm.py")
    expected_validated_code = {
        "cpu": hashlib.sha256("".join(
            sha256(ROOT / name) for name in (
                "src/leakbench/datasets.py", "src/leakbench/mechanisms/__init__.py",
                "src/leakbench/models/core_models.py", "experiments/leakbench/run_corrected_core.py",
            )
        ).encode()).hexdigest(),
        "tabm": hashlib.sha256("".join(
            sha256(ROOT / name) for name in (
                "src/leakbench/models/official_tabm.py",
                "experiments/leakbench/run_corrected_tabm_bundle.py",
            )
        ).encode()).hexdigest(),
        "m10_cpu": hashlib.sha256((core_adapter_hash + runner_hash).encode()).hexdigest(),
        "m10_tabm": hashlib.sha256((tabm_adapter_hash + runner_hash).encode()).hexdigest(),
    }
    if manifest.get("validated_code_sha256") != expected_validated_code:
        raise ValueError("Canonical manifest validated-code hashes changed")

    replacement = m10_amendment["replacement"].copy()
    canonical_m10 = canonical[canonical["mechanism"] == "M10"].copy()
    if len(canonical_m10) != 2500 or set(canonical_m10["run_id"].astype(str)) != set(replacement["run_id"].astype(str)):
        raise ValueError("Canonical M10 rows are not exactly the 2,500 amended run IDs")
    key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    comparison = canonical_m10.merge(
        replacement, on=key, suffixes=("_canonical", "_replacement"), validate="one_to_one"
    )
    string_fields = [
        "run_id", "amendment_version", "strict_policy", "full_policy", "task_hash",
        "source_task_hash", "strict_view_hash", "full_view_hash", "leakage_mask_hash",
        "amendment_config_hash", "runner_sha256", "model_adapter_sha256", "code_hash",
    ]
    numeric_fields = [
        "clean_auc", "strict_auc", "full_auc", "paired_harm", "strict_feature_count",
        "legitimate_injected_count", "contamination_removed_count",
    ]
    for field in string_fields:
        if not (
            comparison[f"{field}_canonical"].astype(str)
            == comparison[f"{field}_replacement"].astype(str)
        ).all():
            raise ValueError(f"Canonical M10 differs from amendment output: {field}")
    for field in numeric_fields:
        if not np.allclose(
            comparison[f"{field}_canonical"], comparison[f"{field}_replacement"], atol=1e-12, rtol=0,
        ):
            raise ValueError(f"Canonical M10 differs from amendment output: {field}")
    if "amendment_version" not in canonical.columns:
        raise ValueError("Canonical table lacks the M10 amendment origin marker")
    non_m10_markers = canonical.loc[canonical["mechanism"] != "M10", "amendment_version"]
    if non_m10_markers.notna().any():
        raise ValueError("M10 amendment origin marker leaked onto non-M10 cells")
    if set(canonical_m10["task_source"].astype(str)) != {"frozen_local_bundle_m10_amendment"}:
        raise ValueError("Canonical M10 rows lack the amendment task-source marker")
    if set(canonical.loc[canonical["mechanism"] != "M10", "task_source"].astype(str)) != {
        "frozen_local_bundle_base_protocol"
    }:
        raise ValueError("Canonical base rows use an unexpected task-source marker")
    if set(canonical["evidence_tier"].astype(str)) != {"confirmatory"}:
        raise ValueError("Canonical rows lack the confirmatory evidence tier")
    return canonical, manifest


def validate_confirmatory_statistics(
    bundle: dict[str, Any], canonical: pd.DataFrame, config: dict[str, Any], canonical_path: Path,
) -> None:
    protocol = config["protocol"]
    expected_models = sorted(protocol["core_models"])
    mechanism = bundle["mechanism"].copy()
    detectability = bundle["detectability"].copy()
    contrasts = bundle["category_contrasts"].copy()
    by_model = bundle["by_model"].copy()
    _validate_exact_mechanisms(mechanism, bundle["paths"]["mechanism"])
    _validate_exact_mechanisms(detectability, bundle["paths"]["detectability"])
    require_finite(mechanism, [
        "paired_harm", "paired_harm_ci_low", "paired_harm_ci_high",
        "diagnostic_normalized_ap", "sign_flip_p", "holm_p",
    ], bundle["paths"]["mechanism"])
    require_finite(detectability, [
        "diagnostic_normalized_ap", "diagnostic_normalized_ap_ci_low",
        "diagnostic_normalized_ap_ci_high", "top5_recall", "top5_recall_ci_low", "top5_recall_ci_high",
    ], bundle["paths"]["detectability"])
    if set(contrasts["contrast"].astype(str)) != {
        "simple_minus_structured", "simple_minus_boundary", "boundary_minus_structured",
    } or contrasts["contrast"].duplicated().any():
        raise ValueError("Category contrast family is incomplete")
    require_finite(
        contrasts, ["difference", "ci_low", "ci_high", "sign_flip_p", "holm_p", "n_tasks"],
        bundle["paths"]["category_contrasts"],
    )
    if (
        set(contrasts["n_tasks"].astype(int)) != {20}
        or set(contrasts["test_method"].astype(str))
        != {"exact_two_sided_task_level_sign_flip"}
        or set(contrasts["multiplicity_family"].astype(str))
        != {"three_declared_category_contrasts_holm"}
    ):
        raise ValueError("Category contrasts do not use the amended exact task-level test")
    if sorted(by_model["model"].astype(str)) != expected_models or by_model["model"].duplicated().any():
        raise ValueError("Model-level simple/structured contrasts are incomplete")
    require_finite(by_model, ["simple_minus_structured", "ci_low", "ci_high", "sign_flip_p", "holm_p"], bundle["paths"]["by_model"])

    integrity = bundle["integrity"]
    expected = int(protocol["expected_model_training_cells"])
    required_integrity = {
        "namespace": "confirmatory", "rows_total": expected, "rows_success": expected,
        "rows_failed_or_invalid": 0, "expected_cells_for_observed_dataset_count": expected,
        "datasets": int(protocol["dataset_count"]), "mechanisms": len(protocol["mechanisms"]),
        "strengths": len(protocol["strengths"]), "bootstrap_repetitions": int(config["statistics"]["bootstrap_repetitions"]),
        "bootstrap_seed": int(config["statistics"]["bootstrap_seed"]),
    }
    for field, expected_value in required_integrity.items():
        if integrity.get(field) != expected_value:
            raise ValueError(f"Statistical integrity mismatch for {field}: {integrity.get(field)!r}")
    if not math.isclose(float(integrity.get("completion_rate", -1)), 1.0, abs_tol=1e-12):
        raise ValueError("Statistical completion rate is not 1.0")
    if sorted(integrity.get("models", [])) != expected_models:
        raise ValueError("Statistical integrity model set changed")
    if sorted(int(seed) for seed in integrity.get("seeds", [])) != sorted(int(seed) for seed in protocol["seeds"]):
        raise ValueError("Statistical integrity seed set changed")
    if integrity.get("source_tables") != [relative(canonical_path)]:
        raise ValueError("Statistics were not derived solely from canonical confirmatory cells")

    canonical_mechanism = canonical.groupby("mechanism")["paired_harm"].mean()
    canonical_detectability = canonical.groupby("mechanism")["diagnostic_normalized_ap"].mean()
    mechanism_index = mechanism.set_index("mechanism")
    detectability_index = detectability.set_index("mechanism")
    for name in CATEGORIES:
        _assert_close(mechanism_index.loc[name, "paired_harm"], canonical_mechanism[name], f"{name} paired harm")
        _assert_close(detectability_index.loc[name, "diagnostic_normalized_ap"], canonical_detectability[name], f"{name} detectability")

    category_atomic = canonical.assign(
        category=canonical["mechanism"].map(CATEGORIES)
    ).groupby(
        ["dataset_id", "seed", "category"], as_index=False, observed=True
    )["paired_harm"].mean().pivot(
        index=["dataset_id", "seed"], columns="category", values="paired_harm"
    )
    contrast_pairs = (
        ("simple_minus_structured", "simple", "structured"),
        ("simple_minus_boundary", "simple", "boundary"),
        ("boundary_minus_structured", "boundary", "structured"),
    )
    observed_contrasts = contrasts.set_index("contrast")
    expected_p_values = []
    for contrast_name, left, right in contrast_pairs:
        seed_effects = category_atomic[left] - category_atomic[right]
        task_effects = seed_effects.groupby(level="dataset_id").mean().sort_index()
        _assert_close(
            observed_contrasts.loc[contrast_name, "difference"], seed_effects.mean(),
            contrast_name,
        )
        expected_p = _exact_task_sign_flip_p(task_effects.to_numpy())
        expected_p_values.append(expected_p)
        _assert_close(
            observed_contrasts.loc[contrast_name, "sign_flip_p"], expected_p,
            f"{contrast_name} exact sign-flip p", tolerance=1e-15,
        )
    expected_holm = multipletests(expected_p_values, method="holm")[1]
    for index, (contrast_name, _, _) in enumerate(contrast_pairs):
        _assert_close(
            observed_contrasts.loc[contrast_name, "holm_p"], expected_holm[index],
            f"{contrast_name} Holm p", tolerance=1e-15,
        )

    atomic = canonical.groupby(
        ["model", "dataset_id", "seed", "mechanism"], as_index=False
    )["paired_harm"].mean()
    atomic["category"] = atomic["mechanism"].map(CATEGORIES)
    model_effect = atomic.groupby(
        ["model", "dataset_id", "seed", "category"], as_index=False
    )["paired_harm"].mean().pivot(
        index=["model", "dataset_id", "seed"], columns="category", values="paired_harm"
    )
    model_effect["effect"] = model_effect["simple"] - model_effect["structured"]
    expected_by_model = model_effect.groupby(level="model")["effect"].mean()
    observed_by_model = by_model.set_index("model")["simple_minus_structured"]
    for model in expected_models:
        _assert_close(observed_by_model[model], expected_by_model[model], f"{model} simple minus structured")

    correlation = bundle["correlation"]
    detection = mechanism_index.loc[list(CATEGORIES), "diagnostic_normalized_ap"]
    harm = mechanism_index.loc[list(CATEGORIES), "paired_harm"]
    _assert_close(correlation["global_spearman"], detection.corr(harm, method="spearman"), "global Spearman")
    _assert_close(correlation["global_pearson"], detection.corr(harm, method="pearson"), "global Pearson")
    correlation_numeric = {
        "global_spearman", "global_spearman_ci", "global_pearson", "global_pearson_ci",
        "excluding_simple_spearman", "within_structured_spearman", "category_r2",
        "category_plus_detectability_r2", "incremental_r2", "incremental_permutation_p",
        "category_lomo_r2", "category_plus_detectability_lomo_r2", "incremental_lomo_r2",
    }
    for field in correlation_numeric:
        value = correlation[field]
        values = value if isinstance(value, list) else [value]
        if not all(math.isfinite(float(item)) for item in values):
            raise ValueError(f"Correlation field is non-finite: {field}")
    if (
        correlation.get("bootstrap_repetitions") != 20000
        or correlation.get("bootstrap_seed") != 20260713
        or correlation.get("datasets") != 20
        or correlation.get("seeds") != 5
        or correlation.get("mechanisms") != 11
    ):
        raise ValueError("Joint D--X bootstrap coverage/parameters changed")

    m08 = bundle["cluster"]["M08"]
    if (
        m08.get("cluster_unit") != "entity_id"
        or m08.get("grouping_key") != ["dataset_id"]
        or m08.get("shared_draw_scope") != ["seed", "model", "strength"]
        or m08.get("shared_cells_per_inner_draw") != 125
        or m08.get("task_groups") != 20
        or m08.get("seed_effects_preserved_within_each_entity_draw") is not True
        or m08.get("inferential_practical_null_claim_allowed") is not False
    ):
        raise ValueError("M08 synchronized entity-draw design changed")
    m09 = bundle["cluster"]["M09"]
    if (
        m09.get("cluster_unit") != "source_id"
        or m09.get("grouping_key") != ["dataset_id", "seed", "strength"]
        or m09.get("shared_draw_scope") != ["model"]
        or m09.get("shared_cells_per_inner_draw") != 5
        or m09.get("task_strength_groups") != 500
        or m09.get("designed_category_count") != 8
        or m09.get("inferential_source_population_claim_allowed") is not False
    ):
        raise ValueError("M09 descriptive designed-category reweighting semantics changed")
    for name, entry, interval_field in (
        ("M08", m08, "synchronized_cluster_ci"),
        ("M09", m09, "descriptive_reweighting_interval"),
    ):
        if (
            entry.get("datasets") != 20
            or entry.get("seeds") != 5
            or entry.get("strengths") != 5
            or entry.get("models") != 5
            or entry.get("cells") != 2500
            or entry.get("inner_repetitions_per_task_group", 0) < 200
            or entry.get("outer_repetitions", 0) < 5000
        ):
            raise ValueError(f"Cluster sensitivity is underpowered/incomplete for {name}")
        values = entry.get(interval_field, [])
        if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Invalid cluster/reweighting interval for {name}")
        _assert_close(entry["paired_harm"], canonical_mechanism[name], f"{name} cluster point estimate")


def validate_statistical_amendment_chain(
    paths: EvidencePaths,
    bundle: dict[str, Any],
    canonical: pd.DataFrame,
) -> dict[str, Any]:
    freeze = validate_freeze(
        paths.statistical_amendment_freeze,
        "FROZEN_BEFORE_FINAL_STATISTICAL_AMENDMENT_V2_ANALYSIS",
        "frozen_files",
    )
    if (
        freeze.get("schema_version") != 2
        or freeze.get("amendment_id") != "statistical_inference_amendment_v2"
        or freeze.get("supersedes")
        != "results/corrected_v2/statistical_amendment_protocol_freeze.json"
        or freeze.get("discovery_phase")
        != "second_post_unblinding_methodological_audit"
        or freeze.get("decision_thresholds") is not None
        or freeze.get("threshold_based_profile_claims_allowed") is not False
        or freeze.get("expected_final_cells") != 27500
        or freeze.get("expected_cluster_prediction_cells") != 5000
    ):
        raise ValueError("Statistical amendment freeze identity/scope changed")
    expected_parameters = {
        "bootstrap_repetitions": 20000,
        "permutation_repetitions": 20000,
        "cluster_inner_repetitions_per_dataset": 200,
        "cluster_outer_repetitions": 5000,
        "seed": 20260713,
    }
    if freeze.get("parameters") != expected_parameters:
        raise ValueError("Statistical amendment repetition counts or seed changed")
    expected_outputs = {
        "category_contrasts": relative(bundle["paths"]["category_contrasts"]),
        "correlation": relative(bundle["paths"]["correlation"]),
        "statistics_manifest": relative(bundle["paths"]["statistical_amendment_manifest"]),
        "cluster": relative(bundle["paths"]["cluster"]),
        "cluster_manifest": relative(bundle["paths"]["cluster_manifest"]),
    }
    if freeze.get("outputs") != expected_outputs:
        raise ValueError("Statistical amendment output paths differ from the freeze")
    if freeze.get("cluster_sensitivity", {}).get("M08", {}).get(
        "inferential_practical_null_claim_allowed"
    ) is not False:
        raise ValueError("M08 cluster interval regained a practical-null inference")
    if freeze.get("cluster_sensitivity", {}).get("M09", {}).get(
        "inferential_source_population_claim_allowed"
    ) is not False:
        raise ValueError("M09 source reweighting lost its descriptive-only scope")
    expected_claim_policy = {
        "simple_vs_structured": (
            "only directional main claim; exact Holm p <= 0.05 and confidence interval low > 0"
        ),
        "M03": "DESCRIPTIVE_ONLY",
        "M08": "DESCRIPTIVE_ONLY",
        "M09": "DESCRIPTIVE_ONLY",
        "detectability_exploitability_relation": "DESCRIPTIVE_ONLY",
        "diagnostic_method_comparison": "DESCRIPTIVE_ONLY",
        "model_specific_contrasts": "DESCRIPTIVE_ONLY",
    }
    if freeze.get("claim_policy") != expected_claim_policy:
        raise ValueError("Statistical amendment descriptive claim policy changed")
    if freeze.get("prediction_lineage") != {
        "frozen_task_count": 1000,
        "frozen_bundle_count": 20,
        "prediction_count": 5000,
        "prediction_arrays_compared_directly": [
            "row_id_to_test_idx", "y", "entity_id_to_entity_ids",
            "source_id_to_source_ids",
        ],
        "frozen_task_hash_reconstructed_from_bundle": True,
        "prediction_metrics_recomputed": ["clean_auc", "full_auc", "paired_harm"],
    }:
        raise ValueError("Statistical amendment prediction lineage policy changed")

    statistics_manifest = bundle["statistical_amendment_manifest"]
    if (
        statistics_manifest.get("status") != "AMENDED_STATISTICS_COMPLETE"
        or statistics_manifest.get("analysis_version")
        != "corrected_v2_statistical_amendment_v1"
        or statistics_manifest.get("bootstrap_repetitions") != 20000
        or statistics_manifest.get("permutation_repetitions") != 20000
        or statistics_manifest.get("seed") != 20260713
    ):
        raise ValueError("Statistical amendment manifest identity/parameters changed")
    analysis_code = statistics_manifest.get("analysis_code", {})
    amended_script = ROOT / "scripts/analyze_corrected_v2_amendment.py"
    if analysis_code != {"path": relative(amended_script), "sha256": sha256(amended_script)}:
        raise ValueError("Statistical amendment manifest does not bind its analysis code")
    if statistics_manifest.get("canonical_inputs") != [
        {"path": relative(paths.canonical), "sha256": sha256(paths.canonical)}
    ]:
        raise ValueError("Statistical amendment manifest does not bind the canonical table")
    if statistics_manifest.get("config") != {
        "path": relative(paths.config), "sha256": sha256(paths.config)
    }:
        raise ValueError("Statistical amendment manifest does not bind the configuration")
    expected_stat_outputs = {}
    for key in ("category_contrasts", "correlation"):
        output_path = bundle["paths"][key]
        expected_stat_outputs[relative(output_path)] = {
            "sha256": sha256(output_path), "size_bytes": output_path.stat().st_size,
        }
    if statistics_manifest.get("outputs") != expected_stat_outputs:
        raise ValueError("Statistical amendment output hashes changed")
    if statistics_manifest.get("category_contrasts") != {
        "independent_unit": "dataset_task",
        "test": "exact_two_sided_task_level_sign_flip",
        "task_count": 20,
        "multiplicity": "holm_over_three_declared_contrasts",
        "ci": "dataset_then_seed_hierarchical_percentile_bootstrap",
    }:
        raise ValueError("Category amendment manifest semantics changed")
    if statistics_manifest.get("correlation") != {
        "status": "DESCRIPTIVE_ONLY",
        "ci": "joint_paired_dataset_then_seed_resampling_of_D_and_X",
    }:
        raise ValueError("D--X amendment manifest semantics changed")

    cluster_manifest = bundle["cluster_manifest"]
    if (
        cluster_manifest.get("status") != "SYNCHRONIZED_CLUSTER_ANALYSIS_COMPLETE"
        or cluster_manifest.get("schema_version") != 2
        or cluster_manifest.get("analysis_version")
        != "synchronized_cluster_sensitivity_amendment_v2"
        or cluster_manifest.get("consumed_prediction_count") != 5000
        or cluster_manifest.get("parameters") != {
            "inner_repetitions_per_task_group": 200,
            "outer_repetitions": 5000,
            "seed": 20260713,
        }
    ):
        raise ValueError("Synchronized cluster manifest identity/parameters changed")
    cluster_script = ROOT / "scripts/analyze_cluster_sensitivity_amendment_v2.py"
    cluster_dependency = ROOT / "scripts/analyze_cluster_sensitivity_v2.py"
    if cluster_manifest.get("analysis_code") != {
        "path": relative(cluster_script), "sha256": sha256(cluster_script)
    }:
        raise ValueError("Cluster manifest does not bind the synchronized analysis code")
    if cluster_manifest.get("dependency_code") != {
        "path": relative(cluster_dependency), "sha256": sha256(cluster_dependency)
    }:
        raise ValueError("Cluster manifest does not bind its frozen helper dependency")
    if cluster_manifest.get("canonical_inputs") != [
        {"path": relative(paths.canonical), "sha256": sha256(paths.canonical)}
    ] or cluster_manifest.get("config") != {
        "path": relative(paths.config), "sha256": sha256(paths.config)
    }:
        raise ValueError("Cluster manifest does not bind canonical/config inputs")
    cluster_output = bundle["paths"]["cluster"]
    if cluster_manifest.get("output") != {
        "path": relative(cluster_output),
        "sha256": sha256(cluster_output),
        "size_bytes": cluster_output.stat().st_size,
    }:
        raise ValueError("Cluster sensitivity output hash changed")
    expected_synchronization = {
        "M08": {
            "grouping_key": ["dataset_id"],
            "cluster_unit": "entity_id",
            "shared_draw_scope": ["seed", "model", "strength"],
            "seed_effects_preserved_within_each_entity_draw": True,
        },
        "M09": {
            "grouping_key": ["dataset_id", "seed", "strength"],
            "cluster_unit": "source_id",
            "shared_draw_scope": ["model"],
            "interpretation": "descriptive_designed_category_reweighting_only",
        },
    }
    if cluster_manifest.get("synchronization") != expected_synchronization:
        raise ValueError("Cluster synchronized-draw semantics changed")

    task_manifest_path = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
    if cluster_manifest.get("task_manifest") != {
        "path": relative(task_manifest_path),
        "sha256": sha256(task_manifest_path),
        "verified_fields": [
            "test_idx", "y", "entity_ids", "source_ids", "task_hash",
        ],
    }:
        raise ValueError("Cluster manifest does not bind the frozen task references")
    task_manifest = pd.read_csv(task_manifest_path)
    bundle_hash_by_path = freeze.get("bundle_sha256_by_path", {})
    expected_bundles = []
    for bundle_name in sorted(bundle_hash_by_path):
        bundle_path = ROOT / bundle_name
        if (
            not bundle_path.is_file()
            or sha256(bundle_path) != bundle_hash_by_path[bundle_name]
        ):
            raise ValueError(f"Frozen cluster bundle hash changed: {bundle_name}")
        expected_bundles.append({
            "path": bundle_name,
            "sha256": bundle_hash_by_path[bundle_name],
            "size_bytes": bundle_path.stat().st_size,
        })
    if len(expected_bundles) != 20 or cluster_manifest.get("frozen_bundles") != expected_bundles:
        raise ValueError("Cluster manifest does not enumerate all 20 frozen bundles")

    prediction_entries = cluster_manifest.get("consumed_predictions")
    if not isinstance(prediction_entries, list) or len(prediction_entries) != 5000:
        raise ValueError("Cluster manifest does not enumerate all 5,000 predictions")
    run_ids = [str(entry.get("run_id")) for entry in prediction_entries]
    prediction_paths = [str(entry.get("path")) for entry in prediction_entries]
    if len(set(run_ids)) != 5000 or len(set(prediction_paths)) != 5000:
        raise ValueError("Cluster manifest contains duplicate prediction identities/paths")
    expected_runs = set(
        canonical.loc[canonical["mechanism"].isin(["M08", "M09"]), "run_id"].astype(str)
    )
    if set(run_ids) != expected_runs:
        raise ValueError("Cluster manifest prediction run IDs differ from canonical M08/M09")
    mechanism_counts = pd.Series(
        [str(entry.get("mechanism")) for entry in prediction_entries]
    ).value_counts().to_dict()
    if mechanism_counts != {"M08": 2500, "M09": 2500}:
        raise ValueError("Cluster manifest prediction mechanism coverage changed")
    canonical_lookup = canonical.set_index("run_id", verify_integrity=True)
    task_lookup = task_manifest.set_index(
        ["dataset_id", "mechanism", "strength", "seed"], verify_integrity=True
    )
    for entry in prediction_entries:
        run_id = str(entry["run_id"])
        canonical_row = canonical_lookup.loc[run_id]
        task_key = (
            str(canonical_row["dataset_id"]), str(canonical_row["mechanism"]),
            str(canonical_row["strength"]), int(canonical_row["seed"]),
        )
        task_row = task_lookup.loc[task_key]
        expected_lineage = {
            "task_hash": str(task_row["task_hash"]),
            "split_hash": str(task_row["split_hash"]),
            "bundle_path": str(task_row["bundle_path"]),
            "bundle_sha256": str(task_row["bundle_sha256"]),
        }
        for field, expected_value in expected_lineage.items():
            if str(entry.get(field)) != expected_value:
                raise ValueError(
                    f"Cluster prediction {run_id} differs from frozen task lineage: {field}"
                )
        prediction_path = ROOT / str(entry["path"])
        try:
            prediction_path.resolve().relative_to(CORRECTED_ROOT.resolve())
        except ValueError as error:
            raise ValueError("Cluster prediction path escaped corrected_v2") from error
        if (
            not prediction_path.is_file()
            or entry.get("sha256") != sha256(prediction_path)
            or entry.get("size_bytes") != prediction_path.stat().st_size
        ):
            raise ValueError(f"Cluster prediction hash/size mismatch: {entry['path']}")
    return freeze


def _diagnostic_metric_digest(frame: pd.DataFrame) -> str:
    columns = [
        "dataset_id", "mechanism", "strength", "seed", "method",
        "localization_ap", "localization_normalized_ap", "top5_recall",
    ]
    ordered = frame[columns].sort_values(columns[:5]).reset_index(drop=True)
    payload = ordered.to_csv(
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_diagnostic_rng_amendment(
    paths: EvidencePaths, task_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Require the scope-locked fixed-seed-42 MI canonical amendment."""
    freeze = load_json(paths.diagnostic_amendment_freeze)
    manifest = load_json(paths.diagnostic_canonical_manifest)
    exact_freeze = {
        "status": "FROZEN_BEFORE_DIAGNOSTIC_RNG_AMENDMENT",
        "amendment_id": "diagnostic_mi_fixed_seed_42_v1",
        "discovery_phase": "post_unblinding",
        "no_tuning": True,
        "thresholds_changed": False,
        "model_outcomes_read_by_builder": False,
        "diagnostic_methods_changed": False,
        "expected_canonical_rows": 22000,
        "expected_replaced_rows": 5500,
        "expected_preserved_rows": 16500,
        "replacement_random_state": 42,
        "selection_rule": "method == mutual_information; no outcome- or value-dependent filtering",
    }
    for field, expected in exact_freeze.items():
        if freeze.get(field) != expected:
            raise ValueError(f"Diagnostic RNG amendment freeze mismatch for {field}")
    if freeze.get("scientific_identity") != [
        "dataset_id", "mechanism", "strength", "seed", "method",
    ] or freeze.get("replacement_metric_fields") != [
        "localization_ap", "localization_normalized_ap", "top5_recall",
    ]:
        raise ValueError("Diagnostic RNG amendment changed identity or replacement scope")

    builder = freeze.get("builder", {})
    builder_path = ROOT / str(builder.get("path", ""))
    if (
        relative(builder_path) != "scripts/build_diagnostic_rng_amendment.py"
        or not builder_path.is_file()
        or builder.get("sha256") != sha256(builder_path)
    ):
        raise ValueError("Diagnostic RNG amendment builder hash mismatch")
    expected_outputs = {
        "canonical": relative(paths.diagnostic_cells),
        "manifest": relative(paths.diagnostic_canonical_manifest),
    }
    if freeze.get("outputs") != expected_outputs:
        raise ValueError(
            "Raw task-seeded diagnostic output is forbidden as final evidence; "
            "the selected input is not the frozen amended canonical table"
        )

    source_raw = freeze.get("source_raw", {})
    raw_path = ROOT / str(source_raw.get("path", ""))
    if raw_path.resolve() == paths.diagnostic_cells.resolve():
        raise ValueError("Raw task-seeded MI rows cannot be used as final diagnostics")
    if (
        source_raw.get("path") != "results/corrected_v2/diagnostic_confirmatory_cells.csv"
        or source_raw.get("rows") != 22000
        or source_raw.get("mi_rng_policy") != "injection_seed"
        or not raw_path.is_file()
        or source_raw.get("sha256") != sha256(raw_path)
    ):
        raise ValueError("Diagnostic RNG amendment raw source mismatch")
    raw_manifest_entry = freeze.get("source_raw_manifest", {})
    raw_manifest_path = ROOT / str(raw_manifest_entry.get("path", ""))
    if (
        not raw_manifest_path.is_file()
        or raw_manifest_entry.get("sha256") != sha256(raw_manifest_path)
        or load_json(raw_manifest_path).get("output_sha256") != sha256(raw_path)
    ):
        raise ValueError("Diagnostic RNG amendment raw manifest mismatch")
    task_path = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
    if freeze.get("replacement_source") != {
        "path": relative(task_path), "sha256": sha256(task_path), "rows": 5500,
        "mi_rng_policy": "fixed_seed_42",
    }:
        raise ValueError("Diagnostic RNG amendment replacement source mismatch")
    if freeze.get("source_diagnostic_protocol") != {
        "path": relative(paths.diagnostic_freeze), "sha256": sha256(paths.diagnostic_freeze),
    }:
        raise ValueError("Diagnostic RNG amendment is not bound to the frozen raw protocol")

    canonical = pd.read_csv(paths.diagnostic_cells)
    raw = pd.read_csv(raw_path)
    key = ["dataset_id", "mechanism", "strength", "seed", "method"]
    task_key = key[:-1]
    required_provenance = {
        "rng_amendment_id", "rng_amendment_applied", "diagnostic_rng_policy", "metric_source",
    }
    require_columns(canonical, required_provenance, paths.diagnostic_cells)
    if (
        len(canonical) != 22000 or len(raw) != 22000 or len(task_manifest) != 5500
        or canonical.duplicated(key).any() or raw.duplicated(key).any()
        or task_manifest.duplicated(task_key).any()
    ):
        raise ValueError("Diagnostic RNG amendment coverage/identity is incomplete")
    if not canonical[key].equals(raw[key]):
        raise ValueError("Diagnostic RNG amendment changed scientific row identities")
    mi = canonical["method"].astype(str) == "mutual_information"
    raw_mi = raw["method"].astype(str) == "mutual_information"
    if int(mi.sum()) != 5500 or int((~mi).sum()) != 16500 or not mi.equals(raw_mi):
        raise ValueError("Diagnostic RNG amendment did not select every and only MI row")
    if not canonical.loc[~mi, raw.columns].reset_index(drop=True).equals(
        raw.loc[~raw_mi].reset_index(drop=True)
    ):
        raise ValueError("Diagnostic RNG amendment changed one of the 16,500 preserved rows")

    checked = canonical.loc[mi].merge(
        task_manifest[
            task_key + [
                "dataset_index", "dataset_namespace", "task_hash", "split_hash", "bundle_sha256",
                "n_leak", "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall",
            ]
        ],
        on=task_key, suffixes=("", "_task"), validate="one_to_one",
    )
    if len(checked) != 5500:
        raise ValueError("Canonical MI rows are not task-manifest complete")
    for field in ("dataset_namespace", "task_hash", "split_hash", "bundle_sha256"):
        if not (checked[field].astype(str) == checked[f"{field}_task"].astype(str)).all():
            raise ValueError(f"Canonical MI {field} differs from the frozen task manifest")
    for field in ("dataset_index", "n_leak"):
        if not np.array_equal(
            pd.to_numeric(checked[field]).to_numpy(),
            pd.to_numeric(checked[f"{field}_task"]).to_numpy(),
        ):
            raise ValueError(f"Canonical MI {field} differs from the frozen task manifest")
    for target, source in (
        ("localization_ap", "diagnostic_ap"),
        ("localization_normalized_ap", "diagnostic_normalized_ap"),
        ("top5_recall", "top5_recall_task"),
    ):
        if not np.allclose(checked[target], checked[source], atol=0.0, rtol=0.0):
            raise ValueError(f"Canonical MI {target} differs from fixed-seed-42 replacement")
    expected_policy = {
        "mutual_information": "fixed_seed_42",
        "absolute_correlation": "deterministic_no_rng",
        "lr_coefficient": "frozen_injection_seed",
        "rf_permutation": "frozen_injection_seed",
    }
    if (
        set(canonical["rng_amendment_id"].astype(str)) != {"diagnostic_mi_fixed_seed_42_v1"}
        or not all(normalize_bool(value) for value in canonical.loc[mi, "rng_amendment_applied"])
        or any(normalize_bool(value) for value in canonical.loc[~mi, "rng_amendment_applied"])
        or canonical.groupby("method")["diagnostic_rng_policy"].first().to_dict() != expected_policy
        or set(canonical.loc[mi, "metric_source"].astype(str)) != {relative(task_path)}
        or set(canonical.loc[~mi, "metric_source"].astype(str)) != {relative(raw_path)}
    ):
        raise ValueError("Diagnostic RNG amendment row-level provenance mismatch")

    expected_manifest = {
        "status": "CANONICAL_DIAGNOSTIC_AMENDED",
        "amendment_id": "diagnostic_mi_fixed_seed_42_v1",
        "expected_rows": 22000,
        "rows_replaced": 5500,
        "rows_preserved": 16500,
        "methods": DIAGNOSTIC_METHODS,
        "replacement_metric_fields": [
            "localization_ap", "localization_normalized_ap", "top5_recall",
        ],
        "builder": builder,
        "raw_source": {"path": relative(raw_path), "sha256": sha256(raw_path)},
        "replacement_source": {"path": relative(task_path), "sha256": sha256(task_path)},
        "canonical": {"path": relative(paths.diagnostic_cells), "sha256": sha256(paths.diagnostic_cells)},
        "amendment_freeze": {
            "path": relative(paths.diagnostic_amendment_freeze),
            "sha256": sha256(paths.diagnostic_amendment_freeze),
        },
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Diagnostic canonical manifest mismatch for {field}")
    identity_checks = manifest.get("identity_checks", {})
    if (
        identity_checks.get("scientific_identity_exact") is not True
        or identity_checks.get("non_mi_existing_columns_exact") is not True
        or identity_checks.get("mi_replacement_metrics_exact") is not True
        or identity_checks.get("task_split_bundle_lineage_exact") is not True
        or identity_checks.get("raw_duplicate_identities") is not False
        or identity_checks.get("replacement_duplicate_identities") is not False
        or identity_checks.get("canonical_duplicate_identities") is not False
    ):
        raise ValueError("Diagnostic canonical manifest identity checks are not fail-closed")
    records = manifest.get("record_hashes", {})
    raw_non_mi_hash = hashlib.sha256(
        raw.loc[~raw_mi].to_csv(
            index=False, lineterminator="\n", float_format="%.17g"
        ).encode("utf-8")
    ).hexdigest()
    if records != {
        "raw_mi_metrics_sha256": _diagnostic_metric_digest(raw.loc[raw_mi]),
        "replacement_mi_metrics_sha256": _diagnostic_metric_digest(canonical.loc[mi]),
        "canonical_mi_metrics_sha256": _diagnostic_metric_digest(canonical.loc[mi]),
        "preserved_non_mi_rows_sha256": raw_non_mi_hash,
    }:
        raise ValueError("Diagnostic canonical manifest record hashes mismatch")
    return {
        "freeze": freeze, "manifest": manifest, "raw_path": raw_path,
        "raw_manifest_path": raw_manifest_path,
    }


def validate_diagnostic_suite(
    paths: EvidencePaths, bundle: dict[str, Any], config: dict[str, Any],
    task_manifest: pd.DataFrame, freeze: dict[str, Any],
) -> dict[str, Any]:
    cells = pd.read_csv(paths.diagnostic_cells)
    required = {
        "diagnostic_run_id", "dataset_id", "dataset_namespace", "mechanism", "strength", "seed", "method",
        "status", "localization_normalized_ap", "task_hash", "split_hash", "bundle_sha256",
        "integrity_verified", "config_hash", "code_hash", "task_manifest_sha256",
    }
    require_columns(cells, required, paths.diagnostic_cells)
    key = ["dataset_id", "mechanism", "strength", "seed", "method"]
    if len(cells) != 22000 or cells.duplicated(key).any():
        raise ValueError("Diagnostic suite is not the exact 22,000-cell design")
    if set(cells["dataset_namespace"].astype(str)) != {"confirmatory"}:
        raise ValueError("Diagnostic suite is not confirmatory")
    if not (cells["status"].astype(str) == "SUCCESS").all():
        raise ValueError("Diagnostic suite contains failed cells")
    if not all(normalize_bool(value) for value in cells["integrity_verified"]):
        raise ValueError("Diagnostic suite contains an unverified task")
    if sorted(cells["method"].astype(str).unique()) != sorted(DIAGNOSTIC_METHODS):
        raise ValueError("Diagnostic method set differs from the frozen four-method suite")
    if cells["diagnostic_run_id"].astype(str).duplicated().any():
        raise ValueError("Diagnostic run IDs are not unique")
    task_manifest_path = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
    if len(task_manifest) != 5500 or task_manifest.duplicated(["dataset_id", "mechanism", "strength", "seed"]).any():
        raise ValueError("Frozen diagnostic task manifest is incomplete")
    if set(cells["config_hash"].astype(str)) != {sha256(paths.config)}:
        raise ValueError("Diagnostic cells use the wrong config hash")
    if set(cells["task_manifest_sha256"].astype(str)) != {sha256(task_manifest_path)}:
        raise ValueError("Diagnostic cells use the wrong task-manifest hash")
    runner_entry = freeze["frozen_files"]["experiments/leakbench/run_diagnostic_suite.py"]
    if set(cells["code_hash"].astype(str)) != {runner_entry["sha256"]}:
        raise ValueError("Diagnostic cells use an unfrozen runner hash")
    require_finite(cells, ["localization_normalized_ap"], paths.diagnostic_cells)

    task_fields = ["dataset_id", "mechanism", "strength", "seed", "task_hash", "split_hash", "bundle_sha256"]
    merged = cells.merge(task_manifest[task_fields], on=key[:-1], suffixes=("", "_task"), validate="many_to_one")
    for field in ("task_hash", "split_hash", "bundle_sha256"):
        if not (merged[field].astype(str) == merged[f"{field}_task"].astype(str)).all():
            raise ValueError(f"Diagnostic {field} differs from the frozen task manifest")

    integrity = bundle["integrity"]
    expected_integrity = {
        "evidence_tier": "confirmatory", "source": relative(paths.diagnostic_cells),
        "rows_success": 22000, "rows_failure": 0, "expected_cells": 22000,
        "datasets": 20, "mechanisms": sorted(CATEGORIES), "methods": sorted(DIAGNOSTIC_METHODS),
        "strengths": config["protocol"]["strengths"], "seeds": sorted(config["protocol"]["seeds"]),
        "bootstrap_repetitions": 20000, "bootstrap_seed": 20260713,
        "primary_diagnostic": "mutual_information", "robust_low_threshold": 0.30,
    }
    for field, expected in expected_integrity.items():
        observed = integrity.get(field)
        if isinstance(expected, list):
            if sorted(observed or []) != sorted(expected):
                raise ValueError(f"Diagnostic integrity mismatch for {field}")
        elif observed != expected:
            raise ValueError(f"Diagnostic integrity mismatch for {field}: {observed!r}")
    if "not a zero-shot ensemble" not in str(integrity.get("best_diagnostic_interpretation", "")):
        raise ValueError("Diagnostic integrity omits the non-deployable best-method warning")

    by_mechanism = bundle["by_mechanism"].copy()
    expected_pairs = {(method, mechanism) for method in DIAGNOSTIC_METHODS for mechanism in CATEGORIES}
    observed_pairs = set(zip(by_mechanism["method"].astype(str), by_mechanism["mechanism"].astype(str)))
    if observed_pairs != expected_pairs or by_mechanism.duplicated(["method", "mechanism"]).any():
        raise ValueError("Diagnostic method-by-mechanism table is incomplete")
    require_finite(by_mechanism, ["diagnostic_normalized_ap", "ci_low", "ci_high"], bundle["paths"]["by_mechanism"])
    observed_categories = dict(zip(by_mechanism["mechanism"].astype(str), by_mechanism["category"].astype(str)))
    if observed_categories != CATEGORIES:
        raise ValueError("Diagnostic method table categories changed")

    method_summary = bundle["method_summary"].copy()
    if sorted(method_summary["method"].astype(str)) != sorted(DIAGNOSTIC_METHODS) or method_summary["method"].duplicated().any():
        raise ValueError("Diagnostic method summary is incomplete")
    require_finite(method_summary, ["diagnostic_normalized_ap", "ci_low", "ci_high"], bundle["paths"]["method_summary"])
    profiles = bundle["profiles"].copy()
    _validate_exact_mechanisms(profiles, bundle["paths"]["profiles"])
    require_finite(profiles, [
        "best_evaluated_diagnostic", "best_ci_low", "best_ci_high", "worst_evaluated_diagnostic",
        "worst_ci_low", "worst_ci_high", "between_diagnostic_range",
    ], bundle["paths"]["profiles"])

    point = cells.groupby(["method", "mechanism"])["localization_normalized_ap"].mean()
    method_point = cells.groupby("method")["localization_normalized_ap"].mean()
    observed = by_mechanism.set_index(["method", "mechanism"])["diagnostic_normalized_ap"]
    for pair, expected in point.items():
        _assert_close(observed[pair], expected, f"diagnostic {pair}")
    method_observed = method_summary.set_index("method")["diagnostic_normalized_ap"]
    for method, expected in method_point.items():
        _assert_close(method_observed[method], expected, f"diagnostic method {method}")
    profile_index = profiles.set_index("mechanism")
    for mechanism in CATEGORIES:
        values = point.xs(mechanism, level="mechanism")
        _assert_close(profile_index.loc[mechanism, "best_evaluated_diagnostic"], values.max(), f"{mechanism} best diagnostic")
        _assert_close(profile_index.loc[mechanism, "worst_evaluated_diagnostic"], values.min(), f"{mechanism} worst diagnostic")
        _assert_close(profile_index.loc[mechanism, "between_diagnostic_range"], values.max() - values.min(), f"{mechanism} diagnostic range")
        expected_low = float(profile_index.loc[mechanism, "best_ci_high"]) < 0.30
        if normalize_bool(profile_index.loc[mechanism, "low_across_all_evaluated_diagnostics"]) != expected_low:
            raise ValueError(f"{mechanism} robust-low flag differs from the frozen rule")
    return {"cells": cells, **bundle}


def validate_natural_freeze(path: Path) -> dict[str, Any]:
    freeze = load_json(path)
    if freeze.get("status") != "FROZEN_BEFORE_NATURAL_TRAINFIT_V2_RERUN":
        raise ValueError("Train-fitted-category natural protocol v2 freeze is not active")
    if (
        freeze.get("amendment_version") != "natural_trainfit_categories_v2"
        or freeze.get("expected_cells") != 60
        or set(freeze.get("tasks", [])) != {"BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"}
        or set(freeze.get("models", [])) != {"lr", "rf", "catboost", "lightgbm"}
        or sorted(int(seed) for seed in freeze.get("seeds", [])) != [13, 42, 2026]
        or freeze.get("fit_scope") != "training rows only"
        or freeze.get("unknown_category_value") != -2.0
        or freeze.get("missing_category_value") != -1.0
        or freeze.get("supersedes") != "results/corrected_v2/natural_protocol_freeze.json"
        or freeze.get("output")
        != "results/corrected_v2/public_natural/natural_cells.csv"
        or freeze.get("task_summary")
        != "results/corrected_v2/public_natural/natural_task_summary.csv"
    ):
        raise ValueError("Natural train-fit protocol v2 freeze design is incomplete")
    policy = freeze.get("category_policy")
    if not isinstance(policy, dict):
        raise ValueError("Natural train-fit protocol lacks a category policy")
    policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(policy_bytes).hexdigest() != freeze.get("category_policy_sha256"):
        raise ValueError("Natural category-policy hash mismatch")
    code_files = freeze.get("code_files", {})
    if not code_files:
        raise ValueError("Natural protocol freeze does not bind code")
    for name, entry in code_files.items():
        code_path = ROOT / name
        if (
            not code_path.is_file()
            or code_path.stat().st_size != entry.get("size_bytes")
            or sha256(code_path) != entry.get("sha256")
        ):
            raise ValueError(f"Natural frozen code hash mismatch: {name}")
    source_files = freeze.get("source_files", {})
    if set(source_files) != set(freeze["tasks"]):
        raise ValueError("Natural protocol freeze source map is incomplete")
    for task, entry in source_files.items():
        source = Path(str(entry.get("path", "")))
        source_hash = str(entry.get("sha256", ""))
        if (
            source.is_absolute()
            or ".." in source.parts
            or source.parts[:1] != ("external_sources",)
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] <= 0
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
        ):
            raise ValueError(f"Public natural source descriptor is invalid: {task}")
    projection = freeze.get("public_projection", {})
    projection_script = ROOT / "scripts/build_public_natural_provenance.py"
    if projection != {
        "version": "natural_public_provenance_v1",
        "generator_path": relative(projection_script),
        "generator_sha256": sha256(projection_script),
        "private_freeze_sha256": projection.get("private_freeze_sha256"),
        "raw_source_files_included": False,
        "source_path_policy": "repo_relative_external_sources_placeholders",
        "scientific_fields_unchanged": True,
    }:
        raise ValueError("Natural public freeze projection metadata changed")
    private_hash = str(projection.get("private_freeze_sha256", ""))
    if len(private_hash) != 64 or any(
        character not in "0123456789abcdef" for character in private_hash
    ):
        raise ValueError("Natural public freeze lacks a typed private freeze hash")
    return freeze


def validate_public_natural_projection(paths: EvidencePaths) -> dict[str, Any]:
    manifest = load_json(paths.natural_public_manifest)
    projection_script = ROOT / "scripts/build_public_natural_provenance.py"
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "PUBLIC_NATURAL_PROVENANCE_PROJECTED"
        or manifest.get("projection_version") != "natural_public_provenance_v1"
        or manifest.get("raw_natural_data_included") is not False
        or manifest.get("all_scientific_invariants_passed") is not True
        or manifest.get("generator") != {
            "path": relative(projection_script), "sha256": sha256(projection_script)
        }
    ):
        raise ValueError("Public natural provenance manifest identity changed")
    expected_public = {
        "freeze": paths.natural_freeze,
        "cells": paths.natural_cells,
        "tasks": paths.natural_tasks,
        "statistics": paths.natural_statistics,
    }
    expected_outputs = {
        logical: {
            "path": relative(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for logical, path in expected_public.items()
    }
    if manifest.get("public_outputs") != expected_outputs:
        raise ValueError("Public natural output hashes changed")
    private = manifest.get("private_provenance", {})
    expected_private_paths = {
        "freeze": "results/corrected_v2/natural_protocol_v2_freeze.json",
        "cells": "results/corrected_v2/natural_cells.csv",
        "tasks": "results/corrected_v2/natural_task_summary.csv",
        "statistics": "results/corrected_v2/natural_statistics.json",
    }
    if (
        private.get("distribution") != "EXCLUDED_FROM_PUBLIC_ARTIFACT"
        or set(private.get("artifacts", {})) != set(expected_private_paths)
    ):
        raise ValueError("Private natural provenance is not explicitly excluded")
    private_hashes: dict[str, str] = {}
    for logical, expected_path in expected_private_paths.items():
        entry = private["artifacts"][logical]
        digest = str(entry.get("sha256", ""))
        if (
            entry.get("artifact_type") != "private_natural_provenance"
            or entry.get("path") != expected_path
            or not isinstance(entry.get("size_bytes"), int)
            or entry["size_bytes"] <= 0
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Private natural provenance descriptor changed: {logical}")
        private_hashes[logical] = digest
    typed_mapping = manifest.get("private_to_public")
    if not isinstance(typed_mapping, list) or len(typed_mapping) != 4:
        raise ValueError("Natural projection lacks four typed private-to-public mappings")
    by_logical = {str(entry.get("logical_name")): entry for entry in typed_mapping}
    if set(by_logical) != set(expected_public):
        raise ValueError("Natural projection typed mapping is incomplete")
    for logical, public_path in expected_public.items():
        entry = by_logical[logical]
        expected_type = "csv" if public_path.suffix == ".csv" else "json"
        if entry != {
            "logical_name": logical,
            "artifact_type": expected_type,
            "private_sha256": private_hashes[logical],
            "public_sha256": sha256(public_path),
            "public_path": relative(public_path),
            "private_distribution": "EXCLUDED_FROM_PUBLIC_ARTIFACT",
        }:
            raise ValueError(f"Natural projection typed mapping changed: {logical}")
    invariants = manifest.get("scientific_invariants", {})
    if set(invariants) != {
        "natural_cells", "natural_task_summary", "natural_protocol_freeze",
        "natural_statistics",
    } or any(entry.get("passed") is not True for entry in invariants.values()):
        raise ValueError("Natural projection scientific invariant set changed")
    for logical in (
        "natural_task_summary", "natural_protocol_freeze", "natural_statistics",
    ):
        entry = invariants[logical]
        if entry.get("private_scientific_sha256") != entry.get(
            "public_scientific_sha256"
        ):
            raise ValueError(f"Natural public scientific digest differs: {logical}")
    freeze_projection = load_json(paths.natural_freeze).get("public_projection", {})
    if freeze_projection.get("private_freeze_sha256") != private_hashes["freeze"]:
        raise ValueError("Public freeze does not bind its typed private freeze hash")
    return manifest


def validate_environment_locks(paths: EvidencePaths) -> None:
    local = load_json(paths.local_environment)
    if (
        local.get("schema_version") != 1
        or local.get("environment") != "local_corrected_v2_cpu"
        or local.get("requirements_path") != relative(paths.requirements)
        or local.get("requirements_sha256") != sha256(paths.requirements)
    ):
        raise ValueError("Local environment lock does not bind requirements-corrected-v2.txt")
    required_local_packages = {
        "numpy": "2.1.3", "pandas": "2.2.3", "scipy": "1.15.3",
        "scikit-learn": "1.6.1", "statsmodels": "0.14.4",
        "catboost": "1.2.10", "lightgbm": "4.6.0", "PyYAML": "6.0.2",
    }
    packages = local.get("packages", {})
    if any(packages.get(name) != version for name, version in required_local_packages.items()):
        raise ValueError("Local environment package lock differs from the frozen CPU stack")

    tabm = load_json(paths.tabm_environment)
    historical_freeze_hash = str(tabm.get("protocol_freeze_sha256", ""))
    if (
        tabm.get("schema_version") != 1
        or tabm.get("environment") != "leakbench-tabm-official"
        or tabm.get("python") != "3.11.15"
        or tabm.get("packages", {}).get("tabm") != "0.0.3"
        or tabm.get("packages", {}).get("torch") != "2.5.1+cu121"
        or tabm.get("cuda_runtime") != "12.1"
        or "RTX 4060" not in str(tabm.get("gpu", ""))
        or len(historical_freeze_hash) != 64
        or any(character not in "0123456789abcdef" for character in historical_freeze_hash)
    ):
        raise ValueError("Official TabM environment lock is incomplete or changed")


def validate_natural(paths: EvidencePaths, freeze: dict[str, Any]) -> dict[str, Any]:
    payload = load_json(paths.natural_statistics)
    cells = pd.read_csv(paths.natural_cells)
    tasks = pd.read_csv(paths.natural_tasks)
    expected_tasks = {"BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"}
    expected_models = {"lr", "rf", "catboost", "lightgbm"}
    require_columns(cells, {
        "preprocessing_protocol", "preprocessing_mapping_sha256", "source_sha256",
        "strict_auc", "permissive_auc", "paired_harm",
    }, paths.natural_cells)
    require_columns(tasks, {
        "preprocessing_protocol", "preprocessing_mapping_sha256",
        "retained_feature_names_sha256", "source_sha256", "lineage",
    }, paths.natural_tasks)
    if len(cells) != 60 or set(cells["task"].astype(str)) != expected_tasks or set(cells["model"].astype(str)) != expected_models:
        raise ValueError("Natural case-study matrix changed")
    if cells.duplicated(["task", "model", "seed"]).any() or not (cells["status"].astype(str) == "SUCCESS").all():
        raise ValueError("Natural case-study matrix is incomplete")
    if (
        set(cells["preprocessing_protocol"].astype(str))
        != {"natural_trainfit_categories_v2"}
        or set(tasks["preprocessing_protocol"].astype(str))
        != {"natural_trainfit_categories_v2"}
    ):
        raise ValueError("Natural outputs do not use the frozen train-fitted-category protocol v2")
    if set(tasks["task"].astype(str)) != expected_tasks or len(tasks) != 5 or tasks["source_sha256"].nunique() != 5:
        raise ValueError("Natural task lineage is incomplete")
    observed_source_hashes = dict(zip(tasks["task"].astype(str), tasks["source_sha256"].astype(str)))
    expected_source_hashes = {
        task: str(entry["sha256"]) for task, entry in freeze["source_files"].items()
    }
    if observed_source_hashes != expected_source_hashes:
        raise ValueError("Natural task summary does not match the frozen per-task source hashes")
    observed_sources = dict(zip(tasks["task"].astype(str), tasks["source"].astype(str)))
    expected_sources = {
        task: str(entry["path"]) for task, entry in freeze["source_files"].items()
    }
    if observed_sources != expected_sources:
        raise ValueError("Natural task summary source paths differ from the frozen lineage")
    cell_source_hashes = cells.groupby("task")["source_sha256"].agg(lambda values: set(values.astype(str)))
    if any(values != {expected_source_hashes[task]} for task, values in cell_source_hashes.items()):
        raise ValueError("Natural result cells do not carry the frozen per-task source hashes")
    task_mapping_hashes = dict(zip(
        tasks["task"].astype(str), tasks["preprocessing_mapping_sha256"].astype(str)
    ))
    for task, values in cells.groupby("task")["preprocessing_mapping_sha256"]:
        if set(values.astype(str)) != {task_mapping_hashes[str(task)]}:
            raise ValueError(f"Natural cell preprocessing map differs from task summary: {task}")
    for row in tasks.itertuples():
        lineage = json.loads(row.lineage)
        preprocessing = lineage.get("preprocessing", {})
        if (
            lineage.get("is_synthetic") is not False
            or lineage.get("source_sha256") != expected_source_hashes[row.task]
            or lineage.get("source_path") != expected_sources[row.task]
            or lineage.get("preprocessing_protocol") != "natural_trainfit_categories_v2"
            or preprocessing.get("protocol_version") != "natural_trainfit_categories_v2"
            or preprocessing.get("fit_rows") != "train_idx_only"
            or preprocessing.get("mapping_sha256") != row.preprocessing_mapping_sha256
            or preprocessing.get("missing_category_value") != -1.0
            or preprocessing.get("unknown_category_value") != -2.0
        ):
            raise ValueError(f"Natural lineage payload differs from the frozen source: {row.task}")
        if row.task == "BTSFlights" and (
            lineage.get("prediction_boundary") != "immediately_before_scheduled_departure"
            or lineage.get("availability_rule") != "schedule_allowlist_v2_all_other_operational_fields_post_event"
        ):
            raise ValueError("BTS lineage does not record the frozen scheduled-departure boundary")
    if payload.get("public_projection_version") != "natural_public_provenance_v1":
        raise ValueError("Natural statistics are not the public provenance projection")
    if payload.get("interpretation") != "fixed real-data case studies; not a population-level dataset sample":
        raise ValueError("Natural evidence interpretation was broadened")
    if payload.get("cells") != 60 or payload.get("tasks") != 5 or set(payload.get("models", [])) != expected_models:
        raise ValueError("Natural statistics coverage changed")
    if payload.get("cells_sha256") != sha256(paths.natural_cells) or payload.get("task_summary_sha256") != sha256(paths.natural_tasks):
        raise ValueError("Natural statistics source hash mismatch")
    task_effects = cells.groupby("task")["paired_harm"].mean()
    require_finite(cells, ["strict_auc", "permissive_auc", "paired_harm"], paths.natural_cells)
    if not np.allclose(
        cells["permissive_auc"] - cells["strict_auc"], cells["paired_harm"], atol=1e-12, rtol=0,
    ):
        raise ValueError("Natural paired harm differs from permissive minus strict AUROC")
    _assert_close(payload["mean_paired_harm"], task_effects.mean(), "natural mean paired harm")
    if bool(payload.get("all_task_effects_positive")) != bool((task_effects > 0).all()):
        raise ValueError("Natural all-positive flag differs from the fixed tasks")
    values = task_effects.to_numpy(dtype=float)
    rng = np.random.RandomState(20260713)
    bootstrap = np.empty(20_000, dtype=float)
    for repetition in range(len(bootstrap)):
        bootstrap[repetition] = rng.choice(
            values, size=len(values), replace=True
        ).mean()
    expected_interval = [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]
    interval = payload.get("task_bootstrap_ci", [])
    if (
        len(interval) != 2
        or not np.allclose(interval, expected_interval, atol=1e-15, rtol=0)
    ):
        raise ValueError("Natural bootstrap interval differs from fixed-seed recomputation")
    signs = np.array(
        np.meshgrid(*[[-1.0, 1.0]] * len(values))
    ).T.reshape(-1, len(values))
    observed = abs(values.mean())
    expected_p = float(
        np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15)
    )
    _assert_close(
        payload.get("exact_two_sided_sign_flip_p"), expected_p,
        "natural exact sign-flip p", tolerance=1e-15,
    )
    expected_task_effects = {str(key): float(value) for key, value in task_effects.items()}
    if payload.get("task_effects") != expected_task_effects:
        raise ValueError("Natural task effects differ from the fixed cells")
    model_effects = cells.groupby("model")["paired_harm"].mean()
    expected_model_effects = {str(key): float(value) for key, value in model_effects.items()}
    if payload.get("model_effects") != expected_model_effects:
        raise ValueError("Natural model effects differ from the fixed cells")
    expected_diagnostic = dict(zip(
        tasks["task"].astype(str),
        pd.to_numeric(tasks["diagnostic_normalized_ap"]).astype(float),
    ))
    if payload.get("diagnostic_normalized_ap") != expected_diagnostic:
        raise ValueError("Natural diagnostic values differ from the task summary")
    return payload


def collect_evidence(paths: EvidencePaths) -> dict[str, Any]:
    for path in (
        paths.canonical, paths.canonical_manifest, paths.diagnostic_cells,
        paths.diagnostic_canonical_manifest, paths.diagnostic_amendment_freeze,
        paths.natural_statistics, paths.natural_cells,
        paths.natural_tasks, paths.natural_freeze, paths.natural_public_manifest,
        paths.config, paths.protocol_freeze, paths.tabm_bundle_freeze,
        paths.diagnostic_freeze, paths.statistical_amendment_freeze,
        paths.superseded, paths.local_environment,
        paths.tabm_environment, paths.requirements, paths.m10_config, paths.m10_freeze,
        paths.m10_cpu, paths.m10_cpu_manifest, paths.m10_tabm, paths.m10_tabm_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(paths.config.read_text(encoding="utf-8"))
    superseded = load_superseded(paths.superseded)
    stat_bundle = validate_statistics_schema(paths.statistics_dir)
    diagnostic_bundle = validate_diagnostic_statistics_schema(paths.diagnostic_statistics_dir)
    evidence_paths = [
        paths.canonical, paths.canonical_manifest, paths.diagnostic_cells,
        paths.diagnostic_canonical_manifest, paths.diagnostic_amendment_freeze,
        paths.statistical_amendment_freeze,
        paths.natural_statistics, paths.natural_cells, paths.natural_tasks,
        paths.m10_cpu, paths.m10_cpu_manifest, paths.m10_tabm, paths.m10_tabm_manifest,
        *stat_bundle["paths"].values(), *diagnostic_bundle["paths"].values(),
    ]
    reject_superseded(evidence_paths, superseded)
    protocol_freeze = validate_freeze(paths.protocol_freeze, "FROZEN_BEFORE_CONFIRMATORY_RUN", "files")
    tabm_freeze = validate_freeze(paths.tabm_bundle_freeze, "FROZEN_BEFORE_BUNDLE_CONFIRMATORY_RUN", "files")
    diagnostic_freeze = validate_freeze(
        paths.diagnostic_freeze, "FROZEN_BEFORE_DIAGNOSTIC_CONFIRMATORY_RUN", "frozen_files"
    )
    natural_freeze = validate_natural_freeze(paths.natural_freeze)
    natural_public_manifest = validate_public_natural_projection(paths)
    validate_environment_locks(paths)
    config_hash = sha256(paths.config)
    if not any(entry.get("path") == relative(paths.config) and entry.get("sha256") == config_hash for entry in protocol_freeze["files"]):
        raise ValueError("Core freeze does not bind the selected config")
    if tabm_freeze.get("confirmatory_tasks") != 5500 or tabm_freeze.get("model_identity") != "tabm.TabM":
        raise ValueError("Official TabM freeze is incomplete")
    if diagnostic_freeze.get("expected_diagnostic_cells") != 22000 or diagnostic_freeze.get("diagnostic_methods") != DIAGNOSTIC_METHODS:
        raise ValueError("Diagnostic freeze is incomplete")

    m10_amendment = validate_m10_amendment(paths, config)
    canonical, canonical_manifest = validate_canonical(paths, config, m10_amendment)
    validate_confirmatory_statistics(stat_bundle, canonical, config, paths.canonical)
    statistical_amendment_freeze = validate_statistical_amendment_chain(
        paths, stat_bundle, canonical
    )
    task_manifest = pd.read_csv(ROOT / "results/corrected_v2/task_bundles/task_manifest.csv")
    diagnostic_amendment = validate_diagnostic_rng_amendment(paths, task_manifest)
    diagnostic = validate_diagnostic_suite(
        paths, diagnostic_bundle, config, task_manifest, diagnostic_freeze
    )
    primary_mi = stat_bundle["detectability"].set_index("mechanism")["diagnostic_normalized_ap"]
    suite_mi = diagnostic["by_mechanism"].loc[
        diagnostic["by_mechanism"]["method"] == "mutual_information"
    ].set_index("mechanism")["diagnostic_normalized_ap"]
    for mechanism in CATEGORIES:
        _assert_close(
            primary_mi[mechanism], suite_mi[mechanism],
            f"{mechanism} primary MI across primary/diagnostic suites", tolerance=1e-12,
        )
    natural = validate_natural(paths, natural_freeze)
    return {
        "paths": paths, "config": config, "canonical": canonical,
        "canonical_manifest": canonical_manifest, "statistics": stat_bundle,
        "diagnostic": diagnostic, "diagnostic_amendment": diagnostic_amendment,
        "natural": natural, "natural_freeze": natural_freeze,
        "natural_public_manifest": natural_public_manifest,
        "statistical_amendment_freeze": statistical_amendment_freeze,
        "m10_amendment": m10_amendment, "superseded": superseded,
    }


def _row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    subset = frame[frame[column].astype(str) == value]
    if len(subset) != 1:
        raise ValueError(f"Expected exactly one {column}={value!r} row")
    return subset.iloc[0]


def derive_claims(evidence: dict[str, Any]) -> dict[str, Any]:
    stats = evidence["statistics"]
    mechanism = stats["mechanism"]
    detectability = stats["detectability"]
    contrast = _row(stats["category_contrasts"], "contrast", "simple_minus_structured")
    by_model = stats["by_model"]
    correlation = stats["correlation"]
    cluster = stats["cluster"]

    positive_count = int((by_model["simple_minus_structured"] > 0).sum())
    excludes_zero_count = int((by_model["ci_low"] > 0).sum())
    simple_supported = bool(
        float(contrast["ci_low"]) > 0.0
        and float(contrast["holm_p"]) <= ALPHA
    )
    simple_claim = {
        "status": "SUPPORTED" if simple_supported else "NOT_SUPPORTED",
        "statement": "Simple mechanisms have greater mean paired AUROC harm than structured mechanisms in the frozen controlled benchmark.",
        "metrics": {
            "difference": float(contrast["difference"]), "ci_low": float(contrast["ci_low"]),
            "ci_high": float(contrast["ci_high"]), "holm_p": float(contrast["holm_p"]),
            "model_positive_direction_count": positive_count,
            "model_ci_excludes_zero_count": excludes_zero_count,
        },
        "criteria": {
            "pooled_ci_low_gt": 0.0,
            "exact_holm_p_lte": ALPHA,
            "model_results_used_for_support_decision": False,
        },
        "allowed_wording": "In the frozen controlled benchmark, the pre-existing directional simple-versus-structured contrast was supported when its exact Holm-adjusted test and confidence interval both excluded the null.",
        "prohibited_wording": "All simple leakage is harmful and all structured leakage is harmless.",
    }

    m03 = _row(mechanism, "mechanism", "M03")
    m08 = _row(mechanism, "mechanism", "M08")
    m09 = _row(mechanism, "mechanism", "M09")
    d03 = _row(detectability, "mechanism", "M03")
    d08 = _row(detectability, "mechanism", "M08")
    d09 = _row(detectability, "mechanism", "M09")
    detectability_separated = bool(
        float(d03["diagnostic_normalized_ap_ci_high"])
        < float(d08["diagnostic_normalized_ap_ci_low"])
    )
    m03_claim = {
        "status": "DESCRIPTIVE_ONLY",
        "statement": "Within the fixed mechanism registry, M03 had lower measured localization than M08 and a positive observed exploitation effect.",
        "metrics": {
            "detectability": float(d03["diagnostic_normalized_ap"]),
            "detectability_ci_low": float(d03["diagnostic_normalized_ap_ci_low"]),
            "detectability_ci_high": float(d03["diagnostic_normalized_ap_ci_high"]),
            "paired_harm": float(m03["paired_harm"]), "paired_harm_ci_low": float(m03["paired_harm_ci_low"]),
            "paired_harm_ci_high": float(m03["paired_harm_ci_high"]), "holm_p": float(m03["holm_p"]),
            "detectability_separated_from_m08": detectability_separated,
        },
        "criteria": {"status_is_capped_at": "DESCRIPTIVE_ONLY", "threshold_based_support_allowed": False},
        "allowed_wording": "In this fixed registry, M03's observed localization and exploitation profile differed from M08; this comparison is descriptive.",
        "prohibited_wording": "Low detectability always implies high exploitation.",
    }

    m08_cluster_low, m08_cluster_high = map(
        float, cluster["M08"]["synchronized_cluster_ci"]
    )
    m08_claim = {
        "status": "DESCRIPTIVE_ONLY",
        "statement": "M08's point estimate and dataset-synchronized entity-cluster interval are reported descriptively.",
        "metrics": {
            "detectability": float(d08["diagnostic_normalized_ap"]),
            "detectability_ci_low": float(d08["diagnostic_normalized_ap_ci_low"]),
            "detectability_ci_high": float(d08["diagnostic_normalized_ap_ci_high"]),
            "paired_harm": float(m08["paired_harm"]), "paired_harm_ci_low": float(m08["paired_harm_ci_low"]),
            "paired_harm_ci_high": float(m08["paired_harm_ci_high"]),
            "cluster_ci_low": m08_cluster_low, "cluster_ci_high": m08_cluster_high,
        },
        "criteria": {"status_is_capped_at": "DESCRIPTIVE_ONLY", "equivalence_or_practical_null_claim_allowed": False},
        "allowed_wording": "M08's observed paired harm and synchronized entity-cluster interval can be reported without an equivalence or practical-null conclusion.",
        "prohibited_wording": "M08 is practically null, equivalent to zero, or has exactly zero exploitation effect.",
    }

    m09_reweighting_low, m09_reweighting_high = map(
        float, cluster["M09"]["descriptive_reweighting_interval"]
    )
    m09_claim = {
        "status": "DESCRIPTIVE_ONLY",
        "statement": "Within this fixed registry and encoded source representation, M09 is a descriptive structured counterexample to a uniform structured-harmlessness pattern.",
        "metrics": {
            "detectability": float(d09["diagnostic_normalized_ap"]),
            "detectability_ci_low": float(d09["diagnostic_normalized_ap_ci_low"]),
            "detectability_ci_high": float(d09["diagnostic_normalized_ap_ci_high"]),
            "paired_harm": float(m09["paired_harm"]), "paired_harm_ci_low": float(m09["paired_harm_ci_low"]),
            "paired_harm_ci_high": float(m09["paired_harm_ci_high"]), "holm_p": float(m09["holm_p"]),
            "designed_category_reweighting_low": m09_reweighting_low,
            "designed_category_reweighting_high": m09_reweighting_high,
            "designed_category_reweighting_is_inferential": False,
            "detectability_unit": "encoded_column",
            "complete_one_hot": True,
            "encoded_column_count": 8,
            "semantic_field_count": 1,
            "representation_conditional": True,
        },
        "criteria": {"status_is_capped_at": "DESCRIPTIVE_ONLY", "threshold_based_support_allowed": False},
        "allowed_wording": "M09 is a representation-conditional descriptive counterexample within this registry; its designed source-category reweighting interval is not population inference.",
        "prohibited_wording": "M09 establishes a population-level source effect or that every structured mechanism is highly exploitable.",
    }

    relation_claim = {
        "status": "DESCRIPTIVE_ONLY",
        "statement": "The global detectability/exploitation association is descriptive and may be category-driven.",
        "metrics": {
            "global_spearman": float(correlation["global_spearman"]),
            "global_spearman_ci_low": float(correlation["global_spearman_ci"][0]),
            "global_spearman_ci_high": float(correlation["global_spearman_ci"][1]),
            "category_r2": float(correlation["category_r2"]),
            "category_plus_detectability_r2": float(correlation["category_plus_detectability_r2"]),
            "incremental_r2": float(correlation["incremental_r2"]),
            "incremental_permutation_p": float(correlation["incremental_permutation_p"]),
            "category_lomo_r2": float(correlation["category_lomo_r2"]),
            "category_plus_detectability_lomo_r2": float(correlation["category_plus_detectability_lomo_r2"]),
            "incremental_lomo_r2": float(correlation["incremental_lomo_r2"]),
        },
        "criteria": {"status_is_capped_at": "DESCRIPTIVE_ONLY"},
        "allowed_wording": "Across the eleven evaluated mechanisms, D and X had a descriptive association; category adjustment and leave-one-mechanism-out results bound interpretation.",
        "prohibited_wording": "Detectability generally predicts or causes exploitation on unseen leakage mechanisms.",
    }

    diagnostic = evidence["diagnostic"]
    profile = diagnostic["profiles"].set_index("mechanism")
    dtable = diagnostic["by_mechanism"]
    m03_methods = dtable[dtable["mechanism"] == "M03"].copy()
    best = m03_methods.loc[m03_methods["diagnostic_normalized_ap"].idxmax()]
    worst = m03_methods.loc[m03_methods["diagnostic_normalized_ap"].idxmin()]
    conservative_separation = bool(float(best["ci_low"]) > float(worst["ci_high"]))
    m03_range = float(profile.loc["M03", "between_diagnostic_range"])
    method_claim = {
        "status": "DESCRIPTIVE_ONLY",
        "statement": "The four evaluated diagnostics show descriptive localization differences; no paired simultaneous method-comparison interval was frozen.",
        "metrics": {
            "m03_method_range": m03_range,
            "conservative_ci_separation": conservative_separation,
            "m03_best_evaluated_method": str(best["method"]),
            "m03_worst_evaluated_method": str(worst["method"]),
        },
        "criteria": {
            "status_is_capped_at": "DESCRIPTIVE_ONLY",
            "paired_simultaneous_method_comparison_available": False,
            "threshold_based_method_labels_allowed": False,
        },
        "allowed_wording": "Across the four evaluated diagnostics, M03 showed descriptive localization variation; best-method results are non-deployable labeled benchmark summaries.",
        "prohibited_wording": "This is an inferential detector ranking or generalizes to unevaluated diagnostics.",
    }
    return {
        "simple_vs_structured": simple_claim,
        "m03_profile": m03_claim,
        "m08_profile": m08_claim,
        "m09_counterexample": m09_claim,
        "detectability_exploitability_relation": relation_claim,
        "D_METHOD_CONDITIONAL": method_claim,
    }


def build_diagnostic_sensitivity(evidence: dict[str, Any]) -> dict[str, Any]:
    diagnostic = evidence["diagnostic"]
    table = diagnostic["by_mechanism"].set_index(["mechanism", "method"])
    mechanisms: dict[str, Any] = {}
    for mechanism in ("M03", "M04", "M05"):
        mechanisms[mechanism] = {}
        for method in DIAGNOSTIC_METHODS:
            row = table.loc[(mechanism, method)]
            mechanisms[mechanism][method] = {
                "detectability": float(row["diagnostic_normalized_ap"]),
                "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"]),
            }
    profile = diagnostic["profiles"].set_index("mechanism")
    return {
        "status": "DESCRIPTIVE_ONLY",
        "method_count": 4,
        "expected_cells": 22000,
        "successful_cells": 22000,
        "completion_rate": 1.0,
        "primary_method": "mutual_information",
        "methods": DIAGNOSTIC_METHODS,
        "mechanisms": mechanisms,
        "robustness_profiles": {
            mechanism: {
                "between_diagnostic_range": float(profile.loc[mechanism, "between_diagnostic_range"]),
                "best_evaluated_detectability": float(
                    profile.loc[mechanism, "best_evaluated_diagnostic"]
                ),
                "worst_evaluated_detectability": float(
                    profile.loc[mechanism, "worst_evaluated_diagnostic"]
                ),
            }
            for mechanism in ("M03", "M04", "M05")
        },
        "best_method_interpretation": "optimistic labeled benchmark summary; explicitly non-deployable and not a zero-shot selector",
    }


def build_document(
    evidence: dict[str, Any], *, generated_at: str | None = None,
) -> dict[str, Any]:
    paths: EvidencePaths = evidence["paths"]
    protocol = evidence["config"]["protocol"]
    natural = evidence["natural"]
    auxiliary_statistics = [
        paths.statistics_dir / name
        for name in (
            "category_summary.csv",
            "model_summary.csv",
            "detectability_category_summary.csv",
            "model_vs_lr_contrasts.csv",
            "mechanism_model_summary.csv",
            "mechanism_model_dispersion.csv",
            "strength_dose_response.csv",
            "secondary_integrity.json",
        )
    ]
    all_inputs = [
        paths.canonical, paths.canonical_manifest, paths.config, paths.protocol_freeze,
        paths.tabm_bundle_freeze, paths.diagnostic_freeze, paths.diagnostic_amendment_freeze,
        paths.statistical_amendment_freeze,
        ROOT / "results/corrected_v2/secondary_analysis_protocol_freeze.json",
        paths.diagnostic_canonical_manifest, paths.natural_freeze,
        paths.natural_public_manifest, paths.superseded,
        paths.local_environment, paths.tabm_environment, paths.requirements,
        paths.m10_config, paths.m10_freeze, paths.m10_cpu, paths.m10_cpu_manifest,
        paths.m10_tabm, paths.m10_tabm_manifest,
        ROOT / "results/corrected_v2/core_cpu_cells_manifest.json",
        ROOT / "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_manifest.json",
        paths.diagnostic_cells, paths.natural_statistics, paths.natural_cells, paths.natural_tasks,
        evidence["diagnostic_amendment"]["raw_path"],
        evidence["diagnostic_amendment"]["raw_manifest_path"],
        *evidence["statistics"]["paths"].values(),
        *auxiliary_statistics,
        *evidence["diagnostic"]["paths"].values(),
        ROOT / "scripts/build_canonical_corrected_v2.py",
        ROOT / "scripts/build_diagnostic_rng_amendment.py",
        ROOT / "scripts/build_public_natural_provenance.py",
        ROOT / "experiments/leakbench/run_m10_amendment.py",
        ROOT / "scripts/analyze_corrected_v2.py",
        ROOT / "scripts/analyze_detectability_v2.py",
        ROOT / "scripts/analyze_model_contrasts_v2.py",
        ROOT / "scripts/analyze_secondary_v2.py",
        ROOT / "scripts/analyze_corrected_v2_amendment.py",
        ROOT / "scripts/analyze_cluster_sensitivity_v2.py",
        ROOT / "scripts/analyze_cluster_sensitivity_amendment_v2.py",
        ROOT / "scripts/analyze_natural_case_studies.py",
        ROOT / "scripts/analyze_diagnostic_suite.py",
    ]
    if len({relative(path) for path in all_inputs}) != len(all_inputs):
        raise ValueError("Duplicate claim provenance input")
    for path in all_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    document = {
        "schema_version": 1,
        "generated_at_utc": generated_at or datetime.now(timezone.utc).isoformat(),
        "evidence_tier": "confirmatory",
        "release_status": "CLAIM_STATE_DERIVED",
        "protocol_integrity": {
            "dataset_count": int(protocol["dataset_count"]),
            "mechanism_count": len(protocol["mechanisms"]),
            "strength_count": len(protocol["strengths"]),
            "model_count": len(protocol["core_models"]),
            "seed_count": len(protocol["seeds"]),
            "expected_cells": int(protocol["expected_model_training_cells"]),
            "successful_cells": int(protocol["expected_model_training_cells"]),
            "completion_rate": 1.0,
            "models": sorted(protocol["core_models"]),
        },
        "protocol_amendments": {
            "M10": {
                "version": "m10_strict_mask_v1",
                "replacement_cells": 2500,
                "strict_policy": "task.X[:, ~task.leakage_mask]",
                "original_m10_rows_accepted": False,
                "protocol_freeze_sha256": sha256(paths.m10_freeze),
            },
            "diagnostic_rng": {
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
                "protocol_freeze_sha256": sha256(paths.diagnostic_amendment_freeze),
                "canonical_manifest_sha256": sha256(paths.diagnostic_canonical_manifest),
            },
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
                "protocol_freeze_sha256": sha256(paths.statistical_amendment_freeze),
            },
            "natural_preprocessing": {
                "version": "natural_trainfit_categories_v2",
                "fit_scope": "training rows only",
                "unseen_category_policy": "map_to_reserved_unknown_value",
                "globally_encoded_date_strings_accepted": False,
                "superseded_full_table_category_vocabulary_accepted": False,
                "replacement_cells": 60,
                "protocol_freeze_sha256": sha256(paths.natural_freeze),
            },
        },
        "claim_policy": {
            "primary_metric": "paired_auroc_harm",
            "primary_detectability": "train_only_mutual_information",
            "familywise_alpha": ALPHA,
            "only_thresholded_main_claim": "simple_vs_structured",
            "simple_vs_structured_support_rule": "exact_holm_p_lte_0.05_and_ci_low_gt_0",
            "profile_thresholds": None,
            "global_d_x_status_cap": "DESCRIPTIVE_ONLY",
            "mechanism_profile_status_cap": "DESCRIPTIVE_ONLY",
            "diagnostic_method_status_cap": "DESCRIPTIVE_ONLY",
            "model_specific_contrasts_status_cap": "DESCRIPTIVE_ONLY",
            "natural_scope_cap": "CASE_STUDY_ONLY",
        },
        "claims": derive_claims(evidence),
        "diagnostic_sensitivity": build_diagnostic_sensitivity(evidence),
        "natural": {
            "status": "CASE_STUDY_ONLY",
            "task_count": 5, "model_count": 4,
            "all_task_effects_positive": bool(natural["all_task_effects_positive"]),
            "mean_paired_harm": float(natural["mean_paired_harm"]),
            "bootstrap_ci_low": float(natural["task_bootstrap_ci"][0]),
            "bootstrap_ci_high": float(natural["task_bootstrap_ci"][1]),
            "exact_sign_flip_p": float(natural["exact_two_sided_sign_flip_p"]),
            "interpretation": "five fixed real-data case studies with train-fitted categorical preprocessing; no population-level dataset inference",
            "allowed_wording": "All five fixed case studies had positive observed effects; this remains a case-study-only description.",
            "prohibited_wording": "The effect generalizes to real-world datasets as a population.",
        },
        "pending": {
            "metadata": {"status": "PENDING", "paper_scope": "excluded from abstract and contributions"},
            "governance": {"status": "PENDING", "paper_scope": "excluded from abstract and contributions"},
        },
        "provenance": {
            "input_sha256": {relative(path): sha256(path) for path in all_inputs},
            "generator": {
                "path": relative(Path(__file__)), "sha256": sha256(Path(__file__)),
            },
            "superseded_manifest": relative(paths.superseded),
            "rule": "Only the listed corrected_v2 inputs may supply paper numbers; superseded evidence is deny-listed.",
        },
    }
    if set(document["claims"]) != MAIN_CLAIM_IDS:
        raise RuntimeError("Internal claim schema drift")
    return document


def write_identical(document: dict[str, Any], outputs: list[Path], force: bool) -> None:
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for output in outputs:
        if output.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite claim evidence without --force: {output}")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", default="results/corrected_v2/canonical_cells.csv")
    parser.add_argument("--canonical-manifest", default="results/corrected_v2/canonical_manifest.json")
    parser.add_argument("--statistics-dir", default="results/corrected_v2/statistics")
    parser.add_argument("--diagnostic-cells", default="results/corrected_v2/diagnostic_canonical_cells.csv")
    parser.add_argument(
        "--diagnostic-canonical-manifest",
        default="results/corrected_v2/diagnostic_canonical_cells.manifest.json",
    )
    parser.add_argument(
        "--diagnostic-amendment-freeze",
        default="results/corrected_v2/diagnostic_rng_amendment_freeze.json",
    )
    parser.add_argument(
        "--statistical-amendment-freeze",
        default="results/corrected_v2/statistical_amendment_protocol_v2_freeze.json",
    )
    parser.add_argument("--diagnostic-statistics-dir", default="results/corrected_v2/statistics")
    parser.add_argument("--output", default="results/corrected_v2/paper_claims.json")
    parser.add_argument("--claim-state-output", default="results/corrected_v2/claim_state.json")
    parser.add_argument("--schema-check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.schema_check_only:
        validate_statistics_schema((ROOT / args.statistics_dir) if not Path(args.statistics_dir).is_absolute() else Path(args.statistics_dir))
        diagnostic_dir = (ROOT / args.diagnostic_statistics_dir) if not Path(args.diagnostic_statistics_dir).is_absolute() else Path(args.diagnostic_statistics_dir)
        validate_diagnostic_statistics_schema(diagnostic_dir)
        print(json.dumps({"status": "SCHEMA_VALID", "claims_written": False}, indent=2))
        return 0

    paths = default_paths(
        canonical=args.canonical, canonical_manifest=args.canonical_manifest,
        statistics_dir=args.statistics_dir, diagnostic_cells=args.diagnostic_cells,
        diagnostic_canonical_manifest=args.diagnostic_canonical_manifest,
        diagnostic_amendment_freeze=args.diagnostic_amendment_freeze,
        statistical_amendment_freeze=args.statistical_amendment_freeze,
        diagnostic_statistics_dir=args.diagnostic_statistics_dir,
    )
    evidence = collect_evidence(paths)
    document = build_document(evidence)
    outputs = [ROOT / args.output, ROOT / args.claim_state_output]
    write_identical(document, outputs, args.force)
    print(json.dumps({
        "status": document["release_status"],
        "paper_claims": relative(outputs[0]), "claim_state": relative(outputs[1]),
        "paper_claims_sha256": sha256(outputs[0]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
