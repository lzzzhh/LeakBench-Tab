#!/usr/bin/env python3
"""Freeze the second post-audit statistical amendment before final analysis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    task_manifest = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
    bundle_summary = task_manifest.parent / "bundle_summary.json"
    prior_amendment = ROOT / "results/corrected_v2/statistical_amendment_protocol_freeze.json"
    paths = [
        ROOT / "configs/paper/corrected_v2.yaml",
        ROOT / "scripts/analyze_corrected_v2_amendment.py",
        ROOT / "scripts/analyze_cluster_sensitivity_v2.py",
        ROOT / "scripts/analyze_cluster_sensitivity_amendment_v2.py",
        ROOT / "scripts/freeze_statistical_amendment_protocol_v2.py",
        ROOT / "tests/test_statistical_amendment.py",
        ROOT / "tests/test_statistical_amendment_v2.py",
        task_manifest,
        bundle_summary,
        prior_amendment,
        ROOT / "results/corrected_v2/protocol_freeze.json",
        ROOT / "results/corrected_v2/secondary_analysis_protocol_freeze.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze statistical amendment v2; missing {missing}")

    tasks = pd.read_csv(task_manifest)
    identity = ["dataset_id", "mechanism", "strength", "seed"]
    if (
        len(tasks) != 5500
        or tasks.duplicated(identity).any()
        or set(tasks["dataset_namespace"].astype(str)) != {"confirmatory"}
        or tasks["dataset_id"].nunique() != 20
    ):
        raise RuntimeError("The immutable confirmatory task matrix is not exactly 5,500 tasks")
    bundle_hashes = tasks.groupby("bundle_path")["bundle_sha256"].nunique()
    if len(bundle_hashes) != 20 or (bundle_hashes != 1).any():
        raise RuntimeError("The task manifest does not bind exactly 20 immutable bundles")
    cluster_tasks = tasks[tasks["mechanism"].astype(str).isin(["M08", "M09"])]
    if len(cluster_tasks) != 1000:
        raise RuntimeError("The frozen M08/M09 matrix is not exactly 1,000 tasks")

    output = ROOT / "results/corrected_v2/statistical_amendment_protocol_v2_freeze.json"
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 2,
        "status": "FROZEN_BEFORE_FINAL_STATISTICAL_AMENDMENT_V2_ANALYSIS",
        "amendment_id": "statistical_inference_amendment_v2",
        "supersedes": "results/corrected_v2/statistical_amendment_protocol_freeze.json",
        "supersession_scope": "M08_cluster_resampling_and_prediction_lineage_validation",
        "evidence_tier": "confirmatory",
        "discovery_phase": "second_post_unblinding_methodological_audit",
        "decision_thresholds": None,
        "threshold_based_profile_claims_allowed": False,
        "retained_from_v1": {
            "category_contrasts": "exact_two_sided_task_level_sign_flip_with_holm",
            "correlation": "joint_paired_dataset_then_seed_resampling_of_D_and_X_descriptive_only",
            "M09": "descriptive_designed_category_reweighting_only",
        },
        "superseded_v1_method": {
            "M08_cluster_sensitivity": (
                "independent entity draws across five injection seeds reused the same test "
                "rows labels and entities"
            ),
        },
        "cluster_sensitivity": {
            "M08": {
                "status": "DESCRIPTIVE_SYNCHRONIZED_CLUSTER_INTERVAL",
                "cluster_unit": "entity_id",
                "grouping_key": ["dataset_id"],
                "shared_draw_scope": ["seed", "model", "strength"],
                "shared_cells_per_inner_draw": 125,
                "seed_specific_effects_preserved_within_each_entity_draw": True,
                "outer_resampling": "dataset_then_seed_with_one_common_entity_draw_per_dataset_copy",
                "inferential_practical_null_claim_allowed": False,
            },
            "M09": {
                "status": "DESCRIPTIVE_DESIGNED_CATEGORY_REWEIGHTING",
                "cluster_unit": "source_id",
                "grouping_key": ["dataset_id", "seed", "strength"],
                "shared_draw_scope": ["model"],
                "inferential_source_population_claim_allowed": False,
            },
        },
        "prediction_lineage": {
            "frozen_task_count": 1000,
            "frozen_bundle_count": 20,
            "prediction_count": 5000,
            "prediction_arrays_compared_directly": [
                "row_id_to_test_idx", "y", "entity_id_to_entity_ids", "source_id_to_source_ids",
            ],
            "frozen_task_hash_reconstructed_from_bundle": True,
            "prediction_metrics_recomputed": ["clean_auc", "full_auc", "paired_harm"],
        },
        "claim_policy": {
            "simple_vs_structured": (
                "only directional main claim; exact Holm p <= 0.05 and confidence interval low > 0"
            ),
            "M03": "DESCRIPTIVE_ONLY",
            "M08": "DESCRIPTIVE_ONLY",
            "M09": "DESCRIPTIVE_ONLY",
            "detectability_exploitability_relation": "DESCRIPTIVE_ONLY",
            "diagnostic_method_comparison": "DESCRIPTIVE_ONLY",
            "model_specific_contrasts": "DESCRIPTIVE_ONLY",
        },
        "parameters": {
            "bootstrap_repetitions": 20000,
            "permutation_repetitions": 20000,
            "cluster_inner_repetitions_per_dataset": 200,
            "cluster_outer_repetitions": 5000,
            "seed": 20260713,
        },
        "expected_final_cells": 27500,
        "expected_cluster_prediction_cells": 5000,
        "outputs": {
            "category_contrasts": "results/corrected_v2/statistics/category_contrasts_amended.csv",
            "correlation": "results/corrected_v2/statistics/correlation_analysis_amended.json",
            "statistics_manifest": "results/corrected_v2/statistics/statistical_amendment_manifest.json",
            "cluster": "results/corrected_v2/statistics/cluster_sensitivity_v3.json",
            "cluster_manifest": "results/corrected_v2/statistics/cluster_sensitivity_v3_manifest.json",
        },
        "frozen_files": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths
        },
        "bundle_sha256_by_path": {
            path: str(tasks.loc[tasks["bundle_path"] == path, "bundle_sha256"].iloc[0])
            for path in sorted(tasks["bundle_path"].unique())
        },
        "rule": (
            "Any change to a frozen method requires a new amendment version; v1 M08 outputs "
            "and all earlier cluster outputs cannot support paper claims."
        ),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
