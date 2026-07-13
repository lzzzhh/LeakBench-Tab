#!/usr/bin/env python3
"""Freeze the post-audit statistical amendment before final amended analysis."""
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
    paths = [
        ROOT / "configs/paper/corrected_v2.yaml",
        ROOT / "scripts/analyze_corrected_v2_amendment.py",
        ROOT / "scripts/analyze_cluster_sensitivity_v2.py",
        ROOT / "scripts/freeze_statistical_amendment_protocol.py",
        ROOT / "tests/test_statistical_amendment.py",
        task_manifest,
        bundle_summary,
        ROOT / "results/corrected_v2/protocol_freeze.json",
        ROOT / "results/corrected_v2/secondary_analysis_protocol_freeze.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze statistical amendment; missing {missing}")

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

    output = ROOT / "results/corrected_v2/statistical_amendment_protocol_freeze.json"
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FINAL_STATISTICAL_AMENDMENT_ANALYSIS",
        "amendment_id": "statistical_inference_amendment_v1",
        "evidence_tier": "confirmatory",
        "discovery_phase": "post_unblinding_methodological_audit",
        "model_outcomes_read_for_threshold_tuning": False,
        "thresholds_changed": False,
        "superseded_methods": {
            "category_p_values": "bootstrap_tail_area_is_not_a_valid_randomization_p_value",
            "correlation_intervals": "bootstrap_fixed_detectability_and_resampled_only_exploitation",
            "cluster_sensitivity": "independent_per_cell_cluster_draws_broke_shared_task_dependence",
        },
        "category_contrasts": {
            "contrasts": [
                "simple_minus_structured", "simple_minus_boundary", "boundary_minus_structured"
            ],
            "independent_unit": "dataset_task",
            "task_effect": "mean_over_seeds_models_strengths_and_mechanisms_within_each_category",
            "test": "exact_two_sided_task_level_sign_flip",
            "multiplicity": "holm_over_three_declared_category_contrasts",
            "ci": "dataset_then_seed_hierarchical_percentile_bootstrap",
        },
        "correlation": {
            "status": "DESCRIPTIVE_ONLY",
            "ci": "joint_paired_dataset_then_seed_resampling_of_D_and_X",
            "resampling_pairing": "identical_dataset_and_seed_indices_for_both_axes",
        },
        "cluster_sensitivity": {
            "M08": {
                "status": "INFERENTIAL_SENSITIVITY",
                "cluster_unit": "entity_id",
                "grouping_key": ["dataset_id", "seed"],
                "shared_draw_scope": ["model", "strength"],
            },
            "M09": {
                "status": "DESCRIPTIVE_DESIGNED_CATEGORY_REWEIGHTING",
                "cluster_unit": "source_id",
                "grouping_key": ["dataset_id", "seed", "strength"],
                "shared_draw_scope": ["model"],
                "inferential_source_population_claim_allowed": False,
            },
        },
        "parameters": {
            "bootstrap_repetitions": 20000,
            "permutation_repetitions": 20000,
            "cluster_inner_repetitions_per_task_group": 200,
            "cluster_outer_repetitions": 5000,
            "seed": 20260713,
        },
        "expected_final_cells": 27500,
        "expected_cluster_prediction_cells": 5000,
        "outputs": {
            "category_contrasts": "results/corrected_v2/statistics/category_contrasts_amended.csv",
            "correlation": "results/corrected_v2/statistics/correlation_analysis_amended.json",
            "statistics_manifest": "results/corrected_v2/statistics/statistical_amendment_manifest.json",
            "cluster": "results/corrected_v2/statistics/cluster_sensitivity_v2.json",
            "cluster_manifest": "results/corrected_v2/statistics/cluster_sensitivity_v2_manifest.json",
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
        "rule": "Any change to a frozen method requires a new amendment version; the superseded outputs cannot support paper claims.",
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
