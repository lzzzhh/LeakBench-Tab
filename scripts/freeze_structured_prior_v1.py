#!/usr/bin/env python3
"""Freeze replacement and independent-replication designs before model runs."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakbench.datasets import ARCHETYPES  # noqa: E402
from src.leakbench.structured_prior_protocol import (  # noqa: E402
    build_frozen_task_plan,
    file_sha256,
    load_protocol_config,
)


OUTPUT_DIR = ROOT / "protocols/structured_prior_v1"
REPLACEMENT_CONFIG = ROOT / "configs/paper/structured_prior_replacement_v1.yaml"
REPLICATION_CONFIG = ROOT / "configs/paper/independent_replication_v1.yaml"
REPLACEMENT_PLAN = OUTPUT_DIR / "structured_prior_replacement_v1_tasks.csv"
REPLICATION_PLAN = OUTPUT_DIR / "independent_replication_v1_tasks.csv"
INFERENCE_PROTOCOL = OUTPUT_DIR / "inference_protocol_v1.json"
FREEZE_MANIFEST = OUTPUT_DIR / "freeze_manifest_v1.json"

FROZEN_CODE_AND_CONFIG = (
    "configs/paper/structured_prior_replacement_v1.yaml",
    "configs/paper/independent_replication_v1.yaml",
    "src/leakbench/datasets.py",
    "src/leakbench/mechanisms/__init__.py",
    "src/leakbench/mechanisms/structured_prior_v1.py",
    "src/leakbench/structured_prior_protocol.py",
    "src/leakbench/models/core_models.py",
    "src/leakbench/models/official_tabm.py",
    "scripts/export_structured_prior_v1_tasks.py",
    "experiments/leakbench/run_structured_prior_v1_bundle.py",
    "scripts/freeze_structured_prior_v1.py",
)


def _csv_bytes(frame):
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _validate_fixed_designs(replacement, replication):
    first = replacement["protocol"]
    second = replication["protocol"]
    if first["version"] != "structured_prior_replacement_v1":
        raise RuntimeError("replacement version is not frozen to v1")
    if (
        first["dataset_namespace"] != "confirmatory"
        or int(first["dataset_count"]) != 20
        or set(first["mechanisms"]) != {"M04", "M05", "M08"}
        or len(first["seeds"]) != 5
        or int(first["expected_task_variants"]) != 1500
        or int(first["expected_model_cells"]) != 7500
    ):
        raise RuntimeError("replacement design differs from its required matrix")
    if second["version"] != "independent_replication_v1":
        raise RuntimeError("replication version is not frozen to v1")
    if (
        second["dataset_namespace"] != "independent_replication_v1"
        or int(second["dataset_count"]) != 25
        or list(second["seeds"]) != [13, 2026, 7777]
        or set(second["simple_mechanisms"]) != {"M01", "M02", "M06", "M10"}
        or set(second["structured_mechanisms"]) != {"M04", "M05", "M08", "M09"}
        or len(second["strengths"]) != 5
        or len(second["core_models"]) != 5
        or int(second["expected_task_variants"]) != 3000
        or int(second["expected_model_cells"]) != 15000
    ):
        raise RuntimeError("independent replication differs from its required matrix")
    if set(first["dataset_indices"]) & set(second["dataset_indices"]):
        raise RuntimeError("replacement and replication generator indices overlap")


def _inference_payload(replication_config):
    protocol = replication_config["protocol"]
    statistics = replication_config["statistics"]
    return {
        "schema_version": 1,
        "protocol_version": "independent_replication_v1",
        "status": "FROZEN_BEFORE_ANY_MODEL_RUN",
        "independent_unit": "generator_task",
        "population": {
            "archetypes": list(ARCHETYPES),
            "tasks_per_archetype": 5,
            "task_count": 25,
            "archetype_weights": {name: 0.2 for name in ARCHETYPES},
            "sampling_frame": "preselected_generator_indices_1000_through_1024",
            "dataset_namespace": "independent_replication_v1",
        },
        "cell_metric": {
            "name": "paired_harm",
            "formula": "full_auc - strict_auc",
            "strict_view": "task.X[:, ~task.leakage_mask]",
            "full_view": "task.X",
            "higher_value": "greater_permissive_minus_strict_AUROC_difference",
        },
        "within_task_aggregation": {
            "order": [
                "require all 600 prespecified model cells for the task",
                "arithmetic mean over mechanism, strength, injection seed, and model within each family",
                "task_effect = simple_family_mean - structured_family_mean",
            ],
            "simple_mechanisms": list(protocol["simple_mechanisms"]),
            "structured_mechanisms": list(protocol["structured_mechanisms"]),
            "strengths": list(protocol["strengths"]),
            "seeds": list(protocol["seeds"]),
            "models": list(protocol["core_models"]),
            "cells_per_family_per_task": 300,
            "cells_per_task": 600,
            "cell_weights_within_family": "equal",
        },
        "point_estimate": {
            "formula": "mean_archetype(mean_task(task_effect))",
            "task_weights_within_archetype": "equal",
            "archetype_weights": "equal_0.2_each",
        },
        "confidence_interval": {
            "method": "stratified_nonparametric_percentile_bootstrap",
            "strata": "archetype",
            "resampling": "sample_5_tasks_with_replacement_independently_within_each_of_5_archetypes",
            "statistic_each_draw": "equal_weight_mean_of_the_5_resampled_archetype_means",
            "repetitions": int(statistics["bootstrap_repetitions"]),
            "seed": int(statistics["bootstrap_seed"]),
            "bounds": [0.025, 0.975],
            "interval": "two_sided_95_percent_percentile",
        },
        "sign_test": {
            "method": "scipy.stats.binomtest_exact_two_sided",
            "trials": 25,
            "success": "task_effect > 0",
            "null_success_probability": 0.5,
            "alternative": "two-sided",
            "exact_zero_rule": "any exactly zero task_effect makes primary inference INVALID; no tie deletion",
        },
        "decision": {
            "multiplicity_family_size": 1,
            "adjustment": "none",
            "supported_iff": [
                "all 15000 prespecified model cells are SUCCESS and integrity_verified",
                "lower endpoint of the frozen 95 percent bootstrap CI is greater than 0",
                "exact two-sided binomial sign-test p-value is less than or equal to 0.05",
            ],
            "otherwise_complete": "NOT_SUPPORTED",
            "incomplete_or_invalid": "INCOMPLETE_OR_INVALID; never SUPPORTED",
        },
        "exclusions_and_missingness": {
            "planned_exclusions": [],
            "post_hoc_exclusions": "forbidden",
            "missing_cell": "invalidates the primary inference",
            "failed_cell": "invalidates the primary inference",
            "task_substitution": "forbidden",
            "mechanism_substitution": "forbidden",
            "model_substitution": "forbidden",
            "seed_substitution": "forbidden",
            "imputation": "forbidden",
            "fallback_model_implementation": "forbidden",
        },
        "outcome_access": {
            "before_freeze": "generator identities and hashes only; no fitted-model outcomes",
            "after_run": "apply this protocol without changing thresholds, weights, tests, or task set",
        },
    }


def _entry(path):
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def _build_outputs():
    replacement_config = load_protocol_config(REPLACEMENT_CONFIG)
    replication_config = load_protocol_config(REPLICATION_CONFIG)
    _validate_fixed_designs(replacement_config, replication_config)
    replacement_plan, _ = build_frozen_task_plan(REPLACEMENT_CONFIG)
    replication_plan, _ = build_frozen_task_plan(REPLICATION_CONFIG)
    replacement_bytes = _csv_bytes(replacement_plan)
    replication_bytes = _csv_bytes(replication_plan)
    inference_bytes = _json_bytes(_inference_payload(replication_config))
    return replacement_plan, replication_plan, replacement_bytes, replication_bytes, inference_bytes


def _write_or_check(path, content, check):
    if check:
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"frozen output differs from deterministic regeneration: {path}")
    else:
        if path.exists():
            raise FileExistsError(path)
        path.write_bytes(content)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    replacement, replication, replacement_bytes, replication_bytes, inference_bytes = _build_outputs()
    if not args.check:
        for config in (load_protocol_config(REPLACEMENT_CONFIG), load_protocol_config(REPLICATION_CONFIG)):
            result_path = ROOT / config["protocol"]["result_output"]
            if result_path.exists() or result_path.with_name(f"{result_path.stem}_manifest.json").exists():
                raise RuntimeError("model output exists; protocol cannot claim a pre-run freeze")
    _write_or_check(REPLACEMENT_PLAN, replacement_bytes, args.check)
    _write_or_check(REPLICATION_PLAN, replication_bytes, args.check)
    _write_or_check(INFERENCE_PROTOCOL, inference_bytes, args.check)

    frozen_paths = list(FROZEN_CODE_AND_CONFIG) + [
        str(REPLACEMENT_PLAN.relative_to(ROOT)),
        str(REPLICATION_PLAN.relative_to(ROOT)),
        str(INFERENCE_PROTOCOL.relative_to(ROOT)),
    ]
    files = {}
    for relative in frozen_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = _entry(path)
    unique_replication = replication.drop_duplicates("dataset_index")
    freeze_payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_ANY_MODEL_RUN",
        "protocol_date": "2026-07-13",
        "model_results_observed": False,
        "models_executed": 0,
        "structured_prior": {
            "type": "outcome_independent_constant",
            "value": 0.5,
            "mechanisms": ["M04", "M05", "M08"],
        },
        "replacement": {
            "task_plan_path": str(REPLACEMENT_PLAN.relative_to(ROOT)),
            "task_plan_sha256": file_sha256(REPLACEMENT_PLAN),
            "task_variants": len(replacement),
            "model_cells": int(replacement["expected_model_cells"].sum()),
        },
        "independent_replication": {
            "namespace": "independent_replication_v1",
            "task_plan_path": str(REPLICATION_PLAN.relative_to(ROOT)),
            "task_plan_sha256": file_sha256(REPLICATION_PLAN),
            "task_variants": len(replication),
            "model_cells": int(replication["expected_model_cells"].sum()),
            "generator_tasks": int(unique_replication["dataset_index"].nunique()),
            "archetype_task_counts": {
                name: int((unique_replication["archetype"] == name).sum())
                for name in ARCHETYPES
            },
        },
        "inference_protocol": {
            "path": str(INFERENCE_PROTOCOL.relative_to(ROOT)),
            "sha256": file_sha256(INFERENCE_PROTOCOL),
        },
        "files": files,
    }
    freeze_bytes = _json_bytes(freeze_payload)
    _write_or_check(FREEZE_MANIFEST, freeze_bytes, args.check)
    print(json.dumps({
        "status": freeze_payload["status"],
        "replacement_task_variants": len(replacement),
        "replacement_model_cells": int(replacement["expected_model_cells"].sum()),
        "replication_task_variants": len(replication),
        "replication_model_cells": int(replication["expected_model_cells"].sum()),
        "freeze_manifest_sha256": file_sha256(FREEZE_MANIFEST),
        "check": bool(args.check),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
