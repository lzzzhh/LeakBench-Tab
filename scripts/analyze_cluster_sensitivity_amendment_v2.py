#!/usr/bin/env python3
"""Second cluster amendment: dataset-synchronized M08 entity resampling.

The first synchronized amendment still drew entities independently for the five
injection seeds of one controlled dataset even though those seeds reuse the
same test rows, labels, and entities.  This version applies one entity draw to
all seeds, strengths, and models within a dataset, preserves seed-specific
effects, and only then performs the outer seed bootstrap.

Every consumed prediction archive is also compared directly with the frozen
task bundle's test indices, labels, entity IDs, and source IDs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cluster_sensitivity_v2 import (  # noqa: E402
    auc_columns,
    load_prediction,
    outer_interval,
    prediction_index,
    relative,
    resolve,
    sha256,
    stable_seed,
    synchronized_inner_effects,
    validate_cells,
)


@dataclass(frozen=True)
class FrozenTaskReference:
    row_id: np.ndarray
    y: np.ndarray
    entity_id: np.ndarray
    source_id: np.ndarray
    task_hash: str
    split_hash: str
    bundle_path: str
    bundle_sha256: str


def scientific_key(row: Any) -> tuple[str, str, str, int]:
    return (
        str(row.dataset_id), str(row.mechanism), str(row.strength), int(row.seed)
    )


def frozen_task_sha256(
    base_x: np.ndarray,
    block: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    leakage_mask: np.ndarray,
    entity_ids: np.ndarray,
    source_ids: np.ndarray,
) -> str:
    """Reconstruct the task digest used when the immutable bundle was exported."""
    digest = hashlib.sha256()
    for values in (
        np.concatenate([base_x, block], axis=1),
        y,
        train_idx,
        val_idx,
        test_idx,
        leakage_mask,
        entity_ids,
        source_ids,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_frozen_task_references(
    task_manifest_path: Path,
    config: dict[str, Any],
) -> tuple[dict[tuple[str, str, str, int], FrozenTaskReference], list[dict[str, Any]]]:
    manifest = pd.read_csv(task_manifest_path)
    required = {
        "dataset_id", "dataset_namespace", "mechanism", "strength", "seed",
        "bundle_key", "task_hash", "split_hash", "bundle_path", "bundle_sha256",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Frozen task manifest is missing {missing}")
    selected = manifest[
        (manifest["dataset_namespace"].astype(str) == "confirmatory")
        & manifest["mechanism"].astype(str).isin(["M08", "M09"])
    ].copy()
    protocol = config["protocol"]
    expected = (
        int(protocol["dataset_count"]) * 2 * len(protocol["strengths"])
        * len(protocol["seeds"])
    )
    identity = ["dataset_id", "mechanism", "strength", "seed"]
    if (
        len(selected) != expected
        or selected.duplicated(identity).any()
        or selected["dataset_id"].nunique() != int(protocol["dataset_count"])
        or set(selected["strength"].astype(str)) != set(map(str, protocol["strengths"]))
        or set(map(int, selected["seed"].unique())) != set(map(int, protocol["seeds"]))
    ):
        raise ValueError(f"Frozen M08/M09 task manifest is incomplete: {len(selected)}/{expected}")

    references: dict[tuple[str, str, str, int], FrozenTaskReference] = {}
    bundle_records: list[dict[str, Any]] = []
    for bundle_name, rows in selected.groupby("bundle_path", sort=True):
        hashes = set(rows["bundle_sha256"].astype(str))
        if len(hashes) != 1:
            raise ValueError(f"Frozen bundle has inconsistent hashes: {bundle_name}")
        bundle_path = resolve(str(bundle_name))
        expected_hash = next(iter(hashes))
        if not bundle_path.is_file() or sha256(bundle_path) != expected_hash:
            raise ValueError(f"Frozen bundle hash mismatch: {bundle_name}")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            common_fields = {"base_X", "y", "train_idx", "val_idx", "test_idx"}
            if not common_fields.issubset(bundle.files):
                raise ValueError(f"Frozen bundle lacks common task arrays: {bundle_name}")
            base_x = np.asarray(bundle["base_X"])
            y = np.asarray(bundle["y"])
            train_idx = np.asarray(bundle["train_idx"])
            val_idx = np.asarray(bundle["val_idx"])
            test_idx = np.asarray(bundle["test_idx"])
            split_hash = hashlib.sha256(test_idx.tobytes()).hexdigest()
            if any(str(value) != split_hash for value in rows["split_hash"]):
                raise ValueError(f"Frozen bundle test_idx differs from task manifest: {bundle_name}")
            for row in rows.itertuples(index=False):
                key = str(row.bundle_key)
                block_name = f"block__{key}"
                mask_name = f"leak_mask__{key}"
                entity_name = f"entity_ids__{key}"
                source_name = f"source_ids__{key}"
                required_task_fields = {block_name, mask_name, entity_name, source_name}
                if not required_task_fields.issubset(bundle.files):
                    raise ValueError(f"Frozen bundle lacks task arrays: {bundle_name}/{key}")
                block = np.asarray(bundle[block_name])
                leakage_mask = np.asarray(bundle[mask_name])
                entity = np.asarray(bundle[entity_name])
                source = np.asarray(bundle[source_name])
                if (
                    y.ndim != 1 or test_idx.ndim != 1
                    or entity.shape != y.shape or source.shape != y.shape
                    or test_idx.min() < 0 or test_idx.max() >= len(y)
                ):
                    raise ValueError(f"Frozen prediction reference has invalid shapes: {bundle_name}/{key}")
                reconstructed_task_hash = frozen_task_sha256(
                    base_x, block, y, train_idx, val_idx, test_idx,
                    leakage_mask, entity, source,
                )
                if reconstructed_task_hash != str(row.task_hash):
                    raise ValueError(
                        f"Frozen bundle task_hash differs from task manifest: {bundle_name}/{key}"
                    )
                reference_key = scientific_key(row)
                references[reference_key] = FrozenTaskReference(
                    row_id=test_idx.copy(),
                    y=y[test_idx].copy(),
                    entity_id=entity[test_idx].copy(),
                    source_id=source[test_idx].copy(),
                    task_hash=str(row.task_hash),
                    split_hash=str(row.split_hash),
                    bundle_path=relative(bundle_path),
                    bundle_sha256=expected_hash,
                )
        bundle_records.append({
            "path": relative(bundle_path),
            "sha256": expected_hash,
            "size_bytes": bundle_path.stat().st_size,
        })
    if len(references) != expected:
        raise ValueError("Frozen task reference map is incomplete")
    return references, bundle_records


def validate_prediction_against_reference(
    prediction: dict[str, np.ndarray],
    reference: FrozenTaskReference,
    run_id: str,
) -> None:
    for field in ("row_id", "y", "entity_id", "source_id"):
        if not np.array_equal(prediction[field], getattr(reference, field)):
            raise ValueError(
                f"Prediction archive {run_id} differs from frozen bundle field {field}"
            )


def load_bound_group(
    rows: pd.DataFrame,
    prediction_paths: dict[str, Path],
    references: dict[tuple[str, str, str, int], FrozenTaskReference],
    sort_fields: list[str],
    shared_fields: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    group_reference: dict[str, np.ndarray] | None = None
    clean_columns: list[np.ndarray] = []
    full_columns: list[np.ndarray] = []
    expected_effects: list[float] = []
    column_seeds: list[int] = []
    for row in rows.sort_values(sort_fields).itertuples(index=False):
        run_id = str(row.run_id)
        prediction = load_prediction(prediction_paths[run_id])
        frozen = references[scientific_key(row)]
        validate_prediction_against_reference(prediction, frozen, run_id)
        if group_reference is None:
            group_reference = prediction
        else:
            for field in shared_fields:
                if not np.array_equal(prediction[field], group_reference[field]):
                    raise ValueError(
                        f"Synchronized group invariant failed for {run_id}: {field} differs"
                    )
        clean_columns.append(prediction["clean_probability"].astype(float))
        full_columns.append(prediction["full_probability"].astype(float))
        expected_effects.append(float(row.paired_harm))
        column_seeds.append(int(row.seed))
    assert group_reference is not None
    clean = np.column_stack(clean_columns)
    full = np.column_stack(full_columns)
    clean_auc = auc_columns(group_reference["y"], clean)
    full_auc = auc_columns(group_reference["y"], full)
    expected_clean_auc = pd.to_numeric(rows.sort_values(sort_fields)["clean_auc"]).to_numpy(float)
    expected_full_auc = pd.to_numeric(rows.sort_values(sort_fields)["full_auc"]).to_numpy(float)
    if not np.allclose(clean_auc, expected_clean_auc, rtol=0.0, atol=5e-10):
        raise ValueError("Prediction/canonical clean_auc mismatch")
    if not np.allclose(full_auc, expected_full_auc, rtol=0.0, atol=5e-10):
        raise ValueError("Prediction/canonical full_auc mismatch")
    observed = full_auc - clean_auc
    if not np.allclose(observed, expected_effects, rtol=0.0, atol=5e-10):
        maximum = float(np.max(np.abs(observed - np.asarray(expected_effects))))
        raise ValueError(
            f"Prediction/canonical paired-harm mismatch; maximum absolute error={maximum}"
        )
    return (
        group_reference["y"], clean, full,
        group_reference["entity_id"] if "entity_id" in shared_fields else group_reference["source_id"],
        np.asarray(column_seeds, dtype=int),
    )


def synchronized_inner_effects_by_seed(
    y: np.ndarray,
    clean: np.ndarray,
    full: np.ndarray,
    clusters: np.ndarray,
    column_seeds: np.ndarray,
    seeds: list[int],
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """One cluster draw yields a separate mean effect for every injection seed."""
    levels = np.unique(clusters)
    if len(levels) < 2:
        raise ValueError("M08 requires at least two entity levels")
    masks = [column_seeds == selected_seed for selected_seed in seeds]
    if any(int(mask.sum()) == 0 for mask in masks):
        raise ValueError("M08 synchronized group lacks one or more injection seeds")
    row_groups = [np.flatnonzero(clusters == level) for level in levels]
    rng = np.random.RandomState(seed)
    effects = np.empty((repetitions, len(seeds)), dtype=float)
    accepted = 0
    attempts = 0
    while accepted < repetitions and attempts < repetitions * 20:
        attempts += 1
        selected = rng.randint(0, len(levels), size=len(levels))
        rows = np.concatenate([row_groups[index] for index in selected])
        if len(np.unique(y[rows])) < 2:
            continue
        cell_effects = auc_columns(y[rows], full[rows]) - auc_columns(y[rows], clean[rows])
        for seed_index, mask in enumerate(masks):
            effects[accepted, seed_index] = float(cell_effects[mask].mean())
        accepted += 1
    if accepted != repetitions:
        raise ValueError(f"Could not obtain {repetitions} valid dataset-synchronized draws")
    return effects


def m08_outer_interval(
    inner: dict[str, np.ndarray],
    datasets: list[str],
    seeds: list[int],
    repetitions: int,
    seed: int,
) -> list[float]:
    """Outer hierarchy with one common entity draw per selected dataset copy."""
    tensor = np.stack([inner[dataset] for dataset in datasets])
    if tensor.ndim != 3 or tensor.shape[2] != len(seeds):
        raise ValueError("M08 inner tensor has the wrong dataset/draw/seed shape")
    inner_repetitions = tensor.shape[1]
    rng = np.random.RandomState(seed)
    outer = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        dataset_draw = rng.randint(0, len(datasets), size=len(datasets))
        entity_draw = rng.randint(0, inner_repetitions, size=len(datasets))
        seed_draw = rng.randint(0, len(seeds), size=(len(datasets), len(seeds)))
        selected = tensor[dataset_draw[:, None], entity_draw[:, None], seed_draw]
        outer[repetition] = float(selected.mean())
    return [float(np.quantile(outer, 0.025)), float(np.quantile(outer, 0.975))]


def analyze(
    cells: pd.DataFrame,
    prediction_paths: dict[str, Path],
    references: dict[tuple[str, str, str, int], FrozenTaskReference],
    config: dict[str, Any],
    inner_repetitions: int,
    outer_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    metric_fields = {"clean_auc", "full_auc", "paired_harm"}
    missing_metrics = sorted(metric_fields - set(cells.columns))
    if missing_metrics:
        raise ValueError(f"Canonical cells lack prediction-bound metrics: {missing_metrics}")
    for field in sorted(metric_fields):
        if not np.isfinite(pd.to_numeric(cells[field], errors="coerce")).all():
            raise ValueError(f"Canonical {field} contains non-finite values")
    protocol = config["protocol"]
    models = sorted(map(str, protocol["core_models"]))
    strengths = list(map(str, protocol["strengths"]))
    seeds = sorted(map(int, protocol["seeds"]))
    datasets = sorted(cells["dataset_id"].astype(str).unique())
    analyses: dict[str, Any] = {}

    m08 = cells[cells["mechanism"] == "M08"].copy()
    inner_m08: dict[str, np.ndarray] = {}
    entity_counts: list[int] = []
    for dataset, rows in m08.groupby("dataset_id", sort=True):
        expected = len(seeds) * len(strengths) * len(models)
        if (
            len(rows) != expected
            or set(map(int, rows["seed"].unique())) != set(seeds)
            or set(rows["strength"].astype(str)) != set(strengths)
            or set(rows["model"].astype(str)) != set(models)
        ):
            raise ValueError(f"M08 dataset group is incomplete: {dataset}")
        y, clean, full, entities, column_seeds = load_bound_group(
            rows, prediction_paths, references,
            ["seed", "strength", "model"],
            ("row_id", "y", "entity_id"),
        )
        entity_counts.append(len(np.unique(entities)))
        dataset_name = str(dataset)
        inner_m08[dataset_name] = synchronized_inner_effects_by_seed(
            y, clean, full, entities, column_seeds, seeds, inner_repetitions,
            stable_seed(seed, "M08-amendment-v2", dataset_name),
        )
    analyses["M08"] = {
        "status": "DESCRIPTIVE_SYNCHRONIZED_CLUSTER_INTERVAL",
        "paired_harm": float(m08["paired_harm"].mean()),
        "synchronized_cluster_ci": m08_outer_interval(
            inner_m08, datasets, seeds, outer_repetitions,
            stable_seed(seed, "M08-amendment-v2", "outer"),
        ),
        "interval_type": "95_percentile_hierarchical_cluster_bootstrap",
        "cluster_unit": "entity_id",
        "grouping_key": ["dataset_id"],
        "shared_draw_scope": ["seed", "model", "strength"],
        "seed_effects_preserved_within_each_entity_draw": True,
        "inferential_practical_null_claim_allowed": False,
        "shared_cells_per_inner_draw": len(seeds) * len(models) * len(strengths),
        "cells": len(m08),
        "task_groups": len(inner_m08),
        "datasets": len(datasets),
        "seeds": len(seeds),
        "strengths": len(strengths),
        "models": len(models),
        "cluster_levels_min": min(entity_counts),
        "cluster_levels_max": max(entity_counts),
        "inner_repetitions_per_task_group": inner_repetitions,
        "outer_repetitions": outer_repetitions,
    }

    m09 = cells[cells["mechanism"] == "M09"].copy()
    inner_m09: dict[tuple[Any, ...], np.ndarray] = {}
    source_counts: list[int] = []
    for (dataset, task_seed, strength), rows in m09.groupby(
        ["dataset_id", "seed", "strength"], sort=True
    ):
        if len(rows) != len(models) or set(rows["model"].astype(str)) != set(models):
            raise ValueError(f"M09 task/strength group is incomplete: {(dataset, task_seed, strength)}")
        y, clean, full, sources, _ = load_bound_group(
            rows, prediction_paths, references,
            ["model"], ("row_id", "y", "source_id"),
        )
        source_counts.append(len(np.unique(sources)))
        key = (str(dataset), int(task_seed), str(strength))
        inner_m09[key] = synchronized_inner_effects(
            y, clean, full, sources, inner_repetitions,
            stable_seed(seed, "M09-amendment-v2", *key),
        )
    expected_sources = int(config["mechanism_parameters"]["M09"]["n_sources"])
    if set(source_counts) != {expected_sources}:
        raise ValueError(f"M09 designed source-category count changed: {sorted(set(source_counts))}")
    analyses["M09"] = {
        "status": "DESCRIPTIVE_DESIGNED_CATEGORY_REWEIGHTING",
        "paired_harm": float(m09["paired_harm"].mean()),
        "descriptive_reweighting_interval": outer_interval(
            inner_m09, datasets, seeds, strengths, outer_repetitions,
            stable_seed(seed, "M09-amendment-v2", "outer"),
        ),
        "interval_type": "95_percentile_designed_category_reweighting_interval",
        "cluster_unit": "source_id",
        "source_unit_semantics": "eight_constructed_source_categories_not_a_sampled_source_population",
        "inferential_source_population_claim_allowed": False,
        "grouping_key": ["dataset_id", "seed", "strength"],
        "shared_draw_scope": ["model"],
        "shared_cells_per_inner_draw": len(models),
        "designed_category_count": expected_sources,
        "cells": len(m09),
        "task_strength_groups": len(inner_m09),
        "datasets": len(datasets),
        "seeds": len(seeds),
        "strengths": len(strengths),
        "models": len(models),
        "inner_repetitions_per_task_group": inner_repetitions,
        "outer_repetitions": outer_repetitions,
    }
    return {
        "schema_version": 3,
        "analysis_version": "synchronized_cluster_sensitivity_amendment_v2",
        "evidence_tier": "confirmatory",
        "claim_scope": "DESCRIPTIVE_ONLY",
        "bootstrap_seed": seed,
        "prediction_bundle_fields_verified": [
            "test_idx", "y", "entity_ids", "source_ids", "task_hash",
        ],
        "prediction_metrics_verified": ["clean_auc", "full_auc", "paired_harm"],
        "analyses": analyses,
    }


def write_manifest(
    path: Path,
    canonical_paths: list[Path],
    config_path: Path,
    task_manifest_path: Path,
    bundle_records: list[dict[str, Any]],
    prediction_dirs: list[Path],
    cells: pd.DataFrame,
    prediction_paths: dict[str, Path],
    references: dict[tuple[str, str, str, int], FrozenTaskReference],
    result_path: Path,
    inner_repetitions: int,
    outer_repetitions: int,
    seed: int,
) -> None:
    prediction_entries = []
    for row in cells.sort_values("run_id").itertuples(index=False):
        run_id = str(row.run_id)
        prediction_path = prediction_paths[run_id]
        reference = references[scientific_key(row)]
        prediction_entries.append({
            "run_id": run_id,
            "mechanism": str(row.mechanism),
            "path": relative(prediction_path),
            "sha256": sha256(prediction_path),
            "size_bytes": prediction_path.stat().st_size,
            "task_hash": reference.task_hash,
            "split_hash": reference.split_hash,
            "bundle_path": reference.bundle_path,
            "bundle_sha256": reference.bundle_sha256,
        })
    dependency = ROOT / "scripts/analyze_cluster_sensitivity_v2.py"
    payload = {
        "schema_version": 2,
        "analysis_version": "synchronized_cluster_sensitivity_amendment_v2",
        "status": "SYNCHRONIZED_CLUSTER_ANALYSIS_COMPLETE",
        "evidence_tier": "confirmatory",
        "analysis_code": {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
        "dependency_code": {"path": relative(dependency), "sha256": sha256(dependency)},
        "canonical_inputs": [
            {"path": relative(source), "sha256": sha256(source)} for source in canonical_paths
        ],
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "task_manifest": {
            "path": relative(task_manifest_path), "sha256": sha256(task_manifest_path),
            "verified_fields": [
                "test_idx", "y", "entity_ids", "source_ids", "task_hash",
            ],
        },
        "frozen_bundles": bundle_records,
        "prediction_directories": [relative(directory) for directory in prediction_dirs],
        "consumed_prediction_count": len(prediction_entries),
        "consumed_predictions": prediction_entries,
        "output": {
            "path": relative(result_path),
            "sha256": sha256(result_path),
            "size_bytes": result_path.stat().st_size,
        },
        "parameters": {
            "inner_repetitions_per_task_group": inner_repetitions,
            "outer_repetitions": outer_repetitions,
            "seed": seed,
        },
        "synchronization": {
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
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", nargs="+", default=["results/corrected_v2/canonical_cells.csv"])
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--task-manifest", default="results/corrected_v2/task_bundles/task_manifest.csv")
    parser.add_argument("--prediction-dirs", nargs="+", default=[
        "results/corrected_v2/predictions",
        "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_predictions",
    ])
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--inner-reps", type=int, default=200)
    parser.add_argument("--outer-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", default="results/corrected_v2/statistics/cluster_sensitivity_v3.json")
    parser.add_argument(
        "--manifest", default="results/corrected_v2/statistics/cluster_sensitivity_v3_manifest.json"
    )
    args = parser.parse_args(argv)
    if args.inner_reps <= 0 or args.outer_reps <= 0:
        raise ValueError("Cluster repetition counts must be positive")

    canonical_paths = [resolve(path) for path in args.canonical]
    config_path = resolve(args.config)
    task_manifest_path = resolve(args.task_manifest)
    prediction_dirs = [resolve(path) for path in args.prediction_dirs]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = pd.concat([pd.read_csv(path) for path in canonical_paths], ignore_index=True, sort=False)
    cells = validate_cells(raw, config, args.namespace)
    prediction_paths = prediction_index(prediction_dirs)
    missing = sorted(set(cells["run_id"].astype(str)) - set(prediction_paths))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required prediction files, e.g. {missing[:3]}")
    references, bundle_records = load_frozen_task_references(task_manifest_path, config)

    result = analyze(
        cells, prediction_paths, references, config,
        args.inner_reps, args.outer_reps, args.seed,
    )
    output = resolve(args.output)
    manifest = resolve(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_manifest(
        manifest, canonical_paths, config_path, task_manifest_path, bundle_records,
        prediction_dirs, cells, prediction_paths, references, output,
        args.inner_reps, args.outer_reps, args.seed,
    )
    print(json.dumps({
        "status": "SYNCHRONIZED_CLUSTER_ANALYSIS_COMPLETE",
        "output": relative(output),
        "manifest": relative(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
