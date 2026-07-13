#!/usr/bin/env python3
"""Synchronized cluster sensitivity amendment for M08 and M09.

M08 resamples one entity draw for every model and strength belonging to the
same dataset/seed task.  M09 resamples one source-category draw for every model
belonging to the same dataset/seed/strength task.  Because M09's eight source
IDs are deliberately constructed categories rather than a sampled source
population, its interval is explicitly descriptive reweighting evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PREDICTION_ARRAYS = {
    "row_id", "y", "clean_probability", "full_probability", "entity_id", "source_id",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def stable_seed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *parts], separators=(",", ":"), ensure_ascii=True)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def auc_columns(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Tie-correct binary AUC for one or more score columns."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, None]
    if y.ndim != 1 or scores.ndim != 2 or len(y) != len(scores):
        raise ValueError("AUC inputs have incompatible shapes")
    positive = y == 1
    negative = y == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if n_positive == 0 or n_negative == 0 or n_positive + n_negative != len(y):
        raise ValueError("AUC labels must be binary and contain both classes")
    ranks = rankdata(scores, method="average", axis=0)
    numerator = ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2.0
    return np.asarray(numerator / (n_positive * n_negative), dtype=float)


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_PREDICTION_ARRAYS - set(archive.files))
        if missing:
            raise ValueError(f"Prediction archive {relative(path)} is missing {missing}")
        arrays = {name: np.asarray(archive[name]).copy() for name in REQUIRED_PREDICTION_ARRAYS}
    lengths = {len(value) for value in arrays.values() if value.ndim == 1}
    if len(lengths) != 1 or any(value.ndim != 1 for value in arrays.values()):
        raise ValueError(f"Prediction archive {relative(path)} does not contain aligned vectors")
    if not np.isfinite(arrays["y"].astype(float)).all():
        raise ValueError(f"Prediction labels are non-finite: {relative(path)}")
    for field in ("clean_probability", "full_probability"):
        if not np.isfinite(arrays[field].astype(float)).all():
            raise ValueError(f"Prediction scores are non-finite: {relative(path)}")
    if set(np.unique(arrays["y"]).tolist()) != {0, 1}:
        raise ValueError(f"Prediction labels are not binary with both classes: {relative(path)}")
    return arrays


def prediction_index(directories: list[Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    collisions: list[str] = []
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.npz")):
            if path.stem in paths:
                collisions.append(path.stem)
            else:
                paths[path.stem] = path
    if collisions:
        raise ValueError(f"Prediction run-ID collisions across input directories: {sorted(set(collisions))[:5]}")
    return paths


def validate_shared_task(
    rows: pd.DataFrame,
    paths: dict[str, Path],
    cluster_field: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a task group and prove that one row/cluster draw is shareable."""
    reference: dict[str, np.ndarray] | None = None
    clean_columns: list[np.ndarray] = []
    full_columns: list[np.ndarray] = []
    expected_effects: list[float] = []
    for row in rows.sort_values(["strength", "model"]).itertuples(index=False):
        run_id = str(row.run_id)
        prediction = load_prediction(paths[run_id])
        if reference is None:
            reference = prediction
        else:
            for field in ("row_id", "y", cluster_field):
                if not np.array_equal(prediction[field], reference[field]):
                    raise ValueError(
                        f"Shared-draw invariant failed for {run_id}: {field} differs within task group"
                    )
        clean_columns.append(prediction["clean_probability"].astype(float))
        full_columns.append(prediction["full_probability"].astype(float))
        expected_effects.append(float(row.paired_harm))
    assert reference is not None
    clean = np.column_stack(clean_columns)
    full = np.column_stack(full_columns)
    observed = auc_columns(reference["y"], full) - auc_columns(reference["y"], clean)
    if not np.allclose(observed, expected_effects, rtol=0.0, atol=5e-10):
        maximum = float(np.max(np.abs(observed - np.asarray(expected_effects))))
        raise ValueError(f"Prediction/canonical paired-harm mismatch; maximum absolute error={maximum}")
    return reference["y"], clean, full, reference[cluster_field]


def synchronized_inner_effects(
    y: np.ndarray,
    clean: np.ndarray,
    full: np.ndarray,
    clusters: np.ndarray,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    """Apply each sampled cluster vector to every score column, then average."""
    levels = np.unique(clusters)
    if len(levels) < 2:
        raise ValueError("Cluster sensitivity requires at least two cluster levels")
    row_groups = [np.flatnonzero(clusters == level) for level in levels]
    rng = np.random.RandomState(seed)
    effects = np.empty(repetitions, dtype=float)
    accepted = 0
    attempts = 0
    while accepted < repetitions and attempts < repetitions * 20:
        attempts += 1
        selected = rng.randint(0, len(levels), size=len(levels))
        rows = np.concatenate([row_groups[index] for index in selected])
        if len(np.unique(y[rows])) < 2:
            continue
        column_effects = auc_columns(y[rows], full[rows]) - auc_columns(y[rows], clean[rows])
        effects[accepted] = float(column_effects.mean())
        accepted += 1
    if accepted != repetitions:
        raise ValueError(f"Could not obtain {repetitions} valid synchronized cluster draws")
    return effects


def outer_interval(
    inner: dict[tuple[Any, ...], np.ndarray],
    datasets: list[str],
    seeds: list[int],
    strengths: list[str] | None,
    repetitions: int,
    seed: int,
) -> list[float]:
    """Dataset/seed bootstrap over synchronized inner task effects."""
    inner_repetitions = len(next(iter(inner.values())))
    rng = np.random.RandomState(seed)
    outer = np.empty(repetitions, dtype=float)
    if strengths is None:
        tensor = np.stack([
            inner[(dataset, selected_seed)]
            for dataset in datasets for selected_seed in seeds
        ]).reshape(len(datasets), len(seeds), inner_repetitions)
        for repetition in range(repetitions):
            dataset_draw = rng.randint(0, len(datasets), size=len(datasets))
            seed_draw = rng.randint(0, len(seeds), size=(len(datasets), len(seeds)))
            inner_draw = rng.randint(0, inner_repetitions, size=(len(datasets), len(seeds)))
            outer[repetition] = float(tensor[dataset_draw[:, None], seed_draw, inner_draw].mean())
    else:
        tensor = np.stack([
            inner[(dataset, selected_seed, strength)]
            for dataset in datasets for selected_seed in seeds for strength in strengths
        ]).reshape(len(datasets), len(seeds), len(strengths), inner_repetitions)
        for repetition in range(repetitions):
            dataset_draw = rng.randint(0, len(datasets), size=len(datasets))
            seed_draw = rng.randint(0, len(seeds), size=(len(datasets), len(seeds)))
            inner_draw = rng.randint(
                0, inner_repetitions, size=(len(datasets), len(seeds), len(strengths))
            )
            selected = tensor[
                dataset_draw[:, None, None],
                seed_draw[:, :, None],
                np.arange(len(strengths))[None, None, :],
                inner_draw,
            ]
            outer[repetition] = float(selected.mean())
    return [float(np.quantile(outer, 0.025)), float(np.quantile(outer, 0.975))]


def validate_cells(
    frame: pd.DataFrame, config: dict[str, Any], namespace: str
) -> pd.DataFrame:
    required = {
        "run_id", "dataset_id", "dataset_namespace", "mechanism", "strength",
        "model", "seed", "status", "paired_harm",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical input is missing columns: {missing}")
    selected = frame[
        (frame["dataset_namespace"].astype(str) == namespace)
        & frame["mechanism"].astype(str).isin(["M08", "M09"])
    ].copy()
    protocol = config["protocol"]
    datasets = int(protocol["dataset_count"])
    strengths = list(map(str, protocol["strengths"]))
    models = list(map(str, protocol["core_models"]))
    seeds = list(map(int, protocol["seeds"]))
    expected = datasets * 2 * len(strengths) * len(models) * len(seeds)
    key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    if (
        len(selected) != expected
        or selected.duplicated(key).any()
        or set(selected["status"].astype(str)) != {"SUCCESS"}
        or selected["dataset_id"].nunique() != datasets
        or set(selected["strength"].astype(str)) != set(strengths)
        or set(selected["model"].astype(str)) != set(models)
        or set(map(int, selected["seed"].unique())) != set(seeds)
        or selected["run_id"].astype(str).duplicated().any()
    ):
        raise ValueError(f"M08/M09 canonical prediction matrix is incomplete: {len(selected)}/{expected}")
    if not np.isfinite(pd.to_numeric(selected["paired_harm"], errors="coerce")).all():
        raise ValueError("Canonical paired harms contain non-finite values")
    return selected


def analyze(
    cells: pd.DataFrame,
    paths: dict[str, Path],
    config: dict[str, Any],
    inner_repetitions: int,
    outer_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    protocol = config["protocol"]
    models = sorted(map(str, protocol["core_models"]))
    strengths = list(map(str, protocol["strengths"]))
    datasets = sorted(cells["dataset_id"].astype(str).unique())
    seeds = sorted(map(int, protocol["seeds"]))
    analyses: dict[str, Any] = {}

    m08 = cells[cells["mechanism"] == "M08"].copy()
    inner_m08: dict[tuple[Any, ...], np.ndarray] = {}
    entity_counts: list[int] = []
    for (dataset, task_seed), rows in m08.groupby(["dataset_id", "seed"], sort=True):
        if len(rows) != len(models) * len(strengths):
            raise ValueError(f"M08 task group is incomplete: {(dataset, task_seed)}")
        if set(rows["model"].astype(str)) != set(models) or set(rows["strength"].astype(str)) != set(strengths):
            raise ValueError(f"M08 task group factors changed: {(dataset, task_seed)}")
        y, clean, full, entities = validate_shared_task(rows, paths, "entity_id")
        entity_counts.append(len(np.unique(entities)))
        key = (str(dataset), int(task_seed))
        inner_m08[key] = synchronized_inner_effects(
            y, clean, full, entities, inner_repetitions,
            stable_seed(seed, "M08", *key),
        )
    m08_interval = outer_interval(
        inner_m08, datasets, seeds, None, outer_repetitions, stable_seed(seed, "M08", "outer")
    )
    analyses["M08"] = {
        "status": "INFERENTIAL_SENSITIVITY",
        "paired_harm": float(m08["paired_harm"].mean()),
        "synchronized_cluster_ci": m08_interval,
        "interval_type": "95_percentile_hierarchical_cluster_bootstrap",
        "cluster_unit": "entity_id",
        "grouping_key": ["dataset_id", "seed"],
        "shared_draw_scope": ["model", "strength"],
        "shared_cells_per_inner_draw": len(models) * len(strengths),
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
        y, clean, full, sources = validate_shared_task(rows, paths, "source_id")
        source_counts.append(len(np.unique(sources)))
        key = (str(dataset), int(task_seed), str(strength))
        inner_m09[key] = synchronized_inner_effects(
            y, clean, full, sources, inner_repetitions,
            stable_seed(seed, "M09", *key),
        )
    expected_sources = int(config["mechanism_parameters"]["M09"]["n_sources"])
    if set(source_counts) != {expected_sources}:
        raise ValueError(f"M09 designed source-category count changed: {sorted(set(source_counts))}")
    m09_interval = outer_interval(
        inner_m09, datasets, seeds, strengths, outer_repetitions,
        stable_seed(seed, "M09", "outer"),
    )
    analyses["M09"] = {
        "status": "DESCRIPTIVE_DESIGNED_CATEGORY_REWEIGHTING",
        "paired_harm": float(m09["paired_harm"].mean()),
        "descriptive_reweighting_interval": m09_interval,
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
        "schema_version": 2,
        "analysis_version": "synchronized_cluster_sensitivity_v2",
        "evidence_tier": "confirmatory",
        "bootstrap_seed": seed,
        "analyses": analyses,
    }


def write_manifest(
    output: Path,
    canonical_paths: list[Path],
    config_path: Path,
    prediction_dirs: list[Path],
    consumed: pd.DataFrame,
    paths: dict[str, Path],
    result_path: Path,
    inner_repetitions: int,
    outer_repetitions: int,
    seed: int,
) -> None:
    prediction_entries = []
    for row in consumed.sort_values("run_id").itertuples(index=False):
        run_id = str(row.run_id)
        path = paths[run_id]
        prediction_entries.append({
            "run_id": run_id,
            "mechanism": str(row.mechanism),
            "path": relative(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        })
    manifest = {
        "schema_version": 1,
        "analysis_version": "synchronized_cluster_sensitivity_v2",
        "status": "SYNCHRONIZED_CLUSTER_ANALYSIS_COMPLETE",
        "evidence_tier": "confirmatory",
        "analysis_code": {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
        "canonical_inputs": [
            {"path": relative(path), "sha256": sha256(path)} for path in canonical_paths
        ],
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "prediction_directories": [relative(path) for path in prediction_dirs],
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
                "grouping_key": ["dataset_id", "seed"],
                "cluster_unit": "entity_id",
                "shared_draw_scope": ["model", "strength"],
            },
            "M09": {
                "grouping_key": ["dataset_id", "seed", "strength"],
                "cluster_unit": "source_id",
                "shared_draw_scope": ["model"],
                "interpretation": "descriptive_designed_category_reweighting_only",
            },
        },
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", nargs="+", default=["results/corrected_v2/canonical_cells.csv"])
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--prediction-dirs", nargs="+", default=[
        "results/corrected_v2/predictions",
        "results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_predictions",
    ])
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--inner-reps", type=int, default=200)
    parser.add_argument("--outer-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", default="results/corrected_v2/statistics/cluster_sensitivity_v2.json")
    parser.add_argument(
        "--manifest", default="results/corrected_v2/statistics/cluster_sensitivity_v2_manifest.json"
    )
    args = parser.parse_args(argv)
    if args.inner_reps <= 0 or args.outer_reps <= 0:
        raise ValueError("Cluster repetition counts must be positive")

    canonical_paths = [resolve(path) for path in args.canonical]
    config_path = resolve(args.config)
    prediction_dirs = [resolve(path) for path in args.prediction_dirs]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = pd.concat([pd.read_csv(path) for path in canonical_paths], ignore_index=True, sort=False)
    cells = validate_cells(raw, config, args.namespace)
    paths = prediction_index(prediction_dirs)
    missing = sorted(set(cells["run_id"].astype(str)) - set(paths))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required prediction files, e.g. {missing[:3]}")

    result = analyze(cells, paths, config, args.inner_reps, args.outer_reps, args.seed)
    output = resolve(args.output)
    manifest = resolve(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_manifest(
        manifest, canonical_paths, config_path, prediction_dirs, cells,
        paths, output, args.inner_reps, args.outer_reps, args.seed,
    )
    print(json.dumps({
        "status": "SYNCHRONIZED_CLUSTER_ANALYSIS_COMPLETE",
        "output": relative(output),
        "manifest": relative(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
