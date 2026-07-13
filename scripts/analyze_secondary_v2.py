#!/usr/bin/env python3
"""Pre-specified model heterogeneity and strength-response analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import yaml


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}


def hierarchical_bootstrap(matrix, repetitions, seed):
    """Resample datasets and then seeds while retaining paired trailing axes."""
    rng = np.random.RandomState(seed)
    n_datasets, n_seeds = matrix.shape[:2]
    samples = np.empty((repetitions,) + matrix.shape[2:], dtype=float)
    for repetition in range(repetitions):
        dataset_draw = rng.randint(0, n_datasets, n_datasets)
        draws = []
        for dataset_index in dataset_draw:
            seed_draw = rng.randint(0, n_seeds, n_seeds)
            draws.append(matrix[dataset_index, seed_draw].mean(axis=0))
        samples[repetition] = np.mean(draws, axis=0)
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--output-dir", default="results/corrected_v2/statistics")
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    mechanisms = list(config["protocol"]["mechanisms"])
    strengths = list(config["protocol"]["strengths"])
    models = list(config["protocol"]["core_models"])
    seeds = [int(value) for value in config["protocol"]["seeds"]]
    cells = pd.read_csv(ROOT / args.core)
    cells = cells[cells["dataset_namespace"] == args.namespace].copy()
    key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    if cells.empty or cells.duplicated(key).any():
        raise ValueError("canonical input is empty or contains duplicate scientific identities")
    failures = cells[cells["status"] != "SUCCESS"]
    successful = cells[cells["status"] == "SUCCESS"].copy()
    datasets = sorted(successful["dataset_id"].unique())
    expected = len(datasets) * len(mechanisms) * len(strengths) * len(models) * len(seeds)
    if args.require_complete and (len(successful) != expected or len(failures)):
        raise ValueError(f"incomplete canonical matrix: success={len(successful)}/{expected}, failures={len(failures)}")

    # Mechanism x model: average over strengths within the resampling unit.
    mm_atomic = successful.groupby(
        ["dataset_id", "seed", "mechanism", "model"], as_index=False
    )["paired_harm"].mean()
    mm_index = pd.MultiIndex.from_product(
        [datasets, seeds, mechanisms, models],
        names=["dataset_id", "seed", "mechanism", "model"],
    )
    mm_matrix = (
        mm_atomic.set_index(mm_index.names)["paired_harm"].reindex(mm_index).to_numpy()
        .reshape(len(datasets), len(seeds), len(mechanisms), len(models))
    )
    if np.isnan(mm_matrix).any():
        raise ValueError("unbalanced mechanism-model matrix")
    mm_bootstrap = hierarchical_bootstrap(mm_matrix, args.repetitions, args.seed)
    mm_point = mm_matrix.mean(axis=(0, 1))
    mm_rows = []
    for mechanism_index, mechanism in enumerate(mechanisms):
        for model_index, model in enumerate(models):
            values = mm_bootstrap[:, mechanism_index, model_index]
            mm_rows.append({
                "mechanism": mechanism, "category": CATEGORIES[mechanism], "model": model,
                "paired_harm": float(mm_point[mechanism_index, model_index]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            })
    mm_summary = pd.DataFrame(mm_rows)

    dispersion_rows = []
    for mechanism_index, mechanism in enumerate(mechanisms):
        values = np.std(mm_bootstrap[:, mechanism_index, :], axis=1, ddof=0)
        dispersion_rows.append({
            "mechanism": mechanism,
            "category": CATEGORIES[mechanism],
            "between_model_sd": float(np.std(mm_point[mechanism_index], ddof=0)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "min_model": models[int(np.argmin(mm_point[mechanism_index]))],
            "max_model": models[int(np.argmax(mm_point[mechanism_index]))],
            "descriptive_extremes_only": True,
        })
    dispersion = pd.DataFrame(dispersion_rows)

    # Strength response: average models within each paired task, then estimate a
    # linear slope across the five pre-ordered levels. No monotonicity is assumed.
    strength_atomic = successful.groupby(
        ["dataset_id", "seed", "mechanism", "strength"], as_index=False
    )["paired_harm"].mean()
    strength_index = pd.MultiIndex.from_product(
        [datasets, seeds, mechanisms, strengths],
        names=["dataset_id", "seed", "mechanism", "strength"],
    )
    strength_matrix = (
        strength_atomic.set_index(strength_index.names)["paired_harm"].reindex(strength_index).to_numpy()
        .reshape(len(datasets), len(seeds), len(mechanisms), len(strengths))
    )
    if np.isnan(strength_matrix).any():
        raise ValueError("unbalanced strength-response matrix")
    strength_bootstrap = hierarchical_bootstrap(strength_matrix, args.repetitions, args.seed + 1)
    strength_point = strength_matrix.mean(axis=(0, 1))
    x = np.arange(len(strengths), dtype=float)
    x = (x - x.mean()) / x.std(ddof=0)
    denominator = float(np.sum(x ** 2))
    slope_point = np.sum((strength_point - strength_point.mean(axis=1, keepdims=True)) * x, axis=1) / denominator
    centered = strength_bootstrap - strength_bootstrap.mean(axis=2, keepdims=True)
    slope_bootstrap = np.sum(centered * x[None, None, :], axis=2) / denominator
    raw_p = 2.0 * np.minimum(
        np.mean(slope_bootstrap <= 0, axis=0),
        np.mean(slope_bootstrap >= 0, axis=0),
    )
    raw_p = np.minimum(raw_p, 1.0)
    holm = multipletests(raw_p, method="holm")[1]
    dose_rows = []
    for mechanism_index, mechanism in enumerate(mechanisms):
        adjacent = np.diff(strength_point[mechanism_index])
        dose_rows.append({
            "mechanism": mechanism,
            "category": CATEGORIES[mechanism],
            "standardized_strength_slope": float(slope_point[mechanism_index]),
            "ci_low": float(np.quantile(slope_bootstrap[:, mechanism_index], 0.025)),
            "ci_high": float(np.quantile(slope_bootstrap[:, mechanism_index], 0.975)),
            "bootstrap_two_sided_p": float(raw_p[mechanism_index]),
            "holm_p": float(holm[mechanism_index]),
            "positive_adjacent_steps": int(np.sum(adjacent > 0)),
            "total_adjacent_steps": len(adjacent),
            **{f"harm_{strength}": float(strength_point[mechanism_index, index]) for index, strength in enumerate(strengths)},
        })
    dose_response = pd.DataFrame(dose_rows)

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mm_summary.to_csv(output / "mechanism_model_summary.csv", index=False)
    dispersion.to_csv(output / "mechanism_model_dispersion.csv", index=False)
    dose_response.to_csv(output / "strength_dose_response.csv", index=False)
    integrity = {
        "schema_version": 1,
        "evidence_tier": args.namespace,
        "rows_success": int(len(successful)), "rows_failure": int(len(failures)),
        "expected_cells": int(expected), "datasets": len(datasets),
        "mechanisms": mechanisms, "models": models, "strengths": strengths, "seeds": seeds,
        "bootstrap_repetitions": args.repetitions, "bootstrap_seed": args.seed,
        "multiplicity_family": "11 strength slopes with Holm correction",
        "model_extremes": "descriptive only; not pre-selected inferential contrasts",
    }
    (output / "secondary_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
