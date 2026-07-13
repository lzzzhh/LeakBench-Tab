#!/usr/bin/env python3
"""Hierarchical analysis for the frozen corrected_v2 diagnostic suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}


def _hierarchical_bootstrap(matrix, repetitions, seed):
    """Resample datasets, then seeds, retaining paired methods/mechanisms."""
    rng = np.random.RandomState(seed)
    n_datasets, n_seeds = matrix.shape[:2]
    samples = np.empty((repetitions,) + matrix.shape[2:], dtype=float)
    for repetition in range(repetitions):
        dataset_draw = rng.randint(0, n_datasets, n_datasets)
        selected = []
        for dataset_index in dataset_draw:
            seed_draw = rng.randint(0, n_seeds, n_seeds)
            selected.append(matrix[dataset_index, seed_draw].mean(axis=0))
        samples[repetition] = np.mean(selected, axis=0)
    return samples


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    source = ROOT / args.input
    cells = pd.read_csv(source)
    cells = cells[cells["dataset_namespace"] == args.namespace].copy()
    key = ["dataset_id", "mechanism", "strength", "seed", "method"]
    if cells.empty or cells.duplicated(key).any():
        raise ValueError("diagnostic input is empty or has duplicate scientific identities")
    failures = cells[cells["status"] != "SUCCESS"]
    if args.require_complete and len(failures):
        raise ValueError(f"diagnostic suite contains {len(failures)} failed cells")
    cells = cells[cells["status"] == "SUCCESS"].copy()
    methods = sorted(cells["method"].unique())
    mechanisms = sorted(cells["mechanism"].unique())
    datasets = sorted(cells["dataset_id"].unique())
    seeds = sorted(int(value) for value in cells["seed"].unique())
    expected = len(datasets) * len(mechanisms) * cells["strength"].nunique() * len(seeds) * len(methods)
    if args.require_complete and len(cells) != expected:
        raise ValueError(f"incomplete successful matrix: {len(cells)}/{expected}")

    atomic = cells.groupby(
        ["dataset_id", "seed", "method", "mechanism"], as_index=False
    )["localization_normalized_ap"].mean()
    index = pd.MultiIndex.from_product(
        [datasets, seeds, methods, mechanisms],
        names=["dataset_id", "seed", "method", "mechanism"],
    )
    matrix = (
        atomic.set_index(index.names)["localization_normalized_ap"]
        .reindex(index).to_numpy().reshape(len(datasets), len(seeds), len(methods), len(mechanisms))
    )
    if np.isnan(matrix).any():
        raise ValueError("unbalanced diagnostic matrix after aggregation")
    bootstrap = _hierarchical_bootstrap(matrix, args.repetitions, args.seed)
    point = matrix.mean(axis=(0, 1))

    rows = []
    for method_index, method in enumerate(methods):
        for mechanism_index, mechanism in enumerate(mechanisms):
            values = bootstrap[:, method_index, mechanism_index]
            rows.append({
                "method": method,
                "mechanism": mechanism,
                "category": CATEGORIES[mechanism],
                "diagnostic_normalized_ap": float(point[method_index, mechanism_index]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
            })
    mechanism_method = pd.DataFrame(rows)

    method_rows = []
    for method_index, method in enumerate(methods):
        values = bootstrap[:, method_index].mean(axis=1)
        method_rows.append({
            "method": method,
            "diagnostic_normalized_ap": float(point[method_index].mean()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        })
    method_summary = pd.DataFrame(method_rows)

    # Pre-specified robustness summaries. "Best" is explicitly optimistic and
    # is never presented as a deployable ensemble selected without labels.
    best_point = point.max(axis=0)
    worst_point = point.min(axis=0)
    best_bootstrap = bootstrap.max(axis=1)
    worst_bootstrap = bootstrap.min(axis=1)
    profile_rows = []
    for mechanism_index, mechanism in enumerate(mechanisms):
        best_values = best_bootstrap[:, mechanism_index]
        worst_values = worst_bootstrap[:, mechanism_index]
        best_high = float(np.quantile(best_values, 0.975))
        profile_rows.append({
            "mechanism": mechanism,
            "category": CATEGORIES[mechanism],
            "best_evaluated_diagnostic": float(best_point[mechanism_index]),
            "best_ci_low": float(np.quantile(best_values, 0.025)),
            "best_ci_high": best_high,
            "worst_evaluated_diagnostic": float(worst_point[mechanism_index]),
            "worst_ci_low": float(np.quantile(worst_values, 0.025)),
            "worst_ci_high": float(np.quantile(worst_values, 0.975)),
            "low_across_all_evaluated_diagnostics": bool(best_high < 0.30),
            "between_diagnostic_range": float(best_point[mechanism_index] - worst_point[mechanism_index]),
        })
    profile_summary = pd.DataFrame(profile_rows)

    integrity = {
        "schema_version": 1,
        "evidence_tier": args.namespace,
        "source": str(source.relative_to(ROOT)),
        "rows_success": int(len(cells)),
        "rows_failure": int(len(failures)),
        "expected_cells": int(expected),
        "datasets": len(datasets),
        "mechanisms": mechanisms,
        "methods": methods,
        "strengths": sorted(cells["strength"].unique()),
        "seeds": seeds,
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": args.seed,
        "primary_diagnostic": "mutual_information",
        "robust_low_threshold": 0.30,
        "best_diagnostic_interpretation": "optimistic labeled benchmark summary, not a zero-shot ensemble",
    }
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mechanism_method.to_csv(output / "diagnostic_method_by_mechanism.csv", index=False)
    method_summary.to_csv(output / "diagnostic_method_summary.csv", index=False)
    profile_summary.to_csv(output / "diagnostic_robustness_profiles.csv", index=False)
    (output / "diagnostic_integrity.json").write_text(
        json.dumps(integrity, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(method_summary.to_string(index=False))
    print(profile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
