#!/usr/bin/env python3
"""Hierarchical intervals for corrected_v2 detectability metrics."""
from __future__ import annotations

import argparse
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output-dir", default="results/corrected_v2/statistics")
    args = parser.parse_args(argv)
    cells = pd.read_csv(ROOT / args.core)
    cells = cells[(cells["dataset_namespace"] == args.namespace) & (cells["status"] == "SUCCESS")]
    atomic = cells.groupby(
        ["dataset_id", "seed", "mechanism", "strength"], as_index=False
    ).agg(
        normalized_ap=("diagnostic_normalized_ap", "mean"),
        normalized_ap_range=("diagnostic_normalized_ap", lambda x: float(x.max() - x.min())),
        top5_recall=("top5_recall", "mean"),
    )
    if atomic["normalized_ap_range"].max() > 1e-10:
        raise ValueError("Diagnostic metric changed across downstream models")
    mechanisms = sorted(atomic["mechanism"].unique())
    datasets = sorted(atomic["dataset_id"].unique())
    seeds = sorted(atomic["seed"].unique())
    index = pd.MultiIndex.from_product(
        [datasets, seeds, mechanisms], names=["dataset_id", "seed", "mechanism"]
    )
    aggregated = atomic.groupby(["dataset_id", "seed", "mechanism"])[["normalized_ap", "top5_recall"]].mean()
    matrix = aggregated.reindex(index).to_numpy().reshape(len(datasets), len(seeds), len(mechanisms), 2)
    if np.isnan(matrix).any():
        raise ValueError("Unbalanced diagnostic matrix")
    rng = np.random.RandomState(args.seed)
    bootstrap = np.empty((args.repetitions, len(mechanisms), 2))
    for repetition in range(args.repetitions):
        selected_datasets = rng.randint(0, len(datasets), size=len(datasets))
        draws = []
        for dataset_index in selected_datasets:
            selected_seeds = rng.randint(0, len(seeds), size=len(seeds))
            draws.append(matrix[dataset_index, selected_seeds].mean(axis=0))
        bootstrap[repetition] = np.mean(draws, axis=0)

    rows = []
    for index, mechanism in enumerate(mechanisms):
        point = matrix[:, :, index].mean(axis=(0, 1))
        rows.append({
            "mechanism": mechanism,
            "category": CATEGORIES[mechanism],
            "diagnostic_normalized_ap": float(point[0]),
            "diagnostic_normalized_ap_ci_low": float(np.quantile(bootstrap[:, index, 0], 0.025)),
            "diagnostic_normalized_ap_ci_high": float(np.quantile(bootstrap[:, index, 0], 0.975)),
            "top5_recall": float(point[1]),
            "top5_recall_ci_low": float(np.quantile(bootstrap[:, index, 1], 0.025)),
            "top5_recall_ci_high": float(np.quantile(bootstrap[:, index, 1], 0.975)),
        })
    mechanism_table = pd.DataFrame(rows)
    category_rows = []
    for category in ("simple", "boundary", "structured"):
        indices = [i for i, mechanism in enumerate(mechanisms) if CATEGORIES[mechanism] == category]
        values = bootstrap[:, indices, 0].mean(axis=1)
        category_rows.append({
            "category": category,
            "diagnostic_normalized_ap": float(mechanism_table.loc[mechanism_table["category"] == category, "diagnostic_normalized_ap"].mean()),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        })
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mechanism_table.to_csv(output / "detectability_mechanism_summary.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output / "detectability_category_summary.csv", index=False)
    print(mechanism_table.to_string(index=False))


if __name__ == "__main__":
    main()
