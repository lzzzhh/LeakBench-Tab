#!/usr/bin/env python3
"""Model-level paired contrasts for the corrected_v2 canonical matrix."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}


def bootstrap_nested(frame, repetitions, seed):
    datasets = sorted(frame["dataset_id"].unique())
    rng = np.random.RandomState(seed)
    values = np.empty(repetitions)
    by_dataset = {dataset: frame[frame["dataset_id"] == dataset] for dataset in datasets}
    for repetition in range(repetitions):
        draws = []
        for dataset in rng.choice(datasets, size=len(datasets), replace=True):
            block = by_dataset[dataset]
            seeds = sorted(block["seed"].unique())
            selected = rng.choice(seeds, size=len(seeds), replace=True)
            draws.append(np.mean([block.loc[block["seed"] == item, "effect"].mean() for item in selected]))
        values[repetition] = np.mean(draws)
    return values


def sign_flip(dataset_values, seed, repetitions=100_000):
    values = np.asarray(dataset_values, dtype=float)
    rng = np.random.RandomState(seed)
    null = np.abs((rng.choice((-1.0, 1.0), size=(repetitions, len(values))) * values).mean(axis=1))
    observed = abs(values.mean())
    return float((1 + np.sum(null >= observed)) / (repetitions + 1))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", required=True)
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output-dir", default="results/corrected_v2/statistics")
    args = parser.parse_args(argv)
    cells = pd.read_csv(ROOT / args.core)
    cells = cells[(cells["dataset_namespace"] == args.namespace) & (cells["status"] == "SUCCESS")].copy()
    cells["category"] = cells["mechanism"].map(CATEGORIES)
    models = sorted(cells["model"].unique())

    category_rows = []
    raw_p = []
    for index, model in enumerate(models):
        table = cells[cells["model"] == model].groupby(
            ["dataset_id", "seed", "category"], as_index=False
        )["paired_harm"].mean().pivot(index=["dataset_id", "seed"], columns="category", values="paired_harm").reset_index()
        table["effect"] = table["simple"] - table["structured"]
        bootstrap = bootstrap_nested(table[["dataset_id", "seed", "effect"]], args.repetitions, args.seed + index)
        dataset_values = table.groupby("dataset_id")["effect"].mean()
        p_value = sign_flip(dataset_values, args.seed + 100 + index)
        raw_p.append(p_value)
        category_rows.append({
            "model": model,
            "simple_minus_structured": float(table["effect"].mean()),
            "ci_low": float(np.quantile(bootstrap, 0.025)),
            "ci_high": float(np.quantile(bootstrap, 0.975)),
            "sign_flip_p": p_value,
        })
    category_table = pd.DataFrame(category_rows)
    category_table["holm_p"] = multipletests(raw_p, method="holm")[1]

    baseline = cells[cells["model"] == "lr"][[
        "dataset_id", "seed", "mechanism", "strength", "paired_harm"
    ]].rename(columns={"paired_harm": "baseline_harm"})
    model_rows = []
    raw_p = []
    alternatives = [model for model in models if model != "lr"]
    for index, model in enumerate(alternatives):
        current = cells[cells["model"] == model][[
            "dataset_id", "seed", "mechanism", "strength", "paired_harm"
        ]]
        paired = current.merge(
            baseline, on=["dataset_id", "seed", "mechanism", "strength"], validate="one_to_one"
        )
        paired["effect"] = paired["paired_harm"] - paired["baseline_harm"]
        bootstrap = bootstrap_nested(paired[["dataset_id", "seed", "effect"]], args.repetitions, args.seed + 200 + index)
        dataset_values = paired.groupby("dataset_id")["effect"].mean()
        p_value = sign_flip(dataset_values, args.seed + 300 + index)
        raw_p.append(p_value)
        model_rows.append({
            "contrast": f"{model}_minus_lr",
            "difference": float(paired["effect"].mean()),
            "ci_low": float(np.quantile(bootstrap, 0.025)),
            "ci_high": float(np.quantile(bootstrap, 0.975)),
            "sign_flip_p": p_value,
        })
    model_table = pd.DataFrame(model_rows)
    if len(model_table):
        model_table["holm_p"] = multipletests(raw_p, method="holm")[1]
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    category_table.to_csv(output / "simple_structured_by_model.csv", index=False)
    model_table.to_csv(output / "model_vs_lr_contrasts.csv", index=False)
    print(category_table.to_string(index=False))
    print(model_table.to_string(index=False))


if __name__ == "__main__":
    main()
