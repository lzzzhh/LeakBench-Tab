#!/usr/bin/env python3
"""Rebuild corrected_v2 claims from canonical per-cell result tables only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests
import yaml


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "M01": "simple",
    "M02": "simple",
    "M03": "boundary",
    "M04": "structured",
    "M05": "structured",
    "M06": "simple",
    "M07": "boundary",
    "M08": "structured",
    "M09": "structured",
    "M10": "simple",
    "M11": "boundary",
}


def hierarchical_mechanism_bootstrap(atomic, mechanisms, repetitions, seed):
    datasets = sorted(atomic["dataset_id"].unique())
    seeds = sorted(atomic["seed"].unique())
    index = pd.MultiIndex.from_product([datasets, seeds, mechanisms], names=["dataset_id", "seed", "mechanism"])
    matrix = (
        atomic.groupby(["dataset_id", "seed", "mechanism"])["paired_harm"]
        .mean()
        .reindex(index)
        .to_numpy()
        .reshape(len(datasets), len(seeds), len(mechanisms))
    )
    if np.isnan(matrix).any():
        raise ValueError("Unbalanced successful matrix; cannot run confirmatory hierarchical bootstrap")
    rng = np.random.RandomState(seed)
    samples = np.empty((repetitions, len(mechanisms)), dtype=float)
    for repetition in range(repetitions):
        dataset_draw = rng.randint(0, len(datasets), size=len(datasets))
        selected = []
        for dataset_index in dataset_draw:
            seed_draw = rng.randint(0, len(seeds), size=len(seeds))
            selected.append(matrix[dataset_index, seed_draw].mean(axis=0))
        samples[repetition] = np.mean(selected, axis=0)
    return matrix, samples


def ci(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def sign_flip_p(values, rng, repetitions=100_000):
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    signs = rng.choice((-1.0, 1.0), size=(repetitions, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1.0 + np.sum(null >= observed)) / (repetitions + 1.0))


def category_design(categories):
    levels = sorted(set(categories))
    baseline = levels[0]
    return np.column_stack([
        np.ones(len(categories)),
        *[(np.asarray(categories) == level).astype(float) for level in levels if level != baseline],
    ])


def regression_diagnostics(mechanism_table, bootstrap_means, mechanisms, bootstrap_seed):
    detectability = mechanism_table.set_index("mechanism").loc[mechanisms, "diagnostic_normalized_ap"].to_numpy()
    harm = mechanism_table.set_index("mechanism").loc[mechanisms, "paired_harm"].to_numpy()
    categories = [CATEGORIES[mechanism] for mechanism in mechanisms]
    base = category_design(categories)
    full = np.column_stack([base, detectability])
    category_model = LinearRegression(fit_intercept=False).fit(base, harm)
    full_model = LinearRegression(fit_intercept=False).fit(full, harm)
    category_r2 = float(category_model.score(base, harm))
    full_r2 = float(full_model.score(full, harm))

    category_predictions = []
    full_predictions = []
    for left_out in range(len(mechanisms)):
        keep = np.arange(len(mechanisms)) != left_out
        category_predictions.append(
            LinearRegression(fit_intercept=False).fit(base[keep], harm[keep]).predict(base[left_out:left_out + 1])[0]
        )
        full_predictions.append(
            LinearRegression(fit_intercept=False).fit(full[keep], harm[keep]).predict(full[left_out:left_out + 1])[0]
        )
    category_lomo_r2 = float(r2_score(harm, category_predictions))
    full_lomo_r2 = float(r2_score(harm, full_predictions))

    rng = np.random.RandomState(bootstrap_seed + 17)
    observed_increment = full_r2 - category_r2
    null = np.empty(20_000)
    for repetition in range(len(null)):
        shuffled = detectability.copy()
        for category in set(categories):
            idx = np.flatnonzero(np.asarray(categories) == category)
            shuffled[idx] = rng.permutation(shuffled[idx])
        permuted = np.column_stack([base, shuffled])
        null[repetition] = LinearRegression(fit_intercept=False).fit(permuted, harm).score(permuted, harm) - category_r2
    permutation_p = float((1 + np.sum(null >= observed_increment)) / (len(null) + 1))

    bootstrap_spearman = np.array([spearmanr(detectability, row).statistic for row in bootstrap_means])
    bootstrap_pearson = np.array([pearsonr(detectability, row).statistic for row in bootstrap_means])
    return {
        "global_spearman": float(spearmanr(detectability, harm).statistic),
        "global_spearman_ci": list(ci(bootstrap_spearman)),
        "global_pearson": float(pearsonr(detectability, harm).statistic),
        "global_pearson_ci": list(ci(bootstrap_pearson)),
        "excluding_simple_spearman": float(spearmanr(
            detectability[[category != "simple" for category in categories]],
            harm[[category != "simple" for category in categories]],
        ).statistic),
        "within_structured_spearman": float(spearmanr(
            detectability[[category == "structured" for category in categories]],
            harm[[category == "structured" for category in categories]],
        ).statistic),
        "category_r2": category_r2,
        "category_plus_detectability_r2": full_r2,
        "incremental_r2": observed_increment,
        "incremental_permutation_p": permutation_p,
        "category_lomo_r2": category_lomo_r2,
        "category_plus_detectability_lomo_r2": full_lomo_r2,
        "incremental_lomo_r2": full_lomo_r2 - category_lomo_r2,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", nargs="+", required=True)
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--output-dir", default="results/corrected_v2/statistics")
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    repetitions = args.bootstrap_reps or int(config["statistics"]["bootstrap_repetitions"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    mechanisms = list(config["protocol"]["mechanisms"])
    strengths = list(config["protocol"]["strengths"])
    seeds = [int(seed) for seed in config["protocol"]["seeds"]]
    expected_models = list(config["protocol"]["core_models"])

    source_paths = [ROOT / path for path in args.core]
    raw = pd.concat([pd.read_csv(path) for path in source_paths], ignore_index=True, sort=False)
    raw = raw[raw["dataset_namespace"] == args.namespace].copy()
    key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    if raw.duplicated(key).any():
        raise ValueError("Duplicate scientific cell keys in canonical inputs")
    successful = raw[raw["status"] == "SUCCESS"].copy()
    if successful.empty:
        raise ValueError("No successful cells")
    successful["category"] = successful["mechanism"].map(CATEGORIES)
    dataset_count = successful["dataset_id"].nunique()
    observed_models = sorted(successful["model"].unique())
    expected_cells = dataset_count * len(mechanisms) * len(strengths) * len(expected_models) * len(seeds)
    if args.require_complete:
        missing_models = set(expected_models) - set(observed_models)
        if missing_models or len(successful) != expected_cells:
            raise ValueError(
                f"Incomplete confirmatory matrix: {len(successful)}/{expected_cells}, missing models={sorted(missing_models)}"
            )

    diagnostic = (
        successful.groupby(["dataset_id", "mechanism", "strength", "seed"], as_index=False)
        .agg(
            diagnostic_ap=("diagnostic_ap", "mean"),
            diagnostic_normalized_ap=("diagnostic_normalized_ap", "mean"),
            diagnostic_range=("diagnostic_normalized_ap", lambda values: float(values.max() - values.min())),
        )
    )
    if diagnostic["diagnostic_range"].max() > 1e-10:
        raise ValueError("Diagnostic scores differ across downstream models for the same injected task")

    atomic = successful[key + ["paired_harm", "category"]].copy()
    matrix, bootstrap = hierarchical_mechanism_bootstrap(atomic, mechanisms, repetitions, bootstrap_seed)
    mechanism_harm = atomic.groupby("mechanism")["paired_harm"].mean()
    mechanism_detectability = diagnostic.groupby("mechanism")["diagnostic_normalized_ap"].mean()
    dataset_mechanism = atomic.groupby(["dataset_id", "mechanism"], as_index=False)["paired_harm"].mean()
    rng = np.random.RandomState(bootstrap_seed + 31)
    p_values = []
    for mechanism in mechanisms:
        p_values.append(sign_flip_p(dataset_mechanism.loc[dataset_mechanism["mechanism"] == mechanism, "paired_harm"], rng))
    adjusted = multipletests(p_values, method="holm")[1]
    mechanism_rows = []
    for index, mechanism in enumerate(mechanisms):
        lower, upper = ci(bootstrap[:, index])
        mechanism_rows.append({
            "mechanism": mechanism,
            "category": CATEGORIES[mechanism],
            "paired_harm": float(mechanism_harm[mechanism]),
            "paired_harm_ci_low": lower,
            "paired_harm_ci_high": upper,
            "diagnostic_normalized_ap": float(mechanism_detectability[mechanism]),
            "sign_flip_p": p_values[index],
            "holm_p": float(adjusted[index]),
        })
    mechanism_table = pd.DataFrame(mechanism_rows)

    category_rows = []
    category_bootstrap = {}
    for category in ("simple", "boundary", "structured"):
        indices = [index for index, mechanism in enumerate(mechanisms) if CATEGORIES[mechanism] == category]
        values = bootstrap[:, indices].mean(axis=1)
        category_bootstrap[category] = values
        lower, upper = ci(values)
        category_rows.append({
            "category": category,
            "paired_harm": float(mechanism_table.loc[mechanism_table["category"] == category, "paired_harm"].mean()),
            "ci_low": lower,
            "ci_high": upper,
            "n_mechanisms": len(indices),
        })
    category_table = pd.DataFrame(category_rows)

    contrast_rows = []
    for left, right in (("simple", "structured"), ("simple", "boundary"), ("boundary", "structured")):
        values = category_bootstrap[left] - category_bootstrap[right]
        lower, upper = ci(values)
        contrast_rows.append({
            "contrast": f"{left}_minus_{right}",
            "difference": float(category_table.set_index("category").loc[left, "paired_harm"] - category_table.set_index("category").loc[right, "paired_harm"]),
            "ci_low": lower,
            "ci_high": upper,
            "bootstrap_two_sided_p": float(2 * min(np.mean(values <= 0), np.mean(values >= 0))),
        })
    contrast_table = pd.DataFrame(contrast_rows)
    contrast_table["holm_p"] = multipletests(contrast_table["bootstrap_two_sided_p"], method="holm")[1]

    model_rows = []
    for model in observed_models:
        values = successful.loc[successful["model"] == model].groupby("dataset_id")["paired_harm"].mean()
        model_rows.append({
            "model": model,
            "paired_harm": float(values.mean()),
            "dataset_sd": float(values.std(ddof=1)),
            "n_datasets": len(values),
        })
    model_table = pd.DataFrame(model_rows)

    correlation = regression_diagnostics(mechanism_table, bootstrap, mechanisms, bootstrap_seed)
    integrity = {
        "schema_version": 1,
        "namespace": args.namespace,
        "source_tables": [str(path.relative_to(ROOT)) for path in source_paths],
        "rows_total": int(len(raw)),
        "rows_success": int(len(successful)),
        "rows_failed_or_invalid": int(len(raw) - len(successful)),
        "expected_cells_for_observed_dataset_count": expected_cells,
        "completion_rate": float(len(successful) / expected_cells),
        "datasets": dataset_count,
        "mechanisms": int(successful["mechanism"].nunique()),
        "strengths": int(successful["strength"].nunique()),
        "models": observed_models,
        "seeds": sorted(int(seed) for seed in successful["seed"].unique()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": bootstrap_seed,
    }

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    mechanism_table.to_csv(output / "mechanism_summary.csv", index=False)
    category_table.to_csv(output / "category_summary.csv", index=False)
    contrast_table.to_csv(output / "category_contrasts.csv", index=False)
    model_table.to_csv(output / "model_summary.csv", index=False)
    (output / "correlation_analysis.json").write_text(json.dumps(correlation, indent=2), encoding="utf-8")
    (output / "integrity_summary.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(integrity, indent=2))
    print("\nCategory summary\n", category_table.to_string(index=False))
    print("\nContrasts\n", contrast_table.to_string(index=False))
    print("\nCorrelation\n", json.dumps(correlation, indent=2))


if __name__ == "__main__":
    main()
