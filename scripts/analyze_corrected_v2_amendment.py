#!/usr/bin/env python3
"""Post-unblinding statistical amendment for corrected_v2.

This script replaces two invalid inferential summaries without changing the
frozen primary analysis script:

* category contrasts use exact task-level sign flips (20 datasets) followed by
  Holm correction over the three declared contrasts; and
* D--X correlation intervals jointly resample detectability and exploitation
  on the same dataset/seed draws.

The complete canonical matrix is mandatory.  Outputs are accompanied by a
hash manifest that binds the canonical input, configuration, analysis code,
and emitted files.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}
CONTRASTS = (
    ("simple", "structured"),
    ("simple", "boundary"),
    ("boundary", "structured"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Interval input must be a finite one-dimensional array")
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def exact_sign_flip_p(values: np.ndarray, *, chunk_size: int = 65_536) -> float:
    """Exact two-sided randomization p-value over independent task effects."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Sign-flip values must be a non-empty finite vector")
    if len(values) > 24:
        raise ValueError("Exact enumeration is intentionally capped at 24 units")
    observed = abs(float(values.mean()))
    total = 1 << len(values)
    extreme = 0
    bits = np.arange(len(values), dtype=np.uint64)
    tolerance = 16.0 * np.finfo(float).eps * max(1.0, observed)
    for start in range(0, total, chunk_size):
        integers = np.arange(start, min(start + chunk_size, total), dtype=np.uint64)
        signs = 2.0 * ((integers[:, None] >> bits) & 1).astype(float) - 1.0
        null = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(null >= observed - tolerance))
    return float(extreme / total)


def category_task_matrix(cells: pd.DataFrame) -> tuple[list[str], list[int], np.ndarray]:
    atomic = cells.groupby(
        ["dataset_id", "seed", "category"], as_index=False, observed=True
    )["paired_harm"].mean()
    datasets = sorted(atomic["dataset_id"].astype(str).unique())
    seeds = sorted(int(value) for value in atomic["seed"].unique())
    index = pd.MultiIndex.from_product(
        [datasets, seeds], names=["dataset_id", "seed"]
    )
    categories = atomic.pivot(
        index=["dataset_id", "seed"], columns="category", values="paired_harm"
    ).reindex(index)
    required = {left for left, _ in CONTRASTS} | {right for _, right in CONTRASTS}
    if required - set(categories.columns) or categories.isna().any().any():
        raise ValueError("Category contrast matrix is incomplete")
    matrix = np.column_stack(
        [categories[left].to_numpy() - categories[right].to_numpy() for left, right in CONTRASTS]
    ).reshape(len(datasets), len(seeds), len(CONTRASTS))
    return datasets, seeds, matrix


def hierarchical_bootstrap(
    matrix: np.ndarray, repetitions: int, seed: int
) -> np.ndarray:
    """Resample datasets, then seeds within each selected dataset."""
    if matrix.ndim != 3 or repetitions <= 0:
        raise ValueError("Hierarchical bootstrap requires a 3-D matrix and positive repetitions")
    n_datasets, n_seeds, n_outputs = matrix.shape
    rng = np.random.RandomState(seed)
    samples = np.empty((repetitions, n_outputs), dtype=float)
    for repetition in range(repetitions):
        dataset_draw = rng.randint(0, n_datasets, size=n_datasets)
        seed_draw = rng.randint(0, n_seeds, size=(n_datasets, n_seeds))
        selected = matrix[dataset_draw[:, None], seed_draw]
        samples[repetition] = selected.mean(axis=(0, 1))
    return samples


def build_category_contrasts(
    cells: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    datasets, _, matrix = category_task_matrix(cells)
    bootstrap = hierarchical_bootstrap(matrix, repetitions, seed)
    task_values = matrix.mean(axis=1)
    raw_p = [exact_sign_flip_p(task_values[:, index]) for index in range(len(CONTRASTS))]
    adjusted = multipletests(raw_p, method="holm")[1]
    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(CONTRASTS):
        low, high = interval(bootstrap[:, index])
        rows.append({
            "contrast": f"{left}_minus_{right}",
            "difference": float(matrix[:, :, index].mean()),
            "ci_low": low,
            "ci_high": high,
            "sign_flip_p": float(raw_p[index]),
            "holm_p": float(adjusted[index]),
            "n_tasks": len(datasets),
            "test_method": "exact_two_sided_task_level_sign_flip",
            "multiplicity_family": "three_declared_category_contrasts_holm",
        })
    return pd.DataFrame(rows)


def category_design(categories: list[str]) -> np.ndarray:
    levels = sorted(set(categories))
    baseline = levels[0]
    return np.column_stack([
        np.ones(len(categories)),
        *[(np.asarray(categories) == level).astype(float) for level in levels if level != baseline],
    ])


def paired_dx_matrices(
    cells: pd.DataFrame, mechanisms: list[str]
) -> tuple[list[str], list[int], np.ndarray, np.ndarray]:
    """Return aligned dataset x seed x mechanism D and X matrices."""
    task_strength = cells.groupby(
        ["dataset_id", "seed", "mechanism", "strength"],
        as_index=False,
        observed=True,
    ).agg(
        detectability=("diagnostic_normalized_ap", "mean"),
        detectability_min=("diagnostic_normalized_ap", "min"),
        detectability_max=("diagnostic_normalized_ap", "max"),
        paired_harm=("paired_harm", "mean"),
    )
    model_range = task_strength["detectability_max"] - task_strength["detectability_min"]
    if float(model_range.max()) > 1e-10:
        raise ValueError("Detectability differs across downstream models for a shared injected task")
    atomic = task_strength.groupby(
        ["dataset_id", "seed", "mechanism"], as_index=False, observed=True
    ).agg(
        detectability=("detectability", "mean"),
        paired_harm=("paired_harm", "mean"),
    )
    datasets = sorted(atomic["dataset_id"].astype(str).unique())
    seeds = sorted(int(value) for value in atomic["seed"].unique())
    index = pd.MultiIndex.from_product(
        [datasets, seeds, mechanisms], names=["dataset_id", "seed", "mechanism"]
    )
    ordered = atomic.set_index(["dataset_id", "seed", "mechanism"]).reindex(index)
    if ordered[["detectability", "paired_harm"]].isna().any().any():
        raise ValueError("Paired D--X matrix is incomplete")
    shape = (len(datasets), len(seeds), len(mechanisms))
    return (
        datasets,
        seeds,
        ordered["detectability"].to_numpy().reshape(shape),
        ordered["paired_harm"].to_numpy().reshape(shape),
    )


def joint_dx_bootstrap(
    detectability: np.ndarray,
    harm: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap correlation while resampling both D and X on identical draws."""
    if detectability.shape != harm.shape or detectability.ndim != 3:
        raise ValueError("D and X must be aligned three-dimensional matrices")
    n_datasets, n_seeds, _ = detectability.shape
    rng = np.random.RandomState(seed)
    spearman = np.empty(repetitions, dtype=float)
    pearson = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        dataset_draw = rng.randint(0, n_datasets, size=n_datasets)
        seed_draw = rng.randint(0, n_seeds, size=(n_datasets, n_seeds))
        d_point = detectability[dataset_draw[:, None], seed_draw].mean(axis=(0, 1))
        x_point = harm[dataset_draw[:, None], seed_draw].mean(axis=(0, 1))
        spearman[repetition] = float(spearmanr(d_point, x_point).statistic)
        pearson[repetition] = float(pearsonr(d_point, x_point).statistic)
    if not np.isfinite(spearman).all() or not np.isfinite(pearson).all():
        raise ValueError("Joint D--X bootstrap produced a non-finite correlation")
    return spearman, pearson


def regression_diagnostics(
    d_point: np.ndarray,
    x_point: np.ndarray,
    mechanisms: list[str],
    permutation_repetitions: int,
    seed: int,
) -> dict[str, float]:
    categories = [CATEGORIES[mechanism] for mechanism in mechanisms]
    base = category_design(categories)
    full = np.column_stack([base, d_point])
    category_model = LinearRegression(fit_intercept=False).fit(base, x_point)
    full_model = LinearRegression(fit_intercept=False).fit(full, x_point)
    category_r2 = float(category_model.score(base, x_point))
    full_r2 = float(full_model.score(full, x_point))

    category_predictions: list[float] = []
    full_predictions: list[float] = []
    for left_out in range(len(mechanisms)):
        keep = np.arange(len(mechanisms)) != left_out
        category_predictions.append(float(
            LinearRegression(fit_intercept=False)
            .fit(base[keep], x_point[keep])
            .predict(base[left_out:left_out + 1])[0]
        ))
        full_predictions.append(float(
            LinearRegression(fit_intercept=False)
            .fit(full[keep], x_point[keep])
            .predict(full[left_out:left_out + 1])[0]
        ))
    category_lomo_r2 = float(r2_score(x_point, category_predictions))
    full_lomo_r2 = float(r2_score(x_point, full_predictions))

    observed_increment = full_r2 - category_r2
    rng = np.random.RandomState(seed)
    null = np.empty(permutation_repetitions, dtype=float)
    categories_array = np.asarray(categories)
    for repetition in range(permutation_repetitions):
        shuffled = d_point.copy()
        for category in sorted(set(categories)):
            indices = np.flatnonzero(categories_array == category)
            shuffled[indices] = rng.permutation(shuffled[indices])
        permuted = np.column_stack([base, shuffled])
        null[repetition] = (
            LinearRegression(fit_intercept=False).fit(permuted, x_point).score(permuted, x_point)
            - category_r2
        )
    permutation_p = float(
        (1 + np.count_nonzero(null >= observed_increment)) / (permutation_repetitions + 1)
    )
    non_simple = np.asarray([category != "simple" for category in categories])
    structured = np.asarray([category == "structured" for category in categories])
    return {
        "excluding_simple_spearman": float(spearmanr(d_point[non_simple], x_point[non_simple]).statistic),
        "within_structured_spearman": float(spearmanr(d_point[structured], x_point[structured]).statistic),
        "category_r2": category_r2,
        "category_plus_detectability_r2": full_r2,
        "incremental_r2": observed_increment,
        "incremental_permutation_p": permutation_p,
        "category_lomo_r2": category_lomo_r2,
        "category_plus_detectability_lomo_r2": full_lomo_r2,
        "incremental_lomo_r2": full_lomo_r2 - category_lomo_r2,
    }


def build_correlation_analysis(
    cells: pd.DataFrame,
    mechanisms: list[str],
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> dict[str, Any]:
    datasets, seeds, d_matrix, x_matrix = paired_dx_matrices(cells, mechanisms)
    d_point = d_matrix.mean(axis=(0, 1))
    x_point = x_matrix.mean(axis=(0, 1))
    bootstrap_spearman, bootstrap_pearson = joint_dx_bootstrap(
        d_matrix, x_matrix, bootstrap_repetitions, seed
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "analysis_version": "joint_paired_dx_bootstrap_v1",
        "status": "DESCRIPTIVE_ONLY",
        "global_spearman": float(spearmanr(d_point, x_point).statistic),
        "global_spearman_ci": interval(bootstrap_spearman),
        "global_pearson": float(pearsonr(d_point, x_point).statistic),
        "global_pearson_ci": interval(bootstrap_pearson),
        "bootstrap_method": "joint_paired_dataset_then_seed_resampling_of_D_and_X",
        "bootstrap_includes_detectability_uncertainty": True,
        "bootstrap_includes_exploitation_uncertainty": True,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_seed": seed,
        "datasets": len(datasets),
        "seeds": len(seeds),
        "mechanisms": len(mechanisms),
        "permutation_repetitions": permutation_repetitions,
        "permutation_method": "within_declared_category_label_permutation_descriptive",
    }
    result.update(regression_diagnostics(
        d_point, x_point, mechanisms, permutation_repetitions, seed + 17
    ))
    return result


def validate_complete_canonical(
    frame: pd.DataFrame, config: dict[str, Any], namespace: str
) -> pd.DataFrame:
    required = {
        "dataset_id", "dataset_namespace", "mechanism", "strength", "model",
        "seed", "status", "paired_harm", "diagnostic_normalized_ap",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical input is missing columns: {missing}")
    selected = frame[frame["dataset_namespace"].astype(str) == namespace].copy()
    protocol = config["protocol"]
    key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    if selected.duplicated(key).any():
        raise ValueError("Duplicate canonical scientific cell keys")
    expected = int(protocol["expected_model_training_cells"])
    if len(selected) != expected or set(selected["status"].astype(str)) != {"SUCCESS"}:
        raise ValueError(f"Canonical matrix is not complete successful confirmatory evidence: {len(selected)}/{expected}")
    exact_factors = {
        "mechanism": set(map(str, protocol["mechanisms"])),
        "strength": set(map(str, protocol["strengths"])),
        "model": set(map(str, protocol["core_models"])),
        "seed": set(map(int, protocol["seeds"])),
    }
    for field, expected_values in exact_factors.items():
        cast = int if field == "seed" else str
        observed = {cast(value) for value in selected[field].unique()}
        if observed != expected_values:
            raise ValueError(f"Canonical factor mismatch for {field}")
    if selected["dataset_id"].nunique() != int(protocol["dataset_count"]):
        raise ValueError("Canonical dataset count changed")
    numeric = selected[["paired_harm", "diagnostic_normalized_ap"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Canonical statistics contain non-finite values")
    selected["category"] = selected["mechanism"].map(CATEGORIES)
    if selected["category"].isna().any():
        raise ValueError("Canonical input contains an unknown mechanism")
    return selected


def write_manifest(
    path: Path,
    canonical_paths: list[Path],
    config_path: Path,
    outputs: list[Path],
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    seed: int,
) -> None:
    payload = {
        "schema_version": 1,
        "analysis_version": "corrected_v2_statistical_amendment_v1",
        "status": "AMENDED_STATISTICS_COMPLETE",
        "evidence_tier": "confirmatory",
        "analysis_code": {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
        "canonical_inputs": [
            {"path": relative(source), "sha256": sha256(source)} for source in canonical_paths
        ],
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "outputs": {
            relative(output): {"sha256": sha256(output), "size_bytes": output.stat().st_size}
            for output in outputs
        },
        "category_contrasts": {
            "independent_unit": "dataset_task",
            "test": "exact_two_sided_task_level_sign_flip",
            "task_count": 20,
            "multiplicity": "holm_over_three_declared_contrasts",
            "ci": "dataset_then_seed_hierarchical_percentile_bootstrap",
        },
        "correlation": {
            "status": "DESCRIPTIVE_ONLY",
            "ci": "joint_paired_dataset_then_seed_resampling_of_D_and_X",
        },
        "bootstrap_repetitions": bootstrap_repetitions,
        "permutation_repetitions": permutation_repetitions,
        "seed": seed,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", nargs="+", required=True)
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--output-dir", default="results/corrected_v2/statistics")
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--permutation-reps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    canonical_paths = [resolve(path) for path in args.canonical]
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repetitions = args.bootstrap_reps or int(config["statistics"]["bootstrap_repetitions"])
    seed = args.seed if args.seed is not None else int(config["statistics"]["bootstrap_seed"])
    if repetitions <= 0 or args.permutation_reps <= 0:
        raise ValueError("Repetition counts must be positive")
    raw = pd.concat([pd.read_csv(path) for path in canonical_paths], ignore_index=True, sort=False)
    cells = validate_complete_canonical(raw, config, args.namespace)
    mechanisms = list(map(str, config["protocol"]["mechanisms"]))

    contrasts = build_category_contrasts(cells, repetitions, seed)
    correlation = build_correlation_analysis(
        cells, mechanisms, repetitions, args.permutation_reps, seed
    )
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contrasts_path = output_dir / "category_contrasts_amended.csv"
    correlation_path = output_dir / "correlation_analysis_amended.json"
    manifest_path = output_dir / "statistical_amendment_manifest.json"
    contrasts.to_csv(contrasts_path, index=False, lineterminator="\n", float_format="%.17g")
    correlation_path.write_text(
        json.dumps(correlation, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        manifest_path,
        canonical_paths,
        config_path,
        [contrasts_path, correlation_path],
        repetitions,
        args.permutation_reps,
        seed,
    )
    print(json.dumps({
        "status": "AMENDED_STATISTICS_COMPLETE",
        "category_contrasts": relative(contrasts_path),
        "correlation": relative(correlation_path),
        "manifest": relative(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
