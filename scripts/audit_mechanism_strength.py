#!/usr/bin/env python3
"""Audit S1--S5 construction severity on disjoint pilot tasks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_corrected_core import build_mechanism_config
from src.leakbench.datasets import build_panel_task
from src.leakbench.mechanisms import LeakBenchInjector


def standardized_mean_difference(values, y):
    positive = values[y > 0.5]
    negative = values[y <= 0.5]
    variance = 0.5 * (positive.var(ddof=1) + negative.var(ddof=1))
    return abs(float(positive.mean() - negative.mean())) / max(1e-12, float(np.sqrt(variance)))


def source_total_variation(source_ids, y):
    levels = np.unique(source_ids)
    positive = np.array([np.mean(source_ids[y > 0.5] == level) for level in levels])
    negative = np.array([np.mean(source_ids[y <= 0.5] == level) for level in levels])
    return 0.5 * float(np.abs(positive - negative).sum())


def construction_separation(mechanism, task):
    block = task.X[:, task.n_original :]
    leak_columns = task.leakage_mask[task.n_original :]
    block = block[:, leak_columns]
    if mechanism == "M09":
        return source_total_variation(task.source_ids, task.y)
    if mechanism == "M03":
        side = task.X[:, 0] >= np.median(task.X[:, 0])
        return float(np.mean([
            standardized_mean_difference(block[side, 0], task.y[side]),
            standardized_mean_difference(block[~side, 0], task.y[~side]),
        ]))
    if mechanism == "M07":
        covered = task.sample_metadata["m07_covered"]
        return standardized_mean_difference(block[covered, 0], task.y[covered])
    return float(np.mean([standardized_mean_difference(block[:, index], task.y) for index in range(block.shape[1])]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--output", default="results/corrected_v2/pilot_strength_audit.csv")
    args = parser.parse_args(argv)
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rows = []
    for dataset_index in range(3):
        base = build_panel_task(dataset_index, namespace="pilot")
        for mechanism in config["protocol"]["mechanisms"]:
            for strength in config["protocol"]["strengths"]:
                for seed in (13, 42, 2026):
                    task = LeakBenchInjector(seed=seed).inject(
                        base.X,
                        base.y,
                        [build_mechanism_config(mechanism, strength, config, seed)],
                        feature_names=list(base.feature_names),
                        timestamps=base.timestamps,
                        entity_ids=base.entity_ids,
                        split_type="time",
                    )
                    row = {
                        "dataset_id": base.dataset_id,
                        "mechanism": mechanism,
                        "strength": strength,
                        "seed": seed,
                        "separation": construction_separation(mechanism, task),
                        "n_injected": task.n_injected,
                    }
                    if mechanism == "M08":
                        row["mean_future_count"] = float(task.sample_metadata["m08_future_count"].mean())
                    if mechanism == "M09":
                        row["js_divergence"] = float(task.mechanism_params[-1]["js_divergence"])
                    rows.append(row)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    means = frame.groupby(["mechanism", "strength"], sort=False)["separation"].mean().unstack()
    failures = []
    strength_order = config["protocol"]["strengths"]
    for mechanism, values in means.iterrows():
        ordered = values[strength_order].to_numpy()
        if np.any(np.diff(ordered) < -0.02):
            failures.append((mechanism, ordered.tolist()))
    print(means[strength_order].round(4).to_string())
    if failures:
        raise SystemExit(f"Non-monotone construction severity: {failures}")


if __name__ == "__main__":
    main()
