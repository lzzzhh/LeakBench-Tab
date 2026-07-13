#!/usr/bin/env python3
"""Validate and summarize the five fixed real-data case studies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "results/corrected_v2/natural_cells.csv"
TASKS = ROOT / "results/corrected_v2/natural_task_summary.csv"
OUTPUT = ROOT / "results/corrected_v2/natural_statistics.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cells = pd.read_csv(CELLS)
    tasks = pd.read_csv(TASKS)
    expected_tasks = {"BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"}
    if set(cells["task"]) != expected_tasks or set(tasks["task"]) != expected_tasks:
        raise ValueError("Natural task set changed")
    if len(cells) != 60 or not (cells["status"] == "SUCCESS").all():
        raise ValueError("Natural matrix is incomplete")
    if cells.duplicated(["task", "model", "seed"]).any():
        raise ValueError("Duplicate natural scientific cells")
    if tasks["source_sha256"].nunique() != 5:
        raise ValueError("Natural task lineage hashes are not unique")

    task_effects = cells.groupby("task")["paired_harm"].mean()
    model_effects = cells.groupby("model")["paired_harm"].mean()
    rng = np.random.RandomState(20260713)
    bootstrap = np.empty(20_000)
    values = task_effects.to_numpy()
    for repetition in range(len(bootstrap)):
        bootstrap[repetition] = rng.choice(values, size=len(values), replace=True).mean()
    # Exact two-sided sign-flip test over the five fixed case studies.
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * len(values))).T.reshape(-1, len(values))
    observed = abs(values.mean())
    p_value = float(np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
    payload = {
        "schema_version": 1,
        "interpretation": "fixed real-data case studies; not a population-level dataset sample",
        "cells": len(cells),
        "tasks": len(tasks),
        "models": sorted(cells["model"].unique()),
        "seeds": sorted(int(seed) for seed in cells["seed"].unique()),
        "all_task_effects_positive": bool((task_effects > 0).all()),
        "mean_paired_harm": float(values.mean()),
        "task_bootstrap_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "exact_two_sided_sign_flip_p": p_value,
        "task_effects": {key: float(value) for key, value in task_effects.items()},
        "model_effects": {key: float(value) for key, value in model_effects.items()},
        "diagnostic_normalized_ap": {
            row.task: float(row.diagnostic_normalized_ap) for row in tasks.itertuples()
        },
        "cells_sha256": sha256(CELLS),
        "task_summary_sha256": sha256(TASKS),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
