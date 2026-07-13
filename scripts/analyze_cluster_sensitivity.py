#!/usr/bin/env python3
"""Cluster-aware sensitivity intervals for corrected M08 and M09 harms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def cluster_bootstrap_effects(path, cluster_field, repetitions, seed):
    data = np.load(path)
    y = data["y"]
    clean = data["clean_probability"]
    full = data["full_probability"]
    clusters = data[cluster_field]
    levels = np.unique(clusters)
    row_groups = [np.flatnonzero(clusters == level) for level in levels]
    rng = np.random.RandomState(seed)
    effects = []
    attempts = 0
    while len(effects) < repetitions and attempts < repetitions * 10:
        attempts += 1
        selected = rng.randint(0, len(levels), size=len(levels))
        rows = np.concatenate([row_groups[index] for index in selected])
        if len(np.unique(y[rows])) < 2:
            continue
        effects.append(roc_auc_score(y[rows], full[rows]) - roc_auc_score(y[rows], clean[rows]))
    if len(effects) != repetitions:
        raise ValueError(f"Could not produce {repetitions} valid cluster draws for {path}")
    return np.asarray(effects)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", nargs="+", required=True)
    parser.add_argument("--prediction-dirs", nargs="+", required=True)
    parser.add_argument("--inner-reps", type=int, default=200)
    parser.add_argument("--outer-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--output", default="results/corrected_v2/statistics/cluster_sensitivity.json")
    args = parser.parse_args(argv)
    cells = pd.concat([pd.read_csv(ROOT / path) for path in args.core], ignore_index=True, sort=False)
    cells = cells[(cells["dataset_namespace"] == args.namespace) & (cells["status"] == "SUCCESS")]
    cells = cells[cells["mechanism"].isin(["M08", "M09"])].copy()
    prediction_dirs = [ROOT / path for path in args.prediction_dirs]
    paths = {}
    for directory in prediction_dirs:
        for path in directory.glob("*.npz"):
            paths[path.stem] = path
    missing = sorted(set(cells["run_id"].astype(str)) - set(paths))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction files, e.g. {missing[:3]}")

    results = {}
    for mechanism, cluster_field in (("M08", "entity_id"), ("M09", "source_id")):
        subset = cells[cells["mechanism"] == mechanism].copy()
        inner = {}
        for index, row in subset.iterrows():
            inner[index] = cluster_bootstrap_effects(
                paths[str(row["run_id"])], cluster_field, args.inner_reps,
                args.seed + int(str(row["run_id"])[:8], 16) % 1_000_003,
            )
        datasets = sorted(subset["dataset_id"].unique())
        seeds = sorted(subset["seed"].unique())
        rng = np.random.RandomState(args.seed + (8 if mechanism == "M08" else 9))
        outer = np.empty(args.outer_reps)
        for repetition in range(args.outer_reps):
            values = []
            for dataset in rng.choice(datasets, size=len(datasets), replace=True):
                selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
                for selected_seed in selected_seeds:
                    rows = subset[(subset["dataset_id"] == dataset) & (subset["seed"] == selected_seed)]
                    for index in rows.index:
                        values.append(inner[index][rng.randint(args.inner_reps)])
            outer[repetition] = float(np.mean(values))
        results[mechanism] = {
            "paired_harm": float(subset["paired_harm"].mean()),
            "cluster_hierarchical_ci": [float(np.quantile(outer, 0.025)), float(np.quantile(outer, 0.975))],
            "cluster_unit": cluster_field,
            "cells": int(len(subset)),
            "datasets": len(datasets),
            "seeds": len(seeds),
            "inner_repetitions_per_cell": args.inner_reps,
            "outer_repetitions": args.outer_reps,
        }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
