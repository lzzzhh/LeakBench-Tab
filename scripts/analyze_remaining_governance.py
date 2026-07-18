#!/usr/bin/env python3
"""Analyze natural-case and semantic-group governance sensitivities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "results/edbt_eab_revision"
BOOTSTRAP_SEED = 20260718
BOOTSTRAP_REPS = 5000
PRIMARY_BUDGET = 0.20


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clustered_interval(values_by_cluster, repetitions=BOOTSTRAP_REPS):
    values = np.asarray(list(values_by_cluster.values()), dtype=float)
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    draws = values[rng.randint(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "bootstrap_mean": float(draws.mean()),
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "probability_positive": float(np.mean(draws > 0)),
        "clusters": int(len(values)),
    }


def paired(frame, keys):
    p3 = frame[frame.policy == "P3_blind_mi"].set_index(keys)[
        ["strict_distance_reduction", "initial_gap"]
    ].rename(columns={"strict_distance_reduction": "p3_sdr"})
    p2 = frame[frame.policy == "P2_random"].groupby(keys)["strict_distance_reduction"].mean()
    out = p3.join(p2.rename("p2_mean_sdr"), how="inner").reset_index()
    out["paired"] = out.p3_sdr - out.p2_mean_sdr
    return out


def analyze_natural(path: Path):
    cells = pd.read_csv(path)
    if len(cells) != 315 or set(cells.status) != {"SUCCESS"} or cells.run_id.duplicated().any():
        raise ValueError("natural governance coverage/integrity failure")
    pairs = paired(cells, ["task", "training_seed"])
    task = pairs.groupby("task", as_index=False).agg(
        initial_gap=("initial_gap", "mean"),
        p2_mean_sdr=("p2_mean_sdr", "mean"),
        p3_sdr=("p3_sdr", "mean"),
        paired=("paired", "mean"),
    )
    p3_diag = cells[cells.policy == "P3_blind_mi"].groupby("task").agg(
        p3_leak_recall=("leak_recall", "mean"),
        p3_legit_retention=("legit_retention", "mean"),
    )
    task = task.merge(p3_diag, on="task", validate="one_to_one")
    task.to_csv(REVISION / "natural_governance_summary.csv", index=False)
    effects = dict(zip(task.task, task.paired))
    interval = clustered_interval(effects)
    values = np.asarray(list(effects.values()))
    signs = np.asarray(np.meshgrid(*[[-1.0, 1.0]] * len(values))).T.reshape(-1, len(values))
    observed = abs(values.mean())
    interval["exact_two_sided_sign_flip_p"] = float(
        np.mean(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15)
    )
    interval["all_case_effects_positive"] = bool((values > 0).all())
    interval["task_effects"] = effects
    return interval


def analyze_semantic(path: Path):
    cells = pd.read_csv(path)
    if len(cells) != 10500 or set(cells.status) != {"SUCCESS"} or cells.run_id.duplicated().any():
        raise ValueError("semantic M09 coverage/integrity failure")
    semantic = paired(cells, ["dataset_index", "mechanism", "strength", "training_seed"])

    encoded_cells = pd.read_csv(REVISION / "b1_multiseed_p2.csv")
    encoded_cells = encoded_cells[np.isclose(encoded_cells.budget_fraction, PRIMARY_BUDGET)]
    encoded = paired(encoded_cells, ["dataset_index", "mechanism", "strength", "training_seed"])
    join = ["dataset_index", "mechanism", "strength", "training_seed"]
    encoded_non_m09 = encoded[encoded.mechanism != "M09"]
    recomposed = pd.concat([encoded_non_m09, semantic], ignore_index=True)
    if len(recomposed) != 5500 or recomposed.duplicated(join).any():
        raise ValueError("semantic recomposition did not preserve 5,500 keys")

    rows = []
    payload = {}
    for label, frame in (
        ("encoded_overall", encoded),
        ("semantic_recomposed_overall", recomposed),
        ("encoded_M09", encoded[encoded.mechanism == "M09"]),
        ("semantic_M09", semantic),
    ):
        clusters = frame.groupby("dataset_index").paired.mean().to_dict()
        result = clustered_interval(clusters)
        result["n_keys"] = int(len(frame))
        payload[label] = result
        rows.append({"analysis": label, **result})
    delta = semantic.merge(
        encoded[encoded.mechanism == "M09"][join + ["paired"]],
        on=join, suffixes=("_semantic", "_encoded"), validate="one_to_one",
    )
    delta["paired"] = delta.paired_semantic - delta.paired_encoded
    payload["semantic_minus_encoded_M09"] = clustered_interval(
        delta.groupby("dataset_index").paired.mean().to_dict()
    )
    rows.append({"analysis": "semantic_minus_encoded_M09", **payload["semantic_minus_encoded_M09"]})
    pd.DataFrame(rows).to_csv(REVISION / "semantic_budget_summary.csv", index=False)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", default="results/edbt_eab_revision/natural_governance_cells.csv")
    parser.add_argument("--semantic", default="results/edbt_eab_revision/semantic_m09_cells.csv")
    parser.add_argument("--output", default="results/edbt_eab_revision/remaining_governance_summary.json")
    args = parser.parse_args(argv)
    natural_path, semantic_path = ROOT / args.natural, ROOT / args.semantic
    payload = {
        "schema_version": 1,
        "primary_budget": PRIMARY_BUDGET,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "natural_interpretation": "five fixed case studies; descriptive sensitivity, not population inference",
        "semantic_interpretation": "full-panel recomposition; only M09 changes because only M09 expands one semantic source into multiple encoded columns",
        "natural": analyze_natural(natural_path),
        "semantic": analyze_semantic(semantic_path),
        "input_hashes": {
            "natural_governance_cells": sha256(natural_path),
            "semantic_m09_cells": sha256(semantic_path),
            "encoded_b1": sha256(REVISION / "b1_multiseed_p2.csv"),
        },
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
