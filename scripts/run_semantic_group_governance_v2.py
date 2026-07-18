#!/usr/bin/env python3
"""Corrected semantic-group runner after the v1 pre-output baseline join failure."""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from scripts.run_remaining_governance import (
    GOVERNANCE_SEEDS,
    PRIMARY_BUDGET,
    ROOT,
    fit_lr_auc,
    load_synthetic,
    m09_groups,
    random_selection,
    selection_hash,
)


FIELDS = [
    "run_id", "dataset_index", "mechanism", "strength", "training_seed",
    "governance_seed", "policy", "budget_fraction", "budget_groups",
    "removed_group_count", "removed_column_count", "status", "failure_reason",
    "strict_auc", "full_auc", "governed_auc", "initial_gap",
    "strict_distance_reduction", "removed_leak_count", "leak_recall",
    "legit_retention", "selection_group_hash", "selection_column_hash",
]


def encoded_baseline(encoded, dataset_index, strength, training_seed):
    """Use the primary-budget P3 row, which carries the frozen strict/full AUCs."""
    match = encoded[
        (encoded.dataset_index == dataset_index)
        & (encoded.mechanism == "M09")
        & (encoded.strength.astype(str) == str(strength))
        & (encoded.training_seed == training_seed)
        & (encoded.policy == "P3_blind_mi")
        & np.isclose(encoded.budget_fraction, PRIMARY_BUDGET)
    ]
    if len(match) != 1:
        raise ValueError(
            f"expected one encoded baseline carrier for dataset={dataset_index}, "
            f"strength={strength}, seed={training_seed}; found {len(match)}"
        )
    return float(match.iloc[0].strict_auc), float(match.iloc[0].full_auc)


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def run(output: Path, resume: bool):
    completed = set()
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = set(pd.read_csv(output).run_id.astype(str))
    manifest = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    manifest = manifest[manifest.mechanism == "M09"]
    encoded = pd.read_csv(ROOT / "results/edbt_eab_revision/b1_multiseed_p2.csv")
    for row in manifest.itertuples(index=False):
        X, y, train, test, truth = load_synthetic(row)
        groups = m09_groups(int(row.n_original), X.shape[1])
        budget_groups = max(1, round(PRIMARY_BUDGET * len(groups)))
        scores = np.nan_to_num(mutual_info_classif(X[train], y[train], random_state=42), nan=0.0)
        group_scores = np.asarray([scores[group].max() for group in groups])
        p3_groups = np.argsort(group_scores, kind="stable")[::-1][:budget_groups]
        strict_auc, full_auc = encoded_baseline(encoded, row.dataset_index, row.strength, row.seed)
        gap = abs(full_auc - strict_auc)
        policies = [("P3_blind_mi", -1, p3_groups)] + [
            ("P2_random", governance_seed,
             random_selection(len(groups), budget_groups, governance_seed, int(row.dataset_index), int(row.seed)))
            for governance_seed in GOVERNANCE_SEEDS
        ]
        for policy, governance_seed, removed_groups in policies:
            identity = (
                f"semantic-v4|{row.dataset_index}|{row.bundle_key}|"
                f"{governance_seed}|{policy}|{budget_groups}"
            )
            run_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            if run_id in completed:
                continue
            removed_columns = np.unique(np.concatenate([groups[index] for index in removed_groups]))
            record = {field: "" for field in FIELDS}
            record.update({
                "run_id": run_id, "dataset_index": row.dataset_index, "mechanism": "M09",
                "strength": row.strength, "training_seed": row.seed,
                "governance_seed": governance_seed, "policy": policy,
                "budget_fraction": PRIMARY_BUDGET, "budget_groups": budget_groups,
                "removed_group_count": len(removed_groups), "removed_column_count": len(removed_columns),
                "status": "FAILURE", "strict_auc": strict_auc, "full_auc": full_auc,
                "initial_gap": gap,
                "selection_group_hash": selection_hash(removed_groups, "semantic_group"),
                "selection_column_hash": selection_hash(removed_columns, "encoded_column"),
            })
            try:
                keep = np.setdiff1d(np.arange(X.shape[1]), removed_columns)
                governed_auc = fit_lr_auc(X, y, train, test, keep, int(row.seed))
                removed_leak = int(truth[removed_columns].sum())
                record.update({
                    "status": "SUCCESS", "governed_auc": governed_auc,
                    "strict_distance_reduction": gap - abs(governed_auc - strict_auc),
                    "removed_leak_count": removed_leak,
                    "leak_recall": removed_leak / int(truth.sum()),
                    "legit_retention": 1.0 - (len(removed_columns) - removed_leak) / int((~truth).sum()),
                })
            except Exception as exc:
                record["failure_reason"] = f"{type(exc).__name__}: {exc}"
            append(output, record)
        print(f"semantic-v2 M09 {row.bundle_key} complete", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/edbt_eab_revision/semantic_m09_cells.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("prospective experiment locked; pass --allow-run")
    run(ROOT / args.output, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
