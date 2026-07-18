#!/usr/bin/env python3
"""Run the two prospective EDBT governance sensitivities.

Natural governance evaluates LR on five fixed real-data case studies. Semantic
group governance reruns only M09 because it is the only registry mechanism
whose single semantic source is expanded into multiple encoded columns; every
other mechanism is an identity mapping under the frozen encoder contract.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_natural_case_studies import BUILDERS  # noqa: E402
from experiments.leakbench.run_natural_case_studies_trainfit import (  # noqa: E402
    apply_train_fitted_category_protocol,
)


GOVERNANCE_SEEDS = tuple(2026071700 + i for i in range(20))
TRAINING_SEEDS = (13, 42, 2026)
PRIMARY_BUDGET = 0.20
NATURAL_TASKS = ("bank", "lending", "bts", "chicago", "nyc311")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_hash(indices, unit: str) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(f"{unit}_indices_v1\0".encode() + values.tobytes()).hexdigest()


def random_selection(n_units: int, k: int, governance_seed: int, task_index: int, training_seed: int):
    seed = (governance_seed * 100 + task_index * 7 + training_seed * 13) % (2**31 - 1)
    return np.random.RandomState(seed).choice(n_units, k, replace=False)


def fit_lr_auc(X, y, train, test, columns, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed, C=1.0),
    )
    model.fit(X[train][:, columns], y[train])
    return float(roc_auc_score(y[test], model.predict_proba(X[test][:, columns])[:, 1]))


def append_row(path: Path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_natural(output: Path, resume: bool):
    fields = [
        "run_id", "task", "training_seed", "governance_seed", "policy",
        "budget_fraction", "budget_k", "status", "failure_reason",
        "strict_auc", "full_auc", "governed_auc", "initial_gap",
        "strict_distance_reduction", "removed_count", "removed_leak_count",
        "leak_recall", "legit_retention", "selection_mask_hash",
        "source_sha256", "preprocessing_mapping_sha256",
    ]
    completed = set()
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = set(pd.read_csv(output)["run_id"].astype(str))
    baselines = pd.read_csv(ROOT / "results/corrected_v2/natural_cells.csv")
    baselines = baselines[(baselines["model"] == "lr") & (baselines["status"] == "SUCCESS")]

    for task_index, task_key in enumerate(NATURAL_TASKS):
        task = BUILDERS[task_key]()
        X, truth, _, audit = apply_train_fitted_category_protocol(task)
        n_features = X.shape[1]
        budget_k = max(1, round(PRIMARY_BUDGET * n_features))
        scores = mutual_info_classif(X[task.train_idx], task.y[task.train_idx], random_state=42)
        scores = np.nan_to_num(scores, nan=0.0)
        p3_selection = np.argsort(scores, kind="stable")[::-1][:budget_k]
        policies = [("P3_blind_mi", -1, p3_selection)]
        policies.extend(
            ("P2_random", governance_seed,
             random_selection(n_features, budget_k, governance_seed, task_index, training_seed))
            for training_seed in TRAINING_SEEDS
            for governance_seed in GOVERNANCE_SEEDS
        )
        for training_seed in TRAINING_SEEDS:
            baseline = baselines[
                (baselines["task"] == task.name) & (baselines["seed"] == training_seed)
            ]
            if len(baseline) != 1:
                raise ValueError(f"missing frozen natural baseline: {task.name}/{training_seed}")
            baseline = baseline.iloc[0]
            strict_auc = float(baseline.strict_auc)
            full_auc = float(baseline.permissive_auc)
            gap = abs(full_auc - strict_auc)
            per_seed_policies = [("P3_blind_mi", -1, p3_selection)] + [
                ("P2_random", governance_seed,
                 random_selection(n_features, budget_k, governance_seed, task_index, training_seed))
                for governance_seed in GOVERNANCE_SEEDS
            ]
            for policy, governance_seed, removed in per_seed_policies:
                identity = f"natural|{task.name}|{training_seed}|{governance_seed}|{policy}|{budget_k}"
                run_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
                if run_id in completed:
                    continue
                row = {field: "" for field in fields}
                row.update({
                    "run_id": run_id, "task": task.name, "training_seed": training_seed,
                    "governance_seed": governance_seed, "policy": policy,
                    "budget_fraction": PRIMARY_BUDGET, "budget_k": budget_k,
                    "status": "FAILURE", "strict_auc": strict_auc, "full_auc": full_auc,
                    "initial_gap": gap, "removed_count": len(removed),
                    "selection_mask_hash": selection_hash(removed, "encoded_column"),
                    "source_sha256": task.lineage["source_sha256"],
                    "preprocessing_mapping_sha256": audit["mapping_sha256"],
                })
                try:
                    keep = np.setdiff1d(np.arange(n_features), removed)
                    governed_auc = fit_lr_auc(X, task.y, task.train_idx, task.test_idx, keep, training_seed)
                    removed_leak = int(truth[removed].sum())
                    row.update({
                        "status": "SUCCESS", "governed_auc": governed_auc,
                        "strict_distance_reduction": gap - abs(governed_auc - strict_auc),
                        "removed_leak_count": removed_leak,
                        "leak_recall": removed_leak / int(truth.sum()),
                        "legit_retention": 1.0 - (len(removed) - removed_leak) / int((~truth).sum()),
                    })
                except Exception as exc:
                    row["failure_reason"] = f"{type(exc).__name__}: {exc}"
                append_row(output, fields, row)
                print(f"natural {task.name} seed={training_seed} {policy} gov={governance_seed} {row['status']}", flush=True)


def load_synthetic(row):
    path = ROOT / row.bundle_path
    if sha256(path) != str(row.bundle_sha256).lower():
        raise ValueError(f"bundle hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        key = str(row.bundle_key)
        X = np.concatenate((payload["base_X"], payload[f"block__{key}"]), axis=1)
        return X, payload["y"], payload["train_idx"], payload["test_idx"], payload[f"leak_mask__{key}"]


def m09_groups(n_original: int, n_features: int):
    """M09's eight one-hot columns are one auditable source-field group."""
    groups = [np.asarray([index], dtype=int) for index in range(n_original)]
    groups.append(np.arange(n_original, n_features, dtype=int))
    return groups


def run_semantic(output: Path, resume: bool):
    fields = [
        "run_id", "dataset_index", "mechanism", "strength", "training_seed",
        "governance_seed", "policy", "budget_fraction", "budget_groups",
        "removed_group_count", "removed_column_count", "status", "failure_reason",
        "strict_auc", "full_auc", "governed_auc", "initial_gap",
        "strict_distance_reduction", "removed_leak_count", "leak_recall",
        "legit_retention", "selection_group_hash", "selection_column_hash",
    ]
    completed = set()
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        completed = set(pd.read_csv(output)["run_id"].astype(str))
    manifest = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    manifest = manifest[manifest["mechanism"] == "M09"]
    encoded = pd.read_csv(ROOT / "results/edbt_eab_revision/b1_multiseed_p2.csv")
    encoded = encoded[(encoded["mechanism"] == "M09") & np.isclose(encoded["budget_fraction"], PRIMARY_BUDGET)]
    for row in manifest.itertuples(index=False):
        X, y, train, test, truth = load_synthetic(row)
        groups = m09_groups(int(row.n_original), X.shape[1])
        budget_groups = max(1, round(PRIMARY_BUDGET * len(groups)))
        scores = mutual_info_classif(X[train], y[train], random_state=42)
        scores = np.nan_to_num(scores, nan=0.0)
        group_scores = np.asarray([scores[group].max() for group in groups])
        p3_groups = np.argsort(group_scores, kind="stable")[::-1][:budget_groups]
        base_key = (
            (encoded["dataset_index"] == row.dataset_index)
            & (encoded["strength"] == row.strength)
            & (encoded["training_seed"] == row.seed)
        )
        baseline = encoded[base_key & (encoded["policy"] == "P0_keep")]
        if len(baseline) != 1:
            raise ValueError(f"missing encoded baseline for M09 key {row.bundle_key}")
        baseline = baseline.iloc[0]
        strict_auc, full_auc = float(baseline.strict_auc), float(baseline.full_auc)
        gap = abs(full_auc - strict_auc)
        policies = [("P3_blind_mi", -1, p3_groups)] + [
            ("P2_random", governance_seed,
             random_selection(len(groups), budget_groups, governance_seed, int(row.dataset_index), int(row.seed)))
            for governance_seed in GOVERNANCE_SEEDS
        ]
        for policy, governance_seed, removed_groups in policies:
            identity = f"semantic|{row.bundle_key}|{governance_seed}|{policy}|{budget_groups}"
            run_id = hashlib.sha256(identity.encode()).hexdigest()[:20]
            if run_id in completed:
                continue
            removed_columns = np.unique(np.concatenate([groups[index] for index in removed_groups]))
            record = {field: "" for field in fields}
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
            append_row(output, fields, record)
        print(f"semantic M09 {row.bundle_key} complete", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("natural", "semantic"))
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("prospective experiment locked; pass --allow-run")
    default = {
        "natural": "results/edbt_eab_revision/natural_governance_cells.csv",
        "semantic": "results/edbt_eab_revision/semantic_m09_cells.csv",
    }[args.mode]
    output = ROOT / (args.output or default)
    if args.mode == "natural":
        run_natural(output, args.resume)
    else:
        run_semantic(output, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
