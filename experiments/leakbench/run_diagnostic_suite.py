#!/usr/bin/env python3
"""Evaluate blind statistical feature-ranking diagnostics on frozen task bundles.

The diagnostic methods may use training labels, but never receive the oracle
leakage mask until all feature scores have been produced.  The mask is used only
to evaluate localization AUPRC and top-k recall.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_corrected_tabm_bundle import (  # noqa: E402
    file_sha256,
    load_bundle_contract,
    load_verified_task,
    parse_selection,
    verify_before_fit,
)


METHODS = ("mutual_information", "absolute_correlation", "lr_coefficient", "rf_permutation")
FIELDS = (
    "diagnostic_run_id", "dataset_id", "dataset_index", "dataset_namespace",
    "mechanism", "strength", "seed", "method", "status", "failure_reason",
    "localization_ap", "localization_normalized_ap", "top5_recall", "n_leak",
    "n_features", "runtime_sec", "task_hash", "split_hash", "bundle_sha256",
    "task_manifest_sha256", "integrity_verified", "config_hash", "code_hash",
)


def _finite_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("diagnostic scores must be a finite one-dimensional array")
    return scores


def compute_blind_scores(X_train, y_train, X_validation, y_validation, seed):
    """Return feature scores without accepting any oracle contamination labels."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    X_validation = np.asarray(X_validation, dtype=float)
    y_validation = np.asarray(y_validation, dtype=int)
    if X_train.ndim != 2 or X_validation.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if X_train.shape[1] != X_validation.shape[1]:
        raise ValueError("train and validation feature counts differ")
    if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
        raise ValueError("both training and validation partitions require two classes")

    mi = mutual_info_classif(X_train, y_train, random_state=int(seed))
    y_centered = y_train - y_train.mean()
    X_centered = X_train - X_train.mean(axis=0, keepdims=True)
    denominator = np.sqrt((X_centered ** 2).sum(axis=0) * (y_centered ** 2).sum())
    correlation = np.divide(
        np.abs(X_centered.T @ y_centered),
        denominator,
        out=np.zeros(X_train.shape[1], dtype=float),
        where=denominator > 0,
    )

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=int(seed)),
    )
    logistic.fit(X_train, y_train)
    coefficient = np.abs(logistic.named_steps["logisticregression"].coef_[0])

    forest = RandomForestClassifier(
        n_estimators=64,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=1,
    )
    forest.fit(X_train, y_train)
    importance = permutation_importance(
        forest,
        X_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=3,
        random_state=int(seed) + 1,
        n_jobs=1,
    ).importances_mean
    return {
        "mutual_information": _finite_scores(mi),
        "absolute_correlation": _finite_scores(correlation),
        "lr_coefficient": _finite_scores(coefficient),
        "rf_permutation": _finite_scores(importance),
    }


def evaluate_localization(scores, leakage_mask):
    """Evaluate already-computed blind scores against the held-back oracle mask."""
    scores = _finite_scores(scores)
    truth = np.asarray(leakage_mask, dtype=bool)
    if truth.shape != scores.shape or truth.sum() == 0:
        raise ValueError("oracle mask must match scores and contain a positive feature")
    prevalence = float(truth.mean())
    ap = float(average_precision_score(truth.astype(int), scores))
    normalized_ap = float((ap - prevalence) / max(1e-12, 1.0 - prevalence))
    top = np.argsort(scores, kind="stable")[::-1][: min(5, len(scores))]
    top5 = float(truth[top].sum() / truth.sum())
    return ap, normalized_ap, top5


def _append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mechanisms", default="all")
    parser.add_argument("--strengths", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-confirmatory", action="store_true")
    args = parser.parse_args(argv)

    config_path = ROOT / args.config
    manifest_path = ROOT / args.task_manifest
    output_path = ROOT / args.output
    config_hash = file_sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest, manifest_hash, _, summary = load_bundle_contract(manifest_path, config_hash)
    if args.namespace == "confirmatory" and not args.allow_confirmatory:
        raise RuntimeError("confirmatory diagnostics require --allow-confirmatory")
    if args.namespace != str(summary["dataset_namespace"]):
        raise RuntimeError("requested namespace does not match frozen bundle summary")

    dataset_universe = sorted(int(value) for value in manifest["dataset_index"].unique())
    mechanism_universe = list(config["protocol"]["mechanisms"])
    strength_universe = list(config["protocol"]["strengths"])
    seed_universe = [int(value) for value in config["protocol"]["seeds"]]
    datasets = parse_selection(args.datasets, dataset_universe)
    mechanisms = parse_selection(args.mechanisms, mechanism_universe)
    strengths = parse_selection(args.strengths, strength_universe)
    seeds = parse_selection(args.seeds, seed_universe)
    methods = list(METHODS) if args.methods == "all" else parse_selection(args.methods, list(METHODS))
    selected = manifest[
        manifest["dataset_index"].isin(datasets)
        & manifest["mechanism"].isin(mechanisms)
        & manifest["strength"].isin(strengths)
        & manifest["seed"].isin(seeds)
    ].copy()
    if selected.empty:
        raise RuntimeError("selection produced no frozen tasks")
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} exists; use --resume or a new path")
    completed = set()
    if output_path.exists():
        previous = pd.read_csv(output_path)
        completed = set(previous.loc[previous["status"] == "SUCCESS", "diagnostic_run_id"].astype(str))

    code_hash = file_sha256(Path(__file__))
    total = len(selected) * len(methods)
    done = 0
    started = time.time()
    for _, row in selected.sort_values(["dataset_index", "mechanism", "strength", "seed"]).iterrows():
        task, bundle_path = load_verified_task(row, ROOT)
        verify_before_fit(task, row, bundle_path)
        task_started = time.time()
        try:
            scores_by_method = compute_blind_scores(
                task.X[task.train_idx], task.y[task.train_idx],
                task.X[task.val_idx], task.y[task.val_idx], int(row["seed"]),
            )
            verify_before_fit(task, row, bundle_path)
            failure = ""
        except Exception as exc:  # retain failures as scientific cells
            scores_by_method = {}
            failure = f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - task_started
        for method in methods:
            identity = f"{row['dataset_id']}|{row['mechanism']}|{row['strength']}|{int(row['seed'])}|{method}"
            run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            if run_id in completed:
                done += 1
                continue
            base = {
                "diagnostic_run_id": run_id,
                "dataset_id": row["dataset_id"], "dataset_index": int(row["dataset_index"]),
                "dataset_namespace": args.namespace, "mechanism": row["mechanism"],
                "strength": row["strength"], "seed": int(row["seed"]), "method": method,
                "n_leak": int(row["n_leak"]), "n_features": int(row["n_original"] + row["n_injected"]),
                "runtime_sec": elapsed, "task_hash": row["task_hash"], "split_hash": row["split_hash"],
                "bundle_sha256": row["bundle_sha256"], "task_manifest_sha256": manifest_hash,
                "config_hash": config_hash, "code_hash": code_hash,
            }
            if failure:
                base.update(status="FAILURE", failure_reason=failure, localization_ap=np.nan,
                            localization_normalized_ap=np.nan, top5_recall=np.nan, integrity_verified=False)
            else:
                ap, normalized_ap, top5 = evaluate_localization(scores_by_method[method], task.leakage_mask)
                base.update(status="SUCCESS", failure_reason="", localization_ap=ap,
                            localization_normalized_ap=normalized_ap, top5_recall=top5,
                            integrity_verified=True)
            _append(output_path, base)
            done += 1
        if done % 100 == 0 or done == total:
            print(f"{done}/{total} diagnostic cells in {time.time() - started:.1f}s", flush=True)

    result = pd.read_csv(output_path)
    selected_result = result[result["dataset_namespace"] == args.namespace]
    expected = len(selected) * len(methods)
    if selected_result.duplicated(["dataset_id", "mechanism", "strength", "seed", "method"]).any():
        raise RuntimeError("duplicate diagnostic identities in output")
    manifest_out = {
        "schema_version": 1,
        "evidence_tier": args.namespace,
        "task_manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
        "code_sha256": code_hash,
        "methods": methods,
        "expected_cells": expected,
        "observed_cells": int(len(selected_result)),
        "successful_cells": int((selected_result["status"] == "SUCCESS").sum()),
        "failed_cells": int((selected_result["status"] != "SUCCESS").sum()),
        "output_sha256": file_sha256(output_path),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest_out, indent=2, sort_keys=True), encoding="utf-8"
    )
    if manifest_out["observed_cells"] != expected or manifest_out["successful_cells"] != expected:
        raise RuntimeError(f"incomplete diagnostic matrix: {manifest_out}")


if __name__ == "__main__":
    main()
