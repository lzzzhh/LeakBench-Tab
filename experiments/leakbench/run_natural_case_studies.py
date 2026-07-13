#!/usr/bin/env python3
"""Run corrected real-data leakage case studies with explicit lineage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmark_v2.core.models import LeakageLabel
from benchmark_v2.datasets.adapters import build_bank_marketing, build_lending_club
from benchmark_v2.datasets.confirmatory_adapters import build_bts_flights, build_chicago_food, build_nyc_311
from src.leakbench.models.core_models import fit_predict_core_model


BUILDERS = {
    "bank": build_bank_marketing,
    "lending": build_lending_club,
    "bts": build_bts_flights,
    "chicago": build_chicago_food,
    "nyc311": build_nyc_311,
}


def train_impute(task):
    X = np.asarray(task.X, dtype=np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    medians = np.nanmedian(X[task.train_idx], axis=0)
    medians[~np.isfinite(medians)] = 0.0
    rows, columns = np.where(np.isnan(X))
    X[rows, columns] = medians[columns]
    return X.astype(np.float32)


def leakage_mask(task):
    forbidden = {LeakageLabel.DIRECT_FORBIDDEN, LeakageLabel.PROXY, LeakageLabel.POST_OUTCOME}
    labels = np.array([ground_truth.label in forbidden for ground_truth in task.ground_truth], dtype=bool)
    unavailable = np.array([not item.available_at_prediction for item in task.availability], dtype=bool)
    if len(labels) != task.X.shape[1] or len(unavailable) != task.X.shape[1]:
        raise ValueError(f"{task.name}: metadata does not match feature matrix")
    return labels | unavailable


def diagnostic(task, X, truth, seed=20260713):
    train = task.train_idx
    if len(train) > 20_000:
        rng = np.random.RandomState(seed)
        positive = train[task.y[train] > 0.5]
        negative = train[task.y[train] <= 0.5]
        positive_n = min(len(positive), int(round(20_000 * len(positive) / len(train))))
        negative_n = min(len(negative), 20_000 - positive_n)
        train = np.concatenate([
            rng.choice(positive, size=positive_n, replace=False),
            rng.choice(negative, size=negative_n, replace=False),
        ])
    scores = mutual_info_classif(X[train], task.y[train], random_state=seed)
    scores = np.nan_to_num(scores, nan=0.0)
    ap = float(average_precision_score(truth.astype(int), scores))
    prevalence = float(truth.mean())
    normalized_ap = (ap - prevalence) / max(1e-12, 1.0 - prevalence)
    order = np.argsort(scores, kind="stable")[::-1]
    relevant_ranks = np.flatnonzero(truth[order]) + 1
    top5_recall = float(truth[order[:5]].sum() / max(1, truth.sum()))
    mrr = 0.0 if len(relevant_ranks) == 0 else float(1.0 / relevant_ranks.min())
    return scores, ap, normalized_ap, top5_recall, mrr, len(train)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="bank,lending,bts,chicago,nyc311")
    parser.add_argument("--models", default="lr,rf,catboost,lightgbm")
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--output", default="results/corrected_v2/natural_cells.csv")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    task_names = [item.strip() for item in args.tasks.split(",")]
    models = [item.strip() for item in args.models.split(",")]
    seeds = [int(item) for item in args.seeds.split(",")]
    output = ROOT / args.output
    completed = set()
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        existing = pd.read_csv(output)
        completed = set(
            existing.loc[existing["status"] == "SUCCESS", ["task", "model", "seed"]]
            .itertuples(index=False, name=None)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task_name in task_names:
        task = BUILDERS[task_name]()
        if task.lineage.get("is_synthetic") is not False:
            raise ValueError(f"{task.name}: natural case study received non-real lineage")
        X = train_impute(task)
        truth = leakage_mask(task)
        if not truth.any() or truth.all():
            raise ValueError(f"{task.name}: case study requires both legitimate and contaminated features")
        scores, ap, normalized_ap, top5_recall, mrr, diagnostic_rows = diagnostic(task, X, truth)
        strict = X[:, ~truth]
        lineage_json = json.dumps(
            task.lineage,
            sort_keys=True,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        summaries.append({
            "task": task.name,
            "n_samples": len(task.y),
            "n_features": X.shape[1],
            "n_leak": int(truth.sum()),
            "prevalence": float(task.y.mean()),
            "diagnostic_ap": ap,
            "diagnostic_normalized_ap": normalized_ap,
            "top5_recall": top5_recall,
            "mrr": mrr,
            "diagnostic_train_rows": diagnostic_rows,
            "source": task.source,
            "source_sha256": task.lineage["source_sha256"],
            "lineage": lineage_json,
        })
        for model in models:
            for seed in seeds:
                if (task.name, model, seed) in completed:
                    continue
                row = {
                    "task": task.name,
                    "model": model,
                    "seed": seed,
                    "status": "FAILURE",
                    "failure_reason": "",
                    "n_samples": len(task.y),
                    "n_features": X.shape[1],
                    "n_leak": int(truth.sum()),
                    "diagnostic_ap": ap,
                    "diagnostic_normalized_ap": normalized_ap,
                    "top5_recall": top5_recall,
                    "mrr": mrr,
                    "source_sha256": task.lineage["source_sha256"],
                }
                try:
                    strict_output = fit_predict_core_model(
                        model,
                        strict[task.train_idx], task.y[task.train_idx],
                        strict[task.val_idx], task.y[task.val_idx],
                        strict[task.test_idx], seed,
                    )
                    full_output = fit_predict_core_model(
                        model,
                        X[task.train_idx], task.y[task.train_idx],
                        X[task.val_idx], task.y[task.val_idx],
                        X[task.test_idx], seed,
                    )
                    strict_auc = float(roc_auc_score(task.y[task.test_idx], strict_output.probabilities))
                    permissive_auc = float(roc_auc_score(task.y[task.test_idx], full_output.probabilities))
                    row.update({
                        "status": "SUCCESS",
                        "strict_auc": strict_auc,
                        "permissive_auc": permissive_auc,
                        "paired_harm": permissive_auc - strict_auc,
                        "implementation": full_output.implementation,
                        "strict_runtime_sec": strict_output.runtime_sec,
                        "permissive_runtime_sec": full_output.runtime_sec,
                    })
                except Exception as exc:
                    row["failure_reason"] = f"{type(exc).__name__}: {exc}"
                pd.DataFrame([row]).to_csv(output, mode="a", header=not output.exists(), index=False)
                print(f"{task.name} {model} seed={seed} {row['status']}", flush=True)
    pd.DataFrame(summaries).to_csv(output.parent / "natural_task_summary.csv", index=False)


if __name__ == "__main__":
    main()
