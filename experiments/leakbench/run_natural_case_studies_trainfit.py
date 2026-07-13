#!/usr/bin/env python3
"""Boundary-corrected natural case studies with train-fitted category coding.

The frozen v1 adapters converted strings to full-table category codes.  This
amendment repairs that unsupervised vocabulary look-ahead without changing the
raw-file selection, row ordering, targets, or prediction boundaries: globally
encoded category IDs are re-indexed from training-observed IDs only, unseen
validation/test IDs receive a dedicated unknown value, and date-string columns
whose old encodings depended on the full vocabulary are excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_natural_case_studies import (  # noqa: E402
    BUILDERS,
    diagnostic,
    leakage_mask,
    train_impute,
)
from src.leakbench.models.core_models import fit_predict_core_model  # noqa: E402


PROTOCOL_VERSION = "natural_trainfit_categories_v2"
UNKNOWN_CATEGORY = -2.0
MISSING_CATEGORY = -1.0

# Names are the adapters' ``name_pool['natural']`` values after their fixed
# renaming rules.  Date strings are deliberately handled by DROP_FEATURES.
CATEGORICAL_FEATURES = {
    "BankMarketing": {
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome",
    },
    "LendingClub": {
        "term", "grade", "sub_grade", "emp_title", "emp_length",
        "home_ownership", "verification_status", "pymnt_plan", "url", "desc",
        "purpose", "title", "zip_code", "addr_state", "initial_list_status",
        "application_type", "verification_status_joint", "hardship_flag",
        "hardship_type", "hardship_reason", "hardship_status",
        "hardship_loan_status", "disbursement_method", "debt_settlement_flag",
        "settlement_status",
    },
    "BTSFlights": {
        "Reporting_Airline", "IATA_CODE_Reporting_Airline", "Tail_Number",
        "Origin", "OriginCityName", "OriginState", "OriginStateName", "Dest",
        "DestCityName", "DestState", "DestStateName", "DepTimeBlk", "ArrTimeBlk",
        "CancellationCode", "Div1Airport", "Div1TailNum", "Div2Airport", "Div2TailNum",
    },
    "ChicagoFood": {
        "chi_dba_name", "chi_aka_name", "chi_facility_type", "chi_risk",
        "chi_address", "chi_city", "chi_state", "chi_inspection_type",
        "chi_results", "chi_violations", "location",
    },
    "NYC311": {
        "agency", "agency_name", "complaint_type", "descriptor", "descriptor_2",
        "location_type", "incident_address", "street_name", "cross_street_1",
        "cross_street_2", "intersection_street_1", "intersection_street_2",
        "address_type", "city", "landmark", "facility_type", "status",
        "resolution_description", "community_board", "police_precinct", "borough",
        "open_data_channel_type", "park_facility_name", "park_borough", "vehicle_type",
        "taxi_company_borough", "taxi_pick_up_location", "bridge_highway_name",
        "bridge_highway_direction", "road_ramp", "bridge_highway_segment", "location",
    },
}

DROP_FEATURES = {
    "BankMarketing": set(),
    "LendingClub": {
        "issue_d", "earliest_cr_line", "last_pymnt_d", "next_pymnt_d",
        "last_credit_pull_d", "hardship_start_date", "hardship_end_date",
        "payment_plan_start_date", "debt_settlement_flag_date", "settlement_date",
    },
    "BTSFlights": {"FlightDate"},
    "ChicagoFood": {"chi_inspection_date"},
    "NYC311": {"due_date", "resolution_action_updated_date"},
}


def _sha256_bytes(values):
    return hashlib.sha256(values).hexdigest()


def feature_names(task):
    return [spec.name_pool.get("natural", spec.feature_id) for spec in task.feature_specs]


def apply_train_fitted_category_protocol(task):
    """Return corrected X/mask and a complete deterministic preprocessing audit."""
    names = feature_names(task)
    if len(names) != len(set(names)):
        raise ValueError(f"{task.name}: natural feature names are not unique")
    drop_requested = DROP_FEATURES[task.name]
    missing_drop = drop_requested - set(names)
    if missing_drop:
        raise ValueError(f"{task.name}: expected date-string features missing: {sorted(missing_drop)}")
    keep = np.asarray([name not in drop_requested for name in names], dtype=bool)
    retained_names = [name for name, retained in zip(names, keep) if retained]
    X = train_impute(task)[:, keep].astype(np.float32, copy=True)
    truth = leakage_mask(task)[keep]

    requested_categories = CATEGORICAL_FEATURES[task.name] - drop_requested
    missing_categories = requested_categories - set(retained_names)
    # Some optional natural columns can be dropped by the v1 adapter if their
    # dtype is unsupported.  Record but do not silently invent them.
    mappings = {}
    unseen_counts = {}
    for name in sorted(requested_categories & set(retained_names)):
        column = retained_names.index(name)
        raw = X[:, column].astype(float)
        if not np.isfinite(raw).all():
            raise ValueError(f"{task.name}/{name}: category codes are non-finite")
        train_values = np.unique(raw[task.train_idx])
        observed = [float(value) for value in train_values if value != MISSING_CATEGORY]
        mapping = {value: float(index) for index, value in enumerate(sorted(observed))}
        corrected = np.full(len(raw), UNKNOWN_CATEGORY, dtype=np.float32)
        corrected[raw == MISSING_CATEGORY] = MISSING_CATEGORY
        for old, new in mapping.items():
            corrected[raw == old] = new
        unknown_mask = (corrected == UNKNOWN_CATEGORY)
        if unknown_mask[task.train_idx].any():
            raise RuntimeError(f"{task.name}/{name}: a training category mapped to unknown")
        X[:, column] = corrected
        mappings[name] = [[old, new] for old, new in sorted(mapping.items())]
        unseen_counts[name] = {
            "validation": int(unknown_mask[task.val_idx].sum()),
            "test": int(unknown_mask[task.test_idx].sum()),
        }
    if not np.isfinite(X).all():
        raise ValueError(f"{task.name}: corrected feature matrix is non-finite")
    mapping_json = json.dumps(mappings, sort_keys=True, separators=(",", ":"))
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "fit_rows": "train_idx_only",
        "unknown_category_value": UNKNOWN_CATEGORY,
        "missing_category_value": MISSING_CATEGORY,
        "categorical_features_requested": sorted(requested_categories),
        "categorical_features_remapped": sorted(mappings),
        "optional_categorical_features_absent": sorted(missing_categories),
        "dropped_date_string_features": sorted(drop_requested),
        "unseen_category_counts": unseen_counts,
        "mapping_sha256": _sha256_bytes(mapping_json.encode("utf-8")),
        "retained_feature_count": int(X.shape[1]),
        "retained_leak_count": int(truth.sum()),
    }
    return X, truth, retained_names, audit


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="bank,lending,bts,chicago,nyc311")
    parser.add_argument("--models", default="lr,rf,catboost,lightgbm")
    parser.add_argument("--seeds", default="13,42,2026")
    parser.add_argument("--output", default="results/corrected_v2/natural_cells.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-confirmatory", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_confirmatory:
        raise RuntimeError("train-fitted natural amendment requires --allow-confirmatory")
    task_names = [item.strip() for item in args.tasks.split(",")]
    models = [item.strip() for item in args.models.split(",")]
    seeds = [int(item) for item in args.seeds.split(",")]
    output = ROOT / args.output
    completed = set()
    if output.exists():
        if not args.resume:
            raise FileExistsError(output)
        existing = pd.read_csv(output)
        completed = set(existing.loc[existing["status"] == "SUCCESS", ["task", "model", "seed"]].itertuples(index=False, name=None))
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for task_name in task_names:
        task = BUILDERS[task_name]()
        if task.lineage.get("is_synthetic") is not False:
            raise ValueError(f"{task.name}: natural case study received non-real lineage")
        X, truth, retained_names, preprocessing = apply_train_fitted_category_protocol(task)
        if not truth.any() or truth.all():
            raise ValueError(f"{task.name}: requires both legitimate and contaminated retained features")
        _, ap, normalized_ap, top5_recall, mrr, diagnostic_rows = diagnostic(task, X, truth)
        strict = X[:, ~truth]
        lineage = {
            **task.lineage,
            "preprocessing": preprocessing,
            "preprocessing_protocol": PROTOCOL_VERSION,
        }
        lineage_json = json.dumps(lineage, sort_keys=True, default=lambda value: value.item() if isinstance(value, np.generic) else str(value))
        summaries.append({
            "task": task.name, "n_samples": len(task.y), "n_features": X.shape[1],
            "n_leak": int(truth.sum()), "prevalence": float(task.y.mean()),
            "diagnostic_ap": ap, "diagnostic_normalized_ap": normalized_ap,
            "top5_recall": top5_recall, "mrr": mrr,
            "diagnostic_train_rows": diagnostic_rows, "source": task.source,
            "source_sha256": task.lineage["source_sha256"], "lineage": lineage_json,
            "preprocessing_protocol": PROTOCOL_VERSION,
            "preprocessing_mapping_sha256": preprocessing["mapping_sha256"],
            "retained_feature_names_sha256": _sha256_bytes("\n".join(retained_names).encode("utf-8")),
        })
        for model in models:
            for seed in seeds:
                if (task.name, model, seed) in completed:
                    continue
                row = {
                    "task": task.name, "model": model, "seed": seed,
                    "status": "FAILURE", "failure_reason": "", "n_samples": len(task.y),
                    "n_features": X.shape[1], "n_leak": int(truth.sum()),
                    "diagnostic_ap": ap, "diagnostic_normalized_ap": normalized_ap,
                    "top5_recall": top5_recall, "mrr": mrr,
                    "source_sha256": task.lineage["source_sha256"],
                    "preprocessing_protocol": PROTOCOL_VERSION,
                    "preprocessing_mapping_sha256": preprocessing["mapping_sha256"],
                }
                try:
                    strict_output = fit_predict_core_model(
                        model, strict[task.train_idx], task.y[task.train_idx],
                        strict[task.val_idx], task.y[task.val_idx], strict[task.test_idx], seed,
                    )
                    full_output = fit_predict_core_model(
                        model, X[task.train_idx], task.y[task.train_idx],
                        X[task.val_idx], task.y[task.val_idx], X[task.test_idx], seed,
                    )
                    strict_auc = float(roc_auc_score(task.y[task.test_idx], strict_output.probabilities))
                    full_auc = float(roc_auc_score(task.y[task.test_idx], full_output.probabilities))
                    row.update({
                        "status": "SUCCESS", "strict_auc": strict_auc,
                        "permissive_auc": full_auc, "paired_harm": full_auc - strict_auc,
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
