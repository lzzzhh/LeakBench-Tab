#!/usr/bin/env python3
"""Freeze the blind diagnostic-suite protocol before confirmatory execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = ROOT / "configs/paper/corrected_v2.yaml"
    runner = ROOT / "experiments/leakbench/run_diagnostic_suite.py"
    analyzer = ROOT / "scripts/analyze_diagnostic_suite.py"
    full_manifest_path = ROOT / "results/corrected_v2/task_bundles/task_manifest.csv"
    full_summary_path = full_manifest_path.parent / "bundle_summary.json"
    pilot_cells_path = ROOT / "results/corrected_v2/diagnostic_pilot_cells.csv"
    pilot_manifest_path = pilot_cells_path.with_suffix(".manifest.json")
    pilot_analysis = ROOT / "results/corrected_v2/diagnostic_pilot_statistics/diagnostic_integrity.json"
    required = [
        config, runner, analyzer, full_manifest_path, full_summary_path,
        pilot_cells_path, pilot_manifest_path, pilot_analysis,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot freeze diagnostic protocol; missing {missing}")

    full = pd.read_csv(full_manifest_path)
    identity = ["dataset_id", "mechanism", "strength", "seed"]
    if len(full) != 5500 or full.duplicated(identity).any():
        raise RuntimeError("full diagnostic task manifest is not the expected 5,500-cell task matrix")
    if set(full["dataset_namespace"].astype(str)) != {"confirmatory"}:
        raise RuntimeError("full diagnostic task manifest is not confirmatory")
    bundle_hashes = full.groupby("bundle_path")["bundle_sha256"].nunique()
    if len(bundle_hashes) != 20 or (bundle_hashes != 1).any():
        raise RuntimeError("expected exactly 20 immutable bundle hashes")

    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    if (
        pilot_manifest.get("evidence_tier") != "pilot"
        or pilot_manifest.get("expected_cells") != 1980
        or pilot_manifest.get("successful_cells") != 1980
        or pilot_manifest.get("failed_cells") != 0
    ):
        raise RuntimeError("diagnostic pilot did not complete its pre-specified matrix")
    pilot_integrity = json.loads(pilot_analysis.read_text(encoding="utf-8"))
    if pilot_integrity.get("evidence_tier") != "pilot":
        raise RuntimeError("diagnostic pilot analysis has the wrong evidence tier")

    frozen_files = {}
    for path in required:
        frozen_files[str(path.relative_to(ROOT))] = {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    freeze = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_DIAGNOSTIC_CONFIRMATORY_RUN",
        "evidence_tier": "confirmatory",
        "task_count": 5500,
        "diagnostic_methods": [
            "mutual_information", "absolute_correlation", "lr_coefficient", "rf_permutation"
        ],
        "expected_diagnostic_cells": 22000,
        "primary_diagnostic": "mutual_information",
        "robustness_rule": {
            "low_across_all_evaluated_diagnostics": "upper 95% hierarchical-bootstrap bound of best evaluated method < 0.30",
            "warning": "best evaluated diagnostic is an optimistic labeled benchmark summary, not a deployable selector",
        },
        "rf_hyperparameters": {
            "n_estimators": 64, "max_depth": 8, "min_samples_leaf": 5,
            "max_features": "sqrt", "class_weight": "balanced",
            "permutation_repeats": 3,
        },
        "bootstrap": {"repetitions": 20000, "seed": 20260713, "units": ["dataset", "seed"]},
        "frozen_files": frozen_files,
        "bundle_sha256_by_path": {
            path: str(full.loc[full["bundle_path"] == path, "bundle_sha256"].iloc[0])
            for path in sorted(full["bundle_path"].unique())
        },
        "confirmatory_output": "results/corrected_v2/diagnostic_confirmatory_cells.csv",
        "rule": "Any modification to a frozen file requires a new protocol version and invalidates use of this freeze for confirmatory claims.",
    }
    output = ROOT / "results/corrected_v2/diagnostic_protocol_freeze.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(freeze, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
