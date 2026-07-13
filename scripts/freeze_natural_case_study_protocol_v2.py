#!/usr/bin/env python3
"""Freeze the train-fitted-category natural case-study amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v2.datasets.adapters import _select_bank_file, _select_lending_file  # noqa: E402
from experiments.leakbench.run_natural_case_studies_trainfit import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DROP_FEATURES,
    PROTOCOL_VERSION,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    v1_freeze = ROOT / "results/corrected_v2/natural_protocol_freeze.json"
    if not v1_freeze.exists():
        raise FileNotFoundError(v1_freeze)
    sources = {
        "BankMarketing": _select_bank_file(ROOT / "data/raw/bank_marketing"),
        "LendingClub": _select_lending_file(ROOT / "data/raw/lending_club"),
        "BTSFlights": ROOT / "data/bts/bts_2023_1.csv.zip",
        "ChicagoFood": ROOT / "data/chicago_food/chicago_food_cache.csv",
        "NYC311": ROOT / "data/nyc311/nyc311_cache.csv",
    }
    if any(path is None or not path.exists() for path in sources.values()):
        raise FileNotFoundError("a fixed real-data source is absent")
    code_paths = [
        ROOT / "benchmark_v2/datasets/adapters.py",
        ROOT / "benchmark_v2/datasets/confirmatory_adapters.py",
        ROOT / "src/leakbench/models/core_models.py",
        ROOT / "experiments/leakbench/run_natural_case_studies.py",
        ROOT / "experiments/leakbench/run_natural_case_studies_trainfit.py",
        ROOT / "scripts/analyze_natural_case_studies.py",
        ROOT / "tests/test_natural_trainfit.py",
    ]
    output = ROOT / "results/corrected_v2/natural_protocol_v2_freeze.json"
    if output.exists():
        raise FileExistsError(output)
    policy = {
        "categorical_features": {key: sorted(value) for key, value in sorted(CATEGORICAL_FEATURES.items())},
        "dropped_date_string_features": {key: sorted(value) for key, value in sorted(DROP_FEATURES.items())},
    }
    policy_bytes = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_NATURAL_TRAINFIT_V2_RERUN",
        "amendment_version": PROTOCOL_VERSION,
        "reason": "v1 category IDs were learned from the full table; v2 re-indexes only training-observed categories, maps unseen categories to unknown, and drops globally encoded date strings",
        "supersedes": "results/corrected_v2/natural_protocol_freeze.json",
        "expected_cells": 60,
        "tasks": ["BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"],
        "models": ["lr", "rf", "catboost", "lightgbm"],
        "seeds": [13, 42, 2026],
        "output": "results/corrected_v2/natural_cells.csv",
        "task_summary": "results/corrected_v2/natural_task_summary.csv",
        "category_policy": policy,
        "category_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "unknown_category_value": -2.0,
        "missing_category_value": -1.0,
        "fit_scope": "training rows only",
        "interpretation": "five fixed boundary-specific case studies; descriptive external-validity audit only, never population inference",
        "code_files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in code_paths
        },
        "source_files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for name, path in sources.items()
        },
        "superseded_v1_outputs": [
            "results/corrected_v2/superseded_snapshots/natural_cells_boundary_corrected_global_category_encoding.csv",
            "results/corrected_v2/superseded_snapshots/natural_task_summary_boundary_corrected_global_category_encoding.csv"
        ],
        "rule": "Any code, policy, source hash, model, seed, or boundary change requires a new version and full 60-cell rerun."
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
