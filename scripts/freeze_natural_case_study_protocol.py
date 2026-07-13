#!/usr/bin/env python3
"""Freeze real-data case-study sources and prediction boundaries before rerun."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v2.datasets.adapters import _select_bank_file, _select_lending_file  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    source_paths = {
        "BankMarketing": _select_bank_file(ROOT / "data/raw/bank_marketing"),
        "LendingClub": _select_lending_file(ROOT / "data/raw/lending_club"),
        "BTSFlights": ROOT / "data/bts/bts_2023_1.csv.zip",
        "ChicagoFood": ROOT / "data/chicago_food/chicago_food_cache.csv",
        "NYC311": ROOT / "data/nyc311/nyc311_cache.csv",
    }
    missing_sources = [name for name, path in source_paths.items() if path is None or not path.exists()]
    if missing_sources:
        raise FileNotFoundError(f"missing real case-study sources: {missing_sources}")
    code_paths = [
        ROOT / "benchmark_v2/datasets/adapters.py",
        ROOT / "benchmark_v2/datasets/confirmatory_adapters.py",
        ROOT / "src/leakbench/models/core_models.py",
        ROOT / "experiments/leakbench/run_natural_case_studies.py",
        ROOT / "scripts/analyze_natural_case_studies.py",
    ]
    output = ROOT / "results/corrected_v2/natural_protocol_freeze.json"
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_BOUNDARY_CORRECTED_NATURAL_RERUN",
        "evidence_tier": "fixed_real_case_studies",
        "expected_cells": 60,
        "tasks": ["BankMarketing", "LendingClub", "BTSFlights", "ChicagoFood", "NYC311"],
        "models": ["lr", "rf", "catboost", "lightgbm"],
        "seeds": [13, 42, 2026],
        "prediction_boundaries": {
            "BankMarketing": "before marketing-call outcome; duration unavailable",
            "LendingClub": "loan origination; repayment, servicing, hardship, and settlement fields unavailable",
            "BTSFlights": "immediately before scheduled departure; explicit schedule allowlist",
            "ChicagoFood": "before inspection outcome; results and violations unavailable",
            "NYC311": "at request creation; closure status and resolution fields unavailable",
        },
        "interpretation": "five fixed case studies only; not a random sample of real-world datasets",
        "code_files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in code_paths
        },
        "source_files": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in source_paths.items()
        },
        "superseded_snapshot": "results/corrected_v2/superseded_snapshots/natural_cells_pre_bts_schedule_boundary_fix.csv",
        "rule": "Any boundary, code, or source-hash change requires a new freeze version and full 60-cell rerun.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
