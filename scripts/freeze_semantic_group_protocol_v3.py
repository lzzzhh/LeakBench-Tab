#!/usr/bin/env python3
"""Freeze semantic-group v3 after two fail-closed zero-row lookup attempts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/edbt_eab_revision/semantic_group_protocol_v3_freeze.json"


def bind(path):
    target = ROOT / path
    return {"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "size_bytes": target.stat().st_size}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    output = ROOT / "results/edbt_eab_revision/semantic_m09_cells.csv"
    if output.exists():
        raise RuntimeError("pre-v3 attempts unexpectedly produced output")
    encoded = pd.read_csv(ROOT / "results/edbt_eab_revision/b1_multiseed_p2.csv")
    carriers = encoded[
        (encoded.mechanism == "M09")
        & (encoded.policy == "P3_blind_mi")
        & (encoded.budget_fraction.round(8) == 0.20)
    ]
    key = ["dataset_index", "mechanism", "strength", "training_seed"]
    if len(carriers) != 500 or carriers.duplicated(key).any():
        raise RuntimeError("real B1 dry-run did not yield 500 unique M09 baseline carriers")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_SEMANTIC_V3_EXECUTION",
        "supersedes": "results/edbt_eab_revision/semantic_group_protocol_v2_freeze.json",
        "pre_output_incidents": [
            "v1 filtered P0 rows before baseline lookup",
            "v2 baseline carrier lookup omitted the mechanism key",
        ],
        "rows_written_before_v3": 0,
        "scientific_protocol_changed": False,
        "dry_run_gate": {"m09_primary_budget_carriers": 500, "duplicate_keys": 0},
        "bindings": [
            bind("scripts/run_semantic_group_governance_v2.py"),
            bind("tests/test_remaining_governance.py"),
            bind("results/edbt_eab_revision/remaining_governance_protocol_freeze.json"),
            bind("results/edbt_eab_revision/b1_multiseed_p2.csv"),
            bind("artifacts/sp6/sp6_bundle_manifest.csv"),
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
