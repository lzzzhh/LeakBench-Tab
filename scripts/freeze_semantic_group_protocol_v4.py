#!/usr/bin/env python3
"""Freeze semantic-group v4 with globally unique run identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/edbt_eab_revision/semantic_group_protocol_v4_freeze.json"


def bind(path):
    target = ROOT / path
    return {"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "size_bytes": target.stat().st_size}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    formal = ROOT / "results/edbt_eab_revision/semantic_m09_cells.csv"
    if formal.exists():
        raise RuntimeError("formal v4 output already exists")
    excluded = "artifacts/archive/semantic_group_v3_duplicate_ids/semantic_m09_cells.csv"
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_SEMANTIC_V4_EXECUTION",
        "supersedes": "results/edbt_eab_revision/semantic_group_protocol_v3_freeze.json",
        "scientific_protocol_changed": False,
        "v3_incident": {
            "cause": "run_id omitted dataset_index while bundle_key repeats across datasets",
            "rows": 10500,
            "successful_rows": 10500,
            "duplicate_run_ids": 8925,
            "disposition": "excluded wholesale; no deduplication or reuse",
            "artifact": bind(excluded),
        },
        "v4_correction": "run_id identity includes dataset_index and the full frozen key",
        "bindings": [
            bind("scripts/run_semantic_group_governance_v2.py"),
            bind("scripts/analyze_remaining_governance.py"),
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
