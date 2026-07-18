#!/usr/bin/env python3
"""Freeze semantic-group v2 after v1 failed before producing any output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/edbt_eab_revision/semantic_group_protocol_v2_freeze.json"


def bind(path):
    target = ROOT / path
    return {"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "size_bytes": target.stat().st_size}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    failed_output = ROOT / "results/edbt_eab_revision/semantic_m09_cells.csv"
    if failed_output.exists():
        raise RuntimeError("v1 failure unexpectedly produced an output; audit before v2")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_SEMANTIC_V2_EXECUTION",
        "supersedes_semantic_runner_only": "scripts/run_remaining_governance.py semantic",
        "v1_incident": {
            "stage": "first-key frozen-baseline lookup",
            "cause": "P0 baseline rows were filtered out by the 20% budget filter before lookup",
            "experiment_rows_written": 0,
            "scientific_protocol_changed": False,
        },
        "correction": "carry strict/full AUCs from the unique encoded-cost P3 row at the primary 20% budget",
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
