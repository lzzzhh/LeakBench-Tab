#!/usr/bin/env python3
"""Freeze the confirmatory corrected_v2 protocol after disjoint pilot review."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "corrected_v2" / "protocol_freeze.json"
FILES = (
    "configs/paper/corrected_v2.yaml",
    "src/leakbench/datasets.py",
    "src/leakbench/mechanisms/__init__.py",
    "src/leakbench/models/core_models.py",
    "experiments/leakbench/run_corrected_core.py",
    "scripts/audit_mechanism_strength.py",
    "scripts/analyze_corrected_v2.py",
    "results/corrected_v2/pilot_strength_audit.csv",
    "results/corrected_v2/pilot_protocol_v2_cells.csv",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Protocol is already frozen: {OUTPUT}")
    entries = []
    for relative in FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        entries.append({"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_CONFIRMATORY_RUN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_namespace": "confirmatory",
        "pilot_namespace": "pilot",
        "pilot_cells": 1980,
        "pilot_successes": 1980,
        "allowed_post_freeze_changes": [
            "runtime-only bug fixes documented with old/new hashes",
            "analysis changes that do not inspect or alter training outcomes",
        ],
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
