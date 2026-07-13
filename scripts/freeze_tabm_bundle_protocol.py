#!/usr/bin/env python3
"""Freeze the cross-environment-safe official TabM bundle protocol."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/corrected_v2/tabm_bundle_protocol_freeze.json"
FILES = (
    "configs/paper/corrected_v2.yaml",
    "src/leakbench/models/official_tabm.py",
    "experiments/leakbench/run_corrected_tabm_bundle.py",
    "results/corrected_v2/task_bundles/task_manifest.csv",
    "results/corrected_v2/task_bundles/bundle_summary.json",
    "results/corrected_v2/tabm_bundle_pilot_tasks/task_manifest.csv",
    "results/corrected_v2/tabm_bundle_pilot_tasks/bundle_summary.json",
    "results/corrected_v2/tabm_bundle_pilot/tabm_official_cells.csv",
    "results/corrected_v2/tabm_bundle_pilot/tabm_official_cells_manifest.json",
    "results/corrected_v2/tabm_cross_environment_incident.json",
    "results/corrected_v2/tabm_official_environment_lock.json",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    entries = []
    for relative in FILES:
        path = ROOT / relative
        entries.append({"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    task_summary = json.loads((ROOT / "results/corrected_v2/task_bundles/bundle_summary.json").read_text())
    pilot = json.loads((ROOT / "results/corrected_v2/tabm_bundle_pilot/tabm_official_cells_manifest.json").read_text())
    if task_summary["task_count"] != 5500:
        raise RuntimeError("Confirmatory task bundle is incomplete")
    if pilot["success_cells"] != 18 or pilot["integrity_verified_cells"] != 18:
        raise RuntimeError("Bundle pilot did not clear its integrity gate")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_BUNDLE_CONFIRMATORY_RUN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": "results/corrected_v2/tabm_protocol_freeze.json",
        "reason": "immutable local task arrays eliminate cross-environment stochastic generation drift",
        "confirmatory_tasks": task_summary["task_count"],
        "pilot_success_cells": pilot["success_cells"],
        "pilot_integrity_verified_cells": pilot["integrity_verified_cells"],
        "model_identity": pilot["model_identity"],
        "adapter_kwargs": pilot["adapter_kwargs"],
        "files": entries,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
