#!/usr/bin/env python3
"""Freeze the official TabM protocol after its disjoint GPU pilot."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "corrected_v2" / "tabm_protocol_freeze.json"
FILES = (
    "configs/paper/corrected_v2.yaml",
    "src/leakbench/datasets.py",
    "src/leakbench/mechanisms/__init__.py",
    "src/leakbench/models/official_tabm.py",
    "experiments/leakbench/run_corrected_tabm.py",
    "results/corrected_v2/tabm_pilot/tabm_official_cells.csv",
    "results/corrected_v2/tabm_pilot/tabm_official_cells_manifest.json",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    files = []
    for relative in FILES:
        path = ROOT / relative
        files.append({"path": relative, "sha256": sha256(path), "size_bytes": path.stat().st_size})
    pilot = json.loads((ROOT / FILES[-1]).read_text(encoding="utf-8"))
    if pilot["success_cells"] != pilot["requested_cells"] or pilot["failure_cells"]:
        raise RuntimeError("Official TabM pilot did not pass its completion gate")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_TABM_CONFIRMATORY_RUN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_cells": pilot["requested_cells"],
        "pilot_result_sha256": pilot["result_sha256"],
        "model_identity": pilot["model_identity"],
        "required_tabm_version": pilot["required_tabm_version"],
        "adapter_kwargs": pilot["adapter_kwargs"],
        "remote_environment": {
            "python": "3.11.15",
            "torch": "2.5.1+cu121",
            "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "environment": "leakbench-tabm-official",
        },
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
