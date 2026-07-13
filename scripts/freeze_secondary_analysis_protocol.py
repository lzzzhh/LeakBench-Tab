#!/usr/bin/env python3
"""Freeze secondary corrected_v2 analyses before confirmatory completion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [
        ROOT / "configs/paper/corrected_v2.yaml",
        ROOT / "scripts/analyze_model_contrasts_v2.py",
        ROOT / "scripts/analyze_secondary_v2.py",
        ROOT / "scripts/analyze_cluster_sensitivity.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    output = ROOT / "results/corrected_v2/secondary_analysis_protocol_freeze.json"
    if output.exists():
        raise FileExistsError(output)
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_CONFIRMATORY_COMPLETION",
        "evidence_tier": "confirmatory",
        "analyses": {
            "model_category_contrasts": "five model-specific simple-minus-structured intervals; Holm family",
            "mechanism_model_heterogeneity": "55 pre-specified mechanism-by-model estimates; model extrema descriptive only",
            "strength_response": "11 standardized linear strength slopes; Holm family",
            "cluster_sensitivity": "M08 entity and M09 source cluster bootstrap using retained test predictions",
        },
        "bootstrap_repetitions": 20000,
        "bootstrap_seed": 20260713,
        "cluster_inner_repetitions": 200,
        "cluster_outer_repetitions": 5000,
        "frozen_files": {
            str(path.relative_to(ROOT)): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in paths
        },
        "rule": "Any modification requires a new protocol version and invalidates this freeze for confirmatory claims.",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
