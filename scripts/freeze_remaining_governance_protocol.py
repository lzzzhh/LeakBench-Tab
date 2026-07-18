#!/usr/bin/env python3
"""Freeze the two prospective remaining-governance protocols before execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/edbt_eab_revision/remaining_governance_protocol_freeze.json"


def binding(path: str):
    target = ROOT / path
    return {"path": path, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "size_bytes": target.stat().st_size}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    natural = pd.read_csv(ROOT / "results/corrected_v2/natural_task_summary.csv")
    manifest = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_EXECUTION",
        "frozen_at_utc": "2026-07-18",
        "natural_governance": {
            "evidence_tier": "descriptive_external_validity_sensitivity",
            "tasks": natural.task.tolist(),
            "model": "lr",
            "training_seeds": [13, 42, 2026],
            "governance_seeds": list(range(2026071700, 2026071720)),
            "budget_fraction": 0.20,
            "cost_unit": "retained natural feature",
            "estimand": "P3 blind-MI SDR minus within-key mean P2 random SDR",
            "uncertainty": "case-level descriptive bootstrap plus exact sign-flip over five fixed cases",
            "population_claim_allowed": False,
            "expected_rows": 315,
        },
        "semantic_group_governance": {
            "evidence_tier": "representation_cost_sensitivity",
            "model": "lr",
            "mechanism_rerun": "M09",
            "reason": "M09 alone maps one semantic source field to eight one-hot encoded columns; all other frozen registry mechanisms have identity semantic-to-encoded group mappings",
            "group_score": "maximum train-side mutual information among columns in a semantic group",
            "random_policy": "uniform without replacement over semantic groups",
            "budget_fraction": 0.20,
            "budget_rounding": "max(1, round(n_semantic_groups * fraction))",
            "governance_seeds": list(range(2026071700, 2026071720)),
            "estimand": "P3 blind-MI SDR minus within-key mean P2 random SDR",
            "expected_m09_rows": int((manifest.mechanism == "M09").sum() * 21),
            "full_panel_recomposition": "replace encoded-cost M09 paired effects with semantic-cost M09 effects; retain 5,000 identity-mapped non-M09 keys",
        },
        "immutable_inputs": [
            binding("results/corrected_v2/natural_protocol_v2_freeze.json"),
            binding("results/corrected_v2/natural_cells.csv"),
            binding("results/corrected_v2/natural_task_summary.csv"),
            binding("artifacts/sp6/sp6_bundle_manifest.csv"),
            binding("results/edbt_eab_revision/b1_multiseed_p2.csv"),
        ],
        "code": [
            binding("scripts/run_remaining_governance.py"),
            binding("scripts/analyze_remaining_governance.py"),
            binding("scripts/freeze_remaining_governance_protocol.py"),
            binding("tests/test_remaining_governance.py"),
        ],
        "failure_rules": [
            "Any source or bundle hash mismatch blocks execution.",
            "Any failed, duplicate, or missing cell blocks claim derivation.",
            "Natural results remain case-study evidence regardless of sign.",
            "No semantic group may be inferred from feature names after observing outcomes.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

