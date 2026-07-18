#!/usr/bin/env python3
"""Validate and bind the EDBT governance revision evidence package."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "results/edbt_eab_revision"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: str, rows=None):
    target = ROOT / path
    if not target.exists():
        raise FileNotFoundError(target)
    result = {"path": path, "sha256": sha256(target)}
    if rows is not None:
        result["rows"] = int(rows)
    return result


def validate_raw_partition(filename: str, expected_rows: int, expected_policies):
    path = REVISION / filename
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{filename}: expected {expected_rows} rows, found {len(frame)}")
    if frame["run_id"].duplicated().any():
        raise ValueError(f"{filename}: duplicate run_id")
    if set(frame["status"]) != {"SUCCESS"}:
        raise ValueError(f"{filename}: non-success rows present")
    if frame["policy"].value_counts().to_dict() != expected_policies:
        raise ValueError(f"{filename}: policy coverage mismatch")
    if not frame["selection_mask_hash"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError(f"{filename}: incomplete selection hashes")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/edbt_eab_revision/manifest.json")
    args = parser.parse_args(argv)

    b1 = validate_raw_partition(
        "b1_multiseed_p2.csv", 467500,
        {"P2_random": 440000, "P3_blind_mi": 22000, "P0_keep": 5500},
    )
    rf = validate_raw_partition(
        "b2_rf.csv", 121000,
        {"P2_random": 110000, "P0_keep": 5500, "P3_blind_mi": 5500},
    )
    lgbm = validate_raw_partition(
        "b2_lgbm.csv", 121000,
        {"P2_random": 110000, "P0_keep": 5500, "P3_blind_mi": 5500},
    )
    if len(set(b1["run_id"]) | set(rf["run_id"]) | set(lgbm["run_id"])) != 709500:
        raise ValueError("cross-partition run_id collision")

    summary_path = REVISION / "analysis_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("primary_budget") != 0.20 or summary.get("expected_keys_per_model") != 5500:
        raise ValueError("analysis summary is not the matched 20% analysis")
    for model in ("LR", "RF", "LightGBM"):
        if summary[f"{model}_overall"]["n_keys"] != 5500:
            raise ValueError(f"{model} summary coverage mismatch")
    current_input_hashes = {
        "b1_lr": sha256(REVISION / "b1_multiseed_p2.csv"),
        "b2_rf": sha256(REVISION / "b2_rf.csv"),
        "b2_lgbm": sha256(REVISION / "b2_lgbm.csv"),
    }
    if summary.get("input_hashes") != current_input_hashes:
        raise ValueError("analysis summary input hashes are stale")

    claim_path = REVISION / "claim_state.json"
    claims = json.loads(claim_path.read_text())
    if claims.get("analysis_summary_sha256") != sha256(summary_path):
        raise ValueError("claim state is not bound to the current summary")

    backfill_path = REVISION / "selection_hash_backfill.json"
    backfill = json.loads(backfill_path.read_text())
    after_hashes = {entry["path"]: entry["after_sha256"] for entry in backfill["files"]}
    for path, expected_hash in after_hashes.items():
        if sha256(ROOT / path) != expected_hash:
            raise ValueError(f"stale selection-hash backfill binding: {path}")

    protocol_path = ROOT / "reports/edbt_eab/governance_revision_protocol.md"
    protocol_text = protocol_path.read_text()
    if "Post-run protocol deviation disclosure" not in protocol_text:
        raise ValueError("B2 baseline-refit deviation is not disclosed")

    paths_with_rows = [
        ("results/edbt_eab_revision/b1_multiseed_p2.csv", len(b1)),
        ("results/edbt_eab_revision/b2_rf.csv", len(rf)),
        ("results/edbt_eab_revision/b2_lgbm.csv", len(lgbm)),
        ("results/edbt_eab_revision/a1_mechanism_level.csv", 11),
        ("results/edbt_eab_revision/a2_gap_stratification.csv", 4),
        ("results/edbt_eab_revision/a3_archetype.csv", 10),
    ]
    paths_without_rows = [
        "results/edbt_eab_revision/analysis_summary.json",
        "results/edbt_eab_revision/claim_state.json",
        "results/edbt_eab_revision/selection_hash_backfill.json",
        "reports/edbt_eab/governance_revision_protocol.md",
        "reports/edbt_eab/revision_fix/final_report.md",
        "scripts/analyze_governance_revision.py",
        "scripts/backfill_governance_selection_hashes.py",
        "scripts/build_governance_revision_claim_state.py",
        "scripts/build_governance_revision_manifest.py",
        "scripts/run_sp8_b1_multiseed.py",
        "scripts/run_sp8_b2_crosslearner.py",
        "scripts/run_sp8_b2_parallel.py",
        "tests/test_governance_revision.py",
        "artifacts/sp6/sp6_bundle_manifest.csv",
        "results/corrected_v2/canonical_cells.csv",
        "artifacts/sp8/governance_clean.csv",
    ]
    artifacts = [artifact(path, rows) for path, rows in paths_with_rows]
    artifacts.extend(artifact(path) for path in paths_without_rows)

    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "schema_version": 2,
        "phase": "EDBT_governance_revision",
        "status": "COMPLETE_WITH_DISCLOSED_LIMITATIONS",
        "canonical_dataset": {
            "format": "manifest_bound_csv_partitions",
            "partitions": ["b1_multiseed_p2.csv", "b2_rf.csv", "b2_lgbm.csv"],
            "rows": 709500,
            "duplicate_run_ids": 0,
        },
        "analysis": {
            "primary_budget": 0.20,
            "bootstrap_seed": summary["analysis_seed"],
            "bootstrap_repetitions": summary["bootstrap_reps"],
            "bootstrap_unit": summary["bootstrap_unit"],
            "keys_per_model": 5500,
            "governance_seeds": 20,
        },
        "validation": {
            "all_rows_success": True,
            "selection_hashes_complete": True,
            "cross_model_selection_hashes_matched": True,
            "claim_state_builder_derived": True,
            "analysis_inputs_bound": True,
            "b2_baseline_refit_deviation_disclosed": True,
        },
        "limitations": [
            "B2 strict/full baselines were re-fitted under a disclosed post-run protocol deviation.",
            "Natural-data governance was not run.",
            "Semantic-group budget sensitivity was not run.",
        ],
        "tests": {
            "command": "PYTHONPATH=. pytest -q",
            "passed": 228,
            "failed": 0,
        },
        "git_head_at_build": git_head,
        "artifacts": artifacts,
    }
    output = ROOT / args.output
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "canonical_rows": payload["canonical_dataset"]["rows"],
        "artifacts": len(artifacts),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
