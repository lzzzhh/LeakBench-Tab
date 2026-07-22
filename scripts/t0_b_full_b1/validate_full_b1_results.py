#!/usr/bin/env python3
"""T0-B Full-B1 result validator — strict shard, merge, and scientific gates."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.merge_contract import (
    build_source_shard_snapshot,
    validate_global_merge_candidate,
    validate_global_scope,
    validate_plan,
    validate_plan_schema,
    validate_shard_set,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_gzip_csv(path: Path, label: str) -> pd.DataFrame:
    try:
        raw = gzip.decompress(path.read_bytes()).decode("utf-8")
        return pd.read_csv(io.StringIO(raw))
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError,
            pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"{label} read error: {exc}") from exc


def validate_counts(
    governed: pd.DataFrame,
    baseline: pd.DataFrame,
    manifest: dict | None = None,
) -> list[str]:
    expected = manifest or {"baseline_rows": 11000, "governed_rows": 792000}
    errors = []
    if len(baseline) != expected["baseline_rows"]:
        errors.append(
            f"baseline row count mismatch: actual={len(baseline)}, "
            f"expected={expected['baseline_rows']}"
        )
    if len(governed) != expected["governed_rows"]:
        errors.append(
            f"governed row count mismatch: actual={len(governed)}, "
            f"expected={expected['governed_rows']}"
        )
    return errors


def validate_run_ids(governed: pd.DataFrame, baseline: pd.DataFrame | None = None) -> list[str]:
    frames = [governed] if baseline is None else [baseline, governed]
    run_ids = pd.concat([frame["run_id"] for frame in frames], ignore_index=True)
    errors = []
    if run_ids.isna().any() or (run_ids.astype(str).str.len() == 0).any():
        errors.append("null or empty run_ids")
    if run_ids.duplicated().any():
        errors.append("duplicate run_ids")
    return errors


def validate_selection_closure(governed: pd.DataFrame, selections: pd.DataFrame) -> list[str]:
    governed_hashes = Counter(governed["selection_hash"].astype(str))
    selection_hashes = Counter(selections["selection_hash"].astype(str))
    if governed_hashes != selection_hashes:
        return ["selection_hash multiset differs between governed and selection ledgers"]
    return []


def validate_scientific_domains(
    governed: pd.DataFrame,
    baseline: pd.DataFrame,
    run_rows: list[dict],
) -> list[str]:
    errors = []
    expected_policies = {row["policy"] for row in run_rows if row["run_type"] == "governed"}
    expected_contracts = {row["contract"] for row in run_rows if row["run_type"] == "governed"}
    expected_budgets = {row["budget_bp"] for row in run_rows if row["run_type"] == "governed"}
    if set(governed["policy"]) != expected_policies:
        errors.append("governed policy domain mismatch")
    if set(governed["contract"]) != expected_contracts:
        errors.append("governed contract domain mismatch")
    if set(governed["budget_bp"]) != expected_budgets:
        errors.append("governed budget domain mismatch")
    if set(baseline["baseline_type"]) != {"strict", "full"}:
        errors.append("baseline_type domain mismatch")
    if set(baseline["learner"]) != {"lr"} or set(governed["learner"]) != {"lr"}:
        errors.append("learner domain mismatch")

    p2 = governed[governed["policy"] == "P2"]
    if set(p2["governance_seed"]) != set(range(20)):
        errors.append("P2 governance_seed domain mismatch")
    deterministic = governed[governed["policy"] != "P2"]
    if set(deterministic["governance_seed"]) != {-1}:
        errors.append("deterministic policy governance_seed must equal -1")

    numeric_columns = ["auc"]
    for column in numeric_columns:
        if not np.isfinite(pd.to_numeric(baseline[column], errors="coerce")).all():
            errors.append(f"baseline {column} contains non-finite values")
    for column in ["strict_auc", "full_auc", "governed_auc", "legacy_sdr", "realized_cost"]:
        if not np.isfinite(pd.to_numeric(governed[column], errors="coerce")).all():
            errors.append(f"governed {column} contains non-finite values")
    for column in ["auc"]:
        values = pd.to_numeric(baseline[column], errors="coerce")
        if ((values < 0) | (values > 1)).any():
            errors.append(f"baseline {column} outside [0,1]")
    for column in ["strict_auc", "full_auc", "governed_auc"]:
        values = pd.to_numeric(governed[column], errors="coerce")
        if ((values < 0) | (values > 1)).any():
            errors.append(f"governed {column} outside [0,1]")
    return errors


def validate_full_result(
    *,
    plan_path: Path,
    shard_root: Path,
    merged_dir: Path,
    expected_mode: str,
) -> tuple[list[str], dict]:
    errors = []
    try:
        plan_manifest = json.loads(plan_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"plan manifest read error: {exc}"], {}
    if not isinstance(plan_manifest, dict):
        return ["plan manifest is not a JSON object"], {}

    errors.extend(validate_plan_schema(plan_manifest, expected_mode))
    key_rows = []
    run_rows = []
    if not errors:
        plan_errors, key_rows, run_rows = validate_plan(plan_manifest, plan_path.parent)
        errors.extend(plan_errors)
    if not errors:
        errors.extend(validate_global_scope(plan_manifest, key_rows, run_rows))
    if errors:
        return errors, {}

    plan_sha = _sha256(plan_path)
    admission = validate_shard_set(
        plan_manifest=plan_manifest,
        plan_manifest_sha256=plan_sha,
        plan_dir=plan_path.parent,
        shard_root=shard_root,
        expected_mode=expected_mode,
    )
    if not admission.is_valid:
        return [f"shard admission: {error}" for error in admission.errors], {}

    planned_shards = sorted({row["shard_id"] for row in key_rows})
    snapshot = build_source_shard_snapshot(shard_root, planned_shards)
    merge_validation = validate_global_merge_candidate(
        merged_dir=merged_dir,
        plan_manifest=plan_manifest,
        plan_manifest_sha256=plan_sha,
        planned_shard_ids=planned_shards,
        snapshot=snapshot,
        shard_root=shard_root,
        run_rows=run_rows,
        key_rows=key_rows,
    )
    if not merge_validation.is_valid:
        return [f"merged candidate: {error}" for error in merge_validation.errors], {}

    try:
        baseline = _read_gzip_csv(merged_dir / "baseline_ledger.csv.gz", "baseline ledger")
        governed = _read_gzip_csv(merged_dir / "governed_ledger.csv.gz", "governed ledger")
        selections = _read_gzip_csv(merged_dir / "selection_ledger.csv.gz", "selection ledger")
    except ValueError as exc:
        return [str(exc)], {}

    errors.extend(validate_counts(governed, baseline, plan_manifest))
    errors.extend(validate_run_ids(governed, baseline))
    errors.extend(validate_selection_closure(governed, selections))
    errors.extend(validate_scientific_domains(governed, baseline, run_rows))

    receipt = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "mode": expected_mode,
        "plan_manifest_sha256": plan_sha,
        "plan_declared_tool_seal_sha": plan_manifest["tool_seal_sha"],
        "merge_manifest_sha256": _sha256(merged_dir / "merge_manifest.json"),
        "canonical_keys": len(key_rows),
        "planned_shards": len(planned_shards),
        "baseline_rows": len(baseline),
        "governed_rows": len(governed),
        "selection_rows": len(selections),
        "failure_rows": merge_validation.failure_rows,
        "downstream_rows": len(baseline) + len(governed),
        "errors": errors,
    }
    return errors, receipt


def _validate_synthetic_fixture(fixture: Path) -> list[str]:
    try:
        baseline = _read_gzip_csv(fixture / "baseline_ledger.csv.gz", "baseline ledger")
        governed = _read_gzip_csv(fixture / "governed_ledger.csv.gz", "governed ledger")
    except ValueError as exc:
        return [str(exc)]
    return validate_run_ids(governed, baseline)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-fixture", default="")
    parser.add_argument("--output-dir", default=str(ROOT / "results/edbt_t0_b_full_b1"))
    parser.add_argument(
        "--plan-manifest",
        default=str(ROOT / "results/edbt_t0_b_full_b1_preflight/full_b1_plan_manifest.json"),
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--receipt-out", default="")
    parser.add_argument("--expected-not-executed-ok", action="store_true")
    args = parser.parse_args()

    if args.synthetic_fixture:
        errors = _validate_synthetic_fixture(Path(args.synthetic_fixture))
        if errors:
            print("FULL_B1_RESULT_VALIDATION_FAIL")
            for error in errors:
                print(f"  {error}")
            sys.exit(1)
        print("FULL_B1_RESULT_VALIDATION_PASS")
        return

    output_dir = Path(args.output_dir)
    shard_root = output_dir / "shards"
    merged_dir = output_dir / "merged"
    if not shard_root.exists() or not any(shard_root.iterdir()):
        print("EXPECTED_NOT_EXECUTED")
        sys.exit(0 if args.expected_not_executed_ok else 42)
    if not merged_dir.exists():
        print("FULL_B1_RESULT_VALIDATION_FAIL")
        print(f"  merged output missing: {merged_dir}")
        sys.exit(1)

    errors, receipt = validate_full_result(
        plan_path=Path(args.plan_manifest),
        shard_root=shard_root,
        merged_dir=merged_dir,
        expected_mode="synthetic" if args.synthetic else "production",
    )
    if args.receipt_out and receipt:
        receipt_path = Path(args.receipt_out)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if errors:
        print("FULL_B1_RESULT_VALIDATION_FAIL")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    print("FULL_B1_RESULT_VALIDATION_PASS")
    for field in [
        "canonical_keys", "planned_shards", "baseline_rows", "governed_rows",
        "selection_rows", "failure_rows", "downstream_rows",
    ]:
        print(f"{field}={receipt[field]}")


if __name__ == "__main__":
    main()
