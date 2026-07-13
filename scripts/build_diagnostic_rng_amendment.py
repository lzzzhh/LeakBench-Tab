#!/usr/bin/env python3
"""Build the post-unblinding canonical diagnostic RNG amendment.

The frozen four-method diagnostic suite used the injection seed for MI, while
the frozen core/headline diagnostic and immutable task manifest used the
pre-specified fixed MI seed 42.  This builder does not rerun or select a
diagnostic.  It replaces the three localization metrics on every MI row with
the already-frozen task-manifest values and preserves every original field on
the other three methods.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_ID = "diagnostic_mi_fixed_seed_42_v1"
METHODS = (
    "mutual_information", "absolute_correlation", "lr_coefficient", "rf_permutation",
)
IDENTITY_COLUMNS = (
    "dataset_id", "mechanism", "strength", "seed", "method",
)
TASK_IDENTITY_COLUMNS = IDENTITY_COLUMNS[:-1]
METRIC_COLUMNS = (
    "localization_ap", "localization_normalized_ap", "top5_recall",
)
TASK_METRIC_COLUMNS = (
    "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall",
)
TASK_LINEAGE_COLUMNS = (
    "dataset_index", "dataset_namespace", "task_hash", "split_hash",
    "bundle_sha256", "n_leak",
)
PROVENANCE_COLUMNS = (
    "rng_amendment_id", "rng_amendment_applied", "diagnostic_rng_policy",
    "metric_source",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise ValueError(f"path is outside repository: {path}") from error


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _finite(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} has non-finite values in {list(columns)}")


def frame_sha256(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """Hash ordered scientific records independent of the input CSV dialect."""
    ordered = frame.loc[:, list(columns)].sort_values(list(IDENTITY_COLUMNS)).reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_canonical_frame(
    raw: pd.DataFrame, task_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return amended cells and fail-closed identity/preservation checks."""
    raw_required = set(IDENTITY_COLUMNS + METRIC_COLUMNS + TASK_LINEAGE_COLUMNS) | {
        "diagnostic_run_id", "status",
    }
    task_required = set(TASK_IDENTITY_COLUMNS + TASK_METRIC_COLUMNS + TASK_LINEAGE_COLUMNS)
    _require_columns(raw, raw_required, "raw diagnostic table")
    _require_columns(task_manifest, task_required, "task manifest")
    if raw.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("raw diagnostic table has a duplicate scientific identity")
    if task_manifest.duplicated(list(TASK_IDENTITY_COLUMNS)).any():
        raise ValueError("task manifest has a duplicate scientific identity")
    if not (raw["status"].astype(str) == "SUCCESS").all():
        raise ValueError("raw diagnostic table contains a non-success row")
    if set(raw["method"].astype(str)) != set(METHODS):
        raise ValueError("raw diagnostic method set differs from the frozen four methods")
    _finite(raw, METRIC_COLUMNS, "raw diagnostic table")
    _finite(task_manifest, TASK_METRIC_COLUMNS, "task manifest")

    mi_mask = raw["method"].astype(str) == "mutual_information"
    raw_mi = raw.loc[mi_mask].copy()
    raw_non_mi = raw.loc[~mi_mask].copy()
    if len(raw_mi) != len(task_manifest):
        raise ValueError(
            f"MI/task-manifest coverage differs: {len(raw_mi)}/{len(task_manifest)}"
        )
    if raw_mi.duplicated(list(TASK_IDENTITY_COLUMNS)).any():
        raise ValueError("MI rows have a duplicate task identity")

    replacement_columns = list(TASK_IDENTITY_COLUMNS + TASK_LINEAGE_COLUMNS + TASK_METRIC_COLUMNS)
    checked = raw_mi.merge(
        task_manifest[replacement_columns],
        on=list(TASK_IDENTITY_COLUMNS),
        how="inner",
        suffixes=("", "_replacement"),
        validate="one_to_one",
    )
    if len(checked) != len(raw_mi):
        raise ValueError("MI rows and replacement task manifest are not identity-complete")
    for column in TASK_LINEAGE_COLUMNS:
        replacement = f"{column}_replacement"
        if column in {"dataset_index", "n_leak"}:
            equal = pd.to_numeric(checked[column]).to_numpy() == pd.to_numeric(checked[replacement]).to_numpy()
        else:
            equal = checked[column].astype(str).to_numpy() == checked[replacement].astype(str).to_numpy()
        if not bool(np.all(equal)):
            raise ValueError(f"MI replacement {column} differs from raw task identity")

    replacement = task_manifest.set_index(list(TASK_IDENTITY_COLUMNS))
    mi_index = pd.MultiIndex.from_frame(raw_mi[list(TASK_IDENTITY_COLUMNS)])
    aligned = replacement.loc[mi_index]
    canonical = raw.copy()
    canonical.loc[mi_mask, "localization_ap"] = aligned["diagnostic_ap"].to_numpy(float)
    canonical.loc[mi_mask, "localization_normalized_ap"] = aligned[
        "diagnostic_normalized_ap"
    ].to_numpy(float)
    canonical.loc[mi_mask, "top5_recall"] = aligned["top5_recall"].to_numpy(float)

    canonical["rng_amendment_id"] = AMENDMENT_ID
    canonical["rng_amendment_applied"] = mi_mask.to_numpy(bool)
    policy = {
        "mutual_information": "fixed_seed_42",
        "absolute_correlation": "deterministic_no_rng",
        "lr_coefficient": "frozen_injection_seed",
        "rf_permutation": "frozen_injection_seed",
    }
    canonical["diagnostic_rng_policy"] = canonical["method"].map(policy)
    canonical["metric_source"] = np.where(
        mi_mask,
        "results/corrected_v2/task_bundles/task_manifest.csv",
        "results/corrected_v2/diagnostic_confirmatory_cells.csv",
    )

    scientific_identity_exact = canonical[list(IDENTITY_COLUMNS)].equals(
        raw[list(IDENTITY_COLUMNS)]
    )
    non_mi_existing_columns_exact = canonical.loc[
        ~mi_mask, raw.columns
    ].reset_index(drop=True).equals(raw_non_mi.reset_index(drop=True))
    replacement_exact = all(
        np.allclose(
            canonical.loc[mi_mask, target].to_numpy(float),
            aligned[source].to_numpy(float),
            atol=0.0,
            rtol=0.0,
        )
        for target, source in zip(METRIC_COLUMNS, TASK_METRIC_COLUMNS)
    )
    if not scientific_identity_exact:
        raise RuntimeError("canonical amendment changed scientific row identities")
    if not non_mi_existing_columns_exact:
        raise RuntimeError("canonical amendment changed a non-MI original field")
    if not replacement_exact:
        raise RuntimeError("canonical MI metrics differ from the task-manifest replacement")
    if canonical.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("canonical amendment introduced duplicate identities")

    digest_columns = IDENTITY_COLUMNS + METRIC_COLUMNS
    audit = {
        "rows_total": int(len(canonical)),
        "rows_replaced": int(mi_mask.sum()),
        "rows_preserved": int((~mi_mask).sum()),
        "scientific_identity_exact": scientific_identity_exact,
        "non_mi_existing_columns_exact": non_mi_existing_columns_exact,
        "mi_replacement_metrics_exact": replacement_exact,
        "raw_mi_metrics_sha256": frame_sha256(raw_mi, digest_columns),
        "replacement_mi_metrics_sha256": frame_sha256(canonical.loc[mi_mask], digest_columns),
        "canonical_mi_metrics_sha256": frame_sha256(canonical.loc[mi_mask], digest_columns),
        "preserved_non_mi_rows_sha256": hashlib.sha256(
            raw_non_mi.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
        ).hexdigest(),
    }
    return canonical, audit


def _validate_exact_design(raw: pd.DataFrame, tasks: pd.DataFrame) -> None:
    if len(raw) != 22_000 or len(tasks) != 5_500:
        raise ValueError(f"expected 22,000 raw rows and 5,500 tasks, got {len(raw)}/{len(tasks)}")
    counts = raw.groupby("method").size().to_dict()
    if counts != {method: 5_500 for method in METHODS}:
        raise ValueError(f"frozen diagnostic method coverage changed: {counts}")
    if raw["diagnostic_run_id"].astype(str).duplicated().any():
        raise ValueError("raw diagnostic run IDs are not unique")


def initialize_freeze(
    freeze_path: Path, raw_path: Path, task_path: Path, raw_manifest_path: Path,
    diagnostic_protocol_path: Path,
) -> None:
    if freeze_path.exists():
        raise FileExistsError(f"refusing to overwrite amendment freeze: {freeze_path}")
    raw = pd.read_csv(raw_path)
    tasks = pd.read_csv(task_path)
    _validate_exact_design(raw, tasks)
    protocol = load_json(diagnostic_protocol_path)
    runner_entry = protocol.get("frozen_files", {}).get(
        "experiments/leakbench/run_diagnostic_suite.py", {}
    )
    if protocol.get("status") != "FROZEN_BEFORE_DIAGNOSTIC_CONFIRMATORY_RUN":
        raise ValueError("source diagnostic protocol is not frozen")
    if runner_entry.get("sha256") != str(raw["code_hash"].iloc[0]):
        raise ValueError("raw diagnostic code hash differs from its frozen protocol")
    raw_manifest = load_json(raw_manifest_path)
    if raw_manifest.get("output_sha256") != sha256(raw_path):
        raise ValueError("raw diagnostic manifest hash mismatch")

    payload = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_DIAGNOSTIC_RNG_AMENDMENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_id": AMENDMENT_ID,
        "discovery_phase": "post_unblinding",
        "reason": (
            "The frozen diagnostic-suite MI rows used random_state equal to the injection seed, "
            "whereas the frozen core/headline MI and immutable task manifest used random_state=42."
        ),
        "correction_rule": (
            "Replace localization_ap, localization_normalized_ap, and top5_recall on every and "
            "only mutual_information row with the identity-matched fixed-seed-42 values already "
            "stored in the frozen task manifest."
        ),
        "selection_rule": "method == mutual_information; no outcome- or value-dependent filtering",
        "no_tuning": True,
        "thresholds_changed": False,
        "model_outcomes_read_by_builder": False,
        "diagnostic_methods_changed": False,
        "expected_canonical_rows": 22_000,
        "expected_replaced_rows": 5_500,
        "expected_preserved_rows": 16_500,
        "replacement_random_state": 42,
        "scientific_identity": list(IDENTITY_COLUMNS),
        "replacement_metric_fields": list(METRIC_COLUMNS),
        "source_raw": {
            "path": relative(raw_path), "sha256": sha256(raw_path), "rows": 22_000,
            "mi_rng_policy": "injection_seed",
        },
        "source_raw_manifest": {
            "path": relative(raw_manifest_path), "sha256": sha256(raw_manifest_path),
        },
        "replacement_source": {
            "path": relative(task_path), "sha256": sha256(task_path), "rows": 5_500,
            "mi_rng_policy": "fixed_seed_42",
        },
        "source_diagnostic_protocol": {
            "path": relative(diagnostic_protocol_path),
            "sha256": sha256(diagnostic_protocol_path),
        },
        "builder": {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))},
        "outputs": {
            "canonical": "results/corrected_v2/diagnostic_canonical_cells.csv",
            "manifest": "results/corrected_v2/diagnostic_canonical_cells.manifest.json",
        },
        "rule": (
            "This amendment is deterministic and scope-locked. Editing the builder, source hashes, "
            "replacement rule, thresholds, or row selection requires a new amendment version."
        ),
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def validate_freeze(
    freeze_path: Path, raw_path: Path, task_path: Path, output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    freeze = load_json(freeze_path)
    expected = {
        "status": "FROZEN_BEFORE_DIAGNOSTIC_RNG_AMENDMENT",
        "amendment_id": AMENDMENT_ID,
        "expected_canonical_rows": 22_000,
        "expected_replaced_rows": 5_500,
        "expected_preserved_rows": 16_500,
        "replacement_random_state": 42,
        "no_tuning": True,
        "thresholds_changed": False,
        "model_outcomes_read_by_builder": False,
        "diagnostic_methods_changed": False,
    }
    for field, value in expected.items():
        if freeze.get(field) != value:
            raise ValueError(f"amendment freeze mismatch for {field}")
    if freeze.get("builder") != {"path": relative(Path(__file__)), "sha256": sha256(Path(__file__))}:
        raise ValueError("amendment freeze builder hash mismatch")
    if freeze.get("source_raw") != {
        "path": relative(raw_path), "sha256": sha256(raw_path), "rows": 22_000,
        "mi_rng_policy": "injection_seed",
    }:
        raise ValueError("amendment freeze raw source mismatch")
    if freeze.get("replacement_source") != {
        "path": relative(task_path), "sha256": sha256(task_path), "rows": 5_500,
        "mi_rng_policy": "fixed_seed_42",
    }:
        raise ValueError("amendment freeze replacement source mismatch")
    if freeze.get("outputs") != {
        "canonical": relative(output_path), "manifest": relative(manifest_path),
    }:
        raise ValueError("amendment freeze output paths mismatch")
    return freeze


def _atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def _atomic_json(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="results/corrected_v2/diagnostic_confirmatory_cells.csv")
    parser.add_argument("--raw-manifest", default="results/corrected_v2/diagnostic_confirmatory_cells.manifest.json")
    parser.add_argument("--task-manifest", default="results/corrected_v2/task_bundles/task_manifest.csv")
    parser.add_argument("--diagnostic-protocol", default="results/corrected_v2/diagnostic_protocol_freeze.json")
    parser.add_argument("--amendment-freeze", default="results/corrected_v2/diagnostic_rng_amendment_freeze.json")
    parser.add_argument("--output", default="results/corrected_v2/diagnostic_canonical_cells.csv")
    parser.add_argument("--manifest", default="results/corrected_v2/diagnostic_canonical_cells.manifest.json")
    parser.add_argument("--initialize-freeze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    paths = {
        name: (Path(value) if Path(value).is_absolute() else ROOT / value)
        for name, value in {
            "raw": args.raw, "raw_manifest": args.raw_manifest,
            "task_manifest": args.task_manifest, "diagnostic_protocol": args.diagnostic_protocol,
            "amendment_freeze": args.amendment_freeze, "output": args.output,
            "manifest": args.manifest,
        }.items()
    }
    if args.initialize_freeze:
        initialize_freeze(
            paths["amendment_freeze"], paths["raw"], paths["task_manifest"],
            paths["raw_manifest"], paths["diagnostic_protocol"],
        )
        print(json.dumps({
            "status": "FROZEN_BEFORE_DIAGNOSTIC_RNG_AMENDMENT",
            "freeze": relative(paths["amendment_freeze"]),
            "sha256": sha256(paths["amendment_freeze"]),
        }, indent=2))
        return 0

    for output in (paths["output"], paths["manifest"]):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite explicitly")
    freeze = validate_freeze(
        paths["amendment_freeze"], paths["raw"], paths["task_manifest"],
        paths["output"], paths["manifest"],
    )
    raw = pd.read_csv(paths["raw"])
    tasks = pd.read_csv(paths["task_manifest"])
    _validate_exact_design(raw, tasks)
    canonical, audit = build_canonical_frame(raw, tasks)
    if audit["rows_total"] != 22_000 or audit["rows_replaced"] != 5_500 or audit["rows_preserved"] != 16_500:
        raise RuntimeError("amendment row coverage differs from the frozen design")
    _atomic_csv(canonical, paths["output"])
    manifest = {
        "schema_version": 1,
        "status": "CANONICAL_DIAGNOSTIC_AMENDED",
        "amendment_id": AMENDMENT_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_freeze": {
            "path": relative(paths["amendment_freeze"]),
            "sha256": sha256(paths["amendment_freeze"]),
        },
        "builder": freeze["builder"],
        "raw_source": {"path": relative(paths["raw"]), "sha256": sha256(paths["raw"])},
        "replacement_source": {
            "path": relative(paths["task_manifest"]), "sha256": sha256(paths["task_manifest"]),
        },
        "canonical": {
            "path": relative(paths["output"]), "sha256": sha256(paths["output"]),
        },
        "expected_rows": 22_000,
        "rows_replaced": 5_500,
        "rows_preserved": 16_500,
        "methods": list(METHODS),
        "replacement_metric_fields": list(METRIC_COLUMNS),
        "identity_checks": {
            "scientific_identity_exact": audit["scientific_identity_exact"],
            "non_mi_existing_columns_exact": audit["non_mi_existing_columns_exact"],
            "mi_replacement_metrics_exact": audit["mi_replacement_metrics_exact"],
            "raw_duplicate_identities": False,
            "replacement_duplicate_identities": False,
            "canonical_duplicate_identities": False,
            "task_split_bundle_lineage_exact": True,
        },
        "record_hashes": {
            key: audit[key]
            for key in (
                "raw_mi_metrics_sha256", "replacement_mi_metrics_sha256",
                "canonical_mi_metrics_sha256", "preserved_non_mi_rows_sha256",
            )
        },
    }
    _atomic_json(manifest, paths["manifest"])
    print(json.dumps({
        "status": manifest["status"], "canonical": manifest["canonical"],
        "rows_replaced": manifest["rows_replaced"],
        "rows_preserved": manifest["rows_preserved"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
