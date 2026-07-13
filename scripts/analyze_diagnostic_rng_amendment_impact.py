#!/usr/bin/env python3
"""Compare the raw task-seeded MI sensitivity summary with its amendment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHOD_RANGE_FLOOR = 0.10


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(directory: Path) -> dict[str, Path]:
    return {
        "by_mechanism": directory / "diagnostic_method_by_mechanism.csv",
        "method_summary": directory / "diagnostic_method_summary.csv",
        "profiles": directory / "diagnostic_robustness_profiles.csv",
        "integrity": directory / "diagnostic_integrity.json",
    }


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"expected strict boolean, got {value!r}")


def _conclusions(paths: dict[str, Path]) -> dict[str, object]:
    table = pd.read_csv(paths["by_mechanism"])
    profiles = pd.read_csv(paths["profiles"]).set_index("mechanism")
    m03 = table[table["mechanism"].astype(str) == "M03"]
    best = m03.loc[m03["diagnostic_normalized_ap"].idxmax()]
    worst = m03.loc[m03["diagnostic_normalized_ap"].idxmin()]
    separation = bool(float(best["ci_low"]) > float(worst["ci_high"]))
    method_range = float(profiles.loc["M03", "between_diagnostic_range"])
    return {
        "D_METHOD_CONDITIONAL": "DESCRIPTIVE_ONLY",
        "paired_simultaneous_method_comparison_available": False,
        "legacy_threshold_pattern_present": bool(
            separation and method_range > METHOD_RANGE_FLOOR
        ),
        "m03_conservative_ci_separation": separation,
        "m03_between_diagnostic_range": method_range,
        "m04_low_across_all_evaluated_diagnostics": _strict_bool(
            profiles.loc["M04", "low_across_all_evaluated_diagnostics"]
        ),
        "m05_low_across_all_evaluated_diagnostics": _strict_bool(
            profiles.loc["M05", "low_across_all_evaluated_diagnostics"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-statistics", default="results/corrected_v2/diagnostic_raw_task_seed_statistics"
    )
    parser.add_argument("--canonical-statistics", default="results/corrected_v2/statistics")
    parser.add_argument(
        "--task-manifest", default="results/corrected_v2/task_bundles/task_manifest.csv"
    )
    parser.add_argument(
        "--output", default="results/corrected_v2/diagnostic_rng_amendment_impact.json"
    )
    args = parser.parse_args(argv)
    raw_dir = ROOT / args.raw_statistics
    canonical_dir = ROOT / args.canonical_statistics
    task_manifest_path = ROOT / args.task_manifest
    output = ROOT / args.output
    raw_paths = _paths(raw_dir)
    canonical_paths = _paths(canonical_dir)
    for path in [*raw_paths.values(), *canonical_paths.values(), task_manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw_table = pd.read_csv(raw_paths["by_mechanism"])
    canonical_table = pd.read_csv(canonical_paths["by_mechanism"])
    key = ["method", "mechanism"]
    if raw_table.duplicated(key).any() or canonical_table.duplicated(key).any():
        raise ValueError("diagnostic summary contains duplicate method/mechanism identities")
    merged = raw_table.merge(
        canonical_table, on=key, suffixes=("_raw", "_canonical"), validate="one_to_one"
    )
    if len(merged) != 44:
        raise ValueError("diagnostic comparison is not the complete 4 x 11 design")
    non_mi = merged["method"].astype(str) != "mutual_information"
    compared = ["diagnostic_normalized_ap", "ci_low", "ci_high"]
    if any(
        not np.array_equal(
            merged.loc[non_mi, f"{column}_raw"].to_numpy(),
            merged.loc[non_mi, f"{column}_canonical"].to_numpy(),
        )
        for column in compared
    ):
        raise ValueError("a non-MI diagnostic summary changed under the MI-only amendment")

    raw_mi = merged.loc[~non_mi].set_index("mechanism")
    task_manifest = pd.read_csv(task_manifest_path)
    primary = task_manifest.groupby("mechanism")["diagnostic_normalized_ap"].mean()
    canonical_mi = raw_mi["diagnostic_normalized_ap_canonical"]
    primary_mi = primary.loc[canonical_mi.index]
    if not np.allclose(canonical_mi, primary_mi, atol=1e-12, rtol=0):
        raise ValueError("canonical sensitivity MI does not match the headline fixed-seed-42 MI")

    before = _conclusions(raw_paths)
    after = _conclusions(canonical_paths)
    central_changed = before != after
    method_summary_raw = pd.read_csv(raw_paths["method_summary"]).set_index("method")
    method_summary_canonical = pd.read_csv(canonical_paths["method_summary"]).set_index("method")
    payload = {
        "schema_version": 1,
        "status": "CONCLUSIONS_CHANGED" if central_changed else "CENTRAL_CONCLUSIONS_UNCHANGED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment_id": "diagnostic_mi_fixed_seed_42_v1",
        "comparison_scope": "diagnostic localization only; no model-outcome columns consumed",
        "no_threshold_tuning": True,
        "before": before,
        "after": after,
        "central_conclusions_changed": central_changed,
        "mi_overall_normalized_ap": {
            "raw_task_seeded": float(
                method_summary_raw.loc["mutual_information", "diagnostic_normalized_ap"]
            ),
            "canonical_fixed_seed_42": float(
                method_summary_canonical.loc[
                    "mutual_information", "diagnostic_normalized_ap"
                ]
            ),
        },
        "mi_mechanism_max_absolute_point_change": float(
            np.max(np.abs(
                raw_mi["diagnostic_normalized_ap_canonical"].to_numpy()
                - raw_mi["diagnostic_normalized_ap_raw"].to_numpy()
            ))
        ),
        "canonical_mi_matches_headline_primary": True,
        "non_mi_method_mechanism_statistics_exact": True,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [*raw_paths.values(), *canonical_paths.values(), task_manifest_path]
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
