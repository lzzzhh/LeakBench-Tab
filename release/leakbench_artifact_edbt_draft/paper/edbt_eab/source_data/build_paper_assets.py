#!/usr/bin/env python3
"""Build the minimal EDBT EA&B paper-table asset set from frozen evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "paper/edbt_eab/source_data/generated"

CLAIMS = ROOT / "results/corrected_v2/paper_claims.json"
CLAIM_STATE = ROOT / "results/corrected_v2/claim_state.json"
CANONICAL = ROOT / "results/corrected_v2/canonical_cells.csv"
CANONICAL_MANIFEST = ROOT / "results/corrected_v2/canonical_manifest.json"
STATS = ROOT / "results/corrected_v2/statistics"
NATURAL = ROOT / "results/corrected_v2/public_natural"
SP8_BOOTSTRAP = ROOT / "artifacts/sp8/bootstrap_analysis.json"
SP8_CLAIMS = ROOT / "artifacts/sp8/claims/claim_evidence_matrix_sp8.json"

MECHANISMS = [f"M{index:02d}" for index in range(1, 12)]
MECHANISM_NAMES = {
    "M01": "Direct Target Copy",
    "M02": "Noisy Target Proxy",
    "M03": "Nonlinear Target Transform",
    "M04": "Post-Outcome Aggregation",
    "M05": "Temporal Look-Ahead",
    "M06": "Redundant Leakage Cluster",
    "M07": "Sparse Subgroup Leakage",
    "M08": "Entity Leakage",
    "M09": "Source Leakage",
    "M10": "Mixed Leakage",
    "M11": "Graph-Mediated Leakage",
}
CATEGORIES = {
    "M01": "simple", "M02": "simple", "M03": "boundary",
    "M04": "structured", "M05": "structured", "M06": "simple",
    "M07": "boundary", "M08": "structured", "M09": "structured",
    "M10": "simple", "M11": "boundary",
}
PROFILE_CLAIMS = {
    "M03": "m03_profile",
    "M08": "m08_profile",
    "M09": "m09_counterexample",
}
NATURAL_BOUNDARIES = {
    "BankMarketing": "Before call completion",
    "LendingClub": "At loan origination",
    "BTSFlights": "Before scheduled departure",
    "ChicagoFood": "Before inspection outcome",
    "NYC311": "At complaint creation",
}
OUTPUT_NAMES = (
    "main_results.csv",
    "governance_results.csv",
    "natural_cases.csv",
    "paper_asset_manifest.json",
)


class AssetError(ValueError):
    """Raised when a paper asset cannot be derived without ambiguity."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or key not in rows[0]:
        raise AssetError(f"{relative(path)} lacks key column {key}")
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise AssetError(f"{relative(path)} has duplicate {key} values")
    return indexed


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_release() -> tuple[dict[str, Any], dict[str, Any]]:
    if CLAIMS.read_bytes() != CLAIM_STATE.read_bytes():
        raise AssetError("paper_claims.json and claim_state.json are not byte-identical")
    claims = load_json(CLAIMS)
    canonical_manifest = load_json(CANONICAL_MANIFEST)
    canonical_sha256 = sha256(CANONICAL)
    if (
        canonical_manifest.get("status") != "CANONICAL"
        or canonical_manifest.get("canonical_sha256") != canonical_sha256
        or canonical_sha256
        != claims.get("provenance", {}).get("input_sha256", {}).get(
            "results/corrected_v2/canonical_cells.csv"
        )
    ):
        raise AssetError("corrected_v2 canonical or claim binding is not current")
    if claims.get("evidence_tier") != "confirmatory":
        raise AssetError("paper claims are not confirmatory evidence")
    return claims, canonical_manifest


def build_main_results(claims: dict[str, Any]) -> list[dict[str, Any]]:
    harm_path = STATS / "mechanism_summary.csv"
    detect_path = STATS / "detectability_mechanism_summary.csv"
    dose_path = STATS / "strength_dose_response.csv"
    harm = read_csv(harm_path, "mechanism")
    detect = read_csv(detect_path, "mechanism")
    dose = read_csv(dose_path, "mechanism")
    if set(harm) != set(MECHANISMS) or set(detect) != set(MECHANISMS) or set(dose) != set(MECHANISMS):
        raise AssetError("mechanism result identity set is not the complete M01-M11 registry")

    contrast = claims["claims"]["simple_vs_structured"]
    metrics = contrast["metrics"]
    rows: list[dict[str, Any]] = [{
        "row_type": "category_contrast",
        "result_id": "C1",
        "label": "Simple minus structured",
        "category": "category_contrast",
        "estimate": metrics["difference"],
        "ci_low": metrics["ci_low"],
        "ci_high": metrics["ci_high"],
        "holm_p": metrics["holm_p"],
        "primary_detectability": "",
        "detectability_ci_low": "",
        "detectability_ci_high": "",
        "strength_slope": "",
        "strength_ci_low": "",
        "strength_ci_high": "",
        "claim_id": "simple_vs_structured",
        "interpretation_status": contrast["status"],
    }]
    for mechanism in MECHANISMS:
        harm_row = harm[mechanism]
        detect_row = detect[mechanism]
        dose_row = dose[mechanism]
        category = CATEGORIES[mechanism]
        if harm_row["category"] != category or detect_row["category"] != category or dose_row["category"] != category:
            raise AssetError(f"category mismatch for {mechanism}")
        claim_id = PROFILE_CLAIMS.get(mechanism, "")
        status = claims["claims"][claim_id]["status"] if claim_id else "DESCRIPTIVE_ONLY"
        rows.append({
            "row_type": "mechanism_profile",
            "result_id": mechanism,
            "label": MECHANISM_NAMES[mechanism],
            "category": category,
            "estimate": harm_row["paired_harm"],
            "ci_low": harm_row["paired_harm_ci_low"],
            "ci_high": harm_row["paired_harm_ci_high"],
            "holm_p": harm_row["holm_p"],
            "primary_detectability": detect_row["diagnostic_normalized_ap"],
            "detectability_ci_low": detect_row["diagnostic_normalized_ap_ci_low"],
            "detectability_ci_high": detect_row["diagnostic_normalized_ap_ci_high"],
            "strength_slope": dose_row["standardized_strength_slope"],
            "strength_ci_low": dose_row["ci_low"],
            "strength_ci_high": dose_row["ci_high"],
            "claim_id": claim_id,
            "interpretation_status": status,
        })
    return rows


def build_governance_results() -> list[dict[str, Any]]:
    bootstrap = load_json(SP8_BOOTSTRAP)
    claim_rows = load_json(SP8_CLAIMS)
    claim_status = {row["id"]: row["status"] for row in claim_rows}
    if claim_status != {
        "G1": "SUPPORTED", "G2": "INCONCLUSIVE",
        "G3": "SUPPORTED", "G4": "SUPPORTED",
    }:
        raise AssetError("SP8 claim-state identity or status changed")
    results = bootstrap.get("results", {})
    expected = {
        "budget_0.01", "budget_0.05", "budget_0.10", "budget_0.20",
        "category_simple", "category_boundary", "category_structured",
    }
    if set(results) != expected:
        raise AssetError("SP8 bootstrap result registry changed")

    rows: list[dict[str, Any]] = []
    for key in ("budget_0.01", "budget_0.05", "budget_0.10", "budget_0.20"):
        item = results[key]
        fraction = float(key.removeprefix("budget_"))
        claim_id = "G4" if fraction == 0.01 else "G1" if fraction == 0.20 else ""
        rows.append({
            "row_type": "budget",
            "scope": f"{fraction:.0%} budget",
            "budget_fraction": fraction,
            "p3_minus_p2_sdr": item["observed_diff"],
            "ci_low": item["ci_lo"],
            "ci_high": item["ci_hi"],
            "p3_better_probability": item["p3_better_prob"],
            "p3_sdr": item["p3_sdr"],
            "p2_sdr": item["p2_sdr"],
            "p3_recall": item["p3_recall"],
            "p3_legitimate_retention": item["p3_retention"],
            "claim_id": claim_id,
            "claim_status": claim_status[claim_id] if claim_id else "DESCRIPTIVE_ONLY",
            "result_interpretation": "P3_ADVANTAGE" if item["ci_lo"] > 0 else "NO_RELIABLE_ADVANTAGE",
        })
    for category in ("simple", "boundary", "structured"):
        item = results[f"category_{category}"]
        rows.append({
            "row_type": "category_at_20pct",
            "scope": category,
            "budget_fraction": 0.20,
            "p3_minus_p2_sdr": item["observed_diff"],
            "ci_low": item["ci_lo"],
            "ci_high": item["ci_hi"],
            "p3_better_probability": item["p3_better_prob"],
            "p3_sdr": item["p3_sdr"],
            "p2_sdr": item["p2_sdr"],
            "p3_recall": item["p3_recall"],
            "p3_legitimate_retention": item["p3_retention"],
            "claim_id": "G3",
            "claim_status": claim_status["G3"],
            "result_interpretation": "P3_ADVANTAGE" if item["ci_lo"] > 0 else "NO_RELIABLE_ADVANTAGE",
        })
    return rows


def build_natural_cases(claims: dict[str, Any]) -> list[dict[str, Any]]:
    summary_path = NATURAL / "natural_task_summary.csv"
    stats_path = NATURAL / "natural_statistics.json"
    summary = read_csv(summary_path, "task")
    statistics = load_json(stats_path)
    effects = statistics.get("task_effects", {})
    if set(summary) != set(NATURAL_BOUNDARIES) or set(effects) != set(NATURAL_BOUNDARIES):
        raise AssetError("natural case identity set changed")
    if claims.get("natural", {}).get("status") != "CASE_STUDY_ONLY":
        raise AssetError("natural claim scope changed")
    rows = []
    for task in NATURAL_BOUNDARIES:
        row = summary[task]
        rows.append({
            "task": task,
            "prediction_boundary": NATURAL_BOUNDARIES[task],
            "n_samples": row["n_samples"],
            "n_features": row["n_features"],
            "n_leak_features": row["n_leak"],
            "primary_detectability": row["diagnostic_normalized_ap"],
            "mean_paired_harm": effects[task],
            "source_sha256": row["source_sha256"],
            "interpretation_status": "CASE_STUDY_ONLY",
        })
    return rows


def build(output: Path) -> dict[str, Any]:
    claims, canonical_manifest = validate_release()
    main_rows = build_main_results(claims)
    governance_rows = build_governance_results()
    natural_rows = build_natural_cases(claims)

    write_csv(output / "main_results.csv", list(main_rows[0]), main_rows)
    write_csv(output / "governance_results.csv", list(governance_rows[0]), governance_rows)
    write_csv(output / "natural_cases.csv", list(natural_rows[0]), natural_rows)

    source_paths = [
        CLAIMS, CLAIM_STATE, CANONICAL, CANONICAL_MANIFEST,
        STATS / "mechanism_summary.csv",
        STATS / "detectability_mechanism_summary.csv",
        STATS / "strength_dose_response.csv",
        NATURAL / "natural_task_summary.csv",
        NATURAL / "natural_statistics.json",
        SP8_BOOTSTRAP, SP8_CLAIMS,
        Path(__file__).resolve(),
    ]
    output_paths = [
        output / "main_results.csv",
        output / "governance_results.csv",
        output / "natural_cases.csv",
    ]
    manifest = {
        "schema_version": 1,
        "status": "EDBT_EAB_PAPER_ASSETS_READY",
        "paper_table_count": 3,
        "evidence_tier": "confirmatory_plus_case_study",
        "canonical_sha256": canonical_manifest["canonical_sha256"],
        "paper_claims_sha256": sha256(CLAIMS),
        "table_policy": {
            "main_results.csv": "MAIN_OR_APPENDIX_SINGLE_SOURCE_TABLE",
            "governance_results.csv": "MAIN_TABLE",
            "natural_cases.csv": "MAIN_OR_COMPACT_APPENDIX_TABLE",
            "mechanism_model_summary.csv": "MACHINE_READABLE_ONLY_OR_FIGURE_SOURCE",
            "diagnostic_method_by_mechanism.csv": "MACHINE_READABLE_ONLY_OR_FIGURE_SOURCE",
            "task_manifest.csv": "ARTIFACT_ONLY",
            "claim_scope": "PROSE_PLUS_MACHINE_READABLE_CLAIMS",
        },
        "legacy_disposition": {
            "artifacts/edbt_eab/claim_evidence_matrix.csv": "STALE_DO_NOT_CITE",
            "artifacts/edbt_eab/mechanism_contract_matrix.csv": "INCOMPLETE_7_OF_11_ARTIFACT_ONLY",
            "artifacts/edbt_eab/baseline_matrix.csv": "METHOD_DOCUMENTATION_NOT_A_RESULTS_TABLE",
        },
        "row_counts": {
            "main_results.csv": len(main_rows),
            "governance_results.csv": len(governance_rows),
            "natural_cases.csv": len(natural_rows),
        },
        "source_sha256": {relative(path): sha256(path) for path in source_paths},
        "output_sha256": {path.name: sha256(path) for path in output_paths},
    }
    manifest_path = output / "paper_asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def check() -> None:
    with tempfile.TemporaryDirectory(prefix="edbt-paper-assets-") as temporary:
        candidate = Path(temporary)
        build(candidate)
        for name in OUTPUT_NAMES:
            expected = DEFAULT_OUTPUT / name
            observed = candidate / name
            if not expected.is_file() or expected.read_bytes() != observed.read_bytes():
                raise AssetError(f"generated asset is missing or stale: {relative(expected)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        print(json.dumps({"status": "EDBT_EAB_PAPER_ASSETS_CURRENT", "table_count": 3}))
    else:
        manifest = build(DEFAULT_OUTPUT)
        print(json.dumps({
            "status": manifest["status"],
            "table_count": manifest["paper_table_count"],
            "row_counts": manifest["row_counts"],
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
