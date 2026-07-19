#!/usr/bin/env python3
"""Build the three governed EDBT paper assets from frozen evidence.

Measurement rows come from corrected-v2. Governance rows come from the final
EDBT revision bundle. The builder verifies every revision source against its
manifest before producing paper-facing CSVs.
"""

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

REVISION = ROOT / "results/edbt_eab_revision"
REVISION_MANIFEST = REVISION / "manifest.json"
REVISION_CLAIMS = REVISION / "claim_state.json"
REVISION_ANALYSIS = REVISION / "analysis_summary.json"
REVISION_REMAINING = REVISION / "remaining_governance_summary.json"
REVISION_A1 = REVISION / "a1_mechanism_level.csv"
REVISION_A2 = REVISION / "a2_gap_stratification.csv"
REVISION_A3 = REVISION / "a3_archetype.csv"
REVISION_NATURAL = REVISION / "natural_governance_summary.csv"
REVISION_SEMANTIC = REVISION / "semantic_budget_summary.csv"
FAILURE = REVISION / "failure_anatomy"
FAILURE_MANIFEST = FAILURE / "failure_anatomy_manifest.json"
FAILURE_SUMMARY = FAILURE / "failure_anatomy_summary.json"
FAILURE_SPARSE = FAILURE / "sparse_failure_anatomy.csv"
FAILURE_NYC = FAILURE / "nyc311_selection_diagnostic.csv"

# The final revision does not duplicate P3 localization outcomes. These values
# are read from the frozen SP8 bootstrap after verifying that the revision
# manifest records complete, cross-model-matched selection hashes.
SP8_BOOTSTRAP = ROOT / "artifacts/sp8/bootstrap_analysis.json"

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
GOV_FIELDS = [
    "row_type", "scope", "learner", "budget_fraction", "cost_unit",
    "effect", "ci_low", "ci_high", "probability_positive",
    "initial_gap", "gap_range", "n_keys", "n_tasks", "p3_recall",
    "p3_legitimate_retention", "negative_task_count", "negative_mechanism_count",
    "mean_signal_fields_removed", "signal_x000_removal_rate",
    "signal_x003_removal_rate", "signal_x005_removal_rate",
    "claim_id", "claim_status", "interpretation",
]


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
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


def validate_release() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if CLAIMS.read_bytes() != CLAIM_STATE.read_bytes():
        raise AssetError("paper_claims.json and claim_state.json are not byte-identical")
    measurement_claims = load_json(CLAIMS)
    canonical_manifest = load_json(CANONICAL_MANIFEST)
    canonical_sha = sha256(CANONICAL)
    if (
        canonical_manifest.get("status") != "CANONICAL"
        or canonical_manifest.get("canonical_sha256") != canonical_sha
        or canonical_sha
        != measurement_claims.get("provenance", {}).get("input_sha256", {}).get(
            "results/corrected_v2/canonical_cells.csv"
        )
    ):
        raise AssetError("corrected-v2 canonical or claim binding is not current")
    if measurement_claims.get("evidence_tier") != "confirmatory":
        raise AssetError("measurement claims are not confirmatory evidence")

    manifest = load_json(REVISION_MANIFEST)
    if manifest.get("status") != "COMPLETE_WITH_DISCLOSED_LIMITATIONS":
        raise AssetError("governance revision is not in its final disclosed state")
    validation = manifest.get("validation", {})
    required_validation = (
        "all_rows_success",
        "selection_hashes_complete",
        "cross_model_selection_hashes_matched",
        "claim_state_builder_derived",
        "analysis_inputs_bound",
        "b2_baseline_refit_deviation_disclosed",
    )
    if not all(validation.get(key) is True for key in required_validation):
        raise AssetError("governance revision validation gate is incomplete")
    manifest_hashes = {entry["path"]: entry["sha256"] for entry in manifest["artifacts"]}
    for path in (
        REVISION_CLAIMS,
        REVISION_ANALYSIS,
        REVISION_REMAINING,
        REVISION_A1,
        REVISION_A2,
        REVISION_A3,
        REVISION_NATURAL,
        REVISION_SEMANTIC,
    ):
        expected = manifest_hashes.get(relative(path))
        if expected != sha256(path):
            raise AssetError(f"revision manifest binding failed: {relative(path)}")
    revision_claims = load_json(REVISION_CLAIMS)
    expected_status = {
        "C1_MULTI_LEARNER_GOVERNANCE": "SUPPORTED",
        "C2_NO_DETECTED_LEARNER_INTERACTION": "SUPPORTED",
        "C3_STRUCTURED_HETEROGENEITY": "NARROWED",
        "C4_ARCHETYPE_SENSITIVITY": "SUPPORTED",
        "C5_NATURAL_GOVERNANCE": "MIXED",
        "C6_SEMANTIC_GROUP_BUDGET": "NARROWED",
    }
    observed_status = {
        key: value["status"] for key, value in revision_claims["claims"].items()
    }
    if observed_status != expected_status:
        raise AssetError("governance revision claim-state identity changed")
    return measurement_claims, canonical_manifest, revision_claims


def validate_failure_anatomy() -> dict[str, Any]:
    manifest = load_json(FAILURE_MANIFEST)
    if (
        manifest.get("status") != "POST_HOC_DESCRIPTIVE_DIAGNOSTIC_COMPLETE"
        or manifest.get("parent_revision_status") != "COMPLETE_WITH_DISCLOSED_LIMITATIONS"
        or manifest.get("selection_hash_validation")
        != {"all_matched": True, "nyc311_rows": 3, "sparse_keys": 1100}
    ):
        raise AssetError("failure-anatomy validation state is incomplete")
    for section in ("input_sha256", "output_sha256"):
        for name, expected in manifest.get(section, {}).items():
            path = ROOT / name
            if not path.is_file() or sha256(path) != expected:
                raise AssetError(f"failure-anatomy binding failed: {name}")
    summary = load_json(FAILURE_SUMMARY)
    if (
        summary.get("status") != "POST_HOC_DESCRIPTIVE_DIAGNOSTIC"
        or summary.get("downstream_model_fits") != 0
        or summary.get("sparse", {}).get("n_keys") != 1100
        or summary.get("nyc311", {}).get("budget_k") != 8
    ):
        raise AssetError("failure-anatomy identity changed")
    return summary


def build_main_results(claims: dict[str, Any]) -> list[dict[str, Any]]:
    harm = read_csv(STATS / "mechanism_summary.csv", "mechanism")
    detect = read_csv(STATS / "detectability_mechanism_summary.csv", "mechanism")
    dose = read_csv(STATS / "strength_dose_response.csv", "mechanism")
    if set(harm) != set(MECHANISMS) or set(detect) != set(MECHANISMS) or set(dose) != set(MECHANISMS):
        raise AssetError("mechanism result identity is not the complete M01-M11 registry")

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
        if any(row["category"] != category for row in (harm_row, detect_row, dose_row)):
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


def gov_row(**values: Any) -> dict[str, Any]:
    row = {field: "" for field in GOV_FIELDS}
    row.update(values)
    return row


def build_governance_results(
    claims: dict[str, Any], failure: dict[str, Any]
) -> list[dict[str, Any]]:
    summary = load_json(REVISION_ANALYSIS)
    remaining = load_json(REVISION_REMAINING)
    p3_profiles = load_json(SP8_BOOTSTRAP)["results"]
    claim_status = {key: value["status"] for key, value in claims["claims"].items()}
    rows: list[dict[str, Any]] = []

    for fraction, item in sorted(summary["LR_budget_curve"].items(), key=lambda value: float(value[0])):
        old = p3_profiles[f"budget_{float(fraction):.2f}"]
        rows.append(gov_row(
            row_type="budget", scope="All mechanisms", learner="LR",
            budget_fraction=float(fraction), cost_unit="encoded column",
            effect=item["paired"], ci_low=item["ci_lo"], ci_high=item["ci_hi"],
            probability_positive=item["P3_better"], n_keys=item["n_keys"], n_tasks=20,
            p3_recall=old["p3_recall"], p3_legitimate_retention=old["p3_retention"],
            claim_id="C1_MULTI_LEARNER_GOVERNANCE",
            claim_status=claim_status["C1_MULTI_LEARNER_GOVERNANCE"],
            interpretation="P3_ADVANTAGE" if item["ci_lo"] > 0 else "NO_RELIABLE_ADVANTAGE",
        ))

    for learner, label in (("LR", "LR"), ("RF", "RF"), ("LightGBM", "LightGBM")):
        item = summary[f"{learner}_overall"]
        rows.append(gov_row(
            row_type="learner_overall", scope="All mechanisms", learner=label,
            budget_fraction=0.2, cost_unit="encoded column", effect=item["paired"],
            ci_low=item["ci_lo"], ci_high=item["ci_hi"],
            probability_positive=item["P3_better"], n_keys=item["n_keys"], n_tasks=20,
            claim_id="C1_MULTI_LEARNER_GOVERNANCE",
            claim_status=claim_status["C1_MULTI_LEARNER_GOVERNANCE"],
            interpretation="P3_ADVANTAGE",
        ))
        for family in ("simple", "boundary", "structured"):
            item = summary[f"{learner}_{family}"]
            rows.append(gov_row(
                row_type="family", scope=family, learner=label,
                budget_fraction=0.2, cost_unit="encoded column", effect=item["paired"],
                ci_low=item["ci_lo"], ci_high=item["ci_hi"],
                probability_positive=item["P3_better"], n_tasks=20,
                claim_id="C3_STRUCTURED_HETEROGENEITY",
                claim_status=claim_status["C3_STRUCTURED_HETEROGENEITY"],
                interpretation="P3_ADVANTAGE" if item["ci_lo"] > 0 else "NO_RELIABLE_ADVANTAGE",
            ))
        for mechanism in MECHANISMS:
            item = summary[f"{learner}_{mechanism}"]
            rows.append(gov_row(
                row_type="mechanism", scope=mechanism, learner=label,
                budget_fraction=0.2, cost_unit="encoded column", effect=item["paired"],
                ci_low=item["ci_lo"], ci_high=item["ci_hi"],
                probability_positive=item["P3_better"], initial_gap=item["initial_gap"],
                n_keys=500, n_tasks=20, claim_id="C3_STRUCTURED_HETEROGENEITY",
                claim_status=claim_status["C3_STRUCTURED_HETEROGENEITY"],
                interpretation="P3_ADVANTAGE" if item["ci_lo"] > 0 else (
                    "P3_DISADVANTAGE" if item["ci_hi"] < 0 else "NO_RELIABLE_ADVANTAGE"
                ),
            ))

    for row in read_rows(REVISION_A2):
        rows.append(gov_row(
            row_type="gap_quartile", scope=row["quartile"], learner="LR",
            budget_fraction=0.2, cost_unit="encoded column", effect=row["paired_effect"],
            ci_low=row["ci_lo"], ci_high=row["ci_hi"], gap_range=row["gap_range"],
            n_keys=row["n_keys"], n_tasks=20, claim_id="C3_STRUCTURED_HETEROGENEITY",
            claim_status=claim_status["C3_STRUCTURED_HETEROGENEITY"],
            interpretation="OPPORTUNITY_SENSITIVITY",
        ))
    for row in read_rows(REVISION_A3):
        row_type = "leave_one_archetype_out" if row["archetype"].startswith("LOAO-") else "archetype"
        rows.append(gov_row(
            row_type=row_type, scope=row["archetype"], learner="LR",
            budget_fraction=0.2, cost_unit="encoded column", effect=row["paired_effect"],
            ci_low=row["ci_lo"], ci_high=row["ci_hi"], n_tasks=row["n_tasks"],
            claim_id="C4_ARCHETYPE_SENSITIVITY",
            claim_status=claim_status["C4_ARCHETYPE_SENSITIVITY"],
            interpretation="NEGATIVE_REGIME" if float(row["ci_hi"]) < 0 else "SENSITIVITY",
        ))
    sparse = failure["sparse"]
    rows.append(gov_row(
        row_type="failure_anatomy", scope="sparse", learner="LR",
        budget_fraction=0.2, cost_unit="encoded column",
        effect=sparse["repair_advantage"], n_keys=sparse["n_keys"], n_tasks=sparse["n_tasks"],
        p3_recall=sparse["p3_leak_recall"],
        p3_legitimate_retention=sparse["p3_legitimate_retention"],
        negative_task_count=sparse["negative_tasks"],
        negative_mechanism_count=sparse["negative_mechanisms"],
        mean_signal_fields_removed=sparse["mean_sparse_signal_fields_removed"],
        signal_x000_removal_rate=sparse["sparse_signal_removal_rates"]["x_000"],
        signal_x003_removal_rate=sparse["sparse_signal_removal_rates"]["x_003"],
        signal_x005_removal_rate=sparse["sparse_signal_removal_rates"]["x_005"],
        claim_id="C4_ARCHETYPE_SENSITIVITY",
        claim_status=claim_status["C4_ARCHETYPE_SENSITIVITY"],
        interpretation="POST_HOC_CONSTRUCTION_DIAGNOSIS",
    ))
    semantic_labels = {
        "encoded_overall": ("All mechanisms", "encoded column"),
        "semantic_recomposed_overall": ("All mechanisms", "semantic group"),
        "encoded_M09": ("M09", "encoded column"),
        "semantic_M09": ("M09", "semantic group"),
        "semantic_minus_encoded_M09": ("M09 semantic - encoded", "cost contrast"),
    }
    for row in read_rows(REVISION_SEMANTIC):
        scope, cost = semantic_labels[row["analysis"]]
        rows.append(gov_row(
            row_type="cost_sensitivity", scope=scope, learner="LR",
            budget_fraction=0.2, cost_unit=cost, effect=row["estimate"],
            ci_low=row["ci_lo"], ci_high=row["ci_hi"],
            probability_positive=row["probability_positive"], n_keys=row["n_keys"],
            n_tasks=row["clusters"], claim_id="C6_SEMANTIC_GROUP_BUDGET",
            claim_status=claim_status["C6_SEMANTIC_GROUP_BUDGET"],
            interpretation="COST_SENSITIVE" if float(row["ci_lo"]) <= 0 <= float(row["ci_hi"]) else "DIRECTIONAL",
        ))

    # Cross-check the paper-facing semantic values against the JSON claim input.
    semantic_map = {row["analysis"]: row for row in read_rows(REVISION_SEMANTIC)}
    for key in ("encoded_overall", "semantic_recomposed_overall", "encoded_M09", "semantic_M09"):
        if abs(float(semantic_map[key]["estimate"]) - float(remaining["semantic"][key]["estimate"])) > 1e-12:
            raise AssetError(f"semantic CSV/JSON mismatch for {key}")
    return rows


def build_natural_cases(
    measurement_claims: dict[str, Any],
    revision_claims: dict[str, Any],
    failure: dict[str, Any],
) -> list[dict[str, Any]]:
    summary = read_csv(NATURAL / "natural_task_summary.csv", "task")
    statistics = load_json(NATURAL / "natural_statistics.json")
    harm = statistics.get("task_effects", {})
    governance = read_csv(REVISION_NATURAL, "task")
    if set(summary) != set(NATURAL_BOUNDARIES) or set(harm) != set(NATURAL_BOUNDARIES) or set(governance) != set(NATURAL_BOUNDARIES):
        raise AssetError("natural case identity set changed")
    if measurement_claims.get("natural", {}).get("status") != "CASE_STUDY_ONLY":
        raise AssetError("natural measurement claim scope changed")
    if revision_claims["claims"]["C5_NATURAL_GOVERNANCE"]["status"] != "MIXED":
        raise AssetError("natural governance claim scope changed")
    rows = []
    nyc = failure["nyc311"]
    for task in NATURAL_BOUNDARIES:
        row = summary[task]
        gov = governance[task]
        rows.append({
            "task": task,
            "prediction_boundary": NATURAL_BOUNDARIES[task],
            "n_samples": row["n_samples"],
            "n_features": row["n_features"],
            "n_leak_features": row["n_leak"],
            "primary_detectability": row["diagnostic_normalized_ap"],
            "mean_paired_harm": harm[task],
            "governance_effect": gov["paired"],
            "initial_gap": gov["initial_gap"],
            "p3_leak_recall": gov["p3_leak_recall"],
            "p3_legitimate_retention": gov["p3_legit_retention"],
            "p3_removed_invalid_count": len(nyc["selected_invalid_fields"]) if task == "NYC311" else "",
            "p3_removed_valid_count": len(nyc["selected_valid_fields"]) if task == "NYC311" else "",
            "selected_invalid_features": ";".join(nyc["selected_invalid_fields"]) if task == "NYC311" else "",
            "missed_invalid_features": ";".join(nyc["missed_invalid_fields"]) if task == "NYC311" else "",
            "failure_diagnostic_status": nyc["status"] if task == "NYC311" else "",
            "source_sha256": row["source_sha256"],
            "interpretation_status": "MIXED_FIXED_CASE_EVIDENCE",
        })
    return rows


def build(output: Path) -> dict[str, Any]:
    measurement_claims, canonical_manifest, revision_claims = validate_release()
    failure = validate_failure_anatomy()
    main_rows = build_main_results(measurement_claims)
    governance_rows = build_governance_results(revision_claims, failure)
    natural_rows = build_natural_cases(measurement_claims, revision_claims, failure)

    write_csv(output / "main_results.csv", list(main_rows[0]), main_rows)
    write_csv(output / "governance_results.csv", GOV_FIELDS, governance_rows)
    write_csv(output / "natural_cases.csv", list(natural_rows[0]), natural_rows)

    source_paths = [
        CLAIMS, CLAIM_STATE, CANONICAL, CANONICAL_MANIFEST,
        STATS / "mechanism_summary.csv",
        STATS / "detectability_mechanism_summary.csv",
        STATS / "strength_dose_response.csv",
        NATURAL / "natural_task_summary.csv",
        NATURAL / "natural_statistics.json",
        REVISION_MANIFEST, REVISION_CLAIMS, REVISION_ANALYSIS,
        REVISION_REMAINING, REVISION_A1, REVISION_A2, REVISION_A3,
        REVISION_NATURAL, REVISION_SEMANTIC, SP8_BOOTSTRAP,
        FAILURE_MANIFEST, FAILURE_SUMMARY, FAILURE_SPARSE, FAILURE_NYC,
        Path(__file__).resolve(),
    ]
    output_paths = [
        output / "main_results.csv",
        output / "governance_results.csv",
        output / "natural_cases.csv",
    ]
    manifest = {
        "schema_version": 2,
        "status": "EDBT_EAB_PAPER_ASSETS_READY",
        "paper_table_count": 3,
        "evidence_tier": "confirmatory_plus_declared_sensitivities_and_fixed_cases",
        "canonical_sha256": canonical_manifest["canonical_sha256"],
        "measurement_claims_sha256": sha256(CLAIMS),
        "governance_claims_sha256": sha256(REVISION_CLAIMS),
        "table_policy": {
            "main_results.csv": "CDX_MEASUREMENT_TABLE_AND_FIGURE_SOURCE",
            "governance_results.csv": "R_LAYER_TABLES_AND_FIGURE_SOURCE",
            "natural_cases.csv": "FIXED_CASE_CONTRACT_TABLE",
            "claim_scope": "PROSE_PLUS_MACHINE_READABLE_CLAIMS",
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
