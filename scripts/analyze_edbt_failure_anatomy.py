#!/usr/bin/env python3
"""Build deterministic post-hoc failure anatomy for the EDBT manuscript.

This analysis never fits a downstream model. It reconstructs the frozen P3
selection masks from immutable task bundles, verifies them against the B1 and
natural-governance ledgers, and summarizes two reviewer-visible negative
regimes: the sparse controlled archetype and NYC311.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_natural_case_studies import BUILDERS  # noqa: E402
from experiments.leakbench.run_natural_case_studies_trainfit import (  # noqa: E402
    apply_train_fitted_category_protocol,
)


REVISION = ROOT / "results/edbt_eab_revision"
DEFAULT_OUTPUT = REVISION / "failure_anatomy"
PRIMARY_BUDGET = 0.20
SPARSE_SIGNAL_INDICES = (0, 3, 5)
KEYS = ["dataset_index", "mechanism", "strength", "training_seed"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def selection_hash(indices: np.ndarray | list[int]) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(
        b"encoded_column_indices_v1\0" + values.tobytes()
    ).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty diagnostic: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_parent_revision() -> dict[str, Any]:
    manifest_path = REVISION / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_WITH_DISCLOSED_LIMITATIONS":
        raise ValueError("parent governance revision is not in its final disclosed state")
    validation = manifest.get("validation", {})
    if not all(
        validation.get(key) is True
        for key in (
            "all_rows_success",
            "selection_hashes_complete",
            "cross_model_selection_hashes_matched",
            "analysis_inputs_bound",
            "b2_baseline_refit_deviation_disclosed",
        )
    ):
        raise ValueError("parent governance revision validation is incomplete")
    declared = {row["path"]: row["sha256"] for row in manifest.get("artifacts", [])}
    required = (
        ROOT / "results/corrected_v2/canonical_cells.csv",
        ROOT / "artifacts/sp6/sp6_bundle_manifest.csv",
        ROOT / "artifacts/sp8/governance_clean.csv",
        REVISION / "b1_multiseed_p2.csv",
        REVISION / "natural_governance_cells.csv",
        REVISION / "natural_governance_summary.csv",
    )
    for path in required:
        if declared.get(relative(path)) != sha256(path):
            raise ValueError(f"parent revision binding failed: {relative(path)}")
    return manifest


def load_sparse_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    canonical = pd.read_csv(
        ROOT / "results/corrected_v2/canonical_cells.csv",
        usecols=["dataset_index", "archetype"],
    ).drop_duplicates()
    if canonical.groupby("dataset_index")["archetype"].nunique().max() != 1:
        raise ValueError("dataset-to-archetype mapping is not unique")
    sparse_indices = set(
        canonical.loc[canonical["archetype"] == "sparse", "dataset_index"].astype(int)
    )
    if len(sparse_indices) != 4:
        raise ValueError(f"expected four sparse tasks, found {sorted(sparse_indices)}")

    b1 = pd.read_csv(
        REVISION / "b1_multiseed_p2.csv",
        usecols=KEYS + [
            "governance_seed", "policy", "budget_k", "budget_fraction", "status",
            "strict_distance_reduction", "initial_gap", "selection_mask_hash",
        ],
    )
    b1 = b1[
        (b1["status"] == "SUCCESS")
        & np.isclose(b1["budget_fraction"], PRIMARY_BUDGET)
        & b1["dataset_index"].isin(sparse_indices)
    ].copy()
    # P0 is stored at budget 0; the 20% slice contains one P3 and 20 P2 rows.
    expected_rows = 4 * 11 * 5 * 5 * 21
    if len(b1) != expected_rows:
        raise ValueError(f"expected {expected_rows} sparse B1 rows, found {len(b1)}")

    manifest = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    manifest = manifest[manifest["dataset_index"].isin(sparse_indices)].copy()
    manifest = manifest.rename(columns={"seed": "training_seed"})
    if len(manifest) != 4 * 11 * 5 * 5 or manifest.duplicated(KEYS).any():
        raise ValueError("sparse bundle manifest is incomplete or duplicated")
    return canonical, b1, manifest


def reconstruct_sparse() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical, b1, manifest = load_sparse_inputs()
    p3 = b1[b1["policy"] == "P3_blind_mi"].copy()
    p2 = (
        b1[b1["policy"] == "P2_random"]
        .groupby(KEYS, as_index=False)
        .agg(p2_sdr=("strict_distance_reduction", "mean"), p2_seeds=("governance_seed", "nunique"))
    )
    if len(p3) != 1100 or len(p2) != 1100 or set(p2["p2_seeds"]) != {20}:
        raise ValueError("sparse P3/P2 key coverage is incomplete")
    paired = p3.merge(p2, on=KEYS, validate="one_to_one")
    bundle_rows = {tuple(getattr(row, key) for key in KEYS): row for row in manifest.itertuples(index=False)}

    rows: list[dict[str, Any]] = []
    verified_hashes = 0
    for row in paired.itertuples(index=False):
        key = tuple(getattr(row, column) for column in KEYS)
        bundle_row = bundle_rows.get(key)
        if bundle_row is None:
            raise ValueError(f"missing sparse bundle key: {key}")
        bundle = ROOT / bundle_row.bundle_path
        if sha256(bundle) != str(bundle_row.bundle_sha256).lower():
            raise ValueError(f"bundle hash mismatch: {bundle_row.bundle_path}")
        with np.load(bundle, allow_pickle=False) as payload:
            bundle_key = str(bundle_row.bundle_key)
            X = np.concatenate(
                (np.asarray(payload["base_X"]), np.asarray(payload[f"block__{bundle_key}"])),
                axis=1,
            )
            y = np.asarray(payload["y"])
            train = np.asarray(payload["train_idx"])
            truth = np.asarray(payload[f"leak_mask__{bundle_key}"], dtype=bool)
        scores = np.nan_to_num(
            mutual_info_classif(X[train], y[train], random_state=42), nan=0.0
        )
        budget_k = int(row.budget_k)
        selected = np.argsort(scores, kind="stable")[::-1][:budget_k]
        observed_hash = selection_hash(selected)
        if observed_hash != str(row.selection_mask_hash):
            raise ValueError(f"P3 selection hash mismatch: {key}")
        verified_hashes += 1

        selected_set = set(int(value) for value in selected)
        removed_leak = int(truth[selected].sum())
        removed_legitimate = int(len(selected) - removed_leak)
        legitimate_count = int((~truth).sum())
        signal_hits = [int(index in selected_set) for index in SPARSE_SIGNAL_INDICES]
        rows.append({
            "dataset_index": int(row.dataset_index),
            "mechanism": str(row.mechanism),
            "strength": str(row.strength),
            "training_seed": int(row.training_seed),
            "budget_k": budget_k,
            "initial_gap": float(row.initial_gap),
            "p3_sdr": float(row.strict_distance_reduction),
            "p2_mean_sdr": float(row.p2_sdr),
            "repair_advantage": float(row.strict_distance_reduction - row.p2_sdr),
            "leak_recall": removed_leak / int(truth.sum()),
            "legitimate_retention": 1.0 - removed_legitimate / legitimate_count,
            "removed_leak_count": removed_leak,
            "removed_legitimate_count": removed_legitimate,
            "removed_x000": signal_hits[0],
            "removed_x003": signal_hits[1],
            "removed_x005": signal_hits[2],
            "removed_sparse_signal_count": sum(signal_hits),
            "selection_mask_hash": observed_hash,
        })

    frame = pd.DataFrame(rows)
    task_effects = frame.groupby("dataset_index")["repair_advantage"].mean()
    mechanism_effects = frame.groupby("mechanism")["repair_advantage"].mean()

    old = pd.read_csv(ROOT / "artifacts/sp8/governance_clean.csv")
    old = old[
        (old["status"] == "SUCCESS")
        & (old["policy"] == "P3_blind_mi")
        & np.isclose(old["budget_fraction"], PRIMARY_BUDGET)
    ].merge(canonical, on="dataset_index", validate="many_to_one")
    archetype_localization = (
        old.groupby("archetype")
        .agg(
            leak_recall=("leak_recall", "mean"),
            legitimate_retention=("legit_retention", "mean"),
        )
        .sort_index()
    )

    summary = {
        "status": "POST_HOC_DESCRIPTIVE_DIAGNOSTIC",
        "scope": "LR sparse archetype at 20% matched encoded-column cost",
        "n_keys": int(len(frame)),
        "n_tasks": int(frame["dataset_index"].nunique()),
        "n_mechanisms": int(frame["mechanism"].nunique()),
        "selection_hashes_verified": verified_hashes,
        "repair_advantage": float(frame["repair_advantage"].mean()),
        "negative_tasks": int((task_effects < 0).sum()),
        "negative_mechanisms": int((mechanism_effects < 0).sum()),
        "p3_leak_recall": float(frame["leak_recall"].mean()),
        "p3_legitimate_retention": float(frame["legitimate_retention"].mean()),
        "mean_sparse_signal_fields_removed": float(frame["removed_sparse_signal_count"].mean()),
        "sparse_signal_removal_rates": {
            "x_000": float(frame["removed_x000"].mean()),
            "x_003": float(frame["removed_x003"].mean()),
            "x_005": float(frame["removed_x005"].mean()),
        },
        "other_archetype_localization_ranges": {
            "leak_recall_min": float(archetype_localization.drop(index="sparse")["leak_recall"].min()),
            "leak_recall_max": float(archetype_localization.drop(index="sparse")["leak_recall"].max()),
            "legitimate_retention_min": float(
                archetype_localization.drop(index="sparse")["legitimate_retention"].min()
            ),
            "legitimate_retention_max": float(
                archetype_localization.drop(index="sparse")["legitimate_retention"].max()
            ),
        },
        "interpretation": (
            "The negative response is present in every sparse task and most mechanisms. "
            "Localization summaries do not show a gross recall or retention collapse relative to "
            "other archetypes; deterministic selection frequently removes the three legitimate "
            "fields that define the sparse generator's clean signal. This is a construction-grounded "
            "failure diagnosis, not a randomized causal decomposition."
        ),
    }
    return rows, summary


def summarize_sparse(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for row_type, key in (("task", "dataset_index"), ("mechanism", "mechanism")):
        for value, group in frame.groupby(key, sort=True):
            output.append({
                "row_type": row_type,
                "scope": value,
                "n_keys": len(group),
                "initial_gap": round(float(group["initial_gap"].mean()), 6),
                "p3_sdr": round(float(group["p3_sdr"].mean()), 6),
                "p2_mean_sdr": round(float(group["p2_mean_sdr"].mean()), 6),
                "repair_advantage": round(float(group["repair_advantage"].mean()), 6),
                "p3_leak_recall": round(float(group["leak_recall"].mean()), 6),
                "p3_legitimate_retention": round(float(group["legitimate_retention"].mean()), 6),
                "mean_sparse_signal_fields_removed": round(
                    float(group["removed_sparse_signal_count"].mean()), 6
                ),
            })
    return output


def reconstruct_nyc311() -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    task = BUILDERS["nyc311"]()
    X, truth, names, audit = apply_train_fitted_category_protocol(task)
    scores = np.nan_to_num(
        mutual_info_classif(X[task.train_idx], task.y[task.train_idx], random_state=42),
        nan=0.0,
    )
    budget_k = max(1, round(PRIMARY_BUDGET * X.shape[1]))
    selected = np.argsort(scores, kind="stable")[::-1][:budget_k]
    observed_hash = selection_hash(selected)

    cells = pd.read_csv(REVISION / "natural_governance_cells.csv")
    p3 = cells[(cells["task"] == "NYC311") & (cells["policy"] == "P3_blind_mi")]
    if len(p3) != 3 or set(p3["status"]) != {"SUCCESS"}:
        raise ValueError("NYC311 P3 ledger coverage changed")
    if set(p3["selection_mask_hash"]) != {observed_hash}:
        raise ValueError("NYC311 reconstructed selection does not match the frozen ledger")
    if p3["initial_gap"].nunique() != 1:
        raise ValueError("NYC311 initial gap differs across training seeds")

    selected_rows = []
    for rank, index in enumerate(selected, start=1):
        selected_rows.append({
            "rank": rank,
            "encoded_index": int(index),
            "feature_name": names[index],
            "contract_label": "invalid" if bool(truth[index]) else "valid",
            "mutual_information": round(float(scores[index]), 9),
            "selection_mask_hash": observed_hash,
        })

    invalid_fields = [names[index] for index in np.flatnonzero(truth)]
    selected_invalid = [names[index] for index in selected if truth[index]]
    selected_valid = [names[index] for index in selected if not truth[index]]
    summary = {
        "status": "POST_HOC_DESCRIPTIVE_DIAGNOSTIC",
        "scope": "NYC311 LR at 20% retained-feature cost",
        "n_features": int(X.shape[1]),
        "budget_k": budget_k,
        "n_invalid_fields": int(truth.sum()),
        "initial_gap": float(p3["initial_gap"].iloc[0]),
        "repair_advantage": float(
            pd.read_csv(REVISION / "natural_governance_summary.csv")
            .set_index("task")
            .loc["NYC311", "paired"]
        ),
        "p3_leak_recall": float(p3["leak_recall"].mean()),
        "p3_legitimate_retention": float(p3["legit_retention"].mean()),
        "invalid_fields": invalid_fields,
        "selected_invalid_fields": selected_invalid,
        "missed_invalid_fields": sorted(set(invalid_fields) - set(selected_invalid)),
        "selected_valid_fields": selected_valid,
        "selection_mask_hash": observed_hash,
        "selection_hashes_verified": int(len(p3)),
        "source_sha256": task.lineage["source_sha256"],
        "preprocessing_mapping_sha256": audit["mapping_sha256"],
        "interpretation": (
            "NYC311 offers little initial repair opportunity. P3 removes one of two invalid "
            "fields together with seven contract-valid context fields, so deletion cost exceeds "
            "the small available strict-full distortion in this fixed case."
        ),
    }
    source_path = Path(task.source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    if not source_path.is_file() or sha256(source_path) != task.lineage["source_sha256"]:
        raise ValueError("NYC311 source binding is not current")
    return selected_rows, summary, source_path


def build(output: Path) -> dict[str, Any]:
    parent_manifest = validate_parent_revision()
    sparse_rows, sparse_summary = reconstruct_sparse()
    nyc_rows, nyc_summary, nyc_source = reconstruct_nyc311()
    sparse_table = summarize_sparse(sparse_rows)

    output.mkdir(parents=True, exist_ok=True)
    sparse_path = output / "sparse_failure_anatomy.csv"
    nyc_path = output / "nyc311_selection_diagnostic.csv"
    summary_path = output / "failure_anatomy_summary.json"
    write_csv(sparse_path, sparse_table)
    write_csv(nyc_path, nyc_rows)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "POST_HOC_DESCRIPTIVE_DIAGNOSTIC",
                "downstream_model_fits": 0,
                "sparse": sparse_summary,
                "nyc311": nyc_summary,
                "allowed_wording": {
                    "sparse": (
                        "The sparse failure is distributed across all four tasks and most mechanisms; "
                        "deterministic selection frequently removes construction-defined legitimate "
                        "signal fields, while recall and retention remain near other archetypes."
                    ),
                    "nyc311": (
                        "NYC311 is a low-opportunity fixed-case failure in which P3 removes one of two "
                        "invalid fields together with seven contract-valid fields."
                    ),
                },
                "forbidden_wording": {
                    "sparse": "Sparse signal concentration causally explains every MI failure.",
                    "nyc311": "NYC311 proves that MI fails on natural data.",
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = [
        ROOT / "results/corrected_v2/canonical_cells.csv",
        ROOT / "artifacts/sp6/sp6_bundle_manifest.csv",
        ROOT / "artifacts/sp8/governance_clean.csv",
        REVISION / "b1_multiseed_p2.csv",
        REVISION / "natural_governance_cells.csv",
        REVISION / "natural_governance_summary.csv",
        REVISION / "manifest.json",
        nyc_source,
        ROOT / "src/leakbench/datasets.py",
        ROOT / "experiments/leakbench/run_natural_case_studies.py",
        ROOT / "experiments/leakbench/run_natural_case_studies_trainfit.py",
        ROOT / "benchmark_v2/datasets/confirmatory_adapters.py",
        Path(__file__).resolve(),
    ]
    outputs = [sparse_path, nyc_path, summary_path]
    manifest = {
        "schema_version": 1,
        "status": "POST_HOC_DESCRIPTIVE_DIAGNOSTIC_COMPLETE",
        "analysis_scope": "failure anatomy; no downstream model fitting or claim-state promotion",
        "parent_revision_status": parent_manifest["status"],
        "selection_hash_validation": {
            "sparse_keys": sparse_summary["selection_hashes_verified"],
            "nyc311_rows": nyc_summary["selection_hashes_verified"],
            "all_matched": True,
        },
        "input_sha256": {relative(path): sha256(path) for path in inputs},
        "output_sha256": {relative(path): sha256(path) for path in outputs},
    }
    manifest_path = output / "failure_anatomy_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--allow-write", action="store_true")
    args = parser.parse_args()
    if not args.allow_write:
        raise RuntimeError("locked; pass --allow-write to create post-hoc diagnostics")
    manifest = build(ROOT / args.output_dir)
    print(json.dumps({
        "status": manifest["status"],
        "output_dir": args.output_dir,
        "sparse_selection_hashes": manifest["selection_hash_validation"]["sparse_keys"],
        "nyc311_selection_hashes": manifest["selection_hash_validation"]["nyc311_rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
